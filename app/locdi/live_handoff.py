"""Explicit controlled live handoff. Default preflight never creates a client."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from app.core.agent_registry import AgentRegistry
from app.llm.prompt_loader import BASE_PHYSICIAN_GUARD
from app.locdi.llm_setup import ROOT,project_file,validate_setup,digest
from app.locdi.ollama_adapter import LocalOllamaAdapter,ModelIdentityError,LLMRequestLimitError
from app.locdi.sequential_handoff import SequentialEngine,link_handoff_requests,HandoffStopped
from app.locdi.run_simulation import SimulationOptions,run_case,_fixed_memory
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.episode_exporter import ExportSession,build_episode,export_summary,_write
from app.locdi.live_runtime import FEATURES
from app.locdi.specialty_evidence_contract import REPAIR_VERSION

DEFAULT_CONFIG=ROOT/'configs/development_handoff_evidence_v2.json'
HANDOFF_ROOT=ROOT/'outputs/development_handoff'
DATASET=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
STAGES=('ER_INITIAL_DECISION','SPECIALTY_ASSESSMENT','ER_REASSESSMENT')
CODE_FILES=('app/locdi/live_handoff.py','app/locdi/sequential_handoff.py','app/locdi/specialty_evidence_contract.py','app/locdi/ollama_adapter.py',
    'app/locdi/run_simulation.py','app/locdi/case_adapter.py','app/locdi/episode_exporter.py',
    'app/core/longitudinal_runner.py','app/core/hospital_parallel_scheduler.py','app/core/hospital_llm_policy.py',
    'app/llm/role_call.py')


class ApprovedSetupChanged(RuntimeError):
    pass


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_config(path=DEFAULT_CONFIG):return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(config):
    """Local-only, fully rechecked bindings. No server or real client construction."""
    expected={'schema_version','ready','base_config_path','target_department','routing','max_requests','runner_retries','stage_bindings'}
    if (not isinstance(config,dict) or set(config)-{'specialty_semantic_repairs'}!=expected or config['schema_version']!='locdi.live_handoff_setup.v1'
            or type(config['ready']) is not bool or config['routing']!='static_development'
            or type(config['runner_retries']) is not int or config['runner_retries']!=0
            or type(config['max_requests']) is not int or not 1<=config['max_requests']<=10):raise ValueError('HANDOFF_CONFIGURATION_INVALID')
    repair_limit=config.get('specialty_semantic_repairs',0)
    if type(repair_limit) is not int or repair_limit not in (0,1):raise ValueError('HANDOFF_CONFIGURATION_INVALID')
    base=json.loads(project_file(config['base_config_path']).read_text(encoding='utf-8'))['llm']
    base['max_requests']=config['max_requests']
    setup=validate_setup(base)
    if setup['static_readiness']!='READY':raise ValueError('BASE_SETUP_NOT_READY')
    prepared=setup['prepared'];registry=AgentRegistry(prepared['registry'])
    er=registry.get('er','')
    try:specialty=registry.get('specialty',config['target_department'])
    except ValueError:raise ValueError('SPECIALTY_UNAVAILABLE') from None
    if not isinstance(config['stage_bindings'],list) or len(config['stage_bindings'])!=3:raise ValueError('STAGE_BINDINGS_INVALID')
    prompts={};bindings={};stage_plan={}
    for item in config['stage_bindings']:
        if set(item)!={'stage','agent_id','model','version','model_identifier','prompt','prompt_sha256'}:raise ValueError('STAGE_BINDINGS_INVALID')
        stage=item['stage'];agent=specialty if stage=='SPECIALTY_ASSESSMENT' else er
        if stage not in STAGES or stage in prompts:raise ValueError('STAGE_BINDINGS_INVALID')
        if any(item[k]!=agent[k] for k in ('agent_id','model','version')):raise ValueError('REGISTRY_BINDING_MISMATCH')
        if item['model_identifier']!=base['model_identifier']:raise ValueError('BINDING_DIGEST_MISMATCH')
        path=project_file(item['prompt'])
        if sha(path)!=item['prompt_sha256']:raise ValueError('PROMPT_FINGERPRINT_MISMATCH')
        text=path.read_text(encoding='utf-8').strip()
        if not text:raise ValueError('PROMPT_FINGERPRINT_MISMATCH')
        system=BASE_PHYSICIAN_GUARD+'\n\n'+('Logical specialty: '+agent['department']+'\n' if stage=='SPECIALTY_ASSESSMENT' else '')+text
        if stage=='ER_INITIAL_DECISION':system+='\nApproved static development target: '+specialty['department']+'. Clinical suitability has not been reviewed.'
        prompts[stage]=system;bindings[stage]=agent
        stage_plan[stage]={**deepcopy(item),'role':agent['role'],'department':agent['department'],
            'system_prompt_fingerprint':digest(system)}
    if set(prompts)!=set(STAGES):raise ValueError('STAGE_BINDINGS_INVALID')
    case=load_cases(DATASET)[0];context=adapt_case(case,registry_manifest=prepared['registry']);_fixed_memory([context],None)
    plan_inputs={'configuration':deepcopy(config),'base_settings_fingerprint':prepared['fingerprint'],
        'base_configuration_file_sha256':sha(project_file(config['base_config_path'])),
        'registry_fingerprint':digest(prepared['registry']),'registry_file_sha256':sha(project_file(base['registry_path'])),
        'stages':stage_plan,'source_fingerprint':sha(DATASET),'case_fingerprint':digest(dict(case)),
        'features':deepcopy(FEATURES),'code_sha256':{p:sha(ROOT/p) for p in CODE_FILES},
        'generation':{k:base[k] for k in ('temperature','seed','max_output_tokens','timeout_seconds','request_retries','max_requests','minute_limit')},
        'model_identifier':base['model_identifier'],'research_evaluation_ready':False,
        'specialty_semantic_repair':{'version':REPAIR_VERSION,'max_attempts':repair_limit,'included_in_global_budget':True},
        'clinical_reference_ready':False,'clinical_metrics':'NOT_EVALUATED',
        'clinical_target_suitability':'NOT_REVIEWED_STATIC_DEVELOPMENT_ONLY'}
    return {'base':base,'prepared':prepared,'registry':registry,'prompts':prompts,'bindings':bindings,
        'case':case,'context':context,'plan_inputs':plan_inputs}


def preflight(*,run_id,config=None):
    result={'schema_version':'locdi.live_handoff_preflight.v1','status':'dry_run','run_id':run_id,
        'case_ref':'Case001','blocker_codes':[],'plan':None,'plan_fingerprint':None,'request_count':0,
        'server_availability':'NOT_CHECKED','observed_server_digest':'NOT_RECORDED',
        'research_evaluation_ready':False,'clinical_metrics':'NOT_EVALUATED'}
    if not isinstance(run_id,str) or not re.fullmatch(r'[0-9a-f]{32}',run_id):
        result['blocker_codes']=['EXPLICIT_NEW_32_HEX_RUN_ID_REQUIRED'];return result
    destination=HANDOFF_ROOT/run_id
    if (destination.resolve().parent!=HANDOFF_ROOT.resolve() or
            not HANDOFF_ROOT.resolve().is_relative_to((ROOT/'outputs').resolve())):
        result['blocker_codes']=['OUTPUT_PATH_INVALID'];return result
    if destination.exists() or (ROOT/'outputs/development_pilot'/run_id).exists():
        result['blocker_codes']=['RUN_ID_OR_OUTPUT_ALREADY_USED'];return result
    try:
        setup=prepare(deepcopy(load_config() if config is None else config))
        plan={'schema_version':'locdi.live_handoff_plan.v1','run_id':run_id,
            'output_namespace':'outputs/development_handoff/'+run_id,**setup['plan_inputs']}
        result.update(plan=plan,plan_fingerprint=digest(plan),static_readiness='READY')
    except Exception as exc:
        safe={'HANDOFF_CONFIGURATION_INVALID','BASE_SETUP_NOT_READY','SPECIALTY_UNAVAILABLE','STAGE_BINDINGS_INVALID',
            'REGISTRY_BINDING_MISMATCH','BINDING_DIGEST_MISMATCH','PROMPT_FINGERPRINT_MISMATCH'}
        result['blocker_codes']=[str(exc) if str(exc) in safe else 'HANDOFF_INPUT_CONTRACT_BLOCKED']
    return result


class LiveSequentialClient(SequentialEngine,LocalOllamaAdapter):
    """Real type, NOT OfflineClient. One adapter/counter/resources across stages."""
    def __init__(self,config,*,approved_plan,authorized=False,output_root):
        if authorized is not True:raise ValueError('EXPLICIT_LIVE_APPROVAL_REQUIRED')
        setup=prepare(config)
        if setup['plan_inputs']!={k:v for k,v in approved_plan.items() if k not in {'schema_version','run_id','output_namespace'}}:raise ValueError('APPROVED_SETUP_CHANGED')
        LocalOllamaAdapter.__init__(self,setup['base'],authorized=True)
        case=setup['case']
        SequentialEngine.__init__(self,None,setup['registry'],episode_id=approved_plan['run_id'],
            patient_id=case['patient_id'],visit_id=case['agent_visible']['index_encounter_id'],
            cutoff=case['agent_visible']['index_encounter_start'],budget=config['max_requests'],
            routing='static_development',static_department=config['target_department'],
            specialty_semantic_repairs=config.get('specialty_semantic_repairs',0))
        self.live_config=deepcopy(config);self.approved_plan=deepcopy(approved_plan)
        self.stage_prompts=setup['prompts'];self.stage_bindings=setup['bindings'];self.output_root=output_root

    def _failure_code(self,exc):
        if isinstance(exc,ApprovedSetupChanged):return 'APPROVED_SETUP_CHANGED'
        if isinstance(exc,HandoffStopped) and str(exc)=='CHECKPOINT_STORAGE_FAILED':return 'CHECKPOINT_STORAGE_FAILED'
        if isinstance(exc,ModelIdentityError):return 'MODEL_IDENTITY_FAILED'
        if isinstance(exc,LLMRequestLimitError):return 'GLOBAL_REQUEST_BUDGET_EXHAUSTED'
        return super()._failure_code(exc)

    def _checkpoint(self,phase,row):
        document={'schema_version':'locdi.live_handoff_checkpoint.v1','run_id':self.episode_id,
            'plan_fingerprint':digest(self.approved_plan),'phase':phase,'recorded_at':now(),
            'request':deepcopy(row),'llm_execution':self.public_metadata()}
        try:_write(document,'private/'+row['request_id']+'_'+phase+'.json',self.output_root,'private')
        except Exception:self._stop('CHECKPOINT_STORAGE_FAILED')

    def _record_specialty_evidence(self,row,summary):
        super()._record_specialty_evidence(row,summary)
        # Transport/schema checkpoint is immutable. Persist the later semantic
        # verdict separately, without raw response or evidence text/indices.
        if row is not None:self._checkpoint('semantic',row)

    def _request(self,stage,payload,system,binding,schema,**kwargs):
        if stage not in self.stage_prompts or binding!=self.stage_bindings[stage]:self._stop('UNAPPROVED_STAGE_BINDING')
        before=len(self.requests)
        try:return super()._request(stage,payload,self.stage_prompts[stage],binding,schema,**kwargs)
        finally:
            if len(self.requests)>before:
                row=self.requests[-1];row['ended_at']=now()
                self._checkpoint('after' if row['status']=='response_received' else 'error',row)

    def _dispatch(self,stage,payload,system,binding,schema):
        row=self.requests[-1]
        row.update(started_at=now(),effective_role=binding['role'],target_department=binding['department'],
            pinned_model_digest=self.prepared['config']['model_identifier'],observed_server_digest='NOT_RECORDED',
            model_fingerprint=digest({k:binding[k] for k in ('agent_id','model','version','role','department')}),
            handoff_id=self.handoffs[-1]['handoff_id'] if self.handoffs else None)
        self._checkpoint('before',row)
        self.last_observed_model_identifier=None
        try:
            try:current=prepare(self.live_config)
            except Exception:raise ApprovedSetupChanged('Pinned local setup changed') from None
            if current['plan_inputs']!={k:v for k,v in self.approved_plan.items() if k not in {'schema_version','run_id','output_namespace'}}:
                raise ApprovedSetupChanged('Approved plan inputs changed')
            return LocalOllamaAdapter.chat_json(self,system,json.dumps(payload,sort_keys=True,ensure_ascii=False),model=binding['model'],schema=schema)
        finally:
            row['observed_server_digest']=self.last_observed_model_identifier or 'NOT_RECORDED'
            row['generation_count_after']=self.request_count;row['identity_check_count_after']=self.identity_check_count

    def public_metadata(self):
        return {**LocalOllamaAdapter.public_metadata(self),'global_budget':self.budget,
            'specialty_semantic_repair_count':sum(r.get('repair_kind')=='SPECIALTY_SEMANTIC' for r in self.requests),
            'core_action_repair_count':sum(r.get('repair_kind')=='CORE_ACTION_CONTRACT' for r in self.requests),
            'retry_count':sum(r['retry'] for r in self.requests),
            'stage_request_attempt_count':len(self.requests),'observed_server_digest':self.last_observed_model_identifier or 'NOT_RECORDED'}


def run_live_handoff(*,run_id,config=None,execute=False,allow_llm=False,accept_release_assumption=False,expected_plan_fingerprint=None):
    if (any(type(v) is not bool for v in (execute,allow_llm,accept_release_assumption)) or execute and not (allow_llm and accept_release_assumption)):
        return {'status':'blocked','blocker_codes':['EXPLICIT_LIVE_APPROVAL_REQUIRED'],'request_count':0}
    definition=deepcopy(load_config() if config is None else config)
    check=preflight(run_id=run_id,config=definition)
    if not execute:return check
    if check['blocker_codes']:return {**check,'status':'blocked'}
    if expected_plan_fingerprint!=check['plan_fingerprint']:return {**check,'status':'blocked','blocker_codes':['APPROVED_PLAN_FINGERPRINT_MISMATCH']}
    root=HANDOFF_ROOT/run_id;client=None
    result={'schema_version':'locdi.live_handoff_pilot.v1','run_id':run_id,'status':'failed','core_status':None,
        'export_status':'NOT_STARTED','handoff_outcome':'NOT_RECORDED','sequential_handoff_verified':False,
        'started_at':now(),'ended_at':None,'plan_fingerprint':check['plan_fingerprint'],
        'blocker_codes':[],'llm_execution':None,'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}
    try:root.mkdir(parents=True,exist_ok=False)
    except Exception:return {**result,'blocker_codes':['OUTPUT_RESERVATION_FAILED'],'ended_at':now()}
    try:
        _write(check,'reports/preflight.json',root,'reports')
        _write({'run_id':run_id,'status':'started','started_at':result['started_at'],'plan_fingerprint':check['plan_fingerprint']},'reports/started.json',root,'reports')
        setup=prepare(definition)
        client=LiveSequentialClient(definition,approved_plan=check['plan'],authorized=True,output_root=root)
        if not isinstance(client,LocalOllamaAdapter):raise ValueError('REAL_ADAPTER_REQUIRED')
        options=SimulationOptions(purpose='development_llm',allow_llm=True,dry_run=False,client=client,
            llm_config=setup['base'],registry_manifest=setup['prepared']['registry'],minute_limit=setup['base']['minute_limit'],
            runner_options=deepcopy(FEATURES),max_retries=0,accept_recorded_release_assumption=True)
        raw=run_case(setup['case'],options=options);result['core_status']=raw['status'];result['status']=raw['status']
        if client.failure:result['blocker_codes'].append(client.failure)
        episode=build_episode(raw,context=setup['context'],session=ExportSession(run_id=run_id))
        link_handoff_requests(client,episode)
        for row in client.requests:
            for handoff in client.handoffs:
                if row['decision_id'] in {handoff['source_decision_id'],handoff['specialty_response_id'],handoff['return_to_er_decision_id']}:row['handoff_id']=handoff['handoff_id']
        result['handoff_outcome']=client.handoffs[-1]['handoff_status'] if client.handoffs else 'NOT_REQUESTED' if raw['status']=='completed' else 'NOT_RECORDED'
        result['sequential_handoff_verified']=result['handoff_outcome']=='handoff_completed'
        audit={'schema_version':'locdi.live_handoff_audit.v1','run_id':run_id,'episode_id':run_id,
            'plan_fingerprint':check['plan_fingerprint'],'configuration_fingerprint':episode['configuration_fingerprint'],
            'requests':client.requests,'handoffs':client.handoffs,'routing':client.routing_records,
            'routing_mode':'static_development','global_budget':client.budget,'request_count':client.request_count,
            'request_attempt_count':len(client.requests),'identity_check_count':client.identity_check_count,
            'specialty_semantic_repair_count':sum(r.get('repair_kind')=='SPECIALTY_SEMANTIC' for r in client.requests),
            'retry_count':sum(r['retry'] for r in client.requests),
            'core_status':raw['status'],'export_status':'EXPORTED','source_consult_count':sum(e['event_type']=='consult_note_added' for e in episode['workflow']['events']),
            'clinical_metrics':'NOT_EVALUATED','clinical_reference_ready':False,'research_evaluation_ready':False}
        _write(episode,'episodes/Case001.json',root,'episodes')
        export_summary({'episodes':[episode]},'metrics/summary.jsonl',output_root=root)
        _write({'run_id':run_id,'plan_fingerprint':check['plan_fingerprint'],'workflow':episode['workflow'],
            'memory':episode['memory'],'terminal':episode['result'],'roles':episode['decision_audit'],
            'core_status':raw['status'],'clinical_metrics':'NOT_EVALUATED'},'reports/runtime.json',root,'reports')
        _write(audit,'reports/handoff.json',root,'reports');result['export_status']='EXPORTED'
    except Exception as exc:
        result['status']='failed';result['export_status']='FAILED_OR_PARTIAL'
        result['blocker_codes'].append('PILOT_EXECUTION_OR_EXPORT_FAILED')
        result['error_type']=type(exc).__name__
        if client is not None and client.failure and client.failure not in result['blocker_codes']:
            result['blocker_codes'].append(client.failure)
    finally:
        if client is not None:
            result['llm_execution']=client.public_metadata();client.close()
        result['ended_at']=now()
    try:_write(result,'reports/pilot.json',root,'reports')
    except Exception:result['status']='failed';result['blocker_codes'].append('FINAL_REPORT_STORAGE_FAILED')
    return result
