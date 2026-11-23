"""New authored-synthetic development collection; no changes to official Gates."""
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json,re,time
from app.locdi.llm_setup import ROOT,digest,validate_setup
from app.locdi.routing_collector import atomic_json
from app.locdi.routing_collection_preparation import response_schema as legacy_response_schema
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient,input_guard
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes
from app.locdi.synthetic_pilot_source import load_zip,snapshot,score_response,file_sha,DEPARTMENTS,VERSION
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import observed_tokens,fit_utility,RoutingUtility
from app.llm.prompt_loader import specialty_system_prompt

NS='SYNTHETIC_PILOT_UTILITY_20261121_v1'
PRIVATE=ROOT/'outputs/private/research_evaluation'/NS
PUBLIC=ROOT/'outputs/research_evaluation'/NS
ZIP=ROOT/'LoCDI_Routing_Pilot_15_Cases.zip'
M5=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
CONFIG=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001/llm_config.json'
CODE=['app/locdi/synthetic_pilot_source.py','app/locdi/synthetic_pilot_collection.py',
      'scripts/synthetic_pilot_utility.py','app/core/routing_utility.py','app/core/agent_registry.py',
      'app/core/hospital_parallel_scheduler.py','app/core/hospital_llm_policy.py','app/core/clinical_observation.py',
      'app/core/hospital_clinical_policy.py','app/core/hospital_case_runtime.py','app/core/hospital_test_model.py',
      'app/core/hospital_flow.py','app/core/diagnostic_evidence.py','app/locdi/routing_collector.py',
      'app/locdi/current_ablation_llm_backend.py','app/locdi/current_ablation_runner.py',
      'app/locdi/routing_collection_preparation.py','app/locdi/llm_setup.py','app/locdi/ollama_adapter.py',
      'app/llm/prompt_loader.py','prompts/er_department_system.txt','scripts/Invoke-LoCDISyntheticPilot.ps1']
MARKERS={'experiment_kind':'DEVELOPMENT_EXPERIMENT','source_label_proxy':True,
         'independent_clinical_review':False,'clinical_accuracy':'NOT_EVALUATED','research_evaluation_ready':False}
RESPONSE_CONTRACT='diagnosis_certainty_pair.v2'

def response_schema(catalog):
    """Narrow the existing wire contract; do not repair contradictory outputs."""
    supported=legacy_response_schema(catalog);unresolved=deepcopy(supported)
    # Plan JSON sorts mapping keys; enum arrays need an explicit stable order too.
    supported['properties']['diagnosis']['enum']=sorted(catalog)
    supported['properties']['certainty']['enum']=['supported']
    unresolved['properties']['diagnosis']['enum']=['Unresolved']
    unresolved['properties']['certainty']['enum']=['unresolved']
    return {'anyOf':[supported,unresolved]}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ident(s):
    if not re.fullmatch('[A-Za-z0-9_]{1,64}',s):raise ValueError('UNSAFE_PLAN_ID')
    return s
def locations(plan):return PUBLIC/ident(plan),PRIVATE/ident(plan)
def signed(d):return {**d,'fingerprint':digest(d)}
def shared_snapshot(case,model,view,stage):
    shared,trace=snapshot(case,model,view=view,stage=stage)
    shared.update(diagnosis_catalog=list(model['diagnoses']),instruction='Return one diagnose response in the fixed catalog or Unresolved. Only released observation is a patient fact. Do not invent missing tests. This is a diagnostic proposal, not Hospital supported finalization.')
    return shared,trace
def check(plan):
    pub,pr=locations(plan);p=read(pr/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    if (p.get('response_contract')!=RESPONSE_CONTRACT or p.get('response_schema_fingerprint')!=digest(response_schema(p['catalog']))):raise ValueError('RESPONSE_CONTRACT_CHANGED_NEW_PLAN_REQUIRED')
    for name,h in p['pins'].items():
        if file_sha(name)!=h:raise ValueError('PIN_CHANGED_NEW_PLAN_REQUIRED')
    setup=validate_setup(p['config'])
    if setup['static_readiness']!='READY' or setup['prepared']['registry']!=p['registry']:raise ValueError('CURRENT_BINDING_CHANGED')
    if setup['prepared']['prompt_fingerprints']!=p['prompt_fingerprints']:raise ValueError('PROMPT_CHANGED')
    model=read(M5);b=load_zip(ZIP,model['diagnoses']);cases={c['case_id']:c for c in b['cases']}
    for t in p['tasks']:
        actual,_=shared_snapshot(cases[t['case_id']],model,p['view'],p['stage'])
        if actual!=t['shared'] or digest(actual)!=t['shared_fingerprint']:raise ValueError('SHARED_SOURCE_CHANGED')
    return p

def prepare(plan,*,count=1,view='full',stage='all_tests',start_ordinal=1):
    if count not in (1,15) or view not in ('full','source_compatible') or stage not in ('initial','all_tests'):raise ValueError('PLAN_SCOPE')
    if type(start_ordinal) is not int or not 1<=start_ordinal<=15 or count==15 and start_ordinal!=1:raise ValueError('DATASET_ORDINAL_SCOPE')
    pub,pr=locations(plan)
    if pr.exists() or pub.exists():raise FileExistsError('NO_OVERWRITE_OR_RESUME')
    model=read(M5);b=load_zip(ZIP,model['diagnoses']);cfg=read(CONFIG);cfg['max_requests']=1
    setup=validate_setup(cfg)
    if setup['static_readiness']!='READY':raise ValueError('MODEL_SETUP')
    registry=setup['prepared']['registry'];agents=[a for a in registry['agents'] if a['role']=='specialty']
    if len(agents)!=5 or {a['department'] for a in agents}!=set(DEPARTMENTS):raise ValueError('FIVE_BINDINGS')
    # Dataset order, no Reference/Agent outcome selection. Full run never reuses smoke outputs.
    tasks=[];source_traces=[];bounds=[]
    for i,c in enumerate(b['cases'][start_ordinal-1:start_ordinal-1+count],start_ordinal):
        shared,trace=shared_snapshot(c,model,view,stage)
        if not observed_tokens(shared['observation']):raise ValueError('NO_INFORMATION_SOURCE_COMPATIBLE_INITIAL_BLOCKED')
        source_traces.append(trace)
        # Only the role/department and existing system prompt vary by Agent.
        for department in DEPARTMENTS:
            a=next(a for a in agents if a['department']==department)
            payload={'department':department,**deepcopy(shared)}
            bound=input_guard(specialty_system_prompt(department),json.dumps(payload,sort_keys=True),response_schema(model['diagnoses']))
            bounds.append(bound['conservative_input_token_bound'])
        tasks.append({'ordinal':i,'case_id':c['case_id'],'patient_id':'authored:'+b['manifest']['dataset_id']+':'+c['case_id'],
            'decision_id':plan+':'+str(i),'split':'train','shared':shared,'shared_fingerprint':digest(shared),
            'reference_fingerprint':digest(b['references'][c['case_id']])})
    files=[ROOT/n for n in CODE]+[ZIP,M5,CONFIG,ROOT/cfg['registry_path']]
    files += [ROOT/'prompts/specialty'/ (d+'.txt') for d in DEPARTMENTS]
    p=signed({'version':'authored_synthetic_5agent_collection.v1','plan_id':plan,'run_id':plan+'_run_001',
        'source_policy':VERSION,'zip_sha256':b['zip_sha256'],'view':view,'stage':stage,
        'start_ordinal':start_ordinal,'response_contract':RESPONSE_CONTRACT,
        'response_schema_fingerprint':digest(response_schema(model['diagnoses'])),
        'config':cfg,'registry':registry,'prompt_fingerprints':setup['prepared']['prompt_fingerprints'],
        'catalog':model['diagnoses'],'agents':[next(a for a in agents if a['department']==d) for d in DEPARTMENTS],
        'tasks':tasks,'generation_cap':5*count,'per_task_generation':1,'retry':0,'repair':0,'warmup':0,
        'wall_seconds':900 if count==1 else 2700,'memory_bytes':4*1024**3,'output_bytes':1024**3,
        'pins':{str(p.resolve()):file_sha(p) for p in files},'reference_policy':'AUTHOR_EXACT_PROXY_EXCLUDE_ILLUSTRATIVE_4',
        'reference_review':'NOT_INDEPENDENTLY_CLINICALLY_VERIFIED','all_cases_train':True,
        'split_limit':'No internal validation cohort; existing evaluation patients excluded, content overlap assessed separately',**MARKERS})
    pr.mkdir(parents=True);pub.mkdir(parents=True)
    atomic_json(pr/'plan.json',p);atomic_json(pr/'source_acquisition_traces.json',source_traces)
    atomic_json(pub/'plan_summary.json',{'plan_id':plan,'run_id':p['run_id'],'plan_fingerprint':p['fingerprint'],
        'source_policy':VERSION,'cases':count,'agents':5,'view':view,'stage':stage,'generation_cap':p['generation_cap'],
        'response_contract':RESPONSE_CONTRACT,'dataset_ordinals':[t['ordinal'] for t in tasks],
        'per_agent_diagnostic_snapshot_calls':1,'native_acquisition_generation':0,'maximum_input_bound':max(bounds),
        'status':'PREPARED_NOT_EXECUTED','model_identity_context_live':'NOT_CHECKED','wall_seconds':p['wall_seconds'],**MARKERS})
    atomic_json(pr/'operator_approval_template.json',{'scope':'AUTHORED_SYNTHETIC_5AGENT_COLLECTION_ONLY',
        'plan_fingerprint':p['fingerprint'],'generation_cap':p['generation_cap'],'operator_approved':False,
        'accept_author_reference_proxy':False,'accept_synthetic_acquisition_timing':False,'no_other_ollama_work':False,
        'operator_ref':'','statement':'','recorded_at':None,'expires_at':None,**MARKERS})
    return read(pub/'plan_summary.json')

def verify_approval(a,p):
    now=datetime.now(timezone.utc)
    if (a.get('scope')!='AUTHORED_SYNTHETIC_5AGENT_COLLECTION_ONLY' or a.get('plan_fingerprint')!=p['fingerprint']
        or a.get('generation_cap')!=p['generation_cap'] or any(a.get(k) is not True for k in ('operator_approved','accept_author_reference_proxy','accept_synthetic_acquisition_timing','no_other_ollama_work'))
        or not a.get('operator_ref') or not a.get('statement') or any(a.get(k)!=v for k,v in MARKERS.items())):raise ValueError('NEW_EXPLICIT_OPERATOR_SCOPE_REQUIRED')
    start=datetime.fromisoformat(a['recorded_at']);end=datetime.fromisoformat(a['expires_at'])
    if start.tzinfo is None or end.tzinfo is None or not start<=now<end or not 0<(end-start).total_seconds()<=86400:raise ValueError('APPROVAL_TIME')

def collect(plan,*,execute=False,allow_llm=False,approval=None):
    if not execute:return {'status':'DRY_RUN','plan':check(plan)['fingerprint'],'actual_LLM_generation':0}
    if allow_llm is not True or approval is None:raise ValueError('EXECUTE_ALLOW_LLM_NEW_APPROVAL_REQUIRED')
    p=check(plan);verify_approval(read(approval),p);pub,pr=locations(plan);run=pr/p['run_id']
    if run.exists() or (pub/'collection_status.json').exists():raise FileExistsError('NO_OVERWRITE_RESUME_OR_SECOND_RUN')
    run.mkdir();atomic_json(run/'started.json',{'plan_fingerprint':p['fingerprint'],'at':datetime.now(timezone.utc).isoformat()})
    start=time.monotonic();records=[];stop=None
    def guard():
        if time.monotonic()-start>=p['wall_seconds'] or _process_private_bytes()>=p['memory_bytes']:raise RuntimeError('GLOBAL_RESOURCE_LIMIT')
        if sum(x.stat().st_size for x in run.rglob('*') if x.is_file())>=p['output_bytes']:raise RuntimeError('GLOBAL_OUTPUT_LIMIT')
        verify_approval(read(approval),p)
    budget=GenerationBudget(p['generation_cap'],1,resource_guard=guard);token=_BUDGET.set(budget)
    try:
        for task in p['tasks']:
            for agent in p['agents']:
                client=None;record={'ordinal':task['ordinal'],'case_id':task['case_id'],'agent_id':agent['agent_id'],
                    'department':agent['department'],'shared_input_fingerprint':task['shared_fingerprint'],
                    'plan_fingerprint':p['fingerprint'],'response':None,'reference_match':None,'status':'REQUEST_FAILED'}
                folder=run/('Case%03d'%task['ordinal'])/agent['department']
                try:
                    # Native client owns exclusive creation of the checkpoint leaf.
                    folder.parent.mkdir(parents=True,exist_ok=True)
                    guard();budget.episode_used=0
                    payload={'department':agent['department'],**deepcopy(task['shared'])}
                    client=BudgetedNativeERClient(p['config'],authorized=True,contract_fingerprint=p['fingerprint'],checkpoint_root=folder)
                    text=client.chat_json(specialty_system_prompt(agent['department']),json.dumps(payload,sort_keys=True),
                        model=agent['model'],schema=response_schema(p['catalog']))
                    response=json.loads(text)
                    # Same semantic check without consulting the private Reference during generation.
                    if (response['certainty']=='unresolved')!=(response['diagnosis']=='Unresolved'):raise ValueError('CERTAINTY_DIAGNOSIS_CONTRADICTION')
                    record.update(response=response,response_fingerprint=digest(response),status='RESPONSE_RECORDED',
                        client_metadata=client.public_metadata(),model_identifier=p['config']['model_identifier'],
                        prompt_fingerprint=p['prompt_fingerprints'][agent['agent_id']])
                    guard()
                except Exception as e:
                    code='CERTAINTY_DIAGNOSIS_CONTRADICTION' if str(e)=='CERTAINTY_DIAGNOSIS_CONTRADICTION' else 'CLIENT_OR_CONTRACT_ERROR'
                    record.update(status='FAILED',error_type=type(e).__name__,error_code=code);stop='FIRST_ERROR_NO_RETRY'
                finally:
                    record['generation_attempts']=client.request_count if client is not None else 0
                    if client is not None:client.close()
                    # Constructor failures still need a marker, without resuming anything.
                    folder.mkdir(parents=True,exist_ok=True)
                    atomic_json(folder/'response.json',record);records.append(record)
                if stop:break
            if stop:break
    finally:
        _BUDGET.reset(token)
        complete=stop is None and len(records)==p['generation_cap'] and all(r['status']=='RESPONSE_RECORDED' for r in records)
        report={'status':'COLLECTED' if complete else 'PARTIAL_OR_FAILED','planned_responses':p['generation_cap'],
            'generation_attempts':budget.used,'normal_responses':sum(r['status']=='RESPONSE_RECORDED' for r in records),
            'not_attempted':p['generation_cap']-len(records),'stop_reason':stop or (None if complete else 'COLLECTION_ABORTED'),
            'wall_seconds':time.monotonic()-start,**MARKERS}
        atomic_json(run/'collection_status.json',report);atomic_json(pub/'collection_status.json',report)
    return report

def analyze(plan,*,fit=False):
    p=check(plan);pub,pr=locations(plan);run=pr/p['run_id']
    if not (run/'collection_status.json').exists():raise ValueError('NO_COMPLETED_COLLECTION_STATUS_READ_CHECKPOINTS_FIRST')
    b=load_zip(ZIP,p['catalog']);rows=[];groups=[];fitrows=[]
    for task in p['tasks']:
        group=[]
        for a in p['agents']:
            path=run/('Case%03d'%task['ordinal'])/a['department']/'response.json'
            r=read(path) if path.exists() else {'status':'NOT_ATTEMPTED','response':None}
            match=None;status=r['status']
            if status=='RESPONSE_RECORDED':
                if (r['plan_fingerprint']!=p['fingerprint'] or r['shared_input_fingerprint']!=task['shared_fingerprint']
                    or r['model_identifier']!=p['config']['model_identifier'] or r['prompt_fingerprint']!=p['prompt_fingerprints'][a['agent_id']]
                    or r['response_fingerprint']!=digest(r['response'])):raise ValueError('ACTUAL_RESPONSE_BINDING_MISMATCH')
                checkpoint=read(path.parent/'Request0001_after.json');before=read(path.parent/'Request0001_before.json')
                if (checkpoint.get('status')!='VALID' or checkpoint['request_fingerprint']!=before['request_fingerprint']
                    or json.loads(checkpoint['response_envelope']['message']['content'])!=r['response']):raise ValueError('RESPONSE_CHECKPOINT_LINK')
                request=before['request'];expected={'department':a['department'],**task['shared']}
                if (digest(request)!=before['request_fingerprint'] or request['model']!=p['config']['model']
                    or request['messages']!=[{'role':'system','content':specialty_system_prompt(a['department'])},
                         {'role':'user','content':json.dumps(expected,sort_keys=True)}]
                    or request['format']!=response_schema(p['catalog'])
                    or request['options']!={'num_ctx':32768,'temperature':0,'seed':42,'num_predict':1024}
                    or checkpoint.get('HTTP_status')!=200):raise ValueError('ACTUAL_PAYLOAD_OR_DECODING_CHANGED')
                from app.locdi.llm_setup import normalize_model_identifier
                for identity in (before['identity_before'],checkpoint['identity_after']):
                    if normalize_model_identifier(identity['digest'])!=p['config']['model_identifier']:raise ValueError('MODEL_DIGEST_CHANGED')
                if checkpoint['identity_after']['context_length']!=32768:raise ValueError('ACTUAL_CONTEXT_CHANGED')
                match,status=score_response(r['response'],b['references'][task['case_id']],p['catalog'])
            row={'case_id':task['case_id'],'ordinal':task['ordinal'],'agent_id':a['agent_id'],'agent':a['department'],
                'predicted_diagnosis':r['response']['diagnosis'] if r.get('response') else None,'reference_match':match,'status':status}
            rows.append(row);group.append(row)
        complete=all(type(r['reference_match']) is bool for r in group)
        groups.append({'ordinal':task['ordinal'],'complete_collection':all(r['status'] in ('AUTHOR_REFERENCE_EXACT_PROXY','UNRESOLVED','REFERENCE_NOT_ELIGIBLE') for r in group),
                       'complete_scorable_group':complete,'positive_agents':sum(r['reference_match'] is True for r in group) if complete else None})
        if complete:
            fitrows.extend({'patient_id':task['patient_id'],'decision_id':task['decision_id'],'split':'train',
                'agent_id':r['agent_id'],'model':p['config']['model'],'version':p['config']['model_version'],
                'correct':r['reference_match'],'observation':deepcopy(task['shared']['observation'])} for r in group)
    report={'complete_collection_groups':sum(g['complete_collection'] for g in groups),
        'complete_scorable_train_groups':sum(g['complete_scorable_group'] for g in groups),
        'all_wrong_groups':sum(g['positive_agents']==0 for g in groups),'multiple_positive_groups':sum((g['positive_agents'] or 0)>1 for g in groups),
        'by_agent':{a['department']:dict(Counter(str(r['reference_match']) for r in rows if r['agent']==a['department'])) for a in p['agents']},
        'utility_fit_ready':len(fitrows)>=10,'clinical_gate':'UNCHANGED_NOT_ADMITTED','fit_called':False,**MARKERS}
    tag='fit' if fit else 'analysis'
    if (pub/(tag+'.json')).exists():raise FileExistsError('ANALYSIS_OUTPUT_EXISTS_NO_OVERWRITE')
    if fit:
        if len(fitrows)<10:raise ValueError('AT_LEAST_TWO_COMPLETE_SCORABLE_GROUPS_REQUIRED_NO_LABEL_IMPUTATION')
        artifact=fit_utility(fitrows,AgentRegistry(p['registry']),min_samples=2)
        RoutingUtility(artifact,AgentRegistry(p['registry']))
        atomic_json(pr/'development_utility.json',{'artifact':artifact,'plan_fingerprint':p['fingerprint'],
            'zip_sha256':p['zip_sha256'],'model_identifier':p['config']['model_identifier'],
            'prompt_fingerprints':p['prompt_fingerprints'],'registry_fingerprint':digest(p['registry']),
            'split_fingerprint':digest({'train':[t['patient_id'] for t in p['tasks']],'validation':[]}),
            'fit_target':'DIAGNOSIS_AUTHOR_REFERENCE_EXACT_PROXY','reference_specialty_used_as_correctness':False,
            'view':p['view'],'stage':p['stage'],'groups':groups,'purpose':'AUTHORED_SYNTHETIC_PROXY_ONLY',**MARKERS})
        report.update(fit_called=True,artifact_created=True)
    atomic_json(pr/(tag+'_rows.json'),{'rows':rows,'groups':groups,'fit_rows':fitrows})
    atomic_json(pub/(tag+'.json'),report)
    return report
