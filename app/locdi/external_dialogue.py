"""D5 lossless external dialogue facade. No HospitalCaseRuntime or grading."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from app.locdi.llm_setup import ROOT, digest, load_setup, validate_setup, normalize_model_identifier
from app.locdi.external_benchmark_contract import validate_healthbench, inventory_healthbench
from app.locdi.source_replay import atomic_rows
from app.locdi.run_simulation import OfflineClient
from app.locdi.ollama_adapter import LocalOllamaAdapter
import app.locdi.ollama_adapter as ollama

REVISION='40ee1968852fc57f625934251ac22be47077a8fb'
MANIFEST=ROOT/'datasets/external/healthbench'/REVISION/'dataset_manifest_canonical_v2.json'
PRIVATE=ROOT/'outputs/private/external_evaluation/healthbench/D5_pilot_v1'
PUBLIC=ROOT/'outputs/reports/external_evaluation/healthbench/D5_pilot_v1'
DATA_NAME='2025-05-07-06-14-12_oss_eval.jsonl'
POLICY='D5_FIRST_THREE_SOURCE_ROWS_BEFORE_RESPONSES_V1'
EXPECTED_MANIFEST='51ec636e60520cff042d13b7b488db71e6ffd95e404e656b02fed85514c87a67'


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_source():
    manifest=read(MANIFEST)
    if manifest.get('revision')!=REVISION or manifest.get('manifest_fingerprint')!=EXPECTED_MANIFEST or digest({k:v for k,v in manifest.items() if k!='manifest_fingerprint'})!=manifest.get('manifest_fingerprint'):
        raise ValueError('MANIFEST_FINGERPRINT_MISMATCH')
    for entry in manifest['files'].values():
        path=(ROOT/entry['path']).resolve()
        if not path.is_relative_to((ROOT/'outputs/private/external_benchmarks').resolve()) or sha(path)!=entry['sha256']:
            raise ValueError('SOURCE_HASH_MISMATCH')
    if sha(MANIFEST.parent/'dataset_manifest.json')!=manifest['original_receipt_sha256']:raise ValueError('RECEIPT_HASH_MISMATCH')
    path=ROOT/manifest['files'][DATA_NAME]['path']
    if inventory_healthbench(path)!=manifest['inventory']:raise ValueError('SOURCE_INVENTORY_MISMATCH')
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    return manifest,rows


def messages_for(row):
    """No system prefix, flattening, reordering, trimming or evaluation fields."""
    validate_healthbench(row)
    return deepcopy(row['prompt'])


def budget(sample_count,rubric_count,*,repeats=1,generation_retries=0,generation_repairs=0,
           grader_retries=0,grader_repairs=0,total_cap=80):
    values=(sample_count,rubric_count,repeats,generation_retries,generation_repairs,grader_retries,grader_repairs,total_cap)
    if any(type(v) is not int for v in values) or sample_count<1 or rubric_count<sample_count or repeats<1 or not 1<=total_cap<=80 or min(values[3:7])<0:
        raise ValueError('INVALID_BUDGET')
    # Retry/repair values are maximum additional dispatches per original request.
    generation=sample_count*repeats*(1+generation_retries+generation_repairs)
    grading=rubric_count*repeats*(1+grader_retries+grader_repairs)
    return {'sample_count':sample_count,'rubric_count':rubric_count,'repeats':repeats,
        'generation_max_requests':generation,'grading_max_requests':grading,'total_llm_max_requests':generation+grading,
        'total_cap':total_cap,'within_cap':generation+grading<=total_cap,
        'generation_retries':generation_retries,'generation_repairs':generation_repairs,
        'grader_retries':grader_retries,'grader_repairs':grader_repairs,
        'unbounded_retry_allowed':False,'rubrics_omitted':0}


def destination(root,area):
    root=Path(root).resolve()
    if not root.is_relative_to((ROOT/'outputs'/area).resolve()):raise ValueError('OUTPUT_PATH_ESCAPE')
    return root


def save(root,name,document):
    # The caller validates the private/public namespace before reserving a run.
    return atomic_rows(root,name,document,jsonl=False)


def prepare_pilot():
    manifest,rows=verify_source();selected=rows[:3]
    private=destination(PRIVATE,'private');destination(PUBLIC,'reports')
    selection={'schema_version':'locdi.D5.pilot_selection.v1','dataset':'HealthBench','source_revision':REVISION,
        'manifest_fingerprint':manifest['manifest_fingerprint'],'source_sha256':manifest['files'][DATA_NAME]['sha256'],
        'selection_policy':POLICY,'selected_before_model_responses':True,'original_split':'test','assigned_internal_split':None,
        'independent_test_approved':False,'exclude_from_later_independent_aggregate':True,
        'patient_level_linkage':'UNAVAILABLE','model_pretraining_contamination':'UNKNOWN','created_at':now(),
        'samples':[{'ordinal':i,'source_row':i,'sample_id':r['prompt_id'],'row_fingerprint':digest(r),
            'input_fingerprint':digest(messages_for(r)),'rubric_count':len(r['rubrics'])} for i,r in enumerate(selected,1)]}
    selection['selection_fingerprint']=digest(selection)
    private.mkdir(parents=True,exist_ok=False)
    save(private,'selection_manifest.json',selection)
    return selection


def load_pilot():
    manifest,rows=verify_source();private=destination(PRIVATE,'private');selection=read(private/'selection_manifest.json')
    if digest({k:v for k,v in selection.items() if k!='selection_fingerprint'})!=selection.get('selection_fingerprint'):
        raise ValueError('SELECTION_FINGERPRINT_MISMATCH')
    if (selection['manifest_fingerprint']!=manifest['manifest_fingerprint'] or selection['source_sha256']!=manifest['files'][DATA_NAME]['sha256']
            or selection['selection_policy']!=POLICY or selection['original_split']!='test' or selection['assigned_internal_split'] is not None
            or selection['selected_before_model_responses'] is not True or selection['exclude_from_later_independent_aggregate'] is not True
            or selection['independent_test_approved'] is not False or len(selection['samples'])!=3):raise ValueError('SELECTION_CONTRACT_MISMATCH')
    for i,item in enumerate(selection['samples'],1):
        row=rows[i-1]
        if item!={'ordinal':i,'source_row':i,'sample_id':row['prompt_id'],'row_fingerprint':digest(row),
                'input_fingerprint':digest(messages_for(row)),'rubric_count':len(row['rubrics'])}:raise ValueError('SELECTION_SOURCE_MISMATCH')
    return manifest,selection,rows[:3]


def preflight(run_id,*,config=None):
    result={'schema_version':'locdi.D5.preflight.v1','status':'DRY_RUN','blocker_codes':[],
        'run_id':run_id,'model_calls':0,'grader_calls':0,'server_identity':'NOT_CHECKED',
        'official_score':None,'research_evaluation_ready':False,'plan':None,'plan_fingerprint':None}
    if not isinstance(run_id,str) or not re.fullmatch('[0-9a-f]{32}',run_id):
        result['blocker_codes']=['NEW_32_HEX_RUN_ID_REQUIRED'];return result
    try:
        private=destination(PRIVATE,'private');public=destination(PUBLIC,'reports')
        if (private/'runs'/run_id).exists() or (public/(run_id+'.json')).exists():raise ValueError('RUN_ID_ALREADY_USED')
        manifest,selection,rows=load_pilot();configuration=deepcopy(load_setup() if config is None else config)
        configuration['max_requests']=3
        setup=validate_setup(configuration)
        if setup['static_readiness']!='READY':raise ValueError('MODEL_STATIC_SETUP_BLOCKED')
        limits=budget(3,sum(len(r['rubrics']) for r in rows))
        if not limits['within_cap']:raise ValueError('COMBINED_REQUEST_CAP_EXCEEDED')
        plan={'schema_version':'locdi.D5.dialogue_plan.v1','run_id':run_id,'purpose':'development_external_dialogue',
            'selection_fingerprint':selection['selection_fingerprint'],'dataset_manifest_fingerprint':manifest['manifest_fingerprint'],
            'source_sha256':selection['source_sha256'],'model_settings_fingerprint':setup['prepared']['fingerprint'],
            'model_identifier':configuration['model_identifier'],'model':configuration['model'],'model_version':configuration['model_version'],
            'input_fingerprints':[i['input_fingerprint'] for i in selection['samples']],
            'generation_options':{k:configuration[k] for k in ('temperature','seed','max_output_tokens','timeout_seconds','request_retries')},
            'code_sha256':{p:sha(ROOT/p) for p in ('app/locdi/external_dialogue.py','app/locdi/external_benchmark_contract.py','app/locdi/ollama_adapter.py')},
            'budget':limits,'original_split':'test','development_exposed_samples':3,'remaining_nonpilot_candidates':manifest['inventory']['sample_count']-3,
            'independent_test_approved':False,'grader_execution_ready':False,'clinical_metrics':'NOT_EVALUATED'}
        result.update(plan=plan,plan_fingerprint=digest(plan))
    except Exception as exc:
        allowed={'RUN_ID_ALREADY_USED','MODEL_STATIC_SETUP_BLOCKED','COMBINED_REQUEST_CAP_EXCEEDED','OUTPUT_PATH_ESCAPE',
            'MANIFEST_FINGERPRINT_MISMATCH','SOURCE_HASH_MISMATCH','RECEIPT_HASH_MISMATCH','SOURCE_INVENTORY_MISMATCH',
            'SELECTION_FINGERPRINT_MISMATCH','SELECTION_CONTRACT_MISMATCH','SELECTION_SOURCE_MISMATCH'}
        result['blocker_codes']=[str(exc) if str(exc) in allowed else 'SOURCE_OR_PREPARATION_NOT_READY']
    return result


class OfflineDialogueClient(OfflineClient):
    """TEST_ONLY extension: chat_messages(messages, *, model) -> free text."""
    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        raise ValueError('EXTERNAL_DIALOGUE_REQUIRES_MESSAGES_NOT_ACTION_JSON')
    def chat_messages(self,messages,*,model):raise NotImplementedError()


class LocalDialogueClient(LocalOllamaAdapter):
    """Reuses LoCDI lazy backend/options/identity. Preserves complete message list."""
    def chat_messages(self,messages,*,model):
        if self.closed or model!=self.prepared['config']['model']:raise ValueError('CLIENT_NOT_AUTHORIZED')
        if self.request_count>=self.prepared['config']['max_requests']:raise RuntimeError('GENERATION_BUDGET_EXHAUSTED')
        if not messages or messages[-1]['role']!='user':raise ValueError('FINAL_USER_REQUIRED')
        setup=validate_setup(self.prepared['config'])
        if setup['static_readiness']!='READY' or setup['prepared']['fingerprint']!=self.prepared['fingerprint']:raise ValueError('MODEL_SETUP_CHANGED')
        if self._resources is None:self._resources=ollama._backend(self.prepared)
        legacy,http,_,_=self._resources;self.identity_check_count+=1;self.last_observed_model_identifier=None
        response=http.get(self.prepared['endpoint'][:-3]+'/api/tags');response.raise_for_status()
        observed={normalize_model_identifier(e.get('digest')) for e in response.json()['models'] if e.get('name')==model or e.get('model')==model}
        if len(observed)==1:self.last_observed_model_identifier=next(iter(observed))
        if self.last_observed_model_identifier!=self.prepared['config']['model_identifier']:raise RuntimeError('MODEL_IDENTITY_MISMATCH')
        self.request_count+=1
        # Existing Ollama backend injects fixed decoding and disables SDK retry.
        response=legacy.client.chat.completions.create(model=model,messages=deepcopy(messages))
        text=response.choices[0].message.content
        if not isinstance(text,str) or not text.strip():raise ValueError('EMPTY_OR_NON_TEXT_RESPONSE')
        return text


def run_pilot(run_id,*,execute=False,allow_llm=False,expected_plan_fingerprint=None,config=None,
              test_only=False,client=None,grade=False):
    if any(type(v) is not bool for v in (execute,allow_llm,test_only,grade)):return {'status':'BLOCKED','blocker_codes':['BOOLEAN_FLAGS_REQUIRED']}
    if grade:return {'status':'BLOCKED','blocker_codes':['GRADER_ENVIRONMENT_AND_BOUNDED_EXECUTOR_NOT_READY'],'model_calls':0,'grader_calls':0}
    if execute and not (allow_llm or test_only):return {'status':'BLOCKED','blocker_codes':['EXPLICIT_GENERATION_APPROVAL_REQUIRED'],'model_calls':0,'grader_calls':0}
    if test_only and (allow_llm or not isinstance(client,OfflineDialogueClient)):return {'status':'BLOCKED','blocker_codes':['TEST_ONLY_OFFLINE_CLIENT_REQUIRED']}
    if not test_only and client is not None:return {'status':'BLOCKED','blocker_codes':['UNAPPROVED_CLIENT_INJECTION']}
    check=preflight(run_id,config=config)
    if not execute:return check
    if check['blocker_codes']:return {**check,'status':'BLOCKED'}
    if expected_plan_fingerprint!=check['plan_fingerprint']:return {**check,'status':'BLOCKED','blocker_codes':['PLAN_APPROVAL_MISMATCH']}
    private=destination(PRIVATE,'private');public=destination(PUBLIC,'reports');root=private/'runs'/run_id
    result={'schema_version':'locdi.D5.dialogue_result.v1','run_id':run_id,'status':'FAILED','test_only':test_only,
        'plan_fingerprint':check['plan_fingerprint'],'selection_fingerprint':check['plan']['selection_fingerprint'],
        'model_calls':0,'grader_calls':0,'dispatch_attempts':0,'fake_dispatches':0,'identity_check_count':0,
        'samples':[],'blocker_codes':[],'official_score':None,'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}
    try:root.mkdir(parents=True,exist_ok=False)
    except Exception:return {**result,'blocker_codes':['OUTPUT_RESERVATION_FAILED']}
    active=client
    try:
        save(root,'started.json',{'started_at':now(),'plan':check['plan'],'test_only':test_only})
        # Immutable reuse/exposure intent is written before the first dispatch.
        save(root,'reuse_history.json',{'run_id':run_id,'selection_fingerprint':result['selection_fingerprint'],
            'exposure_kind':'TEST_ONLY_FAKE' if test_only else 'APPROVED_MODEL_GENERATION','recorded_at':now(),
            'sample_refs':['Sample001','Sample002','Sample003'],'original_split':'test','excluded_from_independent_aggregate':True})
        _,selection,rows=load_pilot();configuration=deepcopy(load_setup() if config is None else config);configuration['max_requests']=3
        if not test_only:active=LocalDialogueClient(configuration,authorized=True)
        for item,row in zip(selection['samples'],rows):
            # Recheck immutable source, selection and setup before every dispatch.
            verify=preflight_inputs(check,configuration)
            if not verify:raise ValueError('APPROVED_PLAN_CHANGED')
            if result['dispatch_attempts']>=check['plan']['budget']['generation_max_requests']:raise RuntimeError('GENERATION_BUDGET_EXHAUSTED')
            ref='Sample{:03d}'.format(item['ordinal']);request={'sample_ref':ref,'input_fingerprint':item['input_fingerprint'],'status':'STARTED','started_at':now()}
            save(root,ref+'_before.json',request);result['dispatch_attempts']+=1
            if test_only:result['fake_dispatches']+=1
            try:
                text=active.chat_messages(messages_for(row),model=configuration['model'])
                if not isinstance(text,str) or not text.strip():raise ValueError('EMPTY_OR_NON_TEXT_RESPONSE')
                record={**request,'status':'RESPONSE_RECORDED','ended_at':now(),'sample_id':item['sample_id'],
                    'response':text,'response_fingerprint':digest(text),'grading_status':'NOT_EVALUATED'}
                save(root,ref+'_response.json',record)
                result['samples'].append({'sample_ref':ref,'status':'RESPONSE_RECORDED','grading_status':'NOT_EVALUATED'})
            except Exception as exc:
                safe={'MODEL_IDENTITY_MISMATCH','MODEL_SETUP_CHANGED','GENERATION_BUDGET_EXHAUSTED','EMPTY_OR_NON_TEXT_RESPONSE'}
                code=str(exc) if str(exc) in safe else 'TIMEOUT' if isinstance(exc,TimeoutError) or 'timeout' in type(exc).__name__.lower() else 'REQUEST_OR_STORAGE_FAILED'
                save(root,ref+'_error.json',{'sample_ref':ref,'status':'FAILED','error_code':code,'ended_at':now()})
                result['samples'].append({'sample_ref':ref,'status':'FAILED','error_code':code});raise
        result['status']='COMPLETED'
    except Exception:
        result['blocker_codes'].append('DIALOGUE_EXECUTION_STOPPED')
    finally:
        if isinstance(active,LocalDialogueClient):
            result.update(model_calls=active.request_count,identity_check_count=active.identity_check_count,
                pinned_model_digest=active.prepared['config']['model_identifier'],observed_server_digest=active.last_observed_model_identifier or 'NOT_RECORDED')
            active.close()
        result['uncollected_samples']=3-len(result['samples'])
    try:
        save(root,'result.json',result);public.mkdir(parents=True,exist_ok=True);save(public,run_id+'.json',result)
    except Exception:result['status']='FAILED';result['blocker_codes'].append('RESULT_EXPORT_FAILED')
    return result


def preflight_inputs(approved,configuration):
    """Ignore occupied output reservation when revalidating an active run."""
    _,selection,_=load_pilot();setup=validate_setup(configuration)
    return (selection['selection_fingerprint']==approved['plan']['selection_fingerprint']
        and setup['static_readiness']=='READY' and setup['prepared']['fingerprint']==approved['plan']['model_settings_fingerprint']
        and all(sha(ROOT/p)==value for p,value in approved['plan']['code_sha256'].items()))
