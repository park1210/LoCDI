"""Explicit authored-proxy domain transfer, never official clinical admission.

Reconstructs labels from actual saved responses; never fits or relabels data.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
from app.locdi.synthetic_pilot_collection import check,locations,shared_snapshot,response_schema,ZIP,M5,MARKERS
from app.locdi.synthetic_pilot_source import load_zip,score_response
from app.locdi.llm_setup import digest,validate_setup,normalize_model_identifier
from app.llm.prompt_loader import specialty_system_prompt
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import observed_tokens,RoutingUtility
from jsonschema import Draft202012Validator

EXPECTED_SHA='429a8177ae4af5a9d63b81b8d354aa0987e5457e7f1dd5c47286ca3147f1360b'
PLAN_ID='syn15_initial_full_005'
SCHEMA='locdi.routing.authored_transfer.v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(utility,registry,patients,config):
    """Non-generating source/response/fit lineage check, no file writes."""
    path=Path(utility)
    if sha(path)!=EXPECTED_SHA:raise ValueError('AUTHORED_UTILITY_SHA_MISMATCH')
    u=read(path);p=check(PLAN_ID);_,pr=locations(PLAN_ID);run=pr/p['run_id'];ready=validate_setup(config)
    if ready['static_readiness']!='READY':raise ValueError('CURRENT_CONFIG_NOT_READY')
    if AgentRegistry(registry).snapshot()!=u['artifact']['registry'] or digest(registry)!=u['registry_fingerprint']:raise ValueError('AUTHORED_REGISTRY_MISMATCH')
    if config['model_identifier']!=u['model_identifier'] or ready['prepared']['prompt_fingerprints']!=u['prompt_fingerprints']:raise ValueError('AUTHORED_MODEL_PROMPT_MISMATCH')
    if u['plan_fingerprint']!=p['fingerprint'] or u['stage']!=p['stage'] or u['view']!=p['view']:raise ValueError('AUTHORED_PLAN_STAGE_VIEW_MISMATCH')
    if u['stage']!='initial' or u['view']!='full' or u['purpose']!='AUTHORED_SYNTHETIC_PROXY_ONLY':raise ValueError('AUTHORED_SCOPE_MISMATCH')
    if u['reference_specialty_used_as_correctness'] is not False or any(u.get(k)!=v for k,v in MARKERS.items()):raise ValueError('AUTHOR_PROXY_ONLY_NO_CLINICAL_PROMOTION')
    if u['fit_target']!='DIAGNOSIS_AUTHOR_REFERENCE_EXACT_PROXY' or u['artifact']['min_samples']!=2:raise ValueError('FIT_TARGET_OR_MIN_SAMPLES_CHANGED')
    source=load_zip(ZIP,p['catalog']);assert source['zip_sha256']==u['zip_sha256']
    if {t['patient_id'] for t in p['tasks']}&set(patients):raise ValueError('AUTHORED_TRAIN_EVAL_PATIENT_OVERLAP')
    expected=[];rows=[];groups=[];origins={str(path.resolve()):EXPECTED_SHA,str((pr/'plan.json').resolve()):sha(pr/'plan.json')}
    for f,h in p['pins'].items():origins[f]=h
    for t in p['tasks']:
        if t['split']!='train':raise ValueError('ACTUAL_TRAIN_SPLIT_REQUIRED')
        group=[]
        for agent in p['agents']:
            folder=run/('Case%03d'%t['ordinal'])/agent['department'];r=read(folder/'response.json');b=read(folder/'Request0001_before.json');a=read(folder/'Request0001_after.json');request=b['request']
            payload={'department':agent['department'],**t['shared']}
            if r['status']!='RESPONSE_RECORDED' or r['plan_fingerprint']!=p['fingerprint'] or r['shared_input_fingerprint']!=t['shared_fingerprint'] or r['response_fingerprint']!=digest(r['response']):raise ValueError('ACTUAL_AUTHORED_RESPONSE_CHANGED')
            if r['model_identifier']!=u['model_identifier'] or r['prompt_fingerprint']!=u['prompt_fingerprints'][agent['agent_id']]:raise ValueError('ACTUAL_AUTHORED_BINDING_CHANGED')
            if request['messages']!=[{'role':'system','content':specialty_system_prompt(agent['department'])},{'role':'user','content':json.dumps(payload,sort_keys=True)}]:raise ValueError('ACTUAL_SHARED_PROMPT_MISMATCH')
            if request['format']!=response_schema(p['catalog']) or request['options']!={'num_ctx':32768,'temperature':0,'seed':42,'num_predict':1024}:raise ValueError('ACTUAL_OUTPUT_OR_DECODING_MISMATCH')
            if digest(request)!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint'] or a['status']!='VALID' or a['HTTP_status']!=200 or json.loads(a['response_envelope']['message']['content'])!=r['response']:raise ValueError('ACTUAL_RESPONSE_CHECKPOINT_MISMATCH')
            Draft202012Validator(request['format']).validate(r['response'])
            for identity in (b['identity_before'],a['identity_after']):
                if normalize_model_identifier(identity['digest'])!=u['model_identifier']:raise ValueError('ACTUAL_MODEL_DIGEST_MISMATCH')
            if a['identity_after']['context_length']!=32768:raise ValueError('ACTUAL_CONTEXT_MISMATCH')
            correctness,status=score_response(r['response'],source['references'][t['case_id']],p['catalog'])
            item={'patient_id':t['patient_id'],'decision_id':t['decision_id'],'agent_id':agent['agent_id'],'correct':correctness,'tokens':observed_tokens(t['shared']['observation'])}
            group.append(item);rows.append({**item,'status':status,'ordinal':t['ordinal']})
            for file in folder.glob('*.json'):origins[str(file.resolve())]=sha(file)
        complete=all(type(item['correct']) is bool for item in group);groups.append({'ordinal':t['ordinal'],'complete_scorable_group':complete})
        if complete:expected.extend(group)
    if len(expected)!=55 or sum(g['complete_scorable_group'] for g in groups)!=11:raise ValueError('EXPECTED_55_ROWS_11_GROUPS_NOT_RECONSTRUCTED')
    if sorted(map(digest,expected))!=sorted(map(digest,u['artifact']['samples'])):raise ValueError('ACTUAL_FIT_LINEAGE_MISMATCH')
    RoutingUtility(u['artifact'],AgentRegistry(registry))
    return {'artifact':deepcopy(u['artifact']),'source_plan':p,'origin_files':origins,'rows':rows,'complete_train_groups':11,
        'admission':'DEVELOPMENT_DOMAIN_TRANSFER_ONLY','native_accuracy_calibration':'NOT_VALIDATED',**MARKERS}


def make_transfer_backend(*args,authorization,**kwargs):
    """Scoped separate adapter; legacy official and Synthea gates unchanged."""
    from unittest.mock import patch
    from app.locdi.augmentation_experiment_backend import make_paired_backend
    from app.locdi.run_simulation import _SOCKET_LOCK
    from app.locdi.common_menu_backend import admit_development_proxy as legacy
    if kwargs.get('experiment_kind')!='DEVELOPMENT_EXPERIMENT' or authorization.get('clinical_release_approved') is not False or any(authorization.get(k)!=v for k,v in MARKERS.items()):raise ValueError('AUTHORED_TRANSFER_NOT_CLINICAL_EVALUATION')
    if authorization.get('accept_authored_utility_transfer') is not True:raise ValueError('EXPLICIT_AUTHORED_DOMAIN_TRANSFER_APPROVAL_REQUIRED')
    def admit(evidence,registry,patients,config):
        if evidence.get('schema_version')!=SCHEMA:return legacy(evidence,registry,patients,config)
        if any(evidence.get(k)!=v for k,v in MARKERS.items()):raise ValueError('DOMAIN_TRANSFER_DEVELOPMENT_ONLY')
        if evidence.get('utility_sha256')!=EXPECTED_SHA:raise ValueError('AUTHORED_UTILITY_SHA_MISMATCH')
        result=verify(evidence['utility_path'],registry,patients,config)
        split=read(evidence['collection_plan']);actual=[{'decision':{'patient_id':t['patient_id'],'split':t['split']}} for t in result['source_plan']['tasks']]
        if split.get('candidates')!=actual or split.get('authored_plan_fingerprint')!=result['source_plan']['fingerprint']:raise ValueError('ACTUAL_SPLIT_INDEX_MISMATCH')
        return result
    backend=make_paired_backend(*args,authorization=authorization,**kwargs)
    def execute(case,features):
        with _SOCKET_LOCK,patch('app.locdi.common_menu_backend.admit_development_proxy',new=admit):return backend(case,features)
    return execute
