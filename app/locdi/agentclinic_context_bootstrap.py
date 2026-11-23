"""Operator-authorized one-call environment probe, distinct from clinical approval.

No server mutation, implicit authorization, retry, repair or automatic resume.
"""
from datetime import datetime, timezone, timedelta
import ctypes
import hashlib
import json
import subprocess
import urllib.request
from app.locdi.llm_setup import ROOT, digest, normalize_model_identifier
from app.locdi.dataset_freeze import sha
from app.locdi.external_reference_readiness import pinned_private_json
from app.locdi.episode_exporter import _write

VERSION = 'agentclinic.context_bootstrap.v1'
SYSTEM = 'You are checking a local JSON interface. Return only the requested JSON object.'
USER = 'Return {"status":"ready"}. No clinical task is requested.'
SCHEMA = {'type':'object','properties':{'status':{'const':'ready'}},
          'required':['status'],'additionalProperties':False}


def verify_operator_evidence(path, expected_sha, *, scope, now=None):
    r=pinned_private_json(path,expected_sha)
    now=now or datetime.now(timezone.utc)
    if (r.get('schema_version')!='agentclinic.operator_authorization.v1'
            or r.get('authorization_source')!='EXPLICIT_USER_INSTRUCTION'
            or r.get('operator_role')!='LOCAL_SERVER_OPERATOR'
            or r.get('operator_reference')!='USER_LOCAL_SERVER_OPERATOR'
            or r.get('operator_authorized') is not True
            or scope not in r.get('permitted_scopes',[])
            or not r.get('user_statement') or not r.get('scope_confirmation_statement')
            or r.get('research_approval',{}).get('status')!='USER_REPORTED_VERBAL_APPROVAL'
            or r['research_approval'].get('signed_document_present') is not False):
        raise ValueError('OPERATOR_AUTHORIZATION_REQUIRED')
    issued=datetime.fromisoformat(r['issued_at']);expires=datetime.fromisoformat(r['expires_at'])
    if (issued.tzinfo is None or expires.tzinfo is None or not issued<=now<expires
            or expires-issued>timedelta(hours=24)
            or r.get('total_generation_limit')!=50 or r.get('bootstrap_generation_limit')!=1):
        raise ValueError('OPERATOR_AUTHORIZATION_SCOPE_OR_VALIDITY')
    return r


def metadata(endpoint, *, payload=None):
    if endpoint not in {'version','tags','ps','show'}:raise ValueError('METADATA_ENDPOINT_ONLY')
    if (endpoint=='show') != (payload is not None):raise ValueError('METADATA_METHOD')
    request=urllib.request.Request('http://127.0.0.1:11434/api/'+endpoint,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'},method='POST' if payload is not None else 'GET')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=10) as response:
        raw=response.read(12_000_001)
    if len(raw)>12_000_000:raise ValueError('METADATA_SIZE_LIMIT')
    return json.loads(raw)


def memory_snapshot():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                  ('total_physical',ctypes.c_ulonglong),('available_physical',ctypes.c_ulonglong),
                  ('total_pagefile',ctypes.c_ulonglong),('available_pagefile',ctypes.c_ulonglong),
                  ('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),
                  ('available_extended',ctypes.c_ulonglong)]
    row={'observed_at':datetime.now(timezone.utc).isoformat()}
    try:
        m=Memory();m.length=ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError()
        row.update(total_physical_bytes=m.total_physical,available_physical_bytes=m.available_physical)
    except Exception:row['ram_status']='NOT_CHECKED'
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,memory.used,driver_version',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        row.update(gpu_status='OBSERVED' if p.returncode==0 else 'NOT_CHECKED',gpu_csv=p.stdout.strip())
    except Exception:row['gpu_status']='NOT_CHECKED'
    return row


def observe_context(config, expected_context=32768):
    ps=metadata('ps');version=metadata('version')
    matches=[r for r in ps.get('models',[]) if r.get('name')==config['model'] or r.get('model')==config['model']]
    codes=[];model=matches[0] if len(matches)==1 else {}
    if len(matches)!=1:codes.append('LOADED_MODEL_NOT_UNIQUE_OR_MISSING')
    try:
        if normalize_model_identifier(model.get('digest'))!=config['model_identifier']:codes.append('LOADED_DIGEST_MISMATCH')
    except ValueError:codes.append('LOADED_DIGEST_MISMATCH')
    if type(model.get('context_length')) is not int or model['context_length']!=expected_context:
        codes.append('ACTUAL_CONTEXT_UNVERIFIED_OR_MISMATCH')
    memory=memory_snapshot()
    if not memory.get('available_physical_bytes') or memory.get('gpu_status')!='OBSERVED':
        codes.append('MEMORY_OBSERVATION_UNAVAILABLE')
    return {'schema_version':'agentclinic.actual_context.v1','observed_at':datetime.now(timezone.utc).isoformat(),
            'status':'VALID' if not codes else 'INVALID','blocker_codes':codes,'server_version':version,
            'loaded_model':model,'expected_context':expected_context,'memory':memory,
            'offload_scope':'size_vram/size from api/ps; GPU utilization does not establish clinical safety'}


def bootstrap_plan(namespace, tokenizer_path, tree):
    from app.locdi.agentclinic_live_preflight import proposal
    from app.locdi.agentclinic_pilot_readiness import output_paths
    public,private=output_paths(namespace)
    cfg=proposal()['legacy_config'];cfg=dict(cfg,max_requests=1)
    return {'schema_version':VERSION,'namespace':namespace,'purpose':'NONCLINICAL_CONTEXT_ENVIRONMENT_VERIFICATION',
            'public_path':public.relative_to(ROOT).as_posix(),'private_path':private.relative_to(ROOT).as_posix(),
            'config':cfg,'context_tokens':32768,'max_generations':1,'retry':0,'repair':0,
            'system_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),'user_sha256':hashlib.sha256(USER.encode()).hexdigest(),
            'schema':SCHEMA,'tokenizer_path':str(tokenizer_path),'tokenizer_tree_sha256':tree,
            'implementation_sha256':sha(ROOT/'app/locdi/agentclinic_context_bootstrap.py'),
            'experiment_generation_limit':50,'pilot_remaining_after_attempt':49}


class PinnedBootstrapApproval:
    def __init__(self,path,expected_sha256):self.path,self.sha256=path,expected_sha256

    def verify(self,plan):
        a=pinned_private_json(self.path,self.sha256)
        if (a.get('schema_version')!='agentclinic.bootstrap_approval.v1'
                or a.get('scope')!='LOCAL_CONTEXT_BOOTSTRAP_ONE_ONLY'
                or a.get('plan_fingerprint')!=digest(plan)
                or a.get('approved_by')!='USER_LOCAL_SERVER_OPERATOR'):
            raise ValueError('BOOTSTRAP_APPROVAL_REQUIRED')
        evidence=verify_operator_evidence(a['operator_evidence_path'],a['operator_evidence_sha256'],scope=a['scope'])
        if plan['namespace']!=evidence.get('bootstrap_namespace'):raise ValueError('BOOTSTRAP_NAMESPACE_BINDING')
        return a


def run_bootstrap(namespace, *, authority=None, execute=False, allow_llm=False):
    from app.locdi.agentclinic_pilot_readiness import output_paths,LocalTokenCounter
    from app.locdi.ollama_adapter import LocalOllamaAdapter
    if execute is not True or allow_llm is not True or not isinstance(authority,PinnedBootstrapApproval):
        raise ValueError('EXPLICIT_BOOTSTRAP_AUTHORITY_REQUIRED')
    a=pinned_private_json(authority.path,authority.sha256)
    plan=bootstrap_plan(namespace,a['tokenizer_path'],a['tokenizer_tree_sha256']);authority.verify(plan)
    public,private=output_paths(namespace)
    if public.exists() or private.exists():raise FileExistsError('BOOTSTRAP_NO_RESUME')
    counter=LocalTokenCounter(plan['tokenizer_path'],plan['tokenizer_tree_sha256'])
    tokens=counter.count(SYSTEM,USER)
    if tokens+1024+512>32768:raise ValueError('BOOTSTRAP_CONTEXT_BUDGET')
    public.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    _write(plan,'reports/plan.json',public,'reports')
    client=LocalOllamaAdapter(plan['config'],authorized=True)
    result=None;error=None;observation=None
    _write({'plan_fingerprint':digest(plan),'system':SYSTEM,'user':USER,'schema':SCHEMA,
            'local_hf_input_tokens':tokens,'issued_at':datetime.now(timezone.utc).isoformat()},
           'checkpoints/Request0001_before.json',private,'checkpoints')
    try:
        authority.verify(plan)
        result=client.chat_json(SYSTEM,USER,model=plan['config']['model'],schema=SCHEMA)
    except Exception as exc:error=type(exc).__name__
    finally:
        if client.request_count:
            try:observation=observe_context(plan['config'])
            except Exception as exc:observation={'status':'INVALID','blocker_codes':['CONTEXT_OBSERVATION_FAILED'],'error_type':type(exc).__name__}
        raw=getattr(client,'last_response_content',None)
        after={'response':raw,'response_sha256':hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw,str) else None,
               'response_scope':'ADAPTER_CONTENT_NOT_RAW_WIRE','error_type':error,'observation':observation,
               'counters':client.public_metadata(),'local_hf_input_tokens':tokens}
        _write(after,'checkpoints/Request0001_after.json',private,'checkpoints')
        client.close()
    status='BOOTSTRAP_ENVIRONMENT_PASS' if result is not None and observation and observation['status']=='VALID' else 'BOOTSTRAP_ENVIRONMENT_FAILED'
    report={'schema_version':VERSION,'status':status,'plan_fingerprint':digest(plan),'generation_count':client.request_count,
            'namespace':namespace,
            'identity_count':client.identity_check_count,'error_type':error,'actual_context_observation':observation,
            'private_after_sha256':sha(private/'checkpoints/Request0001_after.json'),
            'retry':0,'repair':0,'experiment_generation_limit':50,'pilot_remaining_budget':50-client.request_count,
            'full_tokenizer_equivalence':'NOT_VERIFIED','clinical_metrics':'NOT_EVALUATED'}
    _write(report,'reports/bootstrap.json',public,'reports')
    _write(report,'reports/bootstrap_receipt.json',private,'reports')
    return report


def verify_basis(record, scope):
    """Optional explicit operator basis, never manufacture the professor's signature."""
    if record.get('approval_basis_type')!='USER_OPERATOR_WITH_REPORTED_VERBAL_RESEARCH_APPROVAL':
        raise ValueError('APPROVAL_BASIS_TYPE')
    if record.get('approved_by')!='USER_LOCAL_SERVER_OPERATOR':raise ValueError('OPERATOR_APPROVER_BINDING')
    return verify_operator_evidence(record['operator_evidence_path'],record['operator_evidence_sha256'],scope=scope)


def verify_bootstrap_receipt(path,expected_sha):
    r=pinned_private_json(path,expected_sha)
    if (r.get('status')!='BOOTSTRAP_ENVIRONMENT_PASS' or r.get('generation_count')!=1
            or r.get('pilot_remaining_budget')!=49 or r.get('actual_context_observation',{}).get('status')!='VALID'):
        raise ValueError('VERIFIED_BOOTSTRAP_RECEIPT_REQUIRED')
    from app.locdi.agentclinic_live_preflight import proposal
    cfg=proposal()['legacy_config'];obs=r['actual_context_observation'];model=obs.get('loaded_model',{})
    if (model.get('context_length')!=32768 or normalize_model_identifier(model.get('digest'))!=cfg['model_identifier']
            or model.get('name')!=cfg['model']):raise ValueError('BOOTSTRAP_RECEIPT_MODEL_CONTEXT_BINDING')
    from app.locdi.agentclinic_pilot_readiness import output_paths
    _,private=output_paths(r['namespace'])
    if sha(private/'checkpoints/Request0001_after.json')!=r['private_after_sha256']:
        raise ValueError('BOOTSTRAP_CHECKPOINT_HASH')
    return r
