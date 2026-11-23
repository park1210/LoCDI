"""C3-13A pinned runner plan, local tokenizer and execution evidence. No HTTP."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_freeze import sha
from app.locdi.external_reference_readiness import pinned_private_json
from app.locdi.agentclinic_live_preflight import proposal, preflight, PinnedDoctorApproval
from app.locdi.agentclinic_doctor_client import build_request
from app.locdi.agentclinic_turn_runtime import initial_observation
from app.locdi.agentclinic_scenario_adapter import load_scenarios
from app.locdi.agentclinic_live_preflight import SOURCE

VERSION = 'agentclinic.pilot_runner.v1'


def output_paths(namespace):
    if not isinstance(namespace,str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}',namespace):
        raise ValueError('INVALID_NAMESPACE')
    public=(ROOT/'outputs/research_evaluation'/namespace).resolve()
    private=(ROOT/'outputs/private/research_evaluation'/namespace).resolve()
    if (not public.is_relative_to((ROOT/'outputs/research_evaluation').resolve())
            or not private.is_relative_to((ROOT/'outputs/private/research_evaluation').resolve())):
        raise ValueError('OUTPUT_PATH_ESCAPE')
    return public,private


def prepare_plan(namespace, definition=None):
    definition=definition or proposal()
    old=preflight(definition)
    if old['static_blockers'] or definition!=proposal(definition.get('prompt_profile')):
        raise ValueError('PILOT_FIXED_PLAN_INVALID')
    public,private=output_paths(namespace)
    files=['app/locdi/agentclinic_pilot_readiness.py','app/locdi/agentclinic_live_pilot_runner.py',
           'scripts/run_agentclinic_live_pilot.py','app/locdi/agentclinic_context_bootstrap.py']
    if definition.get('prompt_profile') is not None:files.append('app/locdi/agentclinic_prompt_ablation.py')
    return {'schema_version':VERSION,'namespace':namespace,'doctor_plan':old['plan'],
            'doctor_plan_fingerprint':old['plan_fingerprint'],
            'public_path':public.relative_to(ROOT).as_posix(),'private_path':private.relative_to(ROOT).as_posix(),
            'runner_hashes':{n:sha(ROOT/n) for n in files},
            'global_generation_budget':50,'max_total_input_tokens':50*(32768-1024-512),
            'max_total_output_tokens':51200,'retry_budget':0,'repair_budget':0,
            'resume_policy':'NOT_SUPPORTED','stop_on_first_noncompleted_or_export_error':True,
            'development_exposed_ordinals':list(range(1,108)),
            'selection_rule':'FIRST_SOURCE_ORDINALS_1_TO_5_BEFORE_MODEL_OUTPUT',
            'independent_holdout':False,'scoring_status':'NOT_EVALUATED'}


def tokenizer_tree(path):
    root=Path(path).resolve()
    if not root.is_dir():raise ValueError('LOCAL_TOKENIZER_NOT_FOUND')
    rows={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('TOKENIZER_SYMLINK_FORBIDDEN')
        if p.is_file():
            if p.suffix not in {'.json','.txt','.model','.tiktoken','.jinja'} or len(rows)>=100 or p.stat().st_size>100_000_000:
                raise ValueError('TOKENIZER_ARTIFACT_SCOPE')
            rows[p.relative_to(root).as_posix()]=sha(p)
    if 'tokenizer_config.json' not in rows:raise ValueError('TOKENIZER_CONFIG_REQUIRED')
    return digest(rows)


class LocalTokenCounter:
    """Optional local-only transformers; no install/download or remote code."""
    def __init__(self,path,expected_tree_sha256):
        self.path,self.expected=Path(path).resolve(),expected_tree_sha256
        if tokenizer_tree(self.path)!=self.expected:raise ValueError('TOKENIZER_HASH_MISMATCH')
        try:
            from transformers import AutoTokenizer
            self.tokenizer=AutoTokenizer.from_pretrained(str(self.path),local_files_only=True,trust_remote_code=False)
        except Exception:
            raise ValueError('LOCAL_TOKENIZER_UNAVAILABLE') from None

    def count(self,system,user):
        if tokenizer_tree(self.path)!=self.expected:raise ValueError('TOKENIZER_HASH_MISMATCH')
        tokens=self.tokenizer.apply_chat_template([{'role':'system','content':system},
            {'role':'user','content':user}],tokenize=True,add_generation_prompt=True)
        if not isinstance(tokens,list) or not all(type(t) is int for t in tokens):
            raise ValueError('TOKENIZER_RESULT_INVALID')
        return len(tokens)


def _fresh(record):
    now=datetime.now(timezone.utc)
    observed=datetime.fromisoformat(record['observed_at'])
    expires=datetime.fromisoformat(record['expires_at'])
    if (observed.tzinfo is None or expires.tzinfo is None or not observed<=now<expires
            or expires-observed>timedelta(hours=24)):
        raise ValueError('SERVER_CONTEXT_EVIDENCE_EXPIRED')


class PinnedPilotApproval:
    def __init__(self,path,expected_sha256):
        self.path,self.sha256=path,expected_sha256

    def verify(self,plan,*,load_tokenizer=True):
        r=pinned_private_json(self.path,self.sha256)
        if plan['doctor_plan']['definition'].get('prompt_profile') is not None:
            from app.locdi.agentclinic_prompt_ablation import verify_condition_authorization
            verify_condition_authorization(r,plan)
        if 'approval_basis_type' in r:
            from app.locdi.agentclinic_context_bootstrap import verify_basis,verify_bootstrap_receipt
            basis=verify_basis(r,'LIVE_AGENTCLINIC_PILOT_5_ONLY')
            if plan['namespace']!=basis.get('pilot_namespace'):raise ValueError('OPERATOR_PILOT_NAMESPACE_BINDING')
            receipt=verify_bootstrap_receipt(r['bootstrap_receipt_path'],r['bootstrap_receipt_sha256'])
            if receipt['namespace']!=basis.get('bootstrap_namespace'):raise ValueError('BOOTSTRAP_EXPERIMENT_BINDING')
        if (r.get('schema_version')!='agentclinic.pilot_approval.v1'
                or r.get('scope')!='LIVE_AGENTCLINIC_PILOT_5_ONLY'
                or r.get('plan_fingerprint')!=digest(plan) or not r.get('approved_by')):
            raise ValueError('PILOT_APPROVAL_REQUIRED')
        doctor=PinnedDoctorApproval(r['doctor_approval_path'],r['doctor_approval_sha256'])
        doctor.verify(plan['doctor_plan'])
        e=pinned_private_json(r['server_context_review_path'],r['server_context_review_sha256'])
        _fresh(e)
        d=plan['doctor_plan']['definition'];cfg=d['legacy_config']
        if (e.get('review_status')!='APPROVED' or not e.get('reviewed_by')
                or e.get('plan_fingerprint')!=digest(plan)
                or e.get('model_identifier')!=cfg['model_identifier']
                or e.get('model_name')!=cfg['model'] or e.get('model_version')!=cfg['model_version']
                or e.get('context_tokens')!=d['context_tokens']
                or type(e.get('max_context_tokens')) is not int or e['max_context_tokens']<d['context_tokens']
                or not e.get('server_version') or not e.get('context_evidence_reference')
                or not e.get('identity_evidence_reference')
                or type(e.get('available_memory_bytes')) is not int or e['available_memory_bytes']<=0
                or e.get('tokenizer_model_identifier')!=cfg['model_identifier']
                or not e.get('tokenizer_server_equivalence_review_reference')):
            raise ValueError('SERVER_IDENTITY_OR_CONTEXT_UNVERIFIED')
        if tokenizer_tree(r['tokenizer_path'])!=e['tokenizer_tree_sha256']:
            raise ValueError('TOKENIZER_HASH_MISMATCH')
        counter=LocalTokenCounter(r['tokenizer_path'],e['tokenizer_tree_sha256']) if load_tokenizer else None
        return doctor,counter,e


def prompt_sizes(token_counter=None, *, prompt_profile=None):
    """Exact constructed initial payload bytes; tokens only with verified counter."""
    import json
    rows=[]
    for ordinal,s in enumerate(load_scenarios(SOURCE).scenarios,1):
        if ordinal>5:break
        obs=initial_observation(s);obs.update(remaining_requests=10,history=[])
        system,payload,_=build_request(obs,prompt_profile=prompt_profile)
        user=json.dumps(payload,ensure_ascii=False,sort_keys=True)
        count=token_counter.count(system,user) if token_counter else None
        size=len(system.encode('utf8'))+len(user.encode('utf8'))
        rows.append({'ordinal':ordinal,'payload_fingerprint':digest(payload),'utf8_bytes':size,
            'within_16384_bytes':size<=16384,'input_tokens':count,
            'token_status':'MEASURED_LOCAL_CHAT_TEMPLATE' if count is not None else 'NOT_CHECKED',
            'context_headroom_tokens':32768-count-1024-512 if count is not None else None})
    return rows


def readiness(namespace,authority=None, *, definition=None):
    blockers=[];plan=None;rows=[];approved=False;server=False
    try:
        plan=prepare_plan(namespace,definition)
        public,private=output_paths(namespace)
        if public.exists() or private.exists():blockers.append('OUTPUT_NAMESPACE_EXISTS')
        counter=None
        if isinstance(authority,PinnedPilotApproval):
            try:
                _,counter,_=authority.verify(plan);approved=True;server=True
            except Exception:blockers.append('PINNED_APPROVAL_OR_CONTEXT_EVIDENCE_INVALID')
        else:blockers.extend(['PINNED_PILOT_APPROVAL_REQUIRED','SERVER_IDENTITY_NOT_CHECKED','CONTEXT_UNVERIFIED'])
        rows=prompt_sizes(counter,prompt_profile=plan['doctor_plan']['definition'].get('prompt_profile'))
        if any(not r['within_16384_bytes'] or (r['context_headroom_tokens'] is not None and r['context_headroom_tokens']<0) for r in rows):
            blockers.append('INITIAL_PROMPT_CONTEXT_LIMIT')
    except Exception:blockers.append('SOURCE_MODEL_PROMPT_OR_RUNNER_PLAN_INVALID')
    return {'schema_version':'agentclinic.pilot_readiness.v1','mode':'DRY_RUN',
        'plan':plan,'plan_fingerprint':digest(plan) if plan else None,'initial_prompt_sizes':rows,
        'LIVE_SERVER_READY':server,'LIVE_EXECUTION_APPROVED':approved,
        'LIVE_PILOT_READY':not blockers,'observed_server_digest':'NOT_QUERIED_THIS_STAGE',
        'blocker_codes':sorted(set(blockers)),'real_generation_calls':0,'grading_calls':0,
        'research_evaluation_ready':False}
