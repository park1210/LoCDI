"""One user-authorized retry in a fresh namespace; same frozen runtime inputs."""
from copy import deepcopy
from datetime import datetime,timezone
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.prepare_b012_memory_stability as engine
from scripts.audit_research_gaps import read,write,sha,check
from app.locdi.llm_setup import digest

BASE_PUB,BASE_PR,BASE_DOC=engine.PUB,engine.PR,engine.DOC
PUB,PR,DOC=BASE_PUB/'retry_002',BASE_PR/'retry_002',BASE_DOC/'retry_002'
RUN='synthea_b012_subset_stable_002'


def prepare_and_execute():
    if PUB.exists() or PR.exists() or DOC.exists():raise ValueError('FRESH_RETRY_ONLY_NO_RESUME')
    old=read(BASE_PUB/'B012_EXECUTION_PROTOCOL.json')
    assert check({k:v for g in ('input_hashes','binding_hashes','code_hashes') for k,v in old[g].items()})['status']=='PASS'
    protected=read(BASE_PR/'preservation_before.json')
    for base in (BASE_PUB,BASE_PR,BASE_DOC):
        for f in base.rglob('*'):
            if f.is_file() and 'retry_002' not in f.parts:protected[str(f)]=sha(f)
    assert check(protected)['status']=='PASS'
    write(PR/'preservation_before.json',protected)
    for n in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json','history_selection_plans.json'):
        target=PR/n;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((BASE_PR/n).read_bytes())
    for n in ('sequential_policy.json','B012_offline_preflight.json','test_results.json','B1_history_selection_summary.json'):
        target=PUB/n;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((BASE_PUB/n).read_bytes())
    p=deepcopy(old);p.pop('fingerprint')
    p.update(namespace=engine.NS+'/retry_002',run_id=RUN,retry_of='synthea_b012_subset_stable_001',
        prior_protocol=old['fingerprint'],user_reported_reason='Earlier connection disconnected',
        disconnect_causality='USER_REPORTED_NOT_INDEPENDENTLY_VERIFIED',
        runtime_configuration_changed=False,created_at=datetime.now(timezone.utc).isoformat(),
        input_hashes={str(PR/n):sha(PR/n) for n in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json','history_selection_plans.json')})
    p['code_hashes']['scripts/rerun_b012_memory_stability.py']=sha(Path(__file__))
    p['preflight_SHA256']=sha(PUB/'B012_offline_preflight.json');p['tests_SHA256']=sha(PUB/'test_results.json')
    p['fingerprint']=digest(p);write(PUB/'B012_EXECUTION_PROTOCOL.json',p)
    import httpx
    with httpx.Client(timeout=15,trust_env=False,follow_redirects=False) as client:
        version=client.get('http://127.0.0.1:11434/api/version');version.raise_for_status()
        tags=client.get('http://127.0.0.1:11434/api/tags');tags.raise_for_status()
        loaded=client.get('http://127.0.0.1:11434/api/ps');loaded.raise_for_status()
    matches=[m for m in tags.json()['models'] if m.get('name')==p['model'] or m.get('model')==p['model']]
    if len(matches)!=1 or 'sha256:'+matches[0]['digest'].removeprefix('sha256:')!=p['model_identifier']:
        raise ValueError('MODEL_DIGEST_PREFLIGHT_FAILED')
    write(PR/'approved_preflight.json',{'status':'PASS','server':version.json(),'loaded_models':loaded.json(),
        'model_digest_match':True,'pins':check({k:v for g in ('input_hashes','binding_hashes','code_hashes') for k,v in p[g].items()}),
        'generation':0,'at':datetime.now(timezone.utc).isoformat(),
        'other_work_basis':'Earlier explicit no-other-work statement persists; user now requested same-scope retry.'})
    a={'scope':'SYNTHETIC_SEQUENTIAL_ER_PILOT_ONLY','plan_fingerprint':p['fingerprint'],
        'policy_fingerprint':p['policy_fingerprint'],'evidence_type':'USER_EXPLICIT',
        'user_statement':'다시 run을 진행하고 보고서를 작성해도 좋아. 기존에 연결이 끊겨서 그런거니까',
        'scope_interpretation':'One fresh Run under the unchanged immediately preceding approved360-request/30-Episode contract. Not a resume or indefinite retry.',
        'accept_source_record_proxy_cutoff':True,'accept_fixed_five_minute_synthetic_turnaround':True,
        'authorize_sequential_source_and_pilot':True,'accept_reference_timing_limits':True,
        'clinical_release_approved':False,'v1_grants_created':0,'run_id':RUN,'max_generation':360,
        'history_selection_fingerprint':p['history_selection_fingerprint'],'response_contract':p['response_contract'],
        'recorded_at':datetime.now(timezone.utc).isoformat(),'server_changes_authorized':False,
        'additional_experiments_authorized':False}
    write(PR/'execution_authorization.json',a)
    print({'status':'PREFLIGHT_PASS_USER_AUTHORIZED_NEW_RUN','protocol':p['fingerprint'],'Run':RUN},flush=True)
    engine.PUB,engine.PR,engine.DOC,engine.RUN=PUB,PR,DOC,RUN
    engine.execute(PR/'execution_authorization.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true')
    a=parser.parse_args()
    if not(a.execute and a.allow_llm):parser.error('EXPLICIT_EXECUTE_ALLOW_LLM_REQUIRED')
    prepare_and_execute()
