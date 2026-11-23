"""C3-16 fixed A/C preparation and explicitly gated execution; default dry run."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load,sha
from app.locdi.episode_exporter import _write
from app.locdi.agentclinic_state_pilot import plan,StudyApproval,run_live
from app.locdi.agentclinic_pilot_readiness import PinnedPilotApproval
from app.locdi.agentclinic_context_bootstrap import metadata,memory_snapshot

PUBLIC=ROOT/'outputs/research_evaluation/c3_16_v1'
PRIVATE=ROOT/'outputs/private/research_evaluation/c3_16_v1'

def prepare():
    """Record the actual new user instruction, never a professor signature."""
    now=datetime.now(timezone.utc);end=now+timedelta(hours=24)
    process=json.loads((PRIVATE/'reports/server_process.json').read_text(encoding='utf-8-sig'))
    if process['Id']!=27112:raise ValueError('SERVER_SESSION_CHANGED')
    prior=ROOT/'outputs/private/research_evaluation/c3_15_B_001/checkpoints/Request0016_after.json'
    context=load(prior)['context_observation'];observed=datetime.fromisoformat(context['observed_at'])
    if context['status']!='VALID' or not observed<=now<observed+timedelta(hours=24):raise ValueError('CONTEXT_EVIDENCE_STALE')
    current={'observed_at':now.isoformat(),'version':metadata('version'),'tags':metadata('tags'),'ps':metadata('ps'),'memory':memory_snapshot()}
    _write(current,'reports/current_server_metadata.json',PRIVATE,'reports')
    plans={c:plan(c) for c in ('A','C')};cfg=plans['A']['base_plan']['doctor_plan']['definition']['legacy_config']
    if current['version']['version']!='0.40.14':raise ValueError('SERVER_VERSION_CHANGED')
    tags=[t for t in current['tags']['models'] if t['name']==cfg['model']]
    if len(tags)!=1 or 'sha256:'+tags[0]['digest']!=cfg['model_identifier']:raise ValueError('MODEL_DIGEST_CHANGED')
    evidence={'schema_version':'agentclinic.state_study_authorization.v1',
        'approval_source':'EXPLICIT_CURRENT_USER_C3_16_APPROVAL','approved_by':'USER_LOCAL_SERVER_OPERATOR',
        'execute_authorized':True,'user_statement':'위 범위의 새 A/C 실행·Exact Match 승인',
        'issued_at':now.isoformat(),'expires_at':end.isoformat(),'order':['A','C'],'cohort':[1,2,3,4,5],
        'combined_budget':100,'condition_budget':50,'retry':0,'repair':0,
        'scoring_scope':'OFFLINE_EXACT_MATCH_LEXICAL_ONLY',
        'exposure_policy':'DEVELOPMENT_EXPOSED_NOT_INDEPENDENT_TEST',
        'server_changes_authorized':False,'additional_bootstrap_authorized':False,
        'research_approval':{'status':'USER_REPORTED_VERBAL_APPROVAL','signed_document_present':False,
            'scope_details':'Prior user statement retained; no invented professor name, date, expiry or signature'},
        'plans':{c:digest(p) for c,p in plans.items()}}
    _write(evidence,'approvals/study_authorization.json',PRIVATE,'approvals')
    for c,p in plans.items():
        base=p['base_plan'];dp=base['doctor_plan'];definition=dp['definition']
        review={'review_status':'APPROVED','reviewed_by':'CODEX_AUTOMATED_TECHNICAL_CHECK',
            'plan_fingerprint':digest(dp),'exposure_policy':definition['exposure_policy'],
            'verified_server_context_tokens':32768,'server_context_configuration_reference':str(PRIVATE/'reports/server_process.json'),
            'limitation':'Reused actual allocation from same server session; synchronously recheck after every call. No clinical review.'}
        _write(review,'approvals/'+c+'_doctor_review.json',PRIVATE,'approvals')
        doctor={'schema_version':'agentclinic.doctor_approval.v1','scope':'LIVE_AGENTCLINIC_DOCTOR_ONLY',
            'plan_fingerprint':digest(dp),'approved_by':'USER_LOCAL_SERVER_OPERATOR','issued_at':now.isoformat(),'expires_at':end.isoformat(),
            'review_path':str(PRIVATE/('approvals/'+c+'_doctor_review.json')),
            'review_sha256':sha(PRIVATE/('approvals/'+c+'_doctor_review.json')),
            'authorization_reference':str(PRIVATE/'approvals/study_authorization.json')}
        _write(doctor,'approvals/'+c+'_doctor_approval.json',PRIVATE,'approvals')
        previous=load(ROOT/'outputs/private/research_evaluation/c3_15_v1/approvals/A_server_review.json')
        server=dict(previous,plan_fingerprint=digest(base),observed_at=observed.isoformat(),
            expires_at=(observed+timedelta(hours=24)).isoformat(),context_evidence_reference=str(prior),
            identity_evidence_reference=str(PRIVATE/'reports/current_server_metadata.json'),
            available_memory_bytes=current['memory']['available_physical_bytes'],
            allocation_scope='C3_15_ACTUAL_OBSERVATION_SAME_UNCHANGED_SERVER_SESSION_NOT_NEW_ALLOCATION_OBSERVATION')
        _write(server,'approvals/'+c+'_server_review.json',PRIVATE,'approvals')
        old=load(ROOT/'outputs/private/research_evaluation/c3_15_v1/approvals/A_pilot_approval.json')
        pilot={'schema_version':'agentclinic.pilot_approval.v1','scope':'LIVE_AGENTCLINIC_PILOT_5_ONLY',
            'plan_fingerprint':digest(base),'approved_by':'USER_LOCAL_SERVER_OPERATOR',
            'doctor_approval_path':str(PRIVATE/('approvals/'+c+'_doctor_approval.json')),
            'doctor_approval_sha256':sha(PRIVATE/('approvals/'+c+'_doctor_approval.json')),
            'server_context_review_path':str(PRIVATE/('approvals/'+c+'_server_review.json')),
            'server_context_review_sha256':sha(PRIVATE/('approvals/'+c+'_server_review.json')),
            'tokenizer_path':old['tokenizer_path'],'authorization_reference':str(PRIVATE/'approvals/study_authorization.json')}
        _write(pilot,'approvals/'+c+'_pilot_approval.json',PRIVATE,'approvals')
        _write(p,'reports/'+c+'_fixed_plan.json',PUBLIC,'reports')
    return {'status':'PREPARED_NEW_AUTHORIZATION','generation_count':0}

def execute():
    result=load(PUBLIC/'regression/result.json')
    suite=digest({p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'tests/locdi').glob('*.py'))})
    if result['status']!='PASS' or result['source_suite_fingerprint']!=suite:raise ValueError('CURRENT_FULL_REGRESSION_REQUIRED')
    baseline=load(PRIVATE/'reports/baseline_hashes.json')
    if any(not (ROOT/p).is_file() or sha(ROOT/p)!=h for p,h in baseline.items()):raise ValueError('PROTECTED_BASELINE_CHANGED')
    study_path=PRIVATE/'approvals/study_authorization.json';study=StudyApproval(study_path,sha(study_path))
    _write({'started_at':datetime.now(timezone.utc).isoformat(),'study_sha256':sha(study_path),'no_resume':True},
        'reports/execution_started.json',PRIVATE,'reports')
    summaries={}
    for c in ('A','C'):
        pilot=PRIVATE/('approvals/'+c+'_pilot_approval.json')
        summaries[c]=run_live(c,study=study,authority=PinnedPilotApproval(pilot,sha(pilot)),execute=True,allow_llm=True)
        print(json.dumps({'condition':c,'generation_count':summaries[c]['generation_count'],'stop_reason':summaries[c]['stop_reason']},ensure_ascii=True),flush=True)
        if summaries[c]['integrity_stop']:break
    _write({'conditions':summaries,'generation_count':sum(s['generation_count'] for s in summaries.values())},
        'reports/execution_summary.json',PUBLIC,'reports')
    return {'status':'EXECUTED_SAVED_RESULTS','conditions':list(summaries)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true');args=parser.parse_args()
    if args.prepare and args.execute:parser.error('prepare and execute must be separate')
    if args.execute and not args.allow_llm:parser.error('--allow-llm required')
    r=prepare() if args.prepare else execute() if args.execute else {'status':'DRY_RUN','plans':{c:plan(c) for c in ('A','C')},'generation_count':0}
    print(json.dumps(r,ensure_ascii=True))
