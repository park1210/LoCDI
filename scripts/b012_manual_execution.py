"""Operator-only fresh full Run; offline preparation, monitoring and analysis.

No model client is constructed except behind execute + two explicit flags and a
fresh operator authorization. Old Run002 is never resumed or modified.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_research_gaps import read, write, sha, check
from app.locdi.llm_setup import digest
from app.locdi.synthea_sequential_proxy import validate_permission

NS = 'FINAL_B012_FAILURE_RECOVERY_AND_MANUAL_EXECUTION_v1'
PUB = ROOT / 'outputs/research_evaluation' / NS
PR = ROOT / 'outputs/private/research_evaluation' / NS
DOC = ROOT / 'docs' / '연구 평가' / NS
BASE = 'FINAL_B012_MEMORY_STABILITY_RECOVERY_v1/retry_002'
BP = ROOT / 'outputs/research_evaluation' / BASE
BR = ROOT / 'outputs/private/research_evaluation' / BASE
RUN = 'synthea_b012_subset_manual_003'
INPUTS = ('source_rows.json', 'panel.json', 'case_index.json', 'llm_config.json',
          'raw_file_hashes.json', 'history_selection_plans.json')


def utc():
    return datetime.now(timezone.utc)


def copy_exclusive(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        f.write(source.read_bytes())


def validate_authorization(a, p, spec, now=None):
    validate_permission(a, spec, p['fingerprint'], test_only=False)
    expected = {'run_id': RUN, 'max_generation': 360,
                'history_selection_fingerprint': p['history_selection_fingerprint'],
                'response_contract': p['response_contract'], 'operator_approved': True,
                'no_other_ollama_work': True}
    if any(a.get(k) != v for k, v in expected.items()):
        raise ValueError('NEW_MANUAL_OPERATOR_AUTHORIZATION_REQUIRED')
    if not isinstance(a.get('operator_ref'), str) or not a['operator_ref'].strip():
        raise ValueError('ACTUAL_OPERATOR_REFERENCE_REQUIRED')
    issued = datetime.fromisoformat(a['recorded_at'])
    expires = datetime.fromisoformat(a['expires_at'])
    now = now or utc()
    if (issued.tzinfo is None or expires.tzinfo is None or not issued <= now < expires
            or not 0 < (expires-issued).total_seconds() <= 86400):
        raise ValueError('MANUAL_AUTHORIZATION_TIME_INVALID')


def pins(p):
    if p['fingerprint'] != digest({k: v for k, v in p.items() if k != 'fingerprint'}):
        raise ValueError('PROTOCOL_FINGERPRINT_MISMATCH')
    result = check({k: v for group in ('input_hashes', 'binding_hashes', 'code_hashes')
                    for k, v in p[group].items()})
    if result['status'] != 'PASS':
        raise ValueError('INPUT_BINDING_CODE_PIN_CHANGED: ' + str(result['failures']))
    for name, key in (('B012_offline_preflight.json', 'preflight_SHA256'),
                      ('test_results.json', 'tests_SHA256')):
        if sha(PUB/name) != p[key]:
            raise ValueError('PREFLIGHT_OR_TESTS_CHANGED')
    raw = check(read(PR/'raw_file_hashes.json'))
    if raw['status'] != 'PASS':
        raise ValueError('RAW_SOURCE_CHANGED')
    return result


def prepare():
    if (PUB/'B012_EXECUTION_PROTOCOL.json').exists() or (PR/'panel.json').exists():
        raise ValueError('PREPARATION_EXISTS_NO_OVERWRITE')
    old = read(BP/'B012_EXECUTION_PROTOCOL.json')
    if check({k:v for g in ('input_hashes','binding_hashes','code_hashes')
              for k,v in old[g].items()})['status'] != 'PASS':
        raise ValueError('PREVIOUS_PIN_CHANGED')
    for name in INPUTS:
        copy_exclusive(BR/name, PR/name)
    for name in ('sequential_policy.json','B012_offline_preflight.json','test_results.json'):
        copy_exclusive(BP/name, PUB/name)
    protection = read(BR/'preservation_before.json')
    for base in (BP, BR):
        for f in base.rglob('*'):
            if f.is_file(): protection[str(f)] = sha(f)
    write(PR/'preservation_before.json', protection)
    p = deepcopy(old)
    p.pop('fingerprint')
    p.update(namespace=NS, run_id=RUN, created_at=utc().isoformat(),
             preceding_run='synthea_b012_subset_stable_002',
             previous_protocol=old['fingerprint'], execution_mode='MANUAL_FULL_NEW_RUN_ONLY',
             status='NEW_AUTHORIZATION_REQUIRED', no_resume=True,
             result_merge_policy='NO_MERGE_WITH_002; report independent full Run',
             input_hashes={str(PR/n):sha(PR/n) for n in INPUTS})
    p['code_hashes']['scripts/b012_manual_execution.py'] = sha(Path(__file__))
    p['fingerprint'] = digest(p)
    write(PUB/'B012_EXECUTION_PROTOCOL.json', p)
    write(PR/'authorization_template.json', {
        'scope':'SYNTHETIC_SEQUENTIAL_ER_PILOT_ONLY',
        'plan_fingerprint':p['fingerprint'], 'policy_fingerprint':p['policy_fingerprint'],
        'run_id':RUN, 'max_generation':360,
        'history_selection_fingerprint':p['history_selection_fingerprint'],
        'response_contract':p['response_contract'], 'evidence_type':'USER_EXPLICIT',
        'user_statement':None, 'operator_ref':None, 'operator_approved':False,
        'no_other_ollama_work':False, 'recorded_at':None, 'expires_at':None,
        'accept_source_record_proxy_cutoff':False,
        'accept_fixed_five_minute_synthetic_turnaround':False,
        'authorize_sequential_source_and_pilot':False,
        'accept_reference_timing_limits':False,
        'clinical_release_approved':False, 'v1_grants_created':0})
    print(json.dumps({'status':'PREPARED_NOT_APPROVED_NOT_EXECUTED',
                      'protocol':p['fingerprint'], 'generation':0}))


def preflight(authorization=None, server=False):
    p=read(PUB/'B012_EXECUTION_PROTOCOL.json')
    result={'pins':pins(p),'run_unused':not (PR/RUN).exists(),
            'generation':0,'server':'NOT_CHECKED','authorization':'NOT_PROVIDED'}
    if not result['run_unused']:raise ValueError('RUN_USED_NO_RESUME')
    if authorization:
        validate_authorization(read(authorization),p,read(PUB/'sequential_policy.json'))
        result['authorization']='VALID'
    if server:
        import httpx
        with httpx.Client(timeout=15,trust_env=False,follow_redirects=False) as c:
            data={}
            for name in ('version','tags','ps'):
                r=c.get('http://127.0.0.1:11434/api/'+name);r.raise_for_status()
                data[name]=r.json()
        models=[m for m in data['tags']['models'] if m.get('name')==p['model'] or m.get('model')==p['model']]
        if len(models)!=1 or 'sha256:'+models[0]['digest'].removeprefix('sha256:')!=p['model_identifier']:
            raise ValueError('SERVER_MODEL_DIGEST_MISMATCH')
        result['server']=data
    return result


def execute(authorization):
    # Called only by the operator CLI, never by preparation/check/monitor/analyze.
    preflight(authorization,server=True)
    import scripts.prepare_b012_memory_stability as engine
    engine.PUB,engine.PR,engine.DOC,engine.RUN=PUB,PR,DOC,RUN
    engine.execute(authorization)


def monitor():
    run=PR/RUN;trace=run/'episodes.jsonl'
    rows=[]
    if trace.exists():
        for line in trace.read_text(encoding='utf-8').splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass  # writer may be between write/flush
    from collections import Counter
    before=sorted(run.glob('requests/Episode*/Request*_before.json'))
    after=sorted(run.glob('requests/Episode*/Request*_after.json'))
    last_success=None;failed=[]
    for f in after:
        a=read(f)
        if a.get('status')=='VALID':last_success=str(f.relative_to(run))
        else:failed.append({'file':str(f.relative_to(run)),'error_type':a.get('error_type')})
    print(json.dumps({'run_id':RUN,'rows':len(rows),'states':dict(Counter(r['status'] for r in rows)),
        'generation_attempted_before_checkpoints':len(before),'after_checkpoints':len(after),
        'latest_before':str(before[-1].relative_to(run)) if before else None,
        'last_valid_after':last_success,'failures':failed,
        'checkpoint_age_seconds':(utc().timestamp()-before[0].stat().st_mtime) if before else None,
        'partial_Source_counts':{c:{'completed':sum(r['condition']==c and r['status']=='COMPLETED' for r in rows),
            'diagnosis':sum(r['condition']==c and r['status']=='COMPLETED' and
                r['result']['prediction'].get('diagnosis') in
                {p['case_ref']:p['reference']['accepted_diagnoses'] for p in read(PR/'panel.json')}[r['case_ref']]
                for r in rows),
            'department':sum(r['condition']==c and r['status']=='COMPLETED' and
                r['result']['prediction'].get('department') in
                {p['case_ref']:p['reference']['accepted_departments'] for p in read(PR/'panel.json')}[r['case_ref']]
                for r in rows)} for c in ('B0','B1','B2')}},ensure_ascii=True))


def analyze():
    # No backend/client: only saved rows/checkpoints, with the unchanged scorer.
    from app.locdi.b012_pilot import aggregate
    from collections import Counter
    from jsonschema import Draft202012Validator
    p=read(PUB/'B012_EXECUTION_PROTOCOL.json');pins(p)
    run=PR/RUN;collection=read(run/'collection.json')
    rows=[json.loads(line) for line in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if rows!=collection['rows']:raise ValueError('COLLECTION_TRACE_MISMATCH')
    result=aggregate(read(PR/'panel.json'),rows)
    if result!=read(run/'aggregate.json'):raise ValueError('AGGREGATE_MISMATCH')
    usage=Counter();audit=[]
    for bp in sorted(run.glob('requests/Episode*/Request*_before.json')):
        b=read(bp);ap=bp.with_name(bp.name.replace('_before','_after'))
        usage['before_checkpoints']+=1
        if not ap.exists():audit.append({'checkpoint':str(bp.relative_to(run)),'status':'AFTER_MISSING'});continue
        a=read(ap);usage['after_checkpoints']+=1
        if digest(b['request'])!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint']:
            raise ValueError('CHECKPOINT_FINGERPRINT_MISMATCH')
        env=a.get('response_envelope') or {}
        usage['prompt_tokens_reported']+=env.get('prompt_eval_count',0)
        usage['output_tokens_reported']+=env.get('eval_count',0)
        usage['HTTP_wall_seconds']+=a.get('wall_seconds',0)
        usage['valid_responses']+=a.get('status')=='VALID'
        schema_errors=[]
        if env.get('message',{}).get('content'):
            try:
                value=json.loads(env['message']['content'])
                schema_errors=[{'validator':e.validator,'path':list(e.path)}
                               for e in Draft202012Validator(b['request']['format']).iter_errors(value)]
            except (ValueError,KeyError):schema_errors=[{'validator':'JSON_PARSE'}]
        audit.append({'checkpoint':str(bp.relative_to(run)),'status':a['status'],
                      'error_type':a.get('error_type'),'schema_errors':schema_errors})
    if usage['before_checkpoints']!=collection['generation_count']:raise ValueError('GENERATION_COUNT_MISMATCH')
    result.update(run_id=RUN,trace_reaggregation='EXACT_MATCH',usage=dict(usage),
                  wall_seconds=read(PUB/'actual_execution_summary.json')['wall_seconds'],
                  generation=collection['generation_count'],clinical_accuracy='NOT_EVALUATED',
                  old_results_merged=False,global_stop=collection['global_stop_reason'])
    out=PUB/'manual_actual_results_v1';out.mkdir(parents=True,exist_ok=False)
    write(out/'summary.json',result);write(out/'request_audit.json',audit)
    from scripts.run_final_3b_evaluation import csv_file
    csv_file(out/'episodes.csv',result['episodes'],list(result['episodes'][0]))
    lines=['# Independent manual Run results','',f'Run `{RUN}`; registered 10 patients / 30 Episodes.',
           'B1 is CONTEXT_BUDGETED_HISTORY, not Full History. Source agreement is not clinical accuracy.',
           'No old Run results merged. Generation during this saved-trace analysis: 0.','',
           '|Condition|Completed|Failed|Not attempted|Diagnosis /10|Department /10|20% reached|',
           '|---|---:|---:|---:|---:|---:|---|']
    for c,s in result['conditions'].items():
        st=s['states'];fail=sum(v for k,v in st.items() if k in ('CONTRACT_FAILED','TIMEOUT','CONTEXT_BLOCKED'))
        lines.append(f"|{c}|{st.get('COMPLETED',0)}|{fail}|{st.get('NOT_ATTEMPTED',0)}|{s['Source_exact_registered_success_count']}|{s['Source_department_registered_success_count']}|{s['threshold_2_of_10_reached']}|")
    lines+=['','Paired diagnosis results (completed and scorable same-patient pairs only):',
            json.dumps(result['paired'],ensure_ascii=True,indent=2),'',
            'B3/B4 remain BLOCKED; this result does not admit Utility or likelihood artifacts.']
    with (out/'FINAL_MANUAL_RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    write(out/'preservation_verification.json',check(read(PR/'preservation_before.json')))
    print(json.dumps({'status':'SAVED_TRACE_ANALYZED','generation':0,'results':str(out)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','check','execute','monitor','analyze'])
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--server',action='store_true',help='GET version/tags/ps only')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-llm',action='store_true')
    args=parser.parse_args()
    if args.mode=='execute':
        if not (args.execute and args.allow_llm and args.authorization):
            parser.error('MANUAL_EXECUTE_ALLOW_LLM_AND_NEW_AUTHORIZATION_REQUIRED')
        execute(args.authorization)
    elif args.mode=='prepare':prepare()
    elif args.mode=='check':print(json.dumps(preflight(args.authorization,args.server),ensure_ascii=True))
    elif args.mode=='monitor':monitor()
    else:analyze()


if __name__=='__main__':main()
