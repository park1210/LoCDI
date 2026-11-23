"""Read-only Phase 2.7 verification of a saved ER pilot; never executes it."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.llm_setup import ROOT, digest, load_setup, validate_setup
from app.locdi.dataset_loader import load_cases
from app.core.longitudinal_memory import timestamp
from app.locdi.episode_exporter import _write
from app.locdi.routing_collector import _id

DEFAULT_RUN = '20261029000000000000000000002601'


def verify(run_id=DEFAULT_RUN):
    if not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise ValueError('INVALID_RUN_ID')
    root = (ROOT/'outputs/development_pilot'/run_id).resolve()
    if not root.is_relative_to((ROOT/'outputs/development_pilot').resolve()):
        raise ValueError('SOURCE_REDIRECTED')
    required = ('reports/preflight.json', 'reports/started.json', 'reports/pilot.json',
                'reports/runtime.json', 'episodes/Case001.json', 'metrics/summary.jsonl')
    missing = [name for name in required if not (root/name).is_file()]
    if missing:
        return {'schema_version':'locdi.pilot_verification.v1','run_id':run_id,
            'classification':'EXECUTION_NOT_STARTED' if not root.exists() else 'NOT_DETERMINABLE_FROM_AVAILABLE_FILES',
            'missing_files':missing,'new_llm_calls':0,'research_evaluation_ready':False}
    files = sorted(root.rglob('*.json')) + sorted(root.rglob('*.jsonl'))
    docs = {p.relative_to(root).as_posix():json.loads(p.read_text(encoding='utf-8')) for p in files}
    pre, started, pilot, audit, ep, summary = [docs[name] for name in required]
    checks = {}
    def check(name, condition):
        checks[name] = 'PASS' if condition else 'FAIL'
    plan = pilot['plan']; metadata = pilot['llm_execution']
    check('plan_fingerprint', digest(plan)==pilot['plan_fingerprint']==pre['plan_fingerprint']==started['plan_fingerprint'])
    check('preflight_plan', pre['plan']==plan)
    check('run_identity', all(d['run_id']==run_id for d in (pre,started,pilot,audit,ep)))
    check('configuration_fingerprint',digest(ep['configuration'])==ep['configuration_fingerprint']==audit['configuration_fingerprint'])
    trace_keys = ['case_ref','patient_ref','visit_ref','configuration_fingerprint','workflow','memory','capabilities','result','llm_execution']
    check('normalized_trace_fingerprint',digest({k:ep[k] for k in trace_keys})==ep['normalized_trace_fingerprint'])
    role_copy = deepcopy(ep['decision_audit']); role_fp=role_copy.pop('fingerprint')
    check('decision_audit_fingerprint',digest(role_copy)==role_fp)
    check('runtime_episode',audit['workflow']==ep['workflow'] and audit['memory']==ep['memory']
          and audit['roles']==ep['decision_audit'] and audit['terminal']==ep['result'] and audit['capabilities']==ep['capabilities'])
    check('summary_identity',summary['episode_count']==1 and summary['episodes'][0]['run_id']==run_id)
    row=summary['episodes'][0]
    check('summary_episode',all(row[k]==ep[k] for k in ('configuration_fingerprint','normalized_trace_fingerprint','llm_execution','case_ref'))
          and all(row[k]==ep['result'][k] for k in ('status','diagnosis_unresolved','elapsed_simulation_minutes','action_counts')))
    check('model_metadata',metadata==ep['llm_execution']==row['llm_execution']
          and metadata['settings_fingerprint']==plan['settings_fingerprint']==ep['configuration']['llm_settings_fingerprint'])
    check('request_budget',0<=metadata['request_count']<=plan['generation']['max_requests']==5)
    case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0]
    check('source_case001',digest(dict(case))==plan['case_fingerprint'] and ep['case_ref']=='Case001'
          and hashlib.sha256((ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_bytes()).hexdigest()==plan['source_fingerprint'])
    config=load_setup(); config['max_requests']=5
    setup=validate_setup(config)
    check('current_local_binding',setup['static_readiness']=='READY'
          and setup['prepared']['fingerprint']==plan['settings_fingerprint']
          and digest(setup['prepared']['registry'])==plan['registry_fingerprint']
          and setup['prepared']['prompt_fingerprints']==plan['prompt_fingerprints'])
    checkpoints=[]; before_files=sorted((root/'private').glob('*_before.json'))
    after_files=sorted((root/'private').glob('*_after.json'))
    error_files=sorted((root/'private').glob('*_error.json'))
    for file in before_files:
        before=docs[file.relative_to(root).as_posix()]; i=before['attempt']
        after=docs.get('private/{:03d}_after.json'.format(i))
        if after is None: continue
        check('checkpoint_{:03d}'.format(i),before['phase']=='before' and after['phase']=='after'
              and before['llm_execution']['request_count']==i-1 and after['llm_execution']['request_count']==i
              and before['llm_execution']['identity_check_count']==i-1 and after['llm_execution']['identity_check_count']==i
              and before['system_fingerprint'] in plan['prompt_fingerprints'].values()
              and not after['llm_execution']['error_types']
              and datetime.fromisoformat(before['at'])<=datetime.fromisoformat(after['at']))
        checkpoints.append({'attempt':i,'started_at':before['at'],'ended_at':after['at'],
            'checkpoint_interval_seconds':(datetime.fromisoformat(after['at'])-datetime.fromisoformat(before['at'])).total_seconds(),
            'request_count':after['llm_execution']['request_count'],'identity_check_count':after['llm_execution']['identity_check_count'],
            'system_fingerprint':before['system_fingerprint'],'input_fingerprint':before['input_fingerprint'],
            'response_fingerprint':after['response_fingerprint']})
    check('checkpoint_completeness',len(before_files)==len(after_files)==len(checkpoints)==metadata['request_count'] and not error_files)
    check('last_checkpoint_metadata',bool(after_files) and docs[after_files[-1].relative_to(root).as_posix()]['llm_execution']==metadata)
    events=ep['workflow']['events']; decisions=ep['workflow']['decisions']; roles=ep['decision_audit']['records']
    expected={'request_test':'order_test','ask_patient':'ask_history','consult_specialty':'request_specialty','diagnose':'er_finalize'}
    linked=True
    for d in decisions:
        matches=[e for e in events if e['sequence'] in d['execution_event_sequences']]
        linked &= bool(matches) and all(e['minute']==d['minute'] and e['executed_action']==expected[d['selected_action']]
            and (d['selected_action']=='diagnose' or e['target_ref']==d['target_ref']) for e in matches)
    check('selected_executed_actions',linked and all(d['execution_success'] is True for d in decisions))
    check('decision_contract',all(r['selected_in_candidates'] is True and r['action_mapping_matches'] is True
          and r['model_binding_status']=='MATCHED' and r['entry_role']==r['effective_model_role']=='er' for r in roles))
    check('action_counts',dict(Counter(d['selected_action'] for d in decisions))==ep['result']['selected_action_counts']
          and dict(Counter(e['executed_action'] for e in events if e['executed_action'] in expected.values()))==ep['result']['action_counts'])
    check('time_order', [e['minute'] for e in events]==sorted(e['minute'] for e in events))
    deliveries=[]
    for event in events:
        if event['event_type']!='result_received': continue
        ordered=[e for e in events if e['event_type']=='test_ordered' and e['target_ref']==event['target_ref']]
        began=[e for e in events if e['event_type']=='test_started' and e['target_ref']==event['target_ref']]
        good=len(ordered)==len(began)==1 and ordered[0]['minute']<=began[0]['minute']<=event['minute']
        check('delivery_'+event['target_ref'],good)
        deliveries.append({'target_ref':event['target_ref'],'ordered_minute':ordered[0]['minute'] if ordered else None,
            'started_minute':began[0]['minute'] if began else None,'received_minute':event['minute']})
    memory={}; index_start=timestamp(case['agent_visible']['index_encounter_start'])
    for visit in case['agent_visible']['past_visits']:
        for record in visit['records']:
            if record['memory_id'] not in memory:
                memory[record['memory_id']]=('Memory{:04d}'.format(len(memory)+1),record)
    lookup={ref:record for ref,record in memory.values()}
    retrievals=ep['memory']['retrievals']; valid=True
    for r in retrievals:
        valid &= r['cutoff_compliant'] is True and len(r['memory_refs'])==5
        for ref in r['memory_refs']:
            record=lookup.get(ref)
            valid &= bool(record) and record['patient_id']==case['patient_id'] and record['visit_id']!=case['agent_visible']['index_encounter_id']
            if record:
                valid &= timestamp(record['visit_started_at'])<index_start and timestamp(record['observed_at'])<=timestamp(record['available_at'])<=index_start
    check('memory_source_cutoff_recheck',valid and bool(retrievals))
    forbidden={'patient_id','visit_id','case_id','gold','evaluation','provenance','task_annotations','source_event_store','raw_response','user_prompt','system_prompt'}
    def keys(value):
        if isinstance(value,dict):return set(value)|set().union(*(keys(v) for v in value.values()),set())
        if isinstance(value,list):return set().union(*(keys(v) for v in value),set())
        return set()
    text=json.dumps(docs)
    needles={case['patient_id'],case['case_id'],case['agent_visible']['index_encounter_id']}
    needles.update(memory)
    needles.update(v['visit_id'] for v in case['agent_visible']['past_visits'])
    needles.update(e['evidence'] for e in case['environment']['current_encounter_events'])
    check('public_private_stored_boundary',not keys(docs)&forbidden and not any(x and x in text for x in needles))
    check('completed_terminal',pilot['status']==pilot['core_status']==ep['result']['status']=='completed'
          and ep['result']['finalized'] is True and not pilot['blocker_codes'] and not metadata['error_types'])
    baselines=[]
    for tag in 'ABCD':
        p=ROOT/'outputs/test_only/phase25_runtime_v1/episodes'/('TEST_ONLY_'+tag+'.json')
        b=json.loads(p.read_text(encoding='utf-8'))
        baselines.append({'scenario':tag,'status':b['result']['status'],'result':b['result'],
            'decision_count':len(b['workflow']['decisions']), 'actions':[d['selected_action'] for d in b['workflow']['decisions']],
            'retrieval_count':len(b['memory']['retrievals']),'generation_requests':'NOT_APPLICABLE_OFFLINE',
            'configuration_fingerprint':b['configuration_fingerprint']})
    return {'schema_version':'locdi.pilot_verification.v1','run_id':run_id,
        'classification':'DEVELOPMENT_PILOT_COMPLETED' if all(v=='PASS' for v in checks.values()) else 'VERIFICATION_FAILED_REQUIRES_REVIEW',
        'checks':checks,'missing_files':missing,'source_file_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'plan_fingerprint':pilot['plan_fingerprint'],'configuration_fingerprint':ep['configuration_fingerprint'],
        'normalized_trace_fingerprint':ep['normalized_trace_fingerprint'],'llm_execution':metadata,
        'raw_server_digest_response':'NOT_CHECKED_NOT_SAVED','historical_wire_payload':'NOT_CHECKED_NOT_SAVED',
        'raw_response_schema_revalidation':'NOT_CHECKED_NOT_SAVED','explicit_request_decision_correlation_id':'NOT_RECORDED',
        'checkpoints':checkpoints,'decisions':decisions,'result_deliveries':deliveries,'memory':ep['memory'],
        'terminal':ep['result'],'decision_audit':ep['decision_audit'],
        'pilot_wall_seconds':(datetime.fromisoformat(pilot['ended_at'])-datetime.fromisoformat(pilot['started_at'])).total_seconds(),
        'episode_wall_seconds':(datetime.fromisoformat(ep['ended_at'])-datetime.fromisoformat(ep['started_at'])).total_seconds(),
        'baselines':baselines,'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False,
        'new_llm_calls':0,'utility_fit_count':0,'clinical_verdicts_generated':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default=DEFAULT_RUN)
    parser.add_argument('--save',metavar='NEW_ANALYSIS_ID')
    args=parser.parse_args(argv)
    try:
        result=verify(args.run_id)
        if args.save:
            _write(result,'reports/'+_id(args.save)+'.json',ROOT/'outputs','reports')
        print(json.dumps(result,ensure_ascii=True,allow_nan=False))
        return 0 if result['classification']=='DEVELOPMENT_PILOT_COMPLETED' else 2
    except Exception:
        print(json.dumps({'classification':'VERIFICATION_BLOCKED','reason':'SAVED_CONTRACT_OR_STORAGE_ERROR','new_llm_calls':0}))
        return 2


if __name__=='__main__':
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:
            raise RuntimeError('External calls disabled')
    sys.addaudithook(guard)
    raise SystemExit(main())
