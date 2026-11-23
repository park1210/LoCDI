"""D3 actual-source replay with TEST_ONLY deterministic client, no clinical claim."""
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.locdi.source_replay import load_replay_cases,PROXY,STRICT,PUBLIC,PRIVATE,atomic_rows,sha
from app.locdi.llm_setup import ROOT
from app.locdi.case_adapter import adapt_case
from app.locdi.run_simulation import run_case
from app.locdi.episode_exporter import ExportSession,build_episode,export_episode,export_summary
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.longitudinal_memory import timestamp
from test_simulation import SourceActionClient,smoke_options
from smoke_support import _strings,_keys


def boundary(case,context,result,client):
    if result['execution'] is None:return {'checked':False,'reason':'NO_CORE_EXECUTION'}
    final=result['execution']['visits'][0]['final_state']
    delivered={e['payload']['requested_test']:e['minute'] for e in final['event_log'] if e['event_type']=='result_received'}
    ordered={e['payload']['requested_test']:e['minute'] for e in final['event_log'] if e['event_type']=='test_ordered'}
    events=context.runtime_case['current_encounter_events'];visible_tests=set()
    forbidden={'gold','evaluation','task_annotations','provenance','source_event_store','data_replay','current_encounter_events'}
    for call in client.calls:
        payload=call['payload'];assert not _keys(payload)&forbidden
        minute=payload['observation']['elapsed_minutes'];texts=_strings(payload)
        for event in events:
            if any(event['evidence'] in text for text in texts):
                assert minute>=event['available_minute']
                if event['action']=='Test':
                    assert event['target'] in delivered and minute>=delivered[event['target']]>=ordered[event['target']]
                    visible_tests.add(event['target'])
        for record in payload['observation']['longitudinal_memory']['records']:
            assert record['patient_id']==case['patient_id']
            assert record['visit_id']!=case['agent_visible']['index_encounter_id']
            assert timestamp(record['available_at'])<=timestamp(case['agent_visible']['index_encounter_start'])
    assert all(r['cutoff_compliant'] for r in result['telemetry']['retrievals'])
    return {'checked':True,'violations':0,'test_sources':sum(e['action']=='Test' for e in events),
            'tests_ordered':len(ordered),'test_results_received':len(delivered),'test_results_observed':len(visible_tests),
            'remaining_source_tests_not_exercised':sum(e['action']=='Test' for e in events)-len(delivered),
            'payloads_checked':len(client.calls),'retrieval_count':len(result['telemetry']['retrievals'])}


def main():
    cases=load_replay_cases(policy=PROXY,accept_development_proxy=True)
    manifest=json.loads((PUBLIC/'dataset_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/e['path'])==e['sha256'] for e in manifest['files'].values())
    root=ROOT/'outputs/dataset_replay/simulation_source_replay_v1'
    if root.exists():raise FileExistsError('REPLAY_OUTPUT_ALREADY_EXISTS')
    root.mkdir(parents=True,exist_ok=False)
    rounds=[];checks=[];runner_count=scheduler_count=0
    for repeat in range(2):
        session=ExportSession();docs=[];round_checks=[]
        for ordinal,case in enumerate(cases,1):
            context=adapt_case(case);client=SourceActionClient()
            with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner, \
                 patch('app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler, \
                 patch('app.core.hospital_llm_policy.OllamaClient',side_effect=AssertionError('NO_REAL_MODEL')):
                result=run_case(case,options=smoke_options(client))
            runner_count+=runner.call_count;scheduler_count+=scheduler.call_count
            check=boundary(case,context,result,client);round_checks.append(check)
            document=build_episode(result,context=context,session=session,case_ordinal=ordinal);docs.append(document)
            public=json.dumps(document,sort_keys=True)
            assert case['patient_id'] not in public and case['agent_visible']['index_encounter_id'] not in public
            assert document['clinical_metrics']=='NOT_EVALUATED'
            path=export_episode(result,'episodes/round{}_case{:03d}.json'.format(repeat+1,ordinal),
                context=context,session=session,case_ordinal=ordinal,output_root=root)
            assert json.loads(path.read_text(encoding='utf-8'))==document
        export_summary({'episodes':docs},'metrics/round{}_summary.jsonl'.format(repeat+1),output_root=root)
        rounds.append(docs);checks.append(round_checks)
    matches=[a['normalized_trace_fingerprint']==b['normalized_trace_fingerprint'] for a,b in zip(*rounds)]
    try:load_replay_cases(policy=STRICT)
    except ValueError as exc:strict=str(exc)
    else:raise AssertionError('STRICT_NOT_BLOCKED')
    report={'schema_version':'locdi.D3.offline_replay.v1','dataset_version':manifest['dataset_version'],
        'manifest_fingerprint':manifest['manifest_fingerprint'],'purpose':'DEVELOPMENT_ONLY_TEST_ONLY_CLIENT',
        'clinical_metrics':'NOT_EVALUATED','research_readiness':False,'new_llm_generations':0,
        'policy':PROXY,'strict_source':strict,'unique_cases':len(cases),'execution_count':sum(map(len,rounds)),
        'actual_runner_calls':runner_count,'actual_scheduler_calls':scheduler_count,
        'status_counts':dict(Counter(d['result']['status'] for d in rounds[0])),
        'deterministic_matches':sum(matches),'boundary_checks':checks,
        'cases':[{'case_ordinal':i,'result':a['result'],'trace_fingerprint':a['normalized_trace_fingerprint'],
                  'repeat_trace_fingerprint':b['normalized_trace_fingerprint'],'deterministic':same,
                  'validation':{v.validator:v.status for v in adapt_case(case).validation_results}}
                 for i,(a,b,same,case) in enumerate(zip(rounds[0],rounds[1],matches,cases),1)]}
    atomic_rows(root,'offline_validation.json',report,jsonl=False)
    atomic_rows(PUBLIC,'offline_validation.json',report,jsonl=False)
    print(json.dumps({k:v for k,v in report.items() if k not in {'cases','boundary_checks'}},ensure_ascii=True))
    return 0 if all(matches) and all(c.get('checked') for r in checks for c in r) else 1


if __name__=='__main__':
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:raise RuntimeError('EXTERNAL_CALL_DISABLED')
    sys.addaudithook(guard)
    raise SystemExit(main())
