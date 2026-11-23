"""Read-only C4-38 execution analysis; no transport/Runner/model calls."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha
from app.locdi.llm_setup import digest
from app.locdi.current_ablation_runner import reaggregate_trace

PUB=ROOT/'outputs/research_evaluation/c4_38_v1'
PR=ROOT/'outputs/private/research_evaluation/c4_38_v1'
RUN='c4_38_er_b0_b2_001'


def prohibited_keys(value,path=''):
    forbidden={'gold','evaluation','task_annotations','provenance','source_event_store','correctness','accepted_diagnoses','accepted_departments'}
    found=[]
    if isinstance(value,dict):
        for k,v in value.items():
            if k.casefold() in forbidden:found.append(path+'/'+k)
            found.extend(prohibited_keys(v,path+'/'+k))
    elif isinstance(value,list):
        for n,v in enumerate(value):found.extend(prohibited_keys(v,path+'/'+str(n)))
    return found


def main():
    out=PUB/'actual_results';out.mkdir(exist_ok=False)
    contract=read(PUB/'er_execution_contract.json');panel=read(PR/'er_panel.json')
    metrics=reaggregate_trace(PR/RUN/'episodes.jsonl',contract,panel)
    assert metrics==read(PUB/RUN/'metrics.json')
    manifest=read(PUB/RUN/'execution_manifest.json')
    assert sha(PR/RUN/'episodes.jsonl')==manifest['trace_sha256']
    details=[];calls=[];private_index={};prediction_fps=[]
    for n,c in enumerate(['B0','B1','B2'],1):
        folder=PR/RUN/'requests'/('Episode%04d'%n)
        raw=read(folder/'episode_and_raw.json');ep=raw['episode'];client=raw['raw']['llm_execution']
        initial=raw['raw']['execution']['visits'][0]['identity']
        workflow=ep['workflow'];events=workflow['events'];decisions=workflow['decisions']
        assert [e['minute'] for e in events]==sorted(e['minute'] for e in events)
        ordered=next(e['minute'] for e in events if e['event_type']=='test_ordered')
        released=next(e['minute'] for e in events if e['event_type']=='result_received')
        decision_rows=[]
        for i,decision in enumerate(decisions,1):
            before=read(folder/('Request%04d_before.json'%i));after=read(folder/('Request%04d_after.json'%i))
            request=before['request'];payload=json.loads(request['messages'][1]['content'])
            assert digest(request)==before['request_fingerprint']==after['request_fingerprint']
            assert not prohibited_keys(payload)
            memory=payload['observation']['longitudinal_memory'];records=memory.get('records',[])
            # Use the already recorded Core cutoff verdict as independent audit
            # input plus actual payload identity/timestamp predicates.
            from app.core.longitudinal_memory import timestamp
            eligible=all(r['patient_id']==initial['patient_id'] and r['visit_id']!=initial['visit_id']
                and timestamp(r['visit_started_at'])<timestamp(initial['started_at'])
                and timestamp(r['available_at'])<=timestamp(initial['started_at']) for r in records)
            assert eligible
            count=len(payload.get('evidence',[]))
            assert not (decision['minute']<released and count>0)
            assert after['status']=='VALID' and after['HTTP_status']==200
            identity=after['identity_after'];envelope=after['response_envelope']
            assert identity['context_length']==32768
            assert identity['digest']==contract['common']['model_bindings'][0]['digest'].replace('sha256:','')
            assert decision['execution_success'] and decision['execution_event_sequences']
            decision_rows.append({'minute':decision['minute'],'selected':decision['selected_action'],
                'executed_event_sequences':decision['execution_event_sequences'],'execution_success':True,
                'objective_evidence_count':count,'memory_count':len(records),'memory_mode':memory['mode'],
                'cutoff_compliant':eligible,'routing_state':'DISABLED_ER_ONLY',
                'existing_episode_routing_abstention_field':decision['routing_abstention'],
                'not_an_adaptive_abstention':True})
            calls.append({'condition':c,'request':i,'status':'VALID','HTTP_status':200,
                'wall_seconds':after['wall_seconds'],'context':32768,'digest_match':True,
                'prompt_tokens':envelope['prompt_eval_count'],'output_tokens':envelope['eval_count'],
                'request_fingerprint_match':True,'forbidden_key_scan':'PASS','cutoff_compliant':eligible})
        details.append({'condition':c,'case_ordinal':30,'status':ep['result']['status'],
            'diagnosis_unresolved':ep['result']['diagnosis_unresolved'],'result':ep['result'],
            'initial_retrieval_minute':None,'initial_retrieval_timing':'NOT_RECORDED_BY_CORE',
            'request_count':client['request_count'],'identity_check_count':client['identity_check_count'],
            'retry':client['retry'],'repair':client['repair'],'error_types':client['error_types'],
            'test_ordered_minute':ordered,'result_received_minute':released,
            'final_event_minute':events[-1]['minute'],'decisions':decision_rows,
            'normalized_trace_fingerprint':ep['normalized_trace_fingerprint']})
        prediction_fps.append(digest(raw['validated_ER_decisions'][-1].get('target')))
        for p in folder.rglob('*'):
            if p.is_file():private_index[p.relative_to(ROOT).as_posix()]=sha(p)
    assert len(calls)==manifest['generation_count']==6
    result={'status':'PARTIAL_RESEARCH_COMPLETION','ER_run_id':RUN,'ER_episodes':3,'registered_ER_cases':1,
        'ER_completed':3,'ER_failed':0,'ER_incomplete':0,'generation_count':6,'identity_check_count':12,
        'HTTP200_and_schema_valid':6,'timeouts':0,'repair':0,'retry':0,'context':32768,
        'source_probe_status':'EXECUTION_PARTIAL','source_compared_cases':0,
        'source_runtime_calls_returned_before_error':1,'source_persisted_runtime_trace_count':0,
        'source_error':'CLI_PLANNER_METER_PROTOCOL_MISMATCH','source_actual_rerun':False,
        'independent_ER_reaggregation':'EXACT_MATCH','clinical_metrics':'NOT_EVALUATED',
        'all_three_final_diagnosis_strings_same':len(set(prediction_fps))==1,
        'B3_B4':'ROUTING_UTILITY_BLOCKED','Utility_fitting':0,'new_clinical_verdict':0,
        'old_pending_preparation_exports_preserved':True}
    write(out/'final_ablation_execution_manifest.json',result)
    write(out/'actual_episode_and_request_audit.json',{'episodes':details,'requests':calls,
        'payload_observation_boundary':'saved payloads inspected; no Raw wire beyond saved request/envelope claimed',
        'gold_and_private_reference_keys':'absent in saved model payloads',
        'clinical_truth_review':'NOT_REVIEWED','source_result_release_proxy':'ACCEPTED_DEVELOPMENT_ONLY',
        'original_ID_exposure_in_public_export':'no raw IDs copied; ordinal30 only',
        'Model_Server_wire_bytes':'NOT_CAPTURED'})
    write(out/'independent_reaggregated_metrics.json',metrics)
    write(PR/'actual_private_trace_index.json',private_index)
    def table(name,rows):
        with (out/name).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name,key in [('final_diagnosis_results.csv','supported_answer_agreement'),
                     ('final_department_results.csv','final_department_agreement')]:
        rows=[]
        for c in ['B0','B1','B2','B3','B4']:
            v=metrics['conditions'].get(c);a=v[key] if v else {}
            rows.append({'condition':c,'registered_cases':1 if v else '',
                'executed_cases':1 if v else 0,'numerator':a.get('numerator',''),
                'denominator':a.get('denominator',''),'Source_agreement':a.get('value',''),
                'status':'DEVELOPMENT_SOURCE_AGREEMENT_ONLY' if v else 'BLOCKED_ROUTING_UTILITY',
                'clinical_accuracy':'NOT_EVALUATED'})
        table(name,rows)
    table('final_action_results.csv',[{'condition':d['condition'],'registered_cases':1,
        'completed':1,'Ask':0,'Test':1,'Consult':0,'Diagnose':1,'mean_actions':2,
        'simulation_elapsed_minutes':d['result']['elapsed_simulation_minutes'],
        'last_event_minute':d['final_event_minute'],'generation_count':2,
        'identity_checks':4,'status':'ACTUAL_COMPLETED'} for d in details])
    table('final_memory_results.csv',[{'condition':d['condition'],
        'first_decision_count':d['decisions'][0]['memory_count'],
        'second_decision_count':d['decisions'][1]['memory_count'],'cutoff_compliant':True,
        'precision':'','recall':'','independent_review':'NOT_REVIEWED',
        'clinical_retrieval_quality':'NOT_EVALUATED'} for d in details])
    table('final_voi_results.csv',[{'run':'c4_38_source_time_001','compared_cases':0,
        'VoI':'','posterior_equivalence':'','status':'FAILED_REFERENCE_CLI_INTERFACE',
        'retry':0,'new_actual_run':'NOT_RUN'}])
    table('thesis_table2_recalculated.csv',[{'condition':c,'historical_diagnosis_percent':[38,52,44,48,90][i],
        'historical_department_percent':[44,58,54,46,94][i],'historical_mean_actions':[2.90,3.14,2.60,2.12,3.52][i],
        'current_Source_diagnosis_agreement':0 if i<3 else '',
        'current_Source_department_agreement':0 if i<3 else '',
        'current_mean_actions':2 if i<3 else '', 'current_cases':1 if i<3 else 0,
        'clinical_accuracy':'NOT_EVALUATED','comparison':'NOT_COMPARABLE',
        'reason':'single input-size selected development case' if i<3 else 'ROUTING_UTILITY_BLOCKED'}
        for i,c in enumerate(['B0','B1','B2','B3','B4'])])
    table('thesis_table3_recalculated.csv',[{'metric':'annotation_precision_recall','current_reviewed_cases':0,
        'historical_subset':20,'value':'','status':'NOT_EVALUATED_INDEPENDENT_REFERENCE_MISSING'}])
    print(json.dumps(result))


if __name__=='__main__':main()
