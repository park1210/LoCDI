"""Model-free final export of preserved C4-38 and actual C4-39 records."""
import csv,json,re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha
from app.locdi.current_ablation_runner import reaggregate_trace
PUB=ROOT/'outputs/research_evaluation/c4_39_v1'
PR=ROOT/'outputs/private/research_evaluation/c4_39_v1'
SP=ROOT/'outputs/research_evaluation/c4_39_source_final_v1'
SR=ROOT/'outputs/private/research_evaluation/c4_39_source_final_v1/source_time_run'
RUN='c4_39_memory_b0_b2_001'


def no_gold(value):
    forbidden={'gold','evaluation','task_annotations','provenance','source_event_store','correctness','accepted_diagnoses','accepted_departments'}
    if isinstance(value,dict):return not forbidden.intersection(str(k).casefold() for k in value) and all(no_gold(v) for v in value.values())
    if isinstance(value,list):return all(no_gold(v) for v in value)
    return True


def main():
    # Exclusive publication: preserve all prior prepared/actual outputs.
    rows=read(PUB/'reaggregation/final_research_results.json')['rows']
    actual_source=(SP/'source_time_actual_results.json').exists()
    partial_source=(SP/'source_time_actual_partial.json').exists()
    runtime_files=list(SR.glob('Case*_runtime_before_comparison.json'))
    reference_files=list(SR.glob('Case*_reference_before_comparison.json'))
    comparisons=[read(p) for p in sorted(SR.glob('Case*_comparison.json'))
                 if re.fullmatch(r'Case[0-9]{3}_comparison\.json',p.name)]
    if actual_source:source=read(SP/'source_time_actual_results.json')
    elif partial_source:source=read(SP/'source_time_actual_partial.json')
    else:source={'status':'NOT_RUN_APPROVAL_PENDING','runtime_completed':0,'reference_completed':0,'compared_cases':0}
    numeric=[]
    for a in comparisons:
        numeric.extend([a['maximum_final_posterior_difference'],a['maximum_terminal_value_difference']])
        for r in a['action_identity_comparison']:numeric.extend(v for k,v in r.items() if k.startswith('maximum_'))
        for r in a['posterior_comparison']:numeric.extend([r['before_max_abs'],r['after_max_abs']])
    write(PUB/'final_source_runtime_verification.json',{'status':source['status'],
        'planned_visits':5,'runtime_completed':len(runtime_files),'reference_completed':len(reference_files),
        'compared_visits':len(comparisons),'Run_ID':'c4_39_source_time_003',
        'LLM_generation':0,'fitting':0,'retry':0,'source_gate':'DEVELOPMENT_ONLY','clinical_metrics':'NOT_EVALUATED',
        'compute_record':{k:source.get(k) for k in ['planner_calls','evaluated_states','wall_seconds']},
        'hospital_unresolved':sum(read(p)['hospital_unresolved'] for p in runtime_files),
        'full_Hospital_resource_simulation':False,'partial_record':source if partial_source else None})
    write(PUB/'final_source_runtime_comparison.json',{'status':'MATCH' if len(comparisons)==5 else 'INCOMPLETE_OR_NOT_RUN',
        'compared_cases':len(comparisons),'matched_decisions':sum(a['matched_decisions'] for a in comparisons),
        'extra_terminal_decisions':sum(a['extra_terminal_decisions'] for a in comparisons),
        'maximum_numeric_difference':max(numeric) if numeric else None,'tolerance':1e-12,
        'cases':comparisons,'first_mismatch_private_present':(SR/'first_mismatch.json').exists(),
        'policy_input_order_changed':False,'raw_candidate_order_may_differ':True})
    memory={'status':'NOT_RUN_APPROVAL_PENDING','registered_cases':10,'planned_episodes':20,'generation_count':0}
    if (PUB/RUN/'metrics.json').exists():
        c=read(PUB/'memory_execution_contract.json');p=read(PR/'memory_panel.json')
        metrics=reaggregate_trace(PR/RUN/'episodes.jsonl',c,p)
        assert metrics==read(PUB/RUN/'metrics.json')
        execution=read(PUB/RUN/'execution_manifest.json')
        requests=[];payloads_ok=True
        for n,path in enumerate(sorted((PR/RUN/'requests').glob('Episode*')),1):
            for f in sorted(path.glob('Request*_after.json')):
                a=read(f);before=f.with_name(f.name.replace('_after','_before'))
                b=read(before) if before.exists() else None
                if b:
                    u=json.loads(b['request']['messages'][1]['content']);payloads_ok &= no_gold(u)
                    mem=u.get('observation',{}).get('longitudinal_memory',{})
                else:mem={}
                envelope=a.get('response_envelope',{})
                requests.append({'episode_ordinal':n,'generation_attempted':a.get('generation_attempted',True),
                    'HTTP_status':a.get('HTTP_status'),'status':a['status'],'error_type':a.get('error_type'),
                    'prompt_tokens':envelope.get('prompt_eval_count'),'output_tokens':envelope.get('eval_count'),
                    'actual_context':a.get('identity_after',{}).get('context_length'),
                    'memory_records':len(mem.get('records',[])),'memory_mode':mem.get('mode'),
                    'wall_seconds':a.get('wall_seconds')})
        memory={'status':execution['status'],'registered_cases':10,'planned_episodes':20,
            'conditions':metrics['conditions'],'generation_count':execution['generation_count'],
            'independent_reaggregation':'EXACT_MATCH','requests':requests,'structured_Gold_private_key_scan':'PASS' if payloads_ok else 'FAIL',
            'clinical_accuracy':'NOT_EVALUATED','annotation_precision_recall':'NOT_EVALUATED',
            'scope':'B0/B2 supplementary development comparison; not official B0-B4/full50',
            'full_temporal_payload_semantics':'Core projection/validator contract; not independently clinically reviewed'}
        write(PUB/'memory_independent_reaggregation.json',metrics)
        for condition,m in metrics['conditions'].items():
            for key in ['supported_answer_agreement','final_department_agreement']:
                a=m[key]
                rows.append({'group':'C4_39_Memory_ER_development','condition':condition,'metric':key,'value':a['value'],
                    'patients':10,'visits':10,'denominator':a['denominator'],'source':f'c4_39_v1/{RUN}/metrics.json',
                    'scope':'Same fixed10 Original cases; completed supported denominator; Source strings, not clinical truth',
                    'clinical_metrics':'NOT_EVALUATED'})
            rows.append({'group':'C4_39_Memory_ER_development','condition':condition,'metric':'completed_fraction',
                'value':m['completed']/m['registered'],'patients':10,'visits':10,'denominator':m['registered'],
                'source':f'c4_39_v1/{RUN}/metrics.json','scope':'Registered denominator includes failed/not-attempted',
                'clinical_metrics':'NOT_EVALUATED'})
    write(PUB/'memory_actual_evaluation_results.json',memory)
    rows.append({'group':'C4_39_Source_Runtime','condition':'Frozen_M5_H2','metric':'verified_visits',
        'value':len(comparisons),'patients':'','visits':5,'denominator':5,
        'source':'c4_39_source_final_v1/source_time_actual_results.json' if actual_source else 'c4_39_source_final_v1/source_time_actual_partial.json',
        'scope':'Technical equivalence count; not Source agreement or clinical performance','clinical_metrics':'NOT_EVALUATED'})
    with (PUB/'final_research_results.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(PUB/'final_research_results.json',{'rows':rows,'old_rows_preserved':35,'mixed_matched_ablation':False,
        'new_Source_run':source['status'],'new_Memory_run':memory['status'],'clinical_metrics':'NOT_EVALUATED'})
    private_pins={p.relative_to(ROOT).as_posix():sha(p) for d in [PR,SR] for p in d.rglob('*') if p.is_file()}
    write(PR/'final_actual_trace_index.json',private_pins)
    write(PUB/'final_reproducibility_manifest.json',{'new_code':{n:sha(ROOT/n) for n in [
        'app/locdi/source_runtime_comparison.py','scripts/run_current_source_time_probe.py',
        'scripts/run_final_memory_evaluation.py','scripts/finalize_c4_39_results.py']},
        'model_and_Prompt_Registry_decoding':read(PUB/'memory_execution_contract.json')['common'],
        'M5_SHA':sha(ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'),
        'Source_protocol':read(SP/'source_time_execution_protocol.json')['fingerprint'],
        'Memory_contract':read(PUB/'memory_execution_contract.json')['fingerprint'],
        'trace_index_private_SHA':sha(PR/'final_actual_trace_index.json'),
        'saved_ER_trace_reaggregation':memory.get('independent_reaggregation','NOT_RUN'),
        'LLM_repeated_generation_reproducibility':'NOT_TESTED_NO_AUTOMATIC_REPEAT',
        'aggregation_version':'c4_39.final_results.v1','saved_result_SHA':sha(PUB/'final_research_results.json')})
    print(json.dumps({'Source_compared':len(comparisons),'Memory_status':memory['status'],
        'actual_generation':memory['generation_count'],'result_rows':len(rows)}))


if __name__=='__main__':main()
