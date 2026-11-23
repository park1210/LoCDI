# -*- coding: utf-8 -*-
"""Private saved-trace comparison, not a new model run or rescoring policy."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read
from app.locdi.routing_collector import atomic_json
A='MEMORY_RAG_LEGACY_RECOVERY_20261121_v1';B='ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1'

def trace_detail(t):
    if t['request_pointer']:
        ap=ROOT/t['request_pointer'].replace('_before.json','_after.json')
        a=read(ap) if ap.exists() else {};raw=a.get('response_envelope',{}).get('message',{}).get('content')
        try:response=json.loads(raw) if raw else {}
        except ValueError:response={'parse_status':'INVALID_SAVED_CONTENT'}
    else:response={}
    ds=t['decisions'];p=t['last_payload'];o=t['observation']
    return {'plan_id':t['plan_id'],'plan_fingerprint':t['plan_fingerprint'],'status':t['status'],
        'diagnosis':t['prediction']['diagnosis'],'certainty':t['prediction']['certainty'],'department':t['prediction']['department'],
        'source_exact':t['source_exact'],'memory':t['memory'],'public_evidence':p.get('evidence',[]),
        'evidence_citation_wire':response.get('evidence_indices',response.get('evidence_selection')),
        'knowledge':p.get('external_medical_knowledge',{'status':'NONE'}),
        'action_sequence':[{'minute':d['minute'],'action':d['action'],'target':d.get('target')} for d in ds],
        'final_minute':ds[-1]['minute'] if ds else None,'last_rationale':ds[-1].get('rationale') if ds else None,
        'prompt_tokens':t['tokens'].get('prompt'),'output_tokens':t['tokens'].get('output'),
        'last_assessment':o.get('working_assessment'),'source_reference':t['reference']}

def main():
    data=read(ROOT/'outputs/private/research_evaluation'/B/'baseline_trace_details.json')
    old={(r['original_ordinal'],r['condition']):r for r in data['old']};latest={(r['original_ordinal'],r['condition']):r for r in data['latest']}
    private=ROOT/'outputs/private/research_evaluation'/A;public=ROOT/'outputs/research_evaluation'/A;rows=[];detailed=[]
    for key,b in sorted(latest.items()):
        if key[1]!='B2':continue
        a=old[key];old_detail=trace_detail(a);new_detail=trace_detail(b)
        correct_lost=a['source_exact'] is True and b['source_exact'] is False
        transition={'panel_ordinal':b['panel_ordinal'],'old_plan_id':a['plan_id'],'latest_plan_id':b['plan_id'],
            'registered_new_plan_ids':['memrag_m2_s1_002','memrag_k2_s1_002'],'same_contract':False,
            'legacy_final_diagnosis':a['prediction']['diagnosis'],'latest_final_diagnosis':b['prediction']['diagnosis'],
            'legacy_source_exact':a['source_exact'],'latest_source_exact':b['source_exact'],
            'CORRECT_TO_INCORRECT':correct_lost,'INCORRECT_TO_CORRECT':a['source_exact'] is False and b['source_exact'] is True,
            'SUPPORTED_TO_UNRESOLVED':a['prediction']['certainty']=='supported' and b['prediction']['certainty']=='unresolved',
            'last_memory_ids_equal':[r['memory_id'] for r in a['memory']]==[r['memory_id'] for r in b['memory']],
            'last_public_evidence_equal':a['last_payload'].get('evidence')==b['last_payload'].get('evidence'),
            'legacy_final_minute':old_detail['final_minute'],'latest_final_minute':new_detail['final_minute'],
            'legacy_prompt_tokens':old_detail['prompt_tokens'],'latest_prompt_tokens':new_detail['prompt_tokens'],
            'legacy_output_tokens':old_detail['output_tokens'],'latest_output_tokens':new_detail['output_tokens'],
            'new_M2_or_K2_outcome':'NOT_RUN','new_knowledge_results':'NOT_RUN',
            'failure_or_explanation':'NOT_ATTEMPTED' if b['status']=='NOT_ATTEMPTED' else 'CAUSE_UNDETERMINED_CONTRACTS_DIFFER'}
        rows.append(transition);detailed.append({'transition':transition,'old_B2':old_detail,'latest_B2':new_detail,'latest_B0':trace_detail(latest[key[0],'B0'])})
    with (private/'B2_CORRECTNESS_REGRESSION_MATRIX_PRIVATE.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(private/'B2_REGRESSION_FULL_TRACE_COMPARISON.json',detailed)
    doc=ROOT/'docs/B0_B4_실험/16_Memory_RAG_회귀복구_검증'
    p=doc/'TEST_RESULTS.md';p.write_text(p.read_text(encoding='utf-8').replace('165 tests:169','169 tests:169'),encoding='utf-8')
    p=doc/'LEGACY_BASELINE_PRESERVATION.md';p.write_text(p.read_text(encoding='utf-8')+'\n환자별 실제 진단/전이·Memory·Evidence·인용·Knowledge·Action sequence·마지막 minute·Prompt/Output token을 포함한 Private 비교는 `outputs/private/research_evaluation/'+A+'/B2_REGRESSION_FULL_TRACE_COMPARISON.json` 및 B2_CORRECTNESS_REGRESSION_MATRIX_PRIVATE.csv다. Public CSV에는 익명 ordinal/상태·일치 여부/개수만 제공한다. 새 M2/K2의 진단 결과·전이는 NOT_RUN이며 과거 결과로 채우지 않았다.\n',encoding='utf-8')
    tests=read(public/'test_results_closeout.json');plans=read(public/'active_execution_plans.json')
    final={'status':'IMPLEMENTED_OFFLINE_VERIFIED_LIVE_NOT_RUN','tests':tests,'active_plans':plans['active'],
        'saved_request_hashes_verified':628,'selective_context_exact':321,'private_regression_rows':len(rows),
        'actual_LLM_generation':0,'utility_fitting':0,'clinical_accuracy':'NOT_EVALUATED',
        'supersedes_only_provisional_completion_summary':True}
    atomic_json(public/'completion_summary_closeout.json',final)
    p=doc/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n최종 기계 집계는 completion_summary_closeout.json, 실제 사용 계획은 active_execution_plans.json이다. 초기 completion_summary.json/001계획은 미실행 준비 이력으로만 보존한다.\n',encoding='utf-8')
    # Private findings also link the final reference timing instead of the initial missing event-store view.
    p=ROOT/'outputs/private/research_evaluation'/B/'DATA_QUALITY_FINDINGS.md'
    p.write_text(p.read_text(encoding='utf-8')+'\n\n최종 Source Condition 시점은 reference_timing_complete.json/analysis_with_reference_timing.json에서 보완했다. 48명 SAME_DAY_INTRADAY_NOT_VERIFIED,2명 SOURCE_CONDITION_AFTER_INDEX_DAY다. source_event_store에 Conditions가 없는 것은 Gold 비노출 경계이며 원천 부재로 해석하지 않는다. 초기 이벤트 기반 빈 timing 대신 이 결과를 사용한다. 실제 요청/수신은 ORIGINAL50_TEST_RESULTS.csv의 requested_conditions/event_bound_received_conditions 및 Public actual_action_linkage.json을 따른다.\n',encoding='utf-8')
    print(json.dumps({'private_regression_patients':len(rows),'CORRECT_TO_INCORRECT':sum(r['CORRECT_TO_INCORRECT'] for r in rows),'new_M2_K2_predictions':'NOT_RUN','LLM_generation':0,'fitting':0}))
if __name__=='__main__':main()
