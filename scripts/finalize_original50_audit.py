# -*- coding: utf-8 -*-
"""Finish private readable inventory and public aggregates; no runtime changes."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,original_cases
from app.locdi.routing_collector import atomic_json
NS='ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS
DOC=ROOT/'docs/B0_B4_실험/17_Original50_환자별_진단증상검사_분석'
def table(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def safe(v):return str(v).replace('|','\\|').replace('\n',' ')

def main():
    summary=read(PUB/'summary.json');cases=original_cases();trace=read(PR/'baseline_trace_details.json')
    latest={(t['original_ordinal'],t['condition']):t for t in trace['latest']};old={(t['original_ordinal'],t['condition']):t for t in trace['old']}
    base=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001'
    bindings=read(base/'bindings.json');index=read(base/'case_index.json');byoriginal={n:bindings[ref] for ref,n in index.items()}
    with (PR/'ORIGINAL50_TEST_RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:labs=list(csv.DictReader(f))
    groups=defaultdict(list);coverage=Counter();stats=defaultdict(lambda:{'patients':set(),'rows':0,'missing':0,'m5_rows':0,'proxy_eligible':0})
    for row in labs:
        n=int(row['patient_ordinal']);row['requested_conditions']='';row['event_bound_received_conditions']=''
        if row['scope']=='CURRENT_ENCOUNTER':
            br=[r for r in byoriginal.get(n,{}).get('rows',[]) if r['source_event']['event_id']==row['source_event']]
            requested=[];received=[]
            for co in ('B0','B1','B2','B3','B4'):
                t=latest.get((n,co))
                if not t:continue
                for b in br:
                    actions=('order_test','request_test') if b['kind']=='Test' else ('ask_history','ask_patient')
                    if any(d.get('action') in actions and d.get('target')==b['target'] for d in t['decisions']):requested.append(co)
                    o=t['observation']
                    if (b['kind']=='Test' and b['target']+': '+b['evidence'] in o.get('objective_findings',[])) or (b['kind']=='Ask' and any(h.get('question')==b['target'] and h.get('answer')==b['evidence'] and h.get('answer_status')!='unknown' for h in o.get('history',[]))):received.append(co)
            row['requested_conditions']=';'.join(sorted(set(requested)));row['event_bound_received_conditions']=';'.join(sorted(set(received)))
            row['B0_B4_receipts']=row['event_bound_received_conditions']
            row['exposure']='RECEIVED_IN_SAVED_LLM_PAYLOAD_EVENT_BOUND' if received else 'NOT_AVAILABLE' if not row['value'] else 'NOT_RUN' if n not in byoriginal else 'NOT_RELEASED'
            groups[n].append(row);coverage['current_rows']+=1
            coverage['current_missing_values']+=not bool(row['value']);coverage['M5_test_rows']+=row['M5_Test_mappable']=='True'
            coverage['patient_report_pain_rows']+=row['kind']=='PATIENT_REPORT_ASK_ONLY'
            coverage['current_rows_with_saved_receipt']+=bool(received)
            coverage['current_rows_outside_fixed_evaluation_panel']+=n not in byoriginal
            s=stats[(row['code'],row['description'],row['unit'])];s['patients'].add(n);s['rows']+=1
            s['missing']+=not bool(row['value']);s['m5_rows']+=row['M5_Test_mappable']=='True'
            s['proxy_eligible']+=row['cutoff_status']=='PROXY_CUTOFF_ELIGIBLE_NOT_CLINICAL_APPROVAL'
        else:coverage['historical_rows']+=1
    table(PR/'ORIGINAL50_TEST_RESULTS.csv',labs)
    mainfile=PR/'ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md';text=mainfile.read_text(encoding='utf-8')
    # Replace only this new audit's provisional tables; raw Source and old results stay untouched.
    blocks=re.split(r'(?=^## Patient \d+$)',text,flags=re.M);rebuilt=[blocks[0]]
    for block in blocks[1:]:
        n=int(re.search(r'^## Patient (\d+)',block).group(1));lines=['### 현재 Encounter 검사·응답 Source\n\n',
            '|검사/응답|Code|수치|단위|기록시각|Recorded proxy minute|M5 Test|실제 요청 조건|실제 수신 조건|Source 위치|\n',
            '|---|---|---|---|---|---|---|---|---|---|\n']
        for r in groups[n]:lines.append('|'+ '|'.join(safe(r[k]) for k in ('description','code','value','unit','recorded_at','recorded_proxy_available_minute','M5_Test_mappable','requested_conditions','event_bound_received_conditions'))+'|'+safe(r['source_file'])+':'+r['source_row']+'|\n')
        lines.append('\n수신은 고정 Binding의 정확한 Source Event와 실제 마지막 payload의 원문 Test/Ask 수신을 연결한 값이다. 평가하지 않은40명은 NOT_RUN이며 NOT_RELEASED와 구분한다. Raw CSV 값은 변경하지 않았다. 실제 임상 available_at은 NOT_RECORDED이고 cutoff 적격 Proxy는 임상 공개 승인이 아니다.\n\n')
        block=re.sub(r'### 현재 Encounter 검사·응답 Source.*?(?=### 과거 병력)',lambda _:''.join(lines),block,flags=re.S)
        rebuilt.append(block)
    mainfile.write_text(''.join(rebuilt),encoding='utf-8')
    with (PR/'B2_MEMORY_RETRIEVAL_AUDIT.md').open('w',encoding='utf-8') as f:
        f.write('# Private — B2 Memory 및 정답 손실 사례\n\nSource 일치는 독립 임상 정확성이 아니다. 과거/최신 실행의 Code Pin·출력 계약·메뉴가 달라 Memory의 단일 인과 효과로 해석하지 않는다. Query와 Top5는 이 감사에서 변경하지 않았다.\n')
        for n in sorted(byoriginal):
            a=old[n,'B2'];b=latest[n,'B2'];f.write('\n## Patient %02d (평가 패널 %d)\n\n'%(n,b['panel_ordinal']))
            for label,t in [('과거 B2',a),('최신 B2',b)]:
                f.write('### '+label+'\n\n'+t['status']+'; Source Exact='+str(t['source_exact'])+'; prediction='+json.dumps(t['prediction'],ensure_ascii=False)+'\n\n')
                f.write('제공된 마지막 Memory:\n')
                for r in t['memory']:f.write('- '+safe(r['text'])+'; time='+r['observed_at']+'; score='+str(r.get('retrieval_score'))+'; id='+r['memory_id']+'\n')
                f.write('\n실제 공개 검사/답변 및 마지막 판단:\n```json\n'+json.dumps({'objective_findings':t['observation'].get('objective_findings',[]),'history':t['observation'].get('history',[]),'released_measurements':t['last_payload'].get('released_measurements',[]),'last_decision':t['decisions'][-1] if t['decisions'] else None,'tokens':t['tokens'],'output_schema_fingerprint':t['output_schema_fingerprint']},ensure_ascii=False,indent=2)+'\n```\n')
            f.write('판정: '+('CORRECT_TO_INCORRECT' if a['source_exact'] is True and b['source_exact'] is False else 'OTHER_OR_UNEVALUABLE_TRANSITION')+'. 변화 원인은 CAUSE_UNDETERMINED; 비교 가능한 관측 차이는 위 원문에 기록.\n')
    rows=[{'code':code,'description':description,'unit':unit,'patients':len(s['patients']),'records':s['rows'],'missing_value_records':s['missing'],
        'M5_test_mappable_records':s['m5_rows'],'recorded_proxy_cutoff_eligible_records':s['proxy_eligible'],'denominator_patients':50,
        'clinical_availability_reviewed_records':0} for (code,description,unit),s in sorted(stats.items())]
    table(PUB/'current_test_type_coverage.csv',rows)
    detail=read(PR/'native_completeness_details.json');ref=read(PR/'analysis_full.json')['reference_timing'];timing=Counter(t['status'] for r in ref for t in r['timing'])
    contract={'counts':dict(coverage),'unique_current_code_unit_types':len(rows),'M5_test_mappable_fraction':coverage['M5_test_rows']/coverage['current_rows'],
        'missing_current_value_fraction':coverage['current_missing_values']/coverage['current_rows'],
        'reference_timing_record_counts':dict(timing),'clinical_reference_approved_patients':0,
        'reference_timing_denominator':'matched Source Condition rows, not necessarily one per patient',
        'all_original50_processed':True,'fixed_memory_preserved':72590,'Source_Event_receipt_binding':'EXACT_EVENT + SAVED_TEST_OR_ASK_TEXT',
        'unselected_matching_candidates':1206,'unselected_nonpain_candidates':sum(r['omitted_nonpain_candidates'] for r in detail),
        'unselected_pain_candidates':sum(r['omitted_pain_candidates'] for r in detail),'clinical_relevance':'NOT_REVIEWED',
        'actual_LLM_generation':0,'fitting':0,'retrieval_algorithm_change_in_this_audit':False}
    atomic_json(PUB/'final_coverage.json',contract)
    (PR/'README.md').write_text('# PRIVATE Original50 통합 감사\n\n주 문서는 ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md이다. Patient01~50의 Source Reference, 현재 검사 Code/Value/Unit, Source 위치, 실제 요청/수신, 전체72,590개 과거 기록을 포함한다. 용량 약24MB이므로 큰 Markdown 파일을 지원하는 편집기를 권장한다. Native JSON 세부 비교와 export-time 증상은 NATIVE_SOURCE_COMPLETENESS_AND_ASK_APPENDIX.md에 별도 보존한다.\n\n임상 충분성·relevant/stale/conflict·Source 공개 승인을 생성하지 않았다. 40명은 실제 모델 NOT_RUN이며, 평가10명 중 미실행/실패는 원래 상태 그대로다. 새로운 M2/RAG 알고리즘이나 Synthetic15는 이 감사에 사용하지 않았다.\n',encoding='utf-8')
    docs={
    'README.md':'# Original50 환자별 진단·증상·검사 통합 감사\n\n독립 작업: Original50 전원50명, Memory72,590개를 처리했다. 합성 Routing15명을 혼합하지 않았다. 신규 Generation0회/fitting0회/검색 알고리즘 변경0건.\n\n실제 환자별 진단·수치·Source·과거 기록은 모두 Private의 `outputs/private/research_evaluation/'+NS+'/ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md`에 있다. 약24MB의 단일 문서로 Patient01~50을 포함한다. 공개 문서에는 집계만 남겼다.\n\n공개 기계 집계는 `outputs/research_evaluation/'+NS+'/`이다. `SUMMARY_AND_COVERAGE.md`와 `DATA_QUALITY_FINDINGS.md`를 함께 확인한다.\n',
    'SUMMARY_AND_COVERAGE.md':'# 전체 집계와 분모\n\n- 환자50/50; 전문과별10명; Source 진단명15종. 실제 평가 패널10명은 별도이며 나머지40명은 NOT_RUN.\n- 현재 Encounter CSV 관측2,509행, 과거 관측54,591행, 전체 과거 Memory72,590개. 모든 Memory Source/환자/방문/시점 및 54,591개 관측의 Code/Description/Value/Unit을 정확한 CSV로 확인했다.\n- 명시적 초기 chief complaint 및 Encounter에 연결된 초기 symptom narrative는 각각0/50. associated_symptoms50/50은 통증 측정 항목명이며 증상 서술이 아니다. 실제 Ask 답변과 export-time root symptoms는 분리했다.\n\n```json\n'+json.dumps(contract,ensure_ascii=False,indent=2)+'\n```\n\n검사별 환자 수·행 수·결측·M5 연결·Proxy cutoff는 current_test_type_coverage.csv에 있다. 분모는 환자50 및 해당 유형 관측 행이다. 단순 M5 code/unit 연결 비율은 임상적으로 필요한 검사 Coverage가 아니다.\n',
    'DATA_QUALITY_FINDINGS.md':'# 확인된 사실과 미확정 원인\n\n1. Source/입력에 명시적 현재 증상 서술이 부족하다. 일반 질병 증상을 환자에게 있다고 채우지 않았다. 통증 척도 이름과 실제 통증 답변, 과거 통증을 구분했다.\n2. 최신 B2 마지막45개 기록 중43개가 통증이다. 같은 공개 측정 code/unit을 가진 미선택 과거 후보1,206개(통증123/그 외1,083)가9명에게 존재한다. 이는 기계적 일치 후보이며 임상 relevance 정답이 아니다.\n3. 과거 B2의2/10 정답은 최신0/9와 직접 인과 비교할 수 없다. Code Pin/출력 계약/메뉴가 다르며, 옛 정답2건의 손실은 실제 Trace로 보존했지만 원인은 CAUSE_UNDETERMINED다.\n4. Native JSON2,559개 leaf와 CSV2,509행의 literal nonmatch1,911개를 검사했다. 1,137개는 coded value의 display가 정확히 일치,686개는 소수1자리 export와 호환,38개는 표현/값 차이 미확정이다. 50개 JSON-only code/unit은 주소 항목으로 추가 진단 검사로 사용할 자료가 아니다. Native 값을 CSV 대신 공개하거나 수정하지 않았다.\n5. 현재 Source 관측이 있어도 요청/수신·cutoff·공개 승인은 별개다. 수신은 고정 Source Binding의 event_id와 실제 저장된 Test/Ask 텍스트로 연결했다. 실제 임상 available_at과 Source 진단의 intraday 유효성은 검토되지 않았다.\n6. 정보 충분성과 모델 자체의 추론 오류를 독립 임상 검토 없이 확정하지 않았다. 현재 공개값과 Reference의 의학적 식별성은 NOT_REVIEWED다.\n\n검증 코드: scripts/audit_original50_patient_diagnostics.py → supplement_original50_audit.py → classify_original50_export_differences.py → finalize_original50_audit.py. 최초 분석은 Source와 실행물을 읽기 전용으로 검사하고, 마지막 단계는 이 신규 분석의 event-bound 수신 및 집계 표현만 완성한다. 기존 Source/실험/Reference는 수정하지 않는다.\n'}
    for name,text in docs.items():(DOC/name).write_text(text,encoding='utf-8')
    print(json.dumps(contract))
if __name__=='__main__':main()
