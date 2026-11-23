"""Model-free recovery aggregation and final thesis freeze; no regeneration."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_memory_paired_recovery import PUB,PR,RUN,load_protocol
from scripts.audit_research_gaps import read,rows,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from scripts.finalize_c4_39_results import no_gold
from app.locdi.memory_paired_recovery import aggregate,verify_delivered_memory
from app.locdi.dataset_loader import load_cases


def text(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x',encoding='utf-8') as f:f.write(value.rstrip()+'\n')


def compute():
    folder=PR/RUN; finished=read(folder/'finished.json'); panel=read(PR/'panel.json')
    saved=rows(folder/'episodes.jsonl'); summary=aggregate(panel,saved)
    request_records=[];enriched=[];failures=[];memory_checks=[]
    cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl');case_index=read(PR/'case_index.json')
    for row,outcome in zip(saved,summary['episodes']):
        case_folder=folder/'requests'/('Episode%04d'%row['episode_ordinal'])
        requests=[]
        for after in sorted(case_folder.glob('Request*_after.json')):
            a=read(after); before=after.with_name(after.name.replace('_after','_before'))
            b=read(before) if before.exists() else {};u=json.loads(b['request']['messages'][1]['content']) if b else {}
            envelope=a.get('response_envelope',{});mem=u.get('observation',{}).get('longitudinal_memory',{})
            boundary=verify_delivered_memory(cases[case_index[row['case_ref']]-1],mem)
            if row['condition']=='B0' and mem.get('records'):boundary.update(status='FAIL',failure_codes=['B0_MEMORY_PRESENT'])
            memory_checks.append({'patient_ordinal':row['patient_ordinal'],'condition':row['condition'],**boundary})
            requests.append({'patient_ordinal':row['patient_ordinal'],'condition':row['condition'],
                'generation_attempted':a.get('generation_attempted',True),'HTTP_status':a.get('HTTP_status'),
                'status':a['status'],'error_type':a.get('error_type'),
                'prompt_tokens':envelope.get('prompt_eval_count'),'output_tokens':envelope.get('eval_count'),
                'wall_seconds':a.get('wall_seconds'), 'actual_context':a.get('identity_after',{}).get('context_length'),
                'memory_records':len(mem.get('records',[])),'memory_mode':mem.get('mode'),
                'structured_private_gold_absent':no_gold(u)})
        actions=outcome['action_counts'];partial=case_folder/'partial_episode.json'
        if actions is None and partial.exists():
            counts=read(partial)['result']['action_counts']
            actions={'Ask':counts.get('ask_history',0),'Test':counts.get('order_test',0),
                'Consult':counts.get('request_specialty',0),
                'Diagnose':sum(counts.get(k,0) for k in ('er_finalize','specialty_finalize','specialist_finalize'))}
        item={**outcome,'prompt_tokens':sum(r['prompt_tokens'] or 0 for r in requests),
            'output_tokens':sum(r['output_tokens'] or 0 for r in requests),
            'HTTP_latency_seconds':sum(r['wall_seconds'] or 0 for r in requests),
            'nonempty_memory_deliveries':sum(r['memory_records']>0 for r in requests),
            'memory_record_occurrences':sum(r['memory_records'] for r in requests),
            'actual_action_counts':actions,'counts_known':actions is not None}
        enriched.append(item);request_records.extend(requests)
        if row['status'] not in ('COMPLETED','NOT_ATTEMPTED','INCOMPLETE'):
            failures.append({'patient_ordinal':row['patient_ordinal'],'condition':row['condition'],
                'status':row['status'],'error_type':row['error_type'],'actual_requests':row['request_count'],
                'detail':row['failure_audit'],'not_converted_to_incorrect':True})
    totals={}
    for c in ('B0','B2'):
        es=[e for e in enriched if e['condition']==c];rs=[r for r in request_records if r['condition']==c]
        known=[e for e in es if e['counts_known']]
        totals[c]={'registered':20,'completed':sum(e['status']=='COMPLETED' for e in es),
            'prompt_tokens':sum(e['prompt_tokens'] for e in es),'output_tokens':sum(e['output_tokens'] for e in es),
            'HTTP_latency_seconds':sum(e['HTTP_latency_seconds'] for e in es),'request_count':sum(e['request_count'] for e in es),
            'nonempty_memory_deliveries':sum(e['nonempty_memory_deliveries'] for e in es),
            'memory_record_occurrences':sum(e['memory_record_occurrences'] for e in es),
            'known_action_count_episodes':len(known),
            'action_counts':{a:sum(e['actual_action_counts'][a] for e in known) for a in ['Ask','Test','Consult','Diagnose']},
            'unknown_action_count_episodes':20-len(known)}
    pairs=[]
    for i,pair in enumerate(summary['pairs']):
        pair=dict(pair)
        for cond,e in zip(('B0','B2'),enriched[2*i:2*i+2]):
            for key in ('request_count','prompt_tokens','output_tokens','HTTP_latency_seconds','nonempty_memory_deliveries','memory_record_occurrences'):
                pair[cond+'_'+key]=e[key]
            pair[cond+'_certainty']=e['certainty']
            for action in ('Ask','Test','Consult','Diagnose'):
                pair[cond+'_'+action]=e['actual_action_counts'][action] if e['counts_known'] else None
        pairs.append(pair)
    accounting=sum(r['generation_attempted'] for r in request_records)
    protected=check(read(PR/'protected_input_hashes.json'))
    protocol=load_protocol();pins=check(protocol['input_hashes'])
    return {'summary':summary,'pairs':pairs,'episodes':enriched,'requests':request_records,'totals':totals,
        'failure_audit':failures,'generation_count':accounting,'reported_generation_count':finished['generation_count'],
        'wall_seconds':finished['wall_seconds'],'global_stop_reason':finished['global_stop_reason'],
        'protected':protected,'input_pins':pins,'Gold_boundary_scan':all(r['structured_private_gold_absent'] for r in request_records),
        'memory_boundary_checks':memory_checks,'Memory_boundary_status':'PASS' if all(m['status']=='PASS' for m in memory_checks) else 'FAIL'}


def finalize():
    result=compute();s=result['summary'];pairs=result['pairs']
    if s!=read(PR/RUN/'aggregate.json'):raise ValueError('INDEPENDENT_REAGGREGATION_MISMATCH')
    if result['generation_count']!=result['reported_generation_count']:raise ValueError('REQUEST_ACCOUNTING_MISMATCH')
    if result['protected']['status']!='PASS' or result['input_pins']['status']!='PASS' or not result['Gold_boundary_scan'] or result['Memory_boundary_status']!='PASS':
        raise ValueError('SOURCE_PRESERVATION_OR_BOUNDARY_FAILURE')
    paired_dept=[p for p in pairs if p['complete_pair'] and all(type(p[k]) is bool for k in ('B0_department_match','B2_department_match'))]
    dept_difference=sum(int(p['B2_department_match'])-int(p['B0_department_match']) for p in paired_dept)/len(paired_dept) if paired_dept else None
    status='RECOVERY_PAIRED_EVALUATED_WITH_LIMITATIONS' if s['complete_pairs'] else 'RECOVERY_EXECUTED_NO_COMPLETE_PAIRS'
    test_path=PUB/'test_results_final.json'
    tests=read(test_path if test_path.exists() else PUB/'test_results.json')
    execution={'status':status,'registered_patients':20,'registered_episodes':40,
        'conditions':s['conditions'],'complete_pairs':s['complete_pairs'],'source_scorable_pairs':s['source_scorable_pairs'],
        'transitions':s['transitions'],'paired_Source_diagnosis_B2_minus_B0':s['paired_source_exact_B2_minus_B0'],
        'paired_diagnosis_denominator':s['paired_denominator'],'paired_Source_department_B2_minus_B0':dept_difference,
        'paired_department_denominator':len(paired_dept),'generation_requests':result['generation_count'],
        'wall_seconds':result['wall_seconds'],'totals':result['totals'],'global_stop_reason':result['global_stop_reason'],
        'verified_concept_agreement':'NOT_EVALUATED','clinical_accuracy':'NOT_EVALUATED',
        'Core_and_Evidence_Validator_changes':0,'Prompt_changes':0,'Routing_generation':0,'utility_fitting':0,'VoI_evaluation':0,
        'complete_pair_selection_warning':s['failure_selection_bias_warning']}
    write(PUB/'final_memory_execution_status.json',execution)
    write(PUB/'final_memory_failure_audit.json',{'failed_episodes':result['failure_audit'],'failed_count':len(result['failure_audit']),
        'incomplete_episodes':[{'patient_ordinal':e['patient_ordinal'],'condition':e['condition'],'status':e['status']} for e in result['episodes'] if e['status']=='INCOMPLETE'],
        'source_failures_not_imputed_as_incorrect':True,'all_registered_patients_preserved':True,
        'previous_failure_analysis':read(PUB/'prior_failure_evidence_audit.json')})
    csv_file(PUB/'final_memory_paired_results.csv',pairs,list(pairs[0]))
    csv_file(PUB/'final_memory_episode_metrics.csv',result['episodes'],list(result['episodes'][0]))
    write(PUB/'request_accounting.json',{'actual_requests':result['generation_count'],'requests':result['requests'],
        'schema_valid':sum(r['status']=='VALID' for r in result['requests']),
        'HTTP_200':sum(r['HTTP_status']==200 for r in result['requests']),'raw_wire':'NOT_CAPTURED'})
    write(PUB/'memory_cutoff_verification.json',{'status':result['Memory_boundary_status'],'checks':result['memory_boundary_checks']})
    write(PR/'independently_reaggregated_metrics.json',s)
    # Same underlying Source, two fixed conditions; no pooling of earlier9 B0.
    docs=ROOT/'docs/연구 평가/FINAL_EXP_RECOVERY'
    header='|조건|등록|상태|supported/unresolved|Source 진단 일치|Source 진료과 일치|Generation|Prompt tokens|\n|---|---|---|---|---|---|---|---|\n'
    lines=[]
    for c in ('B0','B2'):
        m=s['conditions'][c];t=result['totals'][c];a=m['source_exact_supported'];d=m['source_department_agreement']
        lines.append('|%s|20|%s|%s/%s|%s/%s|%s/%s|%s|%s|'%(c,json.dumps(m['states'],ensure_ascii=False),m['supported'],m['unresolved'],a['numerator'],a['denominator'],d['numerator'],d['denominator'],t['request_count'],t['prompt_tokens']))
    tab=header+'\n'.join(lines)
    scope=('Source 문자열/기존 Source 진료과 정책 일치이며 독립 임상 정확도가 아니다. verified concept agreement는 독립적인 사전 근거가 없어 NOT_EVALUATED다. '
        '완료·실패·미해결·미시도를 구분하며 실패를 Source 오답으로 대입하지 않았다. '
        '기술적으로 완료한 쌍과 두 조건 모두 supported인 Source 채점 쌍의 분모는 다르다. '
        '조건부 완료 쌍의 차이는 실패 선택 편향 가능성이 있어 전체 등록20명 결과와 함께 보고한다.')
    method=('기존 Memory token-intersection Top5/k=5·환자/방문 cutoff·superseded/tie 규칙, Routing Jaccard/평활화/min_samples/abstention, '
        'Bayesian posterior·Bellman·VoI/STOP·원래 H2는 변경하지 않았다. '
        '새 코드는 실제 기존 Native ER backend와 Core Runner를 호출하는 Episode별 수집 루프다. '
        '변경은 기술적 오류 격리와 환자별 B0→B2 실행 순서이며, 연구 방법론이나 모델/Prompt/Context 설정 변경이 아니다. '
        '과거 Run의 first-error-stop 계약을 소급 변경하거나 기존9개 B0를 새 결과와 결합하지 않았다.')
    report=('# FINAL EXP RECOVERY 결과 보고서\n\n'+status+'\n\n'
        '## 실제 새로운 Memory 결과\n\n'+tab+'\n\n'+scope+'\n\n'
        '등록20환자/40Episode, 완료 쌍 '+str(s['complete_pairs'])+', Source 채점 쌍 '+str(s['source_scorable_pairs'])+'다. '
        'paired Source 진단 차이(B2−B0)='+str(s['paired_source_exact_B2_minus_B0'])+', 분모='+str(s['paired_denominator'])+'. '
        'paired Source 진료과 차이='+str(dept_difference)+', 분모='+str(len(paired_dept))+'.\n\n'
        '환자별 전환: '+json.dumps(s['transitions'],ensure_ascii=False)+'. 전체 등록20명의 상세 상태는 paired CSV에 모두 남겼다.\n\n'
        '## Evidence 오류와 보존한 계약\n\n원 실패는 범위 밖 Index가 아니라 존재하지만 비적격인 문진 항목4를 supported 진단에 인용한 것이다. '
        '객관적 후보0~3은 존재했다. 따라서 객관적 목록이 비어 있는 오류와 구분하지만 실제 진단 근거가 충분했다는 임상 판단은 하지 않는다. '
        '입력에는 zero-based Index와 eligible objective citation 지시가 명시돼 있어 Prompt를 바꾸지 않았다. '
        '모델의 의도가1-based 인용이었다고 추정하거나 Index를 자동으로3으로 고치지 않았다. '
        '모든 실패의 원래 HTTP/응답/Evidence/부분 Core Export는 Private에 보존했다.\n\n'
        '## 실행 예산과 실제 요청\n\n새 Protocol '+load_protocol()['fingerprint']+'에 대한 새 사용자 승인 후 한 번 실행했다. '
        '최대480/각12 Generation, 요청120초·출력1024·Context32768·temperature0·seed42·Retry/Repair0, 전체5400초·Python4GiB·출력1GiB였다. '
        '실제 '+str(result['generation_count'])+'회, wall '+str(result['wall_seconds'])+'초. '
        'Source/Identity/전역 상한은 전체 중단하고 일반 Episode 실패는 보존 후 다음 등록 Episode로 진행했다. '
        '단일 호출/쓰기 중 자원 상한 일시 초과 가능성이 있는 호출 전후 감시다.\n\n'
        '## 연구 방법과 논문 정합성\n\n'+method+'\n\n'
        '기존1.5B의10환자 평가와 신규3B의20환자는 다른 패널이므로 모델의 인과 효과로 비교하지 않는다. '
        'FINAL-EXP의 동일20 Snapshot은 전체 Episode 비교가 아니며 원문 Source exact0을 유지한다. '
        '외부 AgentClinic의 고정20사례 Source exact0/20→2/20은 별도 개발 Source-QA 결과로 보존하고 이번에 추가 호출하지 않았다.\n\n'
        '## Routing 및 VoI 종료\n\nRouting은 실제15응답=Source correct1/incorrect5/unresolved9, scorable Train group0으로 ROUTING_INSUFFICIENT_DATA다. '
        '추가 Routing Generation/fitting0, 공식 B3/B4 BLOCKED다. '
        'C4-35 신규 합성35명/71방문의 H3 AUBC52.0952%, Global EIG45.2381%, 차이6.8571pp/95%CI[1.9524,11.9524]pp를 유지한다. '
        'H2는 원래 방법, H3는 확장이다. Budget5 열위와 Top3/NLL/Brier·Dynamic Horizon 한계를 유지하며 Final M5를 재추정하지 않았다. '
        'C4-39 Runtime5/5 수치 비교도 기술 근거로 그대로 보존했다.\n\n'
        '## 자료 보호와 재현성\n\n독립 저장 Trace 재집계 EXACT_MATCH, 보호 '+str(result['protected']['checked'])+'개 Hash 검사 PASS. '
        '관련 테스트 '+str(tests['tests_run'])+'개 중 '+str(tests['pass'])+' PASS이며 전체 저장소 테스트가 아니다. '
        'Source Gold/미공개 Private 키 구조 검사 PASS이며 임상 공개시각은 개발용 Recorded-Time Proxy 수용 범위다. '
        'LLM 재생성의 bitwise 재현성을 주장하지 않는다. 이전 완료 Namespace·원문 PDF·최종 논문 자료는 변경하지 않았다.\n\n'
        '## 최종 논문 수정\n\n초록/§4.1/4.2에는 위 실제 분모와 실패 포함 paired 표를 넣고, §3.2의 공식은 유지한다. '
        '모델·Source·실행 정책과 임상 검증의 차이를 명시한다. 기존90% 통합 성능을 새 AUBC나 다른 패널 값으로 치환하지 않는다. '
        '제목 및 통합 임상 우월성 주장을 기능별 개발/합성 평가 범위로 축소하는 부분은 교수님 확인 사항이다. '
        '새 연구 단계나 자동 재실행 없이 이 결과와 실패 기록으로 논문을 동결한다.\n')
    text(ROOT/'docs/연구 평가/FINAL_EXP_RECOVERY_REPORT.md',report)
    text(PUB/'FINAL_EXP_RECOVERY_REPORT.md',report)
    text(docs/'FINAL_THESIS_UPDATED_RESULTS.md','# 최종 논문 결과 수정 자료\n\n'+tab+'\n\n'+scope+
        '\n\n관측 paired Source 진단 차이='+str(s['paired_source_exact_B2_minus_B0'])+' (분모'+str(s['paired_denominator'])+'). '
        'Source 진료과 차이='+str(dept_difference)+' (분모'+str(len(paired_dept))+'). '
        '공식 B0~B4 전체 임상 평가나 서로 다른 데이터의 통합 Ablation으로 서술하지 않는다.\n\n'+method)
    text(docs/'FINAL_THESIS_FINAL_CHANGE_MATRIX.md','# 최종 논문 변경 구분\n\n'
        '|대상|구분|수정 방향|\n|---|---|---|\n'
        '|Memory/Routing/Bayesian/Bellman/VoI/STOP/H2|연구 방법론 변경 없음|기존 수식 설명 유지|\n'
        '|Episode 실패 격리/환자별B0→B2|기술적 실행 오류 처리|오류·미시도·종료 조건과 새 Protocol 명시|\n'
        '|3B/Context32768/Top5|기존 평가 환경 고정|원래3B 바인딩·Prompt·Source를 그대로 사용|\n'
        '|초록/§4.1/§4.2/결과표|실제 결과 수정|20등록 분모와 완료/채점 쌍·미완료 포함|\n'
        '|Concept Match|평가 범위 제한|사전 독립 근거 없어 NOT_EVALUATED|\n'
        '|Routing|한계 보존|15응답·적격0·fitting0, B3/B4차단|\n'
        '|VoI|기존 결과 유지|원래H2/확장H3 및 합성 평가 한계 구분|\n'
        '|제목/통합 우월성 주장|교수님 확인 필요|기능별 실제 개발/합성 근거에 맞게 범위 한정|\n\n'
        '원문 논문, 기존9개 FINAL_THESIS 및 FINAL_EXP7개는 수정하지 않았다. 저자·사사 정보를 생성하지 않았다.\n')
    text(docs/'FINAL_PROFESSOR_FINAL_BRIEFING.md','# 교수님 최종 보고용 문안\n\n'
        '기존 수식과 Evidence 기준을 유지하고, Episode 오류가 다른 환자의 수집을 중단하지 않도록 기술적 수집 정책만 분리해 동일20환자의 B0/B2를 새 Run에서 평가했습니다. '
        '완료 쌍 '+str(s['complete_pairs'])+'개, 두 조건 모두 supported인 Source 채점 쌍 '+str(s['source_scorable_pairs'])+'개이며 실제 Generation은 '+str(result['generation_count'])+'회입니다.\n\n'+tab+'\n\n'+scope+
        '\n\nRouting은 적격 Utility가 없고 임상 효과는 미검증입니다. 기존 합성 VoI의 사전 주 결과와 부정적 보조 결과는 모두 유지했습니다. '
        '다양한 Source 활용 의견에는 종단 Synthea와 별도 AgentClinic Source-QA 평가를 분리해 대응했습니다. '
        '통합 임상 우월성 주장 대신 이 기능별 실제 평가 범위로 논문을 확정하는 방향과 제목 변경 필요성을 확인 부탁드립니다. '
        '이 문안은 보고용 초안이며 발송하지 않았습니다.\n')
    protocol=load_protocol()
    write(PUB/'final_memory_reproducibility.json',{'status':'EXACT_MATCH','protocol_fingerprint':protocol['fingerprint'],
        'model_identifier':protocol['model_identifier'],'Prompt_changed':False,'Evidence_Validator_changed':False,
        'Core_formulas_changed':False,'parent_final_exp_not_merged':True,'generation_requests':result['generation_count'],
        'source_trace_SHA256':sha(PR/RUN/'episodes.jsonl'),'protected_files':result['protected'],'input_pins':result['input_pins'],
        'tests':tests,'memory_cutoff_checks':'memory_cutoff_verification.json',
        'Frozen_Final_M5_SHA256':sha(ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'),
        'original_paper_SHA256':sha(Path.home()/'Downloads/심사용+논문.pdf'),
        'raw_wire_observed':False,'clinical_visibility_review':'NOT_REVIEWED','concept_agreement':'NOT_EVALUATED',
        'check_only_command':'python -B scripts/finalize_memory_paired_recovery.py --check-only',
        'model_regeneration_bitwise_reproduction':False})
    print(json.dumps({'status':status,'generation_requests':result['generation_count'],'complete_pairs':s['complete_pairs'],
        'scorable_pairs':s['source_scorable_pairs'],'transitions':s['transitions']}))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:
        result=compute();ok=result['summary']==read(PR/RUN/'aggregate.json')
        print(json.dumps({'status':'EXACT_MATCH' if ok else 'FAIL','generation_calls':0,'registered_patients':20,
            'protected':result['protected'],'input_pins':result['input_pins']}));raise SystemExit(not ok)
    else:finalize()
