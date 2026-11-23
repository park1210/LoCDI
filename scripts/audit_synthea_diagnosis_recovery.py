"""FINAL_SYNTHEA_DIAGNOSIS_RECOVERY Phase A + Phase B/C/D/E plans only.

Reads the existing Synthea20 and40 saved Episodes. No transport, generation,
model fitting, raw rewriting, clinical judgment, Runtime or prompt patch.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import csv
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,rows,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.routing_collection_preparation import CATALOG,TAXONOMY
from app.locdi.routing_data_reuse import SYNTHEA,EXPERIMENTS
from app.locdi.dataset_inventory import checked_csv_links
from app.core.longitudinal_memory import timestamp
from app.locdi.synthea_diagnosis_audit import (matches,normalize,diagnosis_branch,schema_selectable,
    proxy_minute,reference_time,source_observation_matches,typed_objective_candidate,disclosure_status)
from scripts.finalize_c4_39_results import no_gold

NS='FINAL_SYNTHEA_DIAGNOSIS_RECOVERY'
PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
DOC=ROOT/'docs/연구 평가'/NS
PREVIOUS=ROOT/'outputs/private/research_evaluation/final_exp_recovery_v1'
DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'


def md(name,text):
    for root in (PUB,DOC):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def main():
    if PUB.exists() or PR.exists() or DOC.exists():raise ValueError('FRESH_AUDIT_NAMESPACE_REQUIRED')
    preserved=[]
    for base in ['outputs/research_evaluation/final_exp_v1','outputs/private/research_evaluation/final_exp_v1',
        'outputs/research_evaluation/final_exp_recovery_v1','outputs/private/research_evaluation/final_exp_recovery_v1',
        'outputs/research_evaluation/c4_35_v1','outputs/research_evaluation/c4_39_v1',
        'docs/연구 평가/FINAL_THESIS','docs/연구 평가/FINAL_EXP','docs/연구 평가/FINAL_EXP_RECOVERY','app/core','prompts','configs']:
        preserved.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    preserved += [DATA,ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json',
        Path.home()/'Downloads/심사용+논문.pdf']
    preserved += list((ROOT/'docs/연구 평가').glob('FINAL*REPORT.md'))
    protected={str(p):sha(p) for p in preserved}
    write(PR/'preservation_before.json',protected)
    cases=load_cases(DATA);panel=read(PREVIOUS/'panel.json');index=read(PREVIOUS/'case_index.json')
    saved=rows(PREVIOUS/'memory_recovery_001/episodes.jsonl')
    previous_metrics=read(PREVIOUS/'memory_recovery_001/aggregate.json')['episodes']
    if len(saved)!=40 or len(panel)!=20 or list(index.values())!=list(range(11,31)):raise ValueError('FROZEN_PANEL_CHANGED')
    catalog=read(CATALOG)
    taxonomy=read(TAXONOMY) if TAXONOMY.exists() else {}
    selected=[cases[index[p['case_ref']]-1] for p in panel]
    links,missing_files=checked_csv_links(selected)
    write(PR/'raw_csv_link_results.json',{'links':[{'source_file':k[0],'source_row':k[1],'matches':v} for k,v in links.items()], 'missing_files':missing_files})
    source_pins={str(DATA):sha(DATA),str(CATALOG):sha(CATALOG)}
    if TAXONOMY.exists():source_pins[str(TAXONOMY)]=sha(TAXONOMY)
    public=[];private=[];snapshots=[];totals=Counter();departments=Counter();target_types=Counter()
    for row in saved:
        original_ordinal=index[row['case_ref']];case=cases[original_ordinal-1]
        folder=PREVIOUS/'memory_recovery_001/requests'/('Episode%04d'%row['episode_ordinal'])
        before_files=sorted(folder.glob('Request*_before.json'))
        steps=[]
        for before_path in before_files:
            b=read(before_path);a=read(before_path.with_name(before_path.name.replace('_before','_after')))
            request=b['request'];u=json.loads(request['messages'][1]['content']);r=json.loads(a['response_envelope']['message']['content'])
            steps.append({'before_path':str(before_path),'after_path':str(before_path.with_name(before_path.name.replace('_before','_after'))),
                'request':request,'payload':u,'response':r,'native_status':a['status'],'HTTP_status':a.get('HTTP_status'),
                'before_SHA256':sha(before_path),'after_SHA256':sha(before_path.with_name(before_path.name.replace('_before','_after')))})
        raw=read(folder/'raw_runner_result.json');visit=raw['execution']['visits'][0]
        decisions=read(folder/'validated_ER_decisions.json');terminal=next(d for d in reversed(decisions) if d['action']=='diagnose')
        final=steps[-1];first=steps[0];pred=row['result']['prediction'];ref=case['evaluation']['diagnosis_labels']
        mm=matches(pred['diagnosis'],ref);gold_branch=diagnosis_branch(final['request']['format'])
        selectable=[schema_selectable(final['request']['format'],final['response'],g) for g in ref]
        obs=first['payload']['observation'];last=final['payload']['observation'];end_minute=last['elapsed_minutes']
        ident=visit['identity'];patient_ok=ident['patient_id']==case['patient_id'];visit_ok=ident['visit_id']==case['agent_visible']['index_encounter_id']
        scoring_ok=(final['response']['target']==terminal['target']==pred['diagnosis']
                    and terminal['department']==pred['department']
                    and (pred['diagnosis'] in ref)==mm['source_exact']==previous_metrics[row['episode_ordinal']-1]['source_diagnosis_exact'])
        idx_ms=int(timestamp(case['agent_visible']['index_encounter_start']).timestamp()*1000)
        projected=[e for e in case['environment']['current_encounter_events'] if e['action']=='Test']
        full=[e for e in case['environment']['source_event_store'] if e['event_type']=='observations']
        projection_records=[];units_pass=True;source_values_pass=True;projected_keys=set()
        for event in projected:
            evidence=json.loads(event['evidence'])
            for rec in evidence.get('records',[]):
                found=[s for s in full if source_observation_matches(rec,s)]
                source_values_pass &= bool(found)
                units_pass &= bool(found) and all(rec['recorded_evidence'].get('units')==s['evidence'].get('units') for s in found)
                projected_keys.update(s['event_id'] for s in found)
                projection_records.append({'projected_event':event,'record':rec,'matched_source_ids':[s['event_id'] for s in found]})
        receipt=[e for e in visit['final_state']['event_log'] if e['event_type']=='result_received']
        delivered={e['payload']['test_result']['test_name'] for e in receipt}
        delivery_ok=all(e['target'] in delivered for e in projected if e['available_minute']<=end_minute)
        finding_checks=[]
        for step in steps:
            observation=step['payload']['observation']
            for finding in observation.get('objective_findings',[]):
                received=[e for e in receipt if finding in (e['payload']['test_result'].get('summary'),e['payload']['test_result'].get('impression'))]
                exact=[]
                for e in received:
                    name=e['payload']['test_result']['test_name']
                    for event in projected:
                        if finding==name+': '+event['evidence']:
                            exact.append(event['available_minute']<=e['minute']<=observation['elapsed_minutes'])
                finding_checks.append(bool(exact) and any(exact))
        source_candidates=[]
        for event in full:
            state=disclosure_status(event,end_minute,idx_ms,event['event_id'] in projected_keys)
            source_candidates.append({'source_event':event,'proxy_available_minute':proxy_minute(event,idx_ms),
                'disclosure_status':state,'typed_numeric_objective':typed_objective_candidate(event)})
        candidate_counts=Counter(s['disclosure_status'] for s in source_candidates)
        typed_not_projected=sum(s['typed_numeric_objective'] and s['disclosure_status']=='PROXY_TIME_ELIGIBLE_RELEASE_NOT_REVIEWED' for s in source_candidates)
        initial_symptom=case['provenance']['source_case_audit']['symptom_sources']
        values=[s.get('visible_value',{}) for s in initial_symptom if s.get('kind')=='symptom']
        symptom_label_only=bool(values) and all(v.get('symptom_name') in obs.get('current_symptoms',[]) for v in values)
        ask_fact_values=[]
        for h in last.get('history',[]):
            try:payload=json.loads(h.get('answer',''));ask_fact_values+=payload.get('facts',[])
            except (ValueError,TypeError):pass
        symptom_values_in_ask=all(any(f.get('value')==v.get('value') and (f.get('description')==v.get('symptom_name') or f.get('symptom_name')==v.get('symptom_name')) for f in ask_fact_values) for v in values)
        refs=[]
        for s in case['provenance']['source_case_audit']['diagnosis_sources']:
            r=s['source_record'];refs.append({'source':s,'same_patient':r['PATIENT']==case['patient_id'],
                'same_encounter':r['ENCOUNTER']==case['agent_visible']['index_encounter_id'],
                'timing':reference_time(r['START'],case['agent_visible']['index_encounter_start'])})
        labels=[e['target'].get('description') for e in full]
        measurement_literal=any(isinstance(label,str) and (pred['diagnosis']==label or pred['diagnosis'].startswith(label+':')) for label in labels)
        literal_type='SOURCE_MEASUREMENT_LABEL_LITERAL' if measurement_literal else 'OTHER_FREE_TEXT_NOT_CLINICALLY_CLASSIFIED'
        target_types[literal_type]+=1
        rec={'episode_ordinal':row['episode_ordinal'],'patient_ordinal':row['patient_ordinal'],'condition':row['condition'],
            'status':row['status'],**mm,'source_department_match':pred['department'] in case['evaluation']['accepted_departments_source_policy'],
            'catalog_present_in_actual_payload':'diagnosis_catalog' in final['payload'],
            'runtime_catalog_size':len(raw['execution']['diagnosis_catalog'] or []),
            'actual_catalog_state':'FREE_TEXT_NO_CATALOG' if raw['execution']['diagnosis_catalog'] is None else 'EXPLICIT_CATALOG',
            'gold_schema_selectable':all(selectable),'preexisting_catalog_gold_coverage':all(g in catalog for g in ref),
            'initial_complaint_present':bool(obs.get('chief_complaint')),'initial_symptom_label_only':symptom_label_only,
            'symptom_score_available_in_source':any(v.get('value') is not None for v in values),
            'symptom_score_in_observed_ask':symptom_values_in_ask,
            'projected_tests':len(projected),'projected_tests_received':len(delivered),
            'source_observations':len(full),'proxy_time_eligible_unreviewed_not_projected':candidate_counts['PROXY_TIME_ELIGIBLE_RELEASE_NOT_REVIEWED'],
            'numeric_objective_unreviewed_not_projected':typed_not_projected,
            'source_values_match':source_values_pass,'source_units_match':units_pass,'projected_delivery_match':delivery_ok,
            'finding_receipt_time_matches':all(finding_checks),'finding_occurrences_checked':len(finding_checks),
            'patient_link_matches':patient_ok,'visit_link_matches':visit_ok,'model_validated_scoring_matches':scoring_ok,
            'panel_reference_matches_dataset':panel[row['patient_ordinal']-1]['reference']=={'accepted_diagnoses':ref,
                'accepted_departments':case['evaluation']['accepted_departments_source_policy']},
            'source_label_description_matches':all(g in [r['source']['source_record']['DESCRIPTION'] for r in refs] for g in ref),
            'Source_department_taxonomy_matches':all(taxonomy.get(g) in case['evaluation']['accepted_departments_source_policy'] for g in ref) if taxonomy else None,
            'reference_after_index_day':any(r['timing']['status']=='SOURCE_CONDITION_AFTER_INDEX_DAY' for r in refs),
            'prediction_type_check':literal_type,'terminal_decision_minute':end_minute,
            'source_objective_final_count':len(last.get('objective_findings',[])),
            'memory_delivery_occurrences':sum(len(s['payload']['observation'].get('longitudinal_memory',{}).get('records',[])) for s in steps),
            'Core_stop_reason':read(folder/'episode_and_raw.json')['episode']['result']['stop_reason'],
            'Core_elapsed_simulation_minutes':read(folder/'episode_and_raw.json')['episode']['result']['elapsed_simulation_minutes']}
        public.append(rec)
        private_row={'audit':rec,'original_ordinal':original_ordinal,'private_identity':case['agent_visible']['longitudinal'],
            'Source_reference_diagnoses':ref,'Source_reference_departments':case['evaluation']['accepted_departments_source_policy'],
            'predicted_diagnosis':pred['diagnosis'],'predicted_department':pred['department'],'actual_catalog':raw['execution']['diagnosis_catalog'],
            'diagnose_schema_branches':gold_branch,'preexisting_catalog':catalog,'all_recorded_requests_and_responses':steps,
            'initial_observation':obs,'terminal_observation':last,'reference_sources_and_time':refs,
            'original_current_initial_state':case['agent_visible']['current_initial_state'],
            'original_past_visits':case['agent_visible']['past_visits'],
            'original_source_event_store':case['environment']['source_event_store'],
            'original_replay_menu':case['environment']['current_encounter_events'],
            'source_symptom_values':values,'source_objective_candidates':source_candidates,'projected_source_links':projection_records,
            'actually_cited_evidence':[final['payload']['evidence_records'][i] for i in final['response'].get('evidence_indices',[])],
            'terminal_rationale':final['response'].get('rationale'),'terminal_assessment':final['response'].get('assessment'),
            'runtime_event_log':visit['final_state']['event_log'],'validated_decisions':decisions,
            'Core_export':read(folder/'episode_and_raw.json')['episode']}
        write(PR/('Episode%03d.json'%row['episode_ordinal']),private_row)
        private.append({'episode_ordinal':row['episode_ordinal'],'patient_ordinal':row['patient_ordinal'],'condition':row['condition'],
            'gold_diagnoses':json.dumps(ref,ensure_ascii=False),'prediction':pred['diagnosis'],
            'gold_departments':json.dumps(case['evaluation']['accepted_departments_source_policy']),'predicted_department':pred['department'],**mm})
        if row['condition']=='B0':
            snapshots.append({'patient_ordinal':row['patient_ordinal'],'original_ordinal':original_ordinal,
                'D0_recorded_payload':first['payload'],'D1_recorded_payload':final['payload'],
                'D2_source_candidates_NOT_AUTHORIZED':source_candidates,'decision_cutoff_minute':end_minute,
                'source_snapshot_sha256':digest(case),'execution':'NOT_RUN','review_status':'NOT_REVIEWED'})
    csv_file(PUB/'SYNTHEA_DIAGNOSIS_CASE_ANALYSIS.csv',public,list(public[0]))
    csv_file(PR/'SYNTHEA_DIAGNOSIS_CASE_ANALYSIS_PRIVATE.csv',private,list(private[0]))
    write(PR/'diagnostic_snapshot_plan_sources.json',snapshots)
    write(PR/'source_input_hashes.json',source_pins)
    stats={k:sum(r[k] is True for r in public) for k in ['source_exact','normalized_match','catalog_present_in_actual_payload',
        'gold_schema_selectable','preexisting_catalog_gold_coverage','initial_complaint_present','initial_symptom_label_only',
        'symptom_score_in_observed_ask','source_values_match','source_units_match','projected_delivery_match',
        'finding_receipt_time_matches','patient_link_matches','visit_link_matches','model_validated_scoring_matches','reference_after_index_day',
        'source_label_description_matches','Source_department_taxonomy_matches','panel_reference_matches_dataset']}
    stats.update(episodes=40,patients=20,source_label_types_original50=len({d for c in cases for d in c['evaluation']['diagnosis_labels']}),
        source_label_types_fixed20=len({d for c in selected for d in c['evaluation']['diagnosis_labels']}),
        preexisting_catalog_size=len(catalog),prediction_type_counts=dict(target_types),
        projected_test_occurrences=sum(r['projected_tests'] for r in public),
        source_observation_occurrences=sum(r['source_observations'] for r in public),
        additional_numeric_objective_proxy_time_eligible_UNREVIEWED=sum(r['numeric_objective_unreviewed_not_projected'] for r in public),
        source_csv_references_checked=len(links),source_csv_reference_matches=sum(links.values()),missing_source_csv_file_count=len(missing_files),
        new_LLM_generations=0,new_fitting=0,Source_release_or_clinical_review_created=0)
    write(PUB/'audit_summary.json',stats)
    write(PUB/'synthea_scope_inventory.json',{'used_only_Synthea':True,'raw_wrapper_cases':50,'raw_wrapper_patients':50,
        'source_diagnosis_types':stats['source_label_types_original50'],'Source_departments':dict(Counter(d for c in cases for d in c['evaluation']['accepted_departments_source_policy'])),
        'fixed_panel_patients':20,'fixed_panel_labels':stats['source_label_types_fixed20'],
        'ten_meanings_verified':{'Original50_patients_per_source_department':10,'C4_39_previous_paired_patients':10,'original50_total_patients':50},
        'ten_independent_datasets_or_total_patients':'NOT_ESTABLISHED_FROM_CURRENT_EVALUATION_ASSETS',
        'external_dataset_execution':0,'new_patient_generation':0})
    publish(stats,public)
    after=check(protected)
    if after['status']!='PASS':raise ValueError('PROTECTED_SOURCE_MODIFIED')
    write(PUB/'preservation_verification.json',after)
    write(PUB/'completion_manifest.json',{'status':'PHASE_A_AUDITED_PHASE_B_PLAN_ONLY','new_generations':0,
        'runtime_or_prompt_patches':0,'preservation':after,'code':{f:sha(ROOT/f) for f in ['scripts/audit_synthea_diagnosis_recovery.py','app/locdi/synthea_diagnosis_audit.py']},
        'files':{str(f.relative_to(ROOT)):sha(f) for f in PUB.iterdir() if f.is_file()}})
    print(json.dumps(stats))


def publish(s,public):
    common=('동일 Synthea20명/40 Episode의 저장 기록만 조사했다. 신규 Generation·fitting·환자 생성0회다. '
        '원본 Source labels와 Exact 결과는 보존하고, 임상 Gold·동의어·공개시각 승인을 생성하지 않았다. '
        '이 보고서에서 Gold는 기존 평가용 Source Reference의 통칭이며 독립 임상 정답 승인이 아니다.\n\n')
    locations=('재현: `python -B scripts/audit_synthea_diagnosis_recovery.py`는 fresh Namespace 전용이며 기존 산출물 덮어쓰기를 거부한다. '
        '핵심 경로: `current_ablation_llm_backend.make_er_backend` → `SimulationOptions(diagnosis_catalog=None)` → '
        '`run_simulation._configuration/run_cases` → `longitudinal_runner.run_with_recovery` → '
        '`HospitalLLMPolicy._decision` → `BudgetedNativeERClient.chat_json`.\n\n')
    md('SYNTHEA_DIAGNOSIS_AUDIT_REPORT.md','# Synthea 진단 전수 감사\n\n'+common+locations+
        '## 실제 규모와 10의 의미\n\nOriginal50는50환자/50현재방문,15개 Source 진단명,5개 Source 진료과별10명이다. '
        '실제 고정 패널은20명/40 Episode, Source 진단명7종, 내분비10/신경10명이다. 이전 C4-39 paired 패널은10명이다. '
        '따라서 현재 Synthea 데이터 전체가10개라고 확정할 수 없으며, 본 감사는 지시된 동일20명으로 제한한다. 외부 데이터는 실행하지 않았다.\n\n'
        '## 확정 사실과 원인 판단\n\n'+json.dumps(s,ensure_ascii=False,indent=2)+'\n\n'
        '1. Source/Normalized 일치가 모두0이다. 출력 파싱·환자/방문 연결·scoring target 전달 오류는 저장 Trace에서 발견되지 않았다.\n'
        '2. 실제 Native Catalog는 None이며40/40 payload에 diagnosis_catalog가 없다. 기존15종 Catalog는 고정20명의 Source labels를 모두 포함하지만 호출 경로에 바인딩되지 않았다. '
        '기존 FINAL-EXP 계약이 explicit free-text였으므로 이는 확인된 설정/연구 비교 차이이며 무조건적인 enum 누락 버그라고 바꾸어 부르지 않는다.\n'
        '3. 초기 chief complaint는 없다. 통증 점수의 이름만 초기 증상으로 노출되며 수치는 Ask로 분리한 upstream 설계다. 실제 Ask 후 복원 여부는 CSV 지표에 기록했다. '
        '이름과 실제 증상 호소의 구분이 필요한 표현 위험이 있으나 없는 complaint를 Gold에서 생성할 수 없다.\n'
        '4. 실행 메뉴4종은 혈압2종·맥박·호흡수다. 원천의 다른 관측이 실행 메뉴에 없으나 이는 upstream의 모든 진단×행동 확률 지원을 요구한 공통 메뉴 설계다. '
        '현재 ER-only/VoI OFF에도 그 제한된 snapshot을 재사용했다. 공개 승인되지 않은 Store 후보를 즉시 공개하지 않는다.\n'
        '5. 검사값/단위와 실제 수신 후 전달 경로는 보존됐다. 기록 시각 proxy의 유효성은 임상 검토되지 않았으며 raw Store의 시간 적격성만으로 공개 가능하다고 표시하지 않는다.\n'
        '6.20명 중2명의 Source Condition 시작일이 index 방문일보다 늦다(조건별 중복합4 Episode). 환자·Encounter 연결은 맞지만 현재 진단 Reference 시점의 적합성은 미검토다. '
        '나머지18명도 같은 날짜라는 사실만으로 intra-day 공개시각을 승인하지 않는다.\n'
        '7. Source 측정항목명과 문자열이 같은 진단 target이 관측됐다. free-text 계약과 입력 정보 범위에서 모델이 부적절한 종류의 target을 생성했을 가능성을 확인한다. '
        '임상적 정오 및 단일 인과 원인은 CAUSE_UNDETERMINED다. 정답을 보강하거나 모델 추론 오류만으로 단정하지 않는다.\n\n'
        '## 최소 수정과 다음 판단\n\n현재 단계에서는 Runtime·Prompt·scorer를 패치하지 않는다. 확인된 표현 손실/메뉴 제한/바인딩 차이 각각을 구분해 Phase B 계획을 작성했다. '
        '기존 수식을 바꾸지 않고 입력 view와 명시적 설정 연결을 검토할 수 있으나, Source 공개 승인·고정 공통 Catalog 사용 범위와 Reference 시점 문제를 먼저 결정해야 한다. '
        '다음 최소 실험은 동일20명 D0/D1/D2 각1응답=최대60 Generation 계획이며 이번에는 실행하지 않았다. 최종 B0/B2는 별도 승인 최대480회 계획이다.\n')
    md('SYNTHEA_CATALOG_COVERAGE_REPORT.md','# 실제 Catalog 경로와 Coverage\n\n'+common+locations+
        f'기존 `routing_collection_preparation.CATALOG`는 인접 Synthea experiment30의 `diagnosis_choices.json`을 가리키며15개다. '
        f'고정20명 Source labels7종은 모두 포함된다. Episode 기준40/40 coverage이며 Native payload 실제 enum은0/40이다. '
        f'저장된 diagnose schema에 Source label을 로컬 메모리에서만 대입한 검사에서는{s["gold_schema_selectable"]}/40이 구조적으로 허용됐다. '
        '이 probe는 LLM 요청·응답을 수정하거나 생성한 실험이 아니다.\n\n'
        '`HospitalLLMPolicy._decision`은 catalog가 truthy일 때만 payload 및 diagnose target enum을 추가한다. '
        '`app/utils/diagnosis_catalog.load_diagnosis_catalog`의 일반 loader가 존재한다는 사실과 이 실행에서 유한 Catalog가 실제 적용됐다는 주장은 다르다. '
        'saved raw.execution.diagnosis_catalog=None, 실제 payload, JSON schema 세 가지로 확인했다.\n\n'
        '판정: coverage 부족으로 Gold가 선택 불가능했던 것은 아니다. 자유 텍스트 공간에서 선택은 가능했다. '
        '선행 source study의15종 후보 설정과 최신 Native free-text 설정이 달라 연구 대비에 표시해야 한다. '
        '기존 공통15종을 사용할 경우 환자별 Gold로 축소·정렬하지 않고 catalog hash·출처·개발 노출을 고정해야 한다. '
        '이는 새로운 출력 계약으로 승인받을 사항이며 현재 평가를 소급 변경하지 않는다.\n')
    md('SYNTHEA_EVIDENCE_DELIVERY_REPORT.md','# Synthea Source와 전달 Evidence\n\n'+common+
        f'실행용 Test160회(4×40)의 수신 및 원문 rendering을 조사했다. 검사값·단위 보존 {s["source_values_match"]}/40, '
        f'단위 대조 {s["source_units_match"]}/40, 메뉴 전달 {s["projected_delivery_match"]}/40, '
        f'수신 후 노출 {s["finding_receipt_time_matches"]}/40이다. Raw CSV 대조는{s["source_csv_reference_matches"]}/{s["source_csv_references_checked"]}개 일치이며 누락 파일은 별도 summary에 기록했다.\n\n'
        f'전체 Store observation occurrence는{s["source_observation_occurrences"]}(조건별 중복 포함)이고, 기존 메뉴 밖의 수치 객관 후보 중 '
        f'종료 decision cutoff 이전 recorded-time proxy 후보는{s["additional_numeric_objective_proxy_time_eligible_UNREVIEWED"]} occurrence다. '
        '이 수를 승인된 누락 검사 수라고 해석하지 않는다. 모든 추가 후보의 reveal/review 상태를 Private에 저장했다.\n\n'
        'upstream `experiments/30_source_enriched_study_50/build_study.py::main`의 내부 `transform`은 모든 진단별 관측 지원을 가진 행동만 실행용 cases에 남기고 추가 검사들은 '
        '`cases_source_enriched_review.json`에 보관한다. `experiments/34_live_action_policy/prepare_package.py::main`은 이 cases를 복사한다. '
        'upstream `experiments/24_action_replay_100/build_action_replay.py`의 초기 증상명/Ask 점수 분리도 추적했다. '
        'LoCDI `case_adapter.adapt_case`는 현재 wrapper 이벤트를 복사하고 `clinical_observation.prepare_encounter/doctor_observation`이 역할 입력을 구성한다.\n\n'
        '정상 전달된 것은 제한된 실행 메뉴다. 전체 Synthea 임상 정보가 충분하게 전달됐다고 결론 내릴 수 없다. '
        'chief complaint/발병·기간·특이 증상은 없는 값을 유지해야 하며 진단에 충분한지 독립 검토가 필요하다. '
        '수식·확률을 건드리지 않고 ER view를 확장할 수 있는지는 동일 source cutoff의 수치 observation에 대한 별도 공개 계약으로 검증할 계획이다.\n')
    md('SYNTHEA_SCORING_VALIDATION_REPORT.md','# 출력과 Scoring 무결성\n\n'+common+
        f'Raw model target → validated ER target → backend prediction → strict Source membership 비교는{s["model_validated_scoring_matches"]}/40 일치했다. '
        f'환자 연결{s["patient_link_matches"]}/40, 방문 연결{s["visit_link_matches"]}/40이다. '
        f'원문 Exact {s["source_exact"]}/40, 별도 NFKC/case/punctuation/whitespace Normalized {s["normalized_match"]}/40이다. '
        'Verified Concept Match는 NOT_EVALUATED이며 동의어·질병군 일치를 만들지 않았다.\n\n'
        '기존 scorer를 수정하지 않았다. Normalized는 이번 감사의 별도 진단 지표일 뿐 예전 Exact를 교체하지 않는다. '
        '진료과는 raw spelling을 그대로 비교하지 않고 기존 Core의 canonical department 처리 후 저장된 prediction과 동일 Source 정책 목록을 비교했다. '
        '유효한 department와 참조 department가 불일치한 것을 파싱 실패로 바꾸지 않는다.\n\n'
        '측정항목명과 동일한 target의 기계적 분류는 진단 ontology나 임상 오답 판정이 아니다. '
        'Source Reference 자체의2명 날짜 문제는 scoring 산술 오류와 별개다. 기존0/20과 모든 분모는 그대로 보존한다.\n')
    md('SYNTHEA_RECOVERY_IMPLEMENTATION_PLAN.md','# Phase B 및 다음 실행 계획\n\n'+common+
        '## Phase B 최소 변경 계획만 작성\n\n'
        '|대상|확인 사실|최소 변경 후보|필요 검증/판정|\n|---|---|---|---|\n'
        '|Native Catalog 연결|make_er_backend가None default를 사용|승인된 공통 Catalog가 실행계약에 명시된 경우에만 SimulationOptions로 전달|원 free-text 보존, 공통15 hash·순서고정·전체 환자동일·Gold별 축소금지·schema	enum 회귀|\n'
        '|초기 symptom 표현|upstream은측정명만, 수치는Ask로분리|측정명과실제증상호소를분리하고 Source값/단위/시각을구조화하는 view|0/null 구분, 없는 complaint 생성금지, 현재Ask값복원여부회귀; 정책변경여부검토|\n'
        '|ER Source 메뉴|VoI 전진단지원조건으로4Vitals만남음|VoI OFF용 Source view를 확률지원여부와 분리하는 opt-in overlay 검토|원 wrapper불변, 기존Action종류유지, Source환자/방문/hash·시간·공개승인 whitelist·미지원결과생성금지|\n'
        '|출력/scoring|현재파싱·매핑·산술불일치0|Runtime 패치 불필요, Normalized는별도지표|Exact불변, 동의어자동승인금지|\n'
        '|Reference 시간|2명Source조건날짜가index후|시점상태 sidecar만추가|Gold/환자교체·제외금지; 임상검토없이유효·무효판정금지|\n\n'
        '현재 코드 패치는 감사 Script와 TEST_ONLY 검증만이다. 문제 원인이 확정되지 않은 임상 판단·의도된 설계를 임의로 고치지 않는다.\n\n'
        '## Phase C Diagnostic Snapshot 계획 — NOT_RUN\n\n'
        '동일20명 입력 순서 고정, 환자별 D0→D1→D2, 각1회로 최대60 Generation. '
        'D0는 B0 첫 실제 payload의 관찰, D1은 B0 마지막 실제 공개 관찰을 사용한다. '
        'D2 cutoff는 두 조건 마지막 진단 decision minute 중 이른 시점으로 사전 고정하고 동일 Source Store의 승인된 객관 관측만 사용한다. '
        '현재 작성한 D2 후보는 time-eligible인 미검토 기록이며 실행용 payload가 아니다. '
        'Source conditions/Gold/진료과 Reference/원본 provenance/약물 REASON 진단정보는 입력하지 않는다. '
        'Model3B/Digest·Context32768·temperature0·seed42·출력1024·120초·Retry/Repair0, Memory OFF로 진단능력 분리한다. '
        'D0 객관근거가 없으면 supported를 강제하지 않고 unresolved를 허용한다. 정적 provisional hypothesis를 추가한다면 Hospital supported 진단과 별도 출력/지표 계약 승인이 필요하다. '
        '고정 Prompt1종만 사용하고20사례를 보고 반복 튜닝하지 않는다. max60×120초=2시간 timeout합계, 전역wall45분·Python4GiB·출력1GiB 제안이며 '
        'Native 기존 평균 HTTP5.51초 외삽은 약5.5분+더긴D2/ColdLoad/처리 overhead로 보장하지 않는다. '
        '추가 Source 공개 계약·최종 Prompt/output hash·실행 예산에 대한 새 승인 전에는 호출0이다.\n\n'
        '## Phase D Prompt 개선 — 결과 확인 전 적용 금지\n\n'
        '현재 chief complaint·증상·병력·객관값과 단위·감별 후보·지지/반증·미확정 사유를 분리해 읽고 '
        '진단을 요청된 검사명이나 통증 측정명과 혼동하지 않도록 하는 일반 형식 안내만 검토한다. '
        '환자별 진단 힌트나 Gold 선택을 추가하지 않는다. eligible citation과 기존 unresolved 정책을 유지한다. '
        'Phase C 없이 문구 개선 효과를 확정하지 않고, 적용 시 별도 Prompt version/hash를 고정한다. 현재 변경0이다.\n\n'
        '## Phase E paired 재평가 — NOT_RUN\n\n'
        'Phase C/Source 공개 계약과 실제 필요한 수정 검증 후에만 동일20명 환자별B0→B2 총40 Episode를 새 Namespace에서 각1회 계획한다. '
        '기존3B·Context32768·temperature0·seed42·Top5·Retry/Repair0·Evidence Validator 유지, 각12/총480Generation·120초·wall90분·Python4GiB·출력1GiB 상한안이다. '
        '이번 완료Run의 HTTP평균5.51초에서 최대480회 단순외삽 약44.1분+overhead이고 개별timeout합계16시간은 전역90분과다르다. '
        '기술적 실패는격리, Source/Identity/전역상한은전체중단, no-resume. Source view/Catalog/Prompt 변경을 양조건에동일적용하고 이전 결과에병합하지않는다. '
        'Source Exact·별도Normalized·Concept NOT_EVALUATED·Source 진료과·certainty·전체등록/완료/실패·환자전환·토큰/실시간을 B0/B2각각보고한다. '
        '이는 계획이며 Phase C60회와 E480회는 현재 승인되거나 실행된 요청이 아니다.\n')


if __name__=='__main__':main()
