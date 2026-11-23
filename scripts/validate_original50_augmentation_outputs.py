# -*- coding: utf-8 -*-
"""Actual 50-case dataset verification and source-preserving final previews."""
from collections import Counter
from pathlib import Path
import ast,csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import original_cases,read,sha_file
from app.locdi.original50_augmentation import validate_augmentation,initial_preview,directly_discloses_label
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.m5_common_source import bind_cases,M5_SHA
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import digest
NS='ORIGINAL50_LITERATURE_AUGMENTATION_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS
DOC=ROOT/'docs/B0_B4_실험/19_Original50_문헌기반_데이터보강'
def main():
    cases=original_cases();catalog=evidence_catalog();rows=[json.loads(s) for s in (PR/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=50 or [r['case_ordinal'] for r in rows]!=list(range(1,51)):raise ValueError('FULL_COHORT_REQUIRED')
    m5=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
    if sha_file(m5)!=M5_SHA:raise ValueError('FROZEN_M5_CHANGED')
    bindings=bind_cases({str(n):c for n,c in enumerate(cases,1)},read(m5));atomic_json(PR/'native_m5_bindings_readonly.json',bindings)
    validated=PR/'validated_previews';validated.mkdir(exist_ok=False);details=[];counts=Counter();rejected_previews=[];nonobs=[]
    for c,r in zip(cases,rows):
        validate_augmentation(c,r,catalog,verify_source_files=True);n=r['case_ordinal']
        preview={'case_ordinal':n,'preview_contract':'SOURCE_PRESERVING_INITIAL_PREVIEW.v1_1','runtime_admitted':False,
            'clinical_reviewed':False,'scope':r['evaluation_scope'],'D0':initial_preview(c,r,catalog,view='D0'),
            'D1_PREVIEW_ONLY':initial_preview(c,r,catalog,view='D1',accept_development_proxy=True),
            'D2_PREVIEW_ONLY':initial_preview(c,r,catalog,view='D2',accept_development_proxy=True)}
        before=c['agent_visible']['current_initial_state'];after=preview['D2_PREVIEW_ONLY']
        if not set(before['patient_visible_info']['associated_symptoms'])<=set(after['patient_visible_info']['associated_symptoms']):raise ValueError('SOURCE_LABEL_DROPPED')
        for k,v in before['doctor_visible_info'].items():
            if k!='chief_complaint' and after['doctor_visible_info'][k]!=v:raise ValueError('SOURCE_INITIAL_DOCTOR_FIELD_CHANGED')
        for k,v in before['patient_visible_info'].items():
            if k!='associated_symptoms' and after['patient_visible_info'][k]!=v:raise ValueError('SOURCE_INITIAL_PATIENT_FIELD_CHANGED')
        if directly_discloses_label(json.dumps(after)):raise ValueError('DIRECT_REFERENCE_NAME_IN_PREVIEW')
        atomic_json(validated/('Patient%02d.json'%n),preview)
        old=read(PR/('Patient%02d_preview.json'%n))
        if old!=preview:
            rejected_previews.append({'case_ordinal':n,'candidate_type':'EARLY_PREVIEW_ONLY','reason':'REPLACED_BY_SOURCE_PRESERVING_PREVIEW_NOT_A_DATASET_ROW_REJECTION',
                'old_file':str(PR/('Patient%02d_preview.json'%n)),'validated_file':str(validated/('Patient%02d.json'%n)),
                'old_file_preserved':True,'LLM_generation':0})
        bound={x['source_event']['event_id'] for x in bindings[str(n)]['rows'] if x['kind']=='Test'}
        for t in r['literature_informed_synthetic']['candidate_tests']:
            match=bool(set(t['M5_compatible_source_event_ids'])&bound)
            details.append({'case_ordinal':n,'test_id':t['test_id'],'candidate_status':t['action_compatibility'],'exact_existing_native_binding':match,
                'clinical_release_approved':False,'new_action_added':False})
            counts['recommendations']+=1;counts['native_bound_recommendations']+=match
            if t['action_compatibility']=='CURRENT_M5_COMPATIBLE' and not match:counts['candidate_mapping_native_mismatch']+=1
        for e in c['environment']['source_event_store']:
            if e['event_type']=='observations':continue
            nonobs.append({'case_ordinal':n,'event_type':e['event_type'],'description':e['target']['description'],'timestamp':e['timestamp'],
                'source':e['source_metadata'],'clinical_available_at':'NOT_RECORDED','automatically_released':False,
                'interpretation':'A procedure/order/medication record is not an interpreted diagnostic test result; diagnosis/treatment chronology requires review.'})
    atomic_json(PR/'exact_native_action_compatibility.json',details)
    with (PR/'REJECTED_EARLY_PREVIEWS.jsonl').open('x',encoding='utf-8') as f:
        for r in rejected_previews:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (PR/'REJECTED_LITERATURE_SOURCES.jsonl').open('x',encoding='utf-8') as f:
        for s in catalog['failed_access_not_used']:f.write(json.dumps({'candidate_type':'EXTERNAL_SOURCE','reason':'ACCESS_FAILED_NOT_USED','source':s})+'\n')
    with (PR/'ORIGINAL_CURRENT_NONOBSERVATION_RECORDS.jsonl').open('x',encoding='utf-8') as f:
        for r in nonobs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (PR/'CURRENT_PROCEDURES_MEDICATIONS_APPENDIX.md').open('x',encoding='utf-8') as f:
        f.write('# PRIVATE — Current procedures, medications and other records\n\nThese are original Source records. Presence does not establish a diagnostic result or initial visibility; all are excluded from new model previews.\n')
        for n in range(1,51):
            f.write('\n## Patient %02d\n\n'%n)
            for r in nonobs:
                if r['case_ordinal']==n:f.write('- '+r['event_type']+' | '+r['description']+' | '+r['timestamp']+' | Source '+r['source']['source_file']+':'+str(r['source']['source_row'])+'\n')
    report={'status':'PASS' if not counts['candidate_mapping_native_mismatch'] else 'ACTION_REVIEW_REQUIRED',
        'actual_cases_structurally_validated':50,'source_file_and_symptom_replay_checked':50,
        'initial_source_fields_preserved':True,'D0_exact_initial_state_match':True,'D2_direct_reference_names_absent':True,
        'counts':dict(counts),'nonobservation_original_records':len(nonobs),'nonobservation_types':dict(Counter(r['event_type'] for r in nonobs)),
        'current_procedure_record_does_not_imply_test_result':True,
        'final_previews_directory':'validated_previews','early_preview_files_preserved_and_superseded':len(rejected_previews),
        'clinical_diagnosability':'UNREVIEWED','clinical_reviewed':False,'runtime_admitted':False,'actual_LLM_generation':0,'utility_fitting':0}
    atomic_json(PUB/'actual_dataset_validation.json',report)
    mainfile=PR/'ORIGINAL50_AUGMENTATION_SUMMARY.md'
    # This new report receives an explicit appendix pointer; no original patient or execution artifact is edited.
    with mainfile.open('a',encoding='utf-8') as f:f.write('\n## 최종 Source 및 Preview 검증\n\n최종 미실행 preview는 validated_previews/Patient01~50.json이다. 초기 *_preview.json은 보존하되 폐기 후보로 분리했다. 원본 associated_symptoms의 통증 척도 항목명도 삭제하지 않고 새 합성 서술을 추가한다. 실제 추가 Source 증상은 임상 척도가 아닌 engine 이벤트이며 별도 공개 승인 없이 Native Runtime에 넣지 않는다. 현재 약물·처치·careplan 원문은 CURRENT_PROCEDURES_MEDICATIONS_APPENDIX.md에서 Patient01~50별로 확인한다. 처치가 있었다는 사실을 검사 판독 결과로 취급하지 않는다.\n')
    (DOC/'ADAPTER_CONTRACT.md').write_text('# Versioned 별도 Adapter\n\napp/locdi/original50_augmentation.py의 initial_preview는 D0/D1/D2의 기존 current_initial_state 형태만 만든다. default D0는 deep-copy로 정확히 동일하다. D1/D2 preview에는 명시적 개발용 opt-in이 필요하며, 실제 LLM 실행/임상 승인과는 다르다. D1은 유일한 Native 방문 구간과 Source 파일/이벤트 Hash를 다시 검사하고 현재 minute0 이전 증상 이벤트만 추가한다. 과거324개 이벤트는 현재 증상으로 복사하지 않는다. D2는 원본 doctor/patient 필드와 원래 associated_symptoms를 보존하면서 합성 chief/symptom만 추가한다. 이미 실제 chief_complaint가 있으면 덮어쓰지 않고 차단한다.\n\n최종 미실행 preview는 Private validated_previews/이며 SOURCE_PRESERVING_INITIAL_PREVIEW.v1_1로 구분한다. 초기50개 *_preview.json은 삭제·덮어쓰기하지 않고 REJECTED_EARLY_PREVIEWS.jsonl로 분리했다. 이는50개 환자 데이터 행을 제외했다는 의미가 아니다. 50개 AUGMENTED_CASES는 그대로 보존된다.\n\nReference/정답 전문과·문헌 profile/provenance·생성 정책/Source cause는 모델 preview에 포함하지 않는다. 권장 검사·미래 결과·과거 증상·임의 신체 소견을 초기 objective Evidence로 추가하지 않는다. Source 공개 grant/clinical reviewer/Native Action/likelihood는 만들지 않는다.\n',encoding='utf-8')
    (DOC/'ACTION_COMPATIBILITY.md').write_text((DOC/'ACTION_COMPATIBILITY.md').read_text(encoding='utf-8')+'\n## 실제 기존 Binding 재검증\n\n```json\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n```\n\n기존 bind_cases의 최신 시각/동시값 충돌/정확한 CSV행 규칙까지 재사용한 비생성 검사다. 절차·약물·careplan 기록은 별도 원본 감사로 보존하며, 실제 검사 해석이나 현재 공개 가능한 정보로 자동 승격하지 않는다.\n',encoding='utf-8')
    summary=read(PUB/'augmentation_summary.json');source=summary['counts']['recovered_facts'];authored=summary['counts']['synthetic_chief_complaints']+summary['counts']['synthetic_symptoms']+summary['counts']['literature_test_recommendations']
    ratio={'source_recovered_event_count':source,'authored_chief_symptom_and_recommendation_item_count':authored,
        'source_fraction_of_these_item_rows':source/(source+authored),'authored_fraction_of_these_item_rows':authored/(source+authored),
        'note':'Counting heterogeneous provenance rows, not clinical information quantity or diagnostic sufficiency; synthetic numeric observations remain zero.'}
    atomic_json(PUB/'provenance_counts_and_ratios.json',ratio)
    (DOC/'NEXT_STEPS.md').write_text((DOC/'NEXT_STEPS.md').read_text(encoding='utf-8')+'\nOriginal50과 D1/D2는 같은 환자들의 파생 View다. 서로를 독립 Train/Test 또는 새로운 Holdout으로 나눌 수 없다. 기존 Original50의 Test 역할·과거 결과는 보존하며, 이 label-conditioned 자료를 학습에 사용하고 같은 환자로 독립 평가했다고 주장하지 않는다. Source Condition2건의 다음날 Reference 문제와48건의 intraday 미검토도 유지한다.\n',encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__':main()
