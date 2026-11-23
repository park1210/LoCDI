# -*- coding: utf-8 -*-
"""Non-generating preservation/privacy closeout. No clinical approval creation."""
from pathlib import Path
import ast,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import original_cases,read,sha_file,DATA
from app.locdi.source_paths import MANIFEST
from app.locdi.routing_collector import atomic_json
NS='ORIGINAL50_LITERATURE_AUGMENTATION_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS
DOC=ROOT/'docs/B0_B4_실험/19_Original50_문헌기반_데이터보강'

def main():
    before=read(PR/'protection_before.json');changed=[p for p,h in before.items() if not Path(p).is_file() or sha_file(p)!=h]
    source=read(PR/'memory_source_verification.json');native=read(PR/'native_source_pins.json')
    source_changed=[p for p,h in {**source['csv']['files'],**native,source['database_path']:source['database_sha256']}.items() if sha_file(p)!=h]
    if sha_file(DATA)!=source['dataset_sha256'] or sha_file(MANIFEST)!=source['csv']['manifest_sha256']:raise ValueError('ORIGINAL_SOURCE_MANIFEST_CHANGED')
    preservation={'status':'PASS' if not changed and not source_changed else 'FAIL','protected_project_files_checked':len(before),
        'changed_protected_files':changed,'changed_native_csv_or_database_files':source_changed,
        'native_json_files_rechecked':len(native),'csv_files_rechecked':len(source['csv']['files']),
        'original_dataset_sha256':sha_file(DATA),'source_manifest_sha256':sha_file(MANIFEST),
        'actual_LLM_generation':0,'real_data_utility_fitting':0}
    atomic_json(PUB/'preservation_verification.json',preservation)
    if preservation['status']!='PASS':raise ValueError('PROTECTED_SOURCE_OR_RUN_CHANGED')
    tests=read(PUB/'test_results_final.json');provenance=read(PUB/'provenance_validation.json')
    if tests['PASS']!=213 or any(tests[k] for k in ('FAIL','ERROR','SKIP')) or provenance['status']!='PASS':raise ValueError('TEST_OR_PROVENANCE_NOT_COMPLETE')
    test_closeout={**tests,'tests_run':214,'PASS':214,
        'additional_executed_test':'test_original50_augmented_dataset.ActualAugmentedDataset.test_validated_provenance_pointers_and_counts',
        'execution_scope_note':'213-test recorded batch plus one new provenance-resolution regression; 214 distinct tests, not whole repository discovery.'}
    atomic_json(PUB/'test_results_closeout.json',test_closeout)
    files=['app/locdi/literature_augmentation_catalog.py','app/locdi/original50_augmentation.py',
        'scripts/build_original50_literature_augmentation.py','scripts/validate_original50_augmentation_outputs.py',
        'scripts/validate_augmentation_provenance.py','scripts/closeout_original50_literature_augmentation.py',
        'tests/locdi/test_original50_literature_augmentation.py','tests/locdi/test_original50_augmented_dataset.py']
    for name in files:ast.parse((ROOT/name).read_text(encoding='utf-8'),filename=name,feature_version=(3,9))
    artifacts={p.name:sha_file(p) for p in PR.iterdir() if p.is_file() and p.name in (
        'AUGMENTED_CASES.jsonl','ORIGINAL50_AUGMENTATION_SUMMARY.md','ORIGINAL50_SOURCE_AUDIT.csv',
        'AUGMENTATION_PROVENANCE_VALIDATED.csv','MISSING_INFORMATION_MATRIX.csv','REJECTED_AUGMENTATIONS.jsonl')}
    atomic_json(PR/'final_reproducibility_manifest.json',{'new_code_sha256':{n:sha_file(ROOT/n) for n in files},
        'private_artifact_sha256':artifacts,'input_sha256':{'Original50':sha_file(DATA),'archive_source_manifest':sha_file(MANIFEST)},
        'LLM_generation':0,'utility_fitting':0,'native_B0_B4_execution':0})
    summary=read(PUB/'augmentation_summary.json');counts=summary['counts'];ratios=read(PUB/'provenance_counts_and_ratios.json');dataset=read(PUB/'actual_dataset_validation.json')
    report=f'''# 검증 결과와 적용 범위

Original50 전체50명을 처리했다. 검증된 결과는 별도 후보 데이터셋이며 기존 Original50, Source, Gold, Scorer, Prompt, Registry, Frozen M5, Memory/Routing/VoI 공식, 임상 Gate와 완료 Run은 변경하지 않았다.

|항목|실제 결과|해석|
|---|---:|---|
|Source 복구가 있는 환자|40/50|현재 증상 복구가 있는 환자는1/50|
|Source 복구 이벤트|325|현재1/과거324; 기록된 방문 ID가 아닌 유일한 Native 방문 시간 구간으로 도출|
|합성 후보 환자|50/50|Source Reference를 사용한 label-conditioned 개발 자료|
|합성 주호소|50|원본의 실제 주호소가 아님|
|합성 증상 항목|51|38명에게1~2개,12명은 새 증상 보고 없음|
|권장 검사|106|실제 주문/결과가 아님|
|생성한 검사 수치/결과|0|기존 측정값·단위·시각 보존|
|기존 Native M5 Binding 일치 후보|16|Source 결과와 action의 구조 연결; 공개/실행 승인 아님|
|기존 Source는 있으나 Action 불가|7|새 메뉴·likelihood를 추가하지 않음|
|문헌 권장이나 결과 미관측|83|없는 결과를 채우지 않음|
|임상 검토 완료|0/50|진단 충분성 UNREVIEWED|
|실제 Ollama 호출/연구 Utility fitting|0/0|B0~B4 실행도0회|

공식1차기관26개 페이지를 직접 확인했고 접근 실패한 NIA2개(405), NIDDK1개(404)는 사용하지 않았다. 문서의 마지막 검토/수정일을 원래 발행 연도로 바꾸지 않았다. 알 수 없는 발행 연도는UNKNOWN이다. 관련 URL·기관·문서명·검토일·확인일과 근거 요약은 DISEASE_EVIDENCE_CATALOG.md 및 Public disease_evidence_catalog.json에 있다.

Source 증상 후보31,189개 중30,864개는 제외했다: 미래4,853개,0이하 engine 값7,968개,유일한 방문 구간을 확인할 수 없는18,043개다. 0점은 환자의 증상 부인으로 만들지 않았으며 최종 resolved 상태를 과거로 투영하지 않았다. SOURCE_RECOVERED는 engine 기록의 사실 복구이지 실제 임상 공개시각·환자 진술·임상 severity scale의 검증이 아니다.

원본의 현재 관측2,509행과 고정 Memory72,590개를 보존했다. 추가로 현재617개 처치/약물/면역/소모품/기기/careplan 기록을 Private에 분리했다. 해당 기록의 존재를 판독 결과로 간주하거나 모델 초기 입력에 넣지 않았다. FHIR2개 파일은 기관·의료인 Bundle뿐이었다. 원본의 initial chief complaint와 초기 narrative는 여전히 NOT_RECORDED이며1명의 별도 engine 증상 이벤트를 이것과 혼동하지 않는다.

Provenance 항목 수 기준 Source325:합성207(주호소50+증상51+권장검사106)이다. 비율 {ratios['source_fraction_of_these_item_rows']:.2%}:{ratios['authored_fraction_of_these_item_rows']:.2%}는 이질적인 항목 행의 구성비일 뿐 실제 정보량·신뢰도·진단 기여 비율이 아니다.

## 실행한 검증

관련16개 모듈의213개 배치와 추가 provenance pointer 검사1개, 총214개 서로 다른 테스트가 통과했다. 전체 저장소 discovery로 확대하지 않는다. Python3.9 AST8개 파일도 통과했다. 기존 Knowledge/Memory/Synthetic Utility/Common Menu/M5/Source-time 회귀를 포함했으며, 테스트 내부 TEST_ONLY 수식 fixture를 실제 Utility 재학습으로 사용하지 않았다.

50명 실제 Case fingerprint, 원본 code/value/unit/date, Native JSON/CSV/SQLite/Manifest Hash, Source 이벤트 포인터, 환자·방문·cutoff, literal 정답명·전문과 비노출, D0 exact-copy, D2 Source 필드 보존, 실제 M5 Binding, 신규 Action 없음, 문헌 feature membership, 기관 URL의 실제 조회 상태를 확인했다. 문헌과 개별 환자의 임상적 정합성/진단 가능성을 자동 PASS시키지 않았다.

보호된 프로젝트 파일 {len(before):,}개와 Native JSON50개, CSV{len(source['csv']['files'])}개 및 기존 Source DB/Manifest/Original50을 전후 비교해 변경0건이다. clinical_reviewed=false, clinical_accuracy=NOT_EVALUATED, research_evaluation_ready=false를 유지한다.

## 작성 중 발견한 기술 문제와 보존

- Windows의 pre-1970 epoch 변환 문제는 새 모듈만 UTC epoch+timedelta로 수정했다. 실제 Source 날짜는 바꾸지 않았다.
- 개발 Preview opt-in을 Source 접근 전에 검사하도록 새 Adapter의 순서만 보완했다. 기존 임상/Source Guard는 변경하지 않았다.
- 초기50개 preview는 원본 associated_symptoms 표시를 빠뜨리지 않는 최종 validated_previews로 대체했지만 기존 파일을 삭제·덮어쓰지 않았다. REJECTED_EARLY_PREVIEWS.jsonl에 분리했으며 환자50개 행은 제외하지 않았다.
- 초기 합성 provenance의 object pointer207개를 별도 검증 파일에서 바로잡았다. 원래 CSV는 보존하고 **AUGMENTATION_PROVENANCE_VALIDATED.csv**를 사용한다. 532개 행의 실제 Source/합성 항목 포인터와 지문을 모두 확인했다. 환자 Source/Reference/합성 내용은 변경하지 않았다.
- 초기 실패 로그도 Private에 보존했다. 이들은 비생성 자료 작성 과정의 오류이며 실제 모델 Run을 재시도한 것이 아니다.

## 남은 경계

D1/D2는 새 분석/개발 preview이며 Native runtime에 자동 등록되지 않는다. 임상 검토가 없다는 이유만으로 모든 개발 실험을 금지하는 것은 아니지만, 실제 실행 전에는 새 Source 공개 Proxy 계약, 역할/데이터 노출·모델·예산·운영자 승인이 필요하다. 기존 v1 Source 공개 Guard를 통과했다고 표시하지 않는다. 임상 성능 주장/CLINICAL_EVALUATION은 기존 독립 검토 Gate를 그대로 따른다.

D2는 Reference를 조건으로 작성한 것이므로 정답명을 숨겨도 강한 힌트가 된다. 원본10명의 오류나 성능은 생성 규칙에 사용하지 않았다. 기존 Original50의 Test 역할은 바꾸지 않고 D0/D1/D2를 독립 Train/Test로 분리하지 않는다. 새로운 Holdout이 아니며, 진단 일치가 증가하더라도 독립 일반화나 기존 Memory/Routing/VoI의 개선으로 주장할 수 없다. 다음날 Reference2건·같은 날 intraday 미검토48건도 보존한다.
'''
    (DOC/'VALIDATION_REPORT.md').write_text(report,encoding='utf-8')
    public_readme=(DOC/'README.md').read_text(encoding='utf-8')
    public_readme+='\n## 최종 사용 파일\n\n검증 결과는 VALIDATION_REPORT.md를 먼저 확인한다. Private main은 ORIGINAL50_AUGMENTATION_SUMMARY.md, 사례는 AUGMENTED_CASES.jsonl, Source 감사는 ORIGINAL50_SOURCE_AUDIT.csv, 부족 정보는 MISSING_INFORMATION_MATRIX.csv다. 최종 lineage는 **AUGMENTATION_PROVENANCE_VALIDATED.csv**, 최종 초기 상태 preview는 **validated_previews/Patient01~50.json**이다. 초기 AUGMENTATION_PROVENANCE.csv와 *_preview.json은 폐기 후보 이력으로만 보존한다. Source event 제외는 REJECTED_AUGMENTATIONS.jsonl, 초기 preview/pointer 문제와 접근 실패 문헌은 각 별도 REJECTED 파일에 남겼다.\n'
    (DOC/'README.md').write_text(public_readme,encoding='utf-8')
    cases=original_cases();public_text='\n'.join(p.read_text(encoding='utf-8-sig') for folder in (DOC,PUB) for p in folder.glob('*') if p.is_file() and p.suffix in ('.json','.md','.csv'))
    for c in cases:
        if c['patient_id'] in public_text or c['agent_visible']['index_encounter_id'] in public_text:raise ValueError('PUBLIC_PATIENT_OR_VISIT_IDENTIFIER_LEAK')
    atomic_json(PUB/'privacy_verification.json',{'status':'PASS','patient_current_encounter_identifiers_checked':50,
        'public_content':'AGGREGATES_AND_GENERAL_LITERATURE_ONLY','case_references_and_clinical_values':'PRIVATE_ONLY'})
    final={'status':'AUGMENTATION_CANDIDATES_BUILT_AND_STRUCTURALLY_VALIDATED','patients':50,
        'source_recovery_patients':40,'current_source_recovery_patients':1,'source_recovered_events':325,
        'authored_chief_complaints':50,'authored_symptoms':51,'literature_recommended_tests':106,'synthetic_test_results':0,
        'primary_pages_checked':26,'source_candidates_rejected':30864,'tests_PASS':214,'protected_files_unchanged':len(before),
        'native_runtime_admitted':False,'clinical_reviewed':False,'clinical_diagnosability':'UNREVIEWED',
        'actual_LLM_generation':0,'utility_fitting':0,'B0_B4_execution':0,'data_sha256':sha_file(PR/'AUGMENTED_CASES.jsonl')}
    atomic_json(PUB/'completion_summary.json',final);print(json.dumps(final))
if __name__=='__main__':main()
