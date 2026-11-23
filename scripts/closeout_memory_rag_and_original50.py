# -*- coding: utf-8 -*-
"""Final non-generating verification and documentation of two isolated tasks."""
from pathlib import Path
import ast,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.routing_collector import atomic_json
from scripts.run_memory_rag_experiment import check
A='MEMORY_RAG_LEGACY_RECOVERY_20261121_v1';B='ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1'

def main():
    cache={};publicA=ROOT/'outputs/research_evaluation'/A;privateA=ROOT/'outputs/private/research_evaluation'/A
    def sha(p):
        key=str(Path(p).resolve())
        if key not in cache:cache[key]=sha_file(key)
        return cache[key]
    for ns in (A,B):
        pr=ROOT/'outputs/private/research_evaluation'/ns;pub=ROOT/'outputs/research_evaluation'/ns
        before=read(pr/'protection_before.json');changed=[n for n,h in before.items() if not Path(n).is_file() or sha(n)!=h]
        report={'status':'PASS' if not changed else 'FAIL','protected_files_checked':len(before),'changed_files':changed,
            'scope':'Recorded pre-work 1850 protected Core/validation/prompts/configs/data/pinned completed run and Utility files; not a claim to hash every repository artifact',
            'actual_LLM_generation':0,'utility_fitting':0}
        atomic_json(pub/'preservation_verification.json',report)
        if changed:raise ValueError('PROTECTED_ASSET_CHANGED')
    old={n:read(publicA/n/'plan.json')['fingerprint'] for n in ('memrag_m2_s1_001','memrag_k2_s1_001')}
    active={}
    for name in ('memrag_m2_s1_002','memrag_k2_s1_002'):
        p,_=check(name);active[name]={'status':'STATIC_PIN_VALID','fingerprint':p['fingerprint'],'arms':p['arms'],
            'registered_patients':p['registered_patients'],'registered_episodes':p['registered_episodes'],'generation_cap':p['generation_total'],
            'operator_approval_exists':(privateA/name/'operator_approval.json').exists(),'actual_run_exists':(privateA/name/p['run_id']).exists()}
        if active[name]['operator_approval_exists'] or active[name]['actual_run_exists']:raise ValueError('UNEXPECTED_LIVE_EXECUTION')
    atomic_json(publicA/'active_execution_plans.json',{'active':active,'superseded_unexecuted_plan_fingerprints_preserved':old,
        'reason':'Read-only repeat analysis contract was finalized after initial preparation. Old pins were not updated. Only new 002 IDs are for user execution.'})
    files=['app/locdi/original_source_inventory.py','app/locdi/diagnostic_memory_rag.py','app/locdi/memory_rag_backend.py',
        'scripts/run_memory_rag_experiment.py','scripts/audit_original50_patient_diagnostics.py','scripts/supplement_original50_audit.py',
        'scripts/classify_original50_export_differences.py','scripts/probe_memory_rag_offline.py','scripts/replay_memory_legacy_offline.py',
        'scripts/replay_diagnostic_memory_rag.py','scripts/finalize_original50_audit.py','scripts/document_memory_rag_completion.py',
        'scripts/complete_original50_reference_timing.py','scripts/complete_original50_action_linkage.py',
        'tests/locdi/test_diagnostic_memory_rag.py','tests/locdi/test_original50_patient_audit.py','tests/locdi/test_memory_rag_execution_contract.py',
        'tests/locdi/test_original50_reference_completeness.py','scripts/closeout_memory_rag_and_original50.py']
    for name in files:ast.parse((ROOT/name).read_text(encoding='utf-8'),filename=name,feature_version=(3,9))
    atomic_json(publicA/'final_code_manifest.json',{'new_python_files':{n:sha(ROOT/n) for n in files},
        'new_powershell_wrapper_sha256':sha(ROOT/'scripts/Invoke-LoCDIMemoryRAG.ps1'),'python39_ast_files':len(files),
        'old_files_changed':[],'actual_LLM_generation':0,'utility_fitting':0})
    docA=ROOT/'docs/B0_B4_실험/16_Memory_RAG_회귀복구_검증'
    for p in docA.glob('*.md'):
        text=p.read_text(encoding='utf-8').replace('memrag_m2_s1_001','memrag_m2_s1_002').replace('memrag_k2_s1_001','memrag_k2_s1_002')
        text=text.replace('165개','169개').replace('165 PASS','169 PASS').replace('165 tests:165','169 tests:169').replace('13개 모듈','14개 모듈').replace('신규 M2/K2/실행 계약35개','신규 M2/K2/실행 계약37개').replace('Original50 감사8개','Original50 감사10개')
        p.write_text(text,encoding='utf-8')
    (docA/'USER_EXECUTION.md').write_text((docA/'USER_EXECUTION.md').read_text(encoding='utf-8')+'\n## 최종 Plan 지문\n\n```json\n'+json.dumps(active,ensure_ascii=False,indent=2)+'\n```\n\n초기 미실행001 계획은 읽기 전용 분석 반복 계약을 보완하기 전 코드 지문을 보존한 상태다. 소급 수정하지 않았으며 **실제 실행은002만 사용한다.** 승인 파일과 Run은 두002 모두 아직 없다.\n',encoding='utf-8')
    (docA/'TEST_RESULTS.md').write_text((docA/'TEST_RESULTS.md').read_text(encoding='utf-8')+'\n최종 추가 검증: 저장 분석 반복 시 byte-identical/read-only, 다른 기존 분석 덮어쓰기 거부, Original50 Condition Reference50행의 CSV/Native 날짜 일치 및 비공개 경계. 최종 로그는 test_results_closeout.txt/json이며169개 전부PASS다.\n',encoding='utf-8')
    docB=ROOT/'docs/B0_B4_실험/17_Original50_환자별_진단증상검사_분석'
    summaryB=read(ROOT/'outputs/research_evaluation'/B/'reference_timing_summary.json');coverage=read(ROOT/'outputs/research_evaluation'/B/'final_coverage_with_reference_timing.json')
    (docB/'TEST_AND_PRESERVATION.md').write_text('# 테스트와 보존\n\nOriginal50 audit8개와 Reference completeness2개, 총10개가 관련169개 회귀에 포함되어 모두PASS했다. 이는 구조·Source·분모·누출 경계 검사이며 임상 Reference 승인이나 검색 품질 평가가 아니다.\n\n각 독립 작업에서 사전1850개 보호 파일을 전후 Hash 비교해 변경0건이다. Memory72,590개 보존, 신규 환자/LLM Generation/Utility fitting0회다. Corpus/Prompt/Registry/Frozen M5/기존 Source·Scorer·완료 Run은 변경하지 않았다.\n\nReference50행은 원래 공개 Event Store 밖의 정확한 conditions.csv 및 Native JSON의 동일 환자·방문·코드로 확인했다. 48행은 같은 날이나 intraday 미검토,2행은 index 이후 날짜다. 이 정보는 Private 평가용으로만 사용하며 실제 모델 입력에 추가하지 않았다.\n\n최종 집계는 final_coverage_with_reference_timing.json, 최종 export 차이 분류는 native_export_difference_summary_refined.json이다. 초기 raw literal nonmatch 집계와 처음의 빈 event-store Reference 조회는 중간 점검 기록으로 보존한다. 최종 Reference 결과는 reference_timing_summary.json을 따른다.\n\n실제 Core Action의 이름(order_test/ask_history)을 사용해 Source Event와 연결했다. 69개 현재 Source 행에 실제 주문/문진이 있었고 조건×Source 행 단위249개 주문이 기록되었다. 69개 행은 적어도 한 조건의 저장 payload에서 수신되었다. 조건 수/요청 수와 고유 Source 행 수의 분모를 혼합하지 않는다.\n',encoding='utf-8')
    # Remove the obsolete empty timing JSON display from this unfinished new report only.
    p=docB/'SUMMARY_AND_COVERAGE.md';text=p.read_text(encoding='utf-8')
    text=text.replace('"reference_timing_record_counts": {}','"reference_timing_record_counts": '+json.dumps(coverage['reference_timing_record_counts'],ensure_ascii=False))
    text=text.replace('위의 최초 timing 빈 집계는 이 실제 원천 조회 결과로 완성했다.','최초 event-store 조회의 timing 빈 집계를 실제 원천 조회 결과로 완성했다.')
    p.write_text(text,encoding='utf-8')
    # Privacy check across public documents and summaries; never copy raw records into docs.
    from app.locdi.original_source_inventory import original_cases
    public_text='\n'.join(p.read_text(encoding='utf-8-sig') for d in (docA,docB,publicA,ROOT/'outputs/research_evaluation'/B) for p in d.glob('*') if p.is_file() and p.suffix in ('.md','.json','.csv'))
    for c in original_cases():
        if c['patient_id'] in public_text or c['agent_visible']['index_encounter_id'] in public_text:raise ValueError('PUBLIC_IDENTIFIER_LEAK')
    for ns in (A,B):atomic_json(ROOT/'outputs/research_evaluation'/ns/'public_privacy_verification.json',{'status':'PASS','patient_and_current_encounter_identifiers_checked':50,'raw_patient_data_in_docs':False})
    print(json.dumps({'status':'TWO_INDEPENDENT_TASKS_COMPLETED','tests':169,'protected_files_each':1850,'active_plans':active,
        'Original50_patients':50,'memory_records':72590,'Source_reference_timing':summaryB['timing_record_counts'],'actual_LLM_generation':0,'fitting':0}))
if __name__=='__main__':main()
