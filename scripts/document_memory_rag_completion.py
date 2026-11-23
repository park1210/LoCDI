# -*- coding: utf-8 -*-
"""Document only new isolated work; no generation, fitting or old result edits."""
from collections import Counter
from pathlib import Path
import csv,json,shutil,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
NS='MEMORY_RAG_LEGACY_RECOVERY_20261121_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS
DOC=ROOT/'docs/B0_B4_실험/16_Memory_RAG_회귀복구_검증'

def main():
    if DOC.exists():raise FileExistsError('NEW_DOCUMENT_FOLDER_ONLY')
    DOC.mkdir(parents=True)
    tests=read(PUB/'test_results_final.json');fake=read(PUB/'TEST_ONLY_core_report.json');replay=read(PUB/'legacy_replay_report.json');projection=read(PUB/'offline_projection_summary.json')
    plans={n:read(PUB/n/'plan.json') for n in ('memrag_m2_s1_001','memrag_k2_s1_001')}
    utility=ROOT/'outputs/private/research_evaluation/SYNTHETIC_PILOT_UTILITY_20261121_v1/syn15_initial_full_005/development_utility.json'
    uh=sha_file(utility)
    if uh!='429a8177ae4af5a9d63b81b8d354aa0987e5457e7f1dd5c47286ca3147f1360b':raise ValueError('SYNTHETIC_UTILITY_CHANGED')
    base=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1';publicbase=ROOT/'outputs/research_evaluation/B34_CONNECTION_v1'
    legacy_state={}
    for name in ('knowledge_e0r_p1_001','knowledge_e1_p1_001'):
        p=read(publicbase/name/'plan.json');run=base/name/p['run_id']
        legacy_state[name]={'plan_fingerprint':p['fingerprint'],'run_exists':run.exists(),
            'status':'SAVED_RUN_EXISTS_INSPECT_SEPARATELY' if run.exists() else 'NOT_EXECUTED',
            'code_pin_matches':all(sha_file(ROOT/k if not Path(k).is_absolute() else k)==v for k,v in p['code_hashes'].items()),
            'input_pin_matches':all(sha_file(k)==v for k,v in p['input_hashes'].items())}
    atomic_json(PUB/'prior_knowledge_plan_audit.json',legacy_state)
    # Archive unchanged current Legacy source/config, not a claim to recreate missing historical code pins.
    before=read(PR/'protection_before.json');snapshot={n:h for n,h in before.items() if Path(n).suffix in ('.py','.json','.txt','.yaml','.yml') and
        any(Path(n).is_relative_to(ROOT/d) for d in ('app','validation','prompts','configs','scripts'))}
    for p in ROOT.glob('requirements*.txt'):snapshot[str(p)]=sha_file(p)
    with zipfile.ZipFile(PR/'legacy_current_code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,h in sorted(snapshot.items()):
            if sha_file(n)!=h:raise ValueError('LEGACY_CODE_CHANGED_BEFORE_SNAPSHOT')
            z.write(n,Path(n).relative_to(ROOT).as_posix())
    atomic_json(PR/'legacy_code_snapshot_manifest.json',{'files':snapshot,'archive_sha256':sha_file(PR/'legacy_current_code_snapshot.zip'),
        'scope':'CURRENT_UNCHANGED_LEGACY_DEPENDENCIES_NOT_A_HISTORICAL_ENVIRONMENT_OR_LLM_REGENERATION'})
    # Detailed references/predictions remain private; public regression matrix has only anonymous aggregate-compatible fields.
    t=read(ROOT/'outputs/private/research_evaluation/ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1/baseline_trace_details.json')
    old={(r['original_ordinal'],r['condition']):r for r in t['old']};latest={(r['original_ordinal'],r['condition']):r for r in t['latest']}
    matrix=[]
    for key,b in latest.items():
        if key[1]!='B2':continue
        a=old[key];pain=lambda rs:sum('Pain severity' in r['text'] for r in rs)
        matrix.append({'panel_ordinal':b['panel_ordinal'],'old_B2_status':a['status'],'latest_B2_status':b['status'],
            'old_B2_source_exact':a['source_exact'],'latest_B2_source_exact':b['source_exact'],
            'latest_B0_source_exact':latest[key[0],'B0']['source_exact'],
            'correct_to_incorrect':a['source_exact'] is True and b['source_exact'] is False,
            'old_memory_count':len(a['memory']),'latest_memory_count':len(b['memory']),
            'old_pain_count':pain(a['memory']),'latest_pain_count':pain(b['memory']),
            'same_source_binding':a['source_binding_fingerprint']==b['source_binding_fingerprint'],
            'same_output_schema':a['output_schema_fingerprint']==b['output_schema_fingerprint'],
            'same_code_pins':a['code_pins']==b['code_pins'],'causal_attribution':'CAUSE_UNDETERMINED',
            'new_C2_C4_diagnosis_outcome':'NOT_RUN'})
    matrix.sort(key=lambda r:r['panel_ordinal'])
    with (DOC/'B2_CORRECTNESS_REGRESSION_MATRIX.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
    code=['app/locdi/original_source_inventory.py','app/locdi/diagnostic_memory_rag.py','app/locdi/memory_rag_backend.py',
        'scripts/run_memory_rag_experiment.py','scripts/Invoke-LoCDIMemoryRAG.ps1','scripts/probe_memory_rag_offline.py',
        'scripts/replay_memory_legacy_offline.py','scripts/replay_diagnostic_memory_rag.py',
        'tests/locdi/test_diagnostic_memory_rag.py','tests/locdi/test_memory_rag_execution_contract.py']
    atomic_json(PUB/'new_files_manifest.json',{'new_files':{n:sha_file(ROOT/n) for n in code},'old_runtime_files_modified':[],
        'actual_LLM_generation':0,'utility_fitting':0,'planned_calls_each':24})
    count=projection['counts']
    documents={
    'README.md':'''# Legacy를 보존한 Memory/RAG 독립 검증

2026-11-22 완료. 기존 Core·Top5·Prompt·Scorer·Source·Frozen M5·Utility 파일은 수정하지 않고 새 opt-in 모듈과 실행기를 추가했다. 이번 작업의 실제 LLM Generation0회, 실데이터 Utility fitting0회다.

신규 기능과 기존 경로의 계약/통합 검증은 완료했다. 새로운 기능의 실제 진단 효과는 NOT_RUN이다. 임상 Reference 검토나 공식 임상 Gate를 대신하지 않는다.

- 기본 `legacy_top5/none`은 기존 make_common_backend에 인수 그대로 직접 위임한다.
- M2는 Source가 검증된 전체 적격 과거 기록에서 BM25 Top5를 검색한다.
- K2는 현재 공개 관측만으로 기존15종 Corpus의 BM25 Top3를 선택하며 외부 지식으로만 전달한다.
- C0/C2와 C0/C4 1명 paired 계획을 따로 준비했다. 각2 Episode/최대24 Generation, 두 계획 모두 실행하면 최대48회다. C0~C5 전체 자동 실행은 없다.

관련 회귀165 PASS. 실제 Core/Source/Scheduler/Exporter를 통과한 Fake 경로7개 PASS. 저장628개 요청 Hash와 selective321개 Memory 재현 PASS. 자세한 범위는 TEST_RESULTS.md와 LEGACY_BASELINE_PRESERVATION.md를 따른다.

Original50 전체 감사는 별도 17번 문서/Private Namespace다. 원본50 감사에서 검색 수식은 변경하지 않았으며, 여기의 M2 진단 성능과 혼합하지 않는다.
''',
    'LEGACY_BASELINE_PRESERVATION.md':'''# Baseline과 보존 범위

과거 recovery_llm_all_001은50/50 완료333 Generation, 최신 runtime_recovery_all_001은45완료/1실패/4미실행295 Generation이다. 과거 B2 Source Exact2/10, 최신 B2 0/9(등록10명 중9완료), 최신 B0 정답은 패널1·3·5다. 결과를 다시 채점해 늘리지 않았다.

두 실행의 Source binding은 동일하지만 Code Pin·출력 Schema·Catalog/Runtime 계약이 다르다. 기존2/10과 최신0/9를 Memory의 단일 인과 효과로 해석하지 않는다. 잃은 두 정답은 패널1·5이며, 구체적인 Prediction·rationale·검사·Memory 원문은 Private 비교 자료에만 보존했다.

저장 요청628개를 변조 없이 읽어 request_fingerprint가 모두 일치함을 확인했다. 그 중selective321개는 원래 memory_query와 수정하지 않은 PatientMemoryStore.retrieve로 선택 ID·순서·시각·점수·전체 컨텍스트까지 정확히 재현했다. B1의 context-budgeted history를 selective 검색으로 재해석하지 않았다.

Fake Core 실행에서 기존 backend → C0 → 신규 기능들 → rollback의 전체 request(model/options/system/user/schema 포함)이 정확히 같았다. 이는 새로운 C0 wire와 현재 Legacy backend 사이의 불변 검증이다. 옛2/10 실행의 사라진 코드 버전을 새 코드라고 주장하거나 실제 모델 재생성의 bitwise 동일성을 주장하지 않는다.

Private legacy_current_code_snapshot.zip과 SHA manifest에 이번 변경 전 현재 Legacy 코드/설정을 보존했다. 데이터/가상환경/옛 코드 버전의 완전한 배포본이 아니므로 historical runtime 복원 성공과 구분한다. 두 작업에서 각각1850개 기존 보호 파일을 전후 Hash 대조한 결과는 preservation_verification.json을 따른다.
''',
    'DIAGNOSTIC_MEMORY_RETRIEVAL.md':'''# M2 — diagnostic_bm25_full_pool.v1

구현: app/locdi/diagnostic_memory_rag.py의 public_projection, DiagnosticMemoryRetriever.retrieve; 연결: memory_rag_backend.make_memory_rag_backend. Core의 기존 retrieve/memory_query는 수정하지 않았다.

실제 현재 Observation의 chief complaint, 답변된 문진, 객관 관측만 Query로 사용한다. 수신된 검사와 정확히 일치하는 기존 Source binding의 description/code/value/unit을 의미 표시로 재사용한다. 실행 시 Runtime 완료 검사/객관 관측 또는 실제 Ask history와 minute를 다시 확인한다. 아직 수신하지 않은 메뉴의 값은 검색할 수 없다. working_assessment·과거 Memory의 진단명·평가 Reference·Gold를 Query로 쓰지 않는다. 초기 symptom에 통증 척도 항목명만 있는 경우 그 이름만으로 임상 증상을 추정하지 않는다. 실제 Ask 후 공개된 통증 값은 남긴다.

후보는 동일 환자·다른 과거 Visit·cutoff 이전 기록 전체다. CSV row/환자/방문/값/시각/Source fingerprint를 검사하고 superseded는 선택하지 않는다. 같은 Source row의 완전 중복만 제거하며 시간별 반복 측정은 보존한다. BM25 k1=1.2,b=0.75, query term 집합, IDF=ln(1+(N-df+0.5)/(df+0.5))다. 동점은 available_at 최신순→Memory ID순, 양의 점수만 Top5, 기존 문자 예산 안에서 완전한 기록만 제공한다. 수치값 단독은 검색어가 아니고 단위 변환/정상·비정상 판정은 하지 않는다.

모든 선택의 원본 Memory ID·Source·observed/available/visit 시각·HISTORICAL·점수·Source 검증 상태를 Private memory_rag_audit.json에 남긴다. CURRENT Evidence와 과거 기록을 섞지 않는다. 코드+단위가 같은 과거 수치 시계열은 sidecar로 보존하며 임상 추세는 판정하지 않는다. M2는 임상 relevance/stale/conflict 정답이 아니다.

전체 적격 후보 검색과 Top5 내부 재정렬은 다르다. 실제 C2는 전체 후보만 사용한다. 오프라인 분석에서는 Top5 재정렬을 별도 TEST_ONLY 비교로 기록했으며61/73요청에서 선택 ID가 달랐다. 이를 같은 알고리즘의 결과로 합치지 않는다.

저장된 최신 B2 73요청에 대한 비생성 투영: 마지막9명45기록의 통증은 Legacy43개→M2 33개다. M2도 통증을 많이 선택하며, 이 수치는 relevance 개선이나 진단 개선을 입증하지 않는다. 초기9요청은 Query 정보가 없어 M2 empty였고 Oracle fallback은 없다.
''',
    'MEDICAL_RAG_IMPLEMENTATION.md':'''# K1/K2 구현과 경계

MedicalKnowledgeIndex와 기존 canonical15 Corpus를 그대로 재사용했다. 새 웹 자료·환자 정보·학습 자료를 추가하지 않았다. 문헌 URL/발행기관/버전/기존 source-check 상태는 corpus snapshot의 references를 전달한다. 이번 작업에서 원격 최신성을 재검증한 것은 아니다.

K1은 동일 common pack15종, K2는 public_projection을 거친 현재 공개 정보의 BM25 Top3다. 검색은 각 모델 요청 직전에 실행되며 external_medical_knowledge라는 독립 payload key로만 제공한다. 문서 ID·score·Source URL·Corpus fingerprint·Query fingerprint·status를 Private checkpoint/sidecar에 남긴다. Common note는 미리 고정된 요약을 쓰고 임의 런타임 요약/환자별 정답 맞춤 수정은 없다.

문헌은 환자의 실제 증상, evidence index, Bayesian likelihood 또는 M5 관측이 아니다. Evidence 배열과 Observation은 수정하지 않고 기존 Schema/Core Evidence Validator를 통과해야 한다. 검색어/결과가 없으면 KNOWLEDGE_UNAVAILABLE, 위조/미검증 Corpus이면 기존 Index가 차단한다.

현재 Native retrieve_medical_knowledge Action은 미등록이며 이번에는 추가하지 않았다. M5 Action/likelihood도 그대로다. B2만으로 C0/C4를 독립 비교하며 B3/B4 지식 결합 효과를 이미 검증했다고 주장하지 않는다.

실제 Core Fake에서 K2 packet은 초기418 bytes(검색 없음), 검사 수신 후3711 bytes였다. 실제 server tokenizer/장기 turn의 Context 적합성은 미검증이므로 기존 byte/token guard를 유지하고 입력 초과 시 절단 없이 중단한다. 저장 B2 73요청 중K2 AVAILABLE64개는 검색 가능성 지표이며 진단 성능이 아니다.
''',
    'SYNTHETIC_UTILITY_PRESERVATION.md':'''# 기존 Synthetic Utility 보존

syn15_initial_full_005 development_utility.json SHA-256:
`'''+uh+'''`

기존11개 적격 환자/55행, positive27·negative28, jaccard_beta_accuracy_v2/min_samples2를 읽기 전용으로 보존했다. 새 fitting0회이며 기존 Model/Prompt/Registry/Source·11개 Train Group의 의미를 변경하지 않았다. 독립 임상 Gate는 UNCHANGED_NOT_ADMITTED다.

이번 M2/K2 paired plan은 B2 ER-only로 Routing OFF/VoI OFF다. Synthetic Utility를 Native B3/B4에 자동 주입하거나 기존 Synthea Utility로 덮어쓰지 않았다. Synthetic15는 Original50 감사의 환자·Memory·Reference 분모에 포함하지 않았다. 신규 모듈의 default delegation과 관련 Synthetic 회귀122개가 포함된 전체 관련165개 검증을 구분한다.
''',
    'EXPERIMENT_MATRIX.md':'''# 비교 조건과 단일 변경 변수

|조건|Memory|Knowledge|용도/실행 상태|
|---|---|---|---|
|C0|legacy_top5|none|공통 기준; 새 실제 실행 NOT_RUN|
|C1|historical_display|none|표시만 변경; Fake PASS, 실제 계획 미등록|
|C2|diagnostic_retrieval|none|Memory 검색만 변경; C0/C2 1명 계획 준비|
|C3|legacy_top5|common_pack|고정 문헌 제공; Fake PASS, 실제 계획 미등록|
|C4|legacy_top5|adaptive_bm25|Knowledge 검색만 변경; C0/C4 1명 계획 준비|
|C5|diagnostic_retrieval|adaptive_bm25|상호작용 후속; 자동 실행/계획 준비 안 함|

두 준비 계획의 환자는 결과를 보지 않고 기존 고정 패널의 첫 환자를 사용한다. 같은 Source/menu/공통15 Catalog/Qwen3B/Prompt/Memory Top5/ER-only/VoI OFF/Evidence wire를 유지한다. C2는 원천 상태가 같아도 Memory Query와 후보 검색 방식이 의도적으로 다르다. C4는 Legacy Memory를 그대로 둔다.

주 지표는 등록 환자 분모의 Source Exact Agreement다. Normalized Match·진료과·certainty·완료/실패/미실행·유효 인용·검사/Ask·실제 HTTP Generation·Prompt token·wall time을 별도 해석한다. Source Exact는 임상 정확도가 아니며 Source Reference의 시점·식별성 한계를 유지한다. 현재 실제 새 결과는 전부 NOT_RUN이다.

1명 기술 실행을 먼저 사용자 검토한 뒤 동일10명 비교를 별도 새 plan으로 prepare할 수 있다(20Episode/최대240 Generation). 10명은 반복 개발 노출 패널이므로 독립 Holdout이 아니다. 평가 정답에 따라 Legacy/New를 선택하는 Oracle fallback이나 양호한 사례만 보고하는 기능은 없다.
''',
    'ROLLBACK_GUIDE.md':'''# 원상 복구와 재현 범위

1. 새 실행기를 사용하지 않으면 기존 B0~B4 CLI의 경로는 전혀 바뀌지 않는다.
2. 새 factory에서 memory_mode=legacy_top5, knowledge_mode=none을 사용하면 원래 make_common_backend에 그대로 위임한다. 추가 payload key나 patch는 실행되지 않는다.
3. Fake Core 비교에서 기존 backend/C0/기능 활성화 후 rollback의 전체 요청 두 개가 EXACT_MATCH였다. 기존 저장 selective321개도 컨텍스트 EXACT_MATCH다.
4. Private legacy_current_code_snapshot.zip/legacy_code_snapshot_manifest.json은 이번 작업 전의 현재 코드와 설정이다. 추후 다른 변경이 있으면 새 작업 디렉터리에서 SHA와 비교하여 검토한다. 현재 파일을 자동 덮어쓰거나 과거 Code Pin을 소급 갱신하는 복구 명령은 제공하지 않는다.

옛2/10의 historical code version과 현재 baseline은 같지 않다. 옛 LLM 응답은 저장 Trace로만 재집계 가능하며 같은 정답률의 재생성이나 bitwise 응답을 보장하지 않는다. Code Pin 불일치 시 옛 Plan 수정이 아니라 새 Plan을 준비한다. 이번의 E0R/E1 옛 준비 Plan은 미실행·현재 Pin 유효로 확인해 그대로 보존했다.
''',
    'TEST_RESULTS.md':'''# 실행한 비생성 검증

- 관련13개 모듈165 tests:165 PASS/0 FAIL/0 ERROR/0 SKIP. 기존 Knowledge/Common/Proxy/M5/Source-time96개, Synthetic 최초18+초기화8개(합122개)의 범위를 포함하고, 신규 M2/K2/실행 계약35개와 Original50 감사8개를 더했다. 전체 저장소 discovery라고 표현하지 않는다.
- 네트워크 socket 차단, 실제 Core/Source/Scheduler/Exporter를 이용한 Fake7개 경로: 기존/C0/C1/C2/C3/C4/rollback 모두 completed. 각2 Fake POST이며 실제 LLM Generation0이다.
- default/C0/rollback의 system/user/schema/model/options 및 순서 EXACT_MATCH. M2 이후 원래 PatientMemoryStore 상태와 selection도 변경되지 않음을 검사했다.
- 저장628개 요청 fingerprint PASS;selective321개 기록 ID/score/order/time/context 전부 EXACT_MATCH. LLM 재실행은 아니다.
- Source 경계·미수신/미래 Test·Ask 실제 이력·Gold 거부·동일 Source 중복·BM25 재현·char budget·단위별 과거 시계열·Knowledge/Evidence 분리·잘못된/만료된 승인·새 Plan overwrite 거부·B0~B4 기본 경로 비간섭을 검증했다.
- Python3.9 AST와 PowerShell parser PASS. 새 두 Plan의 static pins PASS. Source/Artifact 보호 결과는 각 namespace preservation_verification.json을 참조한다.

실제 Ollama가 새 payload/출력 계약을 준수하는지와 진단 성능 개선은 NOT_CHECKED/NOT_RUN이다. Source/임상 Reference의 독립 검토를 테스트 PASS로 승격하지 않는다.
''',
    'USER_EXECUTION.md':'''# 사용자가 직접 수행할 실제 실행

프로젝트 루트, 기존 Python 환경에서 아래 명령을 사용한다. Codex는 실행하지 않았다. 먼저 다른 Ollama 작업이 없는지 확인한다. 서버 재시작/다운로드/모델 변경/Warmup은 포함하지 않는다.

```powershell
# 1. Memory만 변경: 정적 검사(Generation0)
.\scripts\Invoke-LoCDIMemoryRAG.ps1 -PlanId memrag_m2_s1_001
# 2. 실제 scope 확인 후 승인하고 실행: C0→C2, 같은1명
.\scripts\Invoke-LoCDIMemoryRAG.ps1 -PlanId memrag_m2_s1_001 -Execute
# 저장 Trace만 분석. 재호출 없음
python -B scripts/run_memory_rag_experiment.py analyze --plan-id memrag_m2_s1_001

# 3. Knowledge만 변경: 별도 승인·별도 Run, C0→C4
.\scripts\Invoke-LoCDIMemoryRAG.ps1 -PlanId memrag_k2_s1_001
.\scripts\Invoke-LoCDIMemoryRAG.ps1 -PlanId memrag_k2_s1_001 -Execute
python -B scripts/run_memory_rag_experiment.py analyze --plan-id memrag_k2_s1_001
```

각 계획은1명×2조건=2 Episode, Episode당12/총24 Generation, 전체wall900초, Context32768·출력1024·요청120초·temperature0·seed42·retry/repair/warmup0, Python4GiB/출력1GiB다. 두 계획을 모두 실행하면 최대48 Generation이다. 개별 요청 Timeout 합은48분이나 전역wall15분이 우선하며 호출/쓰기 중 일시 자원 초과 가능성이 있다.

실제 입력 내용에 따라 기존 byte/context guard가 절단 없이 차단할 수 있다. Source/Model/Context/코드 pin/예산 오류 또는 첫 미완료 Episode에서 전체 해당 paired 실행을 중단한다. 기존 Run/승인 파일이 존재하면 덮어쓰기/자동 Resume/재시도하지 않는다. 새 Plan이 필요하다.

입력할 값:

- APPROVE_MEMORY_RAG: 화면에 표시된 정확한 조건·예산을 확인한 경우에만 동일 문자열 입력.
- Actual private operator reference: 본인의 실제 운영자 참조명. 교수님 서명이나 임상 reviewer인 것처럼 만들지 않는다.
- acceptance statement: 해당 Plan ID, Source-label 개발 Proxy, Recorded-Time Proxy, 임상 정확도 미평가 및 호출 상한을 본인이 수용한다는 실제 진술. 빈 값은 거부한다.
- NO_OTHER_OLLAMA_WORK: 실제 다른 작업이 없을 때만 입력.
- 기록시각/24시간 만료는 실제 승인 시점에 생성된다. clinical_release_approved=false와 임상 Gate는 그대로다.

공개 plan/summary는 `outputs/research_evaluation/MEMORY_RAG_LEGACY_RECOVERY_20261121_v1/<plan>/`, Private 승인/Source/요청·응답·episode·memory_rag_audit는 `outputs/private/research_evaluation/MEMORY_RAG_LEGACY_RECOVERY_20261121_v1/<plan>/run_<plan>/`이다. 실제 환자 원문·Reference를 docs로 복사하지 않는다.

1명 결과를 먼저 확인한 이후에만, 필요한 경우 새 별도10명 계획을 준비한다(준비 자체는 비생성).

```powershell
python -B scripts/run_memory_rag_experiment.py prepare --plan-id memrag_m2_p10_001 --study memory --patients 10
python -B scripts/run_memory_rag_experiment.py prepare --plan-id memrag_k2_p10_001 --study rag --patients 10
# 각20 Episode/최대240. 자동으로 실행되지 않으며, 각 신규 승인이 필요하다.
```
''',
    'NEXT_STEPS.md':'''# 남은 작업과 정확한 한계

지금 가능한 다음 작업은 위의 독립1명 paired 기술 실행을 사용자가 선택·승인하여 수행하는 것이다. 결과를 보고 실패/미실행을 포함해 분석하며, 효과가 없더라도 결과를 숨기거나 Prompt/검색어를 평가 정답에 맞춰 반복 튜닝하지 않는다.

구현 완료: 독립 flags, source-verified full-pool M2, 현재 공개 정보 K2, Legacy/rollback 지문, 실제 Core Fake, 신규 bounded wrapper/계획, 과거 결과 비교.

미실행/미완료: 실제 C0~C5 진단 효과, 장기 실제 turn의 Context 여유, 10명 신규 비교, C5 상호작용, Native 선택형 Knowledge Action, 임상 relevance/독립 Reference 검토, authored Synthetic Utility의 별도 B3/B4 admission. 이 작업에서 이들을 완료로 승격하지 않았다.

M2의 마지막45기록 중통증33개라는 오프라인 투영은 통증 편중을 완전히 해결하지 못했음을 보여준다. 임상 유용성이나 Source Exact 상승을 보장하지 않는다. Original50의 현재 증상/Reference 시점 한계는 별도17번 감사로 판단하며 두 자료를 하나의 성능 실험으로 합치지 않는다.
'''}
    for name,text in documents.items():(DOC/name).write_text(text,encoding='utf-8')
    # Store public plan digests without any patient references.
    atomic_json(PUB/'completion_summary.json',{'status':'IMPLEMENTED_OFFLINE_VERIFIED_LIVE_NOT_RUN','tests':tests,'fake_profiles':len(fake['profiles']),
        'saved_request_hashes_verified':replay['counts']['saved_requests'],'selective_context_exact':replay['counts']['selective_context_exact'],
        'plans':{n:{'fingerprint':p['fingerprint'],'arms':p['arms'],'generation_cap':p['generation_total']} for n,p in plans.items()},
        'utility_sha256_unchanged':uh,'actual_LLM_generation':0,'fitting':0,'clinical_accuracy':'NOT_EVALUATED'})
    print(json.dumps({'docs':str(DOC.relative_to(ROOT)),'files':len(documents)+1,'plans':list(plans),'generation':0,'fitting':0}))
if __name__=='__main__':main()
