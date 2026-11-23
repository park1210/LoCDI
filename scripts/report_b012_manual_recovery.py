# -*- coding: utf-8 -*-
"""Offline report generation for saved Run002; no experiment execution."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.audit_b012_failure_manual import schema_diagnosis
from scripts.b012_manual_execution import PUB,PR,DOC,RUN,BP,BR


def main():
    out=PUB/'audit_v1';doc=DOC
    if out.exists() or (doc/'FINAL_B012_DECISION.md').exists():raise ValueError('NO_REPORT_OVERWRITE')
    actual=BP/'actual_results_v1';metrics=read(actual/'B012_ACTUAL_METRICS.json')
    s=metrics['summary'];rs=metrics['rows'];oldrun=BR/'synthea_b012_subset_stable_002'
    manifest=read(actual/'FINAL_ACTUAL_MANIFEST.json');protected=check(read(BR/'preservation_before.json'))
    oldhash=check({str(oldrun/k):v for k,v in manifest['Run_hashes'].items()})
    assert protected['status']==oldhash['status']=='PASS'
    failure=oldrun/'requests/Episode0018'
    b=read(failure/'Request0005_before.json');a=read(failure/'Request0005_after.json')
    root=schema_diagnosis(b,a)
    assert root['schema']=='INVALID' and root['errors']==[{'validator':'uniqueItems','path':['evidence_indices'],
        'schema_path':['properties','evidence_indices','uniqueItems']}]
    value=json.loads(a['response_envelope']['message']['content'])
    from app.locdi.memory_subset_wire import allowed_codes,FIELD
    payload=json.loads(b['request']['messages'][1]['content'])
    assert value[FIELD] in allowed_codes(payload['observation']['longitudinal_memory']['records'])
    last=read(failure/'Request0004_after.json');lb=read(failure/'Request0004_before.json')
    audit={'run_id':s['run_id'],'generation_this_task':0,'saved_generation':s['generation_attempts'],
        'root':root,'failure_episode':18,'condition':'B2','selected_patient_ordinal':6,
        'original20_ordinal':12,'request_local_ordinal':5,'request_global_ordinal':209,
        'previous_global_request':208,'previous_status':last['status'],
        'HTTP_status':a['HTTP_status'],'actual_wall_seconds':a['wall_seconds'],
        'previous_wall_seconds':last['wall_seconds'],'recorded_at_before':b['at'],
        'prompt_tokens':a['response_envelope']['prompt_eval_count'],
        'output_tokens':a['response_envelope']['eval_count'],
        'previous_prompt_tokens':last['response_envelope']['prompt_eval_count'],
        'input_guard':b['guard'],'previous_input_guard':lb['guard'],
        'wire_message_UTF8_bytes':sum(len(m['content'].encode('utf-8')) for m in b['request']['messages']),
        'schema_UTF8_bytes':b['guard']['schema_UTF8_bytes'],
        'memory_subset_code_valid':True,'actual_Context':a['identity_after']['context_length'],
        'model_digest':a['identity_after']['digest'],'schema_replay':'EXACT_FAILURE_REPRODUCED_NO_MODEL',
        'Run_files':oldhash,'protected':protected,'pins_at_analysis':manifest['pins']}
    write(out/'root_cause.json',audit)
    pairs={}
    for first,second in (('B0','B1'),('B1','B2'),('B0','B2')):
        rows=[]
        for patient in range(1,11):
            x=next(r for r in rs if r['patient_ordinal']==patient and r['condition']==first)
            y=next(r for r in rs if r['patient_ordinal']==patient and r['condition']==second)
            if x['status']==y['status']=='COMPLETED':
                rows.append({'patient_ordinal':patient,'diagnosis_difference':int(y['Source_exact'])-int(x['Source_exact']),
                    'normalized_difference':int(y['Normalized'])-int(x['Normalized']),
                    'department_difference':int(y['Source_department'])-int(x['Source_department'])})
        pairs[second+'-'+first]={'completed_pairs':len(rows),'rows':rows,
            'diagnosis_difference_pp':100*sum(r['diagnosis_difference'] for r in rows)/len(rows) if rows else None,
            'department_difference_pp':100*sum(r['department_difference'] for r in rows)/len(rows) if rows else None}
    write(out/'paired_analysis.json',pairs)
    p=read(PUB/'B012_EXECUTION_PROTOCOL.json');tests=read(PUB/'manual_regression.json')
    common=f'''\n\n근거: Run `{s['run_id']}`. 저장 위치 `{oldrun.relative_to(ROOT).as_posix()}`.
Protocol `{read(BP/'B012_EXECUTION_PROTOCOL.json')['fingerprint']}`.
집계 원본 `{(actual/'B012_ACTUAL_METRICS.json').relative_to(ROOT).as_posix()}` 및 같은 폴더의 CSV/Manifest.
Source/Model/임상 Prompt/Registry/Scorer/수식/원래 Evidence Validator는 변경하지 않았다.
이번 작업 Generation 0회. 임상 정확도·Verified Concept Match는 NOT_EVALUATED.
등록 분모는 조건별10명이고 실패·미실행에 임상 오답 라벨을 생성하지 않았다.
'''
    def md(name,text):
        for base in (out,doc):
            base.mkdir(parents=True,exist_ok=True)
            with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+common)
    table='|조건|등록|완료/Export|실패|미실행|진단 Exact=Normalized/등록|진료과/등록|진단/완료|진료과/완료|20% 기준|Generation|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|\n'
    details='|조건|Prompt tokens|Output tokens|HTTP wall 초|Memory 제공 occurrence|Memory 인용 occurrence|추가 Test 주문/수신|추가 Evidence 인용 결정|\n|---|---:|---:|---:|---:|---:|---|---:|\n'
    for c,t in s['conditions'].items():
        st=t['states'];done=st.get('COMPLETED',0);fail=st.get('CONTRACT_FAILED',0);diag=t['Source_exact_registered_success_count'];dept=t['Source_department_registered_success_count']
        table+=f"|{c}|10|{done}|{fail}|{st.get('NOT_ATTEMPTED',0)}|{diag}/10|{dept}/10|{diag}/{done}|{dept}/{done}|{'도달' if t['threshold_2_of_10_reached'] else '미달'}|{t['generation']}|\n"
        rr=[r for r in rs if r['condition']==c]
        sums={k:sum(r.get(k) or 0 for r in rr) for k in ('prompt_tokens_reported','output_tokens_reported','HTTP_wall_seconds','memory_record_occurrences','memory_cited_id_occurrences','additional_test_orders','additional_result_receipts','additional_evidence_citation_decisions')}
        details+=f"|{c}|{sums['prompt_tokens_reported']}|{sums['output_tokens_reported']}|{sums['HTTP_wall_seconds']:.3f}|{sums['memory_record_occurrences']}|{sums['memory_cited_id_occurrences']}|{sums['additional_test_orders']}/{sums['additional_result_receipts']}|{sums['additional_evidence_citation_decisions']}|\n"
    md('ACTUAL_209_GENERATION_AUDIT.md',f'''# 209회 실제 저장 결과 감사

등록30, 시작18, Core 완료 및 Export17, 실패1, 미실행12. 세 조건 모두 완료한 환자는5명이다.
조건별 시작 B0=6/B1=6/B2=6, 완료 B0=6/B1=6/B2=5. 실패는 여섯째 선정 환자의 B2(Episode18), 이후 네 환자는 전부 미실행이다.
209 = 17×12 + 실패 Episode의5회. BEFORE/AFTER 각209, HTTP200/응답 수신209, JSON 파싱209,
Schema 및 Raw→Internal/Evidence 검증208, 실패1. Export 파일은17개이며 실패18은 raw_runner_result와 Checkpoint만 보존했다.
전체 wall1457.094초, HTTP wall 합1242.618초, Prompt1040478 / Output26089 tokens, Identity418회.
Runtime 자원 제한 오류는 기록되지 않았다. peak 메모리/서버 내부 지연 및 네트워크 wire 전체는 저장되지 않아 NOT_CHECKED.

{table}

실제 Context32768/Model Digest 및 요청 fingerprint209건 일치. B1 고정 완전 Source 검증72건,
B2 Source/cutoff/Top5 검증65건, Source value/unit 일치470 occurrence, 추가 Test 주문112/수신112,
저장 Trace 재집계 EXACT_MATCH. 관측된 시간 위반0이며 임상적 공개시각 검증을 뜻하지 않는다.
초기 Memory retrieval minute은 NOT_RECORDED이며 추정하지 않았다.
보호1844개·기존 Pin62개 변경0, Run 파일 {oldhash['checked']}개 Hash 유지.
''')
    md('CURRENT_ABLATION_BLOCKED_ROOT_CAUSE.md',f'''# 정확한 중단 원인

Episode18 / 선정환자6(original20 ordinal12) / B2 / 로컬 Request5 / 전역209.
`CurrentAblationBlocked(LLM_EPISODE_TECHNICAL_FAILURE)`는 상위 Wrapper다.
하위 Native 오류는 `LLMResponseError`; Core 실패 상세는 schema_validation / NATIVE_RESPONSE_CONTRACT_FAILED다.
저장 응답을 실제 요청의 JSON Schema로 다시 검사하면 request_test 분기의
`properties.evidence_indices.uniqueItems`에서 실패한다. 값 `[0,1,2,1]`은 네 인용 중 고유3개다.
Memory 코드 `01000`은 허용 enum이며 이 오류와 무관하다. anyOf의 다른 diagnose 분기 오류를 별도 원인으로 세지 않았다.

실패 HTTP200, JSON 정상, 응답 wall{a['wall_seconds']:.3f}초, prompt6535/output134tokens,
conservative bound27603 +1024output +512margin=29139 <32768. 본문 messages UTF8합{audit['wire_message_UTF8_bytes']}bytes,
schema6138bytes. 모델 Digest/Context 정상이다. 이전208번은 VALID, wall{last['wall_seconds']:.3f}초,
prompt6161tokens이며 Evidence 인용이 중복되지 않았다.
발생시각 BEFORE `{b['at']}`, 응답 생성시각 `{a['response_envelope']['created_at']}`.

ReadTimeout/Context·Byte Guard/Identity/Memory enum/Source 공개/Exporter/자원 상한/외부 연결 중단의 근거는 이번 실패에 없다.
파싱된 응답을 거부한 것이므로 수신 실패나 연결 끊김으로 확정하면 안 된다.
상위 예외가 세부 메시지를 가리는 한계는 별도 offline schema 진단기로 보완했다.

관련 코드: `app/locdi/current_ablation_llm_backend.py` generate의 schema validate(254~259),
예외 변환(278~287), make_er_backend 실패 래핑(356).
`app/core/hospital_llm_policy.py` `_action_json_schema` 부근 evidence_indices uniqueItems(662~665).
`scripts/audit_b012_failure_manual.py:schema_diagnosis`로 모델 없이 실제 실패를 재현했다.
수정 필요: 기존 Validator/정책에는 확인된 결함 없음. 모델이 Schema-invalid 출력을 반환한 것이며,
중복 인용을 자동 제거하거나 Schema를 완화하지 않는다. 동일 모델의 재발 가능성은 남는다.
새 코드 수정은 offline 진단·수동 실행 인터페이스뿐이며 임상 Prompt와 Runtime 수식은 그대로다.
''')
    md('B012_PARTIAL_ACCURACY_RESULTS.md','# 실제 부분 실행 Source 일치\n\n'+table+'\n'+details+'''

17개 완료 모두 supported, unresolved0. 실패/미실행의 supported 여부는 NOT_EVALUATED다.
진단 Exact와 Normalized는 각각 같은 건수다. B2의2/5(40%)는 완료된 조건부 분모이고,
등록10명 기준 확인된 성공은2/10이다. 전체10명 평가 완료나 임상 진단 정확도40%로 표현하지 않는다.
B0는 진료과2/10, B2는 진단2/10으로 개발 최적화 중단 기준 도달. B1은 진단0/10·진료과1/10으로 미달.
진료과 Source 일치는 ER 출력 라벨과의 비교이며 Adaptive Routing 성능이 아니다.
Memory tokens만의 서버 계측은 없고 위 Prompt tokens는 전체 요청이다. 실패 요청209의 tokens도 수신 기록대로 포함했다.
''')
    pairtable='|비교|완료쌍|진단 차이|진료과 차이|\n|---|---:|---:|---:|\n'
    for k,t in pairs.items():pairtable+=f"|{k}|{t['completed_pairs']}/10|{t['diagnosis_difference_pp']:.2f}pp|{t['department_difference_pp']:.2f}pp|\n"
    md('B012_MEMORY_PAIRED_ANALYSIS.md','# 같은 환자 완료쌍 비교\n\n'+pairtable+'''

분자는 후자 일치−전자 일치, 분모는 같은 환자의 두 조건이 실제 완료한 쌍만이다.
동일5명에서 B2−B0 진단은+40pp이지만 미실행·실패가 남은 개발 노출 소표본의 Source 일치이며 일반화나 임상 인과효과가 아니다.
환자별 차이는 Public audit_v1/paired_analysis.json에 가명 ordinal로만 기록했다.

B1은 Full History가 아닌 Context-Budgeted History. 완료6명의 후보436/492/509/409/651/12058개에서
사전 고정3/3/3/3/3/1개를 선택했다. Source 값·단위·시각을 부분 절단하지 않았고72요청에서 같은 ID집합을 유지했다.
최신 적격 완전기록/고정 예산 정책이며 B2 관련성 Top5와 정보량이 다르다.
실제 bytes/고유기록수/제공·인용 occurrence는 기존 actual_results_v1/B012_EPISODE_RESULTS.csv와
B1_HISTORY_SELECTION_VALIDATION.md에 있다. 시간/ID 상세는 Private history_selection_plans.json에 유지했다.
Memory cutoff/이전방문/환자 Source 검증 통과, 실제 비어 있지 않은 Memory wire 검증136건이다.
B2 65회 제공은 매회5개, 총325 occurrence이며 실패 Episode의 앞선4개를 포함한 유효 응답64개의 subset enum은 최대4개/미제공·superseded 차단/
기존 Core Schema와 Evidence Validator 경로를 유지했다. 실패 응답의 Memory enum도 유효하지만 Evidence 중복 때문에 전체 응답을 채택하지 않았다.
''')
    md('B012_FAILURE_RECOVERY_OPTIONS.md','''# 후속 실행 선택지

|선택|비용·장점|계약 및 한계|판정|
|---|---|---|---|
|A 기존 결과로 종료|추가0회, 209회 보존, B0/B2 기준 도달|부분실행17/30·완료쌍5/6을 공개, B1 미달|추천|
|B 새 전체30 Episode|최대360회, 독립적인 같은10명 전체 패널|새 승인·미사용 Run·동일 오류 재발 가능, 성공 보장 없음|선택 시 수동 CLI 준비|
|C 미실행 환자만 별도 Run|환자7~10의12 Episode 최대144회|Episode18 실패는 그대로 남음; 기존 Prefix와 조합 시 순서/기간/중단에 따른 선택편향, 독립 paired 분모 명시 필요|미구현·새 독립 계약 필요|

추천 A: 연구 종료 목표와 추가 최적화 중단 기준을 고려해 현재 결과를 고정한다.
B0의 진료과 및 B2의 진단2/10은 확인됐지만 B1이나 전체평가 완료를 뜻하지 않는다.
C는 비용이 작아도 원래 no_resume을 자동으로 해제하는 방법이 아니다.
실패18을 제외하고 미실행12개만 채워 완전한30개 연구로 위장할 수 없다.
B의 새 Run은 기존209회와 절대 병합하지 않으며 Prompt/모델/수식 변경 없이 독립 보고한다.
''')
    md('B012_NEXT_EXECUTION_APPROVAL.md',f'''# 선택 B의 신규 수동 실행 승인 요청 — 현재 미승인

미사용 Run `{RUN}`. Protocol `{p['fingerprint']}`.
동일 선정 ordinal [1,2,5,7,10,12,13,14,15,19], 환자별B0→B1_CONTEXT_BUDGETED_HISTORY→B2,
총30 Episode·각12/최대360 Generation. Qwen2.5:3b/Digest·기존 임상 Prompt·공통15 Catalog·Scorer,
기존 Source345행/선정군175행·Memory B0없음/B1사전고정1~3/B2Top5·Routing/VoI OFF.
Context32768/output1024/temperature0/seed42/timeout120초/Retry·Repair·Warmup0,
wall45분/Python4GiB/output1GiB/no_resume. 첫 기술/Source/Identity/Context/전역 오류에 전체중단.
매 응답 실제 Context/Digest 검사, 모델 미로딩 시 첫 고정 Task만 로딩 요청으로 사용한다.
서버 변경·다운로드·다른 환자·B3/B4·fitting·실패 인용 수정은 없다.
Recorded-Time/합성5분/ordinal12 다음날 Reference 한계와 호출 전후 자원검사의 일시 초과 가능성을 수용하는 개발 계약이다.
고정된 Byte Guard는 후속 실제 입력을 차단할 수 있다. 이번 Schema 오류도 재발할 수 있다.

기존 승인 재사용 없음. Private authorization_template.json의 승인란은 null/false로 남겼다.
운영자가 실제 실행 의사를 확인하고 명령 가이드의 입력 절차로 새 approval.json을 작성해야 한다.
기록부터 최대24시간의 운영자 실행 승인으로만 다루며 교수님 서명·임상 검토·v1 grant를 생성하지 않는다.
Codex는 이 실행을 수행하지 않았다.
''')
    commands=r'''# PowerShell 명령

아래 1·3·4는 조회/집계다. 2는 선택 B를 실제로 원할 때만 운영자가 직접 수행한다.
새 Run은 아직 승인되지 않았고 Codex는 실행하지 않는다. 기존002에 resume하지 않는다.

## 1. 실행 전 점검

```powershell
Set-Location -LiteralPath 'C:\Users\wondt\2026\001_Lecture\007_thesis\002_LoCDI-Framework'
$py = (Get-Command python -ErrorAction Stop).Source
& $py --version
& $py -B -c "import sys,httpx,jsonschema; print(sys.executable); print(sys.version)"
# 검증된 환경은 Python 3.9.7이다. venv 사용 시 해당 환경에서 의존성을 먼저 확인한다. 설치는 자동 수행하지 않는다.
Get-Location
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId,CommandLine
Invoke-RestMethod 'http://127.0.0.1:11434/api/version'
(Invoke-RestMethod 'http://127.0.0.1:11434/api/tags').models | Where-Object name -EQ 'qwen2.5:3b' | Select-Object name,digest
Invoke-RestMethod 'http://127.0.0.1:11434/api/ps'
$privateRoot = 'outputs/private/research_evaluation/FINAL_B012_FAILURE_RECOVERY_AND_MANUAL_EXECUTION_v1'
$publicRoot = 'outputs/research_evaluation/FINAL_B012_FAILURE_RECOVERY_AND_MANUAL_EXECUTION_v1'
Test-Path -LiteralPath "$privateRoot/synthea_b012_subset_manual_003"
# 위 Test-Path는 False여야 한다.
& $py -B scripts/b012_manual_execution.py check --server
if ($LASTEXITCODE -ne 0) { throw 'Preflight blocked; do not execute' }
```

예상 Digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`。
check는 GET version/tags/ps와 Hash 검사만 하며 모델을 로드하지 않는다.
다른 실험이 있으면 임의로 종료하지 말고 이번 실행을 시작하지 않는다.

## 2. 새 승인 및 직접 실행 — 선택 B에 한함

```powershell
$approvalPath = Join-Path $privateRoot 'operator_approval.json'
if (Test-Path -LiteralPath $approvalPath) { throw 'Approval exists: do not overwrite/reuse automatically' }
$confirmation = Read-Host 'Read B012_NEXT_EXECUTION_APPROVAL.md. To approve this new full Run and confirm no other Ollama work, type APPROVE_NEW_30'
if ($confirmation -cne 'APPROVE_NEW_30') { throw 'Not approved' }
$statement = Read-Host 'Enter your own explicit approval statement for the exact new protocol and proxy limitations'
$operator = Read-Host 'Enter your private operator reference (actual approver)'
if ([string]::IsNullOrWhiteSpace($statement) -or [string]::IsNullOrWhiteSpace($operator)) { throw 'Actual approval required' }
$approval = Get-Content -Raw -Encoding UTF8 -LiteralPath "$privateRoot/authorization_template.json" | ConvertFrom-Json
$approval.user_statement = $statement
$approval.operator_ref = $operator
$approval.operator_approved = $true
$approval.no_other_ollama_work = $true
$approval.accept_source_record_proxy_cutoff = $true
$approval.accept_fixed_five_minute_synthetic_turnaround = $true
$approval.authorize_sequential_source_and_pilot = $true
$approval.accept_reference_timing_limits = $true
$issued = [DateTimeOffset]::UtcNow
$approval.recorded_at = $issued.ToString('yyyy-MM-ddTHH:mm:ss.ffffffzzz')
$approval.expires_at = $issued.AddHours(24).ToString('yyyy-MM-ddTHH:mm:ss.ffffffzzz')
$approvalJson = $approval | ConvertTo-Json -Depth 100
$approvalFullPath = Join-Path (Get-Location).Path $approvalPath
$stream = [System.IO.File]::Open($approvalFullPath,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
try { $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($approvalJson); $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
& $py -B scripts/b012_manual_execution.py check --server --authorization $approvalPath
if ($LASTEXITCODE -ne 0) { throw 'Approval/preflight blocked' }
# 다음 한 줄은 실제 LLM 실험을 시작한다. 사용자가 직접 실행한다.
& $py -B scripts/b012_manual_execution.py execute --execute --allow-llm --authorization $approvalPath
```

## 3. 실행 중 모니터링 — 별도 PowerShell, 읽기 전용

```powershell
Set-Location -LiteralPath 'C:\Users\wondt\2026\001_Lecture\007_thesis\002_LoCDI-Framework'
python -B scripts/b012_manual_execution.py monitor
```

states/완료 수, latest_before(최근 Episode/Request), last_valid_after, 실패 파일, BEFORE/AFTER 수와 조건별 Source 일치 수를 표시한다. checkpoint_age_seconds는 첫 Checkpoint 이후 대략적인 경과이며 프로세스 실측 wall이 아니다. BEFORE 수만으로 HTTP 성공 수를 확정하지 않는다. 최종 정확한 wall은 actual_execution_summary.json에 기록된다. 모니터링은 모델에 연결하지 않는다.

## 4. 종료 후 저장 Trace 집계 — Generation 0

```powershell
python -B scripts/b012_manual_execution.py analyze
```

collection.json/episodes.jsonl/aggregate.json을 기존 scorer로 대조하고 manual_actual_results_v1/summary.json·episodes.csv·request_audit.json·FINAL_MANUAL_RESULTS.md를 새로 저장한다. 기존 출력은 덮어쓰지 않는다. 완료·실패·미실행·20% 기준·paired 차이·token/Generation/wall을 기록한다. 강제 종료로 collection.json이 없으면 추정으로 보충하지 않고 분석을 중단한다. 이전002의 Manifest/CSV는 별도 결과이며 새 Run에 합산하지 않는다.
'''
    md('B012_POWERSHELL_COMMANDS.md',commands)
    md('B012_MANUAL_EXECUTION_GUIDE.md',f'''# 수동 실행 가이드

추천은 추가 실행 없이 부분 결과로 종료하는 A다. B를 사용자가 선택하면
B012_NEXT_EXECUTION_APPROVAL.md와 B012_POWERSHELL_COMMANDS.md 순서로 직접 진행한다.
새 Protocol은 `{p['fingerprint']}`, Run `{RUN}`. preparation은 완료했으며 승인 template은 미기입이다.
prepare/check/monitor/analyze에는 Generation 경로가 없다. execute만 --execute/--allow-llm/--authorization,
동일 Protocol/Run/정책/History/응답 계약·실제 운영자 참조·동의·유효24시간을 확인한 후 기존 engine.execute를 호출한다.
check --server는 GET version/tags/ps만 사용하며 모델 로딩/POST/재시작은 없다.
실제 호출은 이 문서를 읽은 사용자가 PowerShell에서만 실행한다. Codex는 호출하지 않았다.

새 코드: scripts/b012_manual_execution.py (수동CLI), scripts/audit_b012_failure_manual.py (저장응답 진단),
scripts/report_b012_manual_recovery.py (이 보고서), tests/locdi/test_b012_manual_execution.py (TEST_ONLY).
기존 engine/Runtime/Core/임상Prompt/Validator 변경0. 기존254 관련 테스트와 신규12개를 함께 다시 실행해
{tests['tests_run']}개 중 {tests['pass']} PASS / {tests['failures']} FAIL / {tests['errors']} ERROR / {tests['skip']} SKIP.
전체 저장소 회귀로 확대하지 않는다. 처음 tests.locdi 모듈명 호출은 패키지 경로 오류3건으로 실패했고,
tests/locdi 경로를 명시한 실행으로 수정했으며 테스트나 Skip을 삭제·완화하지 않았다.
현재 Python 실험 프로세스는 읽기 전용 프로세스 조회에서 없었다. 실제 실행 직전 사용자가 다시 확인해야 한다.
신규 manual003 Run은 미생성. 기존 Run002의209회와 이전001의7회는 별도 보존했다.
''')
    md('FINAL_B012_DECISION.md','# 최종 판정\n\n'+table+'''

PARTIAL_EXECUTION_AUDITED / GENERATION_THIS_TASK=0.
추천 A: 현재 부분 결과로 종료하고 설정을 고정한다. B0 진료과2/10 및 B2 진단2/10 기준은 도달했지만
B1은 미달이고 30 Episode 전체 완료는 아니다. 추가 Prompt 최적화나 무조건적인 재실행을 권하지 않는다.
기존 Schema 거부는 정상적인 계약 집행이며 중복 Evidence를 자동 제거하지 않는다.
선택 B의 새 전체 Run 실행 경로만 준비했으며 신규 운영자 승인 전 NOT_RUN이다.
C는 새 평가 계약·선택편향·실패 보존 방침 결정이 필요하고 자동 Resume는 없다.

B3: 적격 Routing Utility/독립 Correctness/현재 Agent Binding의 평가 가능 Train Group이 부족하여 BLOCKED.
B4: B3 차단과 확대된 Native 검사 메뉴의 Action/Outcome encoding 및 기존 likelihood 불일치로 BLOCKED.
기존 VoI 평가를 이번 B4 진단 성능으로 대신하지 않는다. B0/B2 20%는 개발 종료 기준이며 이 Gate를 해제하지 않는다.
''')
    # No original identifiers or clinical response text in the new public reports.
    from app.locdi.dataset_loader import load_cases
    ids=[v for case in load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
         for v in (case['patient_id'],case['agent_visible']['index_encounter_id'])]
    for base in (out,doc):
        for f in base.rglob('*.md'):
            text=f.read_text(encoding='utf-8')
            assert not any(v in text for v in ids)
    assert check({str(oldrun/k):v for k,v in manifest['Run_hashes'].items()})['status']=='PASS'
    write(out/'FINAL_AUDIT_MANIFEST.json',{'old_Run_hashes_unchanged':oldhash,'protected':protected,
        'trace_reaggregation':'EXACT_MATCH','generation_this_task':0,'tests':tests,
        'manual_protocol_fingerprint':p['fingerprint'],'manual_execution_status':'NOT_RUN_NEW_APPROVAL_REQUIRED',
        'reports':{f.name:sha(f) for f in doc.glob('*.md')},
        'new_code_SHA256':{str(f.relative_to(ROOT)):sha(f) for f in (
            ROOT/'scripts/b012_manual_execution.py',ROOT/'scripts/audit_b012_failure_manual.py',
            Path(__file__),ROOT/'tests/locdi/test_b012_manual_execution.py')}})
    print(json.dumps({'status':'AUDITED_NO_NEW_GENERATION','docs':str(doc),'root':'DUPLICATE_EVIDENCE_INDICES',
        'completed':17,'failed':1,'not_attempted':12,'generation_this_task':0}))


if __name__=='__main__':main()
