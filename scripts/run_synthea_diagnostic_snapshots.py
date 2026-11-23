"""Phase B/C: pinned Synthea20, fail-closed expanded sources, no resume.

prepare/report are offline. execute requires --execute --allow-llm and an
explicit operator statement. Neither existing experiment data nor Core change.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_research_gaps import read, rows, sha, write, check
from scripts.run_final_3b_evaluation import csv_file, metadata, MODEL3, DIGEST3
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest, validate_setup
from app.locdi.routing_collection_preparation import CATALOG
from app.locdi.routing_data_reuse import BANK, SOURCE, EXPERIMENTS
from app.locdi.synthea_diagnosis_audit import normalize, matches
from app.locdi.synthea_diagnostic_snapshot import prepare_snapshot, safe_observation, shared_catalog, source_check
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient, input_guard
from app.locdi.current_ablation_runner import GenerationBudget, _BUDGET, _process_private_bytes
from app.locdi.ollama_adapter import ModelIdentityError
from app.core.longitudinal_memory import timestamp

NS = 'FINAL_SYNTHEA_DIAGNOSIS_RECOVERY_BC_v1'
PUB = ROOT/'outputs/research_evaluation'/NS
PR = ROOT/'outputs/private/research_evaluation'/NS
DOC = ROOT/'docs'/'연구 평가'/NS
OLD = ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY'
OLDPUB = ROOT/'outputs/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY'
RECOVERY = ROOT/'outputs/private/research_evaluation/final_exp_recovery_v1'
DATA = ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
SOURCE_TASK = Path.home()/'.codex/attachments/6022112f-09ca-4840-87e1-57823246b716/Pasted text.txt'
RUN = 'synthea_snapshot_001'
CODE = ['app/locdi/synthea_diagnostic_snapshot.py', 'app/locdi/current_ablation_llm_backend.py',
        'scripts/run_synthea_diagnostic_snapshots.py', 'tests/locdi/test_synthea_diagnostic_snapshot.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def md(name, content):
    for base in (PUB, DOC):
        path = base/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            f.write(content.rstrip()+'\n')


def frozen_record(captured):
    return {'model': captured['model'], 'system': captured['system'],
            'payload': captured['payload'], 'schema': captured['schema']}


def packet(ordinal):
    return read(OLD/('Episode%03d.json' % (2*ordinal-1)))


def load_protocol():
    final = PUB/'snapshot_protocol_final.json'
    return read(final if final.exists() else PUB/'snapshot_protocol.json')


def build_condition(d, condition, registry, catalog, cutoff):
    if condition not in ('D0', 'D1', 'D3'):
        raise ValueError('EXPANDED_SOURCE_RELEASE_NOT_REVIEWED')
    obs = safe_observation(d['initial_observation'] if condition == 'D0' else d['terminal_observation'])
    if condition != 'D0' and obs['elapsed_minutes'] > cutoff:
        raise ValueError('OBSERVATION_AFTER_FIXED_CUTOFF')
    # D0/D1 preserve the existing observation representation for comparison.
    # The separate measurement_view opt-in is tested but not introduced as a
    # third independent variable in the evidence/catalog factorial comparison.
    return prepare_snapshot(obs, registry=registry, catalog=catalog if condition == 'D3' else None)


def prepare():
    if PUB.exists() or PR.exists() or DOC.exists():
        raise ValueError('FRESH_BC_NAMESPACE_REQUIRED')
    pins = read(OLD/'preservation_before.json')
    for base in (OLD, OLDPUB, ROOT/'docs/연구 평가/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY'):
        pins.update({str(p): sha(p) for p in base.rglob('*') if p.is_file()})
    # All previous Core and formula hashes remain protected. The one explicit
    # opt-in backend interface extension is recorded separately, never used to
    # update a historical completed artifact's code fingerprint.
    if check(pins)['status'] != 'PASS':
        raise ValueError('PREVIOUS_ASSET_CHANGED_BEFORE_PREPARATION')
    write(PR/'preservation_before.json', pins)
    config = read(RECOVERY/'llm_config.json')
    ready = validate_setup(config)
    if config['model'] != MODEL3 or config['model_identifier'] != DIGEST3 or ready['static_readiness'] != 'READY':
        raise ValueError('PINNED_3B_SETUP_REQUIRED')
    catalog = shared_catalog(read(CATALOG), expected_sha=sha(CATALOG), path=CATALOG)
    write(PR/'llm_config.json', config)
    inventory = []; summaries = []; reference = []; requests = []; source_hashes = {}
    for ordinal in range(1, 21):
        d = packet(ordinal)
        other = read(OLD/('Episode%03d.json' % (2*ordinal)))
        if (d['audit']['condition'] != 'B0' or other['audit']['condition'] != 'B2'
                or d['private_identity'] != other['private_identity']):
            raise ValueError('FROZEN20_PAIR_MISMATCH')
        cutoff = min(d['audit']['terminal_decision_minute'], other['audit']['terminal_decision_minute'])
        index_ms = int(timestamp(d['private_identity']['started_at']).timestamp()*1000)
        seen = set(); count = Counter(); codes = set()
        for candidate in d['source_objective_candidates']:
            e = candidate['source_event']; key = e['event_id']
            if key in seen:
                continue
            seen.add(key)
            state = source_check(e, patient=d['private_identity']['patient_id'],
                visit=d['private_identity']['visit_id'], index_ms=index_ms, cutoff=cutoff)
            projected = any(key in x['matched_source_ids'] for x in d['projected_source_links'])
            label_risk = any(normalize(g) in normalize(e['target']['description'])
                             for g in d['Source_reference_diagnoses'])
            state = 'DIAGNOSIS_LABEL_LITERAL_RISK' if label_risk else 'EXISTING_REPLAY_MENU' if projected else state
            count[state] += 1
            additional = candidate['typed_numeric_objective'] and candidate['disclosure_status'] == 'PROXY_TIME_ELIGIBLE_RELEASE_NOT_REVIEWED'
            if additional:
                codes.add(str(e['target']['code']))
            inventory.append({'patient_ordinal': ordinal, 'source_event_id': key,
                'source_fingerprint': digest(e), 'source_event': e, 'proxy_available_minute': candidate['proxy_available_minute'],
                'cutoff_minute': cutoff, 'existing_menu': projected, 'additional378_candidate': additional,
                'release_state': state, 'diagnosis_literal_check': 'LITERAL_RISK' if label_risk else 'NO_EXACT_LABEL_LITERAL_NOT_CLINICAL_REVIEW',
                'diagnosis_leakage_reviewed': None, 'grant': None})
        summaries.append({'patient_ordinal': ordinal, 'existing_test_targets': d['audit']['projected_tests'],
            'unique_additional_numeric_codes': len(codes), 'states': dict(count)})
        reference.append({'patient_ordinal': ordinal, 'references': d['reference_sources_and_time'],
            'index_started_at': d['private_identity']['started_at'], 'clinical_reference_status': 'NOT_REVIEWED'})
        for condition in ('D0', 'D1', 'D2', 'D3', 'D4'):
            if condition in ('D2', 'D4'):
                requests.append({'patient_ordinal': ordinal, 'condition': condition, 'status': 'NOT_RUN',
                    'blocker': 'ADDITIONAL_SOURCE_RELEASE_NOT_REVIEWED', 'cutoff_minute': cutoff})
            else:
                capture = build_condition(d, condition, ready['prepared']['registry'], catalog, cutoff)
                payload = frozen_record(capture)
                guard = input_guard(payload['system'], json.dumps(payload['payload'], ensure_ascii=False), payload['schema'])
                requests.append({'patient_ordinal': ordinal, 'condition': condition, 'status': 'PREPARED',
                    'cutoff_minute': cutoff, 'contract': payload, 'fingerprint': digest(payload), 'guard': guard})
        for n in (2*ordinal-1, 2*ordinal):
            source_hashes[str(OLD/('Episode%03d.json'%n))] = sha(OLD/('Episode%03d.json'%n))
    write(PR/'observation_inventory.json', inventory)
    write(PR/'reference_timing.json', reference)
    write(PR/'snapshot_requests.json', requests)
    write(PR/'source_hashes.json', source_hashes)
    write(PR/'release_grants.json', {'grants': [], 'clinical_review': 'NOT_REVIEWED',
        'note': 'No fabricated approvals; old proxy acceptance covers only previously projected menu.'})
    write(PUB/'observation_inventory_summary.json', {'patients': summaries,
        'condition_duplicated_additional_occurrences': 756,
        'unique_additional_source_rows': sum(x['additional378_candidate'] for x in inventory),
        'release_state_counts': dict(Counter(x['release_state'] for x in inventory)),
        'additional_release_approved': 0, 'all_20_retained': True})
    # Explicitly compare evidence equality before model output is observed.
    paired = []
    for ordinal in range(1, 21):
        a = next(x for x in requests if x['patient_ordinal']==ordinal and x['condition']=='D1')
        b = next(x for x in requests if x['patient_ordinal']==ordinal and x['condition']=='D3')
        pa, pb = a['contract']['payload'], b['contract']['payload']
        if any(pa[k] != pb[k] for k in ('observation','evidence','evidence_records','supported_diagnosis_eligible_indices')):
            raise ValueError('PAIRED_EVIDENCE_MISMATCH')
        paired.append({'patient_ordinal': ordinal, 'D1_D3_observation_and_evidence_identical': True,
            'D2_D4': 'NOT_RUN_RELEASE_UNREVIEWED'})
    write(PUB/'paired_input_verification.json', paired)
    investigate_other30()
    protocol = {'version': 'synthea_diagnostic_snapshot.v1', 'namespace': NS, 'run_id': RUN,
        'purpose': 'STATIC_DIAGNOSTIC_DEVELOPMENT_NOT_B0_B2_WORKFLOW', 'source': 'UNCHANGED_SYNTHEA20',
        'registered_patients': 20, 'conditions': ['D0','D1','D2','D3','D4'],
        'executable_conditions': ['D0','D1','D3'], 'blocked_conditions': ['D2','D4'],
        'order': 'patient input order1..20; D0,D1,D2(NOT_RUN),D3,D4(NOT_RUN)',
        'max_generations': 100, 'actual_prepared_request_slots': 60, 'per_patient_condition': 1,
        'model': MODEL3, 'model_identifier': DIGEST3, 'context': 32768, 'temperature': 0, 'seed': 42,
        'max_output': 1024, 'timeout': 120, 'retry': 0, 'repair': 0, 'warmup': 0,
        'memory': 'OFF', 'routing': 'ER_ONLY_NO_SELECTION', 'VoI': False,
        'wall_seconds': 3600, 'process_bytes': 4*1024**3, 'output_bytes': 1024**3,
        'source_approval': 'ONLY_SAVED_DISCLOSED_OBSERVATIONS; expanded378 excluded',
        'snapshot_scope': 'max_decisions0 closes further actions; unchanged evidence eligibility allows unresolved',
        'representation': 'D0 existing initial; D1/D3 existing terminal facts minus prior model assessment and memory; no measurement_view change',
        'catalog': {'path': str(CATALOG), 'SHA256': sha(CATALOG), 'count': 15, 'ordered_fingerprint': digest(catalog),
                    'development_exposed_source_labels': True, 'patient_specific_filter': False},
        'research_evaluation_ready': False, 'clinical_accuracy': 'NOT_EVALUATED', 'concept_match': 'NOT_EVALUATED',
        'stop': 'Source/Identity/global cap => all stop; local response error recorded once, no repair',
        'resource_guard': 'before/after each call, including generation reservation; one call/write may transiently overshoot',
        'no_resume': True, 'no_server_restart_download_unload': True,
        'input_hashes': {str(PR/n): sha(PR/n) for n in ('snapshot_requests.json','source_hashes.json','release_grants.json','llm_config.json')},
        'code_hashes': {n: sha(ROOT/n) for n in CODE},
        'config_pins': {str(p): sha(p) for p in (DATA, CATALOG, SOURCE_TASK)},
        'prepared_at': now()}
    for f in [ROOT/config['registry_path'], ROOT/'prompts/er_department_system.txt', ROOT/'prompts/patient_system.txt']:
        protocol['config_pins'][str(f)] = sha(f)
    protocol['fingerprint'] = digest(protocol)
    write(PUB/'snapshot_protocol.json', protocol)
    print(json.dumps({'status':'PHASE_B_PREPARED','prepared_requests':60,'D2_D4':'BLOCKED', 'fingerprint':protocol['fingerprint']}))


def investigate_other30():
    cases = load_cases(DATA)
    bank = rows(BANK); sources = rows(SOURCE)
    b = {x['training_patient_id'] for x in bank}; s = {x['patient_id'] for x in sources}
    other = [(i,c) for i,c in enumerate(cases,1) if i not in range(11,31)]
    m5 = ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
    provenance = read(m5).get('provenance', {})
    write(PR/'remaining30_exposure_details.json', {'rows': [{'original_ordinal': i,
        'patient_id': c['patient_id'], 'training_bank_patient_overlap': c['patient_id'] in b,
        'training_sources_patient_overlap': c['patient_id'] in s,
        'wrapper_provenance': c['provenance']} for i,c in other], 'frozen_M5_provenance': provenance})
    write(PUB/'remaining30_exposure_summary.json', {'patients':30,
        'training_bank_patient_overlap':sum(c['patient_id'] in b for _,c in other),
        'training_sources_patient_overlap':sum(c['patient_id'] in s for _,c in other),
        'upstream_study_used_Original50_label_set_and_action_support_selection': True,
        'reason': 'Shared15 catalog and common supported action selection; raw inputs/training source patient IDs inspected.',
        'all_prior_prompt_or_probability_exposure': 'NOT_FULLY_ESTABLISHED; absence of patient overlap is not independence',
        'additional_generations_remaining30':0, 'M5_sha256':sha(m5)})


def execute(statement):
    if not statement.strip():
        raise ValueError('EXPLICIT_OPERATOR_STATEMENT_REQUIRED')
    p = load_protocol()
    body = deepcopy(p); fp = body.pop('fingerprint')
    if digest(body) != fp:
        raise ValueError('PLAN_FINGERPRINT_MISMATCH')
    for pins in (p['input_hashes'], p['config_pins'], p['code_hashes']):
        for name, expected in pins.items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            if sha(path) != expected:
                raise ValueError('PIN_CHANGED')
    protected = read(PR/'preservation_before.json')
    if check(protected)['status'] != 'PASS':
        raise ValueError('PROTECTED_ASSET_CHANGED')
    tests = read(PUB/'test_results.json')
    if tests['failures'] or tests['errors'] or not tests['tests_run']:
        raise ValueError('REGRESSION_NOT_PASS')
    run = PR/RUN
    run.mkdir(parents=True, exist_ok=False)
    write(run/'operator_authorization.json', {'source_type':'USER_EXPLICIT_TASK', 'statement':statement,
        'source_instruction_path':str(SOURCE_TASK), 'source_SHA256':sha(SOURCE_TASK), 'recorded_at':now(),
        'scope': 'PhaseC static diagnostic, max100, existing3B, no expanded-source approval', 'plan':fp,
        'professor_or_clinical_approval_created':False})
    loaded = metadata('ps')
    if any(x.get('name') != MODEL3 for x in loaded.get('models', [])):
        write(run/'preflight_blocked.json', {'error':'OTHER_MODEL_LOADED_NO_INTERFERENCE', 'generation':0})
        return
    tags = [x for x in metadata('tags').get('models', []) if x.get('name') == MODEL3]
    if len(tags)!=1 or 'sha256:'+tags[0]['digest'].removeprefix('sha256:') != DIGEST3:
        write(run/'preflight_blocked.json', {'error':'MODEL_IDENTITY_MISMATCH','generation':0})
        return
    write(run/'preflight.json', {'version':metadata('version'), 'loaded':loaded, 'tags':tags,
        'time':now(), 'context':'CHECK_AFTER_EVERY_REQUEST_NO_WARMUP', 'server_modification':False})
    config = read(PR/'llm_config.json'); ready = validate_setup(config)
    catalog = shared_catalog(read(CATALOG), expected_sha=p['catalog']['SHA256'], path=CATALOG)
    slots = read(PR/'snapshot_requests.json')
    result = []; begin = time.monotonic(); cap = 100
    class GlobalStop(ValueError):
        pass
    def guard():
        if time.monotonic()-begin > p['wall_seconds']:
            raise GlobalStop('GLOBAL_WALL_LIMIT')
        if _process_private_bytes() > p['process_bytes']:
            raise GlobalStop('GLOBAL_MEMORY_LIMIT')
        size = sum(f.stat().st_size for root in (PUB,PR) for f in root.rglob('*') if f.is_file())
        if size > p['output_bytes']:
            raise GlobalStop('GLOBAL_OUTPUT_LIMIT')
        if any(sha(Path(k))!=v for k,v in read(PR/'source_hashes.json').items()):
            raise GlobalStop('SOURCE_CHANGED')
    budget = GenerationBudget(cap,1,guard)
    token = _BUDGET.set(budget); stopped = None
    try:
        for slot in slots:
            ordinal, condition = slot['patient_ordinal'], slot['condition']
            row = {'patient_ordinal':ordinal,'condition':condition, 'status':'not_attempted',
                'generation':0,'supported':None,'source_exact':None,'normalized_match':None,
                'source_department_match':None,'measurement_label_target':None,'citation_valid':None,
                'prompt_tokens':None,'wall_seconds':None,'error_type':None,'verified_concept_match':'NOT_EVALUATED'}
            if slot['status']=='NOT_RUN':
                row['reason'] = slot['blocker']; result.append(row); continue
            if stopped:
                row['reason'] = 'GLOBAL_STOP'; result.append(row); continue
            client = None; start = time.monotonic(); budget.episode_used = 0
            try:
                guard()
                d = packet(ordinal)
                capture = build_condition(d,condition,ready['prepared']['registry'],catalog,slot['cutoff_minute'])
                request = frozen_record(capture)
                if digest(request) != slot['fingerprint']:
                    raise GlobalStop('SNAPSHOT_CONTRACT_CHANGED')
                folder = run/('Patient%03d_%s'%(ordinal,condition))
                client = BudgetedNativeERClient(config,authorized=True,contract_fingerprint=fp,checkpoint_root=folder)
                content = client.chat_json(request['system'],json.dumps(request['payload'],ensure_ascii=False,sort_keys=True),
                    model=request['model'],schema=request['schema'])
                output = json.loads(content)
                validated = capture['validate'](content)  # Original Core closure; no rewritten evidence indices.
                supported = validated['certainty']=='supported'
                label = validated['target'] if supported else None
                match = matches(label,d['Source_reference_diagnoses'])
                literal = any(isinstance(label,str) and (label==x['source_event']['target']['description']
                    or label.startswith(x['source_event']['target']['description']+':')) for x in d['source_objective_candidates'])
                row.update(status='completed', supported=supported,source_exact=match['source_exact'] if supported else False,
                    normalized_match=match['normalized_match'] if supported else False,
                    source_department_match=validated['department'] in d['Source_reference_departments'],
                    measurement_label_target=literal,citation_valid=True,prompt_tokens=client.last_server_usage['prompt_tokens'])
                write(folder/'private_scoring.json', {'original_model_output':output,'validated':validated,
                    'reference':d['Source_reference_diagnoses'],'department_reference':d['Source_reference_departments'],
                    'prediction':label,'metrics':row,'clinical_correctness':None})
                guard()
            except Exception as exc:
                row.update(status='failed',error_type=type(exc).__name__)
                if isinstance(exc,(GlobalStop,ModelIdentityError)):
                    stopped = type(exc).__name__
                if client:
                    write(client.checkpoint_root/'semantic_or_execution_failure.json',
                        {'error_type':type(exc).__name__,'message_private':str(exc),'generation':client.request_count})
            finally:
                row['generation'] = client.request_count if client else 0
                row['wall_seconds'] = time.monotonic()-start
                if client:
                    row['identity_checks'] = client.identity_check_count
                    if client.last_server_usage is not None:
                        row['prompt_tokens'] = client.last_server_usage.get('prompt_tokens')
                    client.close()
                result.append(row)
                write(run/('Result%03d_%s.json'%(ordinal,condition)),row)
                print(json.dumps({'patient':ordinal,'condition':condition,'status':row['status'],
                    'generation':row['generation'],'error':row['error_type']}),flush=True)
    finally:
        _BUDGET.reset(token)
        write(run/'results.json',result)
        write(PUB/'execution_summary.json',{'registered_slots':100,'prepared_slots':60,
            'generations':budget.used,'wall_seconds':time.monotonic()-begin,'global_stop':stopped,
            'retry':0,'repair':0,'clinical_correctness_generated':0})
        write(PUB/'preservation_verification_after_execution.json', check(protected))


def report():
    p = load_protocol()
    inv = read(PUB/'observation_inventory_summary.json')
    md('SYNTHEA_ADDITIONAL_OBSERVATION_INVENTORY.md', '# 추가 관측 전수 목록\n\n'
        '동일 Synthea20/기존 Snapshot만 사용했다. B0/B2 중복 756 occurrence는 고유 원천 관측378행이다. '
        '환자별 고유 code 수와 전체 Source 상태 집계는 observation_inventory_summary.json, 값·단위·시각·환자/방문 연결·Source fingerprint는 Private observation_inventory.json에 있다.\n\n'
        f'상태 집계: `{json.dumps(inv["release_state_counts"])}`. 추가 공개 승인0행이다. '
        '최종 세분류(post_execution_source_guard_addendum.json): 환자보고 통증20행은 Ask 전용이며 객관 Test로 승격 금지, '
        '나머지358행은 공개 검토 미완료다. 위 기존 prepared 집계는 초기 기계적 분류로 보존하며 최종 세분류를 함께 적용한다. '
        'NO_EXACT_LABEL_LITERAL은 임상적 누출 없음의 승인이 아니다. 설명·코드·수치의 임상 정답 단서 검토는 NOT_REVIEWED다. '
        '378행은 시간 Proxy 후보이며 승인된378개 검사가 아니다.\n')
    md('SYNTHEA_EVIDENCE_RELEASE_CONTRACT.md','# Source 공개 계약 v1\n\n'
        '기존4 Test와 Ask는 이전 실행의 공개 Snapshot 및 recorded-time 개발 Proxy 계약을 유지한다. '
        '추가 관측에는 같은환자/방문, 원천행 value/unit/date/code 일치, timestamp upper bound≤고정cutoff, numeric laboratory/vital category, '
        '명시적 Source fingerprint·공개 mode·reviewer/evidence/procedure/time·진단문구 검토·Proxy 수용 grant를 모두 요구한다. '
        '이 조건 중 없는 값은 승인으로 만들지 않는다. 현재 추가 grant0, whitelist는 기존 replay menu만이다. D2/D4 NOT_RUN.\n\n'
        'Sequential source_view는 adapt_case 결과의 copy에만 Test 메뉴를 추가하며 B0/B2·ER-only·VoI OFF에 한정한다. '
        '결과는 available_minute를 가진 환경 이벤트이며 요청 전 observation에 삽입하지 않는다. 기존 encounter_evidence와 Scheduler 공개 경로를 사용해야 한다. '
        'Snapshot은 이미 관측한 사실의 정적 재사용이며 전체 순차 워크플로 실행과 구분한다. Source View 실행은 실제 grant 확보 전 차단한다.\n')
    md('SYNTHEA_CATALOG_BINDING_REPORT.md','# Catalog binding\n\n'
        f'기존 공통15 전체, 원래 순서, SHA256 `{p["catalog"]["SHA256"]}`. 환자별 Gold7로 축소하지 않았다. '
        '`make_er_backend(..., diagnosis_catalog=None)`의 기본값은 free-text이며 명시적 Catalog는 SimulationOptions→기존 Policy payload/target enum으로 전달된다. '
        'D3도 실제 Policy가 만든 동일15 enum을 사용한다. Catalog는 이전 Source study label set에 개발 노출된 후보이며 독립 임상 Reference가 아니다. '
        'Department taxonomy는 새로 주입하지 않는다. Evidence·certainty 검증은 그대로다.\n')
    md('SYNTHEA_PHASE_B_IMPLEMENTATION_REPORT.md','# Phase B 구현\n\n'
        '변경: app/locdi/current_ablation_llm_backend.py::make_er_backend(선택적 Catalog 전달). '
        '신규: app/locdi/synthea_diagnostic_snapshot.py(source_check/source_view/measurement_view/prepare_snapshot), '
        'scripts/run_synthea_diagnostic_snapshots.py, tests/locdi/test_synthea_diagnostic_snapshot.py.\n\n'
        'Source View는 opt-in copy overlay이며 기존 wrapper/upstream/Core/Action 종류/VoI 확률/Memory Top5는 변경하지 않는다. '
        'measurement_view는 측정명과 증상을 분리하고 value/unit/time을 이미 공개된 Ask에서만 추출한다. '
        '값이 없으면 null이며 complaint를 만들지 않는다. 이번 D0/D1/D3 비교에는 기존 관찰 표현을 공통 사용해 별도 독립변수를 추가하지 않았다. '
        'prepare_snapshot은 기존 Policy의 실제 schema와 validate closure를 포착한다. max_decisions0은 정적 최종판단만 허용하는 별도 계약이며 실제 Episode가 아니다. '
        'D0 근거0에서 supported 불가/unresolved 가능, 최종진단을 강제 교정하지 않는다. 이전 모델 assessment/Memory는 제거한다.\n\n'
        'Reference 시점: Private reference_timing.json에 원본 날짜와 차이를 보존했다. 환자 ordinal12,16은 index보다1일 늦다. '
        '같은날18명도 intraday 미확정이다. 전체20명·기존 Source label·Exact 분모를 유지하며 clinical Gold 유효성을 자동 판정하지 않는다. '
        '나머지30명의 노출은 remaining30_exposure_summary.json 및 Private 근거를 참조한다. 추가 생성0이다.\n')
    tests = read(PUB/'test_results_final.json') if (PUB/'test_results_final.json').exists() else read(PUB/'test_results.json')
    md('SYNTHEA_PHASE_B_REGRESSION_REPORT.md','# 회귀 결과\n\n'+json.dumps(tests,ensure_ascii=False,indent=2)+
        '\n\nTEST_ONLY Fixture는 Source 공개 승인이나 임상 Reference를 생성하지 않는다. 관련 회귀 결과는 Private related_regression.log와 related_regression_final.log에 있다. '
        '전체 tests/locdi discover는 장시간 외부 Tokenizer/AgentClinic 비관련 검증 도중 중단했으며 전체 회귀 PASS로 보고하지 않는다. '
        '실행 전175개, 실행 후 Source View에서 환자보고 통증/Boolean을 객관검사로 승격하지 못하게 한 별도 guard 보강 후177개를 통과했다. '
        '실행된4개 코드 파일은 Private execution_code에 원래 SHA로 보존했으며, 보강 전후 코드 버전을 혼합하지 않는다. '
        '보호 결과는 preservation_verification_after_execution.json이며 기존 완료 Manifest의 Hash는 갱신하지 않는다.\n')
    run = PR/RUN
    if (run/'results.json').exists():
        result = read(run/'results.json'); execution = read(PUB/'execution_summary.json')
    else:
        result = [{'patient_ordinal':i,'condition':c,'status':'not_attempted','generation':0,
            'reason':'EXECUTION_NOT_STARTED','source_exact':None,'normalized_match':None,'supported':None}
            for i in range(1,21) for c in p['conditions']]
        execution = {'generations':0,'status':'NOT_STARTED'}
    summary = []
    for condition in p['conditions']:
        rr=[r for r in result if r['condition']==condition]; completed=[r for r in rr if r['status']=='completed']
        summary.append({'condition':condition,'registered':20,'completed':len(completed),
            'failed':sum(r['status']=='failed' for r in rr),'not_attempted':sum(r['status']=='not_attempted' for r in rr),
            'supported':sum(r.get('supported') is True for r in completed),
            'unresolved':sum(r.get('supported') is False for r in completed),
            'source_exact_count':sum(r.get('source_exact') is True for r in completed),
            'normalized_count':sum(r.get('normalized_match') is True for r in completed),
            'department_source_count':sum(r.get('source_department_match') is True for r in completed),
            'measurement_label_targets':sum(r.get('measurement_label_target') is True for r in completed),
            'citations_valid':sum(r.get('citation_valid') is True for r in completed),
            'generation':sum(r['generation'] for r in rr),
            'prompt_tokens':sum(r.get('prompt_tokens') or 0 for r in rr),
            'requests_with_reported_prompt_tokens':sum(r.get('prompt_tokens') is not None for r in rr),
            'wall_seconds':sum(r.get('wall_seconds') or 0 for r in rr),
            'concept_match':'NOT_EVALUATED'})
        summary[-1]['source_exact_registered_percent'] = 100*summary[-1]['source_exact_count']/20 if completed else None
        summary[-1]['source_exact_completed_percent'] = 100*summary[-1]['source_exact_count']/len(completed) if completed else None
    write(PUB/'snapshot_results.json', {'conditions':summary,'execution':execution,'rows':result})
    paired=[]
    for before,after in [('D1','D2'),('D1','D3'),('D2','D4'),('D3','D4')]:
        for ordinal in range(1,21):
            a=next(r for r in result if r['patient_ordinal']==ordinal and r['condition']==before)
            b=next(r for r in result if r['patient_ordinal']==ordinal and r['condition']==after)
            ok=a['status']==b['status']=='completed'
            paired.append({'patient_ordinal':ordinal,'comparison':after+'-'+before,'registered_denominator':20,
                'pair_completed':ok,'source_exact_before':a.get('source_exact'),'source_exact_after':b.get('source_exact'),
                'source_exact_delta':int(b['source_exact'])-int(a['source_exact']) if ok else None,
                'normalized_delta':int(b['normalized_match'])-int(a['normalized_match']) if ok else None,
                'status':'COMPARABLE' if ok else 'NOT_COMPARABLE'})
    csv_file(PUB/'SYNTHEA_DIAGNOSTIC_SNAPSHOT_PAIRED_RESULTS.csv',paired,list(paired[0]))
    table='|조건|등록|완료|실패|미실행|Supported|Unresolved|Source Exact|Normalized|Generation|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in summary:
        table+='|'+ '|'.join('NOT_EVALUATED' if not r['completed'] and k in ('source_exact_count','normalized_count') else str(r[k])
            for k in ('condition','registered','completed','failed','not_attempted','supported','unresolved','source_exact_count','normalized_count','generation'))+'|\n'
    metrics='\n|조건|Source 진료과 일치|측정명 target|유효 인용 응답|Prompt tokens|조건 wall초|\n|---|---:|---:|---:|---:|---:|\n'
    for r in summary:
        metrics+='|'+r['condition']+'|'+('|'.join(str(r[k]) for k in ('department_source_count','measurement_label_targets','citations_valid','prompt_tokens','wall_seconds')) if r['completed'] else 'NOT_EVALUATED|NOT_EVALUATED|NOT_EVALUATED|NOT_EVALUATED|NOT_EVALUATED')+'|\n'
    comparisons=[]
    for name in ('D2-D1','D3-D1','D4-D2','D4-D3'):
        rr=[r for r in paired if r['comparison']==name and r['pair_completed']]
        comparisons.append({'comparison':name,'registered_patients':20,'completed_pairs':len(rr),
            'source_exact_net_count':sum(r['source_exact_delta'] for r in rr) if rr else None,
            'paired_pp_difference':100*sum(r['source_exact_delta'] for r in rr)/len(rr) if rr else None,
            'registered_denominator_pp_completed_contribution':100*sum(r['source_exact_delta'] for r in rr)/20 if rr else None,
            'missing_pairs_not_imputed':20-len(rr)})
    write(PUB/'paired_summary.json',comparisons)
    md('SYNTHEA_DIAGNOSTIC_SNAPSHOT_RESULTS.md','# 정적 진단 Snapshot 결과\n\n'+table+metrics+
        '\nSource Exact/Normalized는 독립 임상 정확도가 아니다. 전체 등록20과 완료 수를 구분한다. '
        'D2/D4는 공개 grant 부재로 NOT_RUN이며 0% 성능으로 해석하지 않는다. 실패·미실행의 점수는 null, '
        '완료한 unresolved는 진단 미제출로 등록기준 Source 일치에 기여하지 않으며 incorrect clinical judgment로 변환하지 않는다. '
        'D1/D3은 같은 Evidence·동일 cutoff·기존 system이며 Catalog 계약만 다르다. D0은 근거0이므로 기존 Gate상 unresolved만 가능하다. '
        'Supported는 모델 certainty와 구조적 Evidence 계약 통과를 의미하며 임상적으로 올바른 진단 승인이 아니다. '
        'D1/D3 완료20쌍에서 Source Exact 차이0pp, Normalized 차이0pp다. D2/D4 비교는 완료쌍0으로 NOT_EVALUATED다. '
        '동일 코드/모델의 저장 Trace 검증은 LLM 재생성의 bitwise 재현성 보장이 아니다. '
        'Prompt tokens/시간/인용·측정명 결과는 snapshot_results.json, paired 분모와 차이는 paired_summary.json을 참조한다. '
        '정적 스냅샷이며 기존40 Episode/B0/B2 결과를 대체하지 않는다. Verified Concept Match NOT_EVALUATED.\n')
    md('SYNTHEA_NEXT_ACTION_REPORT.md','# 다음 판단\n\n'
        'D1과 D3의 Source Exact/Normalized 모두0/20이며 Catalog만 적용한 개선은 관측되지 않았다. '
        '따라서 현재 근거만으로 480회 B0/B2 재평가를 바로 실행하는 것은 권장하지 않는다. '
        '이번 실행으로 평가된 범위는 기존 관측 Snapshot과 공통 Catalog 출력 계약이다. 추가 수치 Evidence의 효과는 미평가다. '
        '임상 검토 없이 Recorded-Time 후보378행을 공개하지 않는다. 필요한 실제 검토 입력은 source_check grant 계약에 명시했다. '
        'Source 진단 시점 불확실성·초기 complaint 부재·개발 노출20명 한계는 유지한다. '
        'Catalog 효과가 관측되더라도 이 패널에서 개발된 출력 제약의 탐색 결과이며 Memory 인과 효과나 임상 성능 개선이 아니다. '
        '480회 B0/B2 재실행은 하지 않았다. 기존 결과를 보존하고 실제 효과/Reference/Source 검토를 확인한 다음 별도 실행 여부를 결정해야 한다.\n\n'
        f'프로토콜 fingerprint `{p["fingerprint"]}`. Generation 실제 `{execution.get("generations",0)}`. '
        '실행 명령은 `python -B scripts/run_synthea_diagnostic_snapshots.py execute --execute --allow-llm --operator-statement "<실제 실행 승인 진술>"`이며 '
        '기존 run 재사용·resume는 거부한다. 서버 변경·새 데이터·학습·확률 재추정0.\n')
    write(PUB/'final_manifest.json',{'status':'PHASE_B_IMPLEMENTED_PHASE_C_RECORDED',
        'protocol_fingerprint':p['fingerprint'],'generation':execution.get('generations',0),
        'reports':{str(f.relative_to(ROOT)):sha(f) for f in DOC.glob('*.md')},
        'code_hashes':{n:sha(ROOT/n) for n in CODE},'executed_code_hashes':p['code_hashes'],
        'execution_code_archive':'Private execution_code_archive.json; exact pre-execution code retained',
        'tests':tests,
        'preservation':check(read(PR/'preservation_before.json'))})
    print(json.dumps({'status':'REPORTS_CREATED','conditions':summary},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','execute','report'])
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-llm',action='store_true')
    parser.add_argument('--operator-statement',default='')
    args=parser.parse_args()
    if args.mode=='execute':
        if not (args.execute and args.allow_llm):
            parser.error('execute requires --execute --allow-llm')
        execute(args.operator_statement)
    elif args.mode=='prepare':
        prepare()
    else:
        report()
