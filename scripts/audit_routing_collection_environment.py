"""C4-14A: GET-only environment audit and plans. No execution/timeout options.

Reuses the frozen C4-13 Collector without changing its code, prompts or gates.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.dataset_freeze import load, sha
from app.locdi.llm_setup import digest
from app.locdi.routing_two_decision_collection import load_protocol, signed, verify_fingerprint, verify_pins
from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter
from app.llm.prompt_loader import specialty_system_prompt
from scripts.audit_routing_reference_feasibility import inventory_get
from scripts.enrich_full_cohort_sources import write


def inventory(client):
    result = {}
    for path in ('/api/version', '/api/tags', '/api/ps'):
        try:
            result[path] = inventory_get(client, path)
        except Exception as exc:
            result[path] = {'error_type': type(exc).__name__}
    return result


def assess_environment(observed, model, pin):
    models = observed.get('/api/tags', {}).get('models', [])
    matching = [m for m in models if m.get('name') == model or m.get('model') == model]
    correct = len(matching) == 1 and matching[0].get('digest', '').removeprefix('sha256:') == pin.removeprefix('sha256:')
    blockers = []
    for path in ('/api/version', '/api/tags', '/api/ps'):
        if 'error_type' in observed.get(path, {}): blockers.append('SERVER_INVENTORY_UNAVAILABLE')
    if not correct: blockers.append('PINNED_MODEL_NOT_INSTALLED_OR_DIGEST_MISMATCH')
    loaded = [m for m in observed.get('/api/ps', {}).get('models', []) if m.get('name') == model or m.get('model') == model]
    return {'model_installed': bool(matching), 'model_digest_matches': correct,
        'server_version': observed.get('/api/version', {}).get('version'), 'target_loaded': bool(loaded),
        'actual_loaded_context': loaded[0].get('context_length') if len(loaded) == 1 else None,
        'actual_context_status': 'OBSERVED_LOADED_MODEL' if len(loaded) == 1 and loaded[0].get('context_length') else 'NOT_OBSERVED_TARGET_UNLOADED_OR_FIELD_MISSING',
        'blocker_codes': sorted(set(blockers))}


def verify_stage(plan, cap, expected_fingerprint):
    verify_fingerprint(plan)
    if expected_fingerprint != plan['fingerprint']: raise ValueError('PLAN_FINGERPRINT_MISMATCH')
    if plan['generation'] != {'temperature': 0, 'seed': 42, 'max_output_tokens': 1024, 'timeout_seconds': 120}:
        raise ValueError('FROZEN_GENERATION_POLICY_CHANGED')
    if cap not in (1, 10) or type(cap) is not int: raise ValueError('ONLY_PREREGISTERED_ONE_OR_TEN_REQUEST_STAGE')


def run(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace): raise ValueError('INVALID_NAMESPACE')
    public = ROOT / 'outputs/research_evaluation' / namespace
    private = ROOT / 'outputs/private/research_evaluation' / namespace
    if public.exists() or private.exists(): raise ValueError('NO_OVERWRITE')
    prior = ROOT / 'outputs/private/research_evaluation/c4_13_v1'
    mp = ROOT / 'outputs/research_evaluation/c4_13_v1/output_manifest.json'
    m = load(mp); verify_fingerprint(m)
    ip = prior / 'private_index.json'; index = load(ip); verify_fingerprint(index)
    if sha(ip) != m['private_index_sha256']: raise ValueError('PARENT_INDEX_MISMATCH')
    pins = load(prior / 'protected_hashes.json'); pins.update(m['files']); pins.update(index['files'])
    for path in (mp, ip): pins[path.relative_to(ROOT).as_posix()] = sha(path)
    sources = load(prior / 'source_hashes.json')
    verify_pins(pins); verify_pins(sources)
    print('C4-13 completed assets preserved', flush=True)
    bundle, plan, bindings = load_protocol('c4_13_v1')
    for cap in (1, 10): verify_stage(plan, cap, plan['fingerprint'])
    import httpx
    with httpx.Client(timeout=10, trust_env=False, follow_redirects=False) as client:
        observed = inventory(client)
    environment = assess_environment(observed, plan['model'], plan['model_identifier'])
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        environment['gpu_csv'] = gpu.stdout.strip(); environment['gpu_exit_code'] = gpu.returncode
        if gpu.returncode: environment['blocker_codes'].append('GPU_INVENTORY_FAILED')
    except Exception as exc:
        environment['gpu_error_type'] = type(exc).__name__; environment['blocker_codes'].append('GPU_INVENTORY_FAILED')
    environment.update(observed_at=datetime.now(timezone.utc).isoformat(), endpoint='http://127.0.0.1:11434',
        model=plan['model'], pinned_digest=plan['model_identifier'], adapter_timeout_seconds=120,
        output_limit=1024, registry_fingerprint=plan['registry_fingerprint'], plan_fingerprint=plan['fingerprint'],
        generation_requests=0, inventory_methods=['GET /api/version', 'GET /api/tags', 'GET /api/ps'])
    public.mkdir(parents=True); private.mkdir(parents=True)
    write(private / 'protected_hashes.json', pins); write(private / 'source_hashes.json', sources)
    write(private / 'server_inventory.json', observed)
    log_root = Path(os.environ['LOCALAPPDATA']) / 'Ollama'
    log_index = []
    current = ''
    for path in sorted(log_root.glob('server*.log')):
        raw = path.read_bytes()
        with (private / path.name).open('xb') as stream: stream.write(raw)
        log_index.append({'filename': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        if path.name == 'server.log': current = raw.decode('utf-8', errors='replace')
    write(private / 'server_log_index.json', log_index)
    defaults = re.findall(r'default_num_ctx=(\d+)', current)
    environment['startup_default_context'] = int(defaults[-1]) if defaults else None
    environment['startup_context_env'] = '0 (automatic)' if 'OLLAMA_CONTEXT_LENGTH:0' in current else 'NOT_CHECKED'
    environment['startup_cuda_recognized'] = 'library=CUDA' in current
    environment['current_log_generation_post_lines'] = sum('POST' in line and any(p in line for p in ('/v1/chat/completions', '/api/chat', '/api/generate')) for line in current.splitlines())
    environment['current_log_post_count_is_not_audit_generation_count'] = True
    package = load(ROOT / 'outputs/research_evaluation/c3_13c_v1/reports/tokenizer_package_manifest.json')
    counter = LocalTokenCounter(ROOT / package['runner_tokenizer_path'], package['runner_tokenizer_tree_sha256'])
    tokens = []
    for ref in ('Candidate001', 'Candidate002'):
        c = next(c for c in bundle['candidates'] if c['case_ref'] == ref)
        for a in plan['agents']:
            system = specialty_system_prompt(a['department'])
            user = json.dumps({'department': a['department'], **c['shared_payload']}, ensure_ascii=False)
            prompt_fp = hashlib.sha256(system.encode('utf-8')).hexdigest()
            assert prompt_fp == c['decision']['generation']['prompt_fingerprints'][a['agent_id']]
            assert digest(c['shared_payload']) == c['shared_input_fingerprint']
            tokens.append({'case_ref': ref, 'agent_id': a['agent_id'], 'input_tokens_hf': counter.count(system, user),
                'system_bytes': len(system.encode('utf-8')), 'user_bytes': len(user.encode('utf-8')),
                'prompt_fingerprint': prompt_fp, 'shared_input_fingerprint': c['shared_input_fingerprint']})
    environment['input_measurements'] = tokens
    environment['tokenizer_tree_sha256'] = package['runner_tokenizer_tree_sha256']
    environment['token_measurement_scope'] = 'PINNED_HF_CHAT_TEMPLATE_OVER_FROZEN_SYSTEM_USER; NOT_A_NEW_SERVER_TOKEN_MEASUREMENT'
    write(public / 'current_environment_audit.json', environment)
    historical = load(ROOT / 'outputs/research_evaluation/c4_12_v1/timeout_diagnostics.json')
    matrix = {'A_MODEL_COLD_LOADING': {'status': 'HISTORICALLY_OBSERVED', 'seconds': 50.6, 'current_duration': None},
        'B_GPU_DISCOVERY_INITIALIZATION': {'status': 'HISTORICAL_WATCHDOG_TIMEOUT_OBSERVED', 'duration': None, 'current_cuda_recognized': environment['startup_cuda_recognized']},
        'C_PROMPT_PROCESSING': {'status': 'HISTORICAL_INPUT_1231_TOKENS_SLOT_STATE_512; DURATION_NOT_MEASURED', 'duration': None},
        'D_TOKEN_GENERATION': {'status': 'NO_RECEIVED_CONTENT_NO_COMPLETION_TOKEN_OR_DURATION_MEASUREMENT', 'duration': None},
        'E_HTTP_ADAPTER_TIMEOUT': {'status': 'HISTORICAL_120S_HTTP500_CANCEL_CLIENT_TIMEOUT', 'checkpoint_seconds': historical['checkpoint_elapsed_seconds'], 'retry': 0},
        'F_JSON_SCHEMA': {'status': 'HISTORICAL_CONTENT_NOT_RECEIVED_SCHEMA_NOT_EVALUATED', 'schema_error_as_cause': 'NOT_ESTABLISHED'},
        'root_cause_fully_resolved': False, 'current_context_is_historical_actual_context': False}
    write(public / 'timeout_cause_matrix.json', matrix)
    config = signed({'version': 'locdi.routing_collection_reliability.c4_14a.v1', 'collector_plan_fingerprint': plan['fingerprint'],
        'selection': ['Candidate001', 'Candidate002'], 'generation': plan['generation'], 'model': plan['model'],
        'model_identifier': plan['model_identifier'], 'selected_policy': 'UNCHANGED_BASELINE_SINGLE_REQUEST_FIRST',
        'changes_to_collector_or_server': [], 'warmup_requests': 0,
        'alternatives_not_selected': [
            {'option': 'Warm-up', 'effect': 'UNVERIFIED', 'confounds': 'Additional generation, cache/warm state and total budget; separate authorization required'},
            {'option': 'Timeout increase', 'baseline_seconds': 120, 'comparison_example_seconds': 240,
             'effect': 'UNVERIFIED_NOT_IMPLEMENTED', 'confounds': 'Changes failure censoring and response completion opportunity; new version/approval required'},
            {'option': 'Output cap reduction', 'baseline_tokens': 1024, 'comparison_example_tokens': 512,
             'effect': 'UNVERIFIED_NOT_IMPLEMENTED', 'confounds': 'Changes truncation, rationale and diagnostic behavior; not purely a latency change'}],
        'setting_change_required_before_phase1': False, 'reason': 'Existing evidence does not establish input overflow or a specific parameter defect; measure one unchanged request before redesign',
        'phase1_generation_cap': 1, 'phase2_generation_cap': 10, 'total_prospective_generation_cap_if_separately_approved': 11})
    write(public / 'proposed_execution_config.json', config)
    # Copy verified, identical protocol into the NEW namespace only. The C4-13
    # wrapper can load this without touching its completed parent namespace.
    for name in ('plan.json', 'bindings.json', 'protocol_manifest.json'):
        with (private / name).open('xb') as stream: stream.write((prior / name).read_bytes())
    phase1 = signed({'version': 'locdi.routing_single_probe.c4_14a.v1', 'namespace': namespace, 'run_id': 'c4_14_single_001',
        'case_ref': 'Candidate001', 'agent_id': plan['agents'][0]['agent_id'], 'max_requests': 1,
        'collector_plan_fingerprint': plan['fingerprint'], 'execution_config_fingerprint': config['fingerprint'],
        'authorization': 'NOT_PROVIDED_C4_14B_ONLY', 'retry': 0, 'repair': 0,
        'technical_pass_criteria': 'One received schema-valid response, no request/identity/integrity error, no second call; unresolved allowed',
        'collector_expected_budget_status': 'COLLECTION_PARTIAL_OR_FAILED / GLOBAL_REQUEST_LIMIT after a successful first response; retain unchanged native status',
        'model_loading_measurement': 'NOT_EXPOSED_BY_EXISTING_CLIENT; do not infer from wall time',
        'post_run_inventory': 'GET /api/ps and preserve server logs before deciding phase2; not an existing synchronous per-request context hook',
        'mixing': 'DO_NOT_MERGE_SINGLE_PROBE_WITH_TEN_RESPONSE_RUN'})
    phase2 = signed({'version': 'locdi.routing_ten_collection.c4_14a.v1', 'namespace': namespace, 'run_id': 'c4_14_ten_001',
        'selected': deepcopy(plan['selected']), 'agents': deepcopy(plan['agents']), 'max_requests': 10,
        'collector_plan_fingerprint': plan['fingerprint'], 'execution_config_fingerprint': config['fingerprint'],
        'prerequisite': 'Inspect actual phase1 response/schema/wall/error/checkpoints/context/logs AND obtain separate explicit phase2 approval',
        'authorization': 'NOT_PROVIDED', 'retry': 0, 'repair': 0, 'fail_stop': True, 'no_resume': True,
        'complete_collection_group': 'Five distinct registered Agents for same decision, all actual schema-valid responses without request errors; supported or unresolved',
        'correctness_scorable_group': 'Five supported independently reviewed outcomes and current valid Reference plus unchanged policy/legacy Gate',
        'baseline_or_single_probe_response_replacement': False,
        'fit_requirements': ['At least2 independently scorable Train Groups', 'Source/Decision scope', 'Reference and response review',
            'Same registry/model/prompt', 'Patient split', 'Existing Correctness Gate and legacy Validator', 'min_samples2', 'Positive Jaccard support', 'Validation Reference'],
        'proxy_is_independent_clinical_gold': False, 'fit_permission': False})
    write(public / 'single_request_pilot_plan.json', phase1); write(public / 'ten_request_collection_plan.json', phase2)
    state = 'C4_14A_ENVIRONMENT_BLOCKED' if environment['blocker_codes'] else 'C4_14A_TIMEOUT_UNRESOLVED'
    readiness = {'status': state, 'static_environment_pass': not environment['blocker_codes'], 'environment_blockers': environment['blocker_codes'],
        'phase1': 'PLAN_READY_PENDING_EXPLICIT_APPROVAL', 'phase2': 'BLOCKED_PENDING_PHASE1_EVIDENCE_AND_SEPARATE_APPROVAL',
        'actual_context': environment['actual_context_status'], 'timeout_cause_resolved': False,
        'source_reference_ready': False, 'utility_fit_ready': False, 'research_evaluation_ready': False}
    write(public / 'collection_readiness.json', readiness)
    verify_pins(pins); verify_pins(sources)
    write(public / 'verification.json', {'status': state, 'stage': 'C4_14A_ONLY', 'actual_generation': 0, 'fitting_calls': 0,
        'clinical_verdicts_generated': 0, 'server_restarts': 0, 'timeout_or_server_settings_changes': 0,
        'protected_files': len(pins), 'source_files': len(sources), 'changed_protected_or_source_files': 0,
        'collector_reimplemented': False, 'existing_protocol_copied_byte_identically_to_new_namespace': True})
    return readiness


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--namespace', default='c4_14_v1')
    args = parser.parse_args()
    print(json.dumps(run(args.namespace), ensure_ascii=True))
