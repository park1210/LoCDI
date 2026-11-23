"""C4-12 read-only facts and prospective plan. Never constructs a model client.

Private source facts are not independent clinical judgments. GET diagnostics are
optional, restricted to the existing loopback server's three inventory endpoints.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
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
from app.locdi.routing_collector import load_preparation, validate_preparation, PRIVATE_ROOT
from app.locdi.routing_data_reuse import read, SYNTHEA, SOURCE, BANK
from app.locdi.routing_utility_reconstruction import source_code_bindings, reference_facts, response_evidence
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from validation.routing_correctness_gate import validate_evaluation_gate
from app.core.routing_utility import observed_tokens, jaccard_similarity
from scripts.enrich_full_cohort_sources import write, rows
from scripts.audit_memory_review_progress import check_hashes

VERSION = 'locdi.routing_reference_feasibility.c4_12.v1'


def instant(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('TIMEZONE_REQUIRED')
    return d


def time_fact(value, decision):
    """Recorded time only: same-day date precision cannot prove release order."""
    if not value:
        return 'MISSING'
    t = instant(decision)
    if len(value) == 10:
        lower = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        upper = lower + timedelta(days=1)
        if lower > t:
            return 'DEFINITELY_AFTER_DECISION'
        if upper <= t:
            return 'RECORDED_BEFORE_DECISION_RELEASE_UNKNOWN'
        return 'SAME_DAY_DATE_ONLY_AMBIGUOUS'
    return 'DEFINITELY_AFTER_DECISION' if instant(value) > t else 'RECORDED_BY_DECISION_RELEASE_UNKNOWN'


def code_fact(codes, reference_codes):
    if len(codes) != 1:
        return {'state': 'REVIEW_REQUIRED_NO_UNIQUE_EXPLICIT_CODE', 'correct': None}
    return {'state': 'SAME_EXPLICIT_CODE' if tuple(codes[0]) in set(map(tuple, reference_codes)) else 'DIFFERENT_EXPLICIT_CODE',
            'correct': None, 'clinical_equivalence': 'NOT_REVIEWED'}


def group_fact(case_ref, split, outcomes, agents):
    ids = [r['agent_id'] for r in outcomes]
    complete = len(ids) == len(agents) and len(set(ids)) == len(ids) and set(ids) == set(agents)
    supported = sum(r['response_status'] == 'SUPPORTED' for r in outcomes)
    unresolved = sum(r['response_status'] == 'UNRESOLVED' for r in outcomes)
    return {'case_ref': case_ref, 'split': split, 'agent_states': {r['agent_id']: r['response_status'] for r in outcomes},
            'collected_complete': complete, 'supported': supported, 'unresolved': unresolved,
            'source_reference_ready': False, 'clinical_review_ready': False, 'proxy_reference_ready': False,
            'complete_training_group_eligible': False,
            'minimum_missing_supported_slots_arithmetic_only': len(agents) - supported,
            'new_full_group_requests_if_recollected': len(agents),
            'blocker_codes': ['REFERENCE_SUITABILITY_AND_TIMING_NOT_REVIEWED', 'INDEPENDENT_CORRECTNESS_MISSING',
                              'PROXY_VISIBILITY_NOT_CLINICALLY_REVIEWED'] +
                             (['UNRESOLVED_EXCLUDED_FROM_DIAGNOSIS_FITTING'] if unresolved else []) +
                             ([] if complete else ['INCOMPLETE_OR_DUPLICATE_AGENT_GROUP'])}


def select_prospective(candidates, source_facts, count=2):
    """Predeclared source-only rule; does not accept or consult model outcomes."""
    facts = {r['case_ref']: r for r in source_facts}
    eligible = [c for c in candidates if c['decision']['split'] == 'train'
                and facts[c['case_ref']]['patient_encounter_link']
                and not facts[c['case_ref']]['definitely_future_selected_reference']]
    eligible.sort(key=lambda c: c['case_ref'])
    selected = eligible[:count]
    if len(selected) != count or len({c['decision']['patient_id'] for c in selected}) != count:
        raise ValueError('TWO_DISTINCT_SOURCE_LINKED_TRAIN_PATIENTS_REQUIRED')
    return selected


def classify_attempt(after):
    if after.get('raw_response') is not None:
        return 'RESPONSE_RECEIVED_SEE_SCHEMA_STATUS'
    if after.get('error_type') == 'LLMTimeoutError':
        return 'CLIENT_TIMEOUT_NO_RESPONSE_CONTENT'
    return 'REQUEST_FAILED_NO_RESPONSE_CONTENT' if after.get('error_type') else 'NOT_CHECKED'


def condition_scope_links(facts, alignment):
    """Expose exact code-linked source rows and temporal scope, never relevance."""
    by_case = {f['case_ref']: f for f in facts}
    links = []
    for response in alignment:
        fact = by_case[response['decision_ref']]
        codes = set(map(tuple, response['extracted_codes']))
        matches = []
        for row in fact['all_patient_condition_rows']:
            r = row['source_record']
            if (r['SYSTEM'], r['CODE']) in codes:
                matches.append({'source_file': row['source_file'], 'source_row': row['source_row'],
                    'source_record_fingerprint': row['source_record_fingerprint'],
                    'same_current_encounter': r['ENCOUNTER'] == fact['visit_id'],
                    'recorded_time_state': time_fact(r['START'], fact['decision_time'])})
        links.append({'case_ref': response['decision_ref'], 'agent_id': response['agent_id'],
            'response_fingerprint': response['response_fingerprint'], 'code_linked_source_rows': matches,
            'decision_relevance': 'NOT_REVIEWED', 'correct': None})
    summary = {'responses': len(links), 'responses_with_same_patient_condition_code': sum(bool(x['code_linked_source_rows']) for x in links),
        'matched_source_rows': sum(len(x['code_linked_source_rows']) for x in links),
        'current_encounter_matches': sum(m['same_current_encounter'] for x in links for m in x['code_linked_source_rows']),
        'matched_recorded_time_states': dict(Counter(m['recorded_time_state'] for x in links for m in x['code_linked_source_rows'])),
        'diagnosis_correctness_created': 0, 'independent_review_required': True}
    return links, summary


def inventory_get(client, path):
    if path not in ('/api/version', '/api/tags', '/api/ps'):
        raise ValueError('NON_GENERATIVE_ENDPOINT_ONLY')
    r = client.get('http://127.0.0.1:11434' + path)
    r.raise_for_status()
    return r.json()


def source_response_checks(bundle, facts, records):
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(bundle['response_schema'])
    for record in records:
        validator.validate(record['response'])
    joins = []
    for f in facts:
        csv_keys = Counter((r['source_record']['ENCOUNTER'], r['source_record']['SYSTEM'], r['source_record']['CODE'], r['source_record']['START']) for r in f['all_patient_condition_rows'])
        native_keys = Counter((r['encounter_id'], c['system'], str(c['code']), r['start'][:10]) for r in f['native_condition_inventory'] for c in r['codes'])
        joins.append({'case_ref': f['case_ref'], 'csv_rows': sum(csv_keys.values()), 'native_code_rows': sum(native_keys.values()),
            'csv_native_multiset_equal': csv_keys == native_keys, 'unmatched_csv': sum((csv_keys - native_keys).values()),
            'unmatched_native': sum((native_keys - csv_keys).values())})
    expanded = source_code_bindings([{'source_condition_rows': f['all_patient_condition_rows']} for f in facts])
    return {'schema_valid': len(records), 'parsed': len(records),
        'response_fingerprints_match': all(digest(r['response']) == r['response_fingerprint'] for r in records),
        'all_correct_null': all(r['correct'] is None for r in records), 'csv_native_join_checks': joins,
        'catalog_names_explicitly_bound': sum(bool(expanded.get(name)) for name in bundle['catalog']),
        'chief_complaint_nonempty': sum(bool(c['decision']['observation']['chief_complaint']) for c in bundle['candidates']),
        'current_symptoms_nonempty': sum(bool(c['decision']['observation']['current_symptoms']) for c in bundle['candidates']),
        'objective_findings_count': sum(len(c['decision']['observation']['objective_findings']) for c in bundle['candidates']),
        'memory_records': sum(len(c['decision']['observation']['longitudinal_memory']['records']) for c in bundle['candidates']),
        'selected_reference_native_by_decision': sum(f['native_selected_after_decision'] == 0 for f in facts),
        'actual_available_time_reviewed': 0}


def protect_parent():
    parent = ROOT / 'outputs/private/research_evaluation/c4_11_v1'
    pins = load(parent / 'protected_hashes.json')
    mp = ROOT / 'outputs/research_evaluation/c4_11_v1/output_manifest.json'
    m = load(mp)
    if digest({k: v for k, v in m.items() if k != 'fingerprint'}) != m['fingerprint']:
        raise ValueError('PARENT_MANIFEST')
    ip = parent / 'private_index.json'
    if sha(ip) != m['private_index_sha256']:
        raise ValueError('PARENT_PRIVATE_INDEX')
    pins.update(m['files']); pins.update(load(ip)['files'])
    for p in (mp, ip):
        pins[p.relative_to(ROOT).as_posix()] = sha(p)
    check_hashes(pins)
    return pins, load(parent / 'source_hashes.json')


def source_audit(bundle, references, source_pins):
    patients = {c['decision']['patient_id'] for c in bundle['candidates']}
    paths = {SYNTHEA / x['source_file'] for ref in references for x in ref['source_condition_rows']}
    conditions = defaultdict(list)
    for path in sorted(paths):
        source_pins[str(path)] = sha(path)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for number, row in enumerate(csv.DictReader(stream), 2):
                if row['PATIENT'] in patients:
                    conditions[row['PATIENT']].append({'source_file': str(path), 'source_row': number,
                                                      'source_record': row, 'source_record_fingerprint': digest(row)})
    index = read(ROOT / 'outputs/private/research_evaluation/c4_05_v1/case_source_index.jsonl')
    native = {}
    for descriptor in index:
        if descriptor['dataset'] != 'TrainingBank' or descriptor['patient_id'] not in patients:
            continue
        path = Path(descriptor['raw_path'])
        if str(path) in native:
            continue
        actual = sha(path)
        # TrainingBank descriptors explicitly carry null; use the already sealed
        # C4-11 source hash in that case, never invent an original declaration.
        expected = descriptor['declared_raw_sha256'] or source_pins.get(str(path))
        if expected is None or actual != expected:
            raise ValueError('NATIVE_SOURCE_PIN_MISMATCH')
        source_pins[str(path)] = actual
        native[str(path)] = read(path)
    native_by_patient = {x['attributes']['id']: x for x in native.values()}
    source_rows = {s['case_id']: s for s in read(SOURCE)}
    refs = {r['case_ref']: r for r in references}
    result = []
    for candidate in bundle['candidates']:
        d = candidate['decision']; reference = refs[candidate['case_ref']]
        decision_time = (instant(candidate['visit_started_at']) + timedelta(minutes=d['minute'])).isoformat()
        all_conditions = conditions[d['patient_id']]
        selected = reference['source_condition_rows']
        linked = all(r['source_record']['PATIENT'] == d['patient_id'] and r['source_record']['ENCOUNTER'] == d['visit_id'] for r in selected)
        for selected_row in selected:
            if not any(r['source_row'] == selected_row['source_row'] and r['source_record'] == selected_row['source_record'] for r in all_conditions):
                raise ValueError('SELECTED_SOURCE_ROW_MISMATCH')
        nx = native_by_patient.get(d['patient_id'])
        if nx is None:
            raise ValueError('NATIVE_PATIENT_MISSING')
        encounters = nx['record']['encounters']
        current = [e for e in encounters if e['uuid'] == d['visit_id']]
        if len(current) != 1:
            raise ValueError('NATIVE_ENCOUNTER_LINK_MISMATCH')
        native_conditions = []
        for encounter in encounters:
            for item in encounter.get('conditions', []):
                native_conditions.append({'encounter_id': encounter['uuid'], 'start_epoch_ms': item['start'],
                    'start': (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=item['start'])).isoformat(),
                    'codes': item.get('codes', []), 'source_fingerprint': digest(item), 'current_encounter': encounter['uuid'] == d['visit_id']})
        selected_native = []
        for sr in selected:
            row = sr['source_record']
            matches = [n for n in native_conditions if n['current_encounter'] and any(str(code.get('code')) == row['CODE'] and code.get('system') == row['SYSTEM'] for code in n['codes'])]
            selected_native.append({'source_row': sr['source_row'], 'native_matches': matches,
                                    'csv_precision_state': time_fact(row['START'], decision_time)})
        ledger = list(candidate['visibility'].values())
        src = source_rows[d['source_ref']]
        timing = Counter(time_fact(r['source_record']['START'], decision_time) for r in all_conditions)
        result.append({'case_ref': candidate['case_ref'], 'split': d['split'], 'patient_id': d['patient_id'],
            'visit_id': d['visit_id'], 'decision_id': d['decision_id'], 'decision_time': decision_time,
            'patient_encounter_link': linked, 'source_fingerprint': candidate['source_fingerprint'],
            'all_patient_condition_rows': all_conditions, 'condition_time_counts': dict(timing),
            'patient_condition_count': len(all_conditions), 'current_encounter_condition_count': sum(r['source_record']['ENCOUNTER'] == d['visit_id'] for r in all_conditions),
            'native_condition_count': len(native_conditions), 'native_selected_conditions': selected_native,
            'selected_condition_rows': selected, 'native_condition_inventory': native_conditions,
            'definitely_future_selected_reference': any(time_fact(r['source_record']['START'], decision_time) == 'DEFINITELY_AFTER_DECISION' for r in selected),
            'native_selected_after_decision': sum(instant(n['start']) > instant(decision_time) for s in selected_native for n in s['native_matches']),
            'native_encounter_start_matches': abs(current[0]['start'] / 1000 - instant(candidate['visit_started_at']).timestamp()) < .001,
            'source_target_condition_count': len(src['all_actual_target_conditions']),
            'visibility_ledger_entries': len(ledger), 'actual_release_recorded': sum(x.get('actual_release_status') == 'RECORDED' for x in ledger),
            'proxy_entries': sum('PROXY' in x.get('basis', '') or x.get('actual_release_status') == 'NOT_RECORDED' for x in ledger),
            'allowed_observation_snapshot': d['observation'], 'visibility_ledger': candidate['visibility'],
            'excluded_future_events': candidate['excluded_selected_future_events'],
            'source_reference_status': 'REFERENCE_TIMING_NOT_REVIEWED', 'unique_gold': None, 'correct': None,
            'clinical_current_relevance': 'NOT_REVIEWED', 'proxy_clinically_accepted': None})
    return result


def timeout_audit(bundle, private, live_get):
    base = ROOT / 'outputs/private/research_evaluation/c4_11_v1/live_pilot/c4_11_collect_001'
    before, after = load(base / 'Request001_before.json'), load(base / 'Request001_after.json')
    log_root = Path(os.environ.get('LOCALAPPDATA', '')) / 'Ollama'
    logs = []
    for p in sorted(log_root.glob('server*.log')):
        # Logs are mutable operational evidence: snapshot bytes and hash, not protected sources.
        raw = p.read_bytes(); content = raw.decode('utf-8', errors='replace')
        if '2026/11/08 - 00:08:19' in content:
            captured = private / 'server_log_snapshot.txt'
            captured.parent.mkdir(parents=True, exist_ok=True)
            with captured.open('xb') as stream:
                stream.write(raw)
            logs.append({'file': str(p), 'sha256': __import__('hashlib').sha256(raw).hexdigest(),
                         'lines': [{'line': i, 'text': line} for i, line in enumerate(content.splitlines(), 1)]})
    write(private / 'timeout_log_evidence.json', logs)
    log_text = '\n'.join(r['text'] for log in logs for r in log['lines'])
    facts = {'classification': classify_attempt(after), 'generation_attempts': after['generation_count_after'],
        'identity_checks': after['identity_checks_after'], 'received_content': after.get('raw_response') is not None,
        'checkpoint_elapsed_seconds': (instant(after['recorded_at']) - instant(before['request_started_at'])).total_seconds(),
        'timeout_seconds': before['generation']['timeout_seconds'], 'output_limit': before['generation']['max_output_tokens'],
        'system_utf8_bytes': len(before['system_prompt'].encode('utf-8')),
        'user_utf8_bytes': len(json.dumps(before['payload'], ensure_ascii=False).encode('utf-8')),
        'system_prompt_fingerprint_matches': __import__('hashlib').sha256(before['system_prompt'].encode('utf-8')).hexdigest() == before['prompt_fingerprint'],
        'payload_fingerprint_matches': digest(before['payload']) == before['full_payload_fingerprint'],
        'identity_matches_pin': after['observed_model_identifier'] == bundle['llm_config']['model_identifier'],
        'request_path': 'LocalOllamaAdapter.chat_json -> _backend -> OllamaClient.chat_json/chat -> OpenAI chat.completions.create',
        'adapter_timeout_propagation': 'httpx.Client timeout=120, SDK with_options timeout=120 max_retries=0',
        'response_schema_fingerprint': digest(bundle['response_schema']),
        'response_format': 'json_schema strict clinical_decision', 'messages': ['system', 'user'], 'stream': False,
        'retry': 0, 'repair': 0, 'raw_wire': 'NOT_CAPTURED',
        'server_log_found': bool(logs), 'log_association': 'MATCHING_LOCAL_TIME_ENDPOINT_AND_SINGLE_TASK_NOT_WIRE_REQUEST_ID',
        'gpu_discovery_timeout_logged': 'GPU discovery watchdog timed out' in log_text,
        'model_load_seconds': 50.60 if 'started in 50.60 seconds' in log_text else None,
        'server_prompt_tokens': 1231 if 'task.n_tokens = 1231' in log_text else None,
        'allocated_context': 4096 if 'n_ctx_seq             = 4096' in log_text else None,
        'offloaded_layers': '29/29' if 'offloaded 29/29 layers to GPU' in log_text else 'NOT_CHECKED',
        'server_http_500_after_120s': '00:08:19 | 500 |     2m0s' in log_text,
        'cancelled_task': 'cancel task, id_task = 0' in log_text,
        'release_n_tokens': 512 if 'stop processing: n_tokens = 512, truncated = 0' in log_text else None,
        'release_n_tokens_interpretation': 'INTERNAL_SLOT_STATE_NOT_COMPLETION_TOKENS',
        'truncated_flag_at_release': 0 if 'stop processing: n_tokens = 512, truncated = 0' in log_text else None,
        'root_cause': 'PARTIALLY_OBSERVED_COLD_LOAD_AND_GPU_DISCOVERY_DELAY; PROMPT_PROCESSING_DELAY_CAUSE_UNDETERMINED',
        'context_plus_output_nominal': 2255 if 'task.n_tokens = 1231' in log_text else None, 'context_overflow_proven': False,
        'generation_this_audit': 0, 'settings_changes': 0, 'current_environment': {}}
    if live_get:
        import httpx
        environment = {'observed_at': datetime.now(timezone.utc).isoformat()}
        with httpx.Client(timeout=10, trust_env=False, follow_redirects=False) as client:
            for path in ('/api/version', '/api/tags', '/api/ps'):
                try:
                    environment[path] = inventory_get(client, path)
                except Exception as exc:
                    environment[path] = {'error_type': type(exc).__name__}
        try:
            environment['gpu'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'], capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception as exc:
            environment['gpu'] = {'error_type': type(exc).__name__}
        write(private / 'current_environment.json', environment)
        entries = environment.get('/api/tags', {}).get('models', [])
        expected = bundle['llm_config']['model_identifier'].removeprefix('sha256:')
        matches = [e for e in entries if e.get('name') == bundle['llm_config']['model']]
        facts['current_environment'] = {'version': environment.get('/api/version'),
            'model_digest_matches': len(matches) == 1 and matches[0].get('digest', '').removeprefix('sha256:') == expected,
            'loaded_model_count': len(environment.get('/api/ps', {}).get('models', [])), 'gpu': environment['gpu'],
            'observed_at': environment['observed_at']}
    old = read(PRIVATE_ROOT / 'phase24b_collect_001/Candidate001_Agent001.json')
    facts['successful_baseline_comparison'] = {'schema_valid': old['response_contract_status'] == 'VALID',
        'elapsed_checkpoint_seconds': (instant(old['recorded_at']) - instant(old['request_started_at'])).total_seconds(),
        'same_model_version': old['model'] == before['model'] and old['version'] == before['version'],
        'same_shared_input': old['shared_input_fingerprint'] == before['shared_input_fingerprint'],
        'same_prompt': old['system_fingerprint'] == before['prompt_fingerprint'],
        'same_full_payload': old['full_payload_fingerprint'] == before['full_payload_fingerprint'],
        'generation': old['generation'], 'historical_server_context': 'NOT_RECORDED',
        'historical_server_version': 'NOT_RECORDED', 'historical_server_latency_breakdown': 'NOT_RECORDED'}
    return facts


def run(namespace='c4_12_v1', live_get=False):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace):
        raise ValueError('NAMESPACE')
    public = ROOT / 'outputs/research_evaluation' / namespace
    private = ROOT / 'outputs/private/research_evaluation' / namespace
    if public.exists() or private.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    pins, source_pins = protect_parent(); check_hashes(source_pins)
    print('Parent assets verified', flush=True)
    bundle = load_preparation('phase24b_candidates_v2')
    validate_preparation(bundle)  # Actual CSV/SQLite replay, whitelist, cutoff and prompt binding.
    references = load(PRIVATE_ROOT / 'phase24b_candidates_v2/reference_candidates.json')
    records = [load(p) for p in sorted((PRIVATE_ROOT / 'phase24b_collect_001').glob('Candidate*.json'))]
    if len(records) != 75 or any(r.get('correct') is not None for r in records):
        raise ValueError('BASELINE_CHANGED_OR_REVIEW_ASSUMPTION_INVALID')
    facts = source_audit(bundle, references, source_pins)
    selected = select_prospective(bundle['candidates'], facts)
    bindings = source_code_bindings(references)
    # Expand explicit code bindings using all actual selected patients' conditions.
    expanded = source_code_bindings([{'source_condition_rows': f['all_patient_condition_rows']} for f in facts])
    candidates = {c['case_ref']: c for c in bundle['candidates']}; refs = {r['case_ref']: r for r in references}
    evidence = [response_evidence(r, candidates[r['case_ref']], reference_facts(candidates[r['case_ref']], refs[r['case_ref']]), expanded) for r in records]
    alignment = []
    for record, ev in zip(records, evidence):
        f = next(x for x in facts if x['case_ref'] == ev['decision_ref'])
        codes = [(r['source_record']['SYSTEM'], r['source_record']['CODE']) for r in f['all_patient_condition_rows']]
        alignment.append({**ev, 'response': record['response'],
            'all_patient_condition_code_comparison': code_fact(ev['extracted_codes'], codes),
            'same_code_does_not_establish_current_relevance': True,
            'synonym_or_multiple_diagnosis_adjudication': 'REVIEW_REQUIRED_NO_OFFICIAL_EQUIVALENCE_ASSERTED'})
    agents = [a for a in bundle['registry']['agents'] if a['role'] == 'specialty']
    groups = [group_fact(c['case_ref'], c['decision']['split'], [e for e in evidence if e['decision_ref'] == c['case_ref']], [a['agent_id'] for a in agents]) for c in bundle['candidates']]
    data, options = actual_gate_inputs(); options['min_samples'] = 2
    gate = validate_evaluation_gate(data, **options)
    print('All 15 sources and 75 responses audited; unchanged Gate executed', flush=True)
    splits = bundle['split_manifest']['patients']; train, val, hold = (set(splits[k]) for k in ('train', 'validation', 'test'))
    voi = load(BANK.parent / 'voi_model.json'); voi_train = set(voi['provenance']['training_patients'])
    chosen_patients = {c['decision']['patient_id'] for c in selected}
    if train & val or train & hold or val & hold or chosen_patients & val:
        raise ValueError('PATIENT_SPLIT_OVERLAP')
    support = []
    for c in bundle['candidates']:
        if c['decision']['split'] == 'validation':
            positive = sum(jaccard_similarity(observed_tokens(c['decision']['observation']), observed_tokens(s['decision']['observation'])) > 0 for s in selected)
            support.append({'case_ref': c['case_ref'], 'potential_positive_similarity_groups': positive, 'actual_eligible_groups': 0})
    plan = {'version': VERSION, 'status': 'PROSPECTIVE_PLAN_NOT_AUTHORIZED_NOT_EXECUTED',
        'selection_rule': 'FIRST_TWO_TRAIN_BY_CASE_REF_WITH_EXACT_PATIENT_ENCOUNTER_LINK_NO_DEFINITELY_FUTURE_SELECTED_REFERENCE; MODEL_OUTCOMES_NOT_USED',
        'selected': [{'case_ref': c['case_ref'], 'shared_input_fingerprint': c['shared_input_fingerprint'],
                      'memory_fingerprint': c['memory_fingerprint'], 'source_fingerprint': c['source_fingerprint'],
                      'prompt_fingerprints': c['decision']['generation']['prompt_fingerprints']} for c in selected],
        'preparation_fingerprint': bundle['fingerprint'], 'registry_fingerprint': digest(bundle['registry']),
        'catalog_fingerprint': digest(bundle['catalog']), 'agents': agents,
        'model': bundle['llm_config']['model'], 'model_identifier': bundle['llm_config']['model_identifier'],
        'model_version': bundle['llm_config']['model_version'],
        'generation': {'temperature': 0, 'seed': 42, 'timeout_seconds': 120, 'max_output_tokens': 1024},
        'global_budget': 10, 'identity_checks_max': 10, 'retry': 0, 'repair': 0,
        'checkpoint': 'Private BEFORE and AFTER for every attempted request; actual received content separately counted',
        'stop': 'Any transport/schema/integrity failure stops entire new run; preserve partial results; no resume',
        'unresolved': 'Preserve as unresolved; no forced supported conversion; not eligible diagnosis bool',
        'mixing': 'One new version, both whole 5-Agent groups; no selected replacement with old outcomes',
        'min_samples': 2, 'potential_validation_support': support,
        'execution_blockers': ['DECISION_REFERENCE_AND_VISIBILITY_REVIEW_REQUIRED', 'TIMEOUT_OPERATIONAL_REVIEW_REQUIRED',
                               'NEW_10_REQUEST_OPERATOR_APPROVAL_REQUIRED', 'TWO_DECISION_COLLECTOR_NOT_IMPLEMENTED'],
        'existing_cli_limitation': 'collect_routing_utility_pilot.py always first Train Decision, max5; cannot select second; do not run twice and call it two groups',
        'clinical_judgments_created': 0, 'utility_fit_allowed': False}
    plan['fingerprint'] = digest(plan)
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    write(public / 'next_collection_plan.json', plan)
    rows(private / 'source_decision_eligibility.jsonl', facts)
    rows(private / 'diagnosis_reference_alignment.jsonl', alignment)
    write(public / 'source_and_response_verification.json', source_response_checks(bundle, facts, records))
    scope_links, scope_summary = condition_scope_links(facts, alignment)
    rows(private / 'response_condition_scope_links.jsonl', scope_links)
    write(public / 'diagnosis_alignment_summary.json', scope_summary)
    write(private / 'existing_gate_result.json', gate)
    write(private / 'explicit_source_code_bindings.json', {'previous': bindings, 'expanded_all_selected_patient_conditions': expanded})
    write(public / 'source_decision_summary.json', {'decisions': 15, 'patient_encounter_links': sum(f['patient_encounter_link'] for f in facts),
        'all_patient_csv_conditions': sum(f['patient_condition_count'] for f in facts),
        'native_conditions': sum(f['native_condition_count'] for f in facts),
        'future_selected_reference_cases': [f['case_ref'] for f in facts if f['definitely_future_selected_reference']],
        'cases': [{k: f[k] for k in ('case_ref', 'split', 'patient_condition_count', 'current_encounter_condition_count',
                    'native_condition_count', 'native_selected_after_decision', 'native_encounter_start_matches',
                    'visibility_ledger_entries', 'actual_release_recorded', 'proxy_entries', 'source_reference_status')} for f in facts],
        'independent_clinical_reviews': 0, 'code_bindings_before': len(bindings), 'code_bindings_after': len(expanded)})
    write(public / 'response_support_matrix.json', {'decisions': groups,
        'specialties': {a['department']: dict(Counter(e['response_status'] for e in evidence if e['agent_id'] == a['agent_id'])) for a in agents},
        'supported_code_comparisons_to_all_patient_conditions': dict(Counter(a['all_patient_condition_code_comparison']['state'] for a in alignment if a['response_status'] == 'SUPPORTED')),
        'correctness_created': 0})
    closest = sorted((g for g in groups if g['split'] == 'train'), key=lambda g: (-g['supported'], g['case_ref']))
    write(public / 'complete_group_feasibility.json', {'existing_complete_collected': sum(g['collected_complete'] for g in groups),
        'existing_supported_complete': sum(g['supported'] == 5 and g['collected_complete'] for g in groups),
        'eligible_complete_train_groups': gate['complete_train_groups'], 'closest_train_groups_by_existing_supported_count': closest,
        'selected_prospective_groups': [c['case_ref'] for c in selected], 'new_full_group_requests': 10,
        'success_guaranteed': False, 'gate_blockers': gate['blocker_codes'], 'min_samples_sufficient': False})
    write(public / 'split_integrity.json', {'train': len(train), 'validation': len(val), 'original50': len(hold),
        'train_validation_overlap': len(train & val), 'train_holdout_overlap': len(train & hold), 'validation_holdout_overlap': len(val & hold),
        'selected_distinct_patients': len(chosen_patients), 'selected_validation_overlap': len(chosen_patients & val),
        'validation_voi_train_overlap': len(val & voi_train), 'train_voi_train_overlap': len(train & voi_train),
        'integrated_evaluation_independent': False})
    diagnostics = timeout_audit(bundle, private, live_get)
    write(public / 'timeout_diagnostics.json', diagnostics)
    write(private / 'protected_hashes.json', pins); write(private / 'source_hashes.json', source_pins)
    check_hashes(pins); check_hashes(source_pins)
    result = {'status': 'C4_12_SOURCE_REFERENCE_BLOCKED', 'audited_decisions': 15, 'audited_responses': 75,
        'generation_requests': 0, 'fitting_calls': 0, 'clinical_judgments_created': 0,
        'protected_files': len(pins), 'source_files': len(source_pins), 'changed_protected_files': 0,
        'plan_fingerprint': plan['fingerprint'], 'research_evaluation_ready': False}
    write(public / 'verification.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace', default='c4_12_v1')
    p.add_argument('--inventory-get', action='store_true', help='Only loopback GET version/tags/ps, never generation')
    args = p.parse_args()
    print(json.dumps(run(args.namespace, args.inventory_get), ensure_ascii=True))
