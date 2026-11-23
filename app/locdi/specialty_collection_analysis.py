"""Read-only collection integrity analysis and unreviewed review packets.

Never calls a model, judges diagnosis correctness, resumes, or fits a Utility.
Public results contain counts/ordinals only. Actual responses/reference text
remain in a separate private packet with every correctness value unset.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator
from app.locdi.llm_setup import ROOT, digest
from app.locdi.routing_collector import (PRIVATE_ROOT, PUBLIC_ROOT, load_preparation, validate_preparation, _id, atomic_json)
from app.locdi.routing_collection_preparation import project_input
from app.llm.prompt_loader import specialty_system_prompt
from validation.routing_training_data import _leaves, _private_input, validate_split_manifest, validate_training_data


def _safe_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    def constant(_):
        raise ValueError('NONFINITE_JSON')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def _classification(row, schema_valid, integrity_ok):
    if row is None:
        return 'NOT_EVALUABLE'
    if row.get('response_contract_status') == 'INVALID' or row.get('error_type') == 'LLMResponseError':
        return 'SCHEMA_INVALID'
    if row.get('execution_status') != 'SUCCEEDED':
        return 'REQUEST_FAILED'
    if not schema_valid:
        return 'SCHEMA_INVALID'
    if not integrity_ok:
        return 'NOT_EVALUABLE'
    answer = row['response']
    if answer['certainty'] == 'unresolved' or answer['diagnosis'] == 'Unresolved':
        return 'UNRESOLVED'
    return 'RESPONSE_RECORDED'


def analyze_records(bundle, references, records, *, header=None, completion=None, failure=None,
                    audit=None, manifest=None, current_registry=None, current_prompts=None,
                    source_comparison='NOT_CHECKED', run_id='phase24b_collect_001'):
    """Pure analysis. records contains (filename, parsed private record) pairs.

    Hash equality is consistency with intended inputs, not independently
    captured raw wire payload or signed proof of historical authenticity.
    """
    checks, findings = {}, []
    def check(name, okay):
        checks[name] = 'PASS' if okay else 'FAIL'
        if not okay:
            findings.append(name)
    original = deepcopy(bundle); stored = original.pop('fingerprint', None)
    check('PREPARATION_SELF_FINGERPRINT', digest(original) == stored)
    check('RUN_PREPARATION_LINK', isinstance(header, dict) and header.get('bundle_fingerprint') == stored)
    check('RECORDED_DEVELOPMENT_ASSUMPTIONS', isinstance(header, dict) and header.get('accepted_assumptions') == bundle['assumptions'])
    check('PRIVATE_MANIFEST', isinstance(manifest, dict) and manifest.get('complete') is True
          and manifest.get('preparation_fingerprint') == digest(bundle) and manifest.get('reference_fingerprint') == digest(references))
    if current_registry is not None:
        check('CURRENT_REGISTRY_SNAPSHOT', current_registry == bundle['registry'])
    else:
        checks['CURRENT_REGISTRY_SNAPSHOT'] = 'NOT_CHECKED'
    checks['CURRENT_SOURCE_REBUILD'] = source_comparison
    if source_comparison == 'FAIL':
        findings.append('CURRENT_SOURCE_REBUILD')
    agents = [a for a in bundle['registry']['agents'] if a['role'] == 'specialty']
    candidates = bundle['candidates']
    split = bundle['split_manifest']
    split_result = validate_split_manifest(split, holdout_patient_ids=split['patients']['test'], dataset_fingerprint=split['dataset_fingerprint'])
    check('PATIENT_SPLIT', split_result['status'] == 'VALID')
    check('CANDIDATE_PATIENT_UNIQUENESS', len({c['decision']['patient_id'] for c in candidates}) == len(candidates))
    expected = {(c['case_ref'], 'Agent%03d' % (i + 1)): (c, a) for c in candidates for i, a in enumerate(agents)}
    slots, orphan = {}, 0
    for filename, row in records:
        key = (row.get('case_ref'), row.get('agent_ref'))
        if key not in expected:
            orphan += 1; continue
        slots.setdefault(key, []).append((filename, row))
    check('NO_ORPHAN_OUTCOMES', orphan == 0)
    check('NO_DUPLICATE_TASK_OUTCOMES', all(len(rs) == 1 for rs in slots.values()))
    decisions, review_items, task_rows, source_catalog = [], [], [], []
    schema = Draft202012Validator(bundle['response_schema'])
    initial_integrity = not findings
    all_private_identity = set(split['patients']['train'] + split['patients']['validation'] + split['patients']['test'])
    all_private_identity.update(c['decision']['visit_id'] for c in candidates)
    for candidate in candidates:
        decision, case = candidate['decision'], candidate['case_ref']
        observation, shared = decision['observation'], project_input(decision['observation'], bundle['catalog'])
        candidate_ok = (shared == candidate['shared_payload'] and digest(shared) == candidate['shared_input_fingerprint']
                        and digest(observation['longitudinal_memory']) == candidate['memory_fingerprint']
                        and set(candidate['visibility']) == set(_leaves(observation))
                        and not _private_input(observation))
        candidate_ok = candidate_ok and all(type(r['available_minute']) is int and 0 <= r['available_minute'] <= decision['minute']
                                            for r in candidate['visibility'].values())
        serialized_shared = json.dumps(shared, ensure_ascii=False)
        candidate_ok = candidate_ok and not any(value in serialized_shared for value in all_private_identity)
        check(case + '_SNAPSHOT_LEDGER_AND_INPUT_SCOPE', candidate_ok)
        source_catalog.append({'source_ref': decision['source_ref'], 'purpose': 'routing_training',
            **{k: deepcopy(decision[k]) for k in ('patient_id', 'visit_id', 'decision_id', 'minute', 'observation')},
            'source_fingerprint': candidate['source_fingerprint'], 'visit_started_at': candidate['visit_started_at'],
            'visibility': {p: r['available_minute'] for p, r in candidate['visibility'].items()}, 'review': {'status': 'NOT_REVIEWED'}})
        group, private_group = [], []
        for index, agent in enumerate(agents):
            agent_ref = 'Agent%03d' % (index + 1); matches = slots.get((case, agent_ref), [])
            row = matches[0][1] if len(matches) == 1 else None
            row_codes, schema_valid = [], False
            if row is not None:
                filename = matches[0][0]
                # Collector serializes the saved preparation's insertion order.
                # Canonical equality above independently checks regenerated content.
                payload = {'department': agent['department'], **deepcopy(candidate['shared_payload'])}
                expected_fields = {'agent_id': agent['agent_id'], 'model': agent['model'], 'version': agent['version'],
                    'role': 'specialty', 'department': agent['department'], 'patient_id': decision['patient_id'],
                    'decision_id': decision['decision_id'], 'source_ref': decision['source_ref'], 'split': decision['split'],
                    'generation': decision['generation'], 'tokens': decision['tokens'], 'purpose': 'routing_training',
                    'configuration_fingerprint': digest(decision), 'observation_fingerprint': digest(observation),
                    'shared_input_fingerprint': digest(shared), 'full_payload_fingerprint': digest(payload),
                    'system_fingerprint': decision['generation']['prompt_fingerprints'][agent['agent_id']],
                    'serialized_payload_sha256': hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode('utf-8')).hexdigest(),
                    'response_ref': case + '_' + agent_ref}
                for key, value in expected_fields.items():
                    if row.get(key) != value:
                        row_codes.append('FIELD_MISMATCH_' + key.upper())
                if filename != expected_fields['response_ref'] + '.json':
                    row_codes.append('FILENAME_RESPONSE_REF_MISMATCH')
                if current_prompts is not None and current_prompts.get(agent['department']) != row.get('system_fingerprint'):
                    row_codes.append('CURRENT_PROMPT_MISMATCH')
                if row.get('correct') is not None:
                    row_codes.append('UNREVIEWED_CORRECTNESS_PRESENT')
                if row.get('execution_status') == 'SUCCEEDED':
                    schema_valid = isinstance(row.get('response'), dict) and schema.is_valid(row['response'])
                    if not schema_valid or row.get('response_contract_status') != 'VALID':
                        row_codes.append('RESPONSE_SCHEMA_OR_STATUS_MISMATCH')
                    if row.get('response_fingerprint') != digest(row.get('response')):
                        row_codes.append('RESPONSE_CONTENT_HASH_MISMATCH')
                    if row.get('observed_model_identifier') != bundle['llm_config']['model_identifier']:
                        row_codes.append('MODEL_DIGEST_MISMATCH')
                    if row.get('error_type'):
                        row_codes.append('SUCCESS_ERROR_CONTRADICTION')
            okay = row is not None and not row_codes and candidate_ok and initial_integrity
            classification = _classification(row, schema_valid, okay)
            public = {'case_ref': case, 'agent_ref': agent_ref, 'department': agent['department'], 'split': decision['split'],
                      'record_present': row is not None, 'schema_valid': schema_valid, 'integrity': 'PASS' if okay else 'FAIL' if row else 'MISSING',
                      'classification': classification, 'reference_status': 'REFERENCE_NOT_REVIEWED',
                      'evaluation_status': 'NOT_REVIEWED' if classification == 'RESPONSE_RECORDED' else 'NOT_EVALUABLE',
                      'correct': None, 'issue_codes': row_codes}
            group.append(public); task_rows.append(public)
            findings.extend(row_codes)
            private_group.append({**deepcopy(public), 'source_outcome': deepcopy(row),
                'verdict': {'correct': None, 'status': 'NOT_REVIEWED', 'reviewer_ref': None,
                            'procedure_version': None, 'reviewed_at': None, 'verdict_rationale': None}})
        decisions.append({'case_ref': case, 'split': decision['split'], 'stored_agents': sum(t['record_present'] for t in group),
                          'valid_responses': sum(t['schema_valid'] for t in group),
                          'unresolved': sum(t['classification'] == 'UNRESOLVED' for t in group),
                          'response_recorded': sum(t['classification'] == 'RESPONSE_RECORDED' for t in group),
                          'collected_complete': all(t['record_present'] for t in group),
                          'integrity_complete': all(t['integrity'] == 'PASS' for t in group),
                          'reviewed_train_complete': False})
        refs = [r for r in references if r['case_ref'] == case]
        review_items.append({'case_ref': case, 'split': decision['split'], 'private_decision': deepcopy(decision),
            'agent_visible_snapshot': deepcopy(shared), 'visibility_ledger': deepcopy(candidate['visibility']),
            'memory_fingerprint': candidate['memory_fingerprint'], 'reference_candidates': deepcopy(refs),
            'reference_status': 'REFERENCE_NOT_REVIEWED', 'agent_reviews': private_group,
            'review_instructions': 'Independently review source, time scope and reference validity before judging. Do not infer correctness from string matching, confidence or completion.'})
    values = [r for _, r in records]
    execution_counts = dict(Counter(r.get('execution_status', 'NOT_RECORDED') for r in values))
    correctness_counts = dict(Counter(r.get('correctness_status', 'NOT_RECORDED') for r in values))
    error_counts = Counter(r.get('error_type') for r in values)
    if audit is not None:
        check('PUBLIC_AUDIT_PREPARATION_LINK', audit.get('bundle_fingerprint') == stored)
        check('PUBLIC_AUDIT_RESPONSE_COUNTS', audit.get('execution_status_counts') == execution_counts)
        check('PUBLIC_AUDIT_CORRECTNESS_COUNTS', audit.get('correctness_status_counts') == correctness_counts)
    attempts = completion.get('attempts') if completion is not None else failure.get('attempts') if failure is not None else audit.get('attempts') if audit is not None else None
    requests = completion.get('generation_requests') if completion is not None else failure.get('generation_requests') if failure is not None else audit.get('generation_requests') if audit is not None else None
    if completion is not None and audit is not None:
        check('RUN_AND_AUDIT_COUNTERS', requests == audit.get('generation_requests') and attempts == audit.get('attempts') and completion.get('stop_reason') == audit.get('stop_reason'))
    if attempts is not None and requests is not None and header is not None:
        check('GLOBAL_RECORDED_REQUEST_LIMIT', type(attempts) is int and type(requests) is int
              and 0 <= requests <= attempts <= header.get('global_request_limit', -1))
    if completion is not None and completion.get('status') == 'FINISHED':
        check('FINISHED_RECORD_COUNT', attempts == len(records) == len(expected))
    counts = dict(Counter(t['classification'] for t in task_rows))
    missing = [{'case_ref': t['case_ref'], 'agent_ref': t['agent_ref'], 'department': t['department']} for t in task_rows if not t['record_present']]
    if completion and completion.get('status') == 'FINISHED' and not missing and not findings and all(t['schema_valid'] for t in task_rows):
        state = 'COLLECTION_COMPLETE_NOT_CLINICALLY_EVALUATED'
    elif failure is not None:
        state = 'FAILED_WITH_PRESERVED_PARTIAL_RESULTS' if values else 'FAILED_NO_STORED_OUTCOMES'
    elif values:
        state = 'PARTIAL_OR_INTEGRITY_BLOCKED'
    else:
        state = 'STARTED_NO_STORED_OUTCOMES'
    data = {'schema_version': 'locdi.routing_collection.v1', 'purpose': 'routing_training', 'registry': bundle['registry'],
            'split_manifest': split, 'decisions': [c['decision'] for c in candidates], 'outcomes': deepcopy(values), 'judgments': []}
    training = validate_training_data(data, registry_manifest=bundle['registry'], source_catalog=source_catalog, label_catalog=[],
                                      holdout_patient_ids=split['patients']['test'], dataset_fingerprint=split['dataset_fingerprint'])
    def totals(field, value):
        ts = [t for t in task_rows if t[field] == value]
        return {'planned': len(ts), 'stored': sum(t['record_present'] for t in ts),
                'schema_valid': sum(t['schema_valid'] for t in ts),
                'unresolved': sum(t['classification'] == 'UNRESOLVED' for t in ts),
                'response_recorded': sum(t['classification'] == 'RESPONSE_RECORDED' for t in ts),
                'missing': sum(not t['record_present'] for t in ts)}
    summary = {'schema_version': 'locdi.specialty_collection_analysis.v1', 'run_id': run_id, 'state': state,
        'planned_decisions': len(candidates), 'planned_tasks': len(expected), 'stored_outcome_files': len(records),
        'recorded_attempts': attempts, 'recorded_generation_requests': requests,
        'recorded_identity_checks': audit.get('identity_checks') if audit is not None else None,
        'successful_record_count': execution_counts.get('SUCCEEDED', 0),
        'schema_invalid_count': counts.get('SCHEMA_INVALID', 0), 'timeout_count': error_counts.get('LLMTimeoutError', 0),
        'client_or_identity_error_count': error_counts.get('LLMRequestError', 0) + error_counts.get('ModelIdentityError', 0),
        'classification_counts': counts, 'missing_tasks': missing, 'orphan_record_count': orphan,
        'checks': checks, 'integrity_issue_codes': sorted(set(findings)), 'decisions': decisions,
        'specialties': {a['department']: totals('department', a['department']) for a in agents},
        'splits': {s: {**totals('split', s), 'collected_complete_groups': sum(d['collected_complete'] for d in decisions if d['split'] == s),
                      'integrity_complete_groups': sum(d['integrity_complete'] for d in decisions if d['split'] == s)} for s in ('train', 'validation')},
        'reference_status': 'REFERENCE_NOT_REVIEWED', 'all_correctness_null': all(r.get('correct') is None for r in values),
        'training_validation': training, 'training_readiness': 'BLOCKED', 'eligible_reviewed_train_groups': 0,
        'min_samples_status': 'BLOCKED_NO_REVIEWED_TRAIN_SAMPLES', 'per_observation_similarity_support': 'NOT_CHECKED',
        'raw_wire_payload': 'NOT_CHECKED_NOT_SAVED', 'actual_clinical_result_release': 'NOT_REVIEWED_PROXY_ONLY',
        'historical_overwrite': 'NOT_CHECKED_NO_PRE_RUN_BASELINE', 'new_llm_calls': 0, 'utility_fit_count': 0,
        'clinical_metrics': 'NOT_EVALUATED', 'research_readiness': False}
    return summary, {'schema_version': 'locdi.private_correctness_review.v1', 'run_id': run_id,
                     'preparation_fingerprint': stored, 'review_status': 'NOT_REVIEWED', 'items': review_items}


def analyze_run(run_id='phase24b_collect_001', preparation='phase24b_candidates_v2'):
    run = PRIVATE_ROOT / _id(run_id)
    if not run.is_dir():
        available = [p for p in PRIVATE_ROOT.iterdir() if p.is_dir() and (p/'run_header.json').is_file()]
        if len(available) != 1:
            raise ValueError('RUN_MISSING_OR_AMBIGUOUS_SPECIFY_RUN_ID')
        run = available[0]; run_id = run.name
    bundle = load_preparation(preparation)
    prep = PRIVATE_ROOT / preparation
    refs, manifest = _safe_json(prep/'reference_candidates.json'), _safe_json(prep/'manifest.json')
    optional = lambda p: _safe_json(p) if p.is_file() else None
    records = [(p.name, _safe_json(p)) for p in sorted(run.glob('Candidate*.json'))]
    if not records:
        # Still report genuine start/failure metadata; never synthesize outcomes.
        if not (run/'run_header.json').is_file():
            raise ValueError('NO_COLLECTION_EXECUTION_RESULTS')
    from app.locdi.routing_training_setup import REGISTRY
    current_registry = _safe_json(REGISTRY)
    prompts = {a['department']: hashlib.sha256(specialty_system_prompt(a['department']).encode('utf-8')).hexdigest()
               for a in current_registry['agents'] if a['role'] == 'specialty'}
    try:
        validate_preparation(bundle)
        source_status = 'PASS'
    except Exception:
        source_status = 'FAIL'
    summary, private = analyze_records(bundle, refs, records, header=optional(run/'run_header.json'),
        completion=optional(run/'run_complete.json'), failure=optional(run/'run_failed.json'),
        audit=optional(PUBLIC_ROOT/(run_id+'_routing_collection.json')), manifest=manifest,
        current_registry=current_registry, current_prompts=prompts, source_comparison=source_status, run_id=run_id)
    private['source_file_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.json')}
    return summary, private


def save_review(summary, private, review_id):
    root = ROOT / 'outputs/private/routing_collection_review'
    if ROOT.resolve() not in root.resolve().parents:
        raise ValueError('REVIEW_ROOT_REDIRECTED')
    path = root / _id(review_id)
    root.mkdir(parents=True, exist_ok=True); path.mkdir(exist_ok=False)
    atomic_json(path/'review_packet.json', private)
    atomic_json(path/'manifest.json', {'complete': True, 'run_id': summary['run_id'], 'packet_fingerprint': digest(private)})
    atomic_json(PUBLIC_ROOT/(_id(review_id)+'_summary.json'), summary)
    return path
