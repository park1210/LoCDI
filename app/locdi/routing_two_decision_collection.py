"""Pinned C4-13 development collection; no fitting or clinical adjudication.

Separate wrapper over the existing adapter. No monkeypatching of production
globals, resume, retry, source projection changes or automatic review approval.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from jsonschema import Draft202012Validator
from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_freeze import load, sha
from app.locdi.routing_collector import load_preparation, validate_preparation, atomic_json
from app.locdi.ollama_adapter import LocalOllamaAdapter, _backend, LLMTimeoutError, LLMResponseError, ModelIdentityError
from app.llm.prompt_loader import specialty_system_prompt

VERSION = 'locdi.routing_two_decision.c4_13.v1'
PARENT_FP = '330ed6ea6cc87fece3293fb5a5735647c89a9d05e76a61d9047980790f5272c9'
SELECTED = ['Candidate001', 'Candidate002']
GENERATION = {'temperature': 0, 'seed': 42, 'max_output_tokens': 1024, 'timeout_seconds': 120}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != expected:
            raise ValueError('SOURCE_OR_BINDING_HASH_MISMATCH')


def signed(document):
    return {**document, 'fingerprint': digest(document)}


def verify_fingerprint(document):
    if document.get('fingerprint') != digest({k: v for k, v in document.items() if k != 'fingerprint'}):
        raise ValueError('DOCUMENT_FINGERPRINT_MISMATCH')


def evaluation_protocol():
    return signed({'version': 'locdi.routing_diagnosis_reference.c4_13.v1', 'target': 'diagnosis_correctness',
        'response': 'One named diagnosis in unchanged shared catalog with supported certainty; schema validity alone is not clinical evaluability',
        'unresolved': 'Preserved; separate abstention review; no diagnosis boolean',
        'multiple_conditions': 'Independent reviewer must define current task/reference; multiple source conditions are not automatically accepted answers',
        'past_condition': 'Code match is source linkage only, not current diagnosis correctness',
        'code_mapping': 'Exact System/Code or explicit officially sourced mapping reviewed by a human; no automatic synonym equivalence',
        'date_precision': 'Native recorded onset does not establish clinical release; proxy acceptance requires actual scope review',
        'unknown_reference': 'ABSTAIN_FROM_CORRECTNESS_ADJUDICATION_KEEP_NULL',
        'missing_model_response': 'NOT_EVALUATED_NOT_INCORRECT',
        'forbidden': ['Unresolved->Incorrect', 'Unresolved->Correct', 'Past code match->Clinical Correct',
            'Source code mismatch->Clinical Incorrect', 'Generation failure->Incorrect', 'Source Proxy->Independent Gold'],
        'source_review_status': 'REFERENCE_REVIEW_REQUIRED', 'proxy_scope_status': 'NOT_ACCEPTED',
        'clinical_review_generated': False})


def build_plan(bundle, parent):
    verify_fingerprint(parent)
    if parent['fingerprint'] != PARENT_FP or [x['case_ref'] for x in parent['selected']] != SELECTED:
        raise ValueError('C4_12_SELECTION_OR_PLAN_CHANGED')
    agents = validate_preparation(bundle)
    selected = [next(c for c in bundle['candidates'] if c['case_ref'] == ref) for ref in SELECTED]
    if (any(c['decision']['split'] != 'train' for c in selected)
            or len({c['decision']['patient_id'] for c in selected}) != 2
            or {c['decision']['patient_id'] for c in selected} & set(bundle['split_manifest']['patients']['validation'])):
        raise ValueError('TWO_DISTINCT_TRAIN_PATIENTS_REQUIRED')
    if (digest(bundle['registry']) != parent['registry_fingerprint'] or bundle['fingerprint'] != parent['preparation_fingerprint']
            or digest(bundle['catalog']) != parent['catalog_fingerprint'] or agents != parent['agents']):
        raise ValueError('REGISTRY_CATALOG_PREPARATION_CHANGED')
    cfg = bundle['llm_config']
    if any(cfg.get(k) != v for k, v in GENERATION.items()) or cfg.get('request_retries') != 0:
        raise ValueError('GENERATION_CONFIGURATION_CHANGED')
    if any(cfg[k] != parent[k] for k in ('model', 'model_identifier', 'model_version')):
        raise ValueError('MODEL_BINDING_CHANGED')
    for c, old in zip(selected, parent['selected']):
        if (any(c[k] != old[k] for k in ('shared_input_fingerprint', 'memory_fingerprint', 'source_fingerprint'))
                or c['decision']['generation']['prompt_fingerprints'] != old['prompt_fingerprints']):
            raise ValueError('SOURCE_INPUT_PROMPT_CHANGED')
    return signed({'schema_version': VERSION, 'parent_plan_fingerprint': PARENT_FP,
        'preparation_fingerprint': bundle['fingerprint'], 'registry_fingerprint': digest(bundle['registry']),
        'catalog_fingerprint': digest(bundle['catalog']), 'response_schema_fingerprint': digest(bundle['response_schema']),
        'evaluation_protocol_fingerprint': evaluation_protocol()['fingerprint'],
        'selected': deepcopy(parent['selected']), 'agents': deepcopy(agents),
        'model': cfg['model'], 'model_identifier': cfg['model_identifier'], 'model_version': cfg['model_version'],
        'generation': deepcopy(GENERATION), 'global_budget': 10, 'retry': 0, 'repair': 0,
        'reference_status': 'REFERENCE_REVIEW_REQUIRED', 'proxy_scope_status': 'NOT_ACCEPTED_FOR_FITTING',
        'purpose': 'DEVELOPMENT_ROUTING_RESPONSE_COLLECTION_ONLY', 'clinical_metrics': 'NOT_EVALUATED',
        'mixing': 'NEW_RUN_WHOLE_GROUPS_ONLY_NO_BASELINE_REPLACEMENT', 'stop_on_first_error': True,
        'resume': False, 'server_settings_changed': False})


def load_protocol(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace):
        raise ValueError('INVALID_PROTOCOL_NAMESPACE')
    private = ROOT / 'outputs/private/research_evaluation' / namespace
    manifest = load(private / 'protocol_manifest.json')
    verify_fingerprint(manifest)
    for name, expected in manifest['files'].items():
        if name not in ('plan.json', 'bindings.json') or sha(private / name) != expected:
            raise ValueError('PROTOCOL_MANIFEST_MISMATCH')
    if set(manifest['files']) != {'plan.json', 'bindings.json'}:
        raise ValueError('PROTOCOL_MANIFEST_SCOPE')
    plan = load(private / 'plan.json'); bindings = load(private / 'bindings.json')
    verify_fingerprint(plan); verify_pins(bindings['source_pins']); verify_pins(bindings['binding_pins'])
    bundle = load_preparation('phase24b_candidates_v2')
    parent = load(ROOT / 'outputs/research_evaluation/c4_12_v1/next_collection_plan.json')
    if build_plan(bundle, parent) != plan:
        raise ValueError('FROZEN_PLAN_NO_LONGER_MATCHES')
    return bundle, plan, bindings


class ObservedAdapter(LocalOllamaAdapter):
    """Observe HTTP status without altering base timeout, identity or JSON gates."""
    def chat_json(self, *args, **kwargs):
        self.last_post_status = None
        if self._resources is None:
            self._resources = _backend(self.prepared)  # Lazy HTTP client construction, no request.
            def observe(response):
                if response.request.method == 'POST':
                    self.last_post_status = response.status_code
            self._resources[1].event_hooks.setdefault('response', []).append(observe)
        return super().chat_json(*args, **kwargs)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('DUPLICATE_JSON_KEY')
            out[key] = value
        return out
    def constant(_):
        raise ValueError('NONFINITE_JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def locations(namespace, run_id):
    for value in (namespace, run_id):
        if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}', value):
            raise ValueError('INVALID_NAMESPACE_OR_RUN_ID')
    public = ROOT / 'outputs/research_evaluation' / namespace / 'collection' / run_id
    private = ROOT / 'outputs/private/research_evaluation' / namespace / 'collection' / run_id
    for path in (public, private):
        if not path.resolve().is_relative_to((ROOT / 'outputs').resolve()):
            raise ValueError('OUTPUT_PATH_ESCAPE')
        if path.exists():
            raise ValueError('RUN_ALREADY_USED_NO_RESUME')
    return public, private


def validate_dispatch(bundle, plan, candidate, agent, bindings):
    verify_pins(bindings['binding_pins'])  # Frozen source snapshot, Registry, prompts and wrapper code.
    if digest(bundle['registry']) != plan['registry_fingerprint']:
        raise ValueError('REGISTRY_CHANGED')
    if (agent not in plan['agents'] or agent['model'] != plan['model'] or agent['version'] != plan['model_version']):
        raise ValueError('MODEL_AGENT_CHANGED')
    pin = next(x for x in plan['selected'] if x['case_ref'] == candidate['case_ref'])
    if (digest(candidate['shared_payload']) != pin['shared_input_fingerprint']
            or candidate['source_fingerprint'] != pin['source_fingerprint'] or candidate['memory_fingerprint'] != pin['memory_fingerprint']):
        raise ValueError('SOURCE_SNAPSHOT_CHANGED')
    prompt = specialty_system_prompt(agent['department'])
    if hashlib.sha256(prompt.encode('utf-8')).hexdigest() != pin['prompt_fingerprints'][agent['agent_id']]:
        raise ValueError('PROMPT_CHANGED')
    return prompt, {'department': agent['department'], **deepcopy(candidate['shared_payload'])}


def collect(namespace, run_id, *, execute=False, allow_llm=False, accept_proxy=False,
            acknowledge_timeout=False, expected_plan_fingerprint=None, max_requests=10):
    if any(type(x) is not bool for x in (execute, allow_llm, accept_proxy, acknowledge_timeout)):
        raise ValueError('EXPLICIT_BOOLEAN_FLAGS_REQUIRED')
    if type(max_requests) is not int or not 1 <= max_requests <= 10:
        raise ValueError('GLOBAL_BUDGET_1_TO_10')
    if execute and not (allow_llm and accept_proxy and acknowledge_timeout and expected_plan_fingerprint):
        raise ValueError('EXPLICIT_EXECUTION_PROXY_TIMEOUT_AND_PLAN_APPROVAL_REQUIRED')
    public, private = locations(namespace, run_id)
    bundle, plan, bindings = load_protocol(namespace)
    if expected_plan_fingerprint is not None and expected_plan_fingerprint != plan['fingerprint']:
        raise ValueError('APPROVED_PLAN_MISMATCH')
    if not execute:
        return {'status': 'DRY_RUN', 'plan_fingerprint': plan['fingerprint'], 'selected': SELECTED,
            'ordered_tasks': [{'case_ref': ref, 'agent_id': a['agent_id']} for ref in SELECTED for a in plan['agents']],
            'max_requests': max_requests, 'generation_requests': 0, 'client_constructed': False,
            'reference_status': plan['reference_status'], 'utility_fit_ready': False}
    # Full raw-source SHA verification happens before any client or BEFORE.
    # Per dispatch, recheck immutable snapshot/Registry/Prompt/code file SHA.
    private.mkdir(parents=True, exist_ok=False); public.mkdir(parents=True, exist_ok=False)
    atomic_json(private / 'plan.json', plan)
    atomic_json(private / 'operator_execution_record.json', {'type': 'EXPLICIT_CLI_OPERATOR_ACTION',
        'recorded_at': stamp(), 'plan_fingerprint': plan['fingerprint'], 'max_requests': max_requests,
        'proxy_accepted_for_generation_only': accept_proxy, 'timeout_observations_acknowledged': acknowledge_timeout,
        'professor_or_clinical_approval_claimed': False, 'reference_status': 'REFERENCE_REVIEW_REQUIRED'})
    client = None; results = []; stop = None; attempts = 0
    try:
        cfg = deepcopy(bundle['llm_config']); cfg['max_requests'] = max_requests
        client = ObservedAdapter(cfg, authorized=True)
        validator = Draft202012Validator(bundle['response_schema'])
        for ref in SELECTED:
            candidate = next(c for c in bundle['candidates'] if c['case_ref'] == ref)
            for agent in plan['agents']:
                if attempts >= max_requests:
                    stop = 'GLOBAL_REQUEST_LIMIT'; break
                prompt, payload = validate_dispatch(bundle, plan, candidate, agent, bindings)
                request_ref = 'Request%03d' % (attempts + 1)
                before = {'run_id': run_id, 'request_ref': request_ref, 'case_ref': ref,
                    'agent_id': agent['agent_id'], 'model': agent['model'], 'version': agent['version'],
                    'decision_id': candidate['decision']['decision_id'], 'patient_id': candidate['decision']['patient_id'],
                    'shared_input_fingerprint': candidate['shared_input_fingerprint'], 'full_payload_fingerprint': digest(payload),
                    'source_fingerprint': candidate['source_fingerprint'], 'memory_fingerprint': candidate['memory_fingerprint'],
                    'prompt_fingerprint': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                    'request_started_at': stamp(), 'generation_count_before': client.request_count,
                    'payload': payload, 'system_prompt': prompt, 'generation': plan['generation']}
                atomic_json(private / (request_ref + '_before.json'), before)
                attempts += 1; started = time.monotonic()
                after = {k: v for k, v in before.items() if k not in ('payload', 'system_prompt')}
                after.update(raw_response=None, response=None, response_fingerprint=None, error_type=None,
                    error_category=None, json_valid=False, schema_valid=False, response_status='NOT_EVALUATED',
                    correct=None, correctness_status='NOT_EVALUATED_NO_INDEPENDENT_REVIEW',
                    model_loading_seconds=None, model_loading_status='NOT_EXPOSED_BY_EXISTING_OPENAI_RESPONSE_PATH')
                returned = None
                try:
                    returned = client.chat_json(prompt, json.dumps(payload, ensure_ascii=False), model=agent['model'], schema=bundle['response_schema'])
                except Exception as exc:
                    after['error_type'] = type(exc).__name__
                    status = getattr(client, 'last_post_status', None)
                    after['error_category'] = ('TIMEOUT' if isinstance(exc, (LLMTimeoutError, TimeoutError)) else
                        'SERVER_HTTP_ERROR' if status is not None and status >= 400 else
                        'SCHEMA_OR_JSON_ERROR' if isinstance(exc, LLMResponseError) else
                        'IDENTITY_ERROR' if isinstance(exc, ModelIdentityError) else 'CLIENT_ERROR_HTTP_STATUS_NOT_OBSERVED')
                raw = client.last_response_content if client.last_response_content is not None else returned
                after['raw_response'] = raw
                if raw is not None:
                    after['raw_response_sha256'] = hashlib.sha256(raw.encode('utf-8')).hexdigest()
                    try:
                        value = strict_json(raw); after['json_valid'] = True
                        validator.validate(value); after['schema_valid'] = True
                        after['response'] = value; after['response_fingerprint'] = digest(value)
                        after['response_status'] = 'UNRESOLVED' if value['certainty'] == 'unresolved' or value['diagnosis'] == 'Unresolved' else 'SUPPORTED_REFERENCE_NOT_REVIEWED'
                    except Exception as exc:
                        if after['error_type'] is None:
                            after.update(error_type=type(exc).__name__, error_category='SCHEMA_OR_JSON_ERROR')
                if raw is None and after['error_type'] is None:
                    after.update(error_type='EmptyResponseError', error_category='EMPTY_RESPONSE')
                after.update(request_wall_seconds=time.monotonic() - started, recorded_at=stamp(),
                    generation_count_after=client.request_count, identity_checks_after=client.identity_check_count,
                    observed_model_identifier=client.last_observed_model_identifier,
                    http_post_status=getattr(client, 'last_post_status', None))
                if after['error_type'] is not None:
                    stop = after['error_category']
                atomic_json(private / (request_ref + '_after.json'), after); results.append(after)
                if stop: break
            if stop: break
        groups = {ref: [r for r in results if r['case_ref'] == ref] for ref in SELECTED}
        summary = {'status': 'COLLECTION_COMPLETE_NOT_CLINICALLY_REVIEWED' if len(results) == 10 and not stop else 'COLLECTION_PARTIAL_OR_FAILED',
            'run_id': run_id, 'plan_fingerprint': plan['fingerprint'], 'attempts': attempts,
            'generation_requests': client.request_count, 'identity_checks': client.identity_check_count,
            'received_responses': sum(r['raw_response'] is not None for r in results),
            'json_valid_responses': sum(r['json_valid'] for r in results), 'schema_valid_responses': sum(r['schema_valid'] for r in results),
            'supported_responses': sum(r['response_status'] == 'SUPPORTED_REFERENCE_NOT_REVIEWED' for r in results),
            'unresolved_responses': sum(r['response_status'] == 'UNRESOLVED' for r in results),
            'independently_evaluable_responses': 0, 'complete_clinically_reviewed_train_groups': 0,
            'error_categories': dict(Counter(r['error_category'] for r in results if r['error_category'])),
            'stop_reason': stop, 'not_attempted_tasks': 10 - attempts,
            'groups': [{'case_ref': ref, 'attempts': len(rows), 'schema_complete': len(rows) == 5 and all(r['schema_valid'] and not r['error_type'] for r in rows)} for ref, rows in groups.items()],
            'retry': 0, 'repair': 0, 'correctness_created': 0, 'utility_fit_count': 0, 'clinical_metrics': 'NOT_EVALUATED',
            'training_readiness': 'BLOCKED', 'merged_with_baseline': False}
        atomic_json(private / 'summary.json', summary); atomic_json(public / 'summary.json', summary)
        files = {p.name: sha(p) for p in private.glob('*.json')}
        atomic_json(private / 'run_manifest.json', signed({'version': VERSION, 'complete': True, 'files': files}))
        return summary
    except Exception as exc:
        failure = {'status': 'COLLECTION_STORAGE_OR_INTEGRITY_FAILED', 'error_type': type(exc).__name__,
            'attempts': attempts, 'generation_requests': client.request_count if client else 0,
            'persisted_after_checkpoints': len(results), 'no_resume': True}
        try:
            atomic_json(private / 'failed.json', failure); atomic_json(public / 'failed.json', failure)
        except OSError:
            pass
        raise
    finally:
        if client is not None: client.close()


def assess_run(namespace, run_id, review_package=None):
    """Read-only follow-up. A review package is supplied by humans, never inferred."""
    bundle, plan, bindings = load_protocol(namespace)
    private = ROOT / 'outputs/private/research_evaluation' / namespace / 'collection' / run_id
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', run_id): raise ValueError('INVALID_RUN_ID')
    manifest = load(private / 'run_manifest.json'); verify_fingerprint(manifest)
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(private / name) != expected: raise ValueError('RUN_HASH_MISMATCH')
    if manifest.get('complete') is not True: raise ValueError('RUN_INCOMPLETE_MANIFEST')
    summary = load(private / 'summary.json')
    if summary['plan_fingerprint'] != plan['fingerprint'] or load(private / 'plan.json') != plan:
        raise ValueError('RUN_PLAN_MISMATCH')
    afters = [load(private / n) for n in sorted(manifest['files']) if n.endswith('_after.json')]
    for saved in afters:
        before_name = saved['request_ref'] + '_before.json'
        if before_name not in manifest['files']: raise ValueError('BEFORE_CHECKPOINT_MISSING')
        before = load(private / before_name)
        if any(before[k] != saved[k] for k in ('case_ref', 'agent_id', 'decision_id', 'patient_id', 'shared_input_fingerprint', 'full_payload_fingerprint', 'prompt_fingerprint')):
            raise ValueError('CHECKPOINT_LINEAGE_MISMATCH')
        if saved['response'] is not None and digest(saved['response']) != saved['response_fingerprint']:
            raise ValueError('RESPONSE_FINGERPRINT_MISMATCH')
    actual = {(r['decision_id'], r['agent_id']): r for r in afters}
    reasons = []
    if len(afters) != 10 or len(actual) != 10 or not all(r['schema_valid'] and r['response_status'] == 'SUPPORTED_REFERENCE_NOT_REVIEWED' and r['error_type'] is None for r in afters):
        reasons.append('TWO_FULL_SUPPORTED_FIVE_AGENT_GROUPS_NOT_AVAILABLE')
    report = None
    if review_package is None:
        reasons.append('INDEPENDENT_REFERENCE_AND_RESPONSE_REVIEW_PACKAGE_MISSING')
    else:
        from validation.routing_correctness_gate import validate_evaluation_gate
        data, options = review_package['data'], review_package['options']
        if data.get('purpose') != 'routing_training' or options.get('min_samples', 2) != 2 or options.get('allow_test_only', False):
            raise ValueError('REAL_DIAGNOSIS_GATE_WITH_MIN_SAMPLES_TWO_REQUIRED')
        if options.get('registry_manifest') != bundle['registry'] or data.get('registry') != bundle['registry']:
            raise ValueError('REVIEW_REGISTRY_MISMATCH')
        expected_decisions = {c['decision']['decision_id']: c['decision'] for c in bundle['candidates']}
        if any(expected_decisions.get(d['decision_id']) != d for d in data['decisions']):
            raise ValueError('REVIEW_DECISION_SOURCE_CHANGED')
        new_rows = [r for r in data['outcomes'] if r.get('split') == 'train']
        if len(new_rows) != 10 or {(r['decision_id'], r['agent_id']) for r in new_rows} != set(actual):
            raise ValueError('NO_OLD_NEW_RESPONSE_MIXING')
        for row in new_rows:
            saved = actual[(row['decision_id'], row['agent_id'])]
            if (row['response'] != saved['response'] or row['response_fingerprint'] != saved['response_fingerprint']
                    or row['model'] != saved['model'] or row['version'] != saved['version']
                    or row['shared_input_fingerprint'] != saved['shared_input_fingerprint']):
                raise ValueError('REVIEW_RESPONSE_LINEAGE_MISMATCH')
        report = validate_evaluation_gate(data, **options)
        reasons.extend(report['blocker_codes'])
    return {'status': 'REQUIREMENTS_MET_NO_FITTING_PERFORMED' if report and report['utility_fit_ready'] and not reasons else 'BLOCKED',
        'blocker_codes': sorted(set(reasons)), 'existing_gate': report, 'fitting_called': False,
        'source_proxy_path': 'SEPARATE_REVIEWED_PROXY_CONTRACT_REQUIRED_NOT_INDEPENDENT_GOLD',
        'research_evaluation_ready': False}
