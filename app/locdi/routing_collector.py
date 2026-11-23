"""Opt-in private routing collector. No fitting or correctness inference.

Dry-run never constructs a transport. Execution requires separate positive
authorization and explicit acceptance of the pinned development assumptions.
One generation per task, no repair/retry. Global maximum 75, across all agents.
"""
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import tempfile

from app.locdi.llm_setup import ROOT, digest, validate_setup
from app.locdi.ollama_adapter import LocalOllamaAdapter, LLMTimeoutError, LLMResponseError, LLMRequestError, ModelIdentityError
from app.locdi.routing_collection_preparation import (build_preparation, preparation_audit, project_input,
    ASSUMPTIONS, DEPARTMENTS, response_schema)
from app.llm.prompt_loader import specialty_system_prompt
from validation.routing_training_data import _leaves, _private_input, validate_split_manifest

PRIVATE_ROOT = ROOT / 'outputs/private/routing_collection'
PUBLIC_ROOT = ROOT / 'outputs/reports'


def _id(value):
    if not isinstance(value, str) or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', value):
        raise ValueError('INVALID_RUN_ID')
    return value


def _directory(root, name, *, create=False):
    nominal = Path(root)
    root = nominal.resolve(); path = (root / _id(name)).resolve()
    if nominal == PRIVATE_ROOT and ROOT.resolve() not in root.parents:
        raise ValueError('PRIVATE_ROOT_REDIRECTED_OUTSIDE_WORKSPACE')
    if path.parent != root:
        raise ValueError('OUTPUT_PATH_ESCAPE')
    if create:
        root.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
    return path


def atomic_json(path, document):
    """Same-directory fsync + exclusive hard link; existing data never replaced."""
    serialized = json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=str(path.parent), delete=False) as stream:
            temporary = Path(stream.name); stream.write(serialized); stream.flush(); os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        if temporary is not None:
            temporary.unlink()


def save_preparation(bundle, references, name, *, root=PRIVATE_ROOT):
    path = _directory(root, name, create=True)
    atomic_json(path / 'preparation.json', bundle)
    atomic_json(path / 'reference_candidates.json', references)
    # Manifest last. A partially written directory cannot be loaded as complete.
    atomic_json(path / 'manifest.json', {'schema_version': 'locdi.private_preparation_manifest.v1',
        'preparation_fingerprint': digest(bundle), 'reference_fingerprint': digest(references), 'complete': True})
    return path


def load_preparation(name, *, root=PRIVATE_ROOT):
    path = _directory(root, name)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    bundle = json.loads((path / 'preparation.json').read_text(encoding='utf-8'))
    references = json.loads((path / 'reference_candidates.json').read_text(encoding='utf-8'))
    if manifest.get('complete') is not True or manifest.get('preparation_fingerprint') != digest(bundle) or manifest.get('reference_fingerprint') != digest(references):
        raise ValueError('PRIVATE_PREPARATION_INCOMPLETE_OR_CHANGED')
    return bundle


def validate_preparation(bundle):
    """Reconstruct from current pinned source files; never trust edited inputs.

    Ledger assumptions are inspected without marking source/reference reviewed.
    Reference review is deferred for generation only; fitting stays blocked.
    """
    if bundle.get('schema_version') != 'locdi.routing_preparation.v1' or bundle.get('assumptions') != ASSUMPTIONS:
        raise ValueError('PREPARATION_SCHEMA_INVALID')
    private = deepcopy(bundle); fingerprint = private.pop('fingerprint', None)
    if digest(private) != fingerprint:
        raise ValueError('PREPARATION_FINGERPRINT_MISMATCH')
    fresh, _ = build_preparation()
    if fresh['fingerprint'] != bundle['fingerprint']:
        raise ValueError('PREPARATION_OR_SOURCE_CHANGED_REPREPARE_REQUIRED')
    config = bundle['llm_config']; setup = validate_setup(config)
    if setup['static_readiness'] != 'READY' or setup['prepared']['registry'] != bundle['registry']:
        raise ValueError('CURRENT_MODEL_BINDING_INVALID')
    agents = [a for a in bundle['registry']['agents'] if a['role'] == 'specialty']
    if len(agents) != 5 or {a['department'] for a in agents} != set(DEPARTMENTS):
        raise ValueError('REGISTRY_SCOPE_INVALID')
    if bundle['response_schema'] != response_schema(bundle['catalog']):
        raise ValueError('RESPONSE_SCHEMA_CHANGED')
    manifest = bundle['split_manifest']
    split = validate_split_manifest(manifest, holdout_patient_ids=manifest['patients']['test'], dataset_fingerprint=manifest['dataset_fingerprint'])
    if split['status'] != 'VALID' or len(bundle['candidates']) != 15 or split['train_patient_count'] != 10 or split['validation_patient_count'] != 5:
        raise ValueError('COLLECTION_SPLIT_INVALID')
    for candidate in bundle['candidates']:
        decision, ledger = candidate['decision'], candidate['visibility']
        observation = decision['observation']
        if set(ledger) != set(_leaves(observation)) or _private_input(observation):
            raise ValueError('UNTRACKED_OR_PRIVATE_OBSERVATION')
        for row in ledger.values():
            available = row['available_minute']
            if type(available) is not int or not 0 <= available <= decision['minute']:
                raise ValueError('UNRELEASED_OBSERVATION')
        identity = decision['patient_id'], decision['visit_id']
        shared = project_input(observation, bundle['catalog'])
        if shared != candidate['shared_payload'] or digest(shared) != candidate['shared_input_fingerprint']:
            raise ValueError('SHARED_INPUT_CHANGED')
        text = json.dumps(shared, ensure_ascii=False)
        if any(value in text for value in identity) or re.search(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b', text):
            raise ValueError('PRIVATE_IDENTIFIER_IN_PAYLOAD')
        for agent in agents:
            planned_payload = {'department': agent['department'], **deepcopy(shared)}
            common = {k: v for k, v in planned_payload.items() if k != 'department'}
            if digest(common) != candidate['shared_input_fingerprint']:
                raise ValueError('AGENT_INPUT_FINGERPRINT_MISMATCH')
            if decision['generation']['prompt_fingerprints'][agent['agent_id']] != setup['prepared']['prompt_fingerprints'][agent['agent_id']]:
                raise ValueError('PROMPT_BINDING_CHANGED')
    return agents


def collect(bundle, *, execute=False, allow_llm=False, accept_development_assumptions=False,
            run_id=None, max_requests=75, private_root=PRIVATE_ROOT, public_root=PUBLIC_ROOT):
    # Reject authorization failures before static rebuild, storage or transport.
    if any(type(flag) is not bool for flag in (execute, allow_llm, accept_development_assumptions)):
        raise ValueError('AUTHORIZATION_FLAGS_MUST_BE_EXPLICIT_BOOLEANS')
    if execute and not (allow_llm is True and accept_development_assumptions is True):
        raise ValueError('EXECUTION_AND_DEVELOPMENT_ASSUMPTIONS_REQUIRE_EXPLICIT_OPT_IN')
    if type(max_requests) is not int or not 1 <= max_requests <= 75:
        raise ValueError('GLOBAL_REQUEST_LIMIT_MUST_BE_1_TO_75')
    agents = validate_preparation(bundle)
    report = preparation_audit(bundle)
    report['global_request_limit'] = max_requests
    report['shared_input_checks'] = len(bundle['candidates']) * len(agents)
    report['shared_input_identical'] = True
    if not execute:
        return report
    path = _directory(private_root, run_id, create=True)
    public_nominal = Path(public_root); public_root = public_nominal.resolve()
    if public_nominal == PUBLIC_ROOT and ROOT.resolve() not in public_root.parents:
        raise ValueError('PUBLIC_ROOT_REDIRECTED_OUTSIDE_WORKSPACE')
    public_root.mkdir(parents=True, exist_ok=True)
    public_path = public_root / (_id(run_id) + '_routing_collection.json')
    if public_path.exists():
        raise FileExistsError('PUBLIC_RESULT_EXISTS')
    # Durable run header before the first possible server connection.
    atomic_json(path / 'run_header.json', {'schema_version': 'locdi.routing_collection_run.v1',
        'bundle_fingerprint': bundle['fingerprint'], 'accepted_assumptions': ASSUMPTIONS,
        'global_request_limit': max_requests, 'status': 'STARTED', 'reference_status': 'NOT_REVIEWED'})
    config = deepcopy(bundle['llm_config']); config['max_requests'] = max_requests
    client = None
    records, attempts, stop = [], 0, None
    try:
        client = LocalOllamaAdapter(config, authorized=True)
        for candidate in bundle['candidates']:
            for index, agent in enumerate(agents):
                if attempts >= max_requests:
                    stop = 'GLOBAL_REQUEST_LIMIT'; break
                if digest(candidate['shared_payload']) != candidate['shared_input_fingerprint']:
                    raise ValueError('PAYLOAD_CHANGED_DURING_RUN')
                payload = {'department': agent['department'], **deepcopy(candidate['shared_payload'])}
                serialized_payload = json.dumps(payload, ensure_ascii=False)
                attempts += 1
                record = {'case_ref': candidate['case_ref'], 'agent_ref': 'Agent%03d' % (index + 1),
                    'agent_id': agent['agent_id'], 'model': agent['model'], 'version': agent['version'],
                    'role': agent['role'], 'department': agent['department'],
                    'patient_id': candidate['decision']['patient_id'], 'decision_id': candidate['decision']['decision_id'],
                    'source_ref': candidate['decision']['source_ref'], 'purpose': 'routing_training',
                    'split': candidate['decision']['split'], 'tokens': candidate['decision']['tokens'],
                    'configuration_fingerprint': digest(candidate['decision']),
                    'observation_fingerprint': digest(candidate['decision']['observation']),
                    'generation': deepcopy(candidate['decision']['generation']), 'shared_input_fingerprint': candidate['shared_input_fingerprint'],
                    'full_payload_fingerprint': digest(payload), 'system_fingerprint': candidate['decision']['generation']['prompt_fingerprints'][agent['agent_id']],
                    'serialized_payload_sha256': hashlib.sha256(serialized_payload.encode('utf-8')).hexdigest(),
                    'response_ref': candidate['case_ref'] + '_Agent%03d' % (index + 1),
                    'reference_status': 'NOT_REVIEWED', 'correct': None, 'correctness_status': 'NOT_REVIEWED',
                    'request_started_at': datetime.now(timezone.utc).isoformat(), 'execution_status': 'FAILED',
                    'response_contract_status': 'NOT_RECORDED', 'response_fingerprint': None}
                try:
                    raw = client.chat_json(specialty_system_prompt(agent['department']), serialized_payload,
                        model=agent['model'], schema=bundle['response_schema'])
                    answer = json.loads(raw)
                    record.update(execution_status='SUCCEEDED', response_contract_status='VALID', response=answer,
                        response_fingerprint=digest(answer), observed_model_identifier=config['model_identifier'])
                    unresolved = answer['certainty'] == 'unresolved' or answer['diagnosis'] == 'Unresolved'
                    record['assessment_status'] = 'UNRESOLVED' if unresolved else 'RESPONSE_RECORDED_NOT_CLINICALLY_VERIFIED'
                    record['correctness_status'] = 'NOT_EVALUABLE' if unresolved else 'NOT_REVIEWED'
                except LLMTimeoutError:
                    record.update(error_type='LLMTimeoutError', assessment_status='MISSING_TIMEOUT', correctness_status='NOT_EVALUABLE')
                except LLMResponseError:
                    record.update(error_type='LLMResponseError', response_contract_status='INVALID', assessment_status='INVALID_RESPONSE', correctness_status='NOT_EVALUABLE')
                except (ModelIdentityError, LLMRequestError) as exc:
                    record.update(error_type=type(exc).__name__, assessment_status='CLIENT_OR_IDENTITY_ERROR', correctness_status='NOT_EVALUABLE')
                    stop = 'IDENTITY_OR_CLIENT_ERROR'
                record['recorded_at'] = datetime.now(timezone.utc).isoformat()
                # Storage failure propagates and immediately stops further calls.
                atomic_json(path / (record['response_ref'] + '.json'), record)
                records.append(record)
                if stop:
                    break
            if stop:
                break
        metadata = client.public_metadata()
        report.update(status='executed_development_collection', generation_requests=metadata['request_count'],
            identity_checks=metadata['identity_check_count'], attempts=attempts, stop_reason=stop,
            execution_status_counts=dict(Counter(r['execution_status'] for r in records)),
            correctness_status_counts=dict(Counter(r['correctness_status'] for r in records)),
            complete_train_groups=0, training_readiness='BLOCKED', execution_blockers=[])
        # Public audit includes no response, evidence, raw identity or diagnosis.
        atomic_json(public_path, report)
        atomic_json(path / 'run_complete.json', {'status': 'INCOMPLETE' if stop else 'FINISHED',
                                               'attempts': attempts, 'generation_requests': metadata['request_count'], 'stop_reason': stop})
        return report
    except Exception:
        # Best-effort failure marker, without obscuring the original exception.
        try:
            atomic_json(path / 'run_failed.json', {'status': 'FAILED', 'attempts': attempts,
                'generation_requests': client.request_count if client is not None else 0,
                'persisted_outcomes': len(records), 'reason': 'COLLECTION_OR_STORAGE_FAILURE'})
        except OSError:
            pass
        raise
    finally:
        if client is not None:
            client.close()
