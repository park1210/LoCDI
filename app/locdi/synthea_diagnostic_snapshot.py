"""Opt-in source view and static diagnosis using the unchanged Core validator.

No reference labels are accepted by the model projection. Release grants are
separate from recorded-time eligibility; absent grants always fail closed.
This is a development snapshot, not a Hospital Episode or clinical clearance.
"""
from copy import deepcopy
import json
from unittest.mock import patch

from app.locdi.llm_setup import digest
from app.locdi.synthea_diagnosis_audit import proxy_minute, typed_objective_candidate


def shared_catalog(labels, *, expected_sha, path):
    from hashlib import sha256
    if sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('CATALOG_SOURCE_HASH_MISMATCH')
    actual = json.loads(path.read_text(encoding='utf-8'))
    if (labels != actual or not isinstance(labels, list) or len(labels) != 15
            or len(set(labels)) != 15 or any(not isinstance(x, str) or not x.strip() for x in labels)):
        raise ValueError('EXACT_SHARED15_REQUIRED')
    return deepcopy(labels)


def source_check(event, *, patient, visit, index_ms, cutoff, grant=None, mode='snapshot'):
    """Structural eligibility never establishes a clinical release decision."""
    if mode not in ('snapshot', 'sequential'):
        raise ValueError('EXPLICIT_RELEASE_MODE_REQUIRED')
    record = event.get('source_metadata', {}).get('source_record', {})
    if (event.get('patient_id') != patient or event.get('encounter_id') != visit
            or record.get('PATIENT') != patient or record.get('ENCOUNTER') != visit):
        return 'UNUSABLE_IDENTITY_MISMATCH'
    # Synthea stores the patient-reported pain scale under vital-signs. A
    # numeric storage type is insufficient to promote it to objective evidence.
    if str(event.get('target', {}).get('code')) == '72514-3':
        return 'PATIENT_REPORTED_ASK_ONLY_NOT_OBJECTIVE_TEST'
    if isinstance(event.get('evidence', {}).get('value'), bool):
        return 'UNUSABLE_BOOLEAN_NOT_NUMERIC_MEASUREMENT'
    if not typed_objective_candidate(event) or record.get('TYPE') != 'numeric':
        return 'UNUSABLE_NOT_NUMERIC_OBJECTIVE'
    if (str(record.get('CODE')) != str(event.get('target', {}).get('code'))
            or record.get('VALUE') != event['evidence'].get('value')
            or record.get('UNITS') != event['evidence'].get('units')
            or record.get('DATE') != event.get('timestamp')):
        return 'UNUSABLE_SOURCE_VALUE_MISMATCH'
    minute = proxy_minute(event, index_ms)
    if minute is None:
        return 'RELEASE_TIME_UNKNOWN'
    if minute > cutoff:
        return 'FUTURE_NOT_ELIGIBLE'
    # References/diagnoses in a result description are never silently approved.
    if event.get('event_type') != 'observations' or event['evidence'].get('kind') != 'recorded_observation':
        return 'UNUSABLE_SOURCE_KIND'
    if not isinstance(grant, dict):
        return 'RELEASE_NOT_REVIEWED'
    required = ('reviewer_private_ref', 'evidence_private_ref', 'reviewed_at', 'procedure_version')
    if (grant.get('source_fingerprint') != digest(event) or grant.get('mode') != mode
            or grant.get('patient_id') != patient or grant.get('visit_id') != visit
            or grant.get('release_reviewed') is not True
            or grant.get('diagnosis_leakage_reviewed') is not True
            or grant.get('contains_diagnosis') is not False
            or grant.get('recorded_time_proxy_accepted') is not True
            or not all(isinstance(grant.get(k), str) and grant[k].strip() for k in required)):
        return 'RELEASE_GRANT_INVALID'
    return 'ADDITIONAL_RELEASE_ELIGIBLE_DEVELOPMENT_ONLY'


def source_view(runtime_case, events, grants, *, enabled=False, memory_mode='none', voi_enabled=False,
                primary_router='er_only', cutoff=0, index_ms=0):
    """Copy-isolated sequential Test overlay, never pre-release objective values.

The overlay is applied after case_adapter.adapt_case. Existing Core
encounter_evidence remains responsible for request/available_minute gating.
No wrapper, likelihood model or original action menu is rewritten.
"""
    result = deepcopy(runtime_case)
    if not enabled:
        return result
    if memory_mode not in ('none', 'selective') or voi_enabled or primary_router != 'er_only':
        raise ValueError('SOURCE_VIEW_ONLY_B0_B2_ER_ONLY_VOI_OFF')
    identity = result['longitudinal']
    for event in events:
        state = source_check(event, patient=identity['patient_id'], visit=identity['visit_id'],
            index_ms=index_ms, cutoff=cutoff, grant=grants.get(event['event_id']), mode='sequential')
        if state != 'ADDITIONAL_RELEASE_ELIGIBLE_DEVELOPMENT_ONLY':
            raise ValueError(state)
        name = event['target']['description'] + ' [' + str(event['target']['code']) + ']'
        if name in result['available_tests']:
            raise ValueError('DUPLICATE_TEST_TARGET')
        result['available_tests'].append(name)
        result['current_encounter_events'].append({'event_id': 'source-view:' + digest(event),
            'action': 'Test', 'target': name, 'available_minute': proxy_minute(event, index_ms),
            'evidence': json.dumps({'kind': 'recorded_numeric_observation',
                'code': event['target']['code'], 'description': event['target']['description'],
                'value': event['evidence']['value'], 'units': event['evidence'].get('units'),
                'recorded_at': event['timestamp']}, ensure_ascii=False, sort_keys=True)})
    return result


def measurement_view(observed, names):
    """Opt-in representation only: no undisclosed pain value or new complaint."""
    result = deepcopy(observed)
    names = set(names)
    result['current_symptoms'] = [x for x in result.get('current_symptoms', []) if x not in names]
    result['reported_measurement_labels'] = [
        {'measurement_name': x, 'value': None, 'units': None, 'recorded_at': None,
         'status': 'VALUE_ONLY_AFTER_EXISTING_ASK'} for x in sorted(names)]
    result['chief_complaint'] = result.get('chief_complaint') or None
    result['symptom_narrative_status'] = 'NOT_RECORDED' if not result['current_symptoms'] else 'SOURCE_RECORDED'
    # Typed facts are derived only from answers already in the observed view.
    facts = []
    for item in result.get('history', []):
        try:
            answer = json.loads(item.get('answer', ''))
        except (ValueError, TypeError):
            continue
        for fact in answer.get('facts', []):
            facts.append({k: fact.get(k) for k in ('code', 'description', 'value', 'units', 'recorded_at')})
    result['observed_reported_measurements'] = facts
    return result


def safe_observation(observed):
    """Whitelisted saved public facts; never replay a prior model assessment."""
    keys = ('chief_complaint', 'demographics', 'arrival_info', 'vital_signs', 'elapsed_minutes',
            'health_state', 'current_symptoms', 'history', 'objective_findings')
    result = {k: deepcopy(observed[k]) for k in keys if k in observed}
    result.update(longitudinal_memory={'mode': 'none', 'records': []}, working_assessment='',
                  consults=[], interventions=[], pending_results=[], reassessment_pending=False,
                  reassessment_reason='')
    return result


class _Captured(BaseException):
    pass


def prepare_snapshot(observed, *, registry, catalog=None):
    """Capture the real HospitalLLMPolicy schema AND semantic validator offline.

    Static one-shot scope closes further actions (max_decisions=0), permitting
    an honest unresolved result even at D0. No fabricated Scheduler execution.
    The original evidence eligibility/department/certainty validator is used.
    """
    from app.core.hospital_case_runtime import HospitalCaseRuntime
    from app.core.hospital_llm_policy import HospitalLLMPolicy
    from app.core.agent_registry import AgentRegistry
    runtime = HospitalCaseRuntime(case_id='snapshot', case_data={
        'index_encounter_id': 'snapshot', 'current_encounter_events': [], 'available_tests': [],
        'doctor_visible_info': {k: deepcopy(observed.get(k, '')) for k in
                                ('chief_complaint', 'demographics', 'arrival_info')}},
        arrival_time_min=0, arrival_index=0)
    runtime.current_time_min = observed.get('elapsed_minutes', 0)
    runtime.current_symptoms = deepcopy(observed.get('current_symptoms', []))
    runtime.objective_findings = deepcopy(observed.get('objective_findings', []))
    runtime.question_history = deepcopy(observed.get('history', []))
    runtime.vital_signs = deepcopy(observed.get('vital_signs', {}))
    runtime.memory_context = {'mode': 'none', 'records': []}
    policy = HospitalLLMPolicy({'max_decisions': 0, 'max_questions': 0}, client=object())
    policy.agent_registry = AgentRegistry(deepcopy(registry))
    policy.routing_mode = 'er_only'
    policy.diagnosis_catalog = deepcopy(catalog)
    captured = {}
    def capture(*args, **kwargs):
        captured.update(kwargs)
        kwargs['audit'].append({'role': 'er', 'status': 'captured_offline'})
        raise _Captured()
    with patch('app.core.hospital_llm_policy.role_call', new=capture):
        try:
            policy._decision(runtime, 'er')
        except _Captured:
            pass
    if not captured or captured['payload']['allowed_actions'] != ['diagnose']:
        raise ValueError('STATIC_DIAGNOSIS_CONTRACT_CAPTURE_FAILED')
    if captured['payload']['observation'].get('working_assessment'):
        raise ValueError('PRIOR_MODEL_ASSESSMENT_LEAKAGE')
    return captured
