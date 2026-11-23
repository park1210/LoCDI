"""Episode-isolated collection for the frozen Memory comparison.

This is a technical collection policy, not a clinical validator or a new
retriever. A failed B0 never removes that patient's registered B2. No retries.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from app.locdi.llm_setup import digest
from app.locdi.current_ablation_runner import GenerationBudget, _BUDGET, _validate_result, features_for

VERSION = 'locdi.memory_episode_isolation.v1'
STATUSES = {'COMPLETED', 'CONTRACT_FAILED', 'TIMEOUT', 'CONTEXT_BLOCKED', 'NOT_ATTEMPTED', 'INCOMPLETE'}


class SharedRecoveryStop(ValueError):
    """Source, identity, approval or global resource boundary: stop all calls."""


def schedule(panel):
    if not panel or len({p['patient_ref'] for p in panel}) != len(panel) or len({p['case_ref'] for p in panel}) != len(panel):
        raise ValueError('UNIQUE_NONEMPTY_PATIENT_PANEL_REQUIRED')
    return [{'episode_ordinal': 2*i+j+1, 'patient_ordinal': i+1, 'case_ref': p['case_ref'],
             'condition': condition} for i, p in enumerate(panel) for j, condition in enumerate(('B0', 'B2'))]


def citation_audit(request, response):
    """Structural citation eligibility only; never infer clinical support."""
    user = json.loads(request['messages'][1]['content'])
    value = json.loads(response['message']['content'])
    records = user.get('evidence_records', [])
    indices = value.get('evidence_indices', [])
    valid = isinstance(indices, list) and all(type(i) is int and 0 <= i < len(records) for i in indices)
    eligible = user.get('supported_diagnosis_eligible_indices', [])
    needs = value.get('action') == 'diagnose' and value.get('certainty') == 'supported'
    return {'action': value.get('action'), 'certainty': value.get('certainty'),
        'evidence_count': len(records), 'citation_indices': indices, 'eligible_objective_indices': eligible,
        'cited_kinds': [records[i].get('kind') for i in indices] if valid else None,
        'citation_in_range': valid, 'eligible_objective_cited': bool(set(indices) & set(eligible)) if valid else False,
        'error_code': 'EVIDENCE_INDEX_OUT_OF_RANGE' if not valid else
            'SUPPORTED_DIAGNOSIS_WITHOUT_ELIGIBLE_OBJECTIVE_CITATION' if needs and not set(indices).intersection(eligible) else None,
        'clinical_support_verified': False, 'response_rewritten': False}


def classify_failure(error_names):
    names = ' '.join(str(x) for x in error_names)
    if 'Timeout' in names or 'timeout' in names:
        return 'TIMEOUT'
    if 'Context' in names or 'CONTEXT' in names:
        return 'CONTEXT_BLOCKED'
    return 'CONTRACT_FAILED'


def verify_delivered_memory(case, memory):
    """Read-only Source/patient/visit/cutoff facts, not clinical relevance."""
    records=memory.get('records',[]); failures=[]
    source={r['memory_id']:r for visit in case['agent_visible']['past_visits'] for r in visit['records']}
    def date(value):return datetime.fromisoformat(value.replace('Z','+00:00'))
    if len(records)>5 or len({r['memory_id'] for r in records})!=len(records):failures.append('TOP5_OR_UNIQUE_ID')
    for record in records:
        original=source.get(record['memory_id'])
        if original is None or any(record.get(k)!=v for k,v in original.items()):failures.append('SOURCE_VALUE_MISMATCH')
        if record.get('patient_id')!=case['patient_id']:failures.append('PATIENT_MISMATCH')
        if record.get('visit_id')==case['agent_visible']['index_encounter_id']:failures.append('CURRENT_VISIT_MEMORY')
        try:
            if (date(record['available_at'])>date(memory['as_of']) or date(record['observed_at'])>date(memory['as_of'])
                or date(record['visit_started_at'])>=date(memory['as_of'])):failures.append('FUTURE_OR_VISIT_CUTOFF')
        except (KeyError,ValueError,TypeError):failures.append('TIMESTAMP_NOT_VERIFIABLE')
        if record.get('status_as_of_visit')=='superseded':failures.append('SUPERSEDED_RECORD')
    return {'status':'FAIL' if failures else 'PASS','record_occurrences':len(records),'failure_codes':sorted(set(failures)),
            'clinical_relevance':'NOT_REVIEWED','timestamp_basis':'EXISTING_DEVELOPMENT_SOURCE_PROXY'}


def collect(panel, run_one, *, trace_path, total_cap, per_episode_cap, guard, audit_failure):
    """Reuse an actual existing backend, preserving each failed Episode.

    A backend is required to reserve every POST through the existing ContextVar
    budget. Production CLI constructs the native real Runner backend itself.
    """
    registered = schedule(panel)
    if total_cap < len(registered) or per_episode_cap < 1:
        raise ValueError('EXPLICIT_BOUNDED_REGISTERED_BUDGET_REQUIRED')
    budget = GenerationBudget(total_cap, per_episode_cap, resource_guard=guard)
    stopped = None; result_rows = []; trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open('x', encoding='utf-8', newline='\n') as stream:
        for task in registered:
            row = {**task, 'status': 'NOT_ATTEMPTED', 'request_count': 0, 'result': None,
                   'error_type': None, 'failure_audit': None, 'global_stop_reason': stopped}
            if stopped is None:
                start_count = budget.used; budget.episode_used = 0; token = _BUDGET.set(budget)
                try:
                    guard()
                    if budget.used >= budget.limit:
                        raise SharedRecoveryStop('GLOBAL_GENERATION_LIMIT')
                    result = run_one({'case_ref': task['case_ref']}, features_for(task['condition']))
                    _validate_result(result, budget.used-start_count)
                    row['result'] = deepcopy(result)
                    row['status'] = {'completed': 'COMPLETED', 'incomplete': 'INCOMPLETE', 'failed': 'CONTRACT_FAILED'}[result['status']]
                    guard()
                except SharedRecoveryStop as exc:
                    stopped = str(exc)
                    row.update(status='CONTRACT_FAILED' if budget.used > start_count else 'NOT_ATTEMPTED',
                               error_type=type(exc).__name__, global_stop_reason=stopped)
                except Exception as exc:
                    # Preserve the failure; run the registered B2 and next
                    # patient unless audit identifies a shared integrity error.
                    row['error_type'] = type(exc).__name__
                    row['status'] = classify_failure([type(exc).__name__])
                finally:
                    _BUDGET.reset(token)
                    row['request_count'] = budget.used-start_count
                if row['status'] not in ('COMPLETED', 'INCOMPLETE', 'NOT_ATTEMPTED'):
                    detail = audit_failure(task, row['error_type'])
                    row['failure_audit'] = detail
                    row['status'] = classify_failure(detail.get('error_types', []) + [row['error_type']])
                    if detail.get('shared_integrity_failure'):
                        stopped = 'SHARED_IDENTITY_OR_SOURCE_FAILURE'
                        row['global_stop_reason'] = stopped
                row['global_stop_reason'] = stopped
            row['row_fingerprint'] = digest(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno()); result_rows.append(row)
    return {'rows': result_rows, 'generation_count': budget.used, 'global_stop_reason': stopped,
            'registered_episodes': len(registered), 'ended_at': datetime.now(timezone.utc).isoformat()}


def aggregate(panel, saved_rows):
    registered = schedule(panel)
    if len(saved_rows) != len(registered):
        raise ValueError('REGISTERED_DENOMINATOR_MISSING')
    pi = {p['case_ref']: p for p in panel}; outcomes = []
    for task, row in zip(registered, saved_rows):
        if any(row.get(k) != v for k, v in task.items()) or row.get('status') not in STATUSES:
            raise ValueError('SCHEDULE_OR_STATUS_CHANGED')
        if row.get('row_fingerprint') != digest({k: v for k, v in row.items() if k != 'row_fingerprint'}):
            raise ValueError('TRACE_FINGERPRINT_MISMATCH')
        ref = pi[row['case_ref']]['reference']; result = row.get('result') or {}; pred = result.get('prediction') or {}
        complete = row['status'] == 'COMPLETED'
        supported = complete and pred.get('certainty') == 'supported'
        exact = pred.get('diagnosis') in ref['accepted_diagnoses'] if supported else None
        dept = pred.get('department') in ref['accepted_departments'] if complete and isinstance(pred.get('department'), str) else None
        outcomes.append({**task, 'status': row['status'], 'certainty': pred.get('certainty') if complete else None,
            'source_diagnosis_exact': exact, 'source_department_agreement': dept,
            'verified_concept_agreement': None, 'concept_status': 'NOT_EVALUATED',
            'request_count': row['request_count'], 'action_counts': result.get('action_counts'),
            'clinical_accuracy': 'NOT_EVALUATED'})
    pairs = []
    for i in range(len(panel)):
        b0, b2 = outcomes[2*i:2*i+2]
        observed = b0['status'] == b2['status'] == 'COMPLETED'
        scorable = all(type(x['source_diagnosis_exact']) is bool for x in (b0,b2))
        if not observed: transition = 'one_or_both_unobserved'
        elif not scorable: transition = 'one_or_both_unresolved_not_scorable'
        else:
            transition = {(False,True):'B0_incorrect_to_B2_correct', (True,False):'B0_correct_to_B2_incorrect',
                          (True,True):'both_correct', (False,False):'both_incorrect'}[b0['source_diagnosis_exact'],b2['source_diagnosis_exact']]
        pairs.append({'patient_ordinal': i+1, 'B0_status': b0['status'], 'B2_status': b2['status'],
            'B0_source_exact': b0['source_diagnosis_exact'], 'B2_source_exact': b2['source_diagnosis_exact'],
            'B0_department_match': b0['source_department_agreement'], 'B2_department_match': b2['source_department_agreement'],
            'complete_pair': observed, 'scorable_source_pair': scorable, 'source_transition': transition,
            'verified_concept_agreement': None})
    conditions = {}
    def ratio(n,d): return {'numerator':n,'denominator':d,'value':n/d if d else None,'status':'DEFINED' if d else 'NOT_APPLICABLE'}
    for condition in ('B0','B2'):
        rs = [r for r in outcomes if r['condition']==condition]
        supported = [r for r in rs if type(r['source_diagnosis_exact']) is bool]
        ds = [r for r in rs if type(r['source_department_agreement']) is bool]
        correct = sum(r['source_diagnosis_exact'] is True for r in rs)
        conditions[condition] = {'registered':len(rs), 'states':dict(Counter(r['status'] for r in rs)),
            'supported':len(supported), 'unresolved':sum(r['certainty']=='unresolved' for r in rs),
            'source_exact_supported':ratio(correct,len(supported)),
            'source_exact_success_yield_registered':ratio(correct,len(rs)),
            'source_department_agreement':ratio(sum(r['source_department_agreement'] is True for r in ds),len(ds)),
            'generation_requests':sum(r['request_count'] for r in rs),
            'verified_concept_agreement':'NOT_EVALUATED', 'clinical_accuracy':'NOT_EVALUATED'}
    scorable = [p for p in pairs if p['scorable_source_pair']]
    difference = sum(int(p['B2_source_exact'])-int(p['B0_source_exact']) for p in scorable)/len(scorable) if scorable else None
    return {'registered_patients':len(panel),'registered_episodes':len(saved_rows),'conditions':conditions,
        'complete_pairs':sum(p['complete_pair'] for p in pairs),'source_scorable_pairs':len(scorable),
        'transitions':dict(Counter(p['source_transition'] for p in pairs)),
        'paired_source_exact_B2_minus_B0':difference,'paired_denominator':len(scorable),
        'pairs':pairs,'episodes':outcomes,'failure_selection_bias_warning':'complete-pair conditional effect; keep full registered panel alongside',
        'algorithm_changes':0,'clinical_accuracy':'NOT_EVALUATED'}
