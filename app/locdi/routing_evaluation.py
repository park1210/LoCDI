"""External paired routing evaluator over unchanged Core. Never fits or generates."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import re

from app.core.agent_registry import AgentRegistry
from app.core.longitudinal_memory import timestamp
from app.core.routing_utility import observed_tokens
from app.locdi.llm_setup import digest
from app.locdi.memory_reference_evaluation import reject_hidden, ratio
from validation.routing_correctness_gate import to_policy_fit_rows

VERSION = 'locdi.routing_paired_evaluation.c4_08.v1'
REFERENCE_POLICY = 'locdi.independent_specialty_set.c4_08.v1'
OBSERVATION_KEYS = {'chief_complaint', 'current_symptoms', 'objective_findings', 'history',
                    'longitudinal_memory', 'demographics', 'elapsed_minutes', 'vital_signs'}
PRIVATE_LABEL_KEYS = {'gold_department', 'gold_departments', 'accepted_specialties',
                      'expected_specialty', 'correct', 'declared_label', 'reference_diagnosis'}


def observable(observation):
    if not isinstance(observation, dict) or set(observation) - OBSERVATION_KEYS:
        raise ValueError('OBSERVATION_WHITELIST_REQUIRED')
    def inspect(value):
        if isinstance(value, dict):
            if set(value) & PRIVATE_LABEL_KEYS: raise ValueError('REFERENCE_IN_ROUTING_INPUT')
            for v in value.values(): inspect(v)
        elif isinstance(value, list):
            for v in value: inspect(v)
        elif isinstance(value, str) and value.lstrip().startswith(('{', '[')):
            try: parsed = json.loads(value)
            except ValueError: return
            inspect(parsed)
    reject_hidden(observation); inspect(observation)
    return deepcopy(observation)


def split_check(cases, training_patients=()):
    sets = {name: set() for name in ('train', 'validation', 'test')}
    ids = set()
    for c in cases:
        if c['split'] not in sets: raise ValueError('INVALID_SPLIT')
        if c['case_ref'] in ids: raise ValueError('DUPLICATE_CASE_REFERENCE')
        ids.add(c['case_ref']); sets[c['split']].add(c['patient_id'])
    if any(sets[a] & sets[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('PATIENT_SPLIT_OVERLAP')
    if (sets['validation'] | sets['test']) & set(training_patients):
        raise ValueError('EVALUATION_PATIENT_IN_UTILITY_TRAIN')
    return {k: len(v) for k, v in sets.items()}


def paired_route(cases, registry, *, static_department, seed=42):
    """Fixed-department selection mirrors SequentialEngine's existing static branch.

    Adaptive always uses native abstention. There is no forced-choice variant.
    """
    registry.get('specialty', static_department)
    bound = sorted(a['department'] for a in registry.snapshot()['agents'] if a['role'] == 'specialty')
    results = []
    for c in cases:
        obs = observable(c['observation'])
        candidates = c['candidate_departments']
        if (not candidates or len(candidates) != len(set(candidates))
                or sorted(candidates) != sorted(set(candidates) & set(bound))
                or static_department not in candidates):
            raise ValueError('CANDIDATE_SET_INVALID')
        common = {'observation_fingerprint': digest(obs), 'memory_fingerprint': digest(obs.get('longitudinal_memory', {})),
                  'candidate_fingerprint': digest(sorted(candidates)), 'registry_fingerprint': digest(registry.snapshot()),
                  'source_fingerprint': c['source_fingerprint'], 'seed': seed}
        static = {'decision_status': 'selected', 'selected_specialty': static_department,
                  'candidate_departments': sorted(candidates), 'mode': 'static_development',
                  'reason_code': 'PREDECLARED_FIXED_SPECIALTY'}
        adaptive = registry.route(deepcopy(obs), sorted(candidates), mode='adaptive', seed=seed)
        if digest(obs) != common['observation_fingerprint']: raise ValueError('ROUTER_MUTATED_INPUT')
        adaptive['reason_code'] = ('MISSING_UTILITY' if registry.utility is None else
            'TIED_BEST_SCORE' if adaptive['tied_departments'] else
            'INSUFFICIENT_SUPPORT' if adaptive['decision_status'] == 'abstained' else
            'SELECTED' if adaptive['decision_status'] == 'selected' else 'UNAVAILABLE')
        for condition, decision in (('static_development', static), ('adaptive_native_abstention', adaptive)):
            results.append({'case_ref': c['case_ref'], 'split': c['split'], 'condition': condition,
                            **common, 'decision': deepcopy(decision)})
    return results


def validate_reference(cases, references):
    if not references: raise ValueError('INDEPENDENT_SPECIALTY_REFERENCE_REQUIRED')
    if references.get('policy_version') != REFERENCE_POLICY or references.get('cases_fingerprint') != digest(cases):
        raise ValueError('REFERENCE_POLICY_OR_COHORT_BINDING')
    rows = references.get('reviews', [])
    if [r.get('case_ref') for r in rows] != [c['case_ref'] for c in cases]:
        raise ValueError('FULL_REGISTERED_COHORT_REQUIRED')
    judged = {}
    for c, r in zip(cases, rows):
        if (r.get('source_fingerprint') != c['source_fingerprint']
                or r.get('observation_fingerprint') != digest(c['observation'])
                or r.get('candidate_fingerprint') != digest(sorted(c['candidate_departments']))):
            raise ValueError('REFERENCE_SNAPSHOT_BINDING')
        if r.get('status') == 'NOT_REVIEWED': continue
        reviewer = r.get('reviewer') or {}
        if (reviewer.get('kind') != 'HUMAN_CLINICAL_REVIEWER' or reviewer.get('independent') is not True
                or not isinstance(reviewer.get('private_ref'), str) or not reviewer['private_ref'].strip()
                or not isinstance(reviewer.get('qualification'), str) or not reviewer['qualification'].strip()
                or reviewer.get('procedure_version') != REFERENCE_POLICY):
            raise ValueError('INDEPENDENT_REVIEWER_REQUIRED')
        try:
            when = timestamp(reviewer['reviewed_at'])
            if not timestamp(c['decision_time']) <= when <= datetime.now(timezone.utc): raise ValueError()
        except (KeyError, TypeError, ValueError): raise ValueError('REVIEW_TIME_INVALID')
        if r.get('status') == 'NOT_EVALUABLE':
            if not r.get('not_evaluable_reason') or r.get('accepted_specialties') is not None:
                raise ValueError('NOT_EVALUABLE_REASON_REQUIRED')
            continue
        accepted = r.get('accepted_specialties')
        if (r.get('status') != 'REVIEWED' or not isinstance(accepted, list) or not accepted
                or len(accepted) != len(set(accepted)) or not set(accepted) <= set(c['candidate_departments'])
                or r.get('observation_reviewed') is not True or r.get('memory_scope_reviewed') is not True
                or r.get('visibility_proxy_reviewed') is not True or r.get('timing_status') != 'CURRENT_DECISION_REFERENCE_VALID'
                or not isinstance(r.get('clinical_evidence_rationale'), str) or not r['clinical_evidence_rationale'].strip()
                or not isinstance(r.get('reference_evidence_fingerprint'), str)
                or not re.fullmatch('[0-9a-f]{64}', r['reference_evidence_fingerprint'])):
            raise ValueError('REVIEWED_SPECIALTY_SET_REQUIRED')
        try:
            if timestamp(r['reference_available_at']) > timestamp(c['decision_time']): raise ValueError()
        except (KeyError, TypeError, ValueError): raise ValueError('REFERENCE_TIMING_INVALID')
        judged[c['case_ref']] = set(accepted)
    if not judged: raise ValueError('NO_INDEPENDENT_SCORABLE_REFERENCE')
    return judged


def metrics(rows, judged):
    """All rates expose denominators; abstentions never counted as incorrect selections."""
    selected = abstained = invalid = correct = wrong = 0
    for row in rows:
        if row['case_ref'] not in judged: continue
        d = row['decision']; status = d.get('decision_status'); choice = d.get('selected_specialty')
        if status == 'abstained' and choice is None: abstained += 1
        elif status == 'selected' and choice in d.get('candidate_departments', []):
            selected += 1
            if choice in judged[row['case_ref']]: correct += 1
            else: wrong += 1
        else: invalid += 1
    n = selected + abstained + invalid
    return {'registered_decisions': len(rows), 'reference_evaluable_decisions': n,
            'reference_coverage': ratio(n, len(rows)), 'selected_count': selected, 'abstained_count': abstained,
            'invalid_count': invalid, 'correct_selected_count': correct, 'incorrect_selected_count': wrong,
            'top1_accuracy_all_evaluable': ratio(correct, n), 'selective_accuracy': ratio(correct, selected),
            'abstention_rate': ratio(abstained, n), 'coverage': ratio(selected, n),
            'incorrect_routing_rate': ratio(wrong, n), 'invalid_decision_rate': ratio(invalid, n),
            'top_k_specialty_recall': 'NOT_SUPPORTED_NO_RANKED_SET_CONTRACT',
            'abstention_appropriateness': 'NOT_EVALUATED_SEPARATE_REFERENCE_REQUIRED'}


def validate_utility_provenance(registry, proof):
    """Re-run existing correctness gate; bind supplied artifact to admitted train rows. No fit."""
    if registry.utility is None: raise ValueError('VERIFIED_UTILITY_REQUIRED')
    if not proof: raise ValueError('UTILITY_GATE_PROVENANCE_REQUIRED')
    result = to_policy_fit_rows(proof['data'], **proof['options'])
    if result['rows'] is None: raise ValueError('EXISTING_CORRECTNESS_GATE_BLOCKED')
    expected = [{'patient_id': r['patient_id'], 'decision_id': r['decision_id'], 'agent_id': r['agent_id'],
                 'correct': r['correct'], 'tokens': observed_tokens(r['observation'])} for r in result['rows']]
    key = lambda r: (r['patient_id'], r['decision_id'], r['agent_id'])
    if sorted(expected, key=key) != sorted(registry.utility.artifact['samples'], key=key):
        raise ValueError('UTILITY_NOT_BOUND_TO_GATED_TRAIN_ROWS')
    if proof['options']['registry_manifest'] != registry.snapshot(): raise ValueError('UTILITY_REGISTRY_PROOF_MISMATCH')
    if registry.utility.artifact['min_samples'] != result['report']['positive_similarity_support']['min_samples']:
        raise ValueError('UTILITY_SUPPORT_POLICY_MISMATCH')
    return result['report']


def evaluate(cases, registry, references, *, static_department, utility_proof=None, test_only=False,
             evaluation_split='validation'):
    """Clinical run fails before routing if reference, split or utility provenance is absent."""
    judged = validate_reference(cases, references)
    split_check(cases, registry.utility.training_patients if registry.utility else ())
    if evaluation_split not in ('validation', 'test'): raise ValueError('EVALUATION_SPLIT_REQUIRED_NO_TRAIN_SCORING')
    evaluation_cases = [c for c in cases if c['split'] == evaluation_split]
    if not evaluation_cases or not any(c['case_ref'] in judged for c in evaluation_cases):
        raise ValueError('NO_SCORABLE_EVALUATION_CASES')
    if test_only:
        if (not cases or any(not c['patient_id'].startswith('TEST_ONLY') for c in cases)
                or registry.utility is None or any(not p.startswith('TEST_ONLY') for p in registry.utility.training_patients)):
            raise ValueError('TEST_ONLY_IDENTITY_REQUIRED')
    else:
        if any(c['patient_id'].startswith('TEST_ONLY') for c in cases): raise ValueError('TEST_ONLY_CANNOT_BE_CLINICAL')
        validate_utility_provenance(registry, utility_proof)
    traces = paired_route(evaluation_cases, registry, static_department=static_department)
    summary = {condition: metrics([r for r in traces if r['condition'] == condition], judged)
               for condition in ('static_development', 'adaptive_native_abstention')}
    paired = [{'case_ref': c['case_ref'], 'static_acceptable': traces[i * 2]['decision']['selected_specialty'] in judged[c['case_ref']],
               'adaptive_acceptable': traces[i * 2 + 1]['decision']['selected_specialty'] in judged[c['case_ref']]}
              for i, c in enumerate(evaluation_cases) if c['case_ref'] in judged]
    return {'schema_version': VERSION, 'evidence_level': 'TEST_ONLY' if test_only else 'INDEPENDENT_REVIEWED_SOURCE',
            'registered_cases': len(cases), 'evaluation_split': evaluation_split,
            'planned_evaluation_cases': len(evaluation_cases), 'other_split_cases_not_scored': len(cases) - len(evaluation_cases),
            'clinical_metrics': 'NOT_EVALUATED' if test_only else 'REVIEWED_SPECIALTY_SET_DESCRIPTIVE_ONLY',
            'metrics': summary, 'paired_outcomes': paired,
            'research_evaluation_ready': False}, traces
