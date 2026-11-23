"""Source-QA development utility gate, separate from clinical Correctness Gate.

Calls the existing Core formula only on full, actual, supported Train groups.
An abstention is never converted to a boolean failure or removed within a group.
"""
from collections import Counter
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import fit_utility, observed_tokens
from app.locdi.llm_setup import digest


def complete_groups(questions, bindings, outcomes):
    ids = {b['agent_id'] for b in bindings}
    if len(ids) != 5:
        raise ValueError('EXACTLY_FIVE_AGENTS_REQUIRED')
    index = {}
    qids = {q['question_id'] for q in questions}
    for row in outcomes:
        key = row['question_id'], row['agent_id']
        if key in index or key[0] not in qids or key[1] not in ids:
            raise ValueError('DUPLICATE_OR_UNREGISTERED_OUTCOME')
        expected = {'CORRECT_ANSWER': True, 'INCORRECT_ANSWER': False}.get(row['state'])
        if row.get('answer_correct') is not expected:
            raise ValueError('CORRECTNESS_SEMANTICS')
        index[key] = row
    result = []
    for q in questions:
        rows = [index.get((q['question_id'], a)) for a in sorted(ids)]
        collected = all(r and r['state'] in ('CORRECT_ANSWER', 'INCORRECT_ANSWER', 'UNRESOLVED') for r in rows)
        scorable = all(r and type(r['answer_correct']) is bool for r in rows)
        result.append({'question_id': q['question_id'], 'complete_collection': collected,
                       'complete_scorable': scorable, 'role': q['role']})
    return result


def observation(q):
    # Explicit static Source-QA projection, NOT a claim that the full vignette is
    # a triage chief complaint. Identical disclosed text enters every policy.
    return {'chief_complaint': '', 'current_symptoms': [],
            'objective_findings': [q['input']['question']], 'history': [],
            'longitudinal_memory': {'records': []}}


def fit_source_utility(questions, bindings, rows, registry, *, provenance_verified=False):
    if provenance_verified is not True:
        raise ValueError('ACTUAL_SOURCE_RESPONSE_PROVENANCE_REQUIRED')
    if any(q['role'] != 'train' for q in questions):
        raise ValueError('TRAIN_ONLY_FITTING')
    groups = complete_groups(questions, bindings, rows)
    good = {g['question_id'] for g in groups if g['complete_scorable']}
    if len(good) < 2:
        raise ValueError('MINIMUM_TWO_COMPLETE_TRAIN_GROUPS')
    qi = {q['question_id']: q for q in questions}
    usable = [r for r in rows if r['question_id'] in good]
    if any(not r.get('response_sha256') or not r.get('reference_fingerprint')
           or r.get('model_identifier') != next(b['model_identifier'] for b in bindings if b['agent_id'] == r['agent_id'])
           for r in usable):
        raise ValueError('ACTUAL_BINDING_OR_REFERENCE_MISSING')
    # Same label vector across agents provides no agent-selection signal.
    by_q = {qid: [r['answer_correct'] for r in usable if r['question_id'] == qid] for qid in good}
    if not any(len(set(v)) > 1 for v in by_q.values()):
        raise ValueError('NO_AGENT_DIFFERENTIATION')
    samples = [{'patient_id': qi[r['question_id']]['case_group'],
                'decision_id': r['question_id'], 'agent_id': r['agent_id'],
                'correct': r['answer_correct'], 'observation': observation(qi[r['question_id']]), 'split': 'train'}
               for r in usable]
    return fit_utility(samples, AgentRegistry(registry), min_samples=2)


def evaluate_policies(questions, bindings, rows, registry, artifact=None, er_rows=()):
    """Native abstention kept separate; no forced ER fallback or clinical labels."""
    reg = AgentRegistry(registry, utility_artifact=artifact) if artifact is not None else AgentRegistry(registry)
    departments = sorted(b['department'] for b in bindings)
    by_dept = {b['department']: b['agent_id'] for b in bindings}
    index = {(r['question_id'], r['agent_id']): r for r in rows}
    er = {r['question_id']: r for r in er_rows}
    traces = []
    for q in questions:
        obs = observation(q)
        for name, mode in [('Random', 'random'), ('RuleSimilarity', 'rule'), ('Adaptive', 'adaptive')]:
            if mode == 'adaptive' and artifact is None:
                traces.append({'question_id': q['question_id'], 'policy': name, 'decision_status': 'unavailable',
                               'state': 'NOT_ATTEMPTED', 'answer_correct': None})
                continue
            decision = reg.route(obs, departments, mode=mode, seed=42)
            selected = decision['selected_specialty']
            r = index.get((q['question_id'], by_dept.get(selected)))
            traces.append({'question_id': q['question_id'], 'policy': name,
                           'decision_status': decision['decision_status'],
                           'selected_specialty': selected, 'decision': decision,
                           'state': r['state'] if r else 'NOT_ATTEMPTED',
                           'answer_correct': r['answer_correct'] if r else None})
        fixed = index.get((q['question_id'], by_dept[departments[0]]))
        for name, r in [('StaticFirstAlphabetical', fixed), ('EROnly', er.get(q['question_id']))]:
            traces.append({'question_id': q['question_id'], 'policy': name, 'decision_status': 'selected' if r else 'unavailable',
                           'state': r['state'] if r else 'NOT_ATTEMPTED', 'answer_correct': r['answer_correct'] if r else None})
    summaries = {}
    for name in sorted({t['policy'] for t in traces}):
        rs = [r for r in traces if r['policy'] == name]
        supported = [r for r in rs if type(r['answer_correct']) is bool]
        correct = sum(r['answer_correct'] is True for r in rs)
        summaries[name] = {'registered': len(rs), 'correct_supported': correct,
            'correct_supported_yield': correct / len(rs) if rs else None,
            'supported_denominator': len(supported), 'conditional_source_agreement': correct / len(supported) if supported else None,
            'selection_coverage': sum(r['decision_status'] == 'selected' for r in rs) / len(rs) if rs else None,
            'decision_states': dict(Counter(r['decision_status'] for r in rs)),
            'clinical_accuracy': 'NOT_EVALUATED'}
    return {'summary': summaries, 'traces': traces, 'comparison_scope': 'static Source-QA agent selection; no Hospital state change'}
