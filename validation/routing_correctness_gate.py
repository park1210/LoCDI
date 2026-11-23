"""External policy gate BEFORE the unchanged routing training validator.

Accepts private caller-supplied human reviews; never infers clinical verdicts.
Legacy APIs remain unchanged. New consumers must use this gate/wrapper.
"""
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
import json
import re

from jsonschema import Draft202012Validator
from app.locdi.llm_setup import ROOT, digest
from app.core.routing_utility import observed_tokens, jaccard_similarity
from validation.routing_training_data import (validate_training_data, validate_split_manifest, to_fit_rows,
                                             _time, _text, _review, _index, _leaves)

POLICY_PATH = ROOT/'configs/evaluation/routing_policy_v1.json'
SCHEMA_PATH = ROOT/'schemas/routing_review_results_v1.schema.json'
POLICY = json.loads(POLICY_PATH.read_text(encoding='utf-8'))
REVIEW_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
TIMING_STATES = {'CURRENT_DECISION_REFERENCE_VALID', 'RETROSPECTIVE_OUTCOME_ONLY',
                 'REFERENCE_TIMING_NOT_REVIEWED', 'REFERENCE_REJECTED'}


def empty_review_results(purpose='routing_training'):
    return {'schema_version': 'locdi.routing_review_results.v1', 'purpose': purpose,
            'policy_fingerprint': digest(POLICY), 'reference_reviews': [], 'response_reviews': []}


def _human(review, agent_ids, procedure, *, after=None):
    try:
        return (_review(review) and review.get('reviewer_kind') == 'HUMAN_CLINICAL_REVIEWER'
                and review.get('independent') is True and review['reviewer_ref'] not in agent_ids
                and _text(review.get('reviewer_qualification'))
                and review['procedure_version'] == procedure
                and (after is None or _time(review['reviewed_at']) >= _time(after)))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _axis_valid(axis, name):
    if axis.get('policy_id') != POLICY['policies'][name]['policy_id']:
        return False
    status = axis.get('status')
    if status == 'REVIEWED':
        return type(axis.get('verdict')) is bool and _text(axis.get('clinical_evidence_rationale')) and axis.get('not_evaluable_reason') is None
    if status == 'NOT_EVALUABLE':
        return axis.get('verdict') is None and _text(axis.get('not_evaluable_reason'))
    return status == 'NOT_REVIEWED' and axis.get('verdict') is None


def _current_reference_content(content, decision_at):
    try:
        return (set(content) == {'diagnosis','evidence_ref','available_at','time_scope'}
                and _text(content['diagnosis']) and _text(content['evidence_ref'])
                and content['time_scope'] == 'CURRENT_DECISION' and _time(content['available_at']) <= decision_at)
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def validate_evaluation_gate(data, *, review_results, decision_contexts, registry_manifest,
                             source_catalog, label_catalog, holdout_patient_ids, dataset_fingerprint,
                             evaluation_target='diagnosis_correctness', min_samples=2, allow_test_only=False):
    """Count intact 5-Agent groups, then invoke legacy validator, never fit.

    Positive support uses supplied VALIDATION decisions as queries, TRAIN only
    as samples. It is query-specific support, not clinical performance.
    Every input remains unchanged; no deletion or boolean imputation occurs.
    """
    report = {'schema_version': 'locdi.correctness_gate.v1', 'evaluation_target': evaluation_target,
        'policy_ids': {k:v['policy_id'] for k,v in POLICY['policies'].items()}, 'policy_fingerprint': digest(POLICY),
        'source_reference_ready': False, 'clinical_review_ready': False, 'response_policy_ready': False,
        'complete_train_groups': 0, 'validation_group_count': 0, 'test_group_count': 0,
        'policy_complete_train_groups': 0, 'policy_validation_group_count': 0,
        'positive_similarity_support': {'status': 'NOT_CHECKED', 'min_samples': min_samples, 'queries': []},
        'utility_fit_ready': False, 'development_contract_ready': False,
        'research_evaluation_ready': False, 'training_readiness': 'BLOCKED',
        'contract_status': 'BLOCKED', 'legacy_validator': 'NOT_RUN_GATE_BLOCKED',
        'blocker_codes': [], 'groups': [], 'response_states': {}, 'timing_states': {},
        'new_llm_calls': 0, 'utility_fit_count': 0, 'clinical_gold_generated': 0}
    codes = report['blocker_codes']
    try:
        if evaluation_target != 'diagnosis_correctness':
            codes.append('TARGET_NOT_SUPPORTED_FOR_ACCURACY_FITTING')
        if type(min_samples) is not int or min_samples < 1:
            codes.append('MIN_SAMPLES_INVALID')
        if not isinstance(data, dict) or data.get('purpose') not in {'routing_training', 'TEST_ONLY'}:
            raise ValueError()
        if data['purpose'] == 'TEST_ONLY' and allow_test_only is not True:
            codes.append('TEST_ONLY_NOT_AUTHORIZED')
        if (not Draft202012Validator(REVIEW_SCHEMA).is_valid(review_results)
                or review_results.get('policy_fingerprint') != digest(POLICY)
                or review_results.get('purpose') != data['purpose']):
            codes.append('REVIEW_RESULTS_SCHEMA_POLICY_OR_PURPOSE_INVALID')
            raise ValueError()
        agents = {a['agent_id'] for a in registry_manifest['agents'] if a['role'] == 'specialty'}
        all_agents = {a['agent_id'] for a in registry_manifest['agents']}
        if len(agents) != 5:
            codes.append('FIVE_SPECIALTY_REGISTRY_REQUIRED')
        if data.get('registry') != registry_manifest:
            codes.append('REGISTRY_SNAPSHOT_MISMATCH')
        split = validate_split_manifest(data['split_manifest'], holdout_patient_ids=holdout_patient_ids, dataset_fingerprint=dataset_fingerprint)
        codes.extend(split['blocker_codes'])
        decisions = _index(data['decisions'], 'decision_id')
        contexts = _index(decision_contexts, 'decision_id')
        outcomes = _index(data['outcomes'], 'response_ref')
        judgments = _index(data['judgments'], 'response_ref')
        sources = _index(source_catalog, 'source_ref'); labels = _index(label_catalog, 'label_ref')
        ref_reviews = _index(review_results['reference_reviews'], 'decision_id')
        response_reviews = _index(review_results['response_reviews'], 'response_ref')
        assignments = _index(data['split_manifest']['decisions'], 'decision_id')
        if set(contexts) != set(decisions) or set(assignments) != set(decisions):
            codes.append('DECISION_CONTEXT_OR_SPLIT_SCOPE_MISMATCH')
        if set(ref_reviews) - set(decisions) or set(response_reviews) - set(outcomes) or set(judgments) - set(outcomes):
            codes.append('ORPHAN_REVIEW_OR_JUDGMENT')
        if any(o.get('decision_id') not in decisions for o in outcomes.values()):
            codes.append('ORPHAN_RESPONSE')
        global_ok = not codes
        source_ok_all, clinical_ok_all, policy_ok_all = bool(decisions), bool(outcomes), bool(outcomes)
        state_counts, timing_counts = Counter(), Counter()
        for index, (did,d) in enumerate(decisions.items(), 1):
            source_codes, clinical_codes, policy_codes = [], [], []
            ctx = contexts.get(did, {}); source = sources.get(d.get('source_ref'), {})
            rr = ref_reviews.get(did)
            public_case = ctx.get('case_ref', '')
            if not isinstance(public_case, str) or not re.fullmatch('Candidate[0-9]{3}', public_case):
                public_case = 'Decision%03d' % index
            if any(ctx.get(k) != d.get(k) for k in ('patient_id','visit_id','decision_id')) or ctx.get('source_fingerprint') != source.get('source_fingerprint'):
                source_codes.append('CONTEXT_SOURCE_DECISION_LINK_INVALID')
            if assignments.get(did) != {k:d[k] for k in ('patient_id','decision_id','split')}:
                policy_codes.append('DECISION_SPLIT_LINK_INVALID')
            ledger = ctx.get('visibility', {})
            if (set(ledger) != set(_leaves(d['observation']))
                    or any(type(x.get('available_minute')) is not int or not 0 <= x['available_minute'] <= d['minute'] for x in ledger.values())
                    or source.get('visibility') != {p:r.get('available_minute') for p,r in ledger.items()}):
                source_codes.append('VISIBILITY_CONTEXT_INVALID')
            proxy = any('PROXY' in str(r.get('basis', '')) or r.get('actual_release_status') == 'NOT_RECORDED' for r in ledger.values())
            candidates = ctx.get('reference_candidates', {})
            future_days = []
            decision_at = _time(ctx['encounter_started_at']) + timedelta(minutes=d['minute'])
            for s in candidates.get('source_condition_rows', []):
                record = s['source_record']
                if record.get('PATIENT') != d['patient_id'] or record.get('ENCOUNTER') != d['visit_id']:
                    source_codes.append('CONDITION_ENCOUNTER_LINK_INVALID')
                future_days.append((date.fromisoformat(record['START']) - decision_at.date()).days)
            timing = rr.get('timing_status') if rr else 'REFERENCE_TIMING_NOT_REVIEWED'
            if timing not in TIMING_STATES:
                timing = 'REFERENCE_TIMING_NOT_REVIEWED'
            timing_counts[timing] += 1
            if rr is None:
                source_codes.extend(['REFERENCE_REVIEW_MISSING', 'REFERENCE_TIMING_NOT_REVIEWED', 'OBSERVATION_MEMORY_REVIEW_MISSING'])
            else:
                expected = {'case_ref': ctx['case_ref'], **{k:d[k] for k in ('patient_id','visit_id','decision_id','source_ref')},
                    'source_fingerprint': source.get('source_fingerprint'), 'observation_fingerprint': digest(d['observation']),
                    'memory_fingerprint': digest(d['observation'].get('longitudinal_memory', {})),
                    'condition_candidates_fingerprint': digest(candidates)}
                if any(rr.get(k) != v for k,v in expected.items()):
                    source_codes.append('REFERENCE_INPUT_FINGERPRINT_OR_LINK_INVALID')
                ref_human = _human(rr['review'], all_agents, POLICY['reference_procedure_version'])
                if not ref_human or not _text(rr['clinical_evidence_rationale']) or rr['response_blinded'] is not True:
                    source_codes.append('INDEPENDENT_HUMAN_REFERENCE_REVIEW_REQUIRED')
                if rr['uncertainty_reason'] is not None:
                    source_codes.append('REFERENCE_UNCERTAINTY_UNRESOLVED')
                if source.get('review') != rr['review']:
                    source_codes.append('SOURCE_REVIEW_LINK_INVALID')
                scope = rr['input_scope']
                if not all(scope[k] is True for k in ('observation_reviewed','memory_reviewed','visibility_reviewed')):
                    source_codes.append('OBSERVATION_MEMORY_VISIBILITY_NOT_REVIEWED')
                if proxy and not (scope['proxy_reviewed'] is True and scope['proxy_fitting_approved'] is True and _text(scope['proxy_rationale']) and ref_human):
                    source_codes.append('UNREVIEWED_PROXY_NOT_APPROVED_FOR_FITTING')
                if timing != 'CURRENT_DECISION_REFERENCE_VALID':
                    source_codes.append(timing)
                if max(future_days, default=0) > 0 and rr['reference_basis'] == 'SOURCE_CONDITION':
                    source_codes.append('FUTURE_SOURCE_CONDITION_NOT_CURRENT_REFERENCE')
            if proxy and rr is None:
                source_codes.append('UNREVIEWED_PROXY_NOT_APPROVED_FOR_FITTING')
            label = labels.get(rr.get('reference_ref')) if rr else None
            if not label:
                source_codes.append('INDEPENDENT_REFERENCE_MISSING')
            else:
                content = label.get('reference_content', {})
                if (label.get('independent_reference') is not True or label.get('clinical_review_status') != 'REVIEWED'
                        or label.get('review') != rr['review'] or label.get('label_fingerprint') != digest(content)
                        or rr['reference_fingerprint'] != label.get('label_fingerprint')
                        or any(label.get(k) != d[k] for k in ('patient_id','decision_id'))):
                    source_codes.append('INDEPENDENT_REFERENCE_CONTENT_OR_REVIEW_INVALID')
                if not _current_reference_content(content, decision_at):
                    source_codes.append('REFERENCE_TIME_SCOPE_OR_AVAILABILITY_INVALID')
            rows = [o for o in outcomes.values() if o.get('decision_id') == did]
            originals = ctx.get('original_response_fingerprints', {})
            if set(originals) != {r['response_ref'] for r in rows}:
                policy_codes.append('ORIGINAL_RESPONSE_MANIFEST_SCOPE_MISMATCH')
            if len(rows) != 5 or {r.get('agent_id') for r in rows} != agents:
                policy_codes.append('FIVE_AGENT_EVALUABLE_GROUP_REQUIRED')
            for row in rows:
                rc, pc = [], []
                answer = row.get('response', {}); r = response_reviews.get(row['response_ref'])
                valid_answer = (isinstance(answer, dict) and set(answer) == {'action','diagnosis','certainty','rationale'}
                    and answer.get('action') == 'diagnose' and _text(answer.get('diagnosis')) and _text(answer.get('rationale'))
                    and answer.get('certainty') in {'supported','unresolved'})
                if (not valid_answer or row.get('execution_status') != 'SUCCEEDED' or row.get('response_contract_status') != 'VALID'):
                    pc.append('RESPONSE_NOT_EVALUABLE')
                if answer.get('diagnosis') not in ctx.get('diagnosis_catalog', []) + ['Unresolved']:
                    pc.append('DIAGNOSIS_OUTSIDE_REVIEWED_CATALOG')
                if row.get('response_fingerprint') != digest(answer) or originals.get(row['response_ref']) != row.get('response_fingerprint'):
                    pc.append('ORIGINAL_RESPONSE_FINGERPRINT_MISMATCH')
                unresolved = answer.get('certainty') == 'unresolved' or answer.get('diagnosis') == 'Unresolved'
                state_counts['UNRESOLVED' if unresolved else 'SUPPORTED' if valid_answer else 'INVALID'] += 1
                if answer.get('diagnosis') == 'Unresolved' and answer.get('certainty') == 'supported':
                    pc.append('DIAGNOSIS_CERTAINTY_CONTRADICTION')
                if unresolved:
                    pc.append('UNRESOLVED_EXCLUDED_FROM_DIAGNOSIS_FITTING')
                    j = judgments.get(row['response_ref'], {})
                    if j.get('correct') is not None or row.get('correct') is not None:
                        pc.append('UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN')
                if r is None:
                    rc.append('RESPONSE_CLINICAL_REVIEW_MISSING')
                    pc.append('RESPONSE_POLICY_REVIEW_MISSING')
                else:
                    expected = {k:row[k] for k in ('patient_id','decision_id','agent_id','response_ref','response_fingerprint')}
                    expected.update(case_ref=ctx['case_ref'], reference_ref=rr.get('reference_ref') if rr else None,
                                    reference_fingerprint=rr.get('reference_fingerprint') if rr else None)
                    if any(r.get(k) != v for k,v in expected.items()):
                        rc.append('RESPONSE_REVIEW_REFERENCE_OR_IDENTITY_LINK_INVALID')
                    if (not _human(r['review'], all_agents, POLICY['response_procedure_version'], after=row['recorded_at'])
                            or not rr or _time(r['review']['reviewed_at']) < _time(rr['review']['reviewed_at'])):
                        rc.append('RESPONSE_REVIEW_INDEPENDENCE_PROCEDURE_OR_TIME_INVALID')
                    for axis_name, axis in r['axes'].items():
                        if not _axis_valid(axis, axis_name):
                            pc.append('EVALUATION_TARGET_OR_VERDICT_SEMANTICS_INVALID')
                    axes = r['axes']; diag = axes['diagnosis_correctness']
                    if axes['routing_success']['status'] == 'REVIEWED':
                        pc.append('ROUTING_SUCCESS_REFERENCE_CONTRACT_UNAVAILABLE')
                    if unresolved:
                        if diag['verdict'] is not None or diag['status'] != 'NOT_EVALUABLE' or not _text(diag['not_evaluable_reason']):
                            pc.append('UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN')
                        if axes['abstention_appropriateness']['status'] != 'REVIEWED':
                            rc.append('ABSTENTION_INDEPENDENT_REVIEW_NOT_COMPLETED')
                    else:
                        if diag['status'] != 'REVIEWED' or type(diag['verdict']) is not bool:
                            rc.append('DIAGNOSIS_CORRECTNESS_NOT_REVIEWED')
                        judgment = judgments.get(row['response_ref'], {})
                        if (type(judgment.get('correct')) is not bool or judgment['correct'] != diag['verdict']
                                or judgment.get('review') != r['review'] or judgment.get('label_ref') != r['reference_ref']
                                or judgment.get('label_fingerprint') != r['reference_fingerprint']):
                            pc.append('LEGACY_JUDGMENT_POLICY_LINK_INVALID')
                clinical_codes.extend(rc); policy_codes.extend(pc)
            source_ok_all &= not source_codes; clinical_ok_all &= not clinical_codes
            policy_ok_all &= not policy_codes
            all_codes = sorted(set(source_codes + clinical_codes + policy_codes))
            ready = global_ok and not all_codes
            if ready:
                field = {'train':'complete_train_groups','validation':'validation_group_count','test':'test_group_count'}[d['split']]
                report[field] += 1
            report['groups'].append({'case_ref': public_case, 'split': d['split'], 'stored_responses': len(rows),
                'diagnosis_evaluable_complete': ready, 'reference_timing_status': timing,
                'source_condition_start_after_decision_days': max(future_days, default=0), 'blocker_codes': all_codes})
            codes.extend(all_codes)
        report['policy_complete_train_groups'] = report['complete_train_groups']
        report['policy_validation_group_count'] = report['validation_group_count']
        report.update(source_reference_ready=source_ok_all and global_ok,
                      clinical_review_ready=clinical_ok_all and global_ok,
                      response_policy_ready=policy_ok_all and global_ok,
                      response_states=dict(state_counts), timing_states=dict(timing_counts))
        if not report['complete_train_groups']:
            codes.append('NO_POLICY_COMPLETE_TRAIN_GROUP')
        if not codes:
            options = dict(registry_manifest=registry_manifest, source_catalog=source_catalog, label_catalog=label_catalog,
                holdout_patient_ids=holdout_patient_ids, dataset_fingerprint=dataset_fingerprint, allow_test_only=allow_test_only)
            legacy = validate_training_data(data, **options)
            report['legacy_validator'] = legacy
            codes.extend(legacy['blocker_codes'])
            if not codes:
                report['contract_status'] = 'TEST_ONLY_CONTRACT_VALID' if data['purpose'] == 'TEST_ONLY' else 'POLICY_AND_LEGACY_CONTRACT_VALID'
                samples = [d for d in decisions.values() if d['split'] == 'train']
                queries = [d for d in decisions.values() if d['split'] == 'validation']
                support = []
                for i,q in enumerate(queries,1):
                    n = sum(jaccard_similarity(observed_tokens(q['observation']), observed_tokens(s['observation'])) > 0 for s in samples)
                    support.append({'query_ref':'ValidationQuery%03d' % i, 'matched_samples_per_agent': n, 'meets_min_samples': n >= min_samples})
                status = 'SUPPORTED_FOR_SUPPLIED_VALIDATION_QUERIES' if support and all(q['meets_min_samples'] for q in support) else 'INSUFFICIENT_SUPPORT' if support else 'NOT_CHECKED_NO_VALIDATION_QUERIES'
                report['positive_similarity_support'] = {'status': status, 'min_samples': min_samples, 'queries': support}
                if status == 'SUPPORTED_FOR_SUPPLIED_VALIDATION_QUERIES':
                    report['utility_fit_ready'] = data['purpose'] != 'TEST_ONLY'
                    report['development_contract_ready'] = data['purpose'] == 'TEST_ONLY'
                    report['training_readiness'] = report['contract_status']
                else:
                    codes.append('POSITIVE_SIMILARITY_SUPPORT_NOT_ESTABLISHED')
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        codes.append('GATE_INPUT_CONTRACT_INVALID')
    if not isinstance(report['legacy_validator'], dict) or report['legacy_validator'].get('blocker_codes'):
        report['complete_train_groups'] = 0
        report['validation_group_count'] = 0
        report['test_group_count'] = 0
    report['blocker_codes'] = sorted(set(codes))
    return report


def to_policy_fit_rows(data, **options):
    """Private conversion only after BOTH gates and support; never calls fit."""
    report = validate_evaluation_gate(data, **options)
    if not report['utility_fit_ready']:
        return {'report': report, 'rows': None}
    legacy_keys = ('registry_manifest','source_catalog','label_catalog','holdout_patient_ids','dataset_fingerprint','allow_test_only')
    legacy_options = {k:v for k,v in options.items() if k in legacy_keys}
    result = to_fit_rows(deepcopy(data), **legacy_options)
    if result['rows'] is None:
        return {'report': report, 'rows': None}
    return {'report': report, 'rows': result['rows']}
