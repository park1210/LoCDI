"""TEST_ONLY symbolic review/authority evidence. No real medical verdicts."""
from copy import deepcopy
from functools import lru_cache
import unittest
from unittest.mock import patch

from test_routing_correctness_gate import test_only_reviewed_data
from app.locdi.llm_setup import digest
from app.locdi.clinical_review_bridge import (PrivateContextResolver, EvidenceStore, admit_reviews, bridge_contract)
from validation.review_authority_bridge import TestOnlyAuthority, AuthorityProvider
from validation.routing_correctness_gate import POLICY


def reseal(f):
    """Re-pin TEST_ONLY documents so mutations exercise semantics, not only SHA."""
    for ref in f['refs']:
        e = f['docs'][ref]
        e['submission_fingerprint'] = digest(f['docs'][e['submission_ref']])
        pointers = ('submission_ref', 'reference_label_ref', 'legacy_record_ref', 'source_record_ref',
                    'judgment_ref', 'scope_evidence_ref', 'axis_verdict_ref', 'reference_envelope_ref')
        e['evidence_lineage'] = {e[k]: digest(f['docs'][e[k]]) for k in pointers if e[k] is not None}
    for ref, grant in f['grants'].items():
        grant['subject_fingerprint'] = digest(f['docs'][ref])


def fixture():
    data, options = test_only_reviewed_data()
    resolver = PrivateContextResolver(data, options, dataset_version='TEST_ONLY_DATA_V1', test_only=True)
    docs, refs, grants = {}, [], {}
    policy_ref = 'TEST_ONLY_POLICY'
    docs[policy_ref] = {'schema_version': 'locdi.test_policy_approval.v1', 'purpose': 'TEST_ONLY',
        'policy_fingerprint': digest(POLICY), 'approved': True, 'authority_ref': 'TEST_ONLY_POLICY_REVIEWER',
        'approved_at': '2026-10-28T00:00:00+00:00'}
    def grant(ref, reviewer, axis, sequence=None):
        grants[ref] = {'purpose': 'TEST_ONLY', 'subject_fingerprint': '0'*64, 'reviewer_ref': reviewer,
            'axis': axis, 'policy_fingerprint': digest(POLICY),
            **{k: True for k in ('identity_verified', 'qualification_verified', 'task_suitable', 'independence_verified',
                'conflict_clear', 'institutional_authority', 'signature_verified', 'evidence_source_verified',
                'blinding_verified', 'adjudicator_verified', 'submission_receipt_verified')},
            'submission_sequence': sequence, 'valid_from': '2026-01-01T00:00:00+00:00',
            'valid_until': '2027-01-01T00:00:00+00:00', 'revoked': False}
    grant(policy_ref, 'TEST_ONLY_POLICY_REVIEWER', 'POLICY')
    ref_envelopes = {}
    rows = [(r, None) for r in options['review_results']['reference_reviews']]
    rows += [(r, r['agent_id']) for r in options['review_results']['response_reviews']]
    for index, (legacy, agent_id) in enumerate(rows, 1):
        did = legacy['decision_id']; is_ref = agent_id is None
        identity, snapshot, decision, context = resolver.identity(did, agent_id)
        prefix = 'TEST_ONLY_ENVELOPE_%03d' % index
        sub_ref, legacy_ref, label_ref, scope_ref = (prefix+'_'+n for n in ('SUB', 'LEGACY', 'LABEL', 'SCOPE'))
        label = next(l for l in options['label_catalog'] if l['label_ref'] == legacy['reference_ref'])
        review = legacy['review']
        submission = {'schema_version': 'locdi.review_submission.v1',
            'submission_kind': 'REFERENCE_REVIEW' if is_ref else 'RESPONSE_JUDGMENT',
            'review_case_ref': legacy['case_ref'], 'decision_ref': did, 'reference_source': decision['source_ref'],
            'source_fingerprint': context['source_fingerprint'], 'reference_fingerprint': label['label_fingerprint'],
            'observed_at': '2026-01-01T00:00:00+00:00', 'available_at': label['reference_content']['available_at'],
            'decision_at': '2026-01-01T00:01:00+00:00', 'reviewer_ref': review['reviewer_ref'],
            'reviewer_role': review['reviewer_kind'], 'reviewer_independent': review['independent'],
            'review_timestamp': review['reviewed_at'], 'decision_status': 'REVIEWED',
            'evidence_reference': label['reference_content']['evidence_ref'],
            'correctness_label': None if is_ref else legacy['axes']['diagnosis_correctness']['verdict'],
            'review_policy_version': review['procedure_version'], 'reference_timing_status': 'CURRENT_DECISION_REFERENCE_VALID',
            'evaluation_target': 'diagnosis_correctness', 'response_ref': None if is_ref else legacy['response_ref'],
            'response_fingerprint': None if is_ref else legacy['response_fingerprint'],
            'response_recorded_at': None if is_ref else '2026-10-28T00:00:00+00:00',
            'certainty': None if is_ref else 'supported',
            'clinical_evidence_rationale': legacy['clinical_evidence_rationale'] if is_ref else legacy['axes']['diagnosis_correctness']['clinical_evidence_rationale'],
            'not_evaluable_reason': None, 'approval_proof': None}
        docs[sub_ref], docs[legacy_ref], docs[label_ref] = submission, deepcopy(legacy), deepcopy(label)
        reference_row = next(r for r in options['review_results']['reference_reviews'] if r['decision_id'] == did)
        docs[scope_ref] = {'review_status': 'REVIEWED', 'context_fingerprint': digest(snapshot),
            'input_scope': deepcopy(reference_row['input_scope']), 'submitted_sequence': index,
            'submitted_at': review['reviewed_at']}
        source_ref = judgment_ref = None
        if is_ref:
            source_ref = prefix+'_SOURCE'
            docs[source_ref] = deepcopy(next(s for s in options['source_catalog'] if s['source_ref'] == decision['source_ref']))
        else:
            judgment_ref = prefix+'_JUDGMENT'
            docs[judgment_ref] = deepcopy(next(j for j in data['judgments'] if j['response_ref'] == legacy['response_ref']))
        docs[prefix] = {'schema_version': 'locdi.review_envelope.v1', 'purpose': 'TEST_ONLY',
            'submission_ref': sub_ref, 'submission_fingerprint': digest(submission), 'identity': identity,
            'context_fingerprint': digest(snapshot), 'axis': 'REFERENCE_VALIDITY' if is_ref else 'RESPONSE_CORRECTNESS',
            'reference_validity': 'CURRENT_DECISION_REFERENCE_VALID', 'reference_timing': 'CURRENT_DECISION_REFERENCE_VALID',
            'reference_envelope_ref': None if is_ref else ref_envelopes[did],
            'reference_label_ref': label_ref, 'legacy_record_ref': legacy_ref, 'source_record_ref': source_ref,
            'judgment_ref': judgment_ref, 'scope_evidence_ref': scope_ref, 'procedure_version': review['procedure_version'],
            'policy_fingerprint': digest(POLICY), 'review_timestamp': review['reviewed_at'],
            'uncertainty_status': 'NONE', 'adjudication_status': 'NO_DISAGREEMENT',
            'blinding_status': 'VERIFIED_BY_AUTHORITY' if is_ref else 'NOT_APPLICABLE',
            'axis_verdict_ref': None, 'evidence_lineage': {}}
        if is_ref: ref_envelopes[did] = prefix
        refs.append(prefix); grant(prefix, review['reviewer_ref'], docs[prefix]['axis'], index)
    f = {'resolver': resolver, 'docs': docs, 'refs': refs, 'grants': grants, 'policy_ref': policy_ref}
    reseal(f)
    return f


def run_fixture(f, *, production=False):
    return admit_reviews(f['resolver'], EvidenceStore(f['docs']), f['refs'],
        provider=TestOnlyAuthority(f['grants'], verification_at='2026-11-01T00:00:00+00:00'),
        policy_ref=f['policy_ref'], production=production)


def mutate(f, name):
    env = f['docs'][f['refs'][0]]
    response_env = f['docs'][f['refs'][15]]
    sub = f['docs'][env['submission_ref']]
    rsp = f['docs'][response_env['submission_ref']]
    record = f['docs'][response_env['legacy_record_ref']]
    if name == 'schema_only': f['policy_ref'] = None
    elif name == 'reviewer_name_only':
        f['grants'] = {f['policy_ref']: f['grants'][f['policy_ref']]}
    elif name == 'fake_signature': f['grants'][f['refs'][0]]['signature_verified'] = False
    elif name == 'response_before_reference':
        f['docs'][response_env['scope_evidence_ref']]['submitted_sequence'] = 0
        f['grants'][f['refs'][15]]['submission_sequence'] = 0
    elif name == 'blind_not_verified': f['grants'][f['refs'][0]]['blinding_verified'] = False
    elif name == 'reference_fingerprint_mismatch': sub['reference_fingerprint'] = '9'*64
    elif name == 'response_fingerprint_mismatch': rsp['response_fingerprint'] = '9'*64
    elif name == 'other_patient': env['identity']['patient_id'] = 'TEST_ONLY_OTHER'
    elif name == 'other_decision': env['identity']['decision_id'] = 'TEST_ONLY_OTHER'
    elif name == 'other_agent': response_env['identity']['agent_id'] = 'TEST_ONLY_OTHER'
    elif name == 'different_memory': env['context_fingerprint'] = '9'*64
    elif name == 'future_condition':
        r = f['resolver']; d, o = deepcopy(r._data), deepcopy(r._options)
        o['decision_contexts'][0]['reference_candidates']['source_condition_rows'][0]['source_record']['START'] = '2026-01-05'
        r2 = PrivateContextResolver(d, o, dataset_version=r.dataset_version, test_only=True)
        f['resolver'] = r2
        for e_ref in f['refs']:
            e = f['docs'][e_ref]
            if e['identity']['decision_id'] == d['decisions'][0]['decision_id']:
                _, snapshot, _, ctx = r2.identity(e['identity']['decision_id'], e['identity']['agent_id'])
                e['context_fingerprint'] = digest(snapshot)
                f['docs'][e['scope_evidence_ref']]['context_fingerprint'] = digest(snapshot)
                if e['axis'] == 'REFERENCE_VALIDITY': f['docs'][e['legacy_record_ref']]['condition_candidates_fingerprint'] = digest(ctx['reference_candidates'])
    elif name == 'retrospective_to_current':
        env['reference_validity'] = 'RETROSPECTIVE_OUTCOME_ONLY'
    elif name == 'insufficient_not_timing': env['reference_validity'] = 'INSUFFICIENT_INFORMATION'
    elif name == 'unresolved_false':
        rsp['certainty'] = 'unresolved'; rsp['correctness_label'] = False
    elif name == 'not_evaluable_reason_missing':
        a = record['axes']['abstention_appropriateness']; a.update(status='NOT_EVALUABLE', verdict=None, not_evaluable_reason=None)
    elif name == 'non_diagnosis_verdict_missing':
        record['axes']['routing_success'].update(status='REVIEWED', verdict=None, clinical_evidence_rationale='TEST_ONLY')
    elif name == 'input_scope_missing':
        f['docs'][env['scope_evidence_ref']]['input_scope'].pop('memory_reviewed')
    elif name == 'policy_version_mismatch': env['policy_fingerprint'] = '9'*64
    elif name == 'qualification_not_verified': f['grants'][f['refs'][0]]['qualification_verified'] = False
    elif name == 'authority_expired': f['grants'][f['refs'][0]]['valid_until'] = '2026-01-02T00:00:00+00:00'
    elif name == 'authority_revoked': f['grants'][f['refs'][0]]['revoked'] = True
    elif name == 'disagreement_unresolved': env['adjudication_status'] = 'DISAGREEMENT'
    elif name == 'partial_agent_group':
        # Keep all original responses; omit a review. Do not salvage a complete group.
        f['refs'].pop(15)
    elif name == 'patient_split_overlap':
        r = f['resolver']; d, o = deepcopy(r._data), deepcopy(r._options)
        d['split_manifest']['patients']['validation'].append(d['split_manifest']['patients']['train'][0])
        f['resolver'] = PrivateContextResolver(d, o, dataset_version=r.dataset_version, test_only=True)
        for e_ref in f['refs']:
            e = f['docs'][e_ref]; _, snap, _, _ = f['resolver'].identity(e['identity']['decision_id'], e['identity']['agent_id'])
            e['context_fingerprint'] = digest(snap); f['docs'][e['scope_evidence_ref']]['context_fingerprint'] = digest(snap)
    elif name == 'legacy_independent_block':
        # Genuine bridge inputs; unchanged Gate must still reject zero query support.
        f['resolver']._options['min_samples'] = 11
        r = f['resolver']
        f['resolver'] = PrivateContextResolver(r._data, r._options, dataset_version=r.dataset_version, test_only=True)
    elif name == 'extra_legacy_property': f['docs'][env['legacy_record_ref']]['invented'] = True
    elif name == 'missing_sidecar': env['scope_evidence_ref'] = None
    elif name == 'adjudicator_unverified':
        env['adjudication_status'] = 'RESOLVED_WITH_AUTHORITY'; f['grants'][f['refs'][0]]['adjudicator_verified'] = False
    elif name == 'reference_receipt_after_response':
        f['docs'][env['scope_evidence_ref']]['submitted_at'] = '2026-10-31T00:00:00+00:00'
    elif name == 'llm_reviewer': sub['reviewer_role'] = 'LLM'
    else: raise ValueError('TEST_ONLY_UNKNOWN_CASE')
    reseal(f)


CASE_NAMES = ('schema_only', 'reviewer_name_only', 'fake_signature', 'response_before_reference', 'blind_not_verified',
    'reference_fingerprint_mismatch', 'response_fingerprint_mismatch', 'other_patient', 'other_decision', 'other_agent',
    'different_memory', 'future_condition', 'retrospective_to_current', 'insufficient_not_timing', 'unresolved_false',
    'not_evaluable_reason_missing', 'non_diagnosis_verdict_missing', 'input_scope_missing', 'policy_version_mismatch',
    'qualification_not_verified', 'authority_expired', 'authority_revoked', 'disagreement_unresolved',
    'partial_agent_group', 'patient_split_overlap', 'legacy_independent_block', 'extra_legacy_property', 'missing_sidecar',
    'adjudicator_unverified', 'reference_receipt_after_response', 'llm_reviewer')

EXPECTED_CODES = dict(zip(CASE_NAMES, (
    'REVIEW_POLICY_NOT_APPROVED', 'AUTHORITY_UNVERIFIED', 'AUTHORITY_SIGNATURE_VERIFIED_MISSING',
    'INVALID_REVIEW_SEQUENCE', 'BLINDING_NOT_VERIFIED', 'REFERENCE_FINGERPRINT_OR_IDENTITY_MISMATCH',
    'ORIGINAL_RESPONSE_FINGERPRINT_MISMATCH', 'SOURCE_CONTEXT_MISMATCH', 'SOURCE_CONTEXT_MISMATCH',
    'SOURCE_CONTEXT_MISMATCH', 'SOURCE_CONTEXT_MISMATCH', 'FUTURE_SOURCE_CONDITION_NOT_CURRENT_REFERENCE',
    'REFERENCE_NOT_APPROVED', 'REFERENCE_NOT_APPROVED', 'UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN',
    'NOT_EVALUABLE_REASON_REQUIRED', 'NON_DIAGNOSIS_AXIS_VERDICT_MISSING', 'SOURCE_CONTEXT_MISMATCH',
    'REVIEW_POLICY_NOT_APPROVED', 'AUTHORITY_QUALIFICATION_VERIFIED_MISSING', 'AUTHORITY_EXPIRED_OR_NOT_YET_VALID',
    'AUTHORITY_REVOKED_OR_UNKNOWN', 'UNRESOLVED_DISAGREEMENT', 'RESPONSE_CLINICAL_REVIEW_MISSING',
    'PATIENT_SPLIT_OVERLAP_OR_DUPLICATE', 'POSITIVE_SIMILARITY_SUPPORT_NOT_ESTABLISHED',
    'LEGACY_SCHEMA_INVALID_LOSSLESS_CONVERSION_NOT_PROVEN', 'MISSING_REQUIRED_EVIDENCE',
    'ADJUDICATOR_AUTHORITY_UNVERIFIED', 'INVALID_REVIEW_SEQUENCE', 'ORIGINAL_SUBMISSION_SCHEMA_INVALID')))


@lru_cache(maxsize=1)
def counterexamples():
    base = fixture(); output = []
    with patch('app.core.routing_utility.fit_utility', side_effect=AssertionError('FITTING_FORBIDDEN')) as fit:
        good = run_fixture(base)
        output.append({'case': 'complete_symbolic_evidence', 'expected': 'PASS_TEST_ONLY_CONTRACT', 'result': good})
        for name in CASE_NAMES:
            f = deepcopy(base); mutate(f, name)
            output.append({'case': name, 'expected': 'BLOCKED', 'expected_code': EXPECTED_CODES[name], 'result': run_fixture(f)})
        output.append({'case': 'fake_authority_production', 'expected': 'BLOCKED', 'result': run_fixture(base, production=True)})
    fit.assert_not_called()
    return {'schema_version': 'locdi.c3_04.test_only_results.v1', 'purpose': 'TEST_ONLY', 'cases': output,
        'case_count': len(output), 'status': 'PASS' if all(x['result']['status'] == x['expected'] and
            ('expected_code' not in x or x['expected_code'] in x['result']['blocker_codes']) for x in output) else 'FAIL',
        'clinical_verdicts_generated': 0, 'fitting_calls': 0,
        'scope': 'Declared symbolic judgments and fake ledger only; no real review or signed certificate'}


class ClinicalReviewBridgeTests(unittest.TestCase):
    def check(self, name, code=None):
        row = next(x for x in counterexamples()['cases'] if x['case'] == name)
        self.assertEqual(row['result']['status'], row['expected'], row)
        if 'expected_code' in row: self.assertIn(row['expected_code'], row['result']['blocker_codes'], row)
        if code: self.assertIn(code, row['result']['blocker_codes'])
        self.assertFalse(row['result']['clinical_reference_ready'])
        return row['result']

    def test_complete_symbolic_contract_calls_unchanged_gates(self):
        r = self.check('complete_symbolic_evidence')
        self.assertTrue(r['legacy_gate_called'] and r['legacy_validator_called'] and r['test_only_conversion_ready'])
        self.assertEqual((r['complete_train_groups'], r['validation_group_count']), (10, 5))
        self.assertEqual(r['fit_rows_eligibility'], 'TEST_ONLY_PRIVATE_ROWS_NOT_EXPORTED')
        self.assertFalse(r['research_artifact_gate']['research_evaluation_ready'])

    def test_test_only_provider_cannot_authorize_production(self):
        r = self.check('fake_authority_production', 'TEST_ONLY_AUTHORITY_FORBIDDEN_IN_PRODUCTION')
        self.assertFalse(r['legacy_gate_called'])

    def test_legacy_gate_retains_independent_support_block(self):
        r = self.check('legacy_independent_block', 'POSITIVE_SIMILARITY_SUPPORT_NOT_ESTABLISHED')
        self.assertTrue(r['legacy_gate_called'] and r['legacy_validator_called'])

    def test_future_source_condition_reaches_existing_gate(self):
        r = self.check('future_condition', 'FUTURE_SOURCE_CONDITION_NOT_CURRENT_REFERENCE')
        self.assertTrue(r['legacy_gate_called'])

    def test_default_authority_is_not_configured(self):
        self.assertEqual(AuthorityProvider().verify('TEST_ONLY', '0'*64, {})['status'], 'AUTHENTICATION_NOT_CONFIGURED')

    def test_other_provider_object_cannot_self_authorize(self):
        f = fixture()
        class Pretender:
            def verify(self, *args): return {'verified': True}
        r = admit_reviews(f['resolver'], EvidenceStore(f['docs']), f['refs'], provider=Pretender(), policy_ref=f['policy_ref'])
        self.assertFalse(r['legacy_gate_called'])

    def test_submitted_store_cannot_be_mutated_in_place(self):
        store = EvidenceStore({'TEST_ONLY': {'value': 'original'}})
        store._documents['TEST_ONLY']['value'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'IMMUTABLE'): store.get('TEST_ONLY')

    def test_context_mutation_rejected(self):
        f = fixture(); f['resolver']._data['decisions'][0]['observation']['chief_complaint'] = 'TEST_ONLY_TAMPER'
        r = run_fixture(f); self.assertEqual(r['status'], 'BLOCKED'); self.assertFalse(r['legacy_gate_called'])

    def test_five_axes_and_distinct_validity_timing_types(self):
        c = bridge_contract(); self.assertEqual(len(c['axes']), 5)
        self.assertIn('INSUFFICIENT_INFORMATION', c['reference_validity_types'])
        self.assertNotIn('INSUFFICIENT_INFORMATION', c['legacy_timing_types'])
        self.assertFalse(c['test_only_provider_promotable'])

    def test_real_context_rejected_by_test_only_constructor(self):
        with self.assertRaisesRegex(ValueError, 'TEST_ONLY'):
            PrivateContextResolver({'purpose': 'routing_training'}, {}, dataset_version='TEST_ONLY', test_only=True)

    def test_results_no_private_snapshots_or_labels(self):
        import json
        text = json.dumps(counterexamples())
        self.assertNotIn('TEST_ONLY_REFERENCE_HUMAN', text)
        self.assertNotIn('patient_id', text)
        self.assertNotIn('TEST_ONLY_VERDICT', text)

    def test_identical_tokens_with_different_memory_identity_blocked(self):
        from app.core.routing_utility import observed_tokens
        f = fixture(); r = f['resolver']; d, o = deepcopy(r._data), deepcopy(r._options)
        before = observed_tokens(d['decisions'][0]['observation'])
        memory = d['decisions'][0]['observation']['longitudinal_memory']
        memory['records'].append({'text': '', 'memory_id': 'TEST_ONLY_OTHER_SNAPSHOT'})
        self.assertEqual(before, observed_tokens(d['decisions'][0]['observation']))
        f['resolver'] = PrivateContextResolver(d, o, dataset_version=r.dataset_version, test_only=True)
        result = run_fixture(f)
        self.assertIn('SOURCE_CONTEXT_MISMATCH', result['blocker_codes'])
        self.assertFalse(result['legacy_gate_called'])

    def test_auxiliary_axis_never_becomes_diagnosis_bool(self):
        f = fixture(); e = f['docs'][f['refs'][0]]
        e['axis'] = 'MEMORY_RELEVANCE'
        reseal(f)
        result = run_fixture(f)
        self.assertIn('AXIS_SEMANTIC_MISMATCH', result['blocker_codes'])
        self.assertIn('LOSSLESS_CONVERSION_NOT_PROVEN', result['blocker_codes'])
        self.assertFalse(result['legacy_gate_called'])

    def test_unmodified_submission_hash_is_required(self):
        f = fixture(); e = f['docs'][f['refs'][0]]; e['submission_fingerprint'] = '9'*64
        # Envelope is re-authorized as a TEST_ONLY negative case, original body
        # hash still disagrees. No automatic repair of the submitted document.
        f['grants'][f['refs'][0]]['subject_fingerprint'] = digest(e)
        result = run_fixture(f)
        self.assertIn('ORIGINAL_SUBMISSION_FINGERPRINT_MISMATCH', result['blocker_codes'])

    def test_authority_expired_after_review_before_admission(self):
        f = fixture()
        f['grants'][f['refs'][0]]['valid_until'] = '2026-10-29T23:59:59+00:00'
        result = run_fixture(f)
        self.assertIn('AUTHORITY_EXPIRED_OR_NOT_YET_VALID', result['blocker_codes'])
        self.assertFalse(result['legacy_gate_called'])

    def test_existing_other_agent_cannot_borrow_a_response(self):
        f = fixture(); e = f['docs'][f['refs'][15]]
        e['identity']['agent_id'] = f['docs'][f['refs'][16]]['identity']['agent_id']
        reseal(f)
        result = run_fixture(f)
        self.assertIn('SOURCE_CONTEXT_MISMATCH', result['blocker_codes'])
        self.assertFalse(result['legacy_gate_called'])


def _test_case(name):
    def run(self): self.check(name)
    return run

for _name in CASE_NAMES:
    if _name not in ('future_condition', 'legacy_independent_block'):
        setattr(ClinicalReviewBridgeTests, 'test_counterexample_' + _name, _test_case(_name))


if __name__ == '__main__': unittest.main()
