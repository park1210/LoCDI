"""TEST_ONLY paired routing/metrics. No clinical labels, fitting or generation."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from app.core.agent_registry import AgentRegistry
from app.locdi.llm_setup import ROOT, digest
from app.locdi.routing_evaluation import (REFERENCE_POLICY, observable, split_check, paired_route,
    validate_reference, metrics, evaluate, validate_utility_provenance)
from scripts.evaluate_adaptive_routing import pinned, locations, reference_template
from scripts.enrich_full_cohort_sources import write
from test_routing_voi_readiness import test_only_registry, test_only_routing


def fixture(tie=False):
    manifest = test_only_registry()
    registry = AgentRegistry(manifest, test_only_routing(manifest, tie))
    cases = [{'case_ref': 'Case%03d' % i, 'patient_id': 'TEST_ONLY_EVAL_%d' % i, 'visit_id': 'TEST_ONLY_V_%d' % i,
              'split': 'validation', 'decision_time': '2020-01-01T00:00:00Z', 'source_fingerprint': digest(['TEST_ONLY', i]),
              'candidate_departments': ['cardiology', 'neurology'],
              'observation': {'chief_complaint': 'test_only_signal'}} for i in range(1, 3)]
    ref = reference_template(cases)
    for r in ref['reviews']:
        r.update(status='REVIEWED', accepted_specialties=['neurology'], observation_reviewed=True,
                 memory_scope_reviewed=True, visibility_proxy_reviewed=True,
                 timing_status='CURRENT_DECISION_REFERENCE_VALID', reference_available_at='2020-01-01T00:00:00Z',
                 reference_evidence_fingerprint=digest('TEST_ONLY_REFERENCE'), clinical_evidence_rationale='TEST_ONLY_NO_CLINICAL_MEANING',
                 reviewer={'kind': 'HUMAN_CLINICAL_REVIEWER', 'independent': True, 'private_ref': 'TEST_ONLY_REVIEWER',
                           'qualification': 'TEST_ONLY_NOT_A_REAL_REVIEW', 'procedure_version': REFERENCE_POLICY,
                           'reviewed_at': '2020-01-02T00:00:00Z'})
    return cases, registry, ref


class RoutingEvaluationTests(unittest.TestCase):
    def test_paired_actual_core_decision_and_same_input(self):
        cases, reg, ref = fixture()
        summary, traces = evaluate(cases, reg, ref, static_department='cardiology', test_only=True)
        self.assertEqual(len(traces), 4)
        for a, b in zip(traces[::2], traces[1::2]):
            self.assertEqual(a['observation_fingerprint'], b['observation_fingerprint'])
            self.assertEqual(a['candidate_fingerprint'], b['candidate_fingerprint'])
        self.assertEqual(summary['metrics']['adaptive_native_abstention']['selective_accuracy']['value'], 1)
        self.assertEqual(summary['metrics']['static_development']['top1_accuracy_all_evaluable']['value'], 0)
        self.assertEqual(summary['clinical_metrics'], 'NOT_EVALUATED')

    def test_native_tie_is_abstention_not_incorrect_selection(self):
        cases, reg, ref = fixture(tie=True)
        summary, _ = evaluate(cases, reg, ref, static_department='neurology', test_only=True)
        m = summary['metrics']['adaptive_native_abstention']
        self.assertEqual(m['abstention_rate']['value'], 1)
        self.assertEqual(m['coverage']['value'], 0)
        self.assertIsNone(m['selective_accuracy']['value'])
        self.assertEqual(m['incorrect_selected_count'], 0)
        self.assertEqual(m['invalid_count'], 0)

    def test_missing_utility_distinguished(self):
        cases, _, _ = fixture()
        traces = paired_route(cases, AgentRegistry(test_only_registry()), static_department='neurology')
        self.assertTrue(all(r['decision']['reason_code'] == 'MISSING_UTILITY' for r in traces[1::2]))

    def test_insufficient_similarity_distinguished(self):
        cases, reg, _ = fixture()
        for c in cases: c['observation']['chief_complaint'] = 'test_only_no_overlap'
        traces = paired_route(cases, reg, static_department='neurology')
        self.assertTrue(all(r['decision']['reason_code'] == 'INSUFFICIENT_SUPPORT' for r in traces[1::2]))

    def test_gold_key_rejected(self):
        for key in ('gold', 'gold_department', 'evaluation', 'accepted_specialties'):
            with self.assertRaises(ValueError): observable({'history': [{'answer': {key: 'TEST_ONLY'}}]})

    def test_serialized_hidden_label_rejected(self):
        with self.assertRaises(ValueError): observable({'history': [{'answer': '{"gold_departments": ["neurology"]}'}]})

    def test_unused_root_field_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, 'WHITELIST'): observable({'diagnosis': 'TEST_ONLY'})

    def test_patient_split_overlap_another_visit(self):
        cases, _, _ = fixture(); cases[1]['patient_id'] = cases[0]['patient_id']; cases[1]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'PATIENT_SPLIT_OVERLAP'): split_check(cases)

    def test_utility_training_patient_leak(self):
        cases, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'UTILITY_TRAIN'): split_check(cases, [cases[0]['patient_id']])

    def test_candidate_invalid_or_static_unavailable(self):
        cases, reg, _ = fixture()
        for bad in ([], ['neurology', 'dermatology'], ['neurology', 'neurology'], ['cardiology']):
            cases[0]['candidate_departments'] = bad
            with self.assertRaisesRegex(ValueError, 'CANDIDATE_SET_INVALID'): paired_route(cases, reg, static_department='neurology')

    def test_missing_reference_blocks_before_route(self):
        cases, reg, _ = fixture()
        with patch.object(reg, 'route') as call:
            with self.assertRaisesRegex(ValueError, 'REFERENCE_REQUIRED'):
                evaluate(cases, reg, None, static_department='neurology', test_only=True)
            call.assert_not_called()

    def test_blank_reference_blocks(self):
        cases, reg, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'NO_INDEPENDENT_SCORABLE_REFERENCE'):
            evaluate(cases, reg, reference_template(cases), static_department='neurology', test_only=True)

    def test_reference_binding_and_future_timing(self):
        cases, _, ref = fixture()
        ref['reviews'][0]['reference_available_at'] = '2021-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError, 'TIMING'): validate_reference(cases, ref)
        ref['cases_fingerprint'] = 'BAD'
        with self.assertRaisesRegex(ValueError, 'COHORT_BINDING'): validate_reference(cases, ref)

    def test_reviewer_independence_not_inferred(self):
        cases, _, ref = fixture(); ref['reviews'][0]['reviewer']['independent'] = False
        with self.assertRaisesRegex(ValueError, 'INDEPENDENT_REVIEWER'): validate_reference(cases, ref)

    def test_reference_evidence_fingerprint_required(self):
        cases, _, ref = fixture(); ref['reviews'][0]['reference_evidence_fingerprint'] = 'TEST_ONLY_BAD'
        with self.assertRaisesRegex(ValueError, 'SPECIALTY_SET_REQUIRED'): validate_reference(cases, ref)

    def test_reference_outside_candidate_scope_not_silently_relabelled(self):
        cases, _, ref = fixture(); ref['reviews'][0]['accepted_specialties'] = ['dermatology']
        with self.assertRaisesRegex(ValueError, 'SPECIALTY_SET_REQUIRED'): validate_reference(cases, ref)

    def test_multiple_acceptable_specialties(self):
        cases, reg, ref = fixture()
        for r in ref['reviews']: r['accepted_specialties'] = ['neurology', 'cardiology']
        summary, _ = evaluate(cases, reg, ref, static_department='cardiology', test_only=True)
        self.assertEqual(summary['metrics']['static_development']['selective_accuracy']['value'], 1)

    def test_partial_reference_keeps_registration_denominator(self):
        cases, reg, ref = fixture(); ref['reviews'][1] = reference_template(cases)['reviews'][1]
        summary, _ = evaluate(cases, reg, ref, static_department='neurology', test_only=True)
        m = summary['metrics']['static_development']
        self.assertEqual(m['reference_coverage']['denominator'], 2)
        self.assertEqual(m['reference_coverage']['numerator'], 1)

    def test_invalid_decision_denominators(self):
        rows = [{'case_ref': 'Case001', 'decision': {'decision_status': 'selected', 'selected_specialty': 'BAD',
                                                   'candidate_departments': ['neurology']}}]
        m = metrics(rows, {'Case001': {'neurology'}})
        self.assertEqual(m['invalid_decision_rate']['value'], 1)
        self.assertEqual(m['coverage']['value'], 0)
        self.assertEqual(m['incorrect_routing_rate']['value'], 0)

    def test_empty_denominators_not_zero(self):
        self.assertIsNone(metrics([], {})['selective_accuracy']['value'])
        self.assertEqual(metrics([], {})['top_k_specialty_recall'], 'NOT_SUPPORTED_NO_RANKED_SET_CONTRACT')

    def test_no_clinical_promotion_of_test_only(self):
        cases, reg, ref = fixture()
        with self.assertRaisesRegex(ValueError, 'TEST_ONLY_CANNOT_BE_CLINICAL'):
            evaluate(cases, reg, ref, static_department='neurology')

    def test_no_train_scoring(self):
        cases, reg, ref = fixture()
        with self.assertRaisesRegex(ValueError, 'NO_TRAIN_SCORING'):
            evaluate(cases, reg, ref, static_department='neurology', test_only=True, evaluation_split='train')

    def test_repeat_trace_determinism_and_input_preservation(self):
        cases, reg, ref = fixture(); before = deepcopy((cases, ref, reg.snapshot()))
        a = evaluate(cases, reg, ref, static_department='neurology', test_only=True)
        self.assertEqual(a, evaluate(cases, reg, ref, static_department='neurology', test_only=True))
        self.assertEqual((cases, ref, reg.snapshot()), before)

    def test_actual_fitting_gate_not_bypassed(self):
        _, reg, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'UTILITY_GATE_PROVENANCE_REQUIRED'): validate_utility_provenance(reg, None)
        with patch('app.locdi.routing_evaluation.to_policy_fit_rows', return_value={'rows': None}):
            with self.assertRaisesRegex(ValueError, 'CORRECTNESS_GATE_BLOCKED'):
                validate_utility_provenance(reg, {'data': {}, 'options': {}})

    def test_utility_binding_to_gate_rows_and_support_policy(self):
        _, reg, _ = fixture()
        rows = [dict(r, observation={'chief_complaint': 'test_only_signal'}) for r in reg.utility.artifact['samples']]
        report = {'positive_similarity_support': {'min_samples': 2}}
        proof = {'data': {}, 'options': {'registry_manifest': reg.snapshot()}}
        with patch('app.locdi.routing_evaluation.to_policy_fit_rows', return_value={'rows': rows, 'report': report}):
            self.assertEqual(validate_utility_provenance(reg, proof), report)
            rows[0]['correct'] = not rows[0]['correct']
            with self.assertRaisesRegex(ValueError, 'GATED_TRAIN_ROWS'): validate_utility_provenance(reg, proof)
            rows[0]['correct'] = not rows[0]['correct']
            report['positive_similarity_support']['min_samples'] = 3
            with self.assertRaisesRegex(ValueError, 'SUPPORT_POLICY_MISMATCH'): validate_utility_provenance(reg, proof)

    def test_not_evaluable_reference_keeps_unknown(self):
        cases, reg, ref = fixture()
        ref['reviews'][1].update(status='NOT_EVALUABLE', accepted_specialties=None, not_evaluable_reason='TEST_ONLY_SCOPE_INSUFFICIENT')
        summary, _ = evaluate(cases, reg, ref, static_department='neurology', test_only=True)
        self.assertEqual(summary['metrics']['static_development']['reference_evaluable_decisions'], 1)
        self.assertEqual(summary['metrics']['static_development']['registered_decisions'], 2)

    def test_source_hash_and_private_path(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/private') as folder:
            p = Path(folder)/'TEST_ONLY.json'; write(p, {'TEST_ONLY': True})
            before = p.read_bytes()
            with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'): pinned(p, '0'*64)
            self.assertEqual(p.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'PRIVATE_INPUT_REQUIRED'): pinned(ROOT/'README.md', '0'*64)

    def test_namespace_escape_and_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'INVALID_NAMESPACE'): locations('../BAD')
        with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS'): locations('c4_07_v1')
