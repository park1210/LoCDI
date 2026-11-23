"""TEST_ONLY C3-03: existing symbolic human declarations, never clinical labels."""
from copy import deepcopy
from functools import lru_cache
import unittest
from unittest.mock import patch

from test_routing_correctness_gate import test_only_reviewed_data
from app.locdi.routing_policy_analysis import (test_only_cases, independent_split_check,
    request_budget, policy_comparison, approval_register)
from app.core.routing_utility import observed_tokens, jaccard_similarity, RoutingUtility
from app.core.agent_registry import AgentRegistry


@lru_cache(maxsize=1)
def contract_probe():
    with patch('app.core.routing_utility.fit_utility', side_effect=AssertionError('NO_FITTING')) as fit:
        report = test_only_cases(test_only_reviewed_data)
    fit.assert_not_called()
    return report


class RoutingPolicyAnalysisTests(unittest.TestCase):
    def case(self, name):
        return next(r for r in contract_probe()['cases'] if r['case'] == name)

    def test_all_supported_contract_only(self):
        r = self.case('all_five_supported_independent_test_labels')
        self.assertTrue(r['development_contract_ready'])
        self.assertFalse(r['utility_fit_ready'])
        self.assertEqual(r['complete_train_groups'], 10)
        self.assertEqual(r['blocker_codes'], [])

    def test_one_unresolved_not_boolean(self):
        r = self.case('one_unresolved')
        self.assertIn('UNRESOLVED_EXCLUDED_FROM_DIAGNOSIS_FITTING', r['blocker_codes'])
        self.assertFalse(r['utility_fit_ready'])

    def test_multiple_unresolved(self):
        r = self.case('multiple_unresolved')
        self.assertIn('UNRESOLVED_EXCLUDED_FROM_DIAGNOSIS_FITTING', r['blocker_codes'])
        self.assertFalse(r['development_contract_ready'])

    def test_all_unresolved(self):
        r = self.case('all_unresolved')
        self.assertFalse(r['utility_fit_ready'])
        self.assertIn('UNRESOLVED_EXCLUDED_FROM_DIAGNOSIS_FITTING', r['blocker_codes'])

    def test_imbalanced_partial_rejected_by_core(self):
        self.assertTrue(self.case('candidate_support_imbalance')['partial_artifact_rejected'])

    def test_zero_similarity_abstains_keeps_menu(self):
        r = self.case('all_similarity_zero')
        self.assertEqual((r['decision_status'], r['selected_count'], r['allowed_count']), ('abstained', 0, 5))
        self.assertTrue(all(v is None for v in r['support_per_candidate'].values()))

    def test_some_candidates_supported_not_selected(self):
        r = self.case('only_some_candidates_meet_min_samples')
        self.assertEqual((r['decision_status'], r['selected_count'], r['allowed_count']), ('abstained', 0, 5))
        self.assertIn('INTERFACE_STRESS', r['scope'])
        self.assertEqual(sum(v == 2 for v in r['support_per_candidate'].values()), 4)

    def test_tie_abstains(self):
        r = self.case('tied_best')
        self.assertEqual((r['decision_status'], r['tied_count'], r['allowed_count']), ('abstained', 5, 5))

    def test_unique_best_symbolic(self):
        r = self.case('unique_best_symbolic')
        self.assertEqual((r['decision_status'], r['selected_count'], r['allowed_count']), ('selected', 1, 1))

    def test_reference_timing_not_reviewed(self):
        self.assertIn('REFERENCE_TIMING_NOT_REVIEWED', self.case('reference_timing_not_reviewed')['blocker_codes'])

    def test_snapshot_memory_difference_blocked(self):
        r = self.case('same_current_observation_different_memory')
        self.assertTrue(r['blocker_codes'])
        self.assertFalse(r['development_contract_ready'])

    def test_same_patient_train_validation(self):
        self.assertIn('PATIENT_SPLIT_OVERLAP_OR_DUPLICATE', self.case('same_patient_train_validation')['blocker_codes'])

    def test_routing_validation_voi_overlap(self):
        r = self.case('routing_validation_in_voi_train')
        self.assertEqual(r['combined_validation'], 'BLOCKED')
        self.assertEqual(r['routing_patient_split'], 'STRUCTURALLY_DISJOINT')

    def test_patient_exclusion_is_not_encounter_exclusion(self):
        r = independent_split_check({'routing_train': ['TEST_ONLY_PATIENT'],
            'routing_validation': ['TEST_ONLY_PATIENT'], 'voi_train': []})
        self.assertEqual(r['routing_train_validation_overlap'], 1)
        self.assertEqual(r['combined_validation'], 'BLOCKED')
        self.assertFalse(r['split_reassigned'])

    def test_probe_cannot_accept_real_data(self):
        with self.assertRaisesRegex(ValueError, 'TEST_ONLY'):
            test_only_cases(lambda: ({'purpose': 'routing_training'}, {}))

    def test_test_only_results_contain_no_training_rows(self):
        r = contract_probe()
        self.assertEqual(r['case_count'], 13)
        self.assertEqual(r['fitting_calls'], 0)
        self.assertTrue(all(not v.get('clinical_approval') for v in r['cases']))
        self.assertNotIn('samples', r)
        self.assertNotIn('judgments', r)

    def test_budget_base_and_bounded_repair_retry(self):
        self.assertEqual(request_budget(15)['maximum_generation_requests'], 75)
        self.assertEqual(request_budget(15, repeats=2, retry_limit=1, repair_limit=1)['maximum_generation_requests'], 450)
        self.assertFalse(request_budget(15)['execution_approved'])

    def test_budget_rejects_bool_negative_float(self):
        for value in (True, -1, 2.5):
            with self.assertRaises(ValueError): request_budget(value)
        with self.assertRaises(ValueError): request_budget(1, agents=0)

    def test_all_policies_are_proposals_not_gate_changes(self):
        rows = {r['policy']: r for r in policy_comparison()['policies']}
        self.assertEqual(set(rows), {'A', 'B1', 'B2', 'C', 'D'})
        self.assertFalse(rows['A']['core_change'] or rows['D']['gate_change'] or rows['B1']['math_change'])
        self.assertTrue(rows['B2']['math_change'] and rows['C']['core_change'] and rows['C']['gate_change'])
        self.assertIn('REQUIRES_CODE_REVISION', rows['C']['states'])

    def test_recommendation_is_not_approval(self):
        r = approval_register()
        self.assertIsNone(r['APPROVED_POLICY'])
        self.assertTrue(r['POLICY_APPROVAL_REQUIRED'])
        self.assertFalse(r['clinical_approval'])
        self.assertIn('ROUTING_VALIDATION_IN_VOI_TRAIN', r['UNRESOLVED_CONFLICTS'])

    def test_tokens_do_not_admit_gold_or_superseded_memory(self):
        obs = {'chief_complaint': 'SYMBOL symbol', 'gold': 'TEST_ONLY_SECRET',
            'history': [{'answer_status': 'unknown', 'answer': 'hidden'}],
            'longitudinal_memory': {'records': [{'text': 'old', 'status_as_of_visit': 'superseded'}, {'text': 'past'}]}}
        self.assertEqual(observed_tokens(obs), ['past', 'symbol'])
        self.assertEqual(jaccard_similarity([], []), 0)

    def test_same_tokens_do_not_prove_same_snapshot(self):
        a = {'chief_complaint': 'symbol', 'longitudinal_memory': {'records': [{'text': 'past', 'memory_id': 'TEST_ONLY_A'}]}}
        b = deepcopy(a); b['longitudinal_memory']['records'][0]['memory_id'] = 'TEST_ONLY_B'
        self.assertEqual(observed_tokens(a), observed_tokens(b))
        from app.locdi.llm_setup import digest
        self.assertNotEqual(digest(a), digest(b))

    def test_actual_score_algebra_and_minimum_support_without_fit(self):
        data, _ = test_only_reviewed_data(); registry = AgentRegistry(data['registry'])
        agents = [a for a in registry.snapshot()['agents'] if a['role'] == 'specialty']
        samples = [{'patient_id': 'TEST_ONLY_P%d' % i, 'decision_id': 'TEST_ONLY_D%d' % i,
                    'agent_id': a['agent_id'], 'correct': True, 'tokens': ['x']} for i in range(2) for a in agents]
        artifact = {'schema_version': 2, 'method': 'jaccard_beta_accuracy_v2', 'registry': registry.snapshot(),
                    'min_samples': 2, 'samples': samples}
        utility = RoutingUtility(artifact, registry)
        self.assertEqual(utility.score({'chief_complaint': 'x'}, agents[0])['score'], 3 / 4)
        self.assertIsNone(utility.score({'chief_complaint': 'y'}, agents[0]))


if __name__ == '__main__':
    unittest.main()
