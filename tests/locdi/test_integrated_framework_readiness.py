"""TEST_ONLY C4-36 gates and passive runtime contracts, no model generation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from app.locdi.integrated_framework_readiness import (
    official_contract, memory_voi_compatibility, reject_hidden, source_summary, callable_inventory,
)
from app.locdi.integrated_development_evaluation import (
    technical_protocol, run_technical, block_source_evaluation,
)
from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_loader import load_cases
from runtime_integration_support import options_for
from test_routing_voi_readiness import test_only_voi


class IntegratedFrameworkReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl')

    def test_official_all_five_are_exact_and_blocked(self):
        for name in ['B0', 'B1', 'B2', 'B3', 'B4']:
            config = json.loads((ROOT / ('experiments/configs/' + name + '.json')).read_text(encoding='utf-8'))
            before = deepcopy(config)
            result = official_contract(config)
            self.assertTrue(result['feature_definition_matches'])
            self.assertEqual(result['execution_status'], 'BLOCKED')
            self.assertFalse(result['runner_implemented'])
            self.assertEqual(config, before)

    def test_feature_drift_not_approved(self):
        config = json.loads((ROOT / 'experiments/configs/B2.json').read_text(encoding='utf-8'))
        config['features']['top_k'] = True
        self.assertFalse(official_contract(config)['feature_definition_matches'])

    def test_ready_true_does_not_implement_runner(self):
        config = json.loads((ROOT / 'experiments/configs/B0.json').read_text(encoding='utf-8'))
        config['ready'] = True
        self.assertEqual(official_contract(config)['execution_status'], 'BLOCKED')

    def test_hidden_gold_nested_rejected(self):
        for key in ['evaluation', 'gold_department', 'task_annotations', 'source_event_store', 'correctness']:
            with self.assertRaises(ValueError):
                reject_hidden({'nested': [{key: 'TEST_ONLY'}]})

    def test_memory_is_not_automatically_bayesian_evidence(self):
        model = test_only_voi()
        model['actions'] = [a for a in model['actions'] if a['type'] != 'retrieve_memory']
        before = digest(model)
        result = memory_voi_compatibility(model, [{'text': 'TEST_ONLY history', 'status_as_of_visit': 'historical'}])
        self.assertFalse(result['memory_only_belief_changed'])
        self.assertEqual(result['memory_only_matched_outcomes'], 0)
        self.assertFalse(result['memory_likelihood_ready'])
        self.assertEqual(before, digest(model))

    def test_memory_gold_not_promoted(self):
        with self.assertRaises(ValueError):
            memory_voi_compatibility(test_only_voi(), [{'text': 'TEST_ONLY', 'gold': 'TEST_ONLY'}])

    def test_original50_denominator_and_memory_preserved(self):
        model = test_only_voi()
        model['provenance']['action_key_mapping'] = {}
        result = source_summary(self.cases, [], model)
        self.assertEqual(result['Original50']['cases'], 50)
        self.assertEqual(result['Original50']['past_memory_records'], 72590)
        self.assertEqual(result['Original50']['consult_source_events'], 0)
        self.assertEqual(result['C4_35']['memory_runtime_records_constructed'], 0)

    def test_protocol_does_not_authorize_source_performance(self):
        spec = technical_protocol(self.cases[0], 'TEST_ONLY', {})
        self.assertFalse(spec['M5_injected'])
        self.assertEqual(spec['budget']['LLM_generation'], 0)
        self.assertEqual(block_source_evaluation()['status'], 'BLOCKED')
        self.assertFalse(block_source_evaluation()['C4_35_APPROVAL_REUSABLE'])

    def test_tampered_protocol_blocked_before_runner(self):
        spec = technical_protocol(self.cases[0], 'TEST_ONLY', {})
        spec['conditions'] = []
        with tempfile.TemporaryDirectory() as out, self.assertRaises(ValueError):
            run_technical(self.cases[0], spec, options_for, output_root=out)

    def test_different_patient_blocked(self):
        spec = technical_protocol(self.cases[0], 'TEST_ONLY', {})
        with tempfile.TemporaryDirectory() as out, self.assertRaises(ValueError):
            run_technical(self.cases[1], spec, options_for, output_root=out)

    def test_ast_inventory_matches_actual_multiline_signature(self):
        result = callable_inventory(ROOT, ['app/core/voi_planning.py'])
        planner = next(r for r in result['app/core/voi_planning.py'] if r['name'] == 'plan_voi')
        self.assertIn('max_states=20000', planner['signature'])
        self.assertGreater(planner['line'], 0)

    def test_real_engine_technical_conditions_no_quality_scores(self):
        spec = technical_protocol(self.cases[0], 'TEST_ONLY', {})
        with tempfile.TemporaryDirectory() as out:
            result, private = run_technical(self.cases[0], spec, options_for, output_root=out)
        self.assertTrue(result['all_technical_completed'])
        self.assertEqual(len(private), 6)
        for row in result['conditions']:
            self.assertEqual(row['calls']['runner'], 1)
            self.assertEqual(row['calls']['scheduler'], 1)
            self.assertTrue(row['episode_audit_consistent'])
            self.assertIsNone(row['performance_metrics'])
            self.assertEqual(row['clinical_metrics'], 'NOT_EVALUATED')
        off, on = result['conditions'][2:4]
        self.assertTrue(all(n == 0 for n in off['memory_records_delivered_to_voi']))
        self.assertTrue(any(n > 0 for n in on['memory_records_delivered_to_voi']))
        self.assertEqual(off['voi_matched_counts'], on['voi_matched_counts'])
        self.assertGreater(on['calls']['planner'], 0)
        self.assertGreater(on['calls']['posterior'], 0)


if __name__ == '__main__':
    unittest.main()
