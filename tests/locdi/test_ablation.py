"""User-confirmed config equality and readiness guards; no experimental runs."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.run_locdi_ablation import AblationConfigurationError, run_ablation
from experiments.locdi_metrics import evaluate_episode, aggregate_metrics
from experiments.compare_actions import compare_action_sets
from experiments.analyze_failures import analyze_failures
from evidence.extract_findings import extract_findings
from evidence.review_summary import summarize_reviews


class AblationContracts(unittest.TestCase):
    def test_official_conditions_and_shared_inputs(self):
        root = Path(__file__).resolve().parents[2] / "experiments/configs"
        expected = [
            ("none", None, "er_only", "rule_similarity", False, "specialist_heuristic"),
            ("full_history", None, "er_only", "rule_similarity", False, "specialist_heuristic"),
            ("selective", 5, "er_only", "rule_similarity", False, "specialist_heuristic"),
            ("selective", 5, "adaptive", "adaptive", False, "specialist_heuristic"),
            ("selective", 5, "adaptive", "adaptive", True, "bayesian_voi")]
        keys = ("memory_mode", "top_k", "routing_mode", "primary_router", "voi_enabled", "action_selector")
        common = None
        with patch("experiments.run_locdi_ablation.run_cases") as runner:
            for number, values in enumerate(expected):
                config = json.loads((root / ("B{}.json".format(number))).read_text(encoding="utf-8"))
                self.assertEqual(config["condition"], "B{}".format(number))
                self.assertEqual(tuple(config["features"][k] for k in keys), values)
                self.assertIs(config["ready"], False)
                self.assertEqual(config["common"]["memory_source"], "fixed")
                self.assertEqual(config["common"]["case_count"], 50)
                self.assertIsNone(config["common"]["model"])
                self.assertIsNone(config["common"]["agent_registry"])
                self.assertIsNone(config["common"]["voi_model"])
                if common is not None:
                    self.assertEqual(config["common"], common)
                common = config["common"]
                with self.assertRaises(AblationConfigurationError):
                    run_ablation(config)
                config["ready"] = True
                with self.assertRaises(AblationConfigurationError):
                    run_ablation(config)
            runner.assert_not_called()

    def test_pending_research_functions_never_report_success(self):
        operations = (lambda: evaluate_episode({}, context=None), lambda: aggregate_metrics([]),
                      lambda: compare_action_sets([], []), lambda: analyze_failures([]),
                      lambda: extract_findings({}), lambda: summarize_reviews([]))
        for operation in operations:
            with self.assertRaises(NotImplementedError):
                operation()

    @unittest.skip("TODO: verified shared model/artifacts/labels and reviewed metrics before paired experiments")
    def test_b0_b4_paired_clinical_evaluation(self):
        pass
