"""Exhaustive actual-dataset validation and malformed-record continuation."""
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from app.locdi.dataset_loader import DatasetFormatError, load_cases
from validation.run_validation import validate_dataset

DATASET = Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl"


class DatasetValidationContracts(unittest.TestCase):
    def test_all_50_results_and_unreviewed_scope_are_retained(self):
        report = validate_dataset(DATASET)
        self.assertEqual(report.case_count, 50)
        self.assertEqual(len(report.cases), 50)
        self.assertEqual(len(report.results), 200)
        self.assertEqual(report.counts["schema"]["PASS"], 50)
        self.assertEqual(report.counts["actions"]["PASS"], 50)
        self.assertEqual(report.counts["temporal"]["WARNING"], 50)
        self.assertEqual(report.counts["provenance"]["NOT_REVIEWED"], 50)
        self.assertFalse(any(c.has_failures for c in report.cases))
        self.assertEqual(report.simulation_readiness, "NOT_REVIEWED")

    def test_recoverable_transport_failure_does_not_drop_later_case(self):
        case = load_cases(DATASET)[0]
        lines = "{\n" + json.dumps(case) + "\n"
        with patch("validation.run_validation.load_cases", side_effect=DatasetFormatError("transport failure")):
            with patch.object(Path, "open", return_value=io.StringIO(lines)):
                report = validate_dataset(DATASET)
        self.assertEqual(report.case_count, 2)
        self.assertEqual(report.cases[0].results[0].status, "FAIL")
        self.assertEqual([r.status for r in report.cases[0].results[1:]], ["BLOCKED"] * 3)
        self.assertEqual(report.cases[1].results[0].status, "PASS")

    def test_bad_schema_continues_and_duplicate_ids_are_reported(self):
        case = load_cases(DATASET)[0]
        bad = deepcopy(case)
        bad.pop("evaluation")
        with patch("validation.run_validation.load_cases", return_value=[bad, case]):
            report = validate_dataset(DATASET)
        self.assertEqual(report.case_count, 2)
        self.assertTrue(report.cases[0].has_failures)
        self.assertIn("DUPLICATE_DATASET_CASE", report.issue_counts)
        self.assertIn("DUPLICATE_DATASET_VISIT", report.issue_counts)
