"""Implemented transport contracts, using the immutable real dataset only."""
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from app.locdi.dataset_loader import DatasetFormatError, load_case, load_cases

DATASET = Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl"


class DatasetLoaderContracts(unittest.TestCase):
    def test_real_dataset_transport_and_identity_lookup(self):
        cases = load_cases(DATASET)
        self.assertEqual(len(cases), 50)
        self.assertEqual(load_case(DATASET, cases[0]["case_id"]), cases[0])
        self.assertIn("evaluation", cases[0])
        self.assertNotIn("evaluation", cases[0]["agent_visible"])

    def test_json_error_reports_line_without_echoing_data(self):
        with patch.object(Path, "open", return_value=io.StringIO("{\n")):
            with self.assertRaisesRegex(DatasetFormatError, "line 1: invalid JSON"):
                load_cases(DATASET)

    def test_duplicate_real_case_is_rejected(self):
        line = json.dumps(load_cases(DATASET)[0])
        with patch.object(Path, "open", return_value=io.StringIO(line + "\n" + line + "\n")):
            with self.assertRaisesRegex(DatasetFormatError, "line 2: duplicate case_id"):
                load_cases(DATASET)

    def test_unknown_identity_raises(self):
        with self.assertRaises(KeyError):
            load_case(DATASET, "__missing_case_id__")
