"""Real LoCDI conversion, isolation and legacy encounter contracts."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from app.locdi.case_adapter import AdaptedCase, CaseAdaptationError, adapt_case, adapt_cases
from app.locdi.dataset_loader import load_cases
from app.core.clinical_observation import prepare_encounter
from app.core.longitudinal_memory import PatientMemoryStore, visit_identity

DATASET = Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl"


class CaseAdapterContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(DATASET)
        cls.adapted = adapt_case(cls.cases[0])

    def test_stores_have_distinct_names(self):
        self.assertTrue({
            "case_id", "agent_visible", "environment", "evaluation", "task_annotations",
            "provenance", "runtime_case"}.issubset(AdaptedCase.__dataclass_fields__))
        self.assertIn("data_replay", self.adapted.audit_metadata)
        self.assertNotIn("data_replay", self.adapted.environment)

    def test_invalid_input_has_sanitized_structured_error(self):
        with self.assertRaises(CaseAdaptationError) as context:
            adapt_case({})
        self.assertTrue(context.exception.results)
        self.assertNotIn(self.cases[0]["patient_id"], str(context.exception))

    def test_prepared_runtime_contract(self):
        runtime = self.adapted.runtime_case
        self.assertEqual(prepare_encounter(runtime), runtime)
        self.assertEqual(visit_identity(runtime), self.cases[0]["agent_visible"]["longitudinal"])
        records = [m for v in runtime["past_visits"] for m in v["records"]]
        self.assertEqual(len(PatientMemoryStore(records).snapshot()["records"]), len(records))
        self.assertEqual(runtime["current_encounter_events"], self.cases[0]["environment"]["current_encounter_events"])
        self.assertFalse(self.adapted.simulation_ready)
        self.assertEqual({r.validator for r in self.adapted.validation_results}, {"schema", "temporal", "actions", "provenance"})

    def test_original_and_stores_are_not_aliased(self):
        original = deepcopy(self.cases[0])
        adapted = adapt_case(original)
        self.assertEqual(original, self.cases[0])
        adapted.runtime_case["current_encounter_events"][0]["evidence"] = "changed only in runtime copy"
        self.assertEqual(adapted.environment["current_encounter_events"], original["environment"]["current_encounter_events"])
        self.assertEqual(original, self.cases[0])
        adapted.evaluation.clear()
        self.assertTrue(original["evaluation"])

    def test_all_50_prepare_contracts_without_dataset_changes(self):
        before = hashlib.sha256(DATASET.read_bytes()).hexdigest()
        adapted = adapt_cases(self.cases)
        self.assertEqual(len(adapted), 50)
        self.assertEqual([c.case_id for c in adapted], [c["case_id"] for c in self.cases])
        for item, source in zip(adapted, self.cases):
            self.assertEqual(item.runtime_case["past_visits"], source["agent_visible"]["past_visits"])
            self.assertFalse(any(r.status in {"FAIL", "BLOCKED"} for r in item.validation_results))
        self.assertEqual(hashlib.sha256(DATASET.read_bytes()).hexdigest(), before)

    def test_batch_duplicate_is_not_repaired_or_merged(self):
        with self.assertRaises(CaseAdaptationError):
            adapt_cases([self.cases[0], self.cases[0]])
