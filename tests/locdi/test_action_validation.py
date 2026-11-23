"""Registration is implemented; execution/availability are explicitly pending."""
import unittest
from copy import deepcopy
import json
from pathlib import Path
from app.locdi.dataset_loader import load_cases
from app.locdi.action_registry import ActionSpec, ActionRegistry, register_action, get_available_actions
from app.locdi.action_adapter import normalize_action, map_action_to_runtime
from app.locdi.action_audit import audit_action, summarize_actions
from validation.action_validator import validate_actions


class ActionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases(Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]

    def test_registration_is_unique_and_instance_local(self):
        spec = ActionSpec("ask", "Ask", "question", "ask_history")
        registry = ActionRegistry()
        register_action(registry, spec)
        self.assertEqual(registry.get_action("ask"), spec)
        with self.assertRaises(ValueError):
            register_action(registry, spec)
        with self.assertRaises(KeyError):
            ActionRegistry().get_action("ask")

    def test_pending_functions_fail_explicitly(self):
        registry = ActionRegistry()
        operations = [lambda: get_available_actions(registry, {}),
                      lambda: normalize_action({}, registry=registry),
                      lambda: map_action_to_runtime({}, registry=registry)]
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(NotImplementedError):
                    operation()
        # Phase 1.8 implements audits; unknown input remains explicitly unknown.
        self.assertEqual(audit_action({}, {})["availability"], "NOT_RECORDED")
        self.assertIsNone(audit_action({}, {})["execution_success"])
        self.assertEqual(summarize_actions([])["decision_count"], 0)

    def test_real_action_and_source_compatibility(self):
        result = validate_actions(self.case)
        self.assertEqual(result.status, "PASS")
        self.assertIn("clinical_action_appropriateness", result.not_reviewed)

    def test_action_evidence_tampering_is_rejected(self):
        case = deepcopy(self.case)
        e = next(e for e in case["environment"]["current_encounter_events"] if e["action"] == "Test")
        p = json.loads(e["evidence"])
        p["records"][0]["recorded_evidence"]["value"] += " changed"
        e["evidence"] = json.dumps(p)
        self.assertIn("TEST_EVIDENCE_SOURCE", {i.code for i in validate_actions(case).issues})
        case = deepcopy(self.case)
        e = next(e for e in case["environment"]["current_encounter_events"] if e["action"] == "Ask")
        p = json.loads(e["evidence"])
        p["facts"][0]["value"] += " changed"
        e["evidence"] = json.dumps(p)
        self.assertIn("ASK_EVIDENCE_SOURCE", {i.code for i in validate_actions(case).issues})

    def test_invalid_action_duplicate_target_and_serialization(self):
        for field, value in (("action", "Unsupported"), ("evidence", "not JSON")):
            case = deepcopy(self.case)
            case["environment"]["current_encounter_events"][0][field] = value
            self.assertEqual(validate_actions(case).status, "FAIL")
        case = deepcopy(self.case)
        e = case["environment"]["current_encounter_events"]
        e[2]["target"] = e[1]["target"]
        self.assertIn("DUPLICATE_ACTION", {i.code for i in validate_actions(case).issues})

    def test_missing_test_source_and_question_target(self):
        case = deepcopy(self.case)
        test = next(e for e in case["environment"]["current_encounter_events"] if e["action"] == "Test")
        case["environment"]["source_event_store"] = [s for s in case["environment"]["source_event_store"] if s["event_id"] != test["event_id"]]
        self.assertIn("TEST_SOURCE_MISSING", {i.code for i in validate_actions(case).issues})
        case = deepcopy(self.case)
        case["environment"]["current_encounter_events"][0]["target"] = "not a question"
        self.assertIn("ASK_TARGET", {i.code for i in validate_actions(case).issues})
