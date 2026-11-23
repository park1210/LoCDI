"""Role projection and release gating without constructing an LLM client."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from app.locdi.case_adapter import CaseAdaptationError, adapt_case
from app.locdi.dataset_loader import load_cases
from app.core.clinical_observation import doctor_observation, patient_observation, encounter_evidence, initial_symptoms
from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.hospital_llm_policy import HospitalLLMPolicy
from app.locdi.run_simulation import run_case
from test_simulation import SourceActionClient, smoke_options


class InformationBoundaryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases(Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]
        cls.adapted = adapt_case(cls.case)

    def runtime(self):
        case = deepcopy(self.adapted.runtime_case)
        return HospitalCaseRuntime(case["case_id"], case, 0, 0,
                                   vital_signs=deepcopy(case["doctor_visible_info"].get("initial_vitals", {})),
                                   current_symptoms=initial_symptoms(case))

    def test_gold_provenance_and_future_events_never_reach_model_payload(self):
        rt = self.runtime()
        doctor, patient = doctor_observation(rt), patient_observation(rt.case_data, rt)
        forbidden = {"evaluation", "task_annotations", "gold", "gold_label", "provenance", "data_replay",
                     "source_sha256", "source_event_store", "current_encounter_events", "diagnosis_labels"}
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(v) for v in value.values()), set())
            if isinstance(value, list):
                return set().union(*(keys(v) for v in value), set())
            return set()
        self.assertFalse(forbidden & keys(doctor))
        self.assertFalse(forbidden & keys(patient))
        self.assertFalse((forbidden - {"current_encounter_events"}) & set(self.adapted.runtime_case))
        self.assertEqual(doctor["objective_findings"], [])
        self.assertEqual(doctor["history"], [])
        def strings(value):
            if isinstance(value, str):
                return {value}
            if isinstance(value, dict):
                return set().union(*(strings(v) for v in value.values()), set())
            if isinstance(value, list):
                return set().union(*(strings(v) for v in value), set())
            return set()
        for event in rt.case_data["current_encounter_events"]:
            if event["available_minute"] > rt.current_time_min:
                self.assertNotIn(event["evidence"], strings(doctor))
                self.assertNotIn(event["evidence"], strings(patient))

    def test_future_test_source_is_gated_at_actual_minute(self):
        rt = self.runtime()
        event = next(e for e in rt.case_data["current_encounter_events"] if e["action"] == "Test")
        self.assertGreater(event["available_minute"], 0)
        rt.current_time_min = event["available_minute"] - 1
        self.assertIsNone(encounter_evidence(rt, "Test", event["target"]))
        self.assertEqual(HospitalLLMPolicy.build_objective_finding(rt, event["target"]), "")
        rt.current_time_min = event["available_minute"]
        self.assertEqual(encounter_evidence(rt, "Test", event["target"])["evidence"], event["evidence"])
        self.assertIn(event["evidence"], HospitalLLMPolicy.build_objective_finding(rt, event["target"]))
        # Retrieval alone does not deliver a result to doctor_observation.
        self.assertEqual(doctor_observation(rt)["objective_findings"], [])

    def test_hidden_future_source_store_is_not_in_runtime(self):
        self.assertTrue(any(s["relative_to_decision"] == "after_decision"
                            for s in self.adapted.environment["source_event_store"]))
        self.assertNotIn("source_event_store", self.adapted.runtime_case)
        self.assertNotIn("provenance", self.adapted.runtime_case)
        self.assertNotIn("evaluation", self.adapted.runtime_case)

    def test_gold_injection_into_visible_input_is_rejected(self):
        case = deepcopy(self.case)
        case["agent_visible"]["current_initial_state"]["doctor_visible_info"]["arrival_info"]["gold"] = deepcopy(case["evaluation"])
        with self.assertRaises(CaseAdaptationError) as caught:
            adapt_case(case)
        self.assertIn("PRIVATE_FIELD_IN_VISIBLE", {i.code for r in caught.exception.results for i in r.issues})

    def test_unknown_initial_metadata_is_not_silently_forwarded(self):
        case = deepcopy(self.case)
        case["agent_visible"]["current_initial_state"]["doctor_visible_info"]["arrival_info"]["unreleased"] = deepcopy(case["environment"])
        with self.assertRaises(CaseAdaptationError):
            adapt_case(case)

    def test_gold_inside_serialized_evidence_is_rejected(self):
        case = deepcopy(self.case)
        event = next(e for e in case["environment"]["current_encounter_events"] if e["action"] == "Test")
        payload = json.loads(event["evidence"])
        payload["records"][0]["gold"] = deepcopy(case["evaluation"])
        event["evidence"] = json.dumps(payload)
        with self.assertRaises(CaseAdaptationError) as caught:
            adapt_case(case)
        self.assertIn("TEST_EVIDENCE_FIELDS", {i.code for r in caught.exception.results for i in r.issues})

    def test_actual_role_call_payload_obeys_result_delivery_and_private_boundary(self):
        client = SourceActionClient()
        features = dict(smoke_options().runner_options, primary_router="adaptive")
        # ER-only must ignore the primary router, even without an adaptive utility.
        with patch("app.core.agent_registry.AgentRegistry.select_specialty",
                   side_effect=AssertionError("ER-only must not select a specialist")) as select:
            result = run_case(self.case, options=smoke_options(client, runner_options=features))
        select.assert_not_called()
        self.assertEqual(result["status"], "completed")
        final = result["execution"]["visits"][0]["final_state"]
        delivered = {e["payload"]["requested_test"]: e["minute"] for e in final["event_log"]
                     if e["event_type"] == "result_received"}
        events = [e for e in self.adapted.runtime_case["current_encounter_events"] if e["action"] == "Test"]
        self.assertEqual(set(delivered), {e["target"] for e in events})

        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(v) for v in value.values()), set())
            if isinstance(value, list):
                return set().union(*(keys(v) for v in value), set())
            return set()

        def strings(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, dict):
                return [s for v in value.values() for s in strings(v)]
            if isinstance(value, list):
                return [s for v in value for s in strings(v)]
            return []

        forbidden = {"evaluation", "task_annotations", "gold", "gold_label", "provenance",
                     "source_event_store", "data_replay", "source_sha256", "current_encounter_events"}
        self.assertTrue(client.calls)
        observed = set()
        for call in client.calls:
            payload = call["payload"]  # Actual serialized user_prompt received through role_call.
            self.assertFalse(keys(payload) & forbidden)
            minute = payload["observation"]["elapsed_minutes"]
            visible = strings(payload)
            for event in events:
                present = any(event["evidence"] in text for text in visible)
                if minute < delivered[event["target"]]:
                    self.assertFalse(present)
                if present:
                    self.assertGreaterEqual(minute, event["available_minute"])
                    self.assertGreaterEqual(minute, delivered[event["target"]])
                    observed.add(event["target"])
            for record in payload["observation"]["longitudinal_memory"]["records"]:
                self.assertEqual(record["patient_id"], self.case["patient_id"])
                self.assertNotEqual(record["visit_id"], self.case["agent_visible"]["index_encounter_id"])
        self.assertEqual(observed, set(delivered))  # Also prove results eventually reach the policy.
        self.assertFalse(keys(result) & (forbidden - {"current_encounter_events"}))
