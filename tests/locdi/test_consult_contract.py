"""TEST_ONLY structural fixtures, never real consultation opinions or Gold."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case, CaseAdaptationError, inspect_consult_bindings
from app.locdi.episode_exporter import build_episode, ENGINE_ACTIONS
from app.locdi.run_simulation import run_case
from app.core.clinical_observation import prepare_encounter, doctor_observation, patient_observation, encounter_evidence
from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.hospital_llm_policy import HospitalLLMPolicy
from validation.action_validator import validate_actions, validate_consult_sources, CONSULT_PROPOSAL_VERSION
from validation.schema_validator import validate_schema
from test_simulation import smoke_options, SourceActionClient

ROOT = Path(__file__).resolve().parents[2]
MARKER = "TEST_ONLY_STRUCTURE_NO_CLINICAL_OPINION"


def consult_fixture(base):
    """A proposal shape only. No original source file or clinical review exists."""
    case = deepcopy(base)
    start = datetime.fromisoformat(case["agent_visible"]["index_encounter_start"].replace("Z", "+00:00"))
    recorded, available = start + timedelta(minutes=9), start + timedelta(minutes=10)
    evidence = {"kind": "TEST_ONLY", "placeholder": MARKER}
    link = {"source_file": "TEST_ONLY/consult_source.jsonl", "source_row": 2}
    source = {"event_id": "TEST_ONLY_CONSULT", "case_id": case["case_id"], "patient_id": case["patient_id"],
        "encounter_id": case["agent_visible"]["index_encounter_id"], "timestamp": recorded.isoformat(),
        "timing_precision": "millisecond", "timestamp_lower_epoch_ms": int(recorded.timestamp() * 1000),
        "timestamp_upper_epoch_ms": int(recorded.timestamp() * 1000), "evidence": evidence,
        "target": {"department": "cardiology"}, "source_metadata": {**link, "source_record": {
            "PATIENT": case["patient_id"], "ENCOUNTER": case["agent_visible"]["index_encounter_id"], "TEST_ONLY": True}},
        "availability_status": "TEST_ONLY_UNVERIFIED", "reveal_status": "TEST_ONLY_HIDDEN",
        "relative_to_decision": "after_decision", "event_type": "TEST_ONLY", "action_type_candidate": "Consult"}
    payload = {"schema_version": CONSULT_PROPOSAL_VERSION, "contract_status": "PROPOSED", "purpose": "TEST_ONLY",
        "recorded_at": recorded.isoformat(), "timing_precision": "millisecond", "source_record_link": link,
        "recorded_evidence": evidence, "available_at": available.isoformat(),
        "release_time_basis": "DATASET_DECLARED", "clinical_review_status": "NOT_REVIEWED"}
    event = {"event_id": source["event_id"], "action": "Consult", "target": "cardiology",
             "available_minute": 10, "evidence": json.dumps(payload)}
    case["environment"]["source_event_store"].append(source)
    case["environment"]["current_encounter_events"].append(event)
    return case


def change_payload(case, **changes):
    event = case["environment"]["current_encounter_events"][-1]
    payload = json.loads(event["evidence"])
    payload.update(changes)
    event["evidence"] = json.dumps(payload)


class ConsultSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_cases(ROOT / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]
        cls.registry = deepcopy(smoke_options().registry_manifest)
        cls.registry["agents"].append({"agent_id": "TEST_ONLY_SPECIALTY", "role": "specialty", "department": "cardiology",
            "model": "TEST_ONLY_NO_MODEL", "version": "TEST_ONLY", "keywords": []})

    def check_failure(self, case, code=None):
        result = validate_consult_sources(case)
        self.assertIn(result.status, {"FAIL", "BLOCKED"})
        if code:
            self.assertIn(code, {i.code for i in result.issues})
        self.assertNotIn(MARKER, str(result.issues))

    def test_proposal_structure_is_consistent_but_never_clinically_passes(self):
        case = consult_fixture(self.base)
        self.assertEqual(validate_schema(case).status, "PASS")
        for result in (validate_consult_sources(case), validate_actions(case)):
            self.assertEqual(result.status, "NOT_REVIEWED")
            self.assertFalse(any(i.severity == "FAIL" for i in result.issues))
            self.assertIn("CONSULT_SOURCE_UNREVIEWED", {i.code for i in result.issues})
        with self.assertRaises(CaseAdaptationError):
            adapt_case(case, registry_manifest=self.registry)

    def test_missing_source_and_wrong_source_type(self):
        case = consult_fixture(self.base)
        case["environment"]["source_event_store"].pop()
        self.check_failure(case, "CONSULT_SOURCE_MISSING")
        case = consult_fixture(self.base)
        case["environment"]["source_event_store"][-1]["action_type_candidate"] = "Test"
        self.check_failure(case, "CONSULT_SOURCE_TYPE")

    def test_invalid_department_and_source_department_mismatch(self):
        case = consult_fixture(self.base)
        case["environment"]["current_encounter_events"][-1]["target"] = "TEST_ONLY_UNKNOWN"
        self.check_failure(case, "CONSULT_TARGET")
        case = consult_fixture(self.base)
        case["environment"]["source_event_store"][-1]["target"]["department"] = "neurology"
        self.check_failure(case, "CONSULT_TARGET_SOURCE")

    def test_wrong_patient_encounter_and_original_row_identity(self):
        for field in ("patient_id", "encounter_id"):
            case = consult_fixture(self.base)
            case["environment"]["source_event_store"][-1][field] = "TEST_ONLY_WRONG"
            self.check_failure(case)
        for field in ("PATIENT", "ENCOUNTER"):
            case = consult_fixture(self.base)
            case["environment"]["source_event_store"][-1]["source_metadata"]["source_record"][field] = "TEST_ONLY_WRONG"
            self.check_failure(case, "CONSULT_RECORD_IDENTITY")

    def test_serialized_evidence_source_link_and_extra_fields(self):
        for changes, code in (({"recorded_evidence": {"TEST_ONLY": "changed"}}, "CONSULT_EVIDENCE_SOURCE"),
            ({"source_record_link": {"source_file": "TEST_ONLY/wrong", "source_row": 2}}, "CONSULT_RECORD_LINK_MISMATCH"),
            ({"gold": "TEST_ONLY"}, "CONSULT_EVIDENCE_FIELDS"),
            ({"purpose": []}, "CONSULT_PROPOSAL_STATUS"),
            ({"release_time_basis": {}}, "CONSULT_RELEASE_BASIS"),
            ({"clinical_review_status": "expert_reviewed"}, "CONSULT_REVIEW_CLAIM")):
            case = consult_fixture(self.base)
            change_payload(case, **changes)
            self.check_failure(case, code)
        case = consult_fixture(self.base)
        case["environment"]["current_encounter_events"][-1]["evidence"] = "not-json"
        self.check_failure(case, "CONSULT_EVIDENCE_JSON")

    def test_record_link_safety_and_nonfinite_evidence(self):
        for path in ("../TEST_ONLY", "C:\\TEST_ONLY", "/TEST_ONLY"):
            case = consult_fixture(self.base)
            case["environment"]["source_event_store"][-1]["source_metadata"]["source_file"] = path
            self.check_failure(case, "CONSULT_SOURCE_RECORD_LINK")
        case = consult_fixture(self.base)
        change_payload(case, recorded_evidence={"TEST_ONLY": float("nan")})
        self.check_failure(case, "CONSULT_NONFINITE_EVIDENCE")

    def test_available_minute_and_recording_bounds(self):
        for minute in (-1, True, 1.5):
            case = consult_fixture(self.base)
            case["environment"]["current_encounter_events"][-1]["available_minute"] = minute
            self.check_failure(case)
        case = consult_fixture(self.base)
        case["environment"]["current_encounter_events"][-1]["available_minute"] = 8
        self.check_failure(case, "CONSULT_SOURCE_FUTURE")
        case = consult_fixture(self.base)
        source = case["environment"]["source_event_store"][-1]
        change_payload(case, available_at=source["timestamp"], recorded_at="TEST_ONLY_WRONG")
        self.check_failure(case, "CONSULT_RECORD_TIME_MISMATCH")
        case = consult_fixture(self.base)
        change_payload(case, available_at=self.base["agent_visible"]["index_encounter_start"])
        self.check_failure(case, "CONSULT_AVAILABILITY_TIME")

    def test_precision_and_release_claim_never_clear_review(self):
        case = consult_fixture(self.base)
        change_payload(case, release_time_basis="OBSERVED_DOCUMENTED")
        self.assertEqual(validate_consult_sources(case).status, "NOT_REVIEWED")
        case["environment"]["source_event_store"][-1]["timing_precision"] = "date"
        change_payload(case, timing_precision="date")
        self.assertIn("CONSULT_PRECISION_UNREVIEWED", {i.code for i in validate_consult_sources(case).issues})
        self.assertEqual(validate_consult_sources(self.base).status, "NOT_REVIEWED")

    def test_registry_binding_checks_do_not_clear_source_review(self):
        case = consult_fixture(self.base)
        self.assertEqual(inspect_consult_bindings(case).status, "BLOCKED")
        self.assertEqual(inspect_consult_bindings(case, smoke_options().registry_manifest).status, "FAIL")
        self.assertEqual(inspect_consult_bindings(case, self.registry).status, "PASS")
        with self.assertRaises(CaseAdaptationError) as error:
            adapt_case(case, registry_manifest=smoke_options().registry_manifest)
        self.assertIn("CONSULT_SPECIALTY_UNBOUND", {i.code for r in error.exception.results for i in r.issues})
        self.assertEqual(adapt_case(self.base).runtime_case, adapt_case(self.base, registry_manifest=self.registry).runtime_case)

    def test_low_level_engine_private_visibility_and_release_boundary(self):
        # Bypass execution Adapter ONLY to inspect the existing low-level shape.
        # This is not a validated Consult execution or a new clinical example.
        fixture = consult_fixture(self.base)
        engine = deepcopy(adapt_case(self.base).runtime_case)
        event = fixture["environment"]["current_encounter_events"][-1]
        engine["current_encounter_events"].append(deepcopy(event))
        prepared = prepare_encounter(engine)
        self.assertEqual(prepared["current_encounter_events"][-1], event)
        runtime = HospitalCaseRuntime(engine["case_id"], prepared, 0, 0)
        for minute in (9, 10):
            runtime.current_time_min = minute
            self.assertNotIn(MARKER, json.dumps(doctor_observation(runtime)))
            self.assertNotIn(MARKER, json.dumps(patient_observation(prepared, runtime)))
        runtime.current_time_min = 9
        self.assertIsNone(encounter_evidence(runtime, "Consult", "cardiology"))
        runtime.current_time_min = 10
        self.assertEqual(encounter_evidence(runtime, "Consult", "cardiology"), event)
        policy = HospitalLLMPolicy({}, client=SourceActionClient())
        with patch.object(policy, "_decision", return_value={"action": "er_finalize", "assessment_summary": "TEST_ONLY", "rationale": "TEST_ONLY"}):
            completion = policy.complete_consult(runtime, department="cardiology", specialist_id="TEST_ONLY")
        self.assertEqual(completion["consult_note"], event["evidence"])
        self.assertNotIn("source_event_store", prepared)


class SpecialtyExportContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases(ROOT / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]
        cls.context = adapt_case(cls.case)
        cls.raw = run_case(cls.case, options=smoke_options())

    def test_specialty_finalize_links_without_diagnosis_target_claim(self):
        raw = deepcopy(self.raw)
        raw["test_fixture"] = "TEST_ONLY"
        state = raw["execution"]["visits"][0]["final_state"]
        final = state["event_log"][-1]
        self.assertEqual(final["action"], "er_finalize")
        final["action"] = "specialty_finalize"
        final["actor"]["role"] = "specialist"
        raw["telemetry"]["decisions"][-1]["role"] = "specialty"
        document = build_episode(raw, context=self.context)
        decision = document["workflow"]["decisions"][-1]
        audit = document["decision_audit"]["records"][-1]
        self.assertTrue(decision["execution_success"])
        self.assertTrue(audit["action_mapping_matches"])
        self.assertEqual(audit["executed_action_type"], "specialty_finalize")
        self.assertEqual(audit["execution_target_validation"], "NOT_CHECKED")
        self.assertEqual(document["result"]["action_counts"]["specialty_finalize"], 1)
        self.assertNotIn("er_finalize", document["result"]["action_counts"])
        self.assertTrue(document["mock_execution"])
        self.assertFalse(document["research_evaluation_eligible"])
        self.assertEqual(document["clinical_metrics"], "NOT_EVALUATED")
        text = json.dumps(document)
        for forbidden in (self.case["patient_id"], self.case["case_id"], self.case["agent_visible"]["index_encounter_id"]):
            self.assertNotIn(forbidden, text)

    def test_er_mapping_and_legacy_enum_are_preserved(self):
        doc = build_episode(self.raw, context=self.context)
        self.assertTrue(doc["workflow"]["decisions"][-1]["execution_success"])
        self.assertEqual(doc["decision_audit"]["records"][-1]["executed_action_type"], "er_finalize")
        self.assertIn("specialist_finalize", ENGINE_ACTIONS)
        raw = deepcopy(self.raw)
        raw["execution"]["visits"][0]["final_state"]["event_log"][-1]["action"] = "specialist_finalize"
        doc = build_episode(raw, context=self.context)
        self.assertEqual(doc["workflow"]["events"][-1]["executed_action"], "specialist_finalize")
        self.assertIsNone(doc["workflow"]["decisions"][-1]["execution_success"])


if __name__ == "__main__":
    unittest.main()
