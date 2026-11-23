"""Known chronology failures versus unobserved release-time semantics."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import unittest
from app.locdi.dataset_loader import load_cases
from app.core.longitudinal_memory import timestamp
from validation.temporal_validator import validate_temporal


class TemporalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases(Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]

    def test_real_visit_and_observation_cutoffs(self):
        result = validate_temporal(self.case)
        self.assertEqual(result.status, "WARNING")
        codes = {i.code for i in result.issues}
        self.assertIn("RELEASE_TIME_UNOBSERVED", codes)
        self.assertIn("VISIT_END_NOT_RECORDED", codes)
        self.assertFalse(any(i.severity == "FAIL" for i in result.issues))

    def test_past_visit_order_is_enforced(self):
        case = deepcopy(self.case)
        visit = case["agent_visible"]["past_visits"][0]
        visit["started_at"] = case["agent_visible"]["index_encounter_start"]
        for record in visit["records"]:
            record["visit_started_at"] = visit["started_at"]
        self.assertIn("PAST_VISIT_ORDER", {i.code for i in validate_temporal(case).issues})

    def test_future_and_reversed_memory_times_fail(self):
        case = deepcopy(self.case)
        case["agent_visible"]["past_visits"][0]["records"][0]["available_at"] = (
            timestamp(case["agent_visible"]["index_encounter_start"]) + timedelta(minutes=1)).isoformat()
        self.assertIn("FUTURE_MEMORY", {i.code for i in validate_temporal(case).issues})
        case = deepcopy(self.case)
        m = case["agent_visible"]["past_visits"][0]["records"][0]
        m["available_at"] = (timestamp(m["observed_at"]) - timedelta(seconds=1)).isoformat()
        self.assertIn("MEMORY_ORDER", {i.code for i in validate_temporal(case).issues})

    def test_evidence_recorded_after_projected_release_fails(self):
        case = deepcopy(self.case)
        event = next(e for e in case["environment"]["current_encounter_events"] if e["action"] == "Test")
        payload = json.loads(event["evidence"])
        payload["records"][0]["recorded_at"] = (timestamp(case["agent_visible"]["index_encounter_start"]) + timedelta(minutes=5)).isoformat()
        event["evidence"] = json.dumps(payload)
        self.assertIn("PROJECTED_FUTURE_RECORD", {i.code for i in validate_temporal(case).issues})

    def test_invalid_minute_and_naive_time_do_not_pass(self):
        case = deepcopy(self.case)
        case["environment"]["current_encounter_events"][0]["available_minute"] = True
        self.assertEqual(validate_temporal(case).status, "BLOCKED")
        case = deepcopy(self.case)
        case["agent_visible"]["index_encounter_start"] = "2020-01-01T00:00:00"
        case["agent_visible"]["longitudinal"]["started_at"] = "2020-01-01T00:00:00"
        self.assertIn("TIMESTAMP", {i.code for i in validate_temporal(case).issues})

    def test_source_interval_and_visible_symptom_future_fail(self):
        case = deepcopy(self.case)
        s = case["environment"]["source_event_store"][0]
        s["timestamp_lower_epoch_ms"] = s["timestamp_upper_epoch_ms"] + 1
        self.assertIn("SOURCE_INTERVAL", {i.code for i in validate_temporal(case).issues})
        case = deepcopy(self.case)
        case["provenance"]["source_case_audit"]["symptom_sources"][0]["source_timestamp"] = (
            timestamp(case["agent_visible"]["index_encounter_start"]) + timedelta(seconds=1)).isoformat()
        self.assertIn("FUTURE_INITIAL_SYMPTOM", {i.code for i in validate_temporal(case).issues})
