"""Real engine smoke tests; fixtures and deterministic clients are test-only."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch
from app.locdi.run_simulation import (OfflineClient, ResearchExecutionBlocked, SimulationConfigurationError,
                                     SimulationOptions, run_case, run_cases, translate_runner_options)
from app.locdi.dataset_loader import load_cases
from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.longitudinal_runner import run_with_recovery
from app.locdi.episode_exporter import export_episode, export_summary


def _schema_matches(value, schema):
    """Check the subset of JSON Schema actually supplied by Core, offline."""
    if "anyOf" in schema:
        return any(_schema_matches(value, branch) for branch in schema["anyOf"])
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "not" in schema and _schema_matches(value, schema["not"]):
        return False
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict) or not set(schema.get("required", [])).issubset(value):
            return False
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(props):
            return False
        return all(_schema_matches(v, props[k]) for k, v in value.items() if k in props)
    if kind == "array":
        if not isinstance(value, list) or len(value) > schema.get("maxItems", len(value)):
            return False
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            return False
        return all(_schema_matches(x, schema.get("items", {})) for x in value)
    if kind == "string":
        return isinstance(value, str) and len(value) <= schema.get("maxLength", len(value))
    if kind == "integer":
        return type(value) is int and schema.get("minimum", value) <= value <= schema.get("maximum", value)
    return True


class SourceActionClient(OfflineClient):
    """Never predicts a gold diagnosis; chooses only the Core's source menu."""
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def chat_json(self, system_prompt, user_prompt, model=None, *, schema=None):
        payload = json.loads(user_prompt)
        self.calls.append({"payload": payload, "schema": deepcopy(schema), "model": model})
        if self.failure == "timeout":
            raise TimeoutError("private client details must not appear in results")
        if self.failure == "invalid_json":
            return "not-json"
        if self.failure == "network":
            socket.create_connection(("127.0.0.1", 9))
            raise AssertionError("socket guard failed")
        candidates = payload["candidate_actions"]
        chosen = next((c for action in ("ask_patient", "request_test") for c in candidates if c["action"] == action), None)
        if chosen is None:
            if not payload["allow_unresolved_finalization"]:
                raise AssertionError("Offline fixture cannot invent a supported diagnosis")
            action, target = "diagnose", "Unresolved"
        else:
            action, target = chosen["action"], chosen["target"]
        row = {"action": action, "target": target, "rationale": "Offline source-action contract check.",
               "assessment": "Development test; no diagnosis asserted.", "department": "general_medicine",
               "certainty": "unresolved", "evidence_indices": []}
        if payload["observation"].get("longitudinal_memory", {}).get("records"):
            row["memory_ids"] = []
        if not _schema_matches(row, schema):
            raise AssertionError("Deterministic test response violates the actual Core JSON Schema")
        self.calls[-1]["response"] = deepcopy(row)
        return json.dumps(row)


def smoke_options(client=None, **changes):
    """This minimal registry is a test fixture, not a study artifact."""
    manifest = {"schema_version": 1, "agents": [{"agent_id": "offline-er", "role": "er", "department": "",
                 "model": "offline-contract-client", "version": "test-only-v1", "keywords": []}]}
    options = SimulationOptions(minute_limit=120, registry_manifest=manifest,
                runner_options={"memory_mode": "selective", "top_k": 5, "routing_mode": "er_only",
                                "primary_router": "rule_similarity", "voi_enabled": False,
                                "action_selector": "specialist_heuristic"},
                client=client if client is not None else SourceActionClient(), accept_recorded_release_assumption=True)
    return replace(options, **changes)


class SimulationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl")[:2]

    def test_unprepared_entry_never_reaches_runner(self):
        options = SimulationOptions()
        self.assertFalse(options.allow_llm)
        with patch("app.locdi.run_simulation.run_with_recovery") as runner:
            with self.assertRaises(SimulationConfigurationError):
                run_case(self.cases[0], options=options)
            with self.assertRaises(SimulationConfigurationError):
                run_cases([], options=options)
            runner.assert_not_called()

    def test_export_never_claims_to_write(self):
        with self.assertRaises(ValueError):
            export_episode({}, "not-written", context=None)
        with self.assertRaises(ValueError):
            export_summary({}, "not-written")

    def test_runner_integration_without_external_llm(self):
        client = SourceActionClient()
        original = deepcopy(self.cases[0])
        with patch("app.locdi.run_simulation.run_with_recovery", wraps=run_with_recovery) as runner, \
             patch("app.core.longitudinal_runner.HospitalParallelScheduler", wraps=HospitalParallelScheduler) as scheduler, \
             patch("app.core.hospital_parallel_scheduler.HospitalCaseRuntime", wraps=HospitalCaseRuntime) as runtime, \
             patch("app.core.hospital_llm_policy.OllamaClient", side_effect=AssertionError("No SDK constructor")) as sdk:
            result = run_case(self.cases[0], options=smoke_options(client))
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["development_only"] and result["mock_execution"])
        self.assertFalse(result["research_evaluation_eligible"] or result["simulation_ready"])
        self.assertEqual(result["clinical_metrics"], "NOT_EVALUATED")
        self.assertEqual(result["execution"]["committed_count"], 1)
        runner.assert_called_once()
        scheduler.assert_called_once()
        runtime.assert_called_once()
        sdk.assert_not_called()
        passed = runner.call_args.args[0][0]
        self.assertIn("current_encounter_events", passed)
        self.assertEqual(passed["past_visits"], original["agent_visible"]["past_visits"])
        self.assertFalse({"evaluation", "task_annotations", "provenance", "source_event_store", "data_replay"} & set(passed))
        self.assertEqual(runtime.call_args.kwargs["case_data"]["case_id"], original["case_id"])
        self.assertEqual(runner.call_args.kwargs["memory_source"], "fixed")
        memory = runner.call_args.kwargs["initial_memory"]
        self.assertEqual({r["patient_id"] for r in memory}, {original["patient_id"]})
        self.assertEqual(self.cases[0], original)
        self.assertTrue(client.calls)
        self.assertTrue(all(_schema_matches(call["response"], call["schema"]) for call in client.calls))
        state = next(iter(result["execution"]["visits"][0]["final_state"]["cases"].values()))
        self.assertTrue(state["is_completed"])
        self.assertIn("diagnosis_unresolved", state["alerts"])
        self.assertEqual(len(state["completed_tests"]), 4)
        self.assertEqual(len(state["question_history"]), 1)

    def test_minute_limit_returns_real_incomplete_state(self):
        client = SourceActionClient()
        result = run_case(self.cases[0], options=smoke_options(client, minute_limit=1))
        self.assertEqual(result["status"], "incomplete")
        visit = result["execution"]["visits"][0]
        self.assertEqual(visit["stop_reason"], "visit_minute_limit_reached")
        self.assertEqual(visit["stop_detail"]["current_minute"], 1)
        self.assertEqual(result["execution"]["committed_count"], 0)
        self.assertFalse(next(iter(visit["final_state"]["cases"].values()))["is_completed"])
        self.assertEqual(client.calls, [])

    def test_client_error_is_a_real_failed_report(self):
        for failure in ("timeout", "invalid_json", "network"):
            with self.subTest(failure=failure):
                result = run_case(self.cases[0], options=smoke_options(SourceActionClient(failure)))
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["execution"]["committed_count"], 0)
                self.assertEqual(result["execution"]["visits"][0]["stop_reason"], "role_call_failed")
                state = next(iter(result["execution"]["visits"][0]["final_state"]["cases"].values()))
                self.assertFalse(state["is_completed"])
                self.assertEqual(state["llm_audit"][-1]["status"], "error")
                self.assertNotIn("private client details", json.dumps(result))

    def test_unexpected_core_error_has_no_fabricated_state(self):
        with patch("app.locdi.run_simulation.run_with_recovery", side_effect=RuntimeError("private exception")):
            result = run_case(self.cases[0], options=smoke_options())
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["execution"])
        self.assertEqual(result["errors"][0]["error_type"], "RuntimeError")
        self.assertNotIn("private exception", json.dumps(result))

    def test_research_live_clients_and_missing_prerequisites_are_blocked(self):
        invalid = [smoke_options(purpose="research_evaluation"), smoke_options(allow_llm=True),
                   smoke_options(client=object()), smoke_options(registry_manifest=None),
                   smoke_options(accept_recorded_release_assumption=False), smoke_options(minute_limit=True)]
        with patch("app.locdi.run_simulation.run_with_recovery") as runner:
            for options in invalid:
                with self.assertRaises(SimulationConfigurationError):
                    run_case(self.cases[0], options=options)
            runner.assert_not_called()

    def test_configuration_translation_is_explicit_and_separates_routers(self):
        root = Path(__file__).resolve().parents[2] / "experiments/configs"
        for i in range(5):
            config = json.loads((root / ("B{}.json".format(i))).read_text(encoding="utf-8"))
            features = deepcopy(config["features"])
            core = translate_runner_options(features, memory_source="fixed")
            self.assertEqual(core["mode"], features["memory_mode"])
            self.assertEqual(core["top_k"], 1 if i < 2 else 5)
            self.assertEqual(core["routing_mode"], features["routing_mode"])
            self.assertEqual(core["primary_router"], features["primary_router"])
            self.assertEqual(config["features"], features)
            self.assertFalse(config["ready"])
        bad = dict(smoke_options().runner_options)
        bad["top_k"] = None
        with self.assertRaises(SimulationConfigurationError):
            translate_runner_options(bad, memory_source="fixed")
        bad = dict(smoke_options().runner_options, callback="not allowed")
        with self.assertRaises(SimulationConfigurationError):
            translate_runner_options(bad, memory_source="fixed")

    def test_voi_and_adaptive_require_real_artifacts_only_when_active(self):
        features = dict(smoke_options().runner_options, voi_enabled=True, action_selector="bayesian_voi")
        with self.assertRaises(SimulationConfigurationError):
            run_case(self.cases[0], options=smoke_options(runner_options=features))
        features = dict(smoke_options().runner_options, routing_mode="adaptive", primary_router="adaptive")
        with self.assertRaises(SimulationConfigurationError):
            run_case(self.cases[0], options=smoke_options(runner_options=features))

    def test_extra_or_future_initial_memory_is_rejected(self):
        records = [deepcopy(r) for v in self.cases[0]["agent_visible"]["past_visits"] for r in v["records"]]
        records[0]["available_at"] = self.cases[0]["agent_visible"]["index_encounter_start"]
        with self.assertRaises(SimulationConfigurationError):
            run_case(self.cases[0], options=smoke_options(initial_memory=records, minute_limit=1))

    def test_two_real_patients_have_separate_memory_contexts(self):
        result = run_cases(self.cases, options=smoke_options())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(len(result["execution"]["visits"]), 2)
        for visit in result["execution"]["visits"]:
            self.assertTrue(visit["memory_context"]["records"])
            self.assertEqual({r["patient_id"] for r in visit["memory_context"]["records"]}, {visit["identity"]["patient_id"]})
            for r in visit["memory_context"]["records"]:
                self.assertNotEqual(r["visit_id"], visit["identity"]["visit_id"])
