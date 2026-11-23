"""TEST_ONLY math/contracts. No fitting, model calls or research artifacts.

Toy likelihoods (.9/.1) reuse HospitalWorkflow tests/test_voi_v2_contract.py;
routing sample shape reuses tests/test_routing_abstention.py. Labels/observations
here are abstract TEST_ONLY tokens, never LoCDI labels or clinical outcomes.
All toy artifacts live in memory only and cannot establish clinical readiness.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import RoutingUtility, observed_tokens
from app.core.diagnostic_belief import DiagnosisBelief, posterior, validation_utility
from app.core.information_outcomes import InformationOutcomes
from app.core.information_value import InformationPolicy, candidate_actions, select_action
from app.core.voi_planning import plan_voi
from app.core.clinical_observation import doctor_observation, initial_symptoms
from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.hospital_llm_policy import HospitalLLMPolicy
from app.core.hospital_flow import flow_config
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.run_simulation import _configuration, _capture_execution, SimulationConfigurationError, translate_runner_options
from test_simulation import smoke_options, SourceActionClient


def test_only_registry():
    return {"schema_version": 1, "agents": [
        {"agent_id": "TEST_ONLY_ER", "role": "er", "department": "", "model": "TEST_ONLY_NO_MODEL", "version": "TEST_ONLY", "keywords": []},
        {"agent_id": "TEST_ONLY_N", "role": "specialty", "department": "neurology", "model": "TEST_ONLY_NO_MODEL", "version": "TEST_ONLY", "keywords": ["test_only_signal"]},
        {"agent_id": "TEST_ONLY_C", "role": "specialty", "department": "cardiology", "model": "TEST_ONLY_NO_MODEL", "version": "TEST_ONLY", "keywords": []}]}


def test_only_routing(manifest, tie=False):
    # Explicitly predeclared toy sample table, NOT a fit_utility/training call.
    return {"schema_version": 2, "method": "jaccard_beta_accuracy_v2", "registry": deepcopy(manifest), "min_samples": 2,
            "samples": [{"patient_id": "TEST_ONLY_TRAIN", "decision_id": str(i), "agent_id": agent,
                         "correct": tie or agent == "TEST_ONLY_N", "tokens": ["test_only_signal"]}
                        for i in range(2) for agent in ("TEST_ONLY_N", "TEST_ONLY_C")]}


def test_only_voi():
    return {"schema_version": 2, "model_id": "TEST_ONLY", "revision": "TEST_ONLY",
        "provenance": {"synthetic": True, "review_status": "unreviewed", "assumptions": ["TEST_ONLY independent categorical toy"]},
        "diagnoses": {"TEST_ONLY_A": "general_medicine", "TEST_ONLY_B": "general_medicine"},
        "prior": {"TEST_ONLY_A": .5, "TEST_ONLY_B": .5},
        "terminal_utility": validation_utility(["TEST_ONLY_A", "TEST_ONLY_B"]), "horizon": 2,
        "actions": [{"id": "TEST_ONLY_T", "type": "request_test", "target": "TEST_ONLY_T", "outcomes": [
            {"id": "a", "likelihood": {"TEST_ONLY_A": .9, "TEST_ONLY_B": .1},
             "observation": {"source": "objective_findings", "text": "TEST_ONLY_T: TEST_ONLY_A"}, "supports_diagnoses": ["TEST_ONLY_A"]},
            {"id": "b", "likelihood": {"TEST_ONLY_A": .1, "TEST_ONLY_B": .9},
             "observation": {"source": "objective_findings", "text": "TEST_ONLY_T: TEST_ONLY_B"}, "supports_diagnoses": ["TEST_ONLY_B"]}]}]}


class RoutingReadinessContracts(unittest.TestCase):
    def test_registry_intersection_and_er_only_status(self):
        registry = AgentRegistry(test_only_registry())
        result = registry.route({}, ["neurology", "dermatology"], mode="er_only")
        self.assertEqual(result["candidate_departments"], ["neurology"])
        self.assertEqual(result["allowed_departments"], [])
        self.assertEqual(result["decision_status"], "disabled")
        self.assertEqual(registry.route({}, [], mode="adaptive")["decision_status"], "unavailable")

    def test_missing_utility_abstains_without_invented_estimates(self):
        registry = AgentRegistry(test_only_registry())
        result = registry.route({"chief_complaint": "test_only_signal"}, ["neurology"], mode="adaptive")
        self.assertEqual(result["decision_status"], "abstained")
        self.assertIsNone(result["selected_specialty"])
        self.assertEqual(result["allowed_departments"], ["neurology"])
        self.assertEqual(result["utility_estimates"], {"neurology": None})
        self.assertIsNone(result["utility_artifact_sha256"])

    def test_toy_score_formula_ties_and_insufficient_support(self):
        manifest = test_only_registry()
        registry = AgentRegistry(manifest, test_only_routing(manifest))
        result = registry.select_specialty({"chief_complaint": "test_only_signal"}, {}, ["cardiology", "neurology"], router="adaptive")
        self.assertEqual(result["selected_specialty"], "neurology")
        self.assertAlmostEqual(result["specialty_scores"]["neurology"], .75)
        self.assertAlmostEqual(result["specialty_scores"]["cardiology"], .25)
        tied = AgentRegistry(manifest, test_only_routing(manifest, tie=True)).route(
            {"chief_complaint": "test_only_signal"}, ["cardiology", "neurology"], mode="adaptive")
        self.assertEqual(tied["decision_status"], "abstained")
        self.assertIsNone(tied["selected_specialty"])
        self.assertEqual(tied["allowed_departments"], ["cardiology", "neurology"])
        self.assertEqual(tied["tied_departments"], ["cardiology", "neurology"])
        self.assertEqual(registry.route({"chief_complaint": "TEST_ONLY_UNSEEN"}, ["neurology"], mode="adaptive")["decision_status"], "abstained")

    def test_artifact_registry_and_complete_group_validation(self):
        manifest = test_only_registry()
        for mutation in ("registry", "samples", "min_samples"):
            artifact = test_only_routing(manifest)
            if mutation == "registry":
                artifact["registry"]["agents"][0]["version"] = "TEST_ONLY_CHANGED"
            elif mutation == "samples":
                artifact["samples"].pop()
            else:
                artifact["min_samples"] = True
            with self.assertRaises(ValueError):
                RoutingUtility(artifact, AgentRegistry(manifest))

    def test_missing_artifacts_block_before_runner_and_er_only_translation(self):
        options = smoke_options()
        features = dict(options.runner_options, routing_mode="adaptive", primary_router="adaptive")
        with self.assertRaisesRegex(SimulationConfigurationError, "routing utility"):
            _configuration(smoke_options(runner_options=features))
        features = dict(options.runner_options, voi_enabled=True, action_selector="bayesian_voi")
        with self.assertRaisesRegex(SimulationConfigurationError, "VoI model presence"):
            _configuration(smoke_options(runner_options=features))
        # Existing contract: ER Only ignores the primary adaptive router.
        _configuration(smoke_options(runner_options=dict(options.runner_options, primary_router="adaptive")))
        for mode in ("none", "full_history"):
            value = translate_runner_options(dict(options.runner_options, memory_mode=mode, top_k=None), memory_source="fixed")
            self.assertEqual(value["top_k"], 1)

    def test_token_projection_ignores_private_keys_and_superseded_memory(self):
        tokens = observed_tokens({"chief_complaint": "TEST_ONLY_VISIBLE", "evaluation": "TEST_ONLY_GOLD",
            "source_event_store": "TEST_ONLY_FUTURE", "history": [{"answer": "TEST_ONLY_UNKNOWN", "answer_status": "unknown"}],
            "longitudinal_memory": {"records": [{"text": "TEST_ONLY_OLD", "status_as_of_visit": "superseded"}]}})
        self.assertEqual(tokens, ["test_only_visible"])


class VoIReadinessContracts(unittest.TestCase):
    def test_no_implicit_prior_and_exact_update(self):
        self.assertEqual(DiagnosisBelief().snapshot()["status"], "unavailable")
        self.assertIsNone(DiagnosisBelief().snapshot()["probabilities"])
        model = test_only_voi()
        belief, mass = posterior(model["prior"], model["actions"][0]["outcomes"][0]["likelihood"])
        self.assertAlmostEqual(belief["TEST_ONLY_A"], .9)
        self.assertAlmostEqual(mass, .5)
        with self.assertRaises(ValueError):
            posterior(model["prior"], {d: 0 for d in model["prior"]})

    def test_v2_schema_and_supported_source_bindings(self):
        for kind, target, source in (("request_test", "TEST_ONLY_T", "objective_findings"),
            ("ask_patient", "TEST_ONLY_QUESTION?", "history"), ("consult_specialty", "neurology", "consults"),
            ("retrieve_memory", "longitudinal", "memory")):
            model = test_only_voi()
            action = model["actions"][0]
            action.update(type=kind, target=target)
            for outcome in action["outcomes"]:
                outcome["observation"]["source"] = source
                if source != "objective_findings":
                    outcome["supports_diagnoses"] = []
            parsed = InformationOutcomes(model)
            self.assertEqual(parsed.source_schema_version, 2)
        for kind in ("treat", "diagnose", "TEST_ONLY_UNSUPPORTED"):
            model = test_only_voi()
            model["actions"][0]["type"] = kind
            with self.assertRaises(ValueError):
                InformationOutcomes(model)

    def test_bad_likelihood_cost_fields_and_horizon_rejected(self):
        for mutation in ("likelihood", "cost", "cost_weights"):
            model = test_only_voi()
            if mutation == "likelihood":
                model["actions"][0]["outcomes"][0]["likelihood"]["TEST_ONLY_A"] = .5
            elif mutation == "cost":
                model["actions"][0]["cost"] = {"money": 0, "minutes": 0, "tokens": 0}
            else:
                model["cost_weights"] = {"money": 0, "minutes": 0, "tokens": 0}
            with self.assertRaises(ValueError):
                InformationOutcomes(model)
        for horizon in (0, 4, True, 1.5):
            model = test_only_voi()
            model["horizon"] = horizon
            with self.assertRaises(ValueError):
                InformationOutcomes(model)
            with self.assertRaises(ValueError):
                plan_voi(model["prior"], model["actions"], model["terminal_utility"], horizon=horizon)

    def test_observation_update_ignores_hidden_fields_and_unknown_text_fails(self):
        engine = InformationOutcomes(test_only_voi())
        belief, matched = engine.observe({"hidden_test_results": {"TEST_ONLY_T": "TEST_ONLY_A"},
            "gold": "TEST_ONLY_A", "source_event_store": ["TEST_ONLY_T: TEST_ONLY_A"]})
        self.assertEqual(belief, engine.artifact["prior"])
        self.assertEqual(matched, [])
        observation = {"objective_findings": ["TEST_ONLY_T: TEST_ONLY_A"]}
        self.assertEqual(engine.observe(observation), engine.observe(observation))
        self.assertAlmostEqual(engine.observe(observation)[0]["TEST_ONLY_A"], .9)
        with self.assertRaises(ValueError):
            engine.observe({"objective_findings": ["TEST_ONLY_T: outside_model"]})

    def test_cost_free_bellman_with_three_independent_toy_channels(self):
        model = test_only_voi()
        actions = [dict(deepcopy(model["actions"][0]), id="TEST_ONLY_{}".format(i)) for i in range(3)]
        one = plan_voi(model["prior"], actions, model["terminal_utility"], horizon=1)
        three = plan_voi(model["prior"], actions, model["terminal_utility"], horizon=3)
        self.assertAlmostEqual(one["optimal_value"], .9)
        self.assertAlmostEqual(three["optimal_value"], .972)
        self.assertEqual(three["cost_model"], "none")
        altered = deepcopy(actions)
        for action in altered:
            action["cost"] = {"money": 10**9, "minutes": 10**9, "tokens": 10**9}
        self.assertEqual(three, plan_voi(model["prior"], altered, model["terminal_utility"], horizon=3))

    def test_selector_missing_model_and_unmodeled_candidates_are_unavailable(self):
        missing = select_action([], selector="bayesian_voi")
        self.assertEqual((missing["status"], missing["reason"]), ("unavailable", "missing_outcome_model"))
        model = InformationOutcomes(test_only_voi()).artifact
        candidate = {"action_id": "TEST_ONLY_UNKNOWN", "type": "Test", "outcome_model": None}
        result = select_action([candidate], selector="bayesian_voi", belief=model["prior"], artifact=model)
        self.assertEqual(result["reason"], "missing_action_outcome_model")

    def test_information_policy_rejects_partial_source_menu_and_memory_action(self):
        runtime = SimpleNamespace(case_data={"index_encounter_id": "TEST_ONLY", "current_encounter_events": []},
            question_history=[], voi_state={}, current_time_min=0, voi_memory_pool={})
        policy = InformationPolicy(test_only_voi())
        result = policy.propose(runtime, {}, actions=["request_test"], tests=["TEST_ONLY_UNMODELED"],
                                departments=[], evidence_records=[], allow_unresolved=True)
        self.assertIsNone(result)
        self.assertEqual(runtime.voi_state["selector_status"]["reason"], "missing_action_outcome_model")
        self.assertEqual(candidate_actions(runtime, ["retrieve_memory"], [], []), [])


class LogicalSpecialtyContracts(unittest.TestCase):
    def test_logical_binding_switch_is_distinct_from_source_consult(self):
        case = load_cases(Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]
        engine = adapt_case(case).runtime_case
        self.assertFalse(any(e["action"] == "Consult" for e in engine["current_encounter_events"]))
        runtime = HospitalCaseRuntime(engine["case_id"], engine, 0, 0, current_symptoms=initial_symptoms(engine))
        runtime.current_time_min = 3
        policy = HospitalLLMPolicy(flow_config()["clinical"], client=SourceActionClient())
        policy.agent_registry = AgentRegistry(test_only_registry())
        policy.routing_mode, policy.primary_router_mode = "adaptive", "adaptive"
        calls = []

        def offline_call(client, **kwargs):
            # Router result is TEST_ONLY control flow; no fitted model or LLM.
            calls.append((kwargs["role"], deepcopy(kwargs["payload"])))
            kwargs["audit"].append({"role": kwargs["role"], "status": "success", "fallback_used": False})
            return kwargs["validate"](client.chat_json(kwargs["system"], json.dumps(kwargs["payload"]), schema=kwargs["schema"]))

        trace = {"decisions": [], "retrievals": []}
        with patch.object(policy.agent_registry, "select_specialty", return_value={"selected_specialty": "neurology", "decision_status": "selected"}), \
             patch("app.core.hospital_llm_policy.role_call", side_effect=offline_call), _capture_execution(trace):
            result = policy._decision(runtime, "er")
        self.assertEqual(calls[0][0], "specialty")
        self.assertEqual(calls[0][1]["department"], "neurology")
        self.assertNotIn("consult_specialty", calls[0][1]["allowed_actions"])
        self.assertEqual(runtime.consult_history, [])
        self.assertEqual(result["action"], "ask_history")
        # Legacy entry role stays compatible; new request role records the switch.
        self.assertEqual(runtime.llm_audit[-1]["role"], "specialty")
        self.assertEqual(trace["decisions"][0]["role"], "er")
        self.assertEqual(trace["decisions"][0]["entry_role"], "er")
        self.assertEqual(trace["decisions"][0]["role_observation"]["effective_model_role"], "specialty")
        self.assertEqual(trace["decisions"][0]["role_observation"]["effective_department"], "neurology")
        self.assertEqual(trace["decisions"][0]["role_observation"]["binding_status"], "MATCHED")
        public = doctor_observation(runtime)
        self.assertNotIn("evaluation", public)
        self.assertNotIn("current_encounter_events", public)


if __name__ == "__main__":
    unittest.main()
