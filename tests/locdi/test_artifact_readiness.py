"""TEST_ONLY readiness/fairness checks. Reuse fixed toys; no learning or LLM."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from app.core.agent_registry import AgentRegistry
from app.core.information_outcomes import InformationOutcomes
from app.locdi.action_audit import audit_execution_roles
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.episode_exporter import build_episode
from app.locdi.run_simulation import run_case
from experiments.ablation_fairness import check_ablation_fairness
from validation.artifact_readiness import validate_routing_artifact, validate_voi_artifact, routing_decision_valid
from test_routing_voi_readiness import test_only_registry, test_only_routing, test_only_voi
from test_simulation import smoke_options

ROOT = Path(__file__).resolve().parents[2]


def voi_fixture():
    model = test_only_voi()
    model["provenance"]["training_patients"] = ["TEST_ONLY_TRAIN"]
    return model


def source_fixture():
    return [{"action": "Test", "target": "TEST_ONLY_T", "evidence": "TEST_ONLY_A"}]


def conditions():
    return {"B{}".format(i): json.loads((ROOT / "experiments/configs/B{}.json".format(i)).read_text(encoding="utf-8")) for i in range(5)}


class RoutingArtifactContracts(unittest.TestCase):
    def validate(self, artifact=None, **kwargs):
        manifest = test_only_registry()
        artifact = test_only_routing(manifest) if artifact is None else artifact
        return validate_routing_artifact(manifest, artifact, evaluation_patient_ids=["TEST_ONLY_EVAL"], **kwargs)

    def test_missing_utility_and_registry_mismatch_block(self):
        manifest = test_only_registry()
        result = validate_routing_artifact(manifest, None, evaluation_patient_ids=["TEST_ONLY_EVAL"])
        self.assertEqual(result["development_status"], "BLOCKED")
        artifact = test_only_routing(manifest)
        artifact["registry"]["agents"][0]["version"] = "TEST_ONLY_WRONG"
        self.assertEqual(self.validate(artifact)["structural_status"], "BLOCKED")

    def test_model_version_binding_mismatch_blocks(self):
        expected = {a["agent_id"]: (a["model"], a["version"]) for a in test_only_registry()["agents"]}
        for representation in (expected, {k: list(v) for k, v in expected.items()},
                               {k: {"model": v[0], "version": v[1]} for k, v in expected.items()}):
            self.assertEqual(self.validate(expected_bindings=representation)["development_status"], "DEVELOPMENT_CONTRACT_READY")
        expected["TEST_ONLY_N"] = ("TEST_ONLY_WRONG", "TEST_ONLY")
        result = self.validate(expected_bindings=expected)
        self.assertEqual(result["structural_status"], "STRUCTURAL_VALID")
        self.assertEqual(result["development_status"], "BLOCKED")

    def test_support_shortage_is_valid_abstention_not_failed_execution(self):
        artifact = test_only_routing(test_only_registry())
        artifact["min_samples"] = 3
        result = self.validate(artifact, observations=[{"chief_complaint": "test_only_signal"}])
        self.assertEqual(result["development_status"], "DEVELOPMENT_CONTRACT_READY")
        self.assertEqual(result["details"]["selection_status"], "VALID_ABSTENTION")
        self.assertEqual(result["details"]["support_capacity"], "INSUFFICIENT_COUNT_VALID_ABSTENTION")

    def test_patient_overlap_and_missing_scope_never_ready(self):
        manifest = test_only_registry()
        for ids in (None, ["TEST_ONLY_TRAIN"]):
            result = validate_routing_artifact(manifest, test_only_routing(manifest), evaluation_patient_ids=ids)
            self.assertEqual(result["development_status"], "BLOCKED")
        self.assertNotIn("TEST_ONLY_TRAIN", json.dumps(result))

    def test_invalid_selection_status_contract(self):
        registry = AgentRegistry(test_only_registry())
        result = registry.route({}, ["neurology"], mode="adaptive")
        self.assertTrue(routing_decision_valid(result, registry))
        result["selected_specialty"] = "neurology"
        self.assertFalse(routing_decision_valid(result, registry))

    def test_private_observation_never_reaches_score(self):
        with patch.object(AgentRegistry, "route", side_effect=AssertionError("Unsafe observation reached router")):
            result = self.validate(observations=[{"gold": "TEST_ONLY_PRIVATE"}])
        self.assertEqual(result["development_status"], "BLOCKED")
        self.assertNotIn("TEST_ONLY_PRIVATE", json.dumps(result))


class VoIArtifactContracts(unittest.TestCase):
    def validate(self, artifact=None, source_actions=None, **kwargs):
        return validate_voi_artifact(voi_fixture() if artifact is None else artifact,
            source_actions=source_fixture() if source_actions is None else source_actions,
            evaluation_patient_ids=["TEST_ONLY_EVAL"], expected_horizon=2, **kwargs)

    def test_valid_transport_is_not_research_validation_and_input_is_preserved(self):
        artifact = voi_fixture()
        before = deepcopy(artifact)
        result = self.validate(artifact)
        self.assertEqual(artifact, before)
        self.assertEqual(result["structural_status"], "STRUCTURAL_VALID")
        self.assertEqual(result["development_status"], "DEVELOPMENT_CONTRACT_READY")
        self.assertFalse(result["research_validated"])
        self.assertNotEqual(result["research_status"], "RESEARCH_VALIDATED")

    def test_self_declared_review_cannot_promote_research(self):
        artifact = voi_fixture()
        artifact["provenance"].update(review_status="expert_reviewed", reviewer="TEST_ONLY", review_reference="TEST_ONLY",
                                      research_validated=True, utility_scope="clinical_validated")
        result = self.validate(artifact)
        self.assertFalse(result["research_validated"])
        self.assertEqual(result["stages"]["clinical_evaluation"], "BLOCKED")

    def test_action_coverage_and_missing_artifact_block(self):
        sources = source_fixture() + [{"action": "Ask", "target": "TEST_ONLY_QUESTION?", "evidence": "TEST_ONLY"}]
        result = self.validate(source_actions=sources)
        self.assertEqual(result["development_status"], "BLOCKED")
        self.assertEqual(result["details"]["missing_source_action_count"], 1)
        self.assertEqual(validate_voi_artifact(None)["structural_status"], "BLOCKED")

    def test_prior_likelihood_and_horizon_failures(self):
        for mutation in ("prior", "likelihood", "horizon"):
            artifact = voi_fixture()
            if mutation == "prior":
                artifact["prior"]["TEST_ONLY_A"] = .1
            elif mutation == "likelihood":
                artifact["actions"][0]["outcomes"][0]["likelihood"]["TEST_ONLY_A"] = .1
            else:
                artifact["horizon"] = 4
            self.assertEqual(self.validate(artifact)["structural_status"], "BLOCKED")
        artifact = voi_fixture()
        artifact["horizon"] = 1
        self.assertEqual(self.validate(artifact)["structural_status"], "STRUCTURAL_VALID")
        self.assertEqual(self.validate(artifact)["development_status"], "BLOCKED")

    def test_replay_bin_match_and_mismatch_without_using_source_as_observation(self):
        artifact = voi_fixture()
        artifact["provenance"]["observation_encoding"] = "replay_bins_v1"
        sources = [{**source_fixture()[0], "evidence": json.dumps({"replay_bin": "a", "TEST_ONLY": True})}]
        with patch.object(InformationOutcomes, "observe", side_effect=AssertionError("Source was passed as a current observation")):
            result = self.validate(artifact, source_actions=sources)
        self.assertEqual(result["development_status"], "DEVELOPMENT_CONTRACT_READY")
        self.assertEqual(result["details"]["replay_bin_match_count"], 1)
        sources[0]["evidence"] = json.dumps({"replay_bin": "TEST_ONLY_UNKNOWN"})
        self.assertEqual(self.validate(artifact, source_actions=sources)["development_status"], "BLOCKED")

    def test_unknown_released_observation_and_private_fields_block(self):
        for obs in ({"objective_findings": ["TEST_ONLY_T: outside_model"]}, {"source_event_store": "TEST_ONLY_PRIVATE"}):
            result = self.validate(observations=[obs])
            self.assertEqual(result["development_status"], "BLOCKED")
            self.assertNotIn("TEST_ONLY_PRIVATE", json.dumps(result))

    def test_missing_training_identity_and_overlap_not_verified(self):
        artifact = voi_fixture()
        artifact["provenance"].pop("training_patients")
        self.assertEqual(self.validate(artifact)["development_status"], "BLOCKED")
        result = validate_voi_artifact(voi_fixture(), source_actions=source_fixture(), evaluation_patient_ids=["TEST_ONLY_TRAIN"], expected_horizon=2)
        self.assertEqual(result["development_status"], "BLOCKED")


class FairnessContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(ROOT / "datasets/raw/longitudinal_benchmark_50.jsonl")

    def test_official_full_differences_and_nulls_not_ready(self):
        result = check_ablation_fairness(conditions(), cases=self.cases)
        expected = [{"condition", "features.memory_mode"}, {"condition", "features.memory_mode", "features.top_k"},
            {"condition", "features.routing_mode", "features.primary_router"}, {"condition", "features.voi_enabled", "features.action_selector"}]
        self.assertEqual([{d["path"] for d in p["differences"]} for p in result["comparisons"]], expected)
        self.assertTrue(all(not p["isolated_single_feature_causal_effect"] for p in result["comparisons"]))
        shared_model = next(c for c in result["checks"] if c["code"] == "COMMON_MODEL")
        self.assertTrue(shared_model["equal"])
        self.assertFalse(shared_model["configured"])
        self.assertEqual(shared_model["status"], "BLOCKED")
        self.assertEqual(result["paired_case_identity_status"], "NOT_REVIEWED")

    def test_paired_identity_and_source_event_changes(self):
        sets = {k: [self.cases[0]] for k in conditions()}
        result = check_ablation_fairness(conditions(), cases=[self.cases[0]], case_sets=sets)
        self.assertEqual(result["paired_case_identity_status"], "READY")
        sets = {k: deepcopy(rows) for k, rows in sets.items()}
        sets["B4"][0]["environment"]["current_encounter_events"][0]["target"] = "TEST_ONLY_CHANGED"
        result = check_ablation_fairness(conditions(), cases=[self.cases[0]], case_sets=sets)
        self.assertEqual(next(c["status"] for c in result["checks"] if c["code"] == "PAIRED_SOURCE_ACTIONS_AND_PAYLOADS"), "BLOCKED")
        sets["B4"] = [self.cases[1]]
        self.assertEqual(check_ablation_fairness(conditions(), case_sets=sets)["paired_case_identity_status"], "BLOCKED")

    def test_common_difference_seed_and_false_ready_flags(self):
        configs = conditions()
        configs["B4"]["common"]["model"] = "TEST_ONLY_PRIVATE_MODEL"
        configs["B4"]["common"]["decoding"] = {"seed": 77, "temperature": 0, "max_output_tokens": 100}
        for config in configs.values():
            config["ready"] = True
        result = check_ablation_fairness(configs)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["research_validated"])
        self.assertNotIn("TEST_ONLY_PRIVATE_MODEL", json.dumps(result))
        self.assertEqual(next(c["status"] for c in result["checks"] if c["code"] == "COMMON_DECODING"), "BLOCKED")

    def test_fallback_audit_and_malformed_inputs(self):
        episodes = {"B4": [{"workflow": {"decisions": [{"actual_selector": "specialist_heuristic", "fallback_used": True,
                                                         "selector_status": "unavailable"}]}}]}
        result = check_ablation_fairness(conditions(), episodes=episodes)
        self.assertEqual(result["conditions"]["B4"]["execution_audit"]["fallback_count"], 1)
        self.assertIn("B4_EXECUTION_COVERAGE_OR_FALLBACK_UNVERIFIED", result["warnings"])
        self.assertEqual(check_ablation_fairness({})["status"], "BLOCKED")
        configs = conditions()
        configs["B4"]["common"] = None
        self.assertEqual(check_ablation_fairness(configs)["status"], "BLOCKED")


class RoleObservationContracts(unittest.TestCase):
    def test_missing_roles_and_er_only_public_projection(self):
        missing = audit_execution_roles({})
        self.assertEqual(missing["effective_model_role"], "NOT_RECORDED")
        self.assertEqual(missing["routing_abstained"], "NOT_RECORDED")
        case = load_cases(ROOT / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]
        raw = run_case(case, options=smoke_options())
        document = build_episode(raw, context=adapt_case(case))
        for record in document["decision_audit"]["records"]:
            self.assertEqual((record["entry_role"], record["effective_model_role"]), ("er", "er"))
            self.assertEqual(record["model_binding_status"], "MATCHED")
            self.assertRegex(record["model_binding"]["binding_ref"], r"^Binding[0-9]+$")
        self.assertTrue(all(d["agent_role"] == "er" for d in document["workflow"]["decisions"]))
        self.assertNotIn(case["patient_id"], json.dumps(document))

    def test_actual_specialty_role_binding_projection_and_abstention_contract(self):
        refs = {}
        def reference(kind, value):
            key = (kind, value)
            refs.setdefault(key, "{}{:04d}".format(kind.title(), len(refs) + 1))
            return refs[key]
        binding = test_only_registry()["agents"][1]
        binding["agent_id"] = "TEST_ONLY_PRIVATE_AGENT"
        record = audit_execution_roles({"role": "specialty", "fallback_used": False,
            "routing": {"selected_specialty": "neurology", "decision_status": "selected"}}, entry_role="er",
            observed={"effective_model_role": "specialty", "effective_department": "neurology",
                      "model_binding": binding, "binding_status": "MATCHED"}, reference=reference)
        self.assertEqual(record["entry_role"], "er")
        self.assertEqual(record["effective_model_role"], "specialty")
        self.assertFalse(record["routing_abstained"])
        self.assertNotIn("TEST_ONLY_PRIVATE_AGENT", json.dumps(record))
        abstain = audit_execution_roles({"routing": {"selected_specialty": None, "decision_status": "abstained"}})
        self.assertTrue(abstain["routing_abstained"])


if __name__ == "__main__":
    unittest.main()
