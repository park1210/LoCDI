"""Structural audit contracts, using real source cases and offline clients only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.locdi.action_audit import audit_action, capture_contract, summarize_actions, read_decision_audit
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.episode_exporter import build_episode, export_summary
from app.locdi.run_simulation import _capture_execution, run_case
from app.core.hospital_llm_policy import HospitalLLMPolicy
import app.core.hospital_llm_policy as policy_module
from test_simulation import SourceActionClient, smoke_options

ROOT = Path(__file__).resolve().parents[2]


class ActionAuditContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases(ROOT / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]
        cls.context = adapt_case(cls.case)
        cls.client = SourceActionClient()
        cls.raw = run_case(cls.case, options=smoke_options(cls.client))
        cls.document = build_episode(cls.raw, context=cls.context)

    def test_actual_menu_and_candidates_preserved(self):
        records = self.document["decision_audit"]["records"]
        self.assertEqual(len(records), len(self.client.calls))
        for record, call in zip(records, self.client.calls):
            payload = call["payload"]
            self.assertEqual(record["allowed_action_types"], payload["allowed_actions"])
            self.assertEqual(record["candidate_action_count"], len(payload["candidate_actions"]))
            self.assertTrue(record["selected_in_candidates"])
            self.assertTrue(record["action_mapping_matches"])
            self.assertEqual(record["execution_status"], "OBSERVED")
        self.assertEqual(records[0]["candidate_action_count"], 5)
        self.assertFalse(records[0]["diagnose_allowed"])
        self.assertTrue(records[-1]["diagnose_allowed"])

    def test_finalization_gate_and_citations_keep_core_meaning(self):
        records = self.document["decision_audit"]["records"]
        state = next(iter(self.raw["execution"]["visits"][0]["final_state"]["cases"].values()))
        for record, core in zip(records, state["llm_audit"]):
            self.assertEqual(record["clinical_support_verified"], core["decision_validation"]["clinical_support_verified"])
            self.assertEqual(record["citation_count"], len(core["memory_citations"]["ids"]))
            self.assertEqual(record["citation_validation_status"], "VALID" if core["memory_citations"]["valid"] else "INVALID")
        final = records[-1]
        self.assertEqual(final["diagnosis_finalization_status"], "EXECUTED")
        self.assertEqual(final["certainty"], "unresolved")
        self.assertEqual(final["evidence_gate_status"], "STRUCTURAL_ACCEPTED")
        self.assertFalse(final["clinical_support_verified"])
        self.assertEqual(final["candidate_match_scope"], "TYPE_ONLY")
        self.assertEqual(final["execution_target_validation"], "NOT_CHECKED")
        supported = audit_action({"decision": {"action": "diagnose", "target": "private"},
            "decision_validation": {"cited_evidence": [{"supports_finalization": True}], "clinical_support_verified": False}}, {})
        self.assertEqual(supported["evidence_gate_status"], "STRUCTURAL_ACCEPTED")
        self.assertIsNone(supported["certainty"])
        self.assertFalse(supported["clinical_support_verified"])

    def test_session_refs_link_exact_candidate_without_raw_hash(self):
        records = self.document["decision_audit"]["records"]
        for record in records[:-1]:
            self.assertIn(record["selected_action_ref"], [c["action_ref"] for c in record["candidates"]])
        for record in records:
            self.assertRegex(record["selected_action_ref"], r"^Action[0-9]{4,}$")
        text = json.dumps(self.document)
        for event in self.context.runtime_case["current_encounter_events"]:
            self.assertNotIn(hashlib.sha256(event["target"].encode()).hexdigest(), text)

    def test_missing_and_false_are_not_fake_pass(self):
        missing = audit_action({}, {})
        self.assertEqual(missing["availability"], "NOT_RECORDED")
        for key in ("candidate_action_count", "selected_in_candidates", "execution_success", "diagnose_allowed", "citation_count"):
            self.assertIsNone(missing[key])
        self.assertEqual(missing["citation_validation_status"], "NOT_RECORDED")
        mismatch = audit_action({"decision": {"action": "request_test", "target": "private"}, "candidate_actions": []},
                                {"executed_action": "ask_history", "success": False})
        self.assertFalse(mismatch["selected_in_candidates"])
        self.assertFalse(mismatch["action_mapping_matches"])
        self.assertEqual(mismatch["execution_status"], "FAILED")
        self.assertEqual(summarize_actions([missing, mismatch])["execution_status_counts"], {"NOT_RECORDED": 1, "FAILED": 1})

    def test_source_orderability_does_not_claim_result_visible(self):
        test = next(e for e in self.context.runtime_case["current_encounter_events"] if e["action"] == "Test")
        payload = {"allowed_actions": ["request_test"], "candidate_actions": [{"action": "request_test", "target": test["target"]}]}
        snapshot = capture_contract(payload, source_events=[test], minute=0)
        self.assertEqual(snapshot["candidates"][0]["source_status"], "ORDERABLE_SOURCE_RESULT_GATED")
        ask = next(e for e in self.context.runtime_case["current_encounter_events"] if e["action"] == "Ask")
        payload["candidate_actions"] = [{"action": "ask_patient", "target": ask["target"]}]
        snapshot = capture_contract(payload, source_events=[dict(ask, available_minute=10)], minute=0)
        self.assertEqual(snapshot["candidates"][0]["source_status"], "SOURCE_UNAVAILABLE")

    def test_rejection_records_category_without_invalid_response(self):
        class Rejected(SourceActionClient):
            def chat_json(self, *args, **kwargs):
                return json.dumps({"action": "diagnose", "target": "PRIVATE-INVALID-DIAGNOSIS"})
        raw = run_case(self.case, options=smoke_options(Rejected()))
        self.assertEqual(raw["status"], "failed")
        doc = build_episode(raw, context=self.context)
        record = doc["decision_audit"]["records"][0]
        self.assertEqual(record["decision_validation_status"], "REJECTED")
        self.assertEqual(record["rejection_reason"], "schema_validation")
        self.assertEqual(record["execution_status"], "NOT_EXECUTED")
        self.assertEqual(record["rejected_action_availability"], "NOT_RECORDED")
        self.assertIsNone(record["rejected_action_type"])
        self.assertNotIn("PRIVATE-INVALID-DIAGNOSIS", json.dumps(doc))

    def test_invalid_citations_and_private_fields_are_sanitized(self):
        raw = deepcopy(self.raw)
        state = next(iter(raw["execution"]["visits"][0]["final_state"]["cases"].values()))
        state["llm_audit"][0]["memory_citations"] = {"ids": ["PRIVATE-CITATION"], "invalid_ids": ["PRIVATE-BAD-ID"], "valid": False, "reported": True}
        for row in state["llm_audit"]:
            row["invalid_response"] = row["rationale"] = row["gold"] = "PRIVATE-RAW-TEXT"
            row["validation_detail"] = {"code": "PRIVATE-CODE", "value_repr": "PRIVATE-VALUE"}
        doc = build_episode(raw, context=self.context)
        record = doc["decision_audit"]["records"][0]
        self.assertEqual(record["citation_count"], 1)
        self.assertEqual(record["invalid_citation_count"], 1)
        self.assertEqual(record["citation_validation_status"], "INVALID")
        self.assertNotIn("PRIVATE-", json.dumps(doc))

    def test_capture_restores_functions_and_retriever_after_exception(self):
        decision, call = HospitalLLMPolicy._decision, policy_module.role_call
        with self.assertRaises(RuntimeError):
            with _capture_execution({"decisions": [], "retrievals": []}):
                raise RuntimeError("test")
        self.assertIs(HospitalLLMPolicy._decision, decision)
        self.assertIs(policy_module.role_call, call)
        retriever = lambda runtime, observed: {"records": []}
        policy = SimpleNamespace(memory_retriever=retriever)
        runtime = SimpleNamespace(llm_audit=[], case_id="private", current_time_min=0)
        with patch.object(HospitalLLMPolicy, "_decision", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError), _capture_execution({"decisions": [], "retrievals": []}):
                HospitalLLMPolicy._decision(policy, runtime, "er")
        self.assertIs(policy.memory_retriever, retriever)
        run_case(self.case, options=smoke_options(SourceActionClient("timeout")))
        self.assertIs(HospitalLLMPolicy._decision, decision)
        self.assertIs(policy_module.role_call, call)

    def test_old_real_episodes_readable_without_backfilling(self):
        paths = list((ROOT / "outputs/episodes").glob("955753832e0449eaa9d02f0e9018c59f_case*_real.json"))
        self.assertEqual(len(paths), 5)
        originals = {p: p.read_bytes() for p in paths}
        docs = [json.loads(originals[p]) for p in sorted(paths)]
        self.assertTrue(all("decision_audit" not in d and not d["mock_execution"] for d in docs))
        self.assertTrue(all(read_decision_audit(d)["availability"] == "NOT_RECORDED" for d in docs))
        with tempfile.TemporaryDirectory() as root:
            path = export_summary({"episodes": docs}, "metrics/old.jsonl", output_root=root)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["episode_count"], 5)
        for p, data in originals.items():
            self.assertEqual(p.read_bytes(), data)
        self.assertTrue(self.document["mock_execution"])
        self.assertEqual(self.document["schema_version"], "locdi.episode.v1")
        audit = deepcopy(self.document["decision_audit"])
        recorded = audit.pop("fingerprint")
        digest = hashlib.sha256(json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(recorded, digest)


if __name__ == "__main__":
    unittest.main()
