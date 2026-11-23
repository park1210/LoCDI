"""Real five-case smoke, public projections and failure-safe storage."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.episode_exporter import ExportSession, build_episode, export_episode, export_summary
from app.locdi.run_simulation import run_case
from test_simulation import smoke_options
from smoke_support import execute_smoke, _keys


class EpisodeExportContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke = execute_smoke()
        cls.raw = cls.smoke["raw_first"][0]
        cls.context = cls.smoke["contexts"][0]

    def test_five_real_cases_and_repeat_trace(self):
        self.assertEqual(len(self.smoke["rounds"][0]), 5)
        self.assertEqual(self.smoke["deterministic"], [True] * 5)
        self.assertTrue(all(c["memory"] and c["release"] for c in self.smoke["checks"]))
        for group in self.smoke["rounds"]:
            self.assertEqual([d["case_ref"] for d in group], ["Case{:03d}".format(i) for i in range(1, 6)])
            self.assertEqual(len({d["patient_ref"] for d in group}), 5)
            for document in group:
                self.assertEqual(document["result"]["status"], "completed")
                self.assertTrue(document["result"]["diagnosis_unresolved"])
                self.assertFalse(document["simulation_ready"])
                self.assertEqual(document["clinical_metrics"], "NOT_EVALUATED")
                self.assertTrue(all(r["cutoff_compliant"] for r in document["memory"]["retrievals"]))

    def test_roundtrip_json_jsonl_and_status_summary(self):
        with tempfile.TemporaryDirectory() as root:
            document = build_episode(self.raw, context=self.context)
            session = ExportSession(run_id=document["run_id"])
            path = export_episode(self.raw, "episodes/episode.json", context=self.context, session=session, output_root=root)
            reloaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded, document)
            records = [document] + self.smoke["controls"]
            summary_path = export_summary({"episodes": records}, "metrics/summary.jsonl", output_root=root)
            lines = summary_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            summary = json.loads(lines[0])
            self.assertEqual(summary["status_counts"], {"completed": 1, "incomplete": 1, "failed": 1})
            self.assertEqual(summary["clinical_metrics"], "NOT_EVALUATED")
            self.assertEqual([d["result"]["status"] for d in self.smoke["controls"]], ["incomplete", "failed"])
            self.assertEqual(self.smoke["controls"][0]["result"]["elapsed_simulation_minutes"], 1)
            self.assertIn("RoleCallError", self.smoke["controls"][1]["result"]["error_types"])

    def test_public_projection_excludes_private_identifiers_and_free_text(self):
        raw = deepcopy(self.raw)
        marker = "PRIVATE-PATIENT-GOLD-EVIDENCE-DO-NOT-EXPORT"
        raw["evaluation"] = raw["gold"] = raw["provenance"] = {"private": marker}
        state = next(iter(raw["execution"]["visits"][0]["final_state"]["cases"].values()))
        state["llm_audit"][0]["decision"]["rationale"] = marker
        raw["execution"]["visits"][0]["final_state"]["event_log"][0]["payload"]["source_event_store"] = marker
        raw["errors"].append({"error_type": marker, "message": marker})
        document = build_episode(raw, context=self.context)
        text = json.dumps(document, ensure_ascii=False)
        self.assertNotIn(marker, text)
        forbidden = {"patient_id", "case_id", "visit_id", "evaluation", "gold", "provenance", "source_event_store", "evidence", "rationale"}
        self.assertFalse(_keys(document) & forbidden)
        case = self.smoke["cases"][0]
        for value in (case["case_id"], case["patient_id"], case["agent_visible"]["index_encounter_id"]):
            self.assertNotIn(value, text)
        for past in case["agent_visible"]["past_visits"]:
            self.assertNotIn(past["visit_id"], text)
            for record in past["records"]:
                self.assertNotIn(record["memory_id"], text)
        for event in self.context.runtime_case["current_encounter_events"]:
            self.assertNotIn(event["evidence"], text)
            self.assertNotIn(event["target"], text)

    def test_selected_actions_link_to_executed_events_and_times(self):
        document = self.smoke["rounds"][0][0]
        workflow = document["workflow"]
        events = {e["sequence"]: e for e in workflow["events"]}
        self.assertEqual([d["selected_action"] for d in workflow["decisions"]],
                         ["ask_patient"] + ["request_test"] * 4 + ["diagnose"])
        self.assertEqual([e["minute"] for e in events.values()], sorted(e["minute"] for e in events.values()))
        for decision in workflow["decisions"]:
            self.assertTrue(decision["execution_success"])
            self.assertTrue(decision["execution_event_sequences"])
            for seq in decision["execution_event_sequences"]:
                self.assertEqual(events[seq]["minute"], decision["minute"])
                if decision["selected_action"] != "diagnose":
                    self.assertEqual(events[seq]["target_ref"], decision["target_ref"])
        self.assertEqual([e["minute"] for e in events.values() if e["event_type"] == "result_received"], [11, 21, 31, 41])
        self.assertEqual([r["minute"] for r in document["memory"]["retrievals"]], [3, 5, 15, 25, 35, 45])

    def test_same_case_linkage_and_wrong_context_rejected(self):
        session = ExportSession()
        one = build_episode(self.raw, context=self.context, session=session)
        two = build_episode(self.raw, context=self.context, session=session)
        self.assertEqual(one, two)
        with self.assertRaises(ValueError):
            build_episode(self.raw, context=self.smoke["contexts"][1])
        with self.assertRaises(ValueError):
            build_episode(self.raw, context=self.context, session=session, case_ordinal=2)

    def test_duplicate_and_unsafe_paths_never_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            path = export_episode(self.raw, "episodes/existing.json", context=self.context, output_root=root)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                export_episode(self.raw, "episodes/existing.json", context=self.context, output_root=root)
            self.assertEqual(path.read_bytes(), original)
            for destination in ("../escape.json", "episodes/../escape.json", str(Path(root).parent / "outside.json"),
                                "reports/wrong.json", "episodes/alternate:stream.json", "episodes/raw.txt"):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    export_episode(self.raw, destination, context=self.context, output_root=root)
            self.assertEqual(list(Path(root).rglob("*.json")), [path])

    def test_storage_and_serialization_failures_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as root:
            path = export_episode(self.raw, "episodes/existing.json", context=self.context, output_root=root)
            original = path.read_bytes()
            for symbol in ("os.fsync", "os.link"):
                with patch("app.locdi.episode_exporter." + symbol, side_effect=OSError("injected storage failure")):
                    with self.assertRaises(OSError):
                        export_episode(self.raw, "episodes/failure.json", context=self.context, output_root=root)
                self.assertFalse((Path(root) / "episodes/failure.json").exists())
                self.assertEqual(list(Path(root).rglob("*.tmp")), [])
            for value in (object(), float("nan")):
                bad = dict(self.raw, unknown=value)
                with self.assertRaises(ValueError):
                    export_episode(bad, "episodes/invalid.json", context=self.context, output_root=root)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((Path(root) / "episodes/invalid.json").exists())
            link = os.link

            def racing_writer(source, destination):
                Path(destination).write_bytes(b"existing-racing-writer")
                return link(source, destination)

            with patch("app.locdi.episode_exporter.os.link", side_effect=racing_writer), self.assertRaises(FileExistsError):
                export_episode(self.raw, "episodes/race.json", context=self.context, output_root=root)
            self.assertEqual((Path(root) / "episodes/race.json").read_bytes(), b"existing-racing-writer")
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(root).rglob("*.tmp")), [])

    def test_unexpected_core_failure_exports_unknown_state_without_invention(self):
        marker = self.smoke["cases"][0]["patient_id"]
        with patch("app.locdi.run_simulation.run_with_recovery", side_effect=RuntimeError(marker)):
            result = run_case(self.smoke["cases"][0], options=smoke_options())
        document = build_episode(result, context=self.context)
        self.assertEqual(document["result"]["status"], "failed")
        self.assertEqual(document["result"]["trace_availability"], "core_report_unavailable")
        self.assertIsNone(document["result"]["elapsed_simulation_minutes"])
        self.assertEqual(document["workflow"]["events"], [])
        self.assertNotIn(marker, json.dumps(document))
        with tempfile.TemporaryDirectory() as root:
            path = export_episode(result, "episodes/failed.json", context=self.context, output_root=root)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["result"]["status"], "failed")

    def test_invalid_summary_cannot_export_private_fields_as_counts(self):
        document = deepcopy(self.smoke["rounds"][0][0])
        document["result"]["action_counts"] = {"private-patient-identifier": 3}
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            export_summary({"episodes": [document]}, "metrics/invalid.json", output_root=root)

    def test_configuration_fingerprint_detects_mutation(self):
        bad = deepcopy(self.raw)
        bad["execution_metadata"]["configuration"]["minute_limit"] = 99
        with self.assertRaises(ValueError):
            build_episode(bad, context=self.context)
