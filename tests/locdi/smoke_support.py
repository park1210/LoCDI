"""Real-data smoke harness; all model/registry fixtures stay inside tests."""
import json
from pathlib import Path
import sys
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unittest.mock import patch
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.run_simulation import run_case
from app.locdi.episode_exporter import ExportSession, build_episode, export_episode, export_summary, _write
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.longitudinal_memory import timestamp
from test_simulation import SourceActionClient, smoke_options

DATASET = Path(__file__).resolve().parents[2] / "datasets/raw/longitudinal_benchmark_50.jsonl"


def _strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(v) for v in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(v) for v in value), set())
    return set()


def check_boundary(case, context, result, client):
    if result["execution"] is None:
        return {"memory": False, "release": False}
    visit = result["execution"]["visits"][0]
    final = visit["final_state"]
    delivered = {e["payload"]["requested_test"]: e["minute"] for e in final["event_log"] if e["event_type"] == "result_received"}
    ordered = {e["payload"]["requested_test"]: e["minute"] for e in final["event_log"] if e["event_type"] == "test_ordered"}
    events = [e for e in context.runtime_case["current_encounter_events"] if e["action"] == "Test"]
    seen = set()
    forbidden = {"evaluation", "task_annotations", "gold", "provenance", "source_event_store", "data_replay", "current_encounter_events"}
    for call in client.calls:
        payload = call["payload"]
        assert not _keys(payload) & forbidden, "Private key in actual role payload"
        minute = payload["observation"]["elapsed_minutes"]
        visible = _strings(payload)
        for event in events:
            present = any(event["evidence"] in text for text in visible)
            if present:
                assert event["target"] in delivered, "Unordered/undelivered test was visible"
                assert minute >= delivered[event["target"]] >= ordered[event["target"]], "Early test result"
                assert minute >= event["available_minute"], "Source release cutoff violated"
                seen.add(event["target"])
        for record in payload["observation"]["longitudinal_memory"]["records"]:
            assert record["patient_id"] == case["patient_id"], "Cross-patient memory"
            assert record["visit_id"] != case["agent_visible"]["index_encounter_id"], "Current-visit memory"
            assert timestamp(record["available_at"]) <= timestamp(case["agent_visible"]["index_encounter_start"]), "Future memory"
    if result["status"] == "completed":
        assert seen == set(delivered) == {e["target"] for e in events}, "Test sources not fully exercised"
    assert all(r["cutoff_compliant"] for r in result["telemetry"]["retrievals"]), "Retrieval cutoff failed"
    return {"memory": True, "release": True}


def execute_smoke(output_root=None):
    """First five, separately, twice; retain every genuine status and comparison.

    With a destination, writes only redacted exports. No raw runtime/client data
    is persisted. A mismatch is reported, not rewritten into a success result.
    """
    cases = load_cases(DATASET)[:5]
    contexts = [adapt_case(c) for c in cases]
    rounds, checks, raw_first = [], [], []
    for repeat in range(2):
        session = ExportSession()
        documents = []
        for ordinal, (case, context) in enumerate(zip(cases, contexts), 1):
            client = SourceActionClient()
            with patch("app.locdi.run_simulation.run_with_recovery", wraps=run_with_recovery) as runner, \
                 patch("app.core.longitudinal_runner.HospitalParallelScheduler", wraps=HospitalParallelScheduler) as scheduler, \
                 patch("app.core.hospital_llm_policy.OllamaClient", side_effect=AssertionError("No real model")):
                result = run_case(case, options=smoke_options(client))
            assert runner.call_count == scheduler.call_count == 1, "Real Core path not reached"
            checks.append(check_boundary(case, context, result, client))
            document = build_episode(result, context=context, session=session, case_ordinal=ordinal)
            documents.append(document)
            if repeat == 0:
                raw_first.append(result)
            if output_root is not None:
                path = export_episode(result, "episodes/{}_case{:03d}.json".format(session.run_id, ordinal),
                    context=context, session=session, case_ordinal=ordinal, output_root=output_root)
                assert json.loads(path.read_text(encoding="utf-8")) == document
        rounds.append(documents)
    deterministic = [a["normalized_trace_fingerprint"] == b["normalized_trace_fingerprint"] for a, b in zip(*rounds)]
    # Negative controls use the same real case; they do not replace failed cases.
    controls = []
    for label, client, horizon in (("incomplete", SourceActionClient(), 1), ("failed", SourceActionClient("timeout"), 120)):
        result = run_case(cases[0], options=smoke_options(client, minute_limit=horizon))
        session = ExportSession()
        document = build_episode(result, context=contexts[0], session=session)
        controls.append(document)
        if output_root is not None:
            export_episode(result, "episodes/{}_{}.json".format(session.run_id, label),
                           context=contexts[0], session=session, output_root=output_root)
    summary_path = None
    if output_root is not None:
        summary_path = export_summary({"episodes": rounds[0]},
            "metrics/{}_summary.jsonl".format(rounds[0][0]["run_id"]), output_root=output_root)
        export_summary({"episodes": controls}, "metrics/{}_controls.jsonl".format(rounds[0][0]["run_id"]), output_root=output_root)
    return {"rounds": rounds, "checks": checks, "deterministic": deterministic,
            "controls": controls, "raw_first": raw_first, "contexts": contexts, "cases": cases,
            "summary_path": summary_path}


if __name__ == "__main__":
    import argparse
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    arguments = parser.parse_args()

    def guard(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.bind", "subprocess.Popen", "os.system"}:
            raise RuntimeError("External side effects are disabled")

    sys.addaudithook(guard)
    smoke = execute_smoke(arguments.output_root)
    first = smoke["rounds"][0]
    report = {"schema_version": "locdi.smoke_report.v1", "purpose": "development_smoke",
        "mock_execution": True, "clinical_metrics": "NOT_EVALUATED", "simulation_ready": False,
        "selection": "first_five_in_input_order", "source_schema_version": smoke["cases"][0]["schema_version"],
        "cases": [{"case_ref": d["case_ref"], "result": d["result"],
                   "configuration_fingerprint": d["configuration_fingerprint"],
                   "first_trace_fingerprint": d["normalized_trace_fingerprint"],
                   "repeat_trace_fingerprint": repeat["normalized_trace_fingerprint"],
                   "deterministic": match,
                   "validation": {r.validator: r.status for r in context.validation_results}}
                  for d, repeat, match, context in zip(first, smoke["rounds"][1], smoke["deterministic"], smoke["contexts"])],
        "boundary_checks": smoke["checks"], "negative_controls": [d["result"] for d in smoke["controls"]],
        "run_ids": [group[0]["run_id"] for group in smoke["rounds"]],
        "status_counts": {s: sum(d["result"]["status"] == s for d in first) for s in ("completed", "incomplete", "failed")}}
    path = _write(report, "reports/{}_smoke.json".format(first[0]["run_id"]), arguments.output_root, "reports")
    print(json.dumps({"report_path": str(path), "status_counts": report["status_counts"],
                      "deterministic": smoke["deterministic"],
                      "results": [{"case_ref": d["case_ref"], **d["result"]} for d in first]}, ensure_ascii=True))
    sys.exit(0 if all(smoke["deterministic"]) and all(d["result"]["status"] == "completed" for d in first) else 1)
