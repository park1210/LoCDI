"""First five real-development cases, independently. Default is static dry-run."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import load_setup, validate_setup
from app.locdi.ollama_adapter import LocalOllamaAdapter
from app.locdi.run_simulation import SimulationOptions, run_case, inspect_development_llm
from app.locdi.episode_exporter import ExportSession, build_episode, export_episode, export_summary, _write, ERRORS
from scripts.run_development_llm_case import FEATURES

CASE_LIMIT = 5


def _error_type(exc):
    safe = ERRORS | {"OSError", "FileExistsError", "FileNotFoundError", "SimulationConfigurationError", "CaseAdaptationError"}
    return type(exc).__name__ if type(exc).__name__ in safe else "ExecutionError"


def _execute_case(case, config, expected_fingerprint):
    """Own the real client so counters survive an unexpected runner exception.

    No fallback client exists here. Client construction cannot generate and
    failure before construction therefore has known zero request counters.
    """
    client, result, error = None, None, None
    metadata = {"request_count": 0, "identity_check_count": 0, "real_model_execution": False, "error_types": []}
    try:
        ready = validate_setup(config)
        if ready["static_readiness"] != "READY" or ready["prepared"]["fingerprint"] != expected_fingerprint:
            raise ValueError("Batch setup changed or is not ready")
        client = LocalOllamaAdapter(config, authorized=True)
        options = SimulationOptions(purpose="development_llm", allow_llm=True, dry_run=False,
            client=client, llm_config=config, minute_limit=config["minute_limit"],
            registry_manifest=client.prepared["registry"], runner_options=deepcopy(FEATURES),
            accept_recorded_release_assumption=True)
        result = run_case(case, options=options)
    except Exception as exc:
        error = _error_type(exc)
    finally:
        if client is not None:
            metadata = client.public_metadata()
            client.close()
    return result, metadata, error


def run_batch(cases, config, *, execute=False, allow_llm=False, accept_release_assumption=False, output_root=None):
    """Exactly five inputs. Each case keeps its actual status and storage status.

    Persist public per-case reports even when a case fails before generation.
    Unexpected execution errors have unavailable state, never invented Episodes.
    Files are individually atomic; storage failure does not discard later cases.
    """
    if not isinstance(cases, (list, tuple)) or len(cases) != CASE_LIMIT:
        raise ValueError("Provide exactly the first five cases")
    config = deepcopy(config)
    if execute is False:
        rows = []
        for ordinal, case in enumerate(cases, 1):
            try:
                inspection = inspect_development_llm(case, config=config)
                rows.append({"case_ref": "Case{:03d}".format(ordinal), "status": "dry_run", "inspection": inspection})
            except Exception as exc:
                rows.append({"case_ref": "Case{:03d}".format(ordinal), "status": "blocked", "error_type": _error_type(exc)})
        return {"schema_version": "locdi.real_development_batch.v1", "status": "dry_run", "case_count": CASE_LIMIT,
                "purpose": "development_llm", "development_only": True, "mock_execution": False,
                "clinical_metrics": "NOT_EVALUATED", "research_evaluation_eligible": False, "simulation_ready": False,
                "request_count": 0, "features": deepcopy(FEATURES), "cases": rows}
    if execute is not True or allow_llm is not True or accept_release_assumption is not True:
        raise ValueError("All three explicit execution acknowledgements are required")
    ready = validate_setup(config)
    expected = ready["prepared"]["fingerprint"] if ready["prepared"] else None
    batch_id = uuid4().hex
    session = ExportSession(run_id=batch_id)
    rows, documents = [], []
    for ordinal, case in enumerate(cases, 1):
        ref = "Case{:03d}".format(ordinal)
        result, metadata, error = _execute_case(case, deepcopy(config), expected)
        status = result["status"] if result is not None else "failed"
        if status not in {"completed", "incomplete", "failed"}:
            status, error, result = "failed", "ExecutionError", None
        errors = set(metadata["error_types"])
        if error:
            errors.add(error)
        if result is not None:
            errors.update(e["error_type"] if e.get("error_type") in ERRORS else "ExecutionError" for e in result.get("errors", []))
        row = {"case_ref": ref, "run_id": batch_id, "status": status,
            "outcome_source": "core_report" if result is not None and result.get("execution") is not None else "execution_state_unavailable",
            "request_count": metadata["request_count"], "identity_check_count": metadata["identity_check_count"],
            "real_model_execution": metadata["real_model_execution"], "error_types": sorted(errors),
            "diagnosis_unresolved": None, "elapsed_simulation_minutes": None,
            "files": {}, "storage_status": "completed", "storage_error_types": []}
        if result is not None:
            try:
                context = adapt_case(case)
                document = build_episode(result, context=context, session=session, case_ordinal=ordinal)
                # A completed case may include a rejected call later repaired by
                # Core. Preserve those audit errors as well as client failures.
                audit_errors = {d['error_type'] for d in document['workflow']['decisions']
                                if d['call_status'] == 'error' and d['error_type']}
                row['error_types'] = sorted(set(row['error_types']) | set(document['result']['error_types']) | audit_errors)
                row.update(diagnosis_unresolved=document["result"]["diagnosis_unresolved"],
                           elapsed_simulation_minutes=document["result"]["elapsed_simulation_minutes"])
                if metadata["request_count"] > 0:
                    path = export_episode(result, "episodes/{}_case{:03d}_real.json".format(batch_id, ordinal),
                        context=context, session=session, case_ordinal=ordinal, output_root=output_root)
                    row["files"]["episode"] = "episodes/" + path.name
                    documents.append(document)
            except Exception as exc:
                row.update(storage_status="failed", storage_error_types=[_error_type(exc)])
        try:
            name = "reports/{}_case{:03d}_real_run.json".format(batch_id, ordinal)
            _write({"schema_version": "locdi.real_development_case.v1", "purpose": "development_llm",
                    "development_only": True, "mock_execution": False, "clinical_metrics": "NOT_EVALUATED",
                    "research_evaluation_eligible": False, "simulation_ready": False,
                    "settings_fingerprint": expected, **row}, name, output_root, "reports")
            row["files"]["report"] = name
        except Exception as exc:
            row["storage_status"] = "failed"
            row["storage_error_types"].append(_error_type(exc))
        rows.append(row)
    document = {"schema_version": "locdi.real_development_batch.v1", "batch_id": batch_id,
        "purpose": "development_llm", "development_only": True, "mock_execution": False,
        "clinical_metrics": "NOT_EVALUATED", "research_evaluation_eligible": False, "simulation_ready": False,
        "selection": "first_five_in_input_order", "case_count": CASE_LIMIT,
        "model_identifier": config.get("model_identifier"), "model_version": config.get("model_version"),
        "request_budget_per_case": config.get("max_requests"),
        "settings_fingerprint": expected, "features": deepcopy(FEATURES), "cases": rows,
        "status_counts": {s: sum(r["status"] == s for r in rows) for s in ("completed", "incomplete", "failed")},
        "total_request_count": sum(r["request_count"] for r in rows),
        "total_identity_check_count": sum(r["identity_check_count"] for r in rows),
        "error_type_case_counts": dict(Counter(e for r in rows for e in set(r["error_types"]))),
        "storage_failed_cases": sum(r["storage_status"] == "failed" for r in rows),
        "summary_episode_count": len(documents), "files": {}}
    if documents:
        try:
            path = export_summary({"episodes": documents}, "metrics/{}_real_summary.jsonl".format(batch_id), output_root=output_root)
            document["files"]["summary"] = "metrics/" + path.name
        except Exception as exc:
            document["summary_storage_error_type"] = _error_type(exc)
    document["status"] = "completed" if document["status_counts"]["completed"] == CASE_LIMIT and not document["storage_failed_cases"] and "summary_storage_error_type" not in document else "issues_recorded"
    try:
        path = _write(document, "reports/{}_real_batch.json".format(batch_id), output_root, "reports")
        document["files"]["report"] = "reports/" + path.name
    except Exception as exc:
        document.update(status="issues_recorded", batch_report_storage_error_type=_error_type(exc))
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/development_llm.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-llm", action="store_true")
    parser.add_argument("--accept-release-assumption", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and not (args.allow_llm and args.accept_release_assumption):
        parser.error("Execution requires --allow-llm and --accept-release-assumption")
    cases = load_cases(ROOT / "datasets/raw/longitudinal_benchmark_50.jsonl")[:CASE_LIMIT]
    report = run_batch(cases, load_setup(args.config), execute=args.execute,
                       allow_llm=args.allow_llm, accept_release_assumption=args.accept_release_assumption)
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["status"] in {"dry_run", "completed"} else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"status": "blocked_or_error", "error_type": _error_type(exc)}), file=sys.stderr)
        sys.exit(2)
