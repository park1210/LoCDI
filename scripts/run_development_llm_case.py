"""First LoCDI case only. Default dry-run; generation needs all opt-in flags."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import load_setup
from app.locdi.run_simulation import run_development_llm
from app.locdi.episode_exporter import ExportSession, build_episode, export_episode, export_summary, _write

FEATURES = {"memory_mode": "selective", "top_k": 5, "routing_mode": "er_only",
            "primary_router": "rule_similarity", "voi_enabled": False, "action_selector": "specialist_heuristic"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/development_llm.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-llm", action="store_true")
    parser.add_argument("--accept-release-assumption", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and not (args.allow_llm and args.accept_release_assumption):
        parser.error("Execution requires --allow-llm and --accept-release-assumption")
    config = load_setup(args.config)
    case = load_cases(ROOT / "datasets/raw/longitudinal_benchmark_50.jsonl")[0]
    if not args.execute:
        print(json.dumps(run_development_llm(case, config=config), ensure_ascii=True))
        return 0
    result = run_development_llm(case, config=config, execute=True, allow_llm=True,
                    runner_options=FEATURES, accept_recorded_release_assumption=True)
    metadata = result["llm_execution"]
    run_id = uuid4().hex
    paths = {}
    # A failed pre-generation identity/connection check is not an LLM episode.
    if metadata["request_count"] > 0:
        context = adapt_case(case)
        session = ExportSession(run_id=run_id)
        document = build_episode(result, context=context, session=session)
        episode_path = export_episode(result, "episodes/{}_real_case001.json".format(run_id), context=context, session=session)
        summary_path = export_summary({"episodes": [document]}, "metrics/{}_real_summary.jsonl".format(run_id))
        paths.update(episode=str(episode_path.relative_to(ROOT)), summary=str(summary_path.relative_to(ROOT)))
    report = {"schema_version": "locdi.real_development_run.v1", "run_id": run_id, "case_ref": "Case001",
              "purpose": "development_llm", "development_only": True, "mock_execution": False,
              "clinical_metrics": "NOT_EVALUATED", "research_evaluation_eligible": False, "simulation_ready": False,
              "status": result["status"], "llm_execution": metadata, "files": paths}
    report_path = _write(report, "reports/{}_real_run.json".format(run_id), None, "reports")
    print(json.dumps({"status": result["status"], "request_count": metadata["request_count"],
                      "report": str(report_path.relative_to(ROOT)), **paths}, ensure_ascii=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        # Raw exceptions may contain SDK responses or case data.
        print(json.dumps({"status": "blocked_or_error", "error_type": type(exc).__name__}), file=sys.stderr)
        sys.exit(2)
