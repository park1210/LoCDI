"""Minimal public projections and atomic, no-clobber JSON exports."""
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4
from app.locdi.case_adapter import AdaptedCase
from app.locdi.action_audit import audit_action, summarize_actions, audit_execution_roles, audit_selectors
from app.utils.department_catalog import SUPPORTED_DEPARTMENTS

SCHEMA_VERSION = "locdi.episode.v1"
EXPORTER_VERSION = "1.3.4"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "outputs"
STATUSES = {"completed", "incomplete", "failed"}
ACTIONS = {"ask_patient", "request_test", "consult_specialty", "diagnose", "retrieve_memory", "wait"}
SELECTORS = {"specialist_heuristic", "random", "bayesian_voi", "information_gain", "explicit_outcome_voi",
             "explicit_outcome_ig", "specialist_heuristic_no_supported_voi", "legacy_policy"}
EVENTS = {"case_started", "arrival_entered", "triage_queued", "triage_started", "triage_completed",
          "assessment_completed", "patient_moved", "question_asked", "answer_received", "test_ordered",
          "test_started", "test_resource_released", "test_completed", "result_received", "consult_requested",
          "specialist_dispatched", "specialist_arrived", "consult_started", "consult_note_added",
          "diagnosis_finalized", "observation_started", "discharge_started", "case_completed",
          "deterioration_triggered", "what_if_branch_started", "what_if_branch_completed"}
ENGINE_ACTIONS = {"arrival_entered", "triage_started", "perform_triage", "bed_assigned", "dispatch_er",
                  "ask_history", "order_test", "test_started", "test_resource_released", "test_result_ready",
                  "review_results", "request_specialty", "er_finalize", "specialty_finalize", "specialist_finalize", "discharge",
                  "observe", "consult_started", "consult_completed", "specialist_dispatched", "specialist_arrived"}
ROLES = {"system", "triage_nurse", "er_doctor", "specialist", "lab_slot", "imaging_slot", "er", "specialty", "patient"}
ERRORS = {"RoleCallError", "TimeoutError", "ValueError", "TypeError", "RuntimeError", "PermissionError",
          "ConnectionError", "ContractValidationError", "reported_execution_failure", "LLMRequestError",
          "LLMTimeoutError", "LLMResponseError", "LLMRequestLimitError", "ModelIdentityError"}
STOP_REASONS = {"role_call_failed", "visit_minute_limit_reached"}


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Export input is not finite JSON data") from None


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _enum(value, choices, default="unavailable"):
    return value if isinstance(value, str) and value in choices else default


def _minute(value):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid simulation minute")
    return value


def _ref(table, value, prefix):
    if value not in table:
        table[value] = "{}{:04d}".format(prefix, len(table) + 1)
    return table[value]


@dataclass
class ExportSession:
    """Session-local pseudonyms; raw linkage keys remain only in memory.

    Share a session across exports to link patients/visits/memories. Repeats
    need the same input order. This does not guarantee clinical anonymization.
    """
    run_id: str = field(default_factory=lambda: uuid4().hex)
    _patients: dict = field(default_factory=dict, repr=False)
    _visits: dict = field(default_factory=dict, repr=False)
    _memories: dict = field(default_factory=dict, repr=False)
    _cases: dict = field(default_factory=dict, repr=False)
    _ordinals: dict = field(default_factory=dict, repr=False)
    _actions: dict = field(default_factory=dict, repr=False)
    _bindings: dict = field(default_factory=dict, repr=False)
    _models: dict = field(default_factory=dict, repr=False)
    _versions: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{32}", self.run_id):
            raise ValueError("run_id must be a random UUID hex token")


def _configuration(metadata):
    raw = metadata.get("configuration")
    if not isinstance(raw, dict):
        raise ValueError("Recorded execution configuration is required")
    enums = {"memory_mode": {"none", "full_history", "selective", "summary", "semantic"},
             "routing_mode": {"er_only", "llm", "rule", "random", "adaptive"},
             "primary_router": {"rule_similarity", "adaptive", "llm", "memory_aware_llm"},
             "action_selector": SELECTORS, "memory_source": {"fixed"}}
    out = {}
    for key, choices in enums.items():
        if _enum(raw.get(key), choices, None) is None:
            raise ValueError("Invalid recorded configuration enum")
        out[key] = raw[key]
    for key in ("top_k", "minute_limit", "routing_seed", "max_retries", "max_memory_chars"):
        value = raw.get(key)
        if value is not None and type(value) is not int:
            raise ValueError("Invalid recorded integer configuration")
        out[key] = value
    for key in ("voi_enabled", "accept_recorded_release_assumption"):
        if type(raw.get(key)) is not bool:
            raise ValueError("Invalid recorded boolean configuration")
        out[key] = raw[key]
    out["dependency_fingerprints"] = {}
    for key in ("registry_manifest", "routing_utility", "voi_model", "embedding_artifact", "diagnosis_catalog", "diagnosis_departments"):
        value = raw.get("dependency_fingerprints", {}).get(key)
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
            raise ValueError("Invalid dependency fingerprint")
        out["dependency_fingerprints"][key] = value
    if "llm_settings_fingerprint" in raw:
        value = raw["llm_settings_fingerprint"]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Invalid LLM settings fingerprint")
        out["llm_settings_fingerprint"] = value
    if "sequential_source_view" in raw:
        value = raw["sequential_source_view"]
        expected_keys = {"policy", "fingerprint", "test_only", "turnaround_minutes",
                         "clinical_release_approved", "v1_grants_created", "candidate_wire_projection"}
        if (not isinstance(value, dict) or set(value) != expected_keys
                or value["policy"] not in {"SYNTHEA_SEQUENTIAL_RECORDED_TIME_PROXY.v1",
                                           "SYNTHEA_SEQUENTIAL_RECORDED_TIME_PROXY_B012.v1"}
                or not isinstance(value["fingerprint"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", value["fingerprint"])
                or type(value["test_only"]) is not bool
                or type(value["turnaround_minutes"]) is not int or value["turnaround_minutes"] != 5
                or value["clinical_release_approved"] is not False
                or type(value["v1_grants_created"]) is not int or value["v1_grants_created"] != 0
                or value["candidate_wire_projection"] != "lossless clinical data; action,target pairs unchanged"):
            raise ValueError("Invalid sequential synthetic source configuration")
        # Explicit optional schema extension. Keep full fingerprint validation;
        # unknown keys, altered clinical status and tampered values still fail.
        out["sequential_source_view"] = dict(value)
    if 'context_budgeted_history' in raw:
        value=raw['context_budgeted_history']
        keys={'policy','selection_fingerprint','context','future_reserve_bytes','max_joint_bytes','selection_unit','stable_within_visit'}
        if (not isinstance(value,dict) or set(value)!=keys
                or value['policy']!='B1_CONTEXT_BUDGETED_HISTORY.v1'
                or not isinstance(value['selection_fingerprint'],str) or not re.fullmatch(r'[0-9a-f]{64}',value['selection_fingerprint'])
                or type(value['context']) is not int or value['context']!=32768
                or type(value['future_reserve_bytes']) is not int or value['future_reserve_bytes']!=2048
                or type(value['max_joint_bytes']) is not int or value['max_joint_bytes']!=4096
                or value['selection_unit']!='complete_record_or_declared_visit_block'
                or value['stable_within_visit'] is not True or out['memory_mode']!='full_history'):
            raise ValueError('Invalid Context-Budgeted History metadata')
        out['context_budgeted_history']=dict(value)
    if set(raw) != set(out) or _fingerprint(out) != metadata.get("configuration_fingerprint"):
        raise ValueError("Configuration fingerprint mismatch")
    return out


def _wall_time(value):
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError()
        return parsed.isoformat()
    except (ValueError, TypeError):
        raise ValueError("Timezone-aware execution timestamp is required") from None


def build_episode(episode, *, context, session=None, case_ordinal=1):
    """Project one development run. Never copy free text or raw IDs/evidence.

    Unknown fields are omitted. Missing execution evidence stays unavailable;
    an accepted decision alone is not treated as a successfully executed action.
    """
    _json(episode)
    if (not isinstance(episode, dict) or not isinstance(context, AdaptedCase)
            or episode.get("case_count") != 1 or episode.get("status") not in STATUSES
            or episode.get("development_only") is not True
            or (episode.get("purpose"), episode.get("mock_execution")) not in {("development_smoke", True), ("development_llm", False)}
            or type(episode.get("mock_execution")) is not bool or episode.get("simulation_ready") is not False
            or episode.get("clinical_metrics") != "NOT_EVALUATED"):
        raise ValueError("Expected one recorded development-only mock execution")
    if type(case_ordinal) is not int or case_ordinal < 1:
        raise ValueError("case_ordinal must be positive")
    session = session if session is not None else ExportSession()
    if not isinstance(session, ExportSession):
        raise ValueError("Invalid export session")
    cid = context.case_id
    if (cid in session._cases and session._cases[cid] != case_ordinal
            or case_ordinal in session._ordinals and session._ordinals[case_ordinal] != cid):
        raise ValueError("Conflicting case ordinal in export session")
    session._cases[cid], session._ordinals[case_ordinal] = case_ordinal, cid
    identity = context.runtime_case["longitudinal"]
    patient_ref = _ref(session._patients, identity["patient_id"], "Patient")
    visit_ref = _ref(session._visits, (identity["patient_id"], identity["visit_id"]), "Visit")
    memory_map = {}
    for past in context.runtime_case["past_visits"]:
        for record in past["records"]:
            memory_map[record["memory_id"]] = _ref(session._memories, (identity["patient_id"], record["memory_id"]), "Memory")
    metadata = episode.get("execution_metadata", {})
    config = _configuration(metadata)
    started, ended = _wall_time(metadata.get("started_at")), _wall_time(metadata.get("ended_at"))
    if datetime.fromisoformat(ended) < datetime.fromisoformat(started):
        raise ValueError("Execution timestamps are reversed")
    execution = episode.get("execution")
    visit, final, state = {}, {}, {}
    if execution is not None:
        if (not isinstance(execution, dict) or execution.get("status") != episode["status"]
                or len(execution.get("visits", [])) != 1):
            raise ValueError("Unexpected single-case Core report")
        visit = execution["visits"][0]
        if visit.get("identity") != identity:
            raise ValueError("Export context does not match executed visit")
        final = visit["final_state"]
        if set(final.get("cases", {})) != {cid}:
            raise ValueError("Unexpected Runtime identity")
        state = final["cases"][cid]
    elif episode["status"] != "failed":
        raise ValueError("Nonfailed execution has no Core report")
    targets = {}

    def target(value):
        return _ref(targets, value, "Target") if isinstance(value, str) and value else None

    events = []
    for event in final.get("event_log", []):
        if event.get("case_id") != cid:
            raise ValueError("Cross-case event in single-case report")
        payload = event.get("payload", {})
        raw_target = next((payload[k] for k in ("requested_test", "test_name", "question", "department") if k in payload), None)
        events.append({"sequence": len(events) + 1, "minute": _minute(event.get("minute")),
            "agent_role": _enum(event.get("actor", {}).get("role"), ROLES),
            "event_type": _enum(event.get("event_type"), EVENTS),
            "executed_action": _enum(event.get("action"), ENGINE_ACTIONS),
            "target_ref": target(raw_target), "execution_status": "observed"})
    if [e["minute"] for e in events] != sorted(e["minute"] for e in events):
        raise ValueError("Core events are not in temporal order")
    decisions = []
    structural_audits = []
    def action_ref(kind, raw_target):
        # Private equality keys, scoped to case and session, never raw hashes.
        return _ref(session._actions, (cid, kind, raw_target), "Action")
    def binding_ref(kind, value):
        table, prefix = {"binding": (session._bindings, "Binding"), "model": (session._models, "Model"),
                         "version": (session._versions, "Version")}[kind]
        return _ref(table, value, prefix)
    audit = state.get("llm_audit", [])
    expected = {"ask_patient": "ask_history", "request_test": "order_test", "consult_specialty": "request_specialty", "diagnose": "er_finalize"}
    for trace in episode.get("telemetry", {}).get("decisions", []) if state else []:
        if (trace.get("case_id") != cid or type(trace.get("audit_index")) is not int
                or not 0 <= trace["audit_index"] < len(audit)):
            raise ValueError("Invalid decision telemetry")
        row = audit[trace["audit_index"]]
        chosen = row.get("decision", {})
        action = _enum(chosen.get("action"), ACTIONS, None)
        minute = _minute(trace.get("minute"))
        chosen_target = target(chosen.get("target"))
        # Canonical Scheduler specialty completion; retain the legacy enum for
        # reading/projection compatibility without equating its undocumented use.
        expected_action = "specialty_finalize" if action == "diagnose" and trace.get("role") == "specialty" else expected.get(action)
        matching = [e["sequence"] for e in events if e["minute"] == minute
                    and e["executed_action"] == expected_action
                    and (action == "diagnose" or e["target_ref"] == chosen_target)]
        routing = row.get("routing", {})
        selected = routing.get("selected_specialty")
        selector = row.get("selector_status", {})
        decisions.append({"sequence": len(decisions) + 1, "minute": minute,
            "agent_role": _enum(trace.get("role"), ROLES), "selected_action": action,
            "target_ref": chosen_target, "call_status": _enum(row.get("status"), {"success", "error"}),
            "execution_event_sequences": matching,
            "execution_success": True if matching else False if row.get("status") == "error" else None,
            "actual_selector": _enum(row.get("action_selector"), SELECTORS, None),
            "selector_status": _enum(selector.get("status"), {"available", "unavailable"}),
            "fallback_used": row.get("fallback_used") if type(row.get("fallback_used")) is bool else None,
            "routing_selection": _enum(selected, set(SUPPORTED_DEPARTMENTS), None),
            "routing_abstention": selected is None if "selected_specialty" in routing else None,
            "unavailable_reason": "er_only" if config["routing_mode"] == "er_only" else "not_recorded" if not routing else None,
            "error_type": _enum(row.get("error_type"), ERRORS, "ExecutionError") if row.get("status") == "error" else None})
        linked = [e for e in events if e["sequence"] in matching]
        observed = {"executed_action": linked[0]["executed_action"] if linked else None,
                    "success": decisions[-1]["execution_success"],
                    "target_matched": bool(matching) and action != "diagnose"}
        safe_audit = audit_action(row, observed, contract=trace.get("contract"),
                                  reference=action_ref, certainty=trace.get("certainty"))
        safe_audit.update(sequence=len(structural_audits) + 1, minute=minute)
        safe_audit.update(audit_selectors(row, requested_selector=config['action_selector'],
            artifact_present=config['dependency_fingerprints']['voi_model'] is not None))
        safe_audit.update(audit_execution_roles(row, entry_role=trace.get("entry_role", trace.get("role")),
            observed=trace.get("role_observation"), reference=binding_ref))
        structural_audits.append(safe_audit)
    retrievals = []
    for row in episode.get("telemetry", {}).get("retrievals", []):
        if row.get("case_id") != cid or any(mid not in memory_map for mid in row.get("memory_ids", [])):
            raise ValueError("Unknown memory reference in telemetry")
        retrievals.append({"sequence": len(retrievals) + 1, "minute": _minute(row.get("minute")),
            "mode": _enum(row.get("mode"), {"none", "full_history", "selective", "summary", "semantic"}),
            "memory_refs": [memory_map[mid] for mid in row.get("memory_ids", [])],
            "cutoff_compliant": row.get("cutoff_compliant") if type(row.get("cutoff_compliant")) is bool else None})
    errors = [_enum(e.get("error_type"), ERRORS, "ExecutionError") for e in episode.get("errors", [])]
    if visit.get("error_type"):
        errors.append(_enum(visit["error_type"], ERRORS, "ExecutionError"))
    dependencies = config["dependency_fingerprints"]
    capabilities = {"memory_mode": config["memory_mode"], "memory_source": config["memory_source"],
        "routing_mode": config["routing_mode"], "primary_router": config["primary_router"],
        "voi_enabled": config["voi_enabled"], "requested_selector": config["action_selector"],
        "artifacts": {k: {"present": dependencies[k] is not None,
                           "validation": ("contract_status_unavailable" if k == "embedding_artifact" and config["memory_mode"] != "semantic"
                                          else "contract_validated_not_research_reviewed") if dependencies[k] is not None else "absent"}
                      for k in ("registry_manifest", "routing_utility", "voi_model", "embedding_artifact")},
        "voi_unavailable_reason": "disabled" if not config["voi_enabled"] else None}
    outcome = {"status": episode["status"],
        "diagnosis_unresolved": "diagnosis_unresolved" in state.get("alerts", []) if state else None,
        "finalized": state.get("is_completed") if type(state.get("is_completed")) is bool else None,
        "stop_reason": _enum(visit.get("stop_reason"), STOP_REASONS, None),
        "elapsed_simulation_minutes": _minute(final["current_minute"]) if "current_minute" in final else None,
        "selected_action_counts": dict(Counter(d["selected_action"] for d in decisions if d["selected_action"])),
        "action_counts": dict(Counter(e["executed_action"] for e in events if e["executed_action"] in set(expected.values()) | {"specialty_finalize"})),
        "error_types": sorted(set(errors)),
        "trace_availability": "core_report_available" if state else "core_report_unavailable"}
    document = {"schema_version": SCHEMA_VERSION, "exporter_version": EXPORTER_VERSION,
        "execution_schema_version": "locdi.execution.v1", "run_id": session.run_id,
        "case_ref": "Case{:03d}".format(case_ordinal), "patient_ref": patient_ref, "visit_ref": visit_ref,
        "purpose": episode["purpose"], "development_only": True, "mock_execution": episode["mock_execution"],
        "clinical_metrics": "NOT_EVALUATED", "simulation_ready": False, "research_evaluation_eligible": False,
        "configuration": config, "configuration_fingerprint": _fingerprint(config), "started_at": started, "ended_at": ended,
        "workflow": {"events": events, "decisions": decisions},
        "memory": {"retrievals": retrievals, "initial_retrieval_minute": None, "initial_retrieval_timing": "not_recorded_by_core"},
        "capabilities": capabilities, "result": outcome}
    document["decision_audit"] = {"schema_version": "locdi.decision_audit.v1",
        "reference_scope": "session_case_local_not_cross_run_identity",
        "records": structural_audits, "summary": summarize_actions(structural_audits)}
    document["decision_audit"]["fingerprint"] = _fingerprint(document["decision_audit"])
    if episode["purpose"] == "development_llm":
        document["llm_execution"] = _llm_metadata(episode.get("llm_execution"), config)
    trace_keys = ["case_ref", "patient_ref", "visit_ref", "configuration_fingerprint", "workflow", "memory", "capabilities", "result"]
    if "llm_execution" in document:
        trace_keys.append("llm_execution")
    document["normalized_trace_fingerprint"] = _fingerprint({k: document[k] for k in trace_keys})
    _json(document)
    return document


def _llm_metadata(raw, config):
    if not isinstance(raw, dict):
        raise ValueError("Real execution metadata is required")
    identifier, version, fingerprint = raw.get("model_identifier"), raw.get("model_version"), raw.get("settings_fingerprint")
    if (not isinstance(identifier, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", identifier)
            or not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,120}", version)
            or fingerprint != config.get("llm_settings_fingerprint")):
        raise ValueError("Invalid real model identity metadata")
    for key in ("request_count", "identity_check_count"):
        if type(raw.get(key)) is not int or raw[key] < 0:
            raise ValueError("Invalid model request count")
    if type(raw.get("real_model_execution")) is not bool or raw["real_model_execution"] != (raw["request_count"] > 0):
        raise ValueError("Inconsistent model execution flag")
    return {"model_identifier": identifier, "model_version": version, "settings_fingerprint": fingerprint,
            "request_count": raw["request_count"], "identity_check_count": raw["identity_check_count"],
            "real_model_execution": raw["real_model_execution"],
            "error_types": [_enum(e, ERRORS, "ExecutionError") for e in raw.get("error_types", [])]}


def _write(document, destination, output_root, category):
    """Same-directory fsync + atomic exclusive hard link; never truncate/replace.

    Requires NTFS/POSIX hard links; failures propagate. Individual-file atomicity,
    not a multi-file transaction or protection against hostile directory races.
    """
    serialized = _json(document) + "\n"
    root = Path(output_root).resolve() if output_root is not None else DEFAULT_ROOT.resolve()
    supplied = Path(destination)
    if ".." in supplied.parts or any(":" in part for part in supplied.parts if part != supplied.anchor):
        raise ValueError("Invalid export path")
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("Export path escapes output root") from None
    if len(relative.parts) != 2 or relative.parts[0] != category or path.suffix not in {".json", ".jsonl"}:
        raise ValueError("Use category/filename.json or category/filename.jsonl")
    if path.exists():
        raise FileExistsError("Export destination already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve() != root / category:
        raise ValueError("Export directory redirects outside its category")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=str(path.parent),
                                         prefix=".locdi-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    except FileExistsError:
        raise FileExistsError("Export destination already exists") from None
    except OSError:
        raise OSError("Export storage operation failed; existing files were not replaced") from None
    finally:
        if temporary is not None:
            temporary.unlink()
    return path


def export_episode(episode, destination, *, context, session=None, case_ordinal=1, output_root=None):
    return _write(build_episode(episode, context=context, session=session, case_ordinal=case_ordinal),
                  destination, output_root, "episodes")


def export_summary(summary, destination, *, output_root=None):
    """Accept {'episodes': [public documents]}; export only workflow counts.

    JSONL has one versioned object per file; no append. No caller-provided text,
    clinical metrics, errors, or extra fields pass through.
    """
    _json(summary)
    if not isinstance(summary, dict) or not isinstance(summary.get("episodes"), list) or not summary["episodes"]:
        raise ValueError("Provide exported episode documents")
    rows, seen, modes = [], set(), set()
    for episode in summary["episodes"]:
        mode = (episode.get("purpose"), episode.get("mock_execution"))
        if (episode.get("schema_version") != SCHEMA_VERSION
                or mode not in {("development_smoke", True), ("development_llm", False)}
                or type(episode.get("mock_execution")) is not bool
                or episode.get("clinical_metrics") != "NOT_EVALUATED" or episode.get("simulation_ready") is not False):
            raise ValueError("Invalid public episode for summary")
        modes.add(mode)
        if len(modes) != 1:
            raise ValueError("Mock and real execution summaries must be separate")
        run_id, ref = episode.get("run_id"), episode.get("case_ref")
        if (not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id)
                or not isinstance(ref, str) or not re.fullmatch(r"Case[0-9]{3,}", ref)):
            raise ValueError("Invalid public references")
        if (run_id, ref) in seen:
            raise ValueError("Duplicate run/case in summary")
        seen.add((run_id, ref))
        result = episode.get("result", {})
        if result.get("status") not in STATUSES:
            raise ValueError("Invalid episode status")
        fingerprints = {}
        for key in ("configuration_fingerprint", "normalized_trace_fingerprint"):
            value = episode.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Invalid episode fingerprint")
            fingerprints[key] = value
        counts = result.get("action_counts", {})
        if any(k not in ENGINE_ACTIONS or type(v) is not int or v < 0 for k, v in counts.items()):
            raise ValueError("Invalid action counts")
        minute = result.get("elapsed_simulation_minutes")
        if minute is not None:
            _minute(minute)
        unresolved = result.get("diagnosis_unresolved")
        if unresolved is not None and type(unresolved) is not bool:
            raise ValueError("Invalid finalization flag")
        rows.append({"run_id": run_id, "case_ref": ref, "status": result["status"],
                     "diagnosis_unresolved": unresolved, "elapsed_simulation_minutes": minute,
                     "action_counts": counts, **fingerprints})
        if mode[0] == "development_llm":
            rows[-1]["llm_execution"] = _llm_metadata(episode.get("llm_execution"), episode.get("configuration", {}))
    document = {"schema_version": "locdi.summary.v1", "exporter_version": EXPORTER_VERSION,
                "purpose": next(iter(modes))[0], "development_only": True, "mock_execution": next(iter(modes))[1],
                "clinical_metrics": "NOT_EVALUATED", "simulation_ready": False, "research_evaluation_eligible": False,
                "episode_count": len(rows), "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted(STATUSES)},
                "episodes": rows}
    return _write(document, destination, output_root, "metrics")
