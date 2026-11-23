"""Allowlisted structural audits. No clinical strings, raw IDs or source hashes."""
from collections import Counter
from typing import Any, Dict, Iterable, Mapping

NOT_RECORDED = "NOT_RECORDED"
ACTION_MAP = {"ask_patient": "ask_history", "request_test": "order_test",
              "consult_specialty": "request_specialty", "diagnose": "er_finalize",
              "treat": "treat", "retrieve_memory": "retrieve_memory"}
SELECTORS = {"specialist_heuristic", "random", "bayesian_voi", "information_gain",
             "explicit_outcome_voi", "explicit_outcome_ig", "legacy_policy",
             "specialist_heuristic_no_supported_voi"}
SOURCES = {"NOT_APPLICABLE", "ORDERABLE_SOURCE_RESULT_GATED", "RELEASED_SOURCE", "SOURCE_UNAVAILABLE"}
FAILURES = {"invalid_json", "schema_validation", "timeout", "model_call_error", "output_truncated"}
REASONS = {"diagnosis_is_department", "decision_required_text"}
TERMINATIONS = {"decision_budget_exhausted", "visit_time_budget_exhausted", "available_steps_exhausted"}


def _enum(value, choices, default=NOT_RECORDED):
    return value if isinstance(value, str) and value in choices else default


def _bool(value):
    return value if type(value) is bool else None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def capture_contract(payload, *, source_events=None, minute=None):
    """Observe the actual menu. Private targets stay internal until projection.

    Test orderability is separate from result visibility. Diagnose has a type
    wildcard, not a source event. No retrieval/policy/candidate generator runs.
    """
    result = {"allowed_actions": list(payload["allowed_actions"]),
              "allow_unresolved": _bool(payload.get("allow_unresolved_finalization")),
              "eligible_evidence_count": len(payload["supported_diagnosis_eligible_indices"])
                  if isinstance(payload.get("supported_diagnosis_eligible_indices"), list) else None,
              "candidates": []}
    kinds = {"ask_patient": "Ask", "request_test": "Test", "consult_specialty": "Consult"}
    for c in payload.get("candidate_actions", []):
        kind, target = c.get("action"), c.get("target")
        source = "NOT_APPLICABLE" if kind == "diagnose" else NOT_RECORDED
        if source_events is not None and kind in kinds:
            found = any(e.get("action") == kinds[kind] and e.get("target") == target and
                        (kind == "request_test" or type(minute) is int and e["available_minute"] <= minute)
                        for e in source_events)
            source = ("ORDERABLE_SOURCE_RESULT_GATED" if kind == "request_test" else "RELEASED_SOURCE") if found else "SOURCE_UNAVAILABLE"
        result["candidates"].append({"action": kind, "target": target, "source_status": source})
    return result


def audit_action(selected: Mapping[str, Any], executed: Mapping[str, Any], *,
                 contract=None, reference=None, certainty=None) -> Dict[str, Any]:
    """Project Core audit + observed execution; missing evidence stays unknown.

    reference(kind,target) must assign session-local opaque ordinals. Wildcard
    candidate membership proves action type only, never diagnosis identity.
    """
    chosen = selected.get("decision", {})
    action = _enum(chosen.get("action"), ACTION_MAP, None)
    candidates = contract.get("candidates") if isinstance(contract, Mapping) else selected.get("candidate_actions")
    allowed = contract.get("allowed_actions") if isinstance(contract, Mapping) else None
    rows, hit = None, None
    if isinstance(candidates, list):
        rows = [{"action_ref": reference(c.get("action"), c.get("target")) if reference and c.get("action") in ACTION_MAP else None,
                 "action_type": _enum(c.get("action"), ACTION_MAP, None),
                 "source_availability_status": _enum(c.get("source_status"), SOURCES),
                 "target_scope": "TYPE_ONLY" if c.get("target") is None else "EXACT_SESSION_TARGET"} for c in candidates]
        hit = next((c for c in candidates if action and c.get("action") == action and
                    (c.get("target") is None or c.get("target") == chosen.get("target"))), None)
    validation = selected.get("decision_validation")
    citations = selected.get("memory_citations")
    cited = validation.get("cited_evidence") if isinstance(validation, Mapping) else None
    supports = any(r.get("supports_finalization") is True for r in cited) if isinstance(cited, list) else None
    termination = _enum(validation.get("termination_reason"), TERMINATIONS, None) if isinstance(validation, Mapping) else None
    rejected = selected.get("status") == "error"
    response_rejected = rejected and selected.get("failure_category") in {"invalid_json", "schema_validation", "output_truncated"}
    source = _enum(hit.get("source_status"), SOURCES) if hit else NOT_RECORDED
    actual = _enum(executed.get("executed_action"), set(ACTION_MAP.values()) | {"specialty_finalize"}, None)
    success = _bool(executed.get("success"))
    if rejected and actual is None:
        success = None  # Model failure is not an attempted Scheduler execution.
    gate = ("STRUCTURAL_ACCEPTED" if supports or termination else NOT_RECORDED) if action == "diagnose" else "NOT_APPLICABLE" if action else NOT_RECORDED
    return {"audit_version": "locdi.action_audit.v1",
        "availability": "RECORDED" if contract is not None or candidates is not None or validation is not None else NOT_RECORDED,
        "allowed_action_types": [_enum(a, ACTION_MAP, None) for a in allowed] if isinstance(allowed, list) else None,
        "candidate_action_count": len(candidates) if isinstance(candidates, list) else None, "candidates": rows,
        "selected_action_ref": reference(action, chosen.get("target")) if reference and action else None,
        "selected_action_type": action,
        "selected_in_candidates": hit is not None if isinstance(candidates, list) and action else None,
        "candidate_match_scope": ("TYPE_ONLY" if hit.get("target") is None else "EXACT_SESSION_TARGET") if hit else NOT_RECORDED,
        "source_availability_status": source,
        "source_event_linked": True if source in {"ORDERABLE_SOURCE_RESULT_GATED", "RELEASED_SOURCE"} else False if source == "SOURCE_UNAVAILABLE" else None,
        "executed_action_type": actual,
        "execution_status": "NOT_EXECUTED" if rejected and actual is None else "OBSERVED" if success is True else "FAILED" if success is False else NOT_RECORDED,
        "execution_success": success,
        "action_mapping_matches": (actual == ACTION_MAP[action] or action == "diagnose" and actual == "specialty_finalize") if action and actual else None,
        "execution_target_validation": "NOT_CHECKED" if action == "diagnose" else "SESSION_TARGET_MATCH" if executed.get("target_matched") is True else NOT_RECORDED,
        "decision_validation_status": "REJECTED" if response_rejected else "CALL_FAILED" if rejected else "RECORDED" if isinstance(validation, Mapping) else NOT_RECORDED,
        "rejected_action_type": action if rejected else None,
        "rejection_reason": _enum(selected.get("validation_detail", {}).get("code"), REASONS,
                                    _enum(selected.get("failure_category"), FAILURES)) if rejected else None,
        "rejected_action_availability": NOT_RECORDED if rejected and action is None else "RECORDED" if rejected else "NOT_APPLICABLE",
        "repair_requested": _bool(selected.get("repair_requested")),
        "selector": _enum(selected.get("action_selector"), SELECTORS),
        "selector_status": _enum(selected.get("selector_status", {}).get("status"), {"available", "unavailable"}),
        "fallback_used": _bool(selected.get("fallback_used")),
        "proposal_action_type": _enum(selected.get("specialist_proposal", {}).get("action"), ACTION_MAP, None),
        "diagnose_allowed": "diagnose" in allowed if isinstance(allowed, list) else None,
        "diagnosis_finalization_status": ("EXECUTED" if success is True else "SELECTED_EXECUTION_UNCONFIRMED") if action == "diagnose" else "NOT_SELECTED" if action else NOT_RECORDED,
        "certainty": _enum(certainty, {"supported", "unresolved"}, None),
        "allow_unresolved_finalization": _bool(contract.get("allow_unresolved")) if isinstance(contract, Mapping) else None,
        "eligible_evidence_count": _count(contract.get("eligible_evidence_count")) if isinstance(contract, Mapping) else None,
        "evidence_gate_status": gate, "cited_evidence_count": len(cited) if isinstance(cited, list) else None,
        "eligible_evidence_cited": supports,
        "clinical_support_verified": _bool(validation.get("clinical_support_verified")) if isinstance(validation, Mapping) else None,
        "termination_reason": termination,
        "citation_count": len(citations["ids"]) if isinstance(citations, Mapping) and isinstance(citations.get("ids"), list) else None,
        "invalid_citation_count": len(citations["invalid_ids"]) if isinstance(citations, Mapping) and isinstance(citations.get("invalid_ids"), list) else None,
        "citation_reported": _bool(citations.get("reported")) if isinstance(citations, Mapping) else None,
        "citation_validation_status": ("VALID" if citations.get("valid") is True else "INVALID" if citations.get("valid") is False else NOT_RECORDED) if isinstance(citations, Mapping) else NOT_RECORDED}


def summarize_actions(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Workflow counts only, with unknown execution separate from failure."""
    rows = list(records)
    return {"decision_count": len(rows),
            "selected_action_counts": dict(Counter(r.get("selected_action_type") for r in rows if r.get("selected_action_type") in ACTION_MAP)),
            "execution_status_counts": dict(Counter(_enum(r.get("execution_status"), {"OBSERVED", "FAILED", "NOT_EXECUTED"}) for r in rows)),
            "rejected_count": sum(r.get("decision_validation_status") == "REJECTED" for r in rows),
            "clinical_metrics": "NOT_EVALUATED"}


def audit_selectors(row, *, requested_selector=None, artifact_present=None):
    """Project recorded selector outcomes only; availability is not selection.

    Artifact presence is configuration evidence, not proof it was consumed.
    SUCCESS describes an observed accepted VoI choice, never clinical success.
    """
    actual = _enum(row.get('action_selector'), SELECTORS)
    selection = row.get('selector_status')
    selection = selection if isinstance(selection, Mapping) else {}
    requested = _enum(requested_selector, SELECTORS)
    voi = NOT_RECORDED
    if requested in {'bayesian_voi', 'information_gain'}:
        if selection.get('status') == 'unavailable':
            voi = 'UNAVAILABLE'
        elif (actual in {'explicit_outcome_voi', 'explicit_outcome_ig'}
              and selection.get('status') == 'available'
              and row.get('decision', {}).get('action') in ACTION_MAP):
            voi = 'SUCCESS'
        elif selection.get('status') in {'error', 'failed'}:
            voi = 'FAILED'
    elif requested != NOT_RECORDED:
        voi = 'NOT_APPLICABLE'
    return {'requested_selector': requested, 'actual_selector': actual,
            'voi_artifact_available': artifact_present if type(artifact_present) is bool else NOT_RECORDED,
            'voi_selection_status': voi}


def read_decision_audit(document):
    """Read an optional v1 extension without manufacturing historical menus."""
    audit = document.get("decision_audit")
    if isinstance(audit, Mapping) and audit.get("schema_version") == "locdi.decision_audit.v1":
        return audit
    return {"schema_version": "locdi.decision_audit.v1", "availability": NOT_RECORDED, "records": []}


def audit_execution_roles(row, *, entry_role=None, observed=None, reference=None):
    """Keep Scheduler entry and request roles separate; no inferred routing.

    Request metadata wins. Old Core role/model/executing_agent may be used only
    when actually recorded and mutually consistent. Binding values stay private.
    """
    from app.utils.department_catalog import SUPPORTED_DEPARTMENTS
    roles = {"er", "specialty", "router", "patient"}
    observed = observed if isinstance(observed, Mapping) else {}
    effective = _enum(observed.get("effective_model_role", row.get("role")), roles)
    department = _enum(observed.get("effective_department"), set(SUPPORTED_DEPARTMENTS) | {""})
    binding, status = observed.get("model_binding"), _enum(observed.get("binding_status"), {"MATCHED", "MISMATCH"})
    routing = row.get("routing", {})
    if not isinstance(routing, Mapping):
        routing = {}
    if not observed:
        saved = routing.get("executing_agent")
        if isinstance(saved, Mapping) and saved.get("role") == effective and row.get("model") == saved.get("model") and isinstance(row.get("model"), str):
            binding, status = saved, "MATCHED"
            department = _enum(saved.get("department"), set(SUPPORTED_DEPARTMENTS) | {""})
    public_binding = NOT_RECORDED
    if status == "MATCHED" and isinstance(binding, Mapping) and reference:
        fields = ("agent_id", "role", "department", "model", "version")
        if all(isinstance(binding.get(k), str) for k in fields):
            public_binding = {"binding_ref": reference("binding", tuple(binding[k] for k in fields)),
                "role": _enum(binding.get("role"), roles),
                "department": _enum(binding.get("department"), set(SUPPORTED_DEPARTMENTS) | {""}),
                "model_ref": reference("model", binding["model"]),
                "version_ref": reference("version", binding["version"])}
    decision_status = _enum(routing.get("decision_status"), {"selected", "abstained", "unavailable", "delegated", "disabled"})
    selected = routing.get("selected_specialty", NOT_RECORDED)
    selected = None if selected is None else _enum(selected, set(SUPPORTED_DEPARTMENTS))
    return {"entry_role": _enum(entry_role, roles), "effective_model_role": effective,
        "effective_department": department, "model_binding": public_binding, "model_binding_status": status,
        "routing_decision_status": decision_status, "selected_specialty": selected,
        "routing_abstained": decision_status == "abstained" if decision_status != NOT_RECORDED else NOT_RECORDED,
        "fallback_used": _bool(row.get("fallback_used")) if type(row.get("fallback_used")) is bool else NOT_RECORDED}
