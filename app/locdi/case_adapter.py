"""Validated, copy-isolated LoCDI to legacy encounter conversion."""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple
from app.core.clinical_observation import prepare_encounter
from app.locdi.dataset_loader import LoCDICase
from validation.schema_validator import ValidationResult, validate_schema, _Check, _schema_block
from validation.temporal_validator import validate_temporal
from validation.action_validator import validate_actions
from validation.provenance_validator import validate_provenance


@dataclass
class AdaptedCase:
    """Separate stores; runtime_case is ENGINE PRIVATE, never an LLM payload.

    runtime_case may contain unreleased source events for encounter_evidence.
    Only projected events are copied into runtime_case; no evaluation, gold,
    source store, provenance or replay metadata is passed to the engine. Replay
    metadata lives in audit_metadata. Warnings remain attached and do not
    establish clinical/model-run readiness. Use role observations for models.
    """
    case_id: str
    agent_visible: Dict[str, Any]
    environment: Dict[str, Any]
    evaluation: Dict[str, Any]
    task_annotations: Dict[str, Any]
    provenance: Dict[str, Any]
    runtime_case: Dict[str, Any]
    audit_metadata: Dict[str, Any] = field(default_factory=dict)
    validation_results: Tuple[ValidationResult, ...] = ()
    simulation_ready: bool = False


class CaseAdaptationError(ValueError):
    """Sanitized failure with internal identity and structured diagnostics."""
    def __init__(self, case_id, results=(), message="LoCDI validation failed; inspect structured issue paths"):
        super().__init__(message)
        self.case_id = case_id
        self.results = tuple(results)


def inspect_consult_bindings(case: LoCDICase, registry_manifest=None) -> ValidationResult:
    """Validate canonical runtime bindings without contacting models or routing.

    A binding check cannot clear Consult source/clinical NOT_REVIEWED status.
    No prospective model/version is manufactured when a manifest is missing.
    """
    blocked = _schema_block(case, "consult_bindings")
    if blocked:
        return blocked
    c = _Check("consult_bindings", case)
    consults = [e for e in case["environment"]["current_encounter_events"] if e["action"] == "Consult"]
    if consults:
        if registry_manifest is None:
            c.add("CONSULT_REGISTRY_REQUIRED", "environment.current_encounter_events",
                  "An explicit specialty Registry is required for Consult adaptation", "BLOCKED")
        else:
            from app.core.agent_registry import AgentRegistry
            try:
                registry = AgentRegistry(deepcopy(registry_manifest))
                for event in consults:
                    registry.get("specialty", event["target"])
            except (ValueError, TypeError, KeyError, AttributeError):
                c.add("CONSULT_SPECIALTY_UNBOUND", "environment.current_encounter_events",
                      "Registry is invalid or lacks a requested specialty binding")
    return c.result(checks=("consult_registry_bindings",),
                    not_reviewed=("live_model_identity", "consult_source_clearance", "clinical_review"))


def adapt_case(case: LoCDICase, *, registry_manifest=None) -> AdaptedCase:
    """Validate all four scopes, whitelist engine fields, prepare encounter.

    FAIL/BLOCKED rejects the input. WARNING/NOT_REVIEWED is preserved without
    claiming clinical clearance or observed release times. No patient/evidence
    value, chronology or evaluation label is synthesized to make input work.
    """
    schema = validate_schema(case)
    identity = schema.case_id
    if schema.status == "FAIL":
        raise CaseAdaptationError(identity, (schema,))
    results = (schema, validate_temporal(case), validate_actions(case), validate_provenance(case))
    if any(e["action"] == "Consult" for e in case["environment"]["current_encounter_events"]):
        results += (inspect_consult_bindings(case, registry_manifest),)
    if any(r.status in {"FAIL", "BLOCKED", "SKIPPED", "NOT_IMPLEMENTED"} for r in results) or results[2].status != "PASS":
        raise CaseAdaptationError(identity, results)
    av = deepcopy(case["agent_visible"])
    events = deepcopy(case["environment"]["current_encounter_events"])
    # Explicit fields instead of merging any wrapper or hidden environment.
    engine = {key: deepcopy(av[key]) for key in ("case_id", "current_initial_state", "index_encounter_id",
                                                "index_encounter_start", "longitudinal", "past_visits")}
    engine["current_encounter_events"] = deepcopy(events)
    try:
        runtime = prepare_encounter(engine)
    except (ValueError, KeyError, TypeError, OverflowError):
        raise CaseAdaptationError(identity, results, "Existing prepare_encounter contract rejected the validated input") from None
    return AdaptedCase(case["case_id"], av,
                       {"current_encounter_events": events, "source_event_store": deepcopy(case["environment"]["source_event_store"])},
                       deepcopy(case["evaluation"]), deepcopy(case["task_annotations"]), deepcopy(case["provenance"]), runtime,
                       {"data_replay": deepcopy(case["environment"]["data_replay"])}, results, False)


def adapt_cases(cases: Iterable[LoCDICase], *, registry_manifest=None) -> List[AdaptedCase]:
    """Preserve input order; reject repeated case/patient-visit identities.

    This does not schedule visits or sort them; the legacy runner owns ordering.
    Use validate_dataset first to collect all failures instead of failing fast.
    """
    result, ids, visits = [], set(), set()
    for case in cases:
        adapted = adapt_case(case, registry_manifest=registry_manifest)
        identity = adapted.runtime_case["longitudinal"]
        key = (identity["patient_id"], identity["visit_id"])
        if adapted.case_id in ids or key in visits:
            raise CaseAdaptationError(adapted.case_id, adapted.validation_results, "Duplicate case or patient/visit in adaptation batch")
        ids.add(adapted.case_id)
        visits.add(key)
        result.append(adapted)
    return result
