"""Read-only artifact transport checks, never clinical approval or execution.

No fitting, sampling, server access, path loading or model calls occur here.
Public diagnostics contain codes/counts, never patient IDs, evidence or errors.
DEVELOPMENT_CONTRACT_READY concerns the supplied artifact/data scope only;
it does not authorize a Runner, prove temporal visibility or validate research.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from collections.abc import Mapping

from app.core.agent_registry import AgentRegistry
from app.core.information_value import InformationPolicy
from app.core.routing_utility import RoutingUtility

ACTION_TYPES = {"Ask": "ask_patient", "Test": "request_test", "Consult": "consult_specialty",
                "Memory": "retrieve_memory", "ask_patient": "ask_patient", "request_test": "request_test",
                "consult_specialty": "consult_specialty", "retrieve_memory": "retrieve_memory"}
OBSERVATION_FIELDS = {"chief_complaint", "demographics", "arrival_info", "vital_signs", "elapsed_minutes", "health_state",
    "current_symptoms", "reassessment_pending", "reassessment_reason", "pending_results", "history", "objective_findings",
    "consults", "interventions", "longitudinal_memory", "working_assessment"}
PRIVATE_FIELDS = {"gold", "gold_label", "evaluation", "provenance", "task_annotations", "source_event_store",
                  "current_encounter_events", "hidden_test_results", "data_replay", "future_results"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def observation_scope_valid(observation):
    """Key-scope check only. Caller must establish release times and cutoffs."""
    def private(value):
        if isinstance(value, Mapping):
            return bool(set(value) & PRIVATE_FIELDS) or any(private(v) for v in value.values())
        if isinstance(value, list):
            return any(private(v) for v in value)
        return False
    return isinstance(observation, dict) and not (set(observation) - OBSERVATION_FIELDS) and not private(observation)


def _patients(values):
    if (not isinstance(values, (list, tuple, set)) or not values or
            any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values)):
        return None
    return set(values)


def _check(rows, code, status, *, development_required=True):
    rows.append({"code": code, "status": status, "development_required": development_required})


def _binding_pairs(bindings):
    if not isinstance(bindings, Mapping):
        return None
    result = {}
    for key, value in bindings.items():
        if not isinstance(key, str):
            return None
        if isinstance(value, Mapping):
            value = (value.get("model"), value.get("version"))
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(not isinstance(v, str) for v in value):
            return None
        result[key] = tuple(value)
    return result


def _finish(kind, structural, checks, details, artifact=None):
    ready = structural and all(c["status"] == "READY" for c in checks if c["development_required"])
    fingerprint = None
    try:
        if structural:
            fingerprint = _digest(artifact)
    except (ValueError, TypeError, OverflowError):
        structural, ready = False, False
        _check(checks, "FINITE_ARTIFACT_JSON", "INVALID")
    return {"schema_version": "locdi.artifact_readiness.v1", "artifact_type": kind,
        "structural_status": "STRUCTURAL_VALID" if structural else "BLOCKED",
        "development_status": "DEVELOPMENT_CONTRACT_READY" if ready else "BLOCKED",
        # No independent research review verifier exists in this phase. A
        # caller/metadata declaration can never promote the artifact here.
        "research_status": "NOT_REVIEWED", "research_validated": False,
        "research_blockers": ["INDEPENDENT_RESEARCH_REVIEW_NOT_VERIFIED"],
        "artifact_fingerprint": fingerprint, "checks": checks, "details": details,
        "stages": {"implementation": "READY", "structural_contract": "READY" if structural else "BLOCKED",
                   "offline_contract": "READY" if ready else "BLOCKED",
                   "real_llm_execution": "NOT_REVIEWED" if ready else "BLOCKED", "clinical_evaluation": "BLOCKED"},
        "scope": "supplied_artifact_and_static_data_contract_not_runner_authorization",
        "clinical_metrics": "NOT_EVALUATED"}


def routing_decision_valid(decision, registry):
    """Check the status/selection contract without scoring or replacing it."""
    try:
        bound = {a["department"] for a in registry.snapshot()["agents"] if a["role"] == "specialty"}
        candidates, allowed = decision["candidate_departments"], decision["allowed_departments"]
        if (not isinstance(candidates, list) or not isinstance(allowed, list) or
                any(not isinstance(x, str) for x in candidates + allowed) or
                not set(candidates) <= bound or not set(allowed) <= set(candidates)):
            return False
        selected, status = decision["selected_specialty"], decision["decision_status"]
        if status == "selected":
            return isinstance(selected, str) and selected in candidates and allowed == [selected]
        return status in ("abstained", "unavailable", "delegated", "disabled") and selected is None
    except (KeyError, TypeError, AttributeError):
        return False


def validate_routing_artifact(registry_manifest, utility_artifact, *, evaluation_patient_ids=None,
                              expected_bindings=None, observations=None, departments=None):
    """Core registry/schema checks + split/binding/observation contract checks.

    Insufficient similarity support is a valid abstention, not an execution
    error. Details intentionally omit scores, tokens and private identifiers.
    """
    checks, details = [], {"selection_status": "NOT_REVIEWED", "visibility_scope": "KEY_SCOPE_ONLY"}
    registry = None
    try:
        registry = AgentRegistry(deepcopy(registry_manifest))
        _check(checks, "REGISTRY_SNAPSHOT", "READY")
    except (ValueError, TypeError, KeyError, AttributeError):
        _check(checks, "REGISTRY_SNAPSHOT", "INVALID")
    specialties = [a for a in registry.snapshot()["agents"] if a["role"] == "specialty"] if registry else []
    details["specialty_count"] = len(specialties)
    details["registered_specialties"] = sorted(a["department"] for a in specialties)
    _check(checks, "SPECIALTY_BINDINGS", "READY" if specialties else "BLOCKED")
    model = None
    if registry and utility_artifact is not None:
        try:
            model = RoutingUtility(deepcopy(utility_artifact), registry)
            registry = AgentRegistry(deepcopy(registry_manifest), deepcopy(utility_artifact))
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    _check(checks, "ROUTING_UTILITY_CONTRACT", "READY" if model else "BLOCKED" if utility_artifact is None else "INVALID")
    if expected_bindings is not None:
        expected = {a["agent_id"]: (a["model"], a["version"]) for a in registry.snapshot()["agents"]} if registry else None
        _check(checks, "EXPECTED_MODEL_VERSION_BINDINGS", "READY" if _binding_pairs(expected_bindings) == expected and expected is not None else "INVALID")
    patients = _patients(evaluation_patient_ids)
    _check(checks, "EVALUATION_PATIENT_SCOPE", "READY" if patients is not None else "NOT_REVIEWED")
    if model:
        overlap = len(model.training_patients & patients) if patients is not None else None
        details.update(training_patient_count=len(model.training_patients), evaluation_patient_count=len(patients) if patients else None,
                       train_evaluation_overlap_count=overlap, min_samples=model.artifact["min_samples"])
        _check(checks, "PATIENT_SPLIT", "NOT_REVIEWED" if overlap is None else "READY" if overlap == 0 else "BLOCKED")
        counts = Counter(r["agent_id"] for r in model.artifact["samples"])
        details["support_capacity"] = "SUFFICIENT_COUNT_ONLY" if specialties and all(counts[a["agent_id"]] >= model.artifact["min_samples"] for a in specialties) else "INSUFFICIENT_COUNT_VALID_ABSTENTION"
    if observations is not None:
        valid_scope = isinstance(observations, (list, tuple)) and bool(observations) and all(observation_scope_valid(o) for o in observations)
        _check(checks, "OBSERVATION_KEY_SCOPE", "READY" if valid_scope else "INVALID")
        if valid_scope and registry and model:
            try:
                decisions = [registry.route(deepcopy(o), departments if departments is not None else [a["department"] for a in specialties], mode="adaptive") for o in observations]
                valid = all(routing_decision_valid(d, registry) for d in decisions)
                _check(checks, "ROUTING_DECISION_CONTRACT", "READY" if valid else "INVALID")
                details["decision_status_counts"] = dict(Counter(d["decision_status"] for d in decisions))
                details["selection_status"] = "VALID_ABSTENTION" if all(d["decision_status"] == "abstained" for d in decisions) else "RECORDED"
            except (ValueError, TypeError, KeyError, AttributeError):
                _check(checks, "ROUTING_DECISION_CONTRACT", "INVALID")
    else:
        _check(checks, "OBSERVATION_KEY_SCOPE", "NOT_REVIEWED", development_required=False)
    return _finish("routing", model is not None and bool(specialties), checks, details, utility_artifact)


def validate_voi_artifact(artifact, *, source_actions=None, evaluation_patient_ids=None,
                          expected_horizon=None, observations=None, diagnosis_candidates=None):
    """Validate existing likelihoods and source encodings, never estimate them.

    Source evidence is inspected as static encoding data, never sent to a
    Router/Policy as a current observation. Optional released observations are
    separately key-checked before observe(); no posterior is returned/exported.
    """
    checks, details = [], {"visibility_scope": "STATIC_ENCODING_AND_OPTIONAL_RELEASED_KEY_SCOPE"}
    policy = None
    try:
        policy = InformationPolicy(deepcopy(artifact))
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    _check(checks, "INFORMATION_MODEL_CONTRACT", "READY" if policy else "BLOCKED" if artifact is None else "INVALID")
    if policy is None:
        return _finish("voi", False, checks, details)
    model = policy.model.artifact
    provenance = model["provenance"]
    horizon = model["horizon"]
    _check(checks, "MULTISTEP_HORIZON", "READY" if horizon in (2, 3) and (expected_horizon is None or type(expected_horizon) is int and expected_horizon == horizon) else "BLOCKED")
    if diagnosis_candidates is not None:
        valid_catalog = (isinstance(diagnosis_candidates, (list, tuple)) and all(isinstance(x, str) for x in diagnosis_candidates)
                         and len(set(diagnosis_candidates)) == len(diagnosis_candidates) and set(diagnosis_candidates) == set(model["diagnoses"]))
        _check(checks, "SHARED_DIAGNOSIS_CATALOG", "READY" if valid_catalog else "INVALID")
    patients = _patients(evaluation_patient_ids)
    overlap = len(set(provenance.get("training_patients", [])) & patients) if patients is not None else None
    _check(checks, "TRAINING_PATIENT_DECLARATION", "READY" if provenance.get("training_patients") else "NOT_REVIEWED")
    _check(checks, "PATIENT_SPLIT", "NOT_REVIEWED" if overlap is None else "READY" if overlap == 0 else "BLOCKED")
    encoding = provenance.get("observation_encoding")
    _check(checks, "OBSERVATION_ENCODING", "READY" if encoding in (None, "replay_bins_v1") else "INVALID")
    bindings = {(a["type"], a["target"]): a for a in model["actions"]}
    missing, invalid, unsupported, bins = 0, 0, 0, 0
    required = set()
    if isinstance(source_actions, (list, tuple)) and source_actions:
        for event in source_actions:
            if not isinstance(event, Mapping):
                invalid += 1
                continue
            kind, target = event.get("action"), event.get("target")
            if not isinstance(kind, str) or kind not in ACTION_TYPES or not isinstance(target, str) or not target:
                invalid += 1
                continue
            key = (ACTION_TYPES[kind], target)
            required.add(key)
            action = bindings.get(key)
            if action is None:
                missing += 1
                continue
            evidence = event.get("evidence")
            if not isinstance(evidence, str):
                unsupported += 1
                continue
            if encoding == "replay_bins_v1":
                try:
                    payload = json.loads(evidence)
                    outcome_id = payload["replay_bin"]
                    if not isinstance(outcome_id, str) or outcome_id not in {o["id"] for o in action["outcomes"]}:
                        raise ValueError()
                    bins += 1
                except (ValueError, KeyError, TypeError):
                    unsupported += 1
            else:
                from app.core.information_outcomes import normalized
                text = target + ": " + evidence if key[0] == "request_test" else evidence
                if normalized(text) not in {normalized(o["observation"]["text"]) for o in action["outcomes"]}:
                    unsupported += 1
        _check(checks, "SOURCE_ACTION_COVERAGE", "READY" if not missing and not invalid else "BLOCKED")
        _check(checks, "SOURCE_OUTCOME_ENCODING", "READY" if not unsupported and not invalid and not missing else "BLOCKED")
    else:
        _check(checks, "SOURCE_ACTION_COVERAGE", "NOT_REVIEWED")
    if observations is not None:
        safe = isinstance(observations, (list, tuple)) and bool(observations) and all(observation_scope_valid(o) for o in observations)
        _check(checks, "RELEASED_OBSERVATION_KEY_SCOPE", "READY" if safe else "INVALID")
        if safe:
            try:
                for observation in observations:
                    policy.model.observe(deepcopy(observation))
                _check(checks, "RELEASED_OBSERVATION_SUPPORT", "READY")
            except (ValueError, TypeError, KeyError, AttributeError):
                _check(checks, "RELEASED_OBSERVATION_SUPPORT", "BLOCKED")
    else:
        _check(checks, "RELEASED_OBSERVATION_SUPPORT", "NOT_REVIEWED", development_required=False)
    details.update(source_schema_version=policy.model.source_schema_version, horizon=horizon,
        diagnosis_count=len(model["diagnoses"]), modeled_action_types=dict(Counter(a["type"] for a in model["actions"])),
        required_binding_count=len(required), missing_source_action_count=missing, invalid_source_action_count=invalid,
        unsupported_source_outcome_count=unsupported, replay_bin_match_count=bins,
        extra_modeled_binding_count=len(set(bindings) - required), train_evaluation_overlap_count=overlap,
        declared_review_status=provenance["review_status"], synthetic=provenance["synthetic"],
        diagnostic_support_outcome_count=sum(bool(o["supports_diagnoses"]) for a in model["actions"] for o in a["outcomes"]),
        terminal_utility_scope="ALGORITHM_VALIDATION_ONLY" if artifact.get("terminal_utility") is None or provenance.get("utility_scope") == "algorithm_validation_only" else "SUPPLIED_NOT_INDEPENDENTLY_REVIEWED",
        clinical_support_status="NOT_REVIEWED", likelihood_review_status="NOT_REVIEWED")
    return _finish("voi", True, checks, details, artifact)
