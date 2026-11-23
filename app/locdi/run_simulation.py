"""Development-only LoCDI orchestration, with explicit offline execution."""
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence
from unittest.mock import patch
from app.core.agent_registry import AgentRegistry
from app.core.information_value import InformationPolicy
from app.core.longitudinal_memory import PatientMemoryStore, timestamp
from app.core.memory_retrieval import MemoryEmbeddings
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_llm_policy import HospitalLLMPolicy
from app.locdi.case_adapter import AdaptedCase, adapt_cases
from app.locdi.dataset_loader import LoCDICase
from app.locdi.action_audit import capture_contract
import app.core.hospital_llm_policy as policy_module
from validation.run_validation import DatasetValidationReport


class SimulationConfigurationError(ValueError):
    """Missing, inconsistent, or unsupported execution configuration."""


class ResearchExecutionBlocked(SimulationConfigurationError):
    """Research evaluation is not enabled in this development phase."""


class OfflineClient(ABC):
    """Explicit test-client interface; never an SDK or real-model backend.

    Implementations belong in tests. They may choose source-backed actions or
    unresolved termination, but must not fabricate patient facts/diagnoses.
    """
    @abstractmethod
    def chat_json(self, system_prompt: str, user_prompt: str, model=None, *, schema=None) -> str:
        raise NotImplementedError


FEATURE_KEYS = {"memory_mode", "top_k", "routing_mode", "primary_router", "voi_enabled", "action_selector"}
_SOCKET_LOCK = RLock()
_ACCEPTED_SMOKE_ISSUES = {"VISIT_END_NOT_RECORDED", "RELEASE_TIME_UNOBSERVED", "LATEST_VALUE_REPLAY_UNREVIEWED",
                          "DATE_PRECISION_UNREVIEWED", "CLINICAL_REVIEW_UNVERIFIED", "SOURCE_BYTES_NOT_VERIFIED",
                          "COINCIDENT_PAST_VISITS"}


def translate_runner_options(features: Mapping[str, Any], *, memory_source: str) -> Dict[str, Any]:
    """Translate explicit switches, without changing official experiment JSON.

    None top_k is meaningful only for none/full_history. Core nevertheless
    requires a positive int; use 1 as an UNUSED transport value and log both.
    Artifacts/registry/client are separate dependencies, never inferred here.
    """
    if not isinstance(features, Mapping) or set(features) != FEATURE_KEYS:
        raise SimulationConfigurationError("Provide exactly the six explicit feature switches")
    mode = features["memory_mode"]
    if not isinstance(mode, str) or mode not in {"none", "full_history", "selective", "summary", "semantic"}:
        raise SimulationConfigurationError("Unsupported memory_mode")
    top_k = features["top_k"]
    if top_k is None:
        if mode not in {"none", "full_history"}:
            raise SimulationConfigurationError("Selective/semantic/summary modes require a positive top_k")
        core_top_k = 1
    elif type(top_k) is int and top_k > 0:
        core_top_k = top_k
    else:
        raise SimulationConfigurationError("top_k must be null where unused, or a positive integer")
    routing, primary = features["routing_mode"], features["primary_router"]
    if not isinstance(routing, str) or routing not in {"llm", "er_only", "rule", "random", "adaptive"}:
        raise SimulationConfigurationError("Unsupported routing_mode")
    if not isinstance(primary, str) or primary not in {"rule_similarity", "adaptive", "llm", "memory_aware_llm"}:
        raise SimulationConfigurationError("Unsupported primary_router")
    enabled, selector = features["voi_enabled"], features["action_selector"]
    if type(enabled) is not bool or not isinstance(selector, str):
        raise SimulationConfigurationError("voi_enabled must be boolean and action_selector must be explicit")
    if selector not in ({"bayesian_voi", "information_gain"} if enabled else {"specialist_heuristic", "random"}):
        raise SimulationConfigurationError("Action selector and VoI activation disagree")
    if memory_source != "fixed":
        raise SimulationConfigurationError("This development entry supports fixed memory only")
    return {"mode": mode, "top_k": core_top_k, "routing_mode": routing, "primary_router": primary,
            "memory_source": memory_source, "action_selector": selector}


def _deny_network(*args, **kwargs):
    raise PermissionError("Network access is disabled during development smoke execution")


@contextmanager
def _offline_network_guard():
    """Block ordinary Python socket connections for synchronous smoke runs.

    Process-wide patches are serialized. This is not a sandbox for hostile
    client code or a concurrent production service; only trusted test clients
    implementing OfflineClient are accepted.
    """
    with _SOCKET_LOCK, ExitStack() as stack:
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection", "socket.getaddrinfo"):
            stack.enter_context(patch(name, _deny_network))
        yield


class _OfflineClientProxy(OfflineClient):
    """Always truthy so legacy `client or OllamaClient()` never constructs SDKs."""
    def __init__(self, client: OfflineClient):
        self._client = client

    def chat_json(self, system_prompt, user_prompt, model=None, *, schema=None):
        response = self._client.chat_json(system_prompt, user_prompt, model=model, schema=schema)
        if not isinstance(response, str):
            raise TypeError("Offline client must return a JSON string")
        return response


@dataclass(frozen=True)
class SimulationOptions:
    """Explicit development prerequisites, preserving the first five fields.

    runner_options contains the six experiment feature switches, NOT arbitrary
    legacy kwargs/callbacks/resume reports. None initial_memory means collect
    only the validated wrappers' past records. Explicit initial_memory must
    exactly equal that fixed set. No model/registry is manufactured here.
    """
    allow_llm: bool = False
    minute_limit: Optional[int] = None
    registry_manifest: Optional[Mapping[str, Any]] = None
    initial_memory: Optional[Sequence[Mapping[str, Any]]] = None
    runner_options: Optional[Mapping[str, Any]] = None
    purpose: str = "development_smoke"
    client: Optional[OfflineClient] = None
    accept_recorded_release_assumption: bool = False
    memory_source: str = "fixed"
    routing_seed: int = 0
    max_retries: int = 0
    max_memory_chars: Optional[int] = None
    routing_utility: Optional[Mapping[str, Any]] = None
    voi_model: Optional[Mapping[str, Any]] = None
    embedding_artifact: Optional[Mapping[str, Any]] = None
    diagnosis_catalog: Optional[Sequence[str]] = None
    diagnosis_departments: Optional[Mapping[str, str]] = None
    llm_config: Optional[Mapping[str, Any]] = None
    dry_run: bool = True
    sequential_source_view: Optional[Any] = None
    history_policy: Optional[Any] = None
    # Explicit development experiment only: Core already abstains with no utility.
    allow_unfitted_adaptive_development: bool = False


def _configuration(options: SimulationOptions) -> Dict[str, Any]:
    if not isinstance(options, SimulationOptions):
        raise SimulationConfigurationError("options must be SimulationOptions")
    if options.history_policy is not None:
        from app.locdi.context_budgeted_history import ContextBudgetedHistory
        if (not isinstance(options.history_policy, ContextBudgetedHistory) or not options.runner_options
                or options.runner_options.get('memory_mode') != 'full_history'
                or options.runner_options.get('routing_mode') != 'er_only'
                or options.runner_options.get('voi_enabled') is not False
                or options.memory_source != 'fixed' or options.max_memory_chars is not None):
            raise SimulationConfigurationError('Explicit fixed B1 Context-Budgeted History contract required')
    if options.purpose == "research_evaluation":
        raise ResearchExecutionBlocked("Research evaluation remains blocked; dataset/artifact/gold readiness is unverified")
    if options.purpose == "development_smoke":
        if options.allow_llm is not False or not isinstance(options.client, OfflineClient):
            raise SimulationConfigurationError("Smoke execution requires allow_llm=False and OfflineClient")
    elif options.purpose == "development_llm":
        from app.locdi.ollama_adapter import LocalOllamaAdapter
        from app.locdi.llm_setup import validate_setup
        if options.allow_llm is not True or options.dry_run is not False or not isinstance(options.client, LocalOllamaAdapter):
            raise SimulationConfigurationError("Real development execution requires explicit opt-in and LocalOllamaAdapter")
        ready = validate_setup(options.llm_config)
        if (ready["static_readiness"] != "READY" or options.max_retries != 0
                or ready["prepared"]["fingerprint"] != options.client.prepared["fingerprint"]
                or options.registry_manifest != ready["prepared"]["registry"]
                or options.minute_limit != ready["prepared"]["config"]["minute_limit"]):
            raise SimulationConfigurationError("Real client, registry, limits and static setup must match")
    else:
        raise SimulationConfigurationError("Unsupported execution purpose")
    if options.accept_recorded_release_assumption is not True:
        raise SimulationConfigurationError("Explicit acceptance of dataset-declared release offsets is required for smoke testing")
    if type(options.minute_limit) is not int or options.minute_limit < 1:
        raise SimulationConfigurationError("minute_limit must be an explicit positive integer")
    if type(options.routing_seed) is not int or type(options.max_retries) is not int or not 0 <= options.max_retries <= 3:
        raise SimulationConfigurationError("routing_seed must be integer and max_retries must be 0..3")
    if options.max_memory_chars is not None and (type(options.max_memory_chars) is not int or options.max_memory_chars < 2):
        raise SimulationConfigurationError("max_memory_chars must be null or an integer >= 2")
    core = translate_runner_options(options.runner_options, memory_source=options.memory_source)
    if options.registry_manifest is None:
        raise SimulationConfigurationError("An explicit Agent Registry manifest is required")
    try:
        registry = AgentRegistry(deepcopy(options.registry_manifest), deepcopy(options.routing_utility))
    except (ValueError, TypeError, KeyError, AttributeError):
        raise SimulationConfigurationError("Registry or routing utility does not satisfy the existing Core contract") from None
    if type(options.allow_unfitted_adaptive_development) is not bool:
        raise SimulationConfigurationError("Development abstention flag must be boolean")
    if options.allow_unfitted_adaptive_development and (options.purpose != "development_llm" or not options.allow_llm):
        raise SimulationConfigurationError("Unfitted abstention is restricted to real development opt-in")
    if core["routing_mode"] != "er_only" and (core["routing_mode"] == "adaptive" or core["primary_router"] == "adaptive") and options.routing_utility is None and not options.allow_unfitted_adaptive_development:
        raise SimulationConfigurationError("Adaptive execution requires an explicit fitted routing utility")
    enabled = options.runner_options["voi_enabled"]
    if enabled and options.voi_model is None or not enabled and options.voi_model is not None:
        raise SimulationConfigurationError("VoI model presence must match voi_enabled; inactive VoI needs no artifact")
    try:
        if enabled:
            InformationPolicy(deepcopy(options.voi_model), core["action_selector"])
        if core["mode"] == "semantic":
            if options.embedding_artifact is None:
                raise SimulationConfigurationError("Semantic mode requires explicit embeddings")
            MemoryEmbeddings(deepcopy(options.embedding_artifact))
    except (ValueError, TypeError, KeyError, AttributeError):
        raise SimulationConfigurationError("VoI/embedding artifact contract is invalid or incomplete") from None
    if options.diagnosis_catalog is not None:
        labels = options.diagnosis_catalog
        if (not isinstance(labels, (list, tuple)) or not labels or
                any(not isinstance(x, str) or not x.strip() or len(x) > 240 for x in labels) or len(set(labels)) != len(labels)):
            raise SimulationConfigurationError("Shared diagnosis catalog must contain unique nonempty labels")
    if options.diagnosis_departments is not None:
        from app.utils.department_catalog import SUPPORTED_DEPARTMENTS
        if (options.diagnosis_catalog is None or not isinstance(options.diagnosis_departments, Mapping)
                or set(options.diagnosis_departments) != set(options.diagnosis_catalog)
                or any(x not in SUPPORTED_DEPARTMENTS for x in options.diagnosis_departments.values())):
            raise SimulationConfigurationError("Diagnosis departments must cover the shared catalog with canonical departments")
    core.update(minute_limit=options.minute_limit, registry_manifest=registry.snapshot(), routing_seed=options.routing_seed,
                max_retries=options.max_retries, max_memory_chars=options.max_memory_chars,
                routing_utility=deepcopy(options.routing_utility), voi_model=deepcopy(options.voi_model),
                embedding_artifact=deepcopy(options.embedding_artifact),
                diagnosis_catalog=list(options.diagnosis_catalog) if options.diagnosis_catalog is not None else None,
                diagnosis_departments=deepcopy(options.diagnosis_departments))
    return core


def _fixed_memory(adapted: Sequence[AdaptedCase], supplied) -> list:
    records, cutoffs = {}, {}
    for case in adapted:
        identity = case.runtime_case["longitudinal"]
        patient, start = identity["patient_id"], timestamp(identity["started_at"])
        cutoffs[patient] = min(start, cutoffs.get(patient, start))
        for visit in case.runtime_case["past_visits"]:
            for record in visit["records"]:
                mid = record["memory_id"]
                if mid in records and records[mid] != record:
                    raise SimulationConfigurationError("Conflicting copies of fixed past memory")
                records[mid] = deepcopy(record)
    for record in records.values():
        if timestamp(record["available_at"]) > cutoffs[record["patient_id"]]:
            raise SimulationConfigurationError("Batch fixed memory would include records after an earlier visit cutoff; use visit-scoped batches")
    if supplied is not None:
        if not isinstance(supplied, (list, tuple)) or any(not isinstance(r, dict) for r in supplied):
            raise SimulationConfigurationError("initial_memory must be a sequence of validated past-record objects")
        explicit = {r.get("memory_id"): r for r in supplied if isinstance(r.get("memory_id"), str)}
        if len(explicit) != len(supplied) or explicit != records:
            raise SimulationConfigurationError("Explicit initial memory must exactly match validated past records, with no extra/future content")
    result = list(records.values())
    try:
        PatientMemoryStore(result)
    except (ValueError, TypeError, KeyError):
        raise SimulationConfigurationError("Fixed memory violates the existing MemoryStore contract") from None
    return result


def run_case(case: LoCDICase, *, options: SimulationOptions,
             validation_report: Optional[DatasetValidationReport] = None) -> Dict[str, Any]:
    """Execute one development case; keep evaluation/audit stores with caller."""
    return run_cases([case], options=options, validation_report=validation_report)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@contextmanager
def _capture_execution(trace):
    """Passively observe existing decision/retrieval calls; never change policy.

    run_cases serializes these process-wide patches for BOTH real and offline
    development runs. This is not safe for unrelated concurrent Core callers.
    Calls/validators are forwarded unchanged, including the bounded repair.
    Only the actual call menu is observed; historical menus are not recreated.
    """
    original = HospitalLLMPolicy._decision
    original_call = policy_module.role_call
    contracts, role_observations, active = {}, {}, []

    def call(*args, **kwargs):
        audit, payload = kwargs["audit"], kwargs["payload"]
        before = len(audit)
        policy, runtime = active[-1] if active else (None, None)
        snapshot = None
        role_observation = None
        if runtime is not None and "allowed_actions" in payload:
            snapshot = capture_contract(payload,
                source_events=runtime.case_data.get("current_encounter_events"), minute=runtime.current_time_min)
            role_observation = {"effective_model_role": kwargs.get("role"),
                                "effective_department": payload.get("department"),
                                "model_binding": None, "binding_status": "NOT_RECORDED"}
            if policy.agent_registry is not None:
                try:
                    binding = policy.agent_registry.get(kwargs.get("role"), payload.get("department", ""))
                    if binding["model"] == kwargs.get("model"):
                        role_observation.update(model_binding=deepcopy(binding), binding_status="MATCHED")
                    else:
                        role_observation["binding_status"] = "MISMATCH"
                except (ValueError, TypeError, KeyError):
                    role_observation["binding_status"] = "MISMATCH"
        try:
            return original_call(*args, **kwargs)
        finally:
            for index in range(before, len(audit)):
                contracts[id(audit), index] = snapshot
                role_observations[id(audit), index] = role_observation

    def decision(policy, runtime, role, department=""):
        before = len(runtime.llm_audit)
        retriever = policy.memory_retriever

        def retrieve(rt, observed):
            context = retriever(rt, observed)
            identity = rt.case_data["longitudinal"]
            records = context.get("records", [])
            valid = all(r["patient_id"] == identity["patient_id"] and r["visit_id"] != identity["visit_id"]
                        and timestamp(r["visit_started_at"]) < timestamp(identity["started_at"])
                        and timestamp(r["available_at"]) <= timestamp(identity["started_at"]) for r in records)
            trace["retrievals"].append({"case_id": rt.case_id, "minute": rt.current_time_min,
                "mode": context.get("mode"), "memory_ids": [r["memory_id"] for r in records],
                "cutoff_compliant": valid})
            return context

        if retriever is not None:
            policy.memory_retriever = retrieve
        active.append((policy, runtime))
        result = None
        try:
            result = original(policy, runtime, role, department)
            return result
        finally:
            active.pop()
            policy.memory_retriever = retriever
            for index in range(before, len(runtime.llm_audit)):
                key = (runtime.case_id, index)
                if any((d["case_id"], d["audit_index"]) == key for d in trace["decisions"]):
                    continue  # Nested VoI retrieval may enter _decision again.
                trace["decisions"].append({"case_id": runtime.case_id, "minute": runtime.current_time_min,
                    "role": role, "audit_index": index,
                    "entry_role": role,
                    "role_observation": role_observations.get((id(runtime.llm_audit), index)),
                    "contract": contracts.get((id(runtime.llm_audit), index)),
                    "certainty": result.get("certainty") if isinstance(result, dict) and
                        runtime.llm_audit[index].get("decision") else None})

    with patch.object(HospitalLLMPolicy, "_decision", decision), patch.object(policy_module, "role_call", call):
        yield


def run_cases(cases: Sequence[LoCDICase], *, options: SimulationOptions,
              validation_report: Optional[DatasetValidationReport] = None) -> Dict[str, Any]:
    """Validate -> adapt -> readiness -> legacy runner -> separated result.

    Configuration/validation errors raise before execution. Core failure and
    incomplete reports retain their status. Unexpected execution exceptions
    produce a sanitized failed result, never synthetic success or partial state.
    No evaluation/gold/provenance store is passed to Core or returned here;
    caller retains the original wrapper separately for future evaluation.
    """
    if isinstance(options, SimulationOptions) and options.purpose == "development_llm" and options.dry_run is True:
        if not isinstance(cases, (list, tuple)) or len(cases) != 1:
            raise SimulationConfigurationError("Real development interface requires one case")
        return inspect_development_llm(cases[0], config=options.llm_config)
    core = _configuration(options)
    if not isinstance(cases, (list, tuple)) or not cases:
        raise SimulationConfigurationError("Provide a nonempty sequence of LoCDI wrappers")
    adapted = adapt_cases(cases)  # Recompute validation; never trust a stale report to bypass it.
    if options.purpose == "development_llm" and len(adapted) != 1:
        raise SimulationConfigurationError("Real development execution is single-case only")
    if validation_report is not None:
        if not isinstance(validation_report, DatasetValidationReport):
            raise SimulationConfigurationError("validation_report must be a DatasetValidationReport")
        by_id = {row.case_id: row for row in validation_report.cases}
        if any(c.case_id not in by_id or by_id[c.case_id].has_failures for c in adapted):
            raise SimulationConfigurationError("Dataset validation report lacks a non-failing entry for an input case")
    for case in adapted:
        if any(issue.code not in _ACCEPTED_SMOKE_ISSUES for result in case.validation_results for issue in result.issues):
            raise SimulationConfigurationError("Validation has issues outside the explicitly supported smoke assumptions")
    initial = _fixed_memory(adapted, options.initial_memory)
    execution_cases = [deepcopy(case.runtime_case) for case in adapted]
    source_view_audit = None
    if options.sequential_source_view is not None:
        from app.locdi.synthea_sequential_proxy import SequentialSyntheticSourceView
        if not isinstance(options.sequential_source_view, SequentialSyntheticSourceView):
            raise SimulationConfigurationError("Explicit validated sequential source view required")
        execution_cases, source_view_audit = [], []
        for case in adapted:
            runtime_case, added = options.sequential_source_view.apply(
                case, features=options.runner_options, purpose=options.purpose)
            execution_cases.append(runtime_case)
            source_view_audit.append({"case_id":case.case_id,"added_tests":added})
    core["initial_memory"] = initial
    mock = options.purpose == "development_smoke"
    core["client"] = _OfflineClientProxy(options.client) if mock else options.client
    configuration = {**dict(options.runner_options), "memory_source": options.memory_source,
                     "minute_limit": options.minute_limit, "routing_seed": options.routing_seed,
                     "max_retries": options.max_retries, "max_memory_chars": options.max_memory_chars,
                     "accept_recorded_release_assumption": options.accept_recorded_release_assumption,
                     "dependency_fingerprints": {k: _digest(core[k]) if core[k] is not None else None for k in
                         ("registry_manifest", "routing_utility", "voi_model", "embedding_artifact",
                          "diagnosis_catalog", "diagnosis_departments")}}
    if not mock:
        configuration["llm_settings_fingerprint"] = options.client.prepared["fingerprint"]
    if source_view_audit is not None:
        configuration["sequential_source_view"] = options.sequential_source_view.metadata()
    if options.history_policy is not None:
        configuration['context_budgeted_history'] = options.history_policy.metadata()
    if options.allow_unfitted_adaptive_development:
        configuration['development_unfitted_adaptive'] = {
            'experiment_kind':'DEVELOPMENT_EXPERIMENT','source_label_proxy':True,
            'independent_clinical_review':False,'clinical_accuracy':'NOT_EVALUATED',
            'behavior':'unchanged Core adaptive abstention with utility=None'}
    trace = {"decisions": [], "retrievals": []}
    result = {
        "purpose": options.purpose, "development_only": True, "mock_execution": mock,
        "simulation_ready": False, "research_evaluation_eligible": False,
        "clinical_metrics": "NOT_EVALUATED", "case_count": len(adapted),
        "assumptions": ["Use dataset-declared available_minute offsets only for development replay",
                        "Actual doctor release times and source-file integrity remain unverified"],
        "option_translation": {"memory_mode": options.runner_options["memory_mode"], "core_mode": core["mode"],
                               "requested_top_k": options.runner_options["top_k"], "core_top_k": core["top_k"],
                               "top_k_unused": core["mode"] in {"none", "full_history"},
                               "memory_source": options.memory_source, "initial_memory_source": "validated_dataset_past_visits",
                               "initial_memory_count": len(initial), "routing_mode": core["routing_mode"],
                               "primary_router": core["primary_router"], "voi_enabled": options.runner_options["voi_enabled"]},
        "validation": [{"case_id": c.case_id, "checks": [asdict(r) for r in c.validation_results]} for c in adapted],
        "dataset_report_supplied": validation_report is not None,
        "status": "failed", "execution": None, "errors": [],
        "execution_metadata": {"schema_version": "locdi.execution.v1", "configuration": configuration,
            "configuration_fingerprint": _digest(configuration),
            "started_at": datetime.now(timezone.utc).isoformat(), "ended_at": None},
        "telemetry": trace,
    }
    if source_view_audit is not None:
        result["synthetic_source_view_audit"] = source_view_audit
    try:
        with _SOCKET_LOCK, (_offline_network_guard() if mock else nullcontext()), \
                (options.history_policy.installed() if options.history_policy is not None else nullcontext()), _capture_execution(trace):
            execution = run_with_recovery(execution_cases, **core)
        if not isinstance(execution, dict) or execution.get("status") not in {"completed", "incomplete", "failed"}:
            raise ValueError("Unexpected legacy report contract")
        result.update(status=execution["status"], execution=execution)
        if execution["status"] == "failed":
            result["errors"].append({"stage": "legacy_runner", "error_type": "reported_execution_failure"})
    except Exception as exc:
        result["errors"].append({"stage": "legacy_runner", "error_type": type(exc).__name__,
                                 "message": "Execution raised; no complete Core report or invented partial state is available"})
    finally:
        result["execution_metadata"]["ended_at"] = datetime.now(timezone.utc).isoformat()
        if not mock:
            result["llm_execution"] = options.client.public_metadata()
    return result


def inspect_development_llm(case, *, config=None):
    """Default dry-run: static setup and real case validation, no client/server."""
    from app.locdi.llm_setup import validate_setup, load_setup
    report = validate_setup(load_setup() if config is None else config)
    report.pop("prepared", None)
    adapted = adapt_cases([case])[0]
    report.update(status="dry_run", purpose="development_llm", development_only=True,
                  mock_execution=False, real_model_execution=False, request_count=0,
                  clinical_metrics="NOT_EVALUATED", simulation_ready=False, research_evaluation_eligible=False,
                  dataset_validation={r.validator: r.status for r in adapted.validation_results})
    return report


def run_development_llm(case, *, config=None, execute=False, allow_llm=False,
                        runner_options=None, accept_recorded_release_assumption=False, destination=None):
    """Single-case opt-in entry. This phase uses only the default dry-run.

    The future explicit execute path checks server digest before each model
    request. Core/clinical readiness stays false even if the workflow completes.
    """
    from app.locdi.llm_setup import load_setup
    from app.locdi.ollama_adapter import LocalOllamaAdapter
    config = load_setup() if config is None else config
    if execute is False:
        return inspect_development_llm(case, config=config)
    if execute is not True or allow_llm is not True:
        raise SimulationConfigurationError("Both execute=True and allow_llm=True are required")
    client = LocalOllamaAdapter(config, authorized=True)
    try:
        options = SimulationOptions(purpose="development_llm", allow_llm=True, dry_run=False,
            client=client, llm_config=config, minute_limit=config["minute_limit"],
            registry_manifest=client.prepared["registry"], runner_options=runner_options,
            accept_recorded_release_assumption=accept_recorded_release_assumption)
        result = run_case(case, options=options)
        if destination is not None:
            from app.locdi.episode_exporter import export_episode
            export_episode(result, destination, context=adapt_cases([case])[0])
        return result
    finally:
        client.close()
