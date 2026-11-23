"""Versioned structural validation. PASS never means clinical review."""
from dataclasses import dataclass
import math
from typing import Any, List, Literal, Optional, Tuple
from app.locdi.dataset_loader import LoCDICase

Status = Literal["PASS", "FAIL", "WARNING", "NOT_REVIEWED", "NOT_IMPLEMENTED", "BLOCKED", "SKIPPED"]
SCHEMA_VERSION = "longitudinal_single_file_v34.1"
ROOT_FIELDS = {"schema_version", "case_id", "patient_id", "agent_visible", "environment",
               "evaluation", "task_annotations", "provenance"}
VISIBLE_FIELDS = {"case_id", "current_initial_state", "index_encounter_id",
                  "index_encounter_start", "longitudinal", "past_visits"}
EVENT_FIELDS = {"event_id", "action", "target", "available_minute", "evidence"}
MEMORY_FIELDS = {"memory_id", "patient_id", "visit_id", "visit_started_at", "observed_at",
                 "available_at", "source_id", "kind", "text", "supersedes"}
GOLD_FIELDS = {"acceptable_next_actions", "accepted_departments", "accepted_diagnoses",
               "relevant_memory_ids", "stale_memory_ids"}
READINESS_FIELDS = {"diagnosis_accuracy", "memory_retrieval", "routing_accuracy", "voi_regret"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    severity: Status = "FAIL"


@dataclass(frozen=True)
class ValidationResult:
    """The first three fields retain the Phase 0.3 positional contract.

    case_id is internal diagnostic identity; public reports use ordinal aliases.
    Messages contain invariant descriptions, never raw patient values.
    checks/not_reviewed separate completed scope from unverified meaning.
    """
    validator: str
    status: Status
    issues: Tuple[ValidationIssue, ...] = ()
    case_id: Optional[str] = None
    checks: Tuple[str, ...] = ()
    not_reviewed: Tuple[str, ...] = ()
    verified_source_hashes: int = 0
    source_integrity_verified: Optional[bool] = None


class _Check:
    """Shared diagnostic accumulator. Never interpolate original values."""
    def __init__(self, name: str, case: Any):
        self.name = name
        self.case_id = case.get("case_id") if isinstance(case, dict) and isinstance(case.get("case_id"), str) else None
        self.issues: List[ValidationIssue] = []

    def add(self, code: str, path: str, message: str, severity: Status = "FAIL") -> None:
        self.issues.append(ValidationIssue(code, path, message, severity))

    def result(self, checks=(), not_reviewed=(), **kwargs) -> ValidationResult:
        states = {i.severity for i in self.issues}
        status = next((s for s in ("FAIL", "BLOCKED", "NOT_IMPLEMENTED", "WARNING",
                                    "NOT_REVIEWED", "SKIPPED") if s in states), "PASS")
        return ValidationResult(self.name, status, tuple(self.issues), self.case_id,
                                tuple(checks), tuple(not_reviewed), **kwargs)

    def mapping(self, value: Any, path: str, required=(), allowed=None) -> dict:
        if not isinstance(value, dict):
            self.add("TYPE", path, "Expected an object")
            return {}
        if not set(required).issubset(value):
            self.add("REQUIRED_FIELDS", path, "Required fields are missing")
        if allowed is not None and set(value) - set(allowed):
            self.add("UNSUPPORTED_FIELDS", path, "Fields outside the supported contract")
        return value

    def sequence(self, value: Any, path: str) -> list:
        if not isinstance(value, list):
            self.add("TYPE", path, "Expected an array")
            return []
        return value

    def text(self, value: Any, path: str) -> bool:
        valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            self.add("TYPE", path, "Expected a nonempty string")
        return valid


def validate_schema(case: LoCDICase) -> ValidationResult:
    """Validate v34.1 identities, containers and legacy encounter field shapes.

    Timestamp meaning, clinical correctness, evidence/source integrity and
    release-time semantics are separate checks, not certified by this result.
    """
    c = _Check("schema", case)
    root = c.mapping(case, "$", ROOT_FIELDS, ROOT_FIELDS)
    if root.get("schema_version") != SCHEMA_VERSION:
        c.add("SCHEMA_VERSION", "schema_version", "Unsupported schema version")
    for key in ("case_id", "patient_id"):
        c.text(root.get(key), key)
    av = c.mapping(root.get("agent_visible"), "agent_visible", VISIBLE_FIELDS, VISIBLE_FIELDS)
    forbidden = {"evaluation", "task_annotations", "gold", "gold_label", "ground_truth", "provenance", "source_sha256",
                 "diagnosis_labels", "accepted_diagnoses", "accepted_departments", "environment", "source_event_store",
                 "current_encounter_events", "hidden_test_results", "data_replay"}
    def check_visible_keys(value, path):
        if isinstance(value, dict):
            if set(value) & forbidden:
                c.add("PRIVATE_FIELD_IN_VISIBLE", path, "Evaluation/environment/audit fields are forbidden in the visible store")
            for key, child in value.items():
                # Do not echo unknown/untrusted field names into diagnostics.
                check_visible_keys(child, path + ".field" if key in forbidden else path)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                check_visible_keys(child, path + "[{}]".format(i))
    check_visible_keys(av, "agent_visible")
    for key in ("case_id", "index_encounter_id", "index_encounter_start"):
        c.text(av.get(key), "agent_visible." + key)
    ident = c.mapping(av.get("longitudinal"), "agent_visible.longitudinal",
                      {"patient_id", "visit_id", "started_at"}, {"patient_id", "visit_id", "started_at"})
    for key in ("patient_id", "visit_id", "started_at"):
        c.text(ident.get(key), "agent_visible.longitudinal." + key)
    pairs = ((av.get("case_id"), root.get("case_id"), "agent_visible.case_id"),
             (ident.get("patient_id"), root.get("patient_id"), "agent_visible.longitudinal.patient_id"),
             (ident.get("visit_id"), av.get("index_encounter_id"), "agent_visible.longitudinal.visit_id"),
             (ident.get("started_at"), av.get("index_encounter_start"), "agent_visible.longitudinal.started_at"))
    for left, right, path in pairs:
        if left != right:
            c.add("IDENTITY_MISMATCH", path, "Wrapper and encounter identities disagree")
    initial = c.mapping(av.get("current_initial_state"), "agent_visible.current_initial_state",
                        {"doctor_visible_info", "patient_visible_info"}, {"doctor_visible_info", "patient_visible_info"})
    doctor_keys = {"chief_complaint", "demographics", "arrival_info", "initial_vitals"}
    patient_keys = {"chief_complaint", "history", "associated_symptoms", "past_medical_history",
                    "medications", "allergies", "social_history", "age", "sex"}
    doc = c.mapping(initial.get("doctor_visible_info"), "agent_visible.current_initial_state.doctor_visible_info", allowed=doctor_keys)
    pat = c.mapping(initial.get("patient_visible_info"), "agent_visible.current_initial_state.patient_visible_info", allowed=patient_keys)
    for key in ("demographics", "arrival_info", "initial_vitals"):
        if key in doc:
            c.mapping(doc[key], "agent_visible.current_initial_state.doctor_visible_info." + key)
    if isinstance(doc.get("demographics"), dict):
        dp = "agent_visible.current_initial_state.doctor_visible_info.demographics"
        demo = c.mapping(doc["demographics"], dp, {"age_at_index", "sex"}, {"age_at_index", "sex"})
        if type(demo.get("age_at_index")) is not int or demo["age_at_index"] < 0:
            c.add("TYPE", dp + ".age_at_index", "Expected a nonnegative integer age")
        c.text(demo.get("sex"), dp + ".sex")
    if isinstance(doc.get("arrival_info"), dict):
        ap = "agent_visible.current_initial_state.doctor_visible_info.arrival_info"
        arrival = c.mapping(doc["arrival_info"], ap,
                            {"clinical_information_need", "encounter_class", "reviewed_generic_purposes", "safe_initial_observations"},
                            {"clinical_information_need", "encounter_class", "reviewed_generic_purposes", "safe_initial_observations"})
        c.text(arrival.get("encounter_class"), ap + ".encounter_class")
        if arrival.get("clinical_information_need") is not None and not isinstance(arrival["clinical_information_need"], str):
            c.add("TYPE", ap + ".clinical_information_need", "Expected text or null")
        for key in ("reviewed_generic_purposes", "safe_initial_observations"):
            values = c.sequence(arrival.get(key), ap + "." + key)
            if any(not isinstance(x, str) for x in values):
                c.add("UNSUPPORTED_INITIAL_OBSERVATION", ap + "." + key, "Only explicit text entries are supported; no object is flattened into model input")
    for key, value in doc.get("initial_vitals", {}).items() if isinstance(doc.get("initial_vitals"), dict) else ():
        if type(value) not in (int, float) or not math.isfinite(value):
            c.add("VITAL_TYPE", "agent_visible.current_initial_state.doctor_visible_info.initial_vitals", "Vitals require finite numbers")
    for key in ("chief_complaint",):
        if key in doc:
            c.text(doc[key], "agent_visible.current_initial_state.doctor_visible_info." + key)
    if "age" in pat and (type(pat["age"]) is not int or pat["age"] < 0):
        c.add("TYPE", "agent_visible.current_initial_state.patient_visible_info.age", "Age must be a nonnegative integer")
    for key in patient_keys - {"age"}:
        if key in pat:
            value = pat[key]
            if not (isinstance(value, str) or isinstance(value, list) and all(isinstance(x, str) for x in value)):
                c.add("TYPE", "agent_visible.current_initial_state.patient_visible_info." + key, "Expected text or an array of text")
    if "sex" in pat:
        c.text(pat["sex"], "agent_visible.current_initial_state.patient_visible_info.sex")
    if "associated_symptoms" in pat and (not isinstance(pat["associated_symptoms"], list) or any(not isinstance(x, str) for x in pat["associated_symptoms"])):
        c.add("TYPE", "agent_visible.current_initial_state.patient_visible_info.associated_symptoms", "Expected an array of symptom text")
    visits = c.sequence(av.get("past_visits"), "agent_visible.past_visits")
    visit_ids, memory_ids = set(), set()
    for i, visit in enumerate(visits):
        path = "agent_visible.past_visits[{}]".format(i)
        v = c.mapping(visit, path, {"visit_id", "started_at", "records"}, {"visit_id", "started_at", "ended_at", "records"})
        for key in ("visit_id", "started_at"):
            c.text(v.get(key), path + "." + key)
        vid = v.get("visit_id")
        if isinstance(vid, str):
            if vid in visit_ids or vid == av.get("index_encounter_id"):
                c.add("DUPLICATE_VISIT", path + ".visit_id", "Visit ID is duplicated or equals the current visit")
            visit_ids.add(vid)
        records = c.sequence(v.get("records"), path + ".records")
        for j, record in enumerate(records):
            rp = path + ".records[{}]".format(j)
            m = c.mapping(record, rp, MEMORY_FIELDS, MEMORY_FIELDS)
            for key in MEMORY_FIELDS - {"supersedes"}:
                c.text(m.get(key), rp + "." + key)
            mid = m.get("memory_id")
            if isinstance(mid, str):
                if mid in memory_ids:
                    c.add("DUPLICATE_MEMORY", rp + ".memory_id", "Memory ID is duplicated")
                memory_ids.add(mid)
            if (m.get("patient_id") != root.get("patient_id") or m.get("visit_id") != vid or
                    m.get("visit_started_at") != v.get("started_at")):
                c.add("MEMORY_IDENTITY", rp, "Memory patient/visit identity differs from its container")
            if not isinstance(m.get("kind"), str) or m["kind"] not in {"history", "objective_finding", "prior_assessment", "intervention", "update"}:
                c.add("MEMORY_KIND", rp + ".kind", "Unsupported legacy memory kind")
            supersedes = c.sequence(m.get("supersedes"), rp + ".supersedes")
            if any(not isinstance(x, str) or not x.strip() for x in supersedes):
                c.add("TYPE", rp + ".supersedes", "Expected memory ID strings")
    env = c.mapping(root.get("environment"), "environment", {"current_encounter_events", "source_event_store", "data_replay"},
                    {"current_encounter_events", "source_event_store", "data_replay"})
    events = c.sequence(env.get("current_encounter_events"), "environment.current_encounter_events")
    event_ids = set()
    for i, event in enumerate(events):
        ep = "environment.current_encounter_events[{}]".format(i)
        e = c.mapping(event, ep, EVENT_FIELDS, EVENT_FIELDS)
        for key in EVENT_FIELDS - {"available_minute"}:
            c.text(e.get(key), ep + "." + key)
        if type(e.get("available_minute")) is not int or e["available_minute"] < 0:
            c.add("AVAILABLE_MINUTE", ep + ".available_minute", "Expected a nonnegative integer minute, excluding bool")
        eid = e.get("event_id")
        if isinstance(eid, str):
            if eid in event_ids:
                c.add("DUPLICATE_EVENT", ep + ".event_id", "Projected event ID is duplicated")
            event_ids.add(eid)
    sources = c.sequence(env.get("source_event_store"), "environment.source_event_store")
    source_ids = set()
    required_source = {"event_id", "case_id", "patient_id", "encounter_id", "timestamp", "timing_precision",
                       "timestamp_lower_epoch_ms", "timestamp_upper_epoch_ms", "evidence", "target", "source_metadata",
                       "availability_status", "reveal_status", "relative_to_decision", "event_type", "action_type_candidate"}
    for i, source in enumerate(sources):
        sp = "environment.source_event_store[{}]".format(i)
        s = c.mapping(source, sp, required_source)
        for key in ("event_id", "case_id", "patient_id", "encounter_id", "timestamp", "timing_precision",
                    "availability_status", "reveal_status", "relative_to_decision", "event_type"):
            c.text(s.get(key), sp + "." + key)
        for key in ("timestamp_lower_epoch_ms", "timestamp_upper_epoch_ms"):
            if type(s.get(key)) is not int:
                c.add("TYPE", sp + "." + key, "Expected integer epoch milliseconds")
        for key in ("target", "evidence", "source_metadata"):
            c.mapping(s.get(key), sp + "." + key)
        if s.get("case_id") != root.get("case_id") or s.get("patient_id") != root.get("patient_id") or s.get("encounter_id") != av.get("index_encounter_id"):
            c.add("SOURCE_IDENTITY", sp, "Source identity differs from the current case/encounter")
        eid = s.get("event_id")
        if isinstance(eid, str):
            if eid in source_ids:
                c.add("DUPLICATE_SOURCE_EVENT", sp + ".event_id", "Source event ID is duplicated")
            source_ids.add(eid)
    replay = c.mapping(env.get("data_replay"), "environment.data_replay",
                       {"policy", "original_decision_time", "observed_release_time", "requires_time_series_loader_for_latest_value"})
    for key in ("policy", "original_decision_time"):
        c.text(replay.get(key), "environment.data_replay." + key)
    for key in ("observed_release_time", "requires_time_series_loader_for_latest_value"):
        if type(replay.get(key)) is not bool:
            c.add("TYPE", "environment.data_replay." + key, "Expected a boolean")
    evaluation = c.mapping(root.get("evaluation"), "evaluation",
                           {"diagnosis_labels", "accepted_departments_source_policy", "observed_index_diagnoses"})
    for key in ("diagnosis_labels", "accepted_departments_source_policy"):
        values = c.sequence(evaluation.get(key), "evaluation." + key)
        if any(not isinstance(x, str) or not x.strip() for x in values):
            c.add("TYPE", "evaluation." + key, "Expected nonempty text entries")
    for i, label in enumerate(c.sequence(evaluation.get("observed_index_diagnoses"), "evaluation.observed_index_diagnoses")):
        d = c.mapping(label, "evaluation.observed_index_diagnoses[{}]".format(i), {"code", "description", "system"})
        for key in ("code", "description", "system"):
            c.text(d.get(key), "evaluation.observed_index_diagnoses[{}].{}".format(i, key))
    annotations = c.mapping(root.get("task_annotations"), "task_annotations", {"annotation", "gold", "metric_readiness"})
    annotation = c.mapping(annotations.get("annotation"), "task_annotations.annotation", {"review_status", "source"})
    for key in ("review_status", "source"):
        c.text(annotation.get(key), "task_annotations.annotation." + key)
    gold = c.mapping(annotations.get("gold"), "task_annotations.gold", GOLD_FIELDS, GOLD_FIELDS)
    for key in GOLD_FIELDS:
        if gold.get(key) is not None:
            values = c.sequence(gold[key], "task_annotations.gold." + key)
            expected = (str, dict) if key == "acceptable_next_actions" else (str,)
            if any(not isinstance(x, expected) for x in values):
                c.add("TYPE", "task_annotations.gold." + key, "Unsupported annotation entry type")
    readiness = c.mapping(annotations.get("metric_readiness"), "task_annotations.metric_readiness", READINESS_FIELDS, READINESS_FIELDS)
    for key in READINESS_FIELDS:
        if type(readiness.get(key)) is not bool:
            c.add("TYPE", "task_annotations.metric_readiness." + key, "Expected a boolean readiness flag")
    for metric, fields in {"diagnosis_accuracy": ("accepted_diagnoses",), "routing_accuracy": ("accepted_departments",),
                           "memory_retrieval": ("relevant_memory_ids", "stale_memory_ids"), "voi_regret": ("acceptable_next_actions",)}.items():
        if readiness.get(metric) and any(gold.get(key) is None for key in fields):
            c.add("READINESS_WITHOUT_GOLD", "task_annotations.metric_readiness." + metric, "Metric marked ready without its required annotations")
    for key in ("relevant_memory_ids", "stale_memory_ids"):
        if isinstance(gold.get(key), list) and any(isinstance(x, str) and x not in memory_ids for x in gold[key]):
            c.add("UNKNOWN_MEMORY_ANNOTATION", "task_annotations.gold." + key, "Annotated memory ID does not exist in past records")
    c.mapping(root.get("provenance"), "provenance")
    return c.result(checks=("v34.1_structure", "identity_links", "legacy_field_shapes", "annotation_structure"),
                    not_reviewed=("clinical_correctness", "clinical_gold_review", "timestamp_semantics", "source_integrity"))


def _schema_block(case: Any, name: str) -> Optional[ValidationResult]:
    result = validate_schema(case)
    if result.status == "FAIL":
        return ValidationResult(name, "BLOCKED", (ValidationIssue("SCHEMA_BLOCKED", "$", "Schema errors prevent this check", "BLOCKED"),), result.case_id)
    return None
