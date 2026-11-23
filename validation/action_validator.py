"""Validate existing Ask/Test/Consult evidence without normalizing raw text."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath, PureWindowsPath
from app.locdi.dataset_loader import LoCDICase
from app.utils.department_catalog import SUPPORTED_DEPARTMENTS
from validation.schema_validator import ValidationResult, _Check, _schema_block

CONSULT_PROPOSAL_VERSION = "locdi.consult_source.proposal.v1"


def _consult_time(c, value, path):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError()
        return parsed.astimezone(timezone.utc)
    except (AttributeError, ValueError, TypeError, OverflowError):
        c.add("CONSULT_TIMESTAMP", path, "Expected a timezone-aware timestamp")
        return None


def _check_consult(c, case, event, payload, source, path):
    """Check existing links plus an explicitly PROPOSED envelope, never clearance.

    No inspected real Consult schema exists. Even a structurally consistent
    proposal remains NOT_REVIEWED and cannot pass the execution Adapter.
    """
    c.add("CONSULT_SOURCE_UNREVIEWED", path,
          "Consult clinical meaning and original source contract remain unreviewed", "NOT_REVIEWED")
    if event["target"] not in SUPPORTED_DEPARTMENTS:
        c.add("CONSULT_TARGET", path + ".target", "Consult requires a canonical department")
    if source is None:
        c.add("CONSULT_SOURCE_MISSING", path, "Consult needs an exact source event link")
        return
    identity = case["agent_visible"]
    if (source.get("case_id") != case["case_id"] or source.get("patient_id") != case["patient_id"]
            or source.get("encounter_id") != identity["index_encounter_id"]):
        c.add("CONSULT_IDENTITY", path, "Consult source patient/encounter identity does not match")
    if source.get("action_type_candidate") != "Consult":
        c.add("CONSULT_SOURCE_TYPE", path, "Linked source is not declared as Consult")
    minute = event.get("available_minute")
    if type(minute) is not int or minute < 0:
        c.add("CONSULT_AVAILABLE_MINUTE", path, "Expected nonnegative integer minute, excluding bool")
        return
    start = _consult_time(c, identity["index_encounter_start"], path)
    try:
        release = start + timedelta(minutes=minute) if start is not None else None
    except OverflowError:
        c.add("CONSULT_AVAILABLE_MINUTE", path, "Minute offset exceeds the supported clock")
        release = None
    low, high = source["timestamp_lower_epoch_ms"], source["timestamp_upper_epoch_ms"]
    if low > high:
        c.add("CONSULT_SOURCE_INTERVAL", path, "Source precision interval is reversed")
    if release is not None:
        if low > release.timestamp() * 1000:
            c.add("CONSULT_SOURCE_FUTURE", path, "Source was recorded after projected release")
        elif high > release.timestamp() * 1000:
            c.add("CONSULT_RELEASE_UNCERTAIN", path, "Source recording interval overlaps release", "NOT_REVIEWED")
    meta = source["source_metadata"]
    record = meta.get("source_record")
    source_file, source_row = meta.get("source_file"), meta.get("source_row")
    unsafe = not isinstance(source_file, str) or not source_file.strip()
    if not unsafe:
        unsafe = (PureWindowsPath(source_file).is_absolute() or bool(PureWindowsPath(source_file).drive)
                  or PurePosixPath(source_file.replace("\\", "/")).is_absolute()
                  or ".." in source_file.replace("\\", "/").split("/"))
    if unsafe or type(source_row) is not int or source_row < 1 or not isinstance(record, dict) or not record:
        c.add("CONSULT_SOURCE_RECORD_LINK", path, "Expected a safe relative source reference, positive row and original record")
    if isinstance(record, dict):
        for key, expected in (("PATIENT", case["patient_id"]), ("ENCOUNTER", identity["index_encounter_id"])):
            if key not in record:
                c.add("CONSULT_ROW_IDENTITY_UNREVIEWED", path, "Original row identity fields are not established", "NOT_REVIEWED")
            elif record[key] != expected:
                c.add("CONSULT_RECORD_IDENTITY", path, "Original record patient/encounter identity conflicts")
    if payload.get("schema_version") != CONSULT_PROPOSAL_VERSION:
        c.add("CONSULT_SCHEMA_UNREVIEWED", path, "No reviewed Consult envelope; proposal checks were not applicable", "NOT_REVIEWED")
        return
    required = {"schema_version", "contract_status", "purpose", "recorded_at", "timing_precision",
                "source_record_link", "recorded_evidence", "available_at", "release_time_basis", "clinical_review_status"}
    if set(payload) != required:
        c.add("CONSULT_EVIDENCE_FIELDS", path, "Unexpected or missing proposal evidence fields")
    if payload.get("contract_status") != "PROPOSED" or payload.get("purpose") not in ("TEST_ONLY", "SOURCE_REVIEW_PENDING"):
        c.add("CONSULT_PROPOSAL_STATUS", path, "Proposal cannot claim a reviewed production contract")
    if payload.get("clinical_review_status") != "NOT_REVIEWED":
        c.add("CONSULT_REVIEW_CLAIM", path, "Clinical review cannot be certified by a proposal payload")
    if source["target"].get("department") != event["target"]:
        c.add("CONSULT_TARGET_SOURCE", path, "Proposal source department differs from projected target")
    if payload.get("source_record_link") != {"source_file": source_file, "source_row": source_row}:
        c.add("CONSULT_RECORD_LINK_MISMATCH", path, "Proposal row link differs from source metadata")
    evidence = payload.get("recorded_evidence")
    if not isinstance(evidence, dict) or not evidence or evidence != source["evidence"]:
        c.add("CONSULT_EVIDENCE_SOURCE", path, "Serialized Consult evidence differs from linked source evidence")
    try:
        json.dumps(payload, allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        c.add("CONSULT_NONFINITE_EVIDENCE", path, "Consult evidence must contain finite JSON values")
    if payload.get("recorded_at") != source["timestamp"] or payload.get("timing_precision") != source["timing_precision"]:
        c.add("CONSULT_RECORD_TIME_MISMATCH", path, "Proposal source timestamp/precision differs")
    if source["timing_precision"] not in {"second", "millisecond"}:
        c.add("CONSULT_PRECISION_UNREVIEWED", path, "Non-instant source precision needs review", "NOT_REVIEWED")
    else:
        recorded = _consult_time(c, source["timestamp"], path)
        if recorded is not None and not low <= recorded.timestamp() * 1000 <= high:
            c.add("CONSULT_SOURCE_TIME_BOUNDS", path, "Source timestamp lies outside precision interval")
    available = _consult_time(c, payload.get("available_at"), path)
    if available is not None and (available.timestamp() * 1000 < high or release is not None and available > release):
        c.add("CONSULT_AVAILABILITY_TIME", path, "Declared availability is before recording or after projected release")
    basis = payload.get("release_time_basis")
    if basis not in ("OBSERVED_DOCUMENTED", "DATASET_DECLARED", "UNKNOWN"):
        c.add("CONSULT_RELEASE_BASIS", path, "Unsupported release confidence label")
    # A self-declared label is not proof of actual doctor release time.
    c.add("CONSULT_RELEASE_UNVERIFIED", path, "Actual release evidence and source-row reconstruction require review", "NOT_REVIEWED")


def validate_consult_sources(case: LoCDICase) -> ValidationResult:
    """Standalone structural inspection; NOT_REVIEWED is never execution PASS."""
    blocked = _schema_block(case, "consult_sources")
    if blocked:
        return blocked
    c = _Check("consult_sources", case)
    if not any(e["action"] == "Consult" for e in case["environment"]["current_encounter_events"]):
        c.add("CONSULT_NOT_PRESENT", "environment.current_encounter_events",
              "No Consult sample exists to inspect", "NOT_REVIEWED")
    sources = {s["event_id"]: s for s in case["environment"]["source_event_store"]}
    for i, event in enumerate(case["environment"]["current_encounter_events"]):
        if event["action"] != "Consult":
            continue
        path = "environment.current_encounter_events[{}]".format(i)
        try:
            payload = json.loads(event["evidence"])
        except (ValueError, TypeError):
            payload = None
        if not isinstance(payload, dict):
            c.add("CONSULT_EVIDENCE_JSON", path, "Consult evidence must serialize an object")
            continue
        _check_consult(c, case, event, payload, sources.get(event["event_id"]), path)
    return c.result(checks=("consult_structural_links", "consult_proposal_fields", "consult_record_release_bounds"),
                    not_reviewed=("consult_clinical_meaning", "original_consult_schema", "actual_release_proof"))


def validate_actions(case: LoCDICase) -> ValidationResult:
    """Check legacy targets and linkage; JSON evidence remains a raw string.

    v34.1 Test links to current source_event_store; Ask is a recorded patient
    report linked to provenance symptom_sources, not a Test source event ID.
    No new Action registry, interpretation of clinical value, or result is made.
    """
    blocked = _schema_block(case, "actions")
    if blocked:
        return blocked
    c = _Check("actions", case)
    env = case["environment"]
    sources = {s["event_id"]: s for s in env["source_event_store"]}
    audit = case["provenance"].get("source_case_audit", {})
    symptoms = audit.get("symptom_sources", []) if isinstance(audit, dict) else []
    seen_ids, seen_targets = set(), set()
    for i, event in enumerate(env["current_encounter_events"]):
        ep = "environment.current_encounter_events[{}]".format(i)
        kind, target, eid = event["action"], event["target"], event["event_id"]
        if eid in seen_ids or (kind, target) in seen_targets:
            c.add("DUPLICATE_ACTION", ep, "Duplicate event ID or action/target pair")
        seen_ids.add(eid)
        seen_targets.add((kind, target))
        if kind not in {"Ask", "Test", "Consult"}:
            c.add("ACTION_TYPE", ep + ".action", "Unsupported legacy encounter action")
            continue
        if kind == "Ask" and not target.endswith("?"):
            c.add("ASK_TARGET", ep + ".target", "Ask target must be an explicit question ending in a question mark")
        if kind == "Consult" and target not in SUPPORTED_DEPARTMENTS:
            c.add("CONSULT_TARGET", ep + ".target", "Consult target must be a canonical department")
        if type(event["available_minute"]) is not int or event["available_minute"] < 0:
            c.add("AVAILABLE_MINUTE", ep + ".available_minute", "Expected a nonnegative integer minute")
        try:
            payload = json.loads(event["evidence"])
        except (ValueError, TypeError):
            c.add("EVIDENCE_JSON", ep + ".evidence", "v34.1 evidence must contain a valid serialized JSON object")
            continue
        if not isinstance(payload, dict):
            c.add("EVIDENCE_SHAPE", ep + ".evidence", "Expected a serialized evidence object")
            continue
        if kind == "Test":
            if set(payload) != {"records", "replay_bin"} or not isinstance(payload.get("replay_bin"), str):
                c.add("TEST_PAYLOAD_FIELDS", ep + ".evidence", "Unexpected or missing serialized Test payload fields")
            records = payload.get("records")
            source = sources.get(eid)
            if source is None:
                c.add("TEST_SOURCE_MISSING", ep + ".event_id", "Projected Test does not reference a current source event")
                continue
            if not isinstance(records, list) or not records:
                c.add("TEST_EVIDENCE_SHAPE", ep + ".evidence.records", "Test needs nonempty recorded evidence entries")
                continue
            st = source["target"]
            expected_target = "{} [{}]".format(st.get("description", ""), st.get("code", ""))
            if target != expected_target or source.get("action_type_candidate") != "Test":
                c.add("TEST_TARGET_SOURCE", ep + ".target", "Test target/type differs from the referenced source")
            for j, record in enumerate(records):
                rp = ep + ".evidence.records[{}]".format(j)
                if not isinstance(record, dict):
                    c.add("TEST_EVIDENCE_SHAPE", rp, "Expected a recorded Test object")
                    continue
                required = {"code", "description", "event_type", "recorded_at", "recorded_evidence", "system", "timing_precision"}
                if set(record) != required:
                    c.add("TEST_EVIDENCE_FIELDS", rp, "Unexpected or missing Test evidence fields")
                if source["event_type"] == "observations":
                    content = record.get("recorded_evidence")
                    if not isinstance(content, dict) or set(content) != {"category", "kind", "units", "value"}:
                        c.add("TEST_CONTENT_FIELDS", rp + ".recorded_evidence", "Unexpected observation fields cannot be forwarded to a model")
                else:
                    c.add("TEST_SOURCE_CONTRACT_UNREVIEWED", rp, "Non-observation Test projection is outside the inspected v34.1 contract", "NOT_REVIEWED")
                matches = (record.get("code") == st.get("code") and record.get("description") == st.get("description")
                           and record.get("system") == st.get("system") and record.get("event_type") == source["event_type"]
                           and record.get("recorded_at") == source["timestamp"] and record.get("recorded_evidence") == source["evidence"]
                           and record.get("timing_precision") == source["timing_precision"])
                if not matches:
                    c.add("TEST_EVIDENCE_SOURCE", rp, "Projected Test content differs from its recorded source")
        elif kind == "Ask":
            if set(payload) != {"facts", "kind", "replay_bin"} or not isinstance(payload.get("replay_bin"), str):
                c.add("ASK_PAYLOAD_FIELDS", ep + ".evidence", "Unexpected or missing serialized patient-report payload fields")
            facts = payload.get("facts")
            if payload.get("kind") != "recorded_patient_report" or not isinstance(facts, list) or not facts:
                c.add("ASK_EVIDENCE_SHAPE", ep + ".evidence", "Ask needs nonempty recorded patient-report facts")
                continue
            for j, fact in enumerate(facts):
                matched = False
                if isinstance(fact, dict):
                    if set(fact) != {"active_status", "code", "description", "duration", "onset", "recorded_at", "units", "value"}:
                        c.add("ASK_FACT_FIELDS", ep + ".evidence.facts[{}]".format(j), "Unexpected or missing patient-report fact fields")
                    for symptom in symptoms if isinstance(symptoms, list) else ():
                        if not isinstance(symptom, dict):
                            continue
                        visible = symptom.get("visible_value", {})
                        metadata = symptom.get("source_metadata", {})
                        original = metadata.get("_source", {}).get("source_record", {}) if isinstance(metadata, dict) and isinstance(metadata.get("_source", {}), dict) else {}
                        if isinstance(visible, dict) and isinstance(original, dict):
                            matched = (symptom.get("visibility_allowed") is True and fact.get("description") == visible.get("symptom_name")
                                       and fact.get("recorded_at") == symptom.get("source_timestamp") and fact.get("value") == visible.get("value")
                                       and fact.get("code") == original.get("CODE") and fact.get("units") == original.get("UNITS"))
                            if matched:
                                break
                if not matched:
                    c.add("ASK_EVIDENCE_SOURCE", ep + ".evidence.facts[{}]".format(j), "Patient report cannot be linked to an allowed symptom source")
        else:
            _check_consult(c, case, event, payload, sources.get(eid), ep)
    return c.result(checks=("legacy_action_types", "target_uniqueness", "test_source_equality", "patient_report_source_links"),
                    not_reviewed=("clinical_action_appropriateness", "question_semantics", "consult_source_contract"))
