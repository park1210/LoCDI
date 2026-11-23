"""Check known chronology without equating recorded and released times."""
from datetime import timedelta
from functools import lru_cache
import json
from app.core.longitudinal_memory import timestamp
from app.locdi.dataset_loader import LoCDICase
from validation.schema_validator import ValidationResult, _Check, _schema_block


@lru_cache(maxsize=100000)
def _timestamp(value: str):
    return timestamp(value)


def _time(c: _Check, value, path: str):
    try:
        return _timestamp(value)
    except (ValueError, TypeError, OverflowError):
        c.add("TIMESTAMP", path, "Expected a valid timezone-aware ISO timestamp")
        return None


def validate_temporal(case: LoCDICase) -> ValidationResult:
    """Validate memory cutoffs, replay clocks and source/projection ordering.

    Source intervals describe recording precision, NOT observed release times.
    Future private source records are allowed; a projected record must not be
    released before it was recorded. Missing visit ends are not synthesized.
    """
    blocked = _schema_block(case, "temporal")
    if blocked:
        return blocked
    c = _Check("temporal", case)
    av, env = case["agent_visible"], case["environment"]
    start = _time(c, av["index_encounter_start"], "agent_visible.index_encounter_start")
    if start is None:
        return c.result(not_reviewed=("remaining_temporal_checks",))
    starts, missing_end, records_by_id = set(), False, {}
    for i, visit in enumerate(av["past_visits"]):
        vp = "agent_visible.past_visits[{}]".format(i)
        past = _time(c, visit["started_at"], vp + ".started_at")
        if past is not None:
            if past >= start:
                c.add("PAST_VISIT_ORDER", vp + ".started_at", "Past visit must begin strictly before the index visit")
            if past in starts:
                c.add("COINCIDENT_PAST_VISITS", vp + ".started_at", "Distinct past visit IDs share a recorded start; overlap/end semantics need review", "WARNING")
            starts.add(past)
        end = None
        if visit.get("ended_at") is None:
            missing_end = True
        else:
            end = _time(c, visit["ended_at"], vp + ".ended_at")
            if end is not None and past is not None and not past <= end <= start:
                c.add("VISIT_END_ORDER", vp + ".ended_at", "Past visit end must follow its start and precede the index visit")
        for j, record in enumerate(visit["records"]):
            rp = vp + ".records[{}]".format(j)
            ts = [_time(c, record[k], rp + "." + k) for k in ("visit_started_at", "observed_at", "available_at")]
            began, observed, available = ts
            if all(t is not None for t in ts):
                if not began <= observed <= available:
                    c.add("MEMORY_ORDER", rp, "Memory requires visit start <= observation <= availability")
                if available > start or observed > start:
                    c.add("FUTURE_MEMORY", rp, "Past memory is not available by the index cutoff")
                if past is not None and began != past:
                    c.add("MEMORY_VISIT_TIME", rp + ".visit_started_at", "Memory start differs from its visit")
                if end is not None and observed > end:
                    c.add("OBSERVATION_AFTER_VISIT", rp + ".observed_at", "Observation follows the declared visit end")
                for old_id in record["supersedes"]:
                    old = records_by_id.get(old_id)
                    old_time = _time(c, old["available_at"], rp + ".supersedes") if old is not None else None
                    if old is None or old_time is None or old["patient_id"] != record["patient_id"] or old_time > available:
                        c.add("SUPERSEDES_ORDER", rp + ".supersedes", "Superseded memory must be an earlier available record of the same patient")
            records_by_id[record["memory_id"]] = record
    if missing_end:
        c.add("VISIT_END_NOT_RECORDED", "agent_visible.past_visits", "Past visit end/duration cannot be checked where no end is recorded", "NOT_REVIEWED")
    replay = env["data_replay"]
    original = _time(c, replay["original_decision_time"], "environment.data_replay.original_decision_time")
    if original is not None and original != start:
        c.add("REPLAY_CLOCK_UNREVIEWED", "environment.data_replay", "Index and source decision clocks differ; no implicit time translation is applied", "WARNING")
    if replay["observed_release_time"] is not True:
        c.add("RELEASE_TIME_UNOBSERVED", "environment.data_replay.observed_release_time", "Recorded timestamps do not establish actual doctor release times", "WARNING")
    if replay["requires_time_series_loader_for_latest_value"]:
        c.add("LATEST_VALUE_REPLAY_UNREVIEWED", "environment.data_replay", "Projected records do not verify latest-value time-series replay", "NOT_REVIEWED")
    audit = case["provenance"].get("source_case_audit", {})
    symptoms = audit.get("symptom_sources", []) if isinstance(audit, dict) else []
    for i, symptom in enumerate(symptoms if isinstance(symptoms, list) else []):
        if isinstance(symptom, dict) and symptom.get("visibility_allowed") is True:
            path = "provenance.source_case_audit.symptom_sources[{}].source_timestamp".format(i)
            observed = _time(c, symptom.get("source_timestamp"), path)
            if observed is not None and observed > start:
                c.add("FUTURE_INITIAL_SYMPTOM", path, "An initially visible symptom is recorded after the index cutoff")
    sources = {s["event_id"]: s for s in env["source_event_store"]}
    decision_ms = start.timestamp() * 1000
    for i, source in enumerate(env["source_event_store"]):
        sp = "environment.source_event_store[{}]".format(i)
        lower, upper = source["timestamp_lower_epoch_ms"], source["timestamp_upper_epoch_ms"]
        if lower > upper:
            c.add("SOURCE_INTERVAL", sp, "Recording precision interval is reversed")
        precision = source["timing_precision"]
        if precision not in {"date", "second", "millisecond"}:
            c.add("SOURCE_PRECISION_UNKNOWN", sp + ".timing_precision", "Unsupported recording precision needs review", "NOT_REVIEWED")
        if precision == "date":
            # A day interval is intentionally not interpreted as midnight release.
            c.add("DATE_PRECISION_UNREVIEWED", sp + ".timestamp", "Day-level source time cannot establish an exact recording/release instant", "NOT_REVIEWED")
            continue
        nominal = _time(c, source["timestamp"], sp + ".timestamp")
        if nominal is not None and not lower <= nominal.timestamp() * 1000 <= upper:
            c.add("SOURCE_TIME_BOUNDS", sp, "Recorded timestamp lies outside its precision interval")
        relation = source["relative_to_decision"]
        clock = original.timestamp() * 1000 if original is not None else decision_ms
        expected = "after_decision" if lower > clock else "before_decision" if upper < clock else "same_precision_interval_uncertain"
        if relation != expected:
            c.add("SOURCE_TIME_RELATION", sp + ".relative_to_decision", "Declared source relation conflicts with its recording interval")
    for i, event in enumerate(env["current_encounter_events"]):
        ep = "environment.current_encounter_events[{}]".format(i)
        try:
            release = start + timedelta(minutes=event["available_minute"])
        except OverflowError:
            c.add("AVAILABLE_MINUTE", ep + ".available_minute", "Minute offset exceeds the representable clock")
            continue
        try:
            payload = json.loads(event["evidence"])
        except (ValueError, TypeError):
            c.add("EVIDENCE_JSON", ep + ".evidence", "Cannot check evidence chronology: invalid JSON evidence")
            continue
        if not isinstance(payload, dict):
            c.add("EVIDENCE_SHAPE", ep + ".evidence", "Cannot check evidence chronology: expected an object")
            continue
        entries = payload.get("records", payload.get("facts", []))
        if not isinstance(entries, list):
            c.add("EVIDENCE_SHAPE", ep + ".evidence", "Expected an array of recorded evidence entries")
            continue
        for j, item in enumerate(entries):
            path = ep + ".evidence.entries[{}].recorded_at".format(j)
            observed = _time(c, item.get("recorded_at") if isinstance(item, dict) else None, path)
            if observed is not None and observed > release:
                c.add("PROJECTED_FUTURE_RECORD", path, "Projected evidence would be released before it was recorded")
        source = sources.get(event["event_id"])
        if source is not None:
            lo, hi = source["timestamp_lower_epoch_ms"], source["timestamp_upper_epoch_ms"]
            if lo > release.timestamp() * 1000:
                c.add("SOURCE_NOT_YET_RECORDED", ep, "Projected source recording interval begins after the declared release")
            elif hi > release.timestamp() * 1000:
                c.add("SOURCE_RELEASE_INTERVAL_UNCERTAIN", ep, "Recording precision interval overlaps the projected release boundary", "WARNING")
    return c.result(checks=("visit_order", "memory_cutoff", "source_recording_intervals", "projected_release_bounds"),
                    not_reviewed=("unrecorded_visit_ends", "observed_doctor_release_times", "latest_value_time_series_replay", "clinical_time_course"))
