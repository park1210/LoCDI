"""Whole-dataset validation with explicit transport failures and blocked checks."""
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from app.locdi.dataset_loader import DatasetFormatError, PathLike, load_cases
from validation.schema_validator import ValidationIssue, ValidationResult, validate_schema
from validation.temporal_validator import validate_temporal
from validation.action_validator import validate_actions
from validation.provenance_validator import validate_provenance

VALIDATORS = (validate_schema, validate_temporal, validate_actions, validate_provenance)


@dataclass(frozen=True)
class CaseValidationReport:
    ordinal: int
    case_id: Optional[str]
    results: Tuple[ValidationResult, ...]

    @property
    def has_failures(self) -> bool:
        return any(r.status in {"FAIL", "BLOCKED"} for r in self.results)


@dataclass(frozen=True)
class DatasetValidationReport:
    source: str
    case_count: int
    results: Tuple[ValidationResult, ...]
    cases: Tuple[CaseValidationReport, ...] = ()
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    issue_counts: Dict[str, int] = field(default_factory=dict)
    simulation_readiness: str = "NOT_REVIEWED"
    limitations: Tuple[str, ...] = ()


def validate_dataset(path: PathLike, *, source_root: Optional[PathLike] = None) -> DatasetValidationReport:
    """Load then run schema/temporal/actions/provenance on every record.

    Preserve the loader. On DatasetFormatError only, re-read JSONL to report
    each malformed record explicitly rather than abandoning later records.
    I/O errors propagate. No row is repaired, synthesized, or silently dropped.
    Unsupported schema/structural failures block downstream checks for that row.
    """
    transport_errors = {}
    try:
        records = load_cases(path)
    except DatasetFormatError:
        records = []
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = None
                    transport_errors[len(records)] = "Invalid JSON record"
                records.append(record)
        if not records:
            raise DatasetFormatError("dataset contains no cases")
    identities = Counter(r.get("case_id") for r in records if isinstance(r, dict) and isinstance(r.get("case_id"), str))
    patient_visits = Counter((r.get("patient_id"), r.get("agent_visible", {}).get("index_encounter_id"))
                            for r in records if isinstance(r, dict) and isinstance(r.get("patient_id"), str)
                            and isinstance(r.get("agent_visible"), dict) and isinstance(r["agent_visible"].get("index_encounter_id"), str))
    cases = []
    for index, record in enumerate(records):
        case_id = record.get("case_id") if isinstance(record, dict) and isinstance(record.get("case_id"), str) else None
        schema = validate_schema(record)
        extra = []
        if index in transport_errors:
            extra.append(ValidationIssue("JSONL_RECORD", "$", transport_errors[index]))
        if case_id is not None and identities[case_id] > 1:
            extra.append(ValidationIssue("DUPLICATE_DATASET_CASE", "case_id", "Case identity appears more than once in the dataset"))
        av = record.get("agent_visible", {}) if isinstance(record, dict) else {}
        patient = record.get("patient_id") if isinstance(record, dict) else None
        if isinstance(patient, str) and isinstance(av, dict) and isinstance(av.get("index_encounter_id"), str):
            if patient_visits[(patient, av["index_encounter_id"])] > 1:
                extra.append(ValidationIssue("DUPLICATE_DATASET_VISIT", "agent_visible.index_encounter_id", "Patient/visit identity appears more than once"))
        if extra:
            schema = ValidationResult("schema", "FAIL", schema.issues + tuple(extra), case_id, schema.checks, schema.not_reviewed)
        results = [schema]
        if schema.status == "FAIL":
            for name in ("temporal", "actions", "provenance"):
                results.append(ValidationResult(name, "BLOCKED", (ValidationIssue("SCHEMA_BLOCKED", "$", "Schema/transport failures block this record", "BLOCKED"),), case_id))
        else:
            results.extend((validate_temporal(record), validate_actions(record),
                            validate_provenance(record, source_root=source_root)))
        cases.append(CaseValidationReport(index + 1, case_id, tuple(results)))
    flat = tuple(result for case in cases for result in case.results)
    statuses = ("PASS", "FAIL", "WARNING", "NOT_REVIEWED", "NOT_IMPLEMENTED", "BLOCKED", "SKIPPED")
    counts = {name: {s: sum(r.validator == name and r.status == s for r in flat) for s in statuses}
              for name in ("schema", "temporal", "actions", "provenance")}
    codes = Counter(i.code for r in flat for i in r.issues)
    return DatasetValidationReport(str(path), len(cases), flat, tuple(cases), counts, dict(codes), "NOT_REVIEWED",
                                   ("validation is not a model-run readiness check", "clinical labels and actual release times require review",
                                    "original-file integrity requires explicit source_root and available originals"))
