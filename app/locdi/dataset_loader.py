"""Read the existing LoCDI wrapper; parsing is not clinical validation."""
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union, cast
import json

PathLike = Union[str, Path]


class LoCDICase(TypedDict):
    """Dataset wrapper. Only agent_visible is eligible for agent projection."""
    schema_version: str
    case_id: str
    patient_id: str
    agent_visible: Dict[str, Any]
    environment: Dict[str, Any]
    evaluation: Dict[str, Any]
    task_annotations: Dict[str, Any]
    provenance: Dict[str, Any]


class DatasetFormatError(ValueError):
    """Invalid JSONL/wrapper or duplicate case ID; includes source line."""


def load_cases(path: PathLike) -> List[LoCDICase]:
    """Read UTF-8 JSONL, ignoring blank lines, preserving all wrapper fields.

    Raises OSError for I/O errors and DatasetFormatError for malformed JSON,
    missing/wrong top-level fields, or duplicate IDs. This checks transport
    structure only: temporal, provenance and clinical validity are NOT checked.
    Never flattens environment, evaluation or annotations into agent input.
    """
    cases: List[LoCDICase] = []
    seen = set()
    mappings = ("agent_visible", "environment", "evaluation", "task_annotations", "provenance")
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError("line {}: invalid JSON".format(number)) from exc
            if not isinstance(row, dict):
                raise DatasetFormatError("line {}: expected object".format(number))
            if any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ("schema_version", "case_id", "patient_id")):
                raise DatasetFormatError("line {}: missing string identity/schema".format(number))
            if any(not isinstance(row.get(key), dict) for key in mappings):
                raise DatasetFormatError("line {}: missing wrapper mapping".format(number))
            if row["case_id"] in seen:
                raise DatasetFormatError("line {}: duplicate case_id".format(number))
            seen.add(row["case_id"])
            cases.append(cast(LoCDICase, row))
    if not cases:
        raise DatasetFormatError("dataset contains no cases")
    return cases


def load_case(path: PathLike, case_id: str) -> LoCDICase:
    """Return one case after checking the entire file; KeyError if absent."""
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a nonempty string")
    for case in load_cases(path):
        if case["case_id"] == case_id:
            return case
    raise KeyError(case_id)
