"""Separate provenance shape/linkage from verified original-file hashes."""
import hashlib
from pathlib import Path, PureWindowsPath
import re
from typing import Optional
from app.locdi.dataset_loader import LoCDICase, PathLike
from validation.schema_validator import ValidationResult, _Check, _schema_block


def validate_provenance(case: LoCDICase, *, source_root: Optional[PathLike] = None) -> ValidationResult:
    """Check references; rehash only under an explicitly supplied source_root.

    Never resolve paths against cwd implicitly, read absolute/traversal paths,
    download originals, or treat correctly formatted hashes as verified bytes.
    source_integrity_verified covers declared file hashes only, not clinical
    correctness or reconstruction of source CSV rows.
    """
    blocked = _schema_block(case, "provenance")
    if blocked:
        return blocked
    c = _Check("provenance", case)
    p = c.mapping(case["provenance"], "provenance",
                  {"source_package", "source_sha256", "source_case_audit", "visibility_policy"})
    for key in ("source_package", "visibility_policy"):
        c.text(p.get(key), "provenance." + key)
    hashes = c.mapping(p.get("source_sha256"), "provenance.source_sha256")
    if not hashes:
        c.add("SOURCE_HASHES_EMPTY", "provenance.source_sha256", "Declared source hashes are required")
    audit = c.mapping(p.get("source_case_audit"), "provenance.source_case_audit",
                      {"case_id", "clinical_review_status", "diagnosis_sources", "symptom_sources",
                       "other_current_source_candidates_not_projected", "source_annotation_store",
                       "source_policy_primary_specialty_candidate", "selection_score"})
    if audit.get("case_id") != case["case_id"]:
        c.add("AUDIT_CASE_ID", "provenance.source_case_audit.case_id", "Audit identity differs from wrapper case identity")
    for key in ("clinical_review_status", "source_annotation_store", "source_policy_primary_specialty_candidate"):
        c.text(audit.get(key), "provenance.source_case_audit." + key)
    if audit.get("clinical_review_status") != "expert_reviewed":
        c.add("CLINICAL_REVIEW_UNVERIFIED", "provenance.source_case_audit.clinical_review_status", "Source review status does not certify expert clinical review", "NOT_REVIEWED")
    for key in ("diagnosis_sources", "symptom_sources", "other_current_source_candidates_not_projected"):
        records = c.sequence(audit.get(key), "provenance.source_case_audit." + key)
        for i, record in enumerate(records):
            rp = "provenance.source_case_audit.{}[{}]".format(key, i)
            r = c.mapping(record, rp)
            if key == "diagnosis_sources":
                c.text(r.get("source_file"), rp + ".source_file")
                if type(r.get("source_row")) is not int or r["source_row"] < 1:
                    c.add("SOURCE_ROW", rp + ".source_row", "Expected a positive source row number")
                c.mapping(r.get("source_record"), rp + ".source_record")
            else:
                for field in ("source_id", "source_timestamp", "evidence_grade", "kind"):
                    c.text(r.get(field), rp + "." + field)
                c.mapping(r.get("source_metadata"), rp + ".source_metadata")
                c.mapping(r.get("visible_value"), rp + ".visible_value")
                if type(r.get("visibility_allowed")) is not bool:
                    c.add("VISIBILITY_FLAG", rp + ".visibility_allowed", "Expected an explicit boolean visibility decision")
    for i, source in enumerate(case["environment"]["source_event_store"]):
        sp = "environment.source_event_store[{}].source_metadata".format(i)
        meta = source["source_metadata"]
        c.text(meta.get("source_file"), sp + ".source_file")
        if type(meta.get("source_row")) is not int or meta["source_row"] < 1:
            c.add("SOURCE_ROW", sp + ".source_row", "Expected a positive source row number")
        c.mapping(meta.get("source_record"), sp + ".source_record")
    root = Path(source_root).resolve() if source_root is not None else None
    verified, mismatched, unavailable = 0, False, False
    for i, (name, digest) in enumerate(hashes.items()):
        hp = "provenance.source_sha256.entry[{}]".format(i)
        if not isinstance(name, str) or not name.strip():
            c.add("SOURCE_PATH", hp, "Expected a nonempty relative source reference")
            continue
        normalized = name.replace("\\", "/")
        parts = normalized.split("/")
        if Path(normalized).is_absolute() or PureWindowsPath(name).drive or ".." in parts:
            c.add("UNSAFE_SOURCE_PATH", hp, "Absolute, drive-qualified or parent-traversal source paths are not accepted")
            continue
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            c.add("SOURCE_HASH_FORMAT", hp, "Expected a 64-character SHA-256 hexadecimal digest")
            continue
        if root is None:
            unavailable = True
            continue
        target = (root / normalized).resolve()
        if target != root and root not in target.parents:
            c.add("UNSAFE_SOURCE_PATH", hp, "Resolved source path escapes the declared source root")
            continue
        if not target.is_file():
            unavailable = True
            c.add("SOURCE_FILE_MISSING", hp, "Original source file is unavailable under the declared source root", "NOT_REVIEWED")
            continue
        try:
            actual = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    actual.update(block)
        except OSError:
            unavailable = True
            c.add("SOURCE_FILE_UNREADABLE", hp, "Original source file could not be read", "NOT_REVIEWED")
            continue
        if actual.hexdigest() != digest.lower():
            mismatched = True
            c.add("SOURCE_HASH_MISMATCH", hp, "Original file bytes do not match the declared source hash")
        else:
            verified += 1
    if unavailable and root is None:
        c.add("SOURCE_BYTES_NOT_VERIFIED", "provenance.source_sha256", "Hash format checked only; no original source root was supplied", "NOT_REVIEWED")
    integrity = False if mismatched else True if hashes and verified == len(hashes) else None
    return c.result(checks=("provenance_fields", "hash_format", "case_audit_identity", "local_metadata_links"),
                    not_reviewed=("source_row_reconstruction", "clinical_review", "external_source_identity") +
                                 (() if integrity is True else ("unavailable_original_file_bytes",)),
                    verified_source_hashes=verified, source_integrity_verified=integrity)
