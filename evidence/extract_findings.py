"""Offline review interface. Extracted records are not reviewed evidence."""
from typing import List, TypedDict
from app.locdi.dataset_loader import LoCDICase


class FindingRecord(TypedDict):
    case_id: str
    diagnosis: str
    finding: str
    source_title: str
    source_url: str
    review_status: str


def extract_findings(case: LoCDICase) -> List[FindingRecord]:
    """Future review-only extraction; no literature or clinical verdict made."""
    raise NotImplementedError("Review finding extraction is not implemented")
