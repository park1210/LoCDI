"""Clinical review aggregation interface, independent of model execution."""
from typing import Any, Dict, Iterable
from evidence.extract_findings import FindingRecord


def summarize_reviews(records: Iterable[FindingRecord]) -> Dict[str, Any]:
    """Future status/coverage counts; missing review never means accepted."""
    raise NotImplementedError("Clinical review aggregation is not implemented")
