"""Failure taxonomy interface; missing data is not model diagnostic failure."""
from typing import Any, Dict, Mapping, Sequence


def analyze_failures(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Future split config, data, validation, role-call and clinical failures."""
    raise NotImplementedError("LoCDI failure analysis is not implemented")
