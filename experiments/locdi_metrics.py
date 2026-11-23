"""Readiness-aware LoCDI metric interfaces; gold is evaluator-private."""
from typing import Any, Dict, Mapping, Sequence
from app.core import paper_evaluation
from app.locdi.case_adapter import AdaptedCase


def evaluate_episode(episode: Mapping[str, Any], *, context: AdaptedCase) -> Dict[str, Any]:
    """Future Accuracy/Memory/Routing/VoI/Action values plus readiness/coverage.

    Gold and evaluation are consumed only here. paper_evaluation.episode_rows
    needs legacy identities, explicit labels and a runner report; a translator
    and readiness guards are required before reuse. Unavailable != zero.
    """
    raise NotImplementedError("LoCDI episode metric translation is not implemented")


def aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Future patient-paired denominators, coverage and unknown values.

    Do not call paper_evaluation.aggregate with arbitrary LoCDI-shaped dicts.
    """
    raise NotImplementedError("LoCDI metric aggregation is not implemented")
