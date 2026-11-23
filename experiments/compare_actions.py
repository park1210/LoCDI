"""Compare action vocabularies without changing the base B0-B4 action set."""
from typing import Any, Dict, Sequence
from app.locdi.action_registry import ActionSpec


def compare_action_sets(baseline: Sequence[ActionSpec],
                        extended: Sequence[ActionSpec]) -> Dict[str, Any]:
    """Future vocabulary/coverage comparison, separate from primary ablation."""
    raise NotImplementedError("Action-set comparison is not implemented")
