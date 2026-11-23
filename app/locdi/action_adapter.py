"""Extension boundary for the existing source Ask/Test/Consult contracts."""
from typing import Any, Dict, Mapping
from app.locdi.action_registry import ActionRegistry, ActionSpec

# Documentation of existing policy names, not an executable action catalog.
LEGACY_ACTION_NAMES = {
    "Ask": ("ask_patient", "ask_history"),
    "Test": ("request_test", "order_test"),
    "Consult": ("consult_specialty", "request_specialty"),
}


def normalize_action(action: Mapping[str, Any], *, registry: ActionRegistry) -> Dict[str, Any]:
    """Future canonical identity/target lookup; reject unsupported extensions."""
    raise NotImplementedError("Action normalization is not implemented")


def map_action_to_runtime(action: Mapping[str, Any],
                          *, registry: ActionRegistry) -> Dict[str, Any]:
    """Future mapping to legacy policy result, retaining source/time checks.

    No Ask/Test/Consult target or future result may be invented by the adapter.
    """
    raise NotImplementedError("Runtime action mapping is not implemented")
