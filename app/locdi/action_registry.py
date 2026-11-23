"""Action identity contracts, independent of medical action availability."""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    name: str
    target_kind: str
    runtime_action: Optional[str] = None
    description: str = ""


class ActionRegistry:
    """Explicit instance-owned registry; no global/default invented actions."""

    def __init__(self, specs: Iterable[ActionSpec] = ()) -> None:
        self._specs: Dict[str, ActionSpec] = {}
        for spec in specs:
            self.register_action(spec)

    def register_action(self, spec: ActionSpec) -> None:
        """Register a unique identity, not a claim of runtime executability."""
        if not isinstance(spec, ActionSpec):
            raise TypeError("spec must be ActionSpec")
        if any(not isinstance(value, str) or not value.strip()
               for value in (spec.action_id, spec.name, spec.target_kind)):
            raise ValueError("action identity fields must be nonempty strings")
        if spec.action_id in self._specs:
            raise ValueError("duplicate action_id: " + spec.action_id)
        self._specs[spec.action_id] = spec

    def get_action(self, action_id: str) -> ActionSpec:
        """Lookup only; KeyError for an unregistered identity."""
        return self._specs[action_id]

    def get_available_actions(self, context: Mapping[str, Any]) -> Tuple[ActionSpec, ...]:
        """Future source/time/resource constraints; registered != available."""
        raise NotImplementedError("Action availability rules are not implemented")


def register_action(registry: ActionRegistry, spec: ActionSpec) -> None:
    registry.register_action(spec)


def get_available_actions(registry: ActionRegistry,
                          context: Mapping[str, Any]) -> Tuple[ActionSpec, ...]:
    return registry.get_available_actions(context)
