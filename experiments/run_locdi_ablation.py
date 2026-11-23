"""Official B0-B4 gate and pinned current-version paired Harness entrypoint."""
from typing import Any, Dict, Mapping
from app.locdi.run_simulation import SimulationOptions, run_cases
from experiments.locdi_metrics import evaluate_episode, aggregate_metrics

# User-confirmed design. These are switches, never learned model artifacts.
CONDITIONS = {
    "B0": ("none", None, "er_only", "rule_similarity", False, "specialist_heuristic"),
    "B1": ("full_history", None, "er_only", "rule_similarity", False, "specialist_heuristic"),
    "B2": ("selective", 5, "er_only", "rule_similarity", False, "specialist_heuristic"),
    "B3": ("selective", 5, "adaptive", "adaptive", False, "specialist_heuristic"),
    "B4": ("selective", 5, "adaptive", "adaptive", True, "bayesian_voi"),
}
FEATURE_KEYS = ("memory_mode", "top_k", "routing_mode", "primary_router",
                "voi_enabled", "action_selector")


class AblationConfigurationError(ValueError):
    """Experiment is incomplete, unverified, or inconsistent with its design."""


def validate_ablation_config(config: Mapping[str, Any]) -> None:
    """Reject unresolved settings before any runner/model can be reached.

    Readiness of supplied artifacts cannot be certified by this basic gate;
    run_ablation additionally checks the pinned paired execution contract and
    the existing research artifact admission gate.
    """
    condition = config.get("condition")
    if condition not in CONDITIONS:
        raise AblationConfigurationError("condition must be B0..B4")
    features = config.get("features")
    if not isinstance(features, Mapping) or set(features) != set(FEATURE_KEYS):
        raise AblationConfigurationError("exact official feature keys required")
    expected = dict(zip(FEATURE_KEYS, CONDITIONS[condition]))
    if any(type(features[k]) is not type(expected[k]) or features[k] != expected[k]
           for k in FEATURE_KEYS):
        raise AblationConfigurationError("features differ from the official condition")
    if config.get("ready") is not True:
        raise AblationConfigurationError("ready must be true after execution readiness is verified")
    common = config.get("common")
    if not isinstance(common, Mapping) or common.get("memory_source") != "fixed":
        raise AblationConfigurationError("common fixed-memory settings required")
    required = ("model", "model_version", "decoding", "minute_limit", "agent_registry", "diagnosis_candidates",
                "evaluation", "action_set", "patient_split", "repeats", "fixed_memory")
    if any(common.get(k) is None for k in required):
        raise AblationConfigurationError("unresolved common execution dependencies")
    # No claim that nested artifacts are valid: that checker is future work.
    if condition in {"B3", "B4"} and common.get("routing_utility") is None:
        raise AblationConfigurationError("adaptive routing requires routing_utility")
    if condition == "B4" and (common.get("voi_model") is None or
                              type(common.get("voi_horizon")) is not int or
                              common["voi_horizon"] not in {2, 3}):
        raise AblationConfigurationError("B4 requires a VoI model and multi-step horizon 2..3")


def run_ablation(config: Mapping[str, Any], *, execution_contract=None, panel=None,
                 run_one=None, execute=False, approval_record=None,
                 public_root=None, private_root=None, artifact_inputs=None) -> Dict[str, Any]:
    """Use the actual paired Harness only with pinned official configuration.

    B0/B1 top_k=null is an experiment value; the legacy runner nevertheless
    requires an unused positive top_k. Record translation before implementation.
    All shared inputs, model/version/decoding and patient split must be equal.
    R0 is intentionally outside this mandatory configuration set. Development
    Source comparisons use their separately named entrypoint, never this gate.
    Default is non-executing; existing research admission remains mandatory.
    """
    validate_ablation_config(config)
    if execution_contract is None or panel is None:
        raise AblationConfigurationError("Pinned common execution contract and same-case panel are required")
    if execution_contract.get("purpose") != "official_research_evaluation":
        raise AblationConfigurationError("Development runs cannot be represented as official ablation")
    from app.locdi.llm_setup import digest
    pins = execution_contract.get("common", {}).get("official_config_fingerprints", {})
    if pins.get(config["condition"]) != digest(config):
        raise AblationConfigurationError("Official configuration must be pinned in the execution contract")
    if config["condition"] not in execution_contract.get("conditions", []):
        raise AblationConfigurationError("Condition is absent from paired execution contract")
    from app.locdi.current_ablation_runner import run_paired_ablation
    return run_paired_ablation(execution_contract, panel, run_one=run_one, execute=execute,
        approval_record=approval_record, public_root=public_root, private_root=private_root,
        artifact_inputs=artifact_inputs)
