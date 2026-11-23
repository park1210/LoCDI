"""Static paired-design checks. Does not execute, evaluate or approve a study."""
from collections import Counter
from collections.abc import Mapping
import math
from app.core.agent_registry import AgentRegistry
from experiments.run_locdi_ablation import CONDITIONS, FEATURE_KEYS
from validation.artifact_readiness import validate_routing_artifact, validate_voi_artifact

FIXED = ("dataset", "case_count", "model", "model_version", "decoding", "agent_registry", "diagnosis_candidates",
         "minute_limit", "memory_source", "fixed_memory", "patient_split", "repeats", "action_set", "evaluation")
FEATURE_VALUES = {"none", "full_history", "selective", "er_only", "rule_similarity", "adaptive",
                  "specialist_heuristic", "bayesian_voi"}


def _display(value, path):
    # Full difference paths, but do not echo private labels, IDs or bindings.
    if value is None or type(value) is bool:
        return value
    if path.startswith("features.") and (type(value) is int or isinstance(value, str) and value in FEATURE_VALUES):
        return value
    if path == "condition" and isinstance(value, str) and value in CONDITIONS:
        return value
    return "SET_REDACTED"


def _differences(a, b, path=""):
    result = []
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        # Known keys remain readable; caller-provided clinical dictionary keys
        # under common values are represented as positions, never printed.
        known = set(FIXED) | {"condition", "ready", "features", "common", "routing_utility", "voi_model", "voi_horizon"} | set(FEATURE_KEYS)
        for index, key in enumerate(sorted(set(a) | set(b), key=str)):
            label = key if isinstance(key, str) and key in known and path in ("", "common", "features") else "field[{}]".format(index)
            child = path + "." + label if path else label
            if key not in a or key not in b:
                result.append({"path": child, "before_present": key in a, "after_present": key in b,
                               "before": _display(a.get(key), child), "after": _display(b.get(key), child)})
            else:
                result += _differences(a[key], b[key], child)
    elif type(a) is not type(b) or a != b:
        result.append({"path": path, "before": _display(a, path), "after": _display(b, path)})
    return result


def _configured(common, key):
    value = common.get(key)
    if key in ("case_count", "minute_limit", "repeats"):
        return type(value) is int and value > 0
    if key == "memory_source":
        return value == "fixed"
    if key in ("dataset", "model", "model_version"):
        return isinstance(value, str) and bool(value.strip())
    if key == "decoding":
        return (isinstance(value, Mapping) and type(value.get("seed")) is int and
                type(value.get("temperature")) in (int, float) and math.isfinite(value["temperature"]) and
                type(value.get("max_output_tokens")) is int and value["max_output_tokens"] > 0)
    return value is not None and value != {} and value != [] and value != ""


def _case_keys(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    result = []
    for case in rows:
        try:
            key = (case["patient_id"], case["agent_visible"]["index_encounter_id"], case["case_id"])
            if any(not isinstance(x, str) or not x for x in key):
                return None
            result.append(key)
        except (KeyError, TypeError):
            return None
    return result if len(set(result)) == len(result) and len({k[:2] for k in result}) == len(result) else None


def check_ablation_fairness(configs, *, cases=None, case_sets=None, episodes=None):
    """Compare all fields, incomplete-vs-equal, paired identities and coverage.

    No implicit file resolution. Artifact references must be independently
    loaded by a caller; only inline dictionaries can be structurally inspected.
    Dataset/evaluation stores are never passed to a model. Caller-controlled
    ready/review flags cannot enable the unimplemented run/metric interfaces.
    """
    checks, warnings, per_condition = [], [], {}
    if isinstance(configs, (list, tuple)):
        valid = len(configs) == 5 and all(isinstance(c, Mapping) and isinstance(c.get("condition"), str) and c.get("condition") in CONDITIONS for c in configs)
        configs = {c["condition"]: c for c in configs} if valid else {}
    valid_set = isinstance(configs, Mapping) and set(configs) == set(CONDITIONS) and all(isinstance(c, Mapping) for c in configs.values())
    if not valid_set:
        return {"schema_version": "locdi.ablation_fairness.v1", "status": "BLOCKED", "checks": [{"code": "EXACT_B0_B4_REQUIRED", "status": "INVALID"}],
                "comparisons": [], "research_validated": False, "clinical_metrics": "NOT_EVALUATED"}
    common_by_condition = {k: c.get("common") if isinstance(c.get("common"), Mapping) else {} for k, c in configs.items()}
    for key in FIXED:
        commons = list(common_by_condition.values())
        present = all(_configured(common, key) for common in commons)
        equal = all(common.get(key) == commons[0].get(key) for common in commons)
        checks.append({"code": "COMMON_" + key.upper(), "status": "READY" if present and equal else "BLOCKED",
                       "configured": present, "equal": equal})
    pairs = []
    for i in range(4):
        first, second = "B{}".format(i), "B{}".format(i + 1)
        warnings_for_pair = [["HISTORY_INFORMATION_AND_CONTEXT_LENGTH"], ["MEMORY_INFORMATION_STRATEGY_AND_CONTEXT_LENGTH"],
            ["ROUTING_AND_SPECIALTY_ACCESS_CHANGE", "PRIMARY_BINDING_AND_MENU_ROUTING_CHANGE"],
            ["VOI_SELECTOR_AND_DIAGNOSE_AVAILABILITY_CHANGE", "SELECTOR_FALLBACK_AND_COVERAGE_REQUIRE_RUN_AUDIT",
             "LOCDI_MEMORY_PRECEDES_VOI_SELECTION"]][i]
        pairs.append({"comparison": first + "_TO_" + second, "differences": _differences(configs[first], configs[second]),
                      "isolated_single_feature_causal_effect": False, "warnings": warnings_for_pair})
    private_keys = _case_keys(cases)
    eval_patients = sorted({key[0] for key in private_keys}) if private_keys else None
    source_actions = [e for c in cases for e in c.get("environment", {}).get("current_encounter_events", [])] if private_keys else None
    if case_sets is not None and isinstance(case_sets, Mapping) and set(case_sets) == set(CONDITIONS):
        sets = [_case_keys(case_sets[k]) for k in CONDITIONS]
        paired = all(keys is not None and set(keys) == set(sets[0] or []) for keys in sets)
        order_equal = paired and all(keys == sets[0] for keys in sets)
        paired_status = "READY" if paired else "BLOCKED"
        source_scopes = []
        for k in CONDITIONS:
            source_scopes.append({(c["patient_id"], c["agent_visible"]["index_encounter_id"]): c.get("environment", {}).get("current_encounter_events") for c in case_sets[k]} if _case_keys(case_sets[k]) else None)
        source_equal = (paired and all(scope is not None and all(isinstance(events, list) for events in scope.values()) for scope in source_scopes)
                        and all(scope == source_scopes[0] for scope in source_scopes))
        checks.append({"code": "PAIRED_SOURCE_ACTIONS_AND_PAYLOADS", "status": "READY" if source_equal else "BLOCKED"})
    else:
        paired_status, order_equal = "NOT_REVIEWED", None
        checks.append({"code": "PAIRED_SOURCE_ACTIONS_AND_PAYLOADS", "status": "NOT_REVIEWED"})
    checks.append({"code": "PAIRED_PRIVATE_CASE_IDENTITIES", "status": paired_status})
    warnings.append("PUBLIC_ORDINALS_DO_NOT_PROVE_CROSS_RUN_IDENTITY")
    for condition, expected_values in CONDITIONS.items():
        config = configs[condition]
        features = config.get("features", {})
        official = (config.get("condition") == condition and isinstance(features, Mapping) and set(features) == set(FEATURE_KEYS)
                    and all(type(features[k]) is type(v) and features[k] == v for k, v in zip(FEATURE_KEYS, expected_values)))
        common = common_by_condition[condition]
        item = {"official_features": "READY" if official else "INVALID", "declared_ready": config.get("ready") is True,
                "routing_contract": "NOT_APPLICABLE", "voi_contract": "NOT_APPLICABLE"}
        expected_bindings = None
        try:
            registry = AgentRegistry(common.get("agent_registry"))
            expected_bindings = {a["agent_id"]: (common.get("model"), common.get("model_version")) for a in registry.snapshot()["agents"]}
            item["model_bindings"] = "READY" if all((a["model"], a["version"]) == expected_bindings[a["agent_id"]] for a in registry.snapshot()["agents"]) else "BLOCKED"
        except (ValueError, TypeError, KeyError, AttributeError):
            item["model_bindings"] = "BLOCKED"
        if condition in ("B3", "B4"):
            item["routing_contract"] = validate_routing_artifact(common.get("agent_registry"), common.get("routing_utility"),
                evaluation_patient_ids=eval_patients, expected_bindings=expected_bindings)["development_status"]
        if condition == "B4":
            item["voi_contract"] = validate_voi_artifact(common.get("voi_model"), source_actions=source_actions,
                evaluation_patient_ids=eval_patients, expected_horizon=common.get("voi_horizon"),
                diagnosis_candidates=common.get("diagnosis_candidates"))["development_status"]
            if type(common.get("voi_horizon")) is not int or common["voi_horizon"] not in (2, 3):
                item["voi_contract"] = "BLOCKED"
        per_condition[condition] = item
    route_same = common_by_condition["B3"].get("routing_utility") == common_by_condition["B4"].get("routing_utility")
    checks.append({"code": "B3_B4_ROUTING_UTILITY_EQUAL", "status": "READY" if route_same and common_by_condition["B3"].get("routing_utility") is not None else "BLOCKED"})
    if private_keys:
        checks.append({"code": "DECLARED_CASE_COUNT", "status": "READY" if all(c.get("case_count") == len(private_keys) for c in common_by_condition.values()) else "BLOCKED"})
        metric_ready = all(all(c.get("task_annotations", {}).get("metric_readiness", {}).get(k) is True for k in
                              ("diagnosis_accuracy", "memory_retrieval", "routing_accuracy", "voi_regret")) for c in cases)
        gold_complete = all(all(isinstance(c.get("task_annotations", {}).get("gold", {}).get(k), list)
                               for k in ("acceptable_next_actions", "accepted_departments", "accepted_diagnoses", "relevant_memory_ids", "stale_memory_ids")) for c in cases)
        metric_ready = metric_ready and gold_complete
    else:
        metric_ready = False
    checks.append({"code": "DATASET_METRIC_READINESS", "status": "READY" if metric_ready else "BLOCKED"})
    if episodes is not None and isinstance(episodes, Mapping):
        for condition in CONDITIONS:
            rows = episodes.get(condition)
            if not isinstance(rows, (list, tuple)):
                per_condition[condition]["execution_audit"] = "NOT_REVIEWED"
                continue
            decisions = [d for ep in rows for d in ep.get("workflow", {}).get("decisions", [])]
            unknown = sum(type(d.get("fallback_used")) is not bool or d.get("selector_status") not in ("available", "unavailable") for d in decisions)
            counts = Counter(d.get("actual_selector") if d.get("actual_selector") in ("specialist_heuristic", "bayesian_voi", "explicit_outcome_voi", "information_gain", "random") else "NOT_RECORDED" for d in decisions)
            per_condition[condition]["execution_audit"] = {"decision_count": len(decisions), "actual_selector_counts": dict(counts),
                "fallback_count": sum(d.get("fallback_used") is True for d in decisions), "unknown_count": unknown,
                "unavailable_count": sum(d.get("selector_status") == "unavailable" for d in decisions)}
            if unknown or not decisions or any(d.get("fallback_used") is True or d.get("selector_status") == "unavailable" for d in decisions):
                warnings.append(condition + "_EXECUTION_COVERAGE_OR_FALLBACK_UNVERIFIED")
    else:
        warnings.append("ACTUAL_SELECTOR_FALLBACK_COVERAGE_NOT_RECORDED")
    # Intentionally never promoted by caller flags or self-declared Gold.
    checks.append({"code": "PAIRED_RUNNER_AND_METRIC_TRANSLATOR", "status": "BLOCKED"})
    return {"schema_version": "locdi.ablation_fairness.v1", "status": "BLOCKED", "checks": checks,
        "comparisons": pairs, "conditions": per_condition, "paired_case_identity_status": paired_status,
        "paired_input_order_equal": order_equal, "warnings": sorted(set(warnings)),
        "stages": {"implementation": "READY", "static_design_inspection": "READY",
                   "paired_offline_execution": "BLOCKED", "real_llm_execution": "NOT_REVIEWED", "clinical_evaluation": "BLOCKED"},
        "research_validated": False, "clinical_metrics": "NOT_EVALUATED"}
