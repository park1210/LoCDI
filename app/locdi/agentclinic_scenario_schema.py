"""Separate external OSCE contracts; no LoCDI schema or clinical approval."""
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import json

REPOSITORY = 'SamuelSchmidgall/AgentClinic'
REVISION = 'b6fbe22300e99a267a7ac94eaa465ab552eef741'
SOURCE_SHA256 = 'd91038a2984f21bb1d43edd88c7958d090ef42ba80f5be487b22b903bf3a35ea'
LOCAL_SHA256 = '68dbe4802c58851472a82e053cd98ca5c90331662e019aff45ed57464aae2fbf'
GROUP = 'AGENTCLINIC_EXTERNAL_OSCE'
POLICY_VERSION = 'agentclinic.source_lookup.v1'
POLICY_TEXT = ('OFFLINE doctor chooses one exact catalog target: ASK, EXAMINE, TEST, '
               'or DIAGNOSE with free text. No inference of missing facts, no clinical '
               'minutes, no scoring access. Empty diagnosis terminates unresolved.')


class ScenarioContractError(ValueError):
    """Messages contain fixed codes only, never original values."""


@dataclass(frozen=True)
class Target:
    target_id: str
    label: str
    source_path: Tuple[str, ...]
    value: Any = field(repr=False)
    node_kind: str = 'leaf'


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    source: Dict[str, Any]
    objective: str
    patient_facts: Dict[str, Any] = field(repr=False)
    physical_exams: Dict[str, Any] = field(repr=False)
    test_results: Dict[str, Any] = field(repr=False)
    ask_targets: Tuple[Target, ...] = field(repr=False)
    exam_targets: Tuple[Target, ...] = field(repr=False)
    test_targets: Tuple[Target, ...] = field(repr=False)


@dataclass(frozen=True)
class ScoringKey:
    scenario_id: str
    source_record_fingerprint: str
    correct_diagnosis: str = field(repr=False)


@dataclass(frozen=True)
class ScenarioBundle:
    scenarios: Tuple[Scenario, ...]
    scoring_keys: Tuple[ScoringKey, ...] = field(repr=False)


def validate_record(row):
    if not isinstance(row, dict) or set(row) != {'OSCE_Examination'}:
        raise ScenarioContractError('OSCE_ROOT_FIELDS')
    c = row['OSCE_Examination']
    types = {'Objective_for_Doctor': str, 'Patient_Actor': dict,
             'Physical_Examination_Findings': dict, 'Test_Results': dict,
             'Correct_Diagnosis': str}
    if not isinstance(c, dict) or set(c) != set(types):
        raise ScenarioContractError('OSCE_FIELDS')
    if any(not isinstance(c[k], t) for k, t in types.items()):
        raise ScenarioContractError('OSCE_FIELD_TYPES')
    if not c['Objective_for_Doctor'].strip() or not c['Correct_Diagnosis'].strip():
        raise ScenarioContractError('OSCE_EMPTY_REQUIRED_TEXT')
    actor = c['Patient_Actor']
    required = {'Demographics', 'History', 'Symptoms', 'Past_Medical_History',
                'Social_History', 'Review_of_Systems'}
    allowed = required | {'Current_Medications', 'Medications', 'Drug_History'}
    if not required <= set(actor) or set(actor) - allowed:
        raise ScenarioContractError('PATIENT_FIELDS')
    if not isinstance(actor['Symptoms'], dict):
        raise ScenarioContractError('PATIENT_SYMPTOM_TYPE')
    def private_key(value):
        if isinstance(value, dict):
            return any(k.casefold() in {'correct_diagnosis', 'gold', 'scoring_key', 'evaluation'}
                       or private_key(v) for k, v in value.items())
        return isinstance(value, list) and any(private_key(v) for v in value)
    if any(private_key(c[k]) for k in types if k != 'Correct_Diagnosis'):
        raise ScenarioContractError('PRIVATE_KEY_IN_FACT_STORE')
    try:
        json.dumps(row, allow_nan=False)
    except (TypeError, ValueError):
        raise ScenarioContractError('NONFINITE_OR_NONJSON_SOURCE') from None
    return c
