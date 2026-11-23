"""C4-36 read-only contract audit; no artifact promotion or model estimation."""
from copy import deepcopy
import ast
from pathlib import Path

from app.core.information_outcomes import InformationOutcomes
from app.locdi.llm_setup import digest
from experiments.run_locdi_ablation import (
    CONDITIONS, FEATURE_KEYS, validate_ablation_config, AblationConfigurationError,
)
from validation.artifact_readiness import validate_voi_artifact

HIDDEN = {'gold', 'gold_department', 'gold_diagnosis', 'evaluation', 'task_annotations',
          'source_event_store', 'provenance', 'correctness', 'reference_diagnosis'}


def reject_hidden(value):
    if isinstance(value, dict):
        if HIDDEN & {str(k).casefold() for k in value}:
            raise ValueError('HIDDEN_POLICY_INPUT')
        for item in value.values():
            reject_hidden(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_hidden(item)


def official_contract(config):
    """Inspect the actual official JSON without changing ready or calling a runner."""
    name = config.get('condition')
    if name not in CONDITIONS:
        raise ValueError('UNKNOWN_OFFICIAL_CONDITION')
    expected = dict(zip(FEATURE_KEYS, CONDITIONS[name]))
    exact = (isinstance(config.get('features'), dict) and set(config['features']) == set(expected)
             and all(type(config['features'][k]) is type(v) and config['features'][k] == v
                     for k, v in expected.items()))
    try:
        validate_ablation_config(deepcopy(config))
        config_gate = 'PASS'
    except AblationConfigurationError:
        config_gate = 'BLOCKED'
    common = config.get('common', {})
    return {'condition': name, 'features': deepcopy(config.get('features')),
            'feature_definition_matches': exact, 'declared_ready': config.get('ready'),
            'common': deepcopy(common), 'config_gate': config_gate,
            'unresolved_common_fields': sorted(k for k, v in common.items() if v is None),
            'runner_implemented': False, 'execution_status': 'BLOCKED',
            'blocker_codes': (['OFFICIAL_FEATURE_MISMATCH'] if not exact else []) +
                ['OFFICIAL_EXECUTION_NOT_IMPLEMENTED', 'RESEARCH_READINESS_NOT_VERIFIED'],
            'horizon': common.get('voi_horizon'), 'config_fingerprint': digest(config)}


def memory_voi_compatibility(model, memory_records):
    """Check an unchanged model with/without memory; never invent memory likelihoods."""
    reject_hidden({'longitudinal_memory': {'records': memory_records}})
    before = digest(model)
    outcomes = InformationOutcomes(deepcopy(model))
    empty, matches0 = outcomes.observe({})
    with_memory, matches1 = outcomes.observe({'longitudinal_memory': {'records': deepcopy(memory_records)}})
    memory_actions = [a for a in model['actions'] if a['type'] == 'retrieve_memory']
    gate = validate_voi_artifact(deepcopy(model))
    if digest(model) != before:
        raise ValueError('MODEL_MUTATED')
    return {'InformationOutcomes_schema': 'PASS', 'model_fingerprint': before,
            'observation_encoding': model['provenance'].get('observation_encoding'),
            'memory_action_count': len(memory_actions), 'memory_record_count_checked': len(memory_records),
            'memory_only_belief_changed': empty != with_memory,
            'memory_only_matched_outcomes': len(matches1) - len(matches0),
            'runtime_artifact_gate': gate,
            'memory_likelihood_ready': False,
            'memory_model_binding_exists': bool(memory_actions),
            'integrated_performance_status': 'BLOCKED',
            'blocker_codes': ['NO_VALIDATED_MEMORY_LIKELIHOOD', 'NO_CURRENT_AND_PAST_DEPENDENCE_MODEL',
                              'HOSPITAL_SOURCE_ENCODING_AND_ACTION_BINDING_NOT_CLEARED'],
            'interpretation': 'Memory delivery is distinct from posterior evidence; no prior/posterior reuse.'}


def source_summary(cases, panel, model):
    records = [r for c in cases for v in c['agent_visible']['past_visits'] for r in v['records']]
    events = [e for c in cases for e in c['environment']['current_encounter_events']]
    bindings = {(a['type'], a['target']) for a in model['actions']}
    names = {'Ask': 'ask_patient', 'Test': 'request_test', 'Consult': 'consult_specialty'}
    return {'Original50': {'cases': len(cases), 'patients': len({c['patient_id'] for c in cases}),
                'past_memory_records': len(records), 'past_visits': sum(len(c['agent_visible']['past_visits']) for c in cases),
                'source_event_count': len(events),
                'consult_source_events': sum(e['action'] == 'Consult' for e in events),
                'Final_M5_exact_action_target_matches': sum((names.get(e['action']), e['target']) in bindings for e in events),
                'clinical_reference': 'UNREVIEWED', 'clinical_release_time': 'MISSING_PROXY_ONLY',
                'same_patient_memory_runtime': 'IMPLEMENTED', 'full_integrated_performance': 'BLOCKED'},
            'C4_35': {'patients': len({v['patient_id'] for v in panel}), 'visits': len(panel),
                'complete_menu': all(set(v['values']) == set(model['provenance']['action_key_mapping']) for v in panel),
                'memory_runtime_records_constructed': 0, 'raw_patient_history': 'AVAILABLE_IN_C4_34',
                'validated_raw_to_runtime_memory_projection': 'MISSING',
                'routing_reference': 'MISSING_INDEPENDENT_REVIEW', 'exposure': 'DEVELOPMENT_EXPOSED_AFTER_C4_35',
                'full_integrated_performance': 'BLOCKED'}}


def callable_inventory(root, paths):
    """Exact line/signature from source, including methods; not paper equation numbers."""
    result = {}
    for name in paths:
        p = Path(root) / name
        source = p.read_text(encoding='utf-8')
        tree = ast.parse(source)
        items = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                items.append({'name': node.name, 'line': node.lineno,
                              'signature': '\n'.join(source.splitlines()[node.lineno-1:node.body[0].lineno-1])})
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        items.append({'name': node.name + '.' + child.name, 'line': child.lineno,
                            'signature': '\n'.join(source.splitlines()[child.lineno-1:child.body[0].lineno-1])})
        result[name] = items
    return result
