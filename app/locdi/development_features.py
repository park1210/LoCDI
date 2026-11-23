"""Static feature setup, not Runner authorization. No client/server creation.

Private source events are inspected only by artifact validators, never used as
current observations or returned. External JSON is explicitly opt-in, read-only.
"""
import json
from pathlib import Path

from app.core.agent_registry import AgentRegistry
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT, digest, load_setup, project_file, validate_setup
from validation.artifact_readiness import validate_routing_artifact, validate_voi_artifact

DEFAULT_FEATURE_CONFIG = ROOT / 'configs/development_specialty_llm.json'
ARCHIVE = (ROOT.parent / '001_synthea-docker/experiments/34_live_action_policy/output/live_action_policy_v1').resolve()
MODES = ('er_only', 'adaptive_routing', 'bayesian_voi')
KEYS = {'schema_version', 'ready', 'llm', 'routing_utility_path', 'voi_model_path',
        'diagnosis_catalog_path', 'horizon', 'voi_binding'}


def read_artifact(value):
    """Only explicit JSON files inside project or the named read-only archive."""
    if not isinstance(value, str) or not value:
        raise ValueError('ARTIFACT_PATH_REQUIRED')
    path = (ROOT / value).resolve()
    if path.suffix.lower() != '.json' or not any(path == base or base in path.parents for base in (ROOT, ARCHIVE)):
        raise ValueError('ARTIFACT_PATH_OUTSIDE_ALLOWED_ROOTS')
    return json.loads(path.read_text(encoding='utf-8'))


def check_development_features(mode='er_only', *, config=None):
    """Public sanitized static report. All modes prohibit generation here.

    Specialty config is an envelope; its llm member uses the unchanged exact
    llm_setup schema. Never feed this envelope to the existing execution CLI.
    """
    if mode not in MODES:
        raise ValueError('UNSUPPORTED_MODE')
    report = {'schema_version': 'locdi.development_setup.v1', 'mode': mode,
              'status': 'dry_run', 'ready': False, 'research_readiness': False,
              'server': 'NOT_CHECKED', 'server_model_identity': 'NOT_CHECKED',
              'execution_readiness': 'BLOCKED', 'generation_requests': 0,
              'clinical_metrics': 'NOT_EVALUATED', 'blockers': [],
              'routing_utility': {'status': 'NOT_REQUIRED'}, 'voi_artifact': {'status': 'NOT_REQUIRED'},
              'action_coverage': 'NOT_REQUIRED', 'diagnosis_catalog': 'NOT_REQUIRED',
              'runtime_audit': {k: 'NOT_EXECUTED' for k in ('entry_role', 'effective_model_role',
                  'effective_department', 'model_binding', 'routing_decision_status', 'selected_specialty',
                  'routing_abstained', 'actual_selector', 'fallback_used', 'voi_selection', 'action_execution')}}
    blockers = report['blockers']
    report['requested_features'] = {'memory_mode': 'selective', 'top_k': 5,
        'routing_mode': 'er_only' if mode == 'er_only' else 'adaptive',
        'primary_router': 'rule_similarity' if mode == 'er_only' else 'adaptive',
        'voi_enabled': mode == 'bayesian_voi',
        'action_selector': 'bayesian_voi' if mode == 'bayesian_voi' else 'specialist_heuristic'}
    def load(value, code):
        try:
            return read_artifact(value)
        except (OSError, ValueError, TypeError):
            blockers.append(code)
            return None
    envelope = None
    if mode == 'er_only':
        llm = load_setup()  # Existing ER settings remain authoritative.
    else:
        try:
            envelope = load_setup(DEFAULT_FEATURE_CONFIG) if config is None else config
            if (not isinstance(envelope, dict) or set(envelope) != KEYS or
                    envelope['schema_version'] != 'locdi.development_features.v1' or envelope['ready'] is not False):
                raise ValueError()
            llm = envelope['llm']
        except (OSError, ValueError, KeyError, TypeError):
            report.update(registry='INVALID', model_binding='INVALID', development_readiness='BLOCKED')
            blockers.append('FEATURE_CONFIGURATION_INVALID')
            return report
    setup = validate_setup(llm)
    report['setup_checks'] = setup['checks']
    report['registry'] = setup['checks'].get('registry', {}).get('status', 'INVALID')
    report['model_binding'] = setup['checks'].get('agent_bindings', {}).get('status', 'INVALID')
    if setup['static_readiness'] != 'READY':
        blockers.append('LLM_STATIC_BINDING_NOT_READY')
    manifest = None
    try:
        manifest = AgentRegistry(json.loads(project_file(llm['registry_path']).read_text(encoding='utf-8'))).snapshot()
        specialties = sorted(a['department'] for a in manifest['agents'] if a['role'] == 'specialty')
        report.update(specialties=specialties, registry_fingerprint=digest(manifest))
        if mode == 'er_only' and specialties or mode != 'er_only' and not specialties:
            blockers.append('MODE_REGISTRY_MISMATCH')
    except (OSError, ValueError, KeyError, TypeError):
        blockers.append('REGISTRY_INVALID')
    report['runtime_audit']['requested_selector'] = 'bayesian_voi' if mode == 'bayesian_voi' else 'specialist_heuristic'
    if mode != 'er_only':
        cases = load_cases(ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl')
        patients = [c['patient_id'] for c in cases]
        utility = load(envelope['routing_utility_path'], 'ROUTING_UTILITY_MISSING_OR_INVALID')
        bindings = {a['agent_id']: (a['model'], a['version']) for a in manifest['agents']} if manifest else {}
        routing = validate_routing_artifact(manifest, utility, evaluation_patient_ids=patients, expected_bindings=bindings)
        report['routing_utility'] = routing
        if routing['development_status'] != 'DEVELOPMENT_CONTRACT_READY':
            blockers.append('ROUTING_CONTRACT_BLOCKED')
        if mode == 'bayesian_voi':
            artifact = load(envelope['voi_model_path'], 'VOI_ARTIFACT_MISSING_OR_INVALID')
            catalog = load(envelope['diagnosis_catalog_path'], 'SHARED_DIAGNOSIS_CATALOG_MISSING_OR_INVALID')
            # A catalog must be explicitly supplied, never derived from gold.
            report['diagnosis_catalog'] = 'SUPPLIED' if catalog is not None else 'BLOCKED'
            sources = [e for c in cases for e in c['environment']['current_encounter_events']]
            voi = validate_voi_artifact(artifact, source_actions=sources, evaluation_patient_ids=patients,
                                       expected_horizon=envelope['horizon'], diagnosis_candidates=catalog)
            report['voi_artifact'] = voi
            report['action_coverage'] = next((c['status'] for c in voi['checks'] if c['code'] == 'SOURCE_ACTION_COVERAGE'), 'NOT_CHECKED')
            if type(envelope['horizon']) is not int or envelope['horizon'] not in (2, 3):
                blockers.append('MULTISTEP_HORIZON_INVALID')
            if voi['development_status'] != 'DEVELOPMENT_CONTRACT_READY':
                blockers.append('VOI_CONTRACT_BLOCKED')
            expected = {'artifact_fingerprint': voi['artifact_fingerprint'],
                        'registry_fingerprint': digest(manifest) if manifest else None,
                        'model_identifier': llm.get('model_identifier')}
            # Explicit artifact-to-binding declaration, not inferred from role count.
            if envelope['voi_binding'] != expected or not all(expected.values()):
                blockers.append('VOI_MODEL_BINDING_NOT_CONFIRMED')
            report['voi_model_binding'] = 'BLOCKED' if 'VOI_MODEL_BINDING_NOT_CONFIRMED' in blockers else 'STATIC_DECLARATION_MATCHED_NOT_LIVE_VERIFIED'
    report['development_readiness'] = 'BLOCKED' if blockers else 'STATIC_CONTRACT_READY'
    report['execution_readiness'] = 'BLOCKED' if blockers else 'NOT_CHECKED'
    return report
