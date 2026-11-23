"""Explicitly authorized, single-case ER development pilot; default static only.

This boundary never upgrades the offline integration client or clinical artifacts.
The existing real adapter owns transport, identity checks and JSON validation.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re

from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.episode_exporter import ExportSession, build_episode, export_summary, _write
from app.locdi.llm_setup import ROOT, digest, load_setup, validate_setup
from app.locdi.ollama_adapter import LocalOllamaAdapter, LLMRequestError
from app.locdi.run_simulation import SimulationOptions, run_case, _fixed_memory

DATASET = ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl'
PILOT_ROOT = ROOT / 'outputs/development_pilot'
FEATURES = {'memory_mode': 'selective', 'top_k': 5, 'routing_mode': 'er_only',
            'primary_router': 'rule_similarity', 'voi_enabled': False,
            'action_selector': 'specialist_heuristic'}
MAX_REQUESTS = 5


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def preflight(*, run_id, config=None, mode='er_only'):
    """Read local files only. No server probe, client, directory or write."""
    report = {'schema_version': 'locdi.live_preflight.v1', 'status': 'dry_run',
        'execution_class': 'DEVELOPMENT_PILOT', 'run_id': run_id, 'case_ref': 'Case001',
        'mode': mode, 'request_count': 0, 'max_requests': MAX_REQUESTS,
        'server_availability': 'NOT_CHECKED', 'server_model_identity': 'NOT_CHECKED',
        'clinical_metrics': 'NOT_EVALUATED', 'research_evaluation_ready': False,
        'blocker_codes': [], 'plan': None, 'plan_fingerprint': None}
    if not isinstance(run_id, str) or not re.fullmatch(r'[0-9a-f]{32}', run_id):
        report['blocker_codes'].append('EXPLICIT_32_HEX_RUN_ID_REQUIRED')
        return report
    if mode != 'er_only':
        report['blocker_codes'].append({
            'specialty': 'SPECIALTY_RUNTIME_APPROVAL_NOT_IMPLEMENTED',
            'adaptive': 'REVIEWED_ROUTING_UTILITY_REQUIRED',
            'adaptive_voi': 'REVIEWED_ROUTING_AND_VOI_ARTIFACTS_REQUIRED',
            'research': 'RESEARCH_EXECUTION_BLOCKED'}.get(mode, 'UNSUPPORTED_MODE'))
        return report
    destination = PILOT_ROOT / run_id
    if (destination.resolve().parent != PILOT_ROOT.resolve()
            or not PILOT_ROOT.resolve().is_relative_to((ROOT / 'outputs').resolve())):
        report['blocker_codes'].append('OUTPUT_PATH_INVALID')
        return report
    if destination.exists():
        report['blocker_codes'].append('OUTPUT_ALREADY_EXISTS')
    try:
        config = deepcopy(load_setup() if config is None else config)
        # Pilot-specific in-memory budget; original configuration is preserved.
        config['max_requests'] = MAX_REQUESTS
        setup = validate_setup(config)
        report['static_readiness'] = setup['static_readiness']
        report['checks'] = setup['checks']
        if setup['static_readiness'] != 'READY':
            report['blocker_codes'].append('STATIC_SETUP_NOT_READY')
            return report
        prepared = setup['prepared']
        if (config['model'] != 'qwen2.5:1.5b-instruct' or config['request_retries'] != 0
                or len(prepared['registry']['agents']) != 1
                or prepared['registry']['agents'][0]['role'] != 'er'
                or 'test' in config['model_version'].lower()):
            report['blocker_codes'].append('EXISTING_ER_BINDING_REQUIRED')
            return report
        case = load_cases(DATASET)[0]
        context = adapt_case(case, registry_manifest=prepared['registry'])
        _fixed_memory([context], None)
        report['dataset_validation'] = {
            'source_review' if r.validator == 'provenance' else r.validator: r.status
            for r in context.validation_results}
        plan = {'schema_version': 'locdi.live_plan.v1', 'run_id': run_id,
            'case_ref': 'Case001', 'execution_class': 'DEVELOPMENT_PILOT',
            'source_fingerprint': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
            'case_fingerprint': digest(dict(case)), 'features': deepcopy(FEATURES),
            'model': config['model'], 'model_identifier': config['model_identifier'],
            'model_version': config['model_version'], 'settings_fingerprint': prepared['fingerprint'],
            'registry_fingerprint': digest(prepared['registry']),
            'prompt_fingerprints': prepared['prompt_fingerprints'],
            'generation': {k: config[k] for k in ('temperature', 'seed', 'max_output_tokens',
                'timeout_seconds', 'request_retries', 'max_requests', 'minute_limit')},
            'memory_cutoff': 'validated_fixed_past_only',
            'release_policy': 'dataset_declared_proxy_requires_explicit_acceptance',
            'output_namespace': 'outputs/development_pilot/' + run_id}
        report.update(plan=plan, plan_fingerprint=digest(plan))
    except Exception:
        report['blocker_codes'].append('PREFLIGHT_INPUT_CONTRACT_ERROR')
    return report


class PilotOllamaAdapter(LocalOllamaAdapter):
    """Fail closed on audit storage errors; count all calls including repairs.

    Journals contain only fingerprints, times and actual adapter counters.
    A successful JSON response is never an independent correctness judgment.
    """
    def __init__(self, config, *, authorized=False, output_root):
        super().__init__(config, authorized=authorized)
        if self.prepared['config']['max_requests'] != MAX_REQUESTS:
            raise ValueError('Pilot budget must be five')
        self.output_root = output_root
        self.attempt = 0
        self.fatal = False

    def _checkpoint(self, phase, **fields):
        document = {'schema_version': 'locdi.pilot_checkpoint.v1',
                    'attempt': self.attempt, 'phase': phase, 'at': _now(),
                    'llm_execution': self.public_metadata(), **fields}
        try:
            _write(document, 'private/{:03d}_{}.json'.format(self.attempt, phase),
                   self.output_root, 'private')
        except Exception:
            self.fatal = True
            self.errors.append('PilotStorageError')
            raise LLMRequestError('Pilot audit storage failed; further requests blocked') from None

    def chat_json(self, system_prompt, user_prompt, model=None, *, schema=None):
        if self.fatal:
            raise LLMRequestError('Pilot stopped after a previous failure')
        if _sha(system_prompt) not in self.prepared['prompt_fingerprints'].values():
            self.fatal = True
            raise LLMRequestError('Pilot prompt identity mismatch')
        self.attempt += 1
        self._checkpoint('before', system_fingerprint=_sha(system_prompt),
                         input_fingerprint=_sha(user_prompt))
        try:
            response = super().chat_json(system_prompt, user_prompt, model=model, schema=schema)
        except Exception:
            self.fatal = True
            self._checkpoint('error')
            raise
        self._checkpoint('after', response_fingerprint=_sha(response))
        return response


def run_pilot(*, run_id, config=None, mode='er_only', execute=False, allow_llm=False,
              accept_release_assumption=False, expected_plan_fingerprint=None):
    """Only the explicit all-true approval branch can construct a live client."""
    if (any(type(v) is not bool for v in (execute, allow_llm, accept_release_assumption))
            or execute and not (allow_llm and accept_release_assumption)):
        return {'status': 'blocked', 'blocker_codes': ['EXPLICIT_LIVE_APPROVAL_REQUIRED'],
                'request_count': 0, 'research_evaluation_ready': False}
    report = preflight(run_id=run_id, config=config, mode=mode)
    if not execute:
        return report
    if report['blocker_codes']:
        report['status'] = 'blocked'
        return report
    if expected_plan_fingerprint != report['plan_fingerprint']:
        report.update(status='blocked', blocker_codes=['APPROVED_PLAN_FINGERPRINT_MISMATCH'])
        return report
    root = PILOT_ROOT / run_id
    client = None
    result = {'schema_version': 'locdi.live_pilot.v1', 'execution_class': 'DEVELOPMENT_PILOT',
        'run_id': run_id, 'case_ref': 'Case001', 'status': 'failed', 'core_status': None,
        'started_at': _now(), 'ended_at': None, 'plan': report['plan'],
        'plan_fingerprint': report['plan_fingerprint'], 'blocker_codes': [],
        'llm_execution': None, 'files': {}, 'clinical_metrics': 'NOT_EVALUATED',
        'research_evaluation_ready': False, 'clinical_reference_ready': False}
    try:
        root.mkdir(parents=True, exist_ok=False)
    except Exception:
        result['blocker_codes'] = ['OUTPUT_RESERVATION_FAILED']
        result['ended_at'] = _now()
        return result
    try:
        _write(report, 'reports/preflight.json', root, 'reports')
        _write({'schema_version': 'locdi.pilot_started.v1', 'run_id': run_id,
                'started_at': result['started_at'], 'status': 'started',
                'plan_fingerprint': report['plan_fingerprint']}, 'reports/started.json', root, 'reports')
        # Recheck local files after reserving storage, still before client creation.
        current_config = deepcopy(load_setup() if config is None else config)
        current_config['max_requests'] = MAX_REQUESTS
        setup = validate_setup(current_config)
        if (setup['static_readiness'] != 'READY'
                or setup['prepared']['fingerprint'] != report['plan']['settings_fingerprint']):
            raise ValueError('Approved setup changed')
        case = load_cases(DATASET)[0]
        if (digest(dict(case)) != report['plan']['case_fingerprint']
                or hashlib.sha256(DATASET.read_bytes()).hexdigest() != report['plan']['source_fingerprint']):
            raise ValueError('Approved source changed')
        client = PilotOllamaAdapter(current_config, authorized=True, output_root=root)
        if not isinstance(client, LocalOllamaAdapter):
            raise ValueError('Real adapter required')
        options = SimulationOptions(purpose='development_llm', allow_llm=True, dry_run=False,
            client=client, llm_config=current_config, minute_limit=current_config['minute_limit'],
            registry_manifest=setup['prepared']['registry'], runner_options=deepcopy(FEATURES),
            max_retries=0, accept_recorded_release_assumption=True)
        raw = run_case(case, options=options)
        result['core_status'] = raw['status']
        result['llm_execution'] = client.public_metadata()
        context = adapt_case(case, registry_manifest=setup['prepared']['registry'])
        episode = build_episode(raw, context=context, session=ExportSession(run_id=run_id))
        _write(episode, 'episodes/Case001.json', root, 'episodes')
        result['files']['episode'] = 'episodes/Case001.json'
        export_summary({'episodes': [episode]}, 'metrics/summary.jsonl', output_root=root)
        result['files']['summary'] = 'metrics/summary.jsonl'
        audit = {'schema_version': 'locdi.live_runtime_audit.v1',
            'execution_class': 'DEVELOPMENT_PILOT', 'run_id': run_id,
            'configuration_fingerprint': episode['configuration_fingerprint'],
            'workflow': episode['workflow'], 'memory': episode['memory'],
            'capabilities': episode['capabilities'], 'roles': episode['decision_audit'],
            'terminal': episode['result'], 'clinical_metrics': 'NOT_EVALUATED',
            'diagnosis_correctness': 'NOT_REVIEWED', 'abstention_appropriateness': 'NOT_REVIEWED',
            'routing_success': 'NOT_REVIEWED'}
        _write(audit, 'reports/runtime.json', root, 'reports')
        result['files']['audit'] = 'reports/runtime.json'
        result['status'] = raw['status']
        if client.fatal:
            result['status'] = 'failed'
            result['blocker_codes'].append('CLIENT_OR_STORAGE_STOP')
        if 'LLMRequestLimitError' in client.errors:
            result['blocker_codes'].append('REQUEST_LIMIT_REACHED')
    except Exception:
        result['status'] = 'failed'
        result['blocker_codes'].append('PILOT_EXECUTION_OR_EXPORT_FAILED')
    finally:
        if client is not None:
            result['llm_execution'] = client.public_metadata()
            client.close()
        result['ended_at'] = _now()
    try:
        _write(result, 'reports/pilot.json', root, 'reports')
    except Exception:
        result['status'] = 'failed'
        result['blocker_codes'].append('FINAL_REPORT_STORAGE_FAILED')
    return result
