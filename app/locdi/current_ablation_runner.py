"""Pinned current-version B0..B4 execution and independent trace aggregation.

Clinical research admission is delegated to the unchanged existing gate.
Development Source agreement is explicitly separate from official evaluation.
No model implementation, fitting, labels, or transport is manufactured here.
"""
from collections import Counter
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time

from app.locdi.llm_setup import digest
from experiments.run_locdi_ablation import CONDITIONS, FEATURE_KEYS

VERSION = 'locdi.current_paired_ablation.c4_38.v1'
METRICS = 'locdi.source_agreement_and_workflow.c4_38.v1'
PURPOSES = {'development_source_evaluation', 'official_research_evaluation', 'TEST_ONLY'}
REQUIRED_COMMON = {'dataset_sha256', 'patient_split_sha256', 'model_bindings',
    'registry_sha256', 'prompt_sha256', 'decoding', 'seed', 'memory_source',
    'memory_sha256', 'action_catalog_sha256', 'diagnosis_catalog', 'source_visibility',
    'time_contract', 'reference_policy', 'minute_limit', 'repeats', 'request_budget', 'resource_budget'}
HIDDEN = {'gold', 'evaluation', 'task_annotations', 'reference', 'reference_diagnosis',
          'correctness', 'accepted_diagnoses', 'accepted_departments'}
_BUDGET = ContextVar('locdi_current_ablation_budget', default=None)


class CurrentAblationBlocked(ValueError):
    pass


def _no_reference(value):
    if isinstance(value, dict):
        if HIDDEN & {str(k).casefold() for k in value}:
            raise CurrentAblationBlocked('REFERENCE_IN_EXECUTION_INPUT')
        for item in value.values():
            _no_reference(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _no_reference(item)
    elif isinstance(value, str) and value.lstrip().startswith(('{', '[')):
        try:
            parsed = json.loads(value)
        except ValueError:
            return
        _no_reference(parsed)


def features_for(condition):
    if condition not in CONDITIONS:
        raise CurrentAblationBlocked('UNKNOWN_CONDITION')
    return dict(zip(FEATURE_KEYS, CONDITIONS[condition]))


def make_contract(common, panel, *, conditions=('B0', 'B1', 'B2'), purpose='development_source_evaluation'):
    """Pin caller-provided real settings; no null substitution or ready mutation."""
    contract = {'version': VERSION, 'purpose': purpose, 'conditions': list(conditions),
        'features': {c: features_for(c) for c in conditions}, 'common': deepcopy(common),
        'panel_fingerprint': digest(panel), 'registered_cases': len(panel), 'voi_horizon': 2,
        'metrics_version': METRICS, 'stop_on_failure': True,
        'clinical_metrics': 'NOT_EVALUATED' if purpose != 'official_research_evaluation' else 'GATE_REQUIRED'}
    contract['fingerprint'] = digest(contract)
    return contract


def _positive_int(value):
    return type(value) is int and value > 0


def validate_contract(contract, panel):
    if not isinstance(contract, dict) or contract.get('fingerprint') != digest({k: v for k, v in contract.items() if k != 'fingerprint'}):
        raise CurrentAblationBlocked('CONTRACT_FINGERPRINT')
    if contract.get('version') != VERSION or contract.get('metrics_version') != METRICS or contract.get('purpose') not in PURPOSES:
        raise CurrentAblationBlocked('CONTRACT_VERSION_OR_PURPOSE')
    names = contract.get('conditions')
    if not isinstance(names, list) or not names or len(names) != len(set(names)) or any(c not in CONDITIONS for c in names):
        raise CurrentAblationBlocked('CONDITIONS')
    expected_features = {c: features_for(c) for c in names}
    actual_features = contract.get('features')
    if (not isinstance(actual_features, dict) or set(actual_features) != set(expected_features)
            or any(not isinstance(actual_features[c], dict) or set(actual_features[c]) != set(expected_features[c])
                   or any(type(actual_features[c][k]) is not type(v) or actual_features[c][k] != v
                          for k, v in expected_features[c].items()) for c in names)):
        raise CurrentAblationBlocked('OFFICIAL_FEATURE_DRIFT')
    if type(contract.get('voi_horizon')) is not int or contract['voi_horizon'] != 2:
        raise CurrentAblationBlocked('THESIS_H2_REQUIRED')
    if type(contract.get('stop_on_failure')) is not bool:
        raise CurrentAblationBlocked('ERROR_POLICY_REQUIRED')
    common = contract.get('common')
    if not isinstance(common, dict) or REQUIRED_COMMON - set(common) or any(common[k] is None for k in REQUIRED_COMMON):
        raise CurrentAblationBlocked('UNRESOLVED_COMMON_CONTRACT')
    for key in ('dataset_sha256', 'patient_split_sha256', 'registry_sha256', 'prompt_sha256', 'memory_sha256', 'action_catalog_sha256'):
        v = common[key]
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise CurrentAblationBlocked('UNPINNED_' + key.upper())
    if common['memory_source'] != 'fixed' or not _positive_int(common['minute_limit']) or not _positive_int(common['repeats']) or type(common['seed']) is not int:
        raise CurrentAblationBlocked('COMMON_EXECUTION_VALUES')
    if not isinstance(common['model_bindings'], list) or not common['model_bindings']:
        raise CurrentAblationBlocked('MODEL_BINDINGS')
    for b in common['model_bindings']:
        if not isinstance(b, dict) or any(not isinstance(b.get(k), str) or not b[k] for k in ('agent_id', 'model', 'version', 'digest', 'prompt_sha256')):
            raise CurrentAblationBlocked('MODEL_VERSION_DIGEST_PROMPT_BINDING')
        model_digest = b['digest'].removeprefix('sha256:') if hasattr(str, 'removeprefix') else b['digest'].replace('sha256:', '', 1)
        if len(model_digest) != 64 or any(c not in '0123456789abcdef' for c in model_digest):
            raise CurrentAblationBlocked('MODEL_DIGEST')
        if len(b['prompt_sha256']) != 64 or any(c not in '0123456789abcdef' for c in b['prompt_sha256']):
            raise CurrentAblationBlocked('MODEL_PROMPT_DIGEST')
    catalog = common['diagnosis_catalog']
    free_text = (isinstance(catalog, dict) and set(catalog) == {'mode', 'validator_sha256', 'source_label_injected'}
                 and catalog['mode'] == 'existing_free_text' and catalog['source_label_injected'] is False
                 and isinstance(catalog['validator_sha256'], str) and len(catalog['validator_sha256']) == 64
                 and all(c in '0123456789abcdef' for c in catalog['validator_sha256']))
    if not free_text and (not isinstance(catalog, list) or not catalog or any(not isinstance(d, str) or not d for d in catalog) or len(catalog) != len(set(catalog))):
        raise CurrentAblationBlocked('DIAGNOSIS_UNIVERSE')
    for k in ('decoding', 'source_visibility', 'time_contract', 'reference_policy', 'resource_budget'):
        if not isinstance(common[k], dict) or not common[k]:
            raise CurrentAblationBlocked('EXPLICIT_' + k.upper())
    for key in ('wall_seconds', 'output_bytes', 'memory_bytes'):
        value = common['resource_budget'].get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise CurrentAblationBlocked('RESOURCE_CAPS_REQUIRED')
    requests = common['request_budget']
    if (not isinstance(requests, dict) or not _positive_int(requests.get('total'))
            or not _positive_int(requests.get('per_episode'))
            or type(requests.get('retry')) is not int or requests['retry'] != 0
            or type(requests.get('repair')) is not int or requests['repair'] != 0):
        raise CurrentAblationBlocked('EXPLICIT_REQUEST_CAP_RETRY_REPAIR0')
    if not isinstance(panel, list) or not panel or len(panel) != contract.get('registered_cases') or digest(panel) != contract['panel_fingerprint']:
        raise CurrentAblationBlocked('SAME_REGISTERED_PANEL_REQUIRED')
    seen = set()
    for r in panel:
        if not isinstance(r, dict) or any(not isinstance(r.get(k), str) or not r[k] for k in ('case_ref', 'patient_ref')):
            raise CurrentAblationBlocked('PANEL_IDENTITY')
        if r['case_ref'] in seen:
            raise CurrentAblationBlocked('DUPLICATE_CASE')
        seen.add(r['case_ref'])
        if not isinstance(r.get('execution_case'), dict) or not isinstance(r.get('reference'), dict):
            raise CurrentAblationBlocked('PRIVATE_REFERENCE_SEPARATION')
        _no_reference(r['execution_case'])
        reference = r['reference']
        if (not isinstance(reference.get('accepted_diagnoses'), list) or not reference['accepted_diagnoses']
                or any(not isinstance(x, str) or not x for x in reference['accepted_diagnoses'])):
            raise CurrentAblationBlocked('SOURCE_REFERENCE_REQUIRED')
    return common


def execution_readiness(contract, panel, *, artifact_inputs=None):
    """Re-run existing research gate; never trust a caller readiness declaration."""
    validate_contract(contract, panel)
    blockers = []
    gate = None
    adaptive = any(c in {'B3', 'B4'} for c in contract['conditions'])
    official = contract['purpose'] == 'official_research_evaluation'
    if official or adaptive:
        from validation.research_artifact_gate import assess_artifacts
        args = deepcopy(artifact_inputs or {})
        registry = args.pop('registry_manifest', None)
        try:
            gate = assess_artifacts(registry, **args)
        except (TypeError, ValueError, KeyError, AttributeError):
            blockers.append('EXISTING_RESEARCH_GATE_INPUT_INVALID')
        if official and not (gate and gate.get('research_evaluation_ready') is True):
            blockers.append('EXISTING_RESEARCH_GATE_BLOCKED')
        if adaptive and not (gate and gate['artifacts']['routing']['status'] == 'READY'):
            blockers.append('ROUTING_UTILITY_NOT_ADMITTED')
        if 'B4' in contract['conditions'] and not (gate and gate['artifacts']['voi']['status'] == 'READY'):
            blockers.append('VOI_RESEARCH_ARTIFACT_NOT_ADMITTED')
    if contract['purpose'] != 'TEST_ONLY' and contract['common']['source_visibility'].get('basis') == 'RECORDED_TIME_PROXY' and contract['common']['source_visibility'].get('accepted_development_assumption') is not True:
        blockers.append('RECORDED_TIME_PROXY_NOT_ACCEPTED')
    return {'status': 'BLOCKED' if blockers else 'CONTRACT_READY_REQUIRES_EXECUTION_APPROVAL',
        'blocker_codes': blockers, 'existing_gate': gate,
        'clinical_metrics': 'NOT_EVALUATED' if not official else 'GATE_REQUIRED',
        'official_condition_features_changed': False}


class GenerationBudget:
    def __init__(self, total, per_episode, resource_guard=None):
        self.limit, self.per_episode = total, per_episode
        self.used = self.episode_used = 0
        self.resource_guard = resource_guard
    def reserve(self):
        if self.resource_guard is not None:
            self.resource_guard()
        if self.used >= self.limit or self.episode_used >= self.per_episode:
            raise CurrentAblationBlocked('REQUEST_BUDGET_EXHAUSTED')
        self.used += 1
        self.episode_used += 1


def reserve_generation():
    """A real generation adapter must call this immediately before every request."""
    value = _BUDGET.get()
    if value is None:
        raise CurrentAblationBlocked('NO_ACTIVE_GENERATION_BUDGET')
    value.reserve()


def _process_private_bytes():
    if os.name == 'nt':
        import ctypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise CurrentAblationBlocked('PROCESS_MEMORY_UNOBSERVABLE')
        return counters.PrivateUsage
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024)


def _atomic_json(path, value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2).encode('utf-8')
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(temporary, path)  # Exclusive, atomic publication; never replace.
    finally:
        Path(temporary).unlink()


def _validate_result(result, budget_delta):
    if not isinstance(result, dict) or result.get('status') not in {'completed', 'incomplete', 'failed'}:
        raise CurrentAblationBlocked('BACKEND_STATUS_CONTRACT')
    if type(result.get('request_count')) is not int or result['request_count'] != budget_delta:
        raise CurrentAblationBlocked('UNMETERED_GENERATION_OR_REQUEST_COUNT')
    actions = result.get('action_counts')
    if not isinstance(actions, dict) or set(actions) != {'Ask', 'Test', 'Consult', 'Diagnose'} or any(type(v) is not int or v < 0 for v in actions.values()):
        raise CurrentAblationBlocked('EXECUTED_ACTION_COUNTS_REQUIRED')
    prediction = result.get('prediction')
    if not isinstance(prediction, dict) or prediction.get('certainty') not in {'supported', 'unresolved', 'not_available'}:
        raise CurrentAblationBlocked('PREDICTION_CONTRACT')
    if prediction['certainty'] == 'unresolved' and prediction.get('diagnosis') not in (None, ''):
        raise CurrentAblationBlocked('UNRESOLVED_NOT_A_SUPPORTED_DIAGNOSIS')
    for key in ('diagnosis', 'department', 'algorithmic_diagnosis_choice'):
        if prediction.get(key) is not None and not isinstance(prediction[key], str):
            raise CurrentAblationBlocked('PREDICTION_TEXT_OR_NULL')
    json.dumps(result, allow_nan=False)


def _ratio(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None,
            'status': 'DEFINED' if denominator else 'NOT_APPLICABLE'}


def aggregate_saved_rows(rows, contract, panel):
    """Reference scoring is post-execution only; unknown != incorrect."""
    validate_contract(contract, panel)
    reference = {r['case_ref']: r['reference'] for r in panel}
    expected = {(c, r['case_ref'], k) for c in contract['conditions'] for r in panel
                for k in range(contract['common']['repeats'])}
    actual = [(r['condition'], r['case_ref'], r['repeat']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise CurrentAblationBlocked('INCOMPLETE_OR_DUPLICATE_PAIRED_TRACE')
    summary = {}
    for condition in contract['conditions']:
        rs = [r for r in rows if r['condition'] == condition]
        states = Counter(r['status'] for r in rs)
        correct = scorable = unresolved = algorithmic_matches = algorithmic_count = 0
        dept_correct = dept_scorable = dept_reference = 0
        memory_selected = memory_hits = memory_relevant = memory_cases = 0
        source_selected = source_hits = source_relevant = source_cases = 0
        actions = Counter(); requests = 0; routing_calls = abstentions = 0
        unknown_action_cases = 0; recorded_routing_cases = 0
        voi_values = []; eig_values = []; planner_calls = states_evaluated = 0
        wall = []
        for r in rs:
            ref = reference[r['case_ref']]; result = r['result']; p = result.get('prediction', {})
            if p.get('certainty') == 'unresolved':
                unresolved += 1
            if r['status'] == 'completed' and p.get('certainty') == 'supported' and p.get('diagnosis'):
                scorable += 1; correct += p['diagnosis'] in ref['accepted_diagnoses']
            if r['status'] == 'completed' and p.get('algorithmic_diagnosis_choice'):
                algorithmic_count += 1; algorithmic_matches += p['algorithmic_diagnosis_choice'] in ref['accepted_diagnoses']
            departments = ref.get('accepted_departments')
            if isinstance(departments, list) and departments:
                dept_reference += 1
                if r['status'] == 'completed' and p.get('department'):
                    dept_scorable += 1; dept_correct += p['department'] in departments
            if isinstance(result.get('action_counts'), dict):
                actions.update(result['action_counts'])
            else:
                unknown_action_cases += 1
            requests += result.get('request_count', 0)
            if type(result.get('routing_call_count')) is int and type(result.get('routing_abstention_count')) is int:
                recorded_routing_cases += 1
                routing_calls += result['routing_call_count']; abstentions += result['routing_abstention_count']
            selected = result.get('memory_selected_ids')
            if isinstance(selected, list):
                if len(selected) != len(set(selected)):
                    raise CurrentAblationBlocked('DUPLICATE_SELECTED_MEMORY')
                for name in ['memory_annotation', 'source_identity_annotation']:
                    annotation = ref.get(name)
                    if isinstance(annotation, dict) and isinstance(annotation.get('relevant_ids'), list):
                        gold = set(annotation['relevant_ids']); hit = len(set(selected) & gold)
                        if name == 'memory_annotation':
                            memory_cases += 1; memory_selected += len(selected); memory_hits += hit; memory_relevant += len(gold)
                        else:
                            source_cases += 1; source_selected += len(selected); source_hits += hit; source_relevant += len(gold)
            for event in result.get('voi_trace', []):
                if type(event.get('voi')) in (int, float): voi_values.append(event['voi'])
                if type(event.get('eig_nats')) in (int, float): eig_values.append(event['eig_nats'])
            computation = result.get('computation', {})
            planner_calls += computation.get('planner_calls', 0); states_evaluated += computation.get('evaluated_states', 0)
            if type(computation.get('wall_seconds')) in (int, float): wall.append(computation['wall_seconds'])
        summary[condition] = {'registered': len(rs), 'states': dict(states),
            'completed': states['completed'], 'incomplete': states['incomplete'], 'failed': states['failed'],
            'not_attempted': states['not_attempted'], 'unresolved': unresolved,
            'source_diagnosis_correct_supported_count': correct,
            'correct_supported_yield_registered': _ratio(correct, len(rs)),
            'supported_answer_agreement': _ratio(correct, scorable),
            'unknown_unresolved_failure_not_imputed_as_correctness_false': True,
            'algorithmic_choice_agreement': _ratio(algorithmic_matches, algorithmic_count),
            'final_department_agreement': _ratio(dept_correct, dept_scorable),
            'department_correct_yield_reference_cases': _ratio(dept_correct, dept_reference),
            'executed_action_counts_recorded_sum': dict(actions), 'unknown_action_count_cases': unknown_action_cases,
            'mean_executed_actions_registered': sum(actions.values()) / len(rs) if not unknown_action_cases else None,
            'generation_count': requests, 'routing_call_count_recorded_sum': routing_calls,
            'routing_abstention_count_recorded_sum': abstentions, 'routing_recorded_cases': recorded_routing_cases,
            'memory_annotation_metrics': {'cases': memory_cases, 'precision': _ratio(memory_hits, memory_selected), 'recall': _ratio(memory_hits, memory_relevant)},
            'source_identity_metrics': {'cases': source_cases, 'precision': _ratio(source_hits, source_selected), 'recall': _ratio(source_hits, source_relevant)},
            'VoI_mean_recorded': sum(voi_values) / len(voi_values) if voi_values else None,
            'EIG_mean_recorded_nats': sum(eig_values) / len(eig_values) if eig_values else None,
            'planner_calls_recorded': planner_calls, 'evaluated_states_recorded': states_evaluated,
            'wall_seconds_recorded': sum(wall) if wall else None,
            'clinical_metrics': 'NOT_EVALUATED' if contract['purpose'] != 'official_research_evaluation' else 'GATE_DEPENDENT'}
    return {'version': METRICS, 'conditions': summary, 'same_panel_all_conditions': True,
        'registered_patients': len({r['patient_ref'] for r in panel}), 'registered_cases': len(panel),
        'purpose': contract['purpose'], 'not_official_when_development': contract['purpose'] != 'official_research_evaluation'}


def reaggregate_trace(path, contract, panel):
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('row_fingerprint') != digest({k: v for k, v in row.items() if k != 'row_fingerprint'}):
                raise CurrentAblationBlocked('TRACE_ROW_FINGERPRINT')
            if row.get('contract_fingerprint') != contract['fingerprint']:
                raise CurrentAblationBlocked('TRACE_CONTRACT_BINDING')
            rows.append(row)
    return aggregate_saved_rows(rows, contract, panel)


def run_paired_ablation(contract, panel, *, run_one=None, execute=False, approval_record=None,
                        public_root=None, private_root=None, artifact_inputs=None):
    ready = execution_readiness(contract, panel, artifact_inputs=artifact_inputs)
    if not execute:
        return {'status': 'DRY_RUN', 'readiness': ready, 'generation_count': 0,
                'planned_episodes': len(panel) * len(contract['conditions']) * contract['common']['repeats']}
    if ready['status'] == 'BLOCKED':
        raise CurrentAblationBlocked(','.join(ready['blocker_codes']))
    from app.locdi.new_cohort_execution import approval
    scope = 'C4_38_CURRENT_ABLATION_OFFICIAL' if contract['purpose'] == 'official_research_evaluation' else 'C4_38_CURRENT_ABLATION_DEVELOPMENT'
    approval(approval_record, scope, contract['fingerprint'])
    if not callable(run_one) or public_root is None or private_root is None:
        raise CurrentAblationBlocked('EXECUTION_BACKEND_AND_SEPARATE_OUTPUTS_REQUIRED')
    pub, private = Path(public_root).resolve(), Path(private_root).resolve()
    if pub == private or pub in private.parents or private in pub.parents:
        raise CurrentAblationBlocked('PUBLIC_PRIVATE_OUTPUT_SEPARATION')
    if pub.exists() or private.exists():
        raise CurrentAblationBlocked('NO_OVERWRITE_NO_RESUME')
    pub.mkdir(parents=True); private.mkdir(parents=True)
    _atomic_json(private / 'contract.json', contract); _atomic_json(private / 'approval.json', approval_record)
    _atomic_json(private / 'started.json', {'at': datetime.now(timezone.utc).isoformat(), 'contract': contract['fingerprint']})
    common = contract['common']; requests = common['request_budget']
    budget = GenerationBudget(requests['total'], requests['per_episode'])
    begin = time.monotonic(); caps = common['resource_budget']
    def guard():
        if time.monotonic() - begin >= caps['wall_seconds']:
            raise CurrentAblationBlocked('WALL_BUDGET_EXHAUSTED')
        if _process_private_bytes() >= caps['memory_bytes']:
            raise CurrentAblationBlocked('MEMORY_BUDGET_EXHAUSTED')
        size = sum(p.stat().st_size for p in private.rglob('*') if p.is_file())
        size += sum(p.stat().st_size for p in pub.rglob('*') if p.is_file())
        if size >= caps['output_bytes']:
            raise CurrentAblationBlocked('OUTPUT_BUDGET_EXHAUSTED')
    budget.resource_guard = guard
    stopped = False; rows = []
    with (private / 'episodes.jsonl').open('x', encoding='utf-8') as f:
        for condition in contract['conditions']:
            for repeat in range(common['repeats']):
                for ordinal, entry in enumerate(panel, 1):
                    features = features_for(condition)
                    result = {'prediction': {'certainty': 'not_available', 'diagnosis': None, 'department': None},
                              'action_counts': {'Ask': 0, 'Test': 0, 'Consult': 0, 'Diagnose': 0}, 'request_count': 0}
                    status = 'not_attempted'
                    if not stopped:
                        before = budget.used; budget.episode_used = 0
                        token = _BUDGET.set(budget)
                        try:
                            guard()
                            result = run_one(deepcopy(entry['execution_case']), deepcopy(features))
                            guard()
                            _validate_result(result, budget.used - before)
                            status = result['status']
                            if status == 'failed' and contract['stop_on_failure']: stopped = True
                        except Exception as exc:
                            # Preserve successful earlier episodes. A backend failure is
                            # never a clinical false label or a synthetic supported reply.
                            status = 'failed'; stopped = True
                            rejected = result
                            try:
                                json.dumps(rejected, allow_nan=False)
                            except (TypeError, ValueError):
                                rejected = None
                            result = {'status': 'failed', 'error_type': type(exc).__name__,
                                'error_code': str(exc) if isinstance(exc, CurrentAblationBlocked) else None,
                                'request_count': budget.used - before,
                                'prediction': {'certainty': 'not_available', 'diagnosis': None, 'department': None},
                                'action_counts': None, 'action_count_status': 'UNKNOWN_BACKEND_ERROR',
                                'partial_backend_result_not_invented': True,
                                'rejected_backend_result_private': rejected}
                        finally:
                            _BUDGET.reset(token)
                    row = {'condition': condition, 'case_ref': entry['case_ref'], 'case_ordinal': ordinal,
                        'patient_ref': entry['patient_ref'], 'repeat': repeat, 'status': status, 'result': result,
                        'execution_input_fingerprint': digest(entry['execution_case']), 'feature_fingerprint': digest(features),
                        'contract_fingerprint': contract['fingerprint']}
                    row['row_fingerprint'] = digest(row)
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
                    rows.append(row)
    metrics = aggregate_saved_rows(rows, contract, panel)
    independent = reaggregate_trace(private / 'episodes.jsonl', contract, panel)
    if independent != metrics:
        raise CurrentAblationBlocked('INDEPENDENT_REAGGREGATION_MISMATCH')
    _atomic_json(pub / 'metrics.json', metrics)
    _atomic_json(pub / 'execution_manifest.json', {'version': VERSION, 'status': 'PARTIAL' if stopped else 'COMPLETED',
        'contract_fingerprint': contract['fingerprint'], 'trace_sha256': __import__('hashlib').sha256((private / 'episodes.jsonl').read_bytes()).hexdigest(),
        'registered_episodes': len(rows), 'generation_count': budget.used, 'independent_reaggregation': 'EXACT_MATCH',
        'resource_enforcement': 'before/after each backend episode and before every generation reservation; single bounded call/write may overshoot',
        'clinical_metrics': 'NOT_EVALUATED', 'official_condition_definitions_changed': False})
    return {'status': 'PARTIAL' if stopped else 'COMPLETED', 'metrics': metrics,
            'generation_count': budget.used, 'trace_reaggregation': 'EXACT_MATCH'}
