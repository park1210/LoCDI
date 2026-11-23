"""FINAL-EXP: exclusive preparation; newly approved bounded live execution.

No default generation, no resume, no model download/server changes. Core and
all previous namespaces are read-only. Python 3.9 compatible.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_research_gaps import read, rows, sha, write, check
from app.locdi.llm_setup import digest, load_setup, inspect_local_model, validate_setup
from app.locdi.dataset_loader import load_cases
from app.locdi.current_ablation_runner import make_contract, run_paired_ablation, GenerationBudget, _BUDGET, _process_private_bytes
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient, make_er_backend, input_guard, _strict_object
from app.locdi.final_diagnosis_evaluation import diagnosis_match, normalized
from app.locdi.answer_grounded_compact import compact_binding, schema, score
from app.locdi.answer_grounded_routing import model_input, validate_pair
from app.locdi.final_3b_routing_evaluation import complete_groups, fit_source_utility, evaluate_policies

PUB = ROOT / 'outputs/research_evaluation/final_exp_v1'
PR = ROOT / 'outputs/private/research_evaluation/final_exp_v1'
OLD = ROOT / 'outputs/private/research_evaluation/c4_39_v1'
DATA = ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl'
MODEL3 = 'qwen2.5:3b'
DIGEST3 = 'sha256:357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b'
SCOPE = 'FINAL_EXP_PINNED_DEVELOPMENT_ONLY'
GIB = 1024**3


class SharedIntegrityError(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def metadata(endpoint):
    with urllib.request.urlopen('http://127.0.0.1:11434/api/' + endpoint, timeout=10) as response:
        raw = response.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError('METADATA_SIZE_LIMIT')
        return json.loads(raw)


def csv_file(path, values, columns):
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(values)


def freeze_config(model, identifier, suffix):
    config = load_setup()
    registry = read(ROOT / config['registry_path'])
    registry['agents'][0].update(model=model, version='final-exp-' + suffix)
    relative = 'outputs/research_evaluation/final_exp_v1/er_registry_' + suffix + '.json'
    write(ROOT / relative, registry)
    config.update(model=model, model_identifier=identifier, model_version='final-exp-' + suffix,
                  allowed_models=[model], registry_path=relative)
    for b in config['agent_bindings']:
        b['model_identifier'] = identifier
    if validate_setup(config)['static_readiness'] != 'READY':
        raise ValueError('NEW_MODEL_STATIC_BINDING_BLOCKED')
    write(PR / ('llm_config_' + suffix + '.json'), config)
    return config


def duplicate_clusters(questions):
    """Conservative source-only components. No answer or model output consulted.

    Original source group, full text, full Patient Actor, or >=.85 word 3-gram
    Jaccard joins a component. Patient identity beyond provided source is UNKNOWN.
    """
    parent = list(range(len(questions)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    features = []
    for q in questions:
        text = normalized(q['input']['question']); words = text.split()
        try:
            actor = normalized(json.dumps(json.loads(q['input']['question'])['Patient_Actor'], sort_keys=True))
        except (KeyError, ValueError, TypeError):
            actor = None
        features.append((text, actor, {tuple(words[n:n+3]) for n in range(max(0, len(words)-2))}))
    for i, q in enumerate(questions):
        for j in range(i):
            a, b = features[i], features[j]
            sim = len(a[2] & b[2]) / len(a[2] | b[2]) if a[2] | b[2] else 0
            if (q['case_group'] == questions[j]['case_group'] or a[0] == b[0]
                    or (a[1] and a[1] == b[1]) or sim >= .85):
                parent[root(i)] = root(j)
    return [root(i) for i in range(len(questions))]


def prepare():
    if PUB.exists() or PR.exists():
        raise ValueError('FRESH_NAMESPACE_REQUIRED_NO_OVERWRITE')
    tags = metadata('tags'); version = metadata('version'); ps = metadata('ps')
    matches = [r for r in tags['models'] if r['name'] == MODEL3]
    if len(matches) != 1 or 'sha256:' + matches[0]['digest'] != DIGEST3:
        raise ValueError('INSTALLED_3B_PIN_MISMATCH')
    local = inspect_local_model(MODEL3)
    if local['status'] != 'READY' or local['model_identifier'] != DIGEST3:
        raise ValueError('LOCAL_MANIFEST_MISMATCH')
    # Freeze protected inputs before creating new evaluation data.
    protected = list((ROOT / 'app/core').glob('*.py'))
    protected += list((ROOT / 'docs/연구 평가/FINAL_THESIS').glob('*.md'))
    protected += list((ROOT / 'outputs/research_evaluation/c4_39_v1').glob('*.*'))
    protected += [DATA, ROOT / 'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json']
    protected += list((ROOT / 'configs').glob('*.json')) + list((ROOT / 'prompts').rglob('*.txt'))
    protected += list((OLD / 'c4_39_memory_b0_b2_001').rglob('*.json'))
    protected += [OLD / 'c4_39_memory_b0_b2_001/episodes.jsonl']
    pins = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.is_file()}
    write(PR / 'protected_input_hashes.json', pins)
    write(PUB / 'local_model_inventory.json', {'observed_at': now(), 'server': version,
        'models': [m for m in tags['models'] if m['name'] in (MODEL3, 'qwen2.5:1.5b-instruct')],
        'loaded_models': ps, 'generation_requests': 0})
    write(PUB / 'qwen3b_model_manifest.json', {'model': matches[0], 'local_manifest_check': local,
        'maximum_context': 32768, 'actual_context': 'NOT_CHECKED_MODEL_UNLOADED',
        'weights_content_hash': 'NOT_REHASHED; existing manifest and referenced blob sizes verified'})
    import shutil
    disk = shutil.disk_usage(ROOT)
    import psutil
    write(PUB / 'model_resource_preflight.json', {'observed_at': now(), 'system_RAM': dict(psutil.virtual_memory()._asdict()),
        'disk_free_bytes': disk.free, 'actual_context': 'NOT_CHECKED',
        'gpu': 'RTX 4050 Laptop GPU; 6141 MiB total, 0 MiB used at read-only nvidia-smi observation',
        'requested_context': 32768, 'CPU_offload_permitted': True,
        'VRAM_fit_guaranteed': False, 'post_first_task_ps_verification_required': True,
        'server_restart': False, 'warmup': False, 'generation_requests': 0})
    c1 = freeze_config('qwen2.5:1.5b-instruct', load_setup()['model_identifier'], '1b5')
    c3 = freeze_config(MODEL3, DIGEST3, '3b')
    # Same terminal public information as each original Episode, including
    # B0/B2 distinction; NOT the later private source store or reference.
    oldrows = rows(OLD / 'c4_39_memory_b0_b2_001/episodes.jsonl')
    panel_old = {r['case_ref']: r for r in read(OLD / 'memory_panel.json')}
    pilot = []; reviews = []
    for i, row in enumerate(oldrows, 1):
        folder = OLD / 'c4_39_memory_b0_b2_001/requests' / ('Episode%04d' % i)
        requests = sorted(folder.glob('Request*_before.json'))
        request = read(requests[-1])['request']
        guard = input_guard(request['messages'][0]['content'], request['messages'][1]['content'], request['format'])
        ref = panel_old[row['case_ref']]['reference']
        prediction = row['result']['prediction']
        match = diagnosis_match(prediction['diagnosis'], ref['accepted_diagnoses'], certainty=prediction['certainty'])
        reviews.append({'episode': i, 'condition': row['condition'], 'case_ordinal': row['case_ordinal'], **match})
        pilot.append({'episode': i, 'case_ref': row['case_ref'], 'condition': row['condition'],
                      'request': request, 'request_source_sha256': sha(requests[-1]),
                      'reference': ref, 'old_prediction': prediction, 'guard': guard})
    write(PR / 'diagnostic_snapshot_panel.json', pilot)
    write(PUB / 'c4_39_diagnosis_match_review.json', {'registered': 20, 'rows': reviews,
        'classification_counts': dict(Counter(r['classification'] for r in reviews)),
        'clinical_equivalence_unreviewed': True, 'private_predictions_preserved': True})
    write(PUB / 'concept_match_rules.json', {'version': 'conservative_source_match.v1',
        'normalization_only': 'NFKC/case/punctuation/whitespace', 'verified_equivalence_rules': [],
        'uncertain': 'REVIEW_REQUIRED', 'clinical_accuracy': 'NOT_EVALUATED'})
    write(PUB / 'concept_match_provenance.json', {'new_synonyms_or_clinical_verdicts_created': 0,
        'source_episode_sha256': sha(OLD / 'c4_39_memory_b0_b2_001/episodes.jsonl'),
        'independent_clinical_review': 'NOT_PROVIDED', 'unmatched_concept_agreement': None})
    csv_file(PUB / 'original_exact_vs_concept_results.csv', reviews,
             ['episode', 'condition', 'case_ordinal', 'original_exact', 'normalized_exact', 'concept_match', 'classification', 'rule_id'])
    cases = load_cases(DATA); memory = []; index = {}
    for n in range(11, 31):
        case = cases[n-1]; ref = 'Final3BCase%03d' % n; index[ref] = n
        memory.append({'case_ref': ref, 'patient_ref': 'Patient' + digest({'dataset': sha(DATA), 'patient': case['patient_id']})[:16],
                       'execution_case': {'case_ref': ref}, 'reference': {
                       'accepted_diagnoses': case['evaluation']['diagnosis_labels'],
                       'accepted_departments': case['evaluation']['accepted_departments_source_policy']}})
    if len({r['patient_ref'] for r in memory}) != 20:
        raise ValueError('MEMORY_PANEL_PATIENT_DUPLICATE')
    write(PR / 'memory_panel.json', memory); write(PR / 'memory_case_index.json', index)
    common = deepcopy(read(ROOT / 'outputs/research_evaluation/c4_39_v1/memory_execution_contract.json')['common'])
    new_binding = deepcopy(common['model_bindings'])
    for b in new_binding:
        b.update(model=MODEL3, version='final-exp-3b', digest=DIGEST3)
    common.update(model_bindings=new_binding, registry_sha256=sha(ROOT / c3['registry_path']),
        patient_split_sha256=digest([r['patient_ref'] for r in memory]),
        memory_sha256=digest([cases[n-1]['agent_visible']['past_visits'] for n in range(11, 31)]),
        request_budget={'total': 480, 'per_episode': 12, 'retry': 0, 'repair': 0},
        resource_budget={'wall_seconds': 7200, 'output_bytes': GIB, 'memory_bytes': 4*GIB})
    contract = make_contract(common, memory, conditions=('B0', 'B2'))
    write(PR / 'memory_execution_contract.json', contract)
    write(PUB / 'final_memory_evaluation_panel.json', {'ordinals': list(range(11, 31)), 'patients': 20,
        'episodes': 40, 'prior_pilot_ordinals_excluded': list(range(1, 11)),
        'case030_previous_development_exposure': True, 'selection': 'input order, fixed before outputs',
        'independent_holdout': False, 'reference_private': True})
    write(PUB / 'final_memory_evaluation_protocol.json', {'contract_fingerprint': contract['fingerprint'],
        'conditions': ['B0', 'B2'], 'features': contract['features'], 'readiness': run_paired_ablation(contract, memory),
        'B1': 'SKIPPED_INITIAL_FULL_HISTORY_BYTE_GUARD_RESOURCE', 'clinical_metrics': 'NOT_EVALUATED'})
    external = read(ROOT / 'outputs/private/research_evaluation/c4_24_v1/question_reference_manifest.json')
    if len(external['questions']) != 92 or len(external['excluded']) != 15:
        raise ValueError('FROZEN_ELIGIBILITY_DENOMINATOR')
    if check(external['source_pins'])['changed_or_missing']:
        raise ValueError('EXTERNAL_SOURCE_HASH')
    clusters = duplicate_clusters(external['questions']); used = set(); selected = []
    for i, cluster in enumerate(clusters):
        if cluster not in used:
            used.add(cluster); selected.append(i)
    if len(selected) < 43:
        raise ValueError('INSUFFICIENT_DISJOINT_SOURCE_COMPONENTS')
    assignments = [('pilot', selected[:3]), ('train', selected[3:23]), ('evaluation', selected[23:43])]
    routing = {}
    for role, indices in assignments:
        routing[role] = []
        for i in indices:
            q = deepcopy(external['questions'][i]); q['role'] = 'train' if role == 'train' else 'validation'
            routing[role].append({'question': q, 'reference': external['references'][i],
                                  'source_ordinal': external['original_ordinals'][i], 'duplicate_component': clusters[i]})
    write(PR / 'routing_panel.json', routing)
    inventory = read(ROOT / 'outputs/private/research_evaluation/c4_24_v1/agent_binding_inventory.json')
    registry = deepcopy(inventory['registry'])
    for a in registry['agents']:
        a.update(model=MODEL3, version='final-exp-3b')
    bindings = [compact_binding(b) for b in inventory['bindings']]
    for b in bindings:
        b.update(model=MODEL3, version='final-exp-3b', model_identifier=DIGEST3)
    write(PR / 'routing_bindings.json', bindings)
    write(PUB / 'routing_3b_agent_registry.json', {'registry': registry,
        'agent_kind': 'ROLE_CONDITIONED_AGENT', 'distinct_weight_models': False,
        'model_identifier': DIGEST3, 'prompt_hashes': {b['agent_id']: digest(b['system_prompt']) for b in bindings}})
    write(PUB / 'routing_train_eval_split.json', {'pilot_cases': 3, 'train_cases': 20, 'evaluation_cases': 20,
        'source_ordinals': {role: [r['source_ordinal'] for r in rs] for role, rs in routing.items()},
        'disjoint_source_duplicate_components': True, 'source_case_patient_identifier': 'NOT_AVAILABLE',
        'real_patient_independence': 'UNKNOWN; AgentClinic source scenarios, not longitudinal patient IDs',
        'model_training_and_LoRA_exposure': 'UNKNOWN', 'all_scenarios_development_exposed': True,
        'Original50_VoI_patient_overlap': 'NOT_LINKABLE_DIFFERENT_SOURCE; not a verified identity exclusion',
        'near_duplicate_rule': 'connected components of same source case, full actor, normalized full text, word3gram Jaccard>=0.85',
        'split_selected_using_answers': False, 'clinical_reference_approved': False})
    # Pin existing source/model/contracts plus new code, without overwriting any
    # original experiment or pretending a planned output is an actual result.
    protocol = {'version': 'final_exp.v1', 'created_at': now(), 'scope': SCOPE,
        'namespace': 'final_exp_v1', 'model': MODEL3, 'model_identifier': DIGEST3,
        'context': 32768, 'output_tokens': 1024, 'timeout_seconds': 120,
        'temperature': 0, 'seed': 42, 'retry': 0, 'repair': 0, 'warmup': 0,
        'max_generations': 755, 'per_stage_caps': {'diagnostic_snapshots': 20, 'memory': 480,
            'routing_pilot': 15, 'routing_train': 100, 'routing_evaluation': 100, 'external': 40},
        'wall_seconds': 10800, 'process_private_memory_bytes': 4*GIB, 'output_bytes': 2*GIB,
        'order': ['diagnostic_snapshots', 'memory', 'routing_pilot', 'routing_train', 'routing_evaluation', 'external'],
        'routing_train_gate': 'at least2 of3 complete supported/scorable pilot groups; actual differences in correctness across agents',
        'routing_evaluation_gate': 'at least2 complete supported Train groups with differing agent labels; source-only fit provenance and Core validation',
        'external_gate': '20 evaluation source cases fixed now; skip if remaining global wall<2400s before stage, no outcome-based resizing',
        'diagnostic_pilot': '20 frozen final decision payloads from old B0/B2 x10; old1.5B response retained, new3B one action only; nonterminal is not a diagnosis',
        'memory_contract': contract['fingerprint'], 'source_proxy_accepted_for': 'development generation only, requires operator approval',
        'first_task_model_load': True, 'actual_digest_context_check_before_continuing': True,
        'server_restart_download_settings_change': False,
        'failure_policy': 'first technical error stops that stage; identity/source/gold breach stops all; no retry/resume',
        'per_call_guard': 'unchanged 32768 conservative byte guard including full schema; no truncation',
        'conditional_source_utility_fitting': 'Core existing min_samples2; development lexical Source QA only; clinical Gate unchanged',
        'official_B3_B4': 'BLOCKED', 'clinical_verdicts': 'NOT_GENERATED', 'method_formula_changes': 0}
    inputs = list(PR.glob('*.json')) + list(PUB.glob('*.json'))
    code = ['scripts/run_final_3b_evaluation.py', 'app/locdi/final_diagnosis_evaluation.py',
            'app/locdi/final_3b_routing_evaluation.py', 'app/locdi/current_ablation_llm_backend.py',
            'app/locdi/current_ablation_runner.py', 'app/locdi/llm_setup.py', 'app/locdi/answer_grounded_compact.py',
            'app/locdi/answer_grounded_routing.py', 'app/locdi/run_simulation.py', 'app/locdi/case_adapter.py']
    protocol['pins'] = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    protocol['pins'].update({p: sha(ROOT / p) for p in code})
    protocol['fingerprint'] = digest(protocol)
    write(PUB / 'final_exp_protocol.json', protocol)
    write(PUB / 'actual_execution_plan.json', protocol)
    write(PUB / 'actual_execution_approval_status.json', {'status': 'NEW_EXPLICIT_APPROVAL_PENDING',
        'generation_requests': 0, 'protocol_fingerprint': protocol['fingerprint'],
        'prior_C4_approvals_reused': False})
    write(PUB / 'diagnosis_model_pilot_protocol.json', {'snapshots': 20, 'cases': 10, 'conditions': ['B0', 'B2'],
        'same_system_user_schema': True, 'old_1b5_original_responses_reused_not_regenerated': True,
        '3b_request_only_one_action_per_snapshot': True, 'nonterminal_outputs': 'NO_FINAL_DIAGNOSIS',
        'not_a_fresh_paired_episode_experiment': True})
    write(PUB / 'final_prompt_contract.json', {'ER': sha(ROOT / 'prompts/er_department_system.txt'),
        'specialty': {b['agent_id']: digest(b['system_prompt']) for b in bindings},
        'changes_to_existing_prompt': 0, 'planned_prompt_optimization': False})
    write(PUB / 'external_diagnosis_evaluation_protocol.json', {'source': 'AgentClinic-MedQA', 'eligible': 92,
        'excluded_literal_leakage': 15, 'selected': 20, 'selected_from': 'routing evaluation group',
        'models': ['qwen2.5:1.5b-instruct', MODEL3], 'max_generations': 40,
        'prompt': 'existing Compact general diagnostic prompt; identical across models',
        'ER_baseline_reuse': '3B external outputs reused as ER-only routing baseline; no duplicate generation',
        'static_all_disclosed_scenario': True, 'not_interactive_OSCE_accuracy': True})
    command = 'python -B scripts/run_final_3b_evaluation.py execute --execute --allow-llm --approval-file <NEW_APPROVAL_JSON>'
    (PUB / 'actual_execution_commands.md').write_text('# FINAL-EXP\n\nPending new explicit approval.\n\n```powershell\n' + command + '\n```\n', encoding='utf-8')
    print(json.dumps({'status': 'PREPARED_NEW_APPROVAL_PENDING', 'fingerprint': protocol['fingerprint'],
                      'max_generations': 755, 'protected_files': len(pins)}))


def verify_approval(record, protocol):
    if (record.get('approved') is not True or record.get('scope') != SCOPE
            or record.get('plan_fingerprint') != protocol['fingerprint']
            or record.get('evidence_type') != 'USER_EXPLICIT' or not record.get('user_statement')):
        raise ValueError('NEW_EXACT_EXPLICIT_APPROVAL_REQUIRED')
    issued = datetime.fromisoformat(record['recorded_at']); expires = datetime.fromisoformat(record['expires_at'])
    if issued.tzinfo is None or expires.tzinfo is None or not issued <= datetime.now(timezone.utc) < expires:
        raise ValueError('APPROVAL_NOT_CURRENT')
    if not 0 < (expires-issued).total_seconds() <= 86400:
        raise ValueError('APPROVAL_MAX_24_HOURS')


class Execution:
    def __init__(self, protocol, record):
        verify_approval(record, protocol)
        if protocol['fingerprint'] != digest({k: v for k, v in protocol.items() if k != 'fingerprint'}):
            raise ValueError('PROTOCOL_FINGERPRINT')
        if check(protocol['pins'])['changed_or_missing']:
            raise ValueError('PINNED_CODE_OR_INPUT_CHANGED')
        if check(read(PR / 'protected_input_hashes.json'))['changed_or_missing']:
            raise SharedIntegrityError('PROTECTED_SOURCE_CHANGED')
        self.protocol = protocol; self.record = record; self.start = time.monotonic()
        self.budget = GenerationBudget(protocol['max_generations'], protocol['max_generations'])
        self.budget.resource_guard = self.guard
        tracked = set(protocol['pins']) | set(read(PR / 'protected_input_hashes.json'))
        self.input_stats = {p: ((ROOT/p).stat().st_size, (ROOT/p).stat().st_mtime_ns) for p in tracked}
        self.run = PR / 'live/final_exp_001'
        self.run.mkdir(parents=True, exist_ok=False)
        write(self.run / 'started.json', {'at': now(), 'approval': record, 'protocol_fingerprint': protocol['fingerprint']})
        self.c1 = read(PR / 'llm_config_1b5.json'); self.c3 = read(PR / 'llm_config_3b.json')
        self.all_rows = []; self.errors = []

    def guard(self):
        verify_approval(self.record, self.protocol)
        for p, before in self.input_stats.items():
            try:
                stat = (ROOT/p).stat()
            except OSError:
                raise SharedIntegrityError('PINNED_INPUT_DISAPPEARED') from None
            if (stat.st_size, stat.st_mtime_ns) != before:
                raise SharedIntegrityError('PINNED_INPUT_MODIFIED')
        if time.monotonic() - self.start >= self.protocol['wall_seconds']:
            raise ValueError('GLOBAL_WALL_LIMIT')
        if _process_private_bytes() >= self.protocol['process_private_memory_bytes']:
            raise ValueError('PROCESS_MEMORY_LIMIT')
        size = sum(p.stat().st_size for root in (PUB, PR) for p in root.rglob('*') if p.is_file())
        if size >= self.protocol['output_bytes']:
            raise ValueError('GLOBAL_OUTPUT_LIMIT')

    def call(self, name, n, system, user, response_schema, config):
        self.guard()
        folder = self.run / name / ('Task%04d' % n)
        client = BudgetedNativeERClient(config, authorized=True,
            contract_fingerprint=self.protocol['fingerprint'], checkpoint_root=folder)
        self.budget.episode_used = 0
        token = _BUDGET.set(self.budget)
        try:
            result = client.chat_json(system, user, model=config['model'], schema=response_schema)
            self.guard()
            return result
        finally:
            _BUDGET.reset(token); client.close()

    def collect(self, stage, pairs, bindings, config):
        outputs = []; stop = None; n = 0
        for p in pairs:
            q, reference = p['question'], p['reference']
            validate_pair(q, reference)
            user = json.dumps(model_input(q), ensure_ascii=False, sort_keys=True)
            for b in bindings:
                n += 1
                if stop:
                    r = score(q, reference, attempted=False)
                else:
                    try:
                        content = self.call(stage, n, b['system_prompt'], user, schema(), config)
                        r = score(q, reference, content)
                        after = self.run / stage / ('Task%04d' % n) / 'Request0001_after.json'
                        r.update(response_sha256=sha(after), reference_fingerprint=digest(reference),
                                 model_identifier=config['model_identifier'], prompt_fingerprint=digest(b['system_prompt']),
                                 input_fingerprint=digest(model_input(q)), response_path=str(after.relative_to(ROOT)))
                        if r['state'] == 'INVALID_RESPONSE':
                            stop = 'INVALID_RESPONSE'
                    except Exception as exc:
                        r = score(q, reference, error_type=type(exc).__name__); stop = type(exc).__name__
                        if 'Identity' in stop or isinstance(exc, SharedIntegrityError):
                            raise
                r['agent_id'] = b['agent_id']; outputs.append(r)
        write(self.run / (stage + '_outcomes.json'), outputs)
        self.all_rows.extend({'stage': stage, **r} for r in outputs)
        if stop:
            self.errors.append({'stage': stage, 'error_type': stop})
        return outputs


def execute(path):
    protocol_path = PUB / 'final_exp_protocol_final.json'
    protocol = read(protocol_path if protocol_path.exists() else PUB / 'final_exp_protocol.json'); record = read(path)
    run = Execution(protocol, record)
    fatal = None; results = {}
    try:
        snapshots = read(PR / 'diagnostic_snapshot_panel.json'); pilot = []
        for n, p in enumerate(snapshots, 1):
            req = p['request']; prediction = None; error = None
            try:
                raw = run.call('diagnostic_snapshots', n, req['messages'][0]['content'],
                               req['messages'][1]['content'], req['format'], run.c3)
                value = _strict_object(raw)
                prediction = {'certainty': value.get('certainty'), 'diagnosis': value.get('target') if value.get('action') == 'diagnose' else None,
                              'department': value.get('department'), 'action': value.get('action')}
                match = diagnosis_match(prediction['diagnosis'], p['reference']['accepted_diagnoses'], certainty=prediction['certainty'])
                pilot.append({'episode': n, 'condition': p['condition'], 'prediction': prediction, 'match': match})
            except Exception as exc:
                error = type(exc).__name__; pilot.append({'episode': n, 'error_type': error})
                if 'Identity' in error or isinstance(exc, SharedIntegrityError):
                    raise
                break
        write(PR / 'diagnosis_model_pilot_results.json', pilot)
        write(PUB / 'diagnosis_model_pilot_results.json', {'registered': 20, 'attempted': len(pilot),
            'new_model': MODEL3, 'original_exact': sum(r.get('match', {}).get('original_exact', False) for r in pilot),
            'normalized_exact': sum(r.get('match', {}).get('normalized_exact', False) for r in pilot),
            'states': dict(Counter(r.get('prediction', {}).get('action', 'FAILED') for r in pilot)),
            'full_episode_comparison': False, 'clinical_metrics': 'NOT_EVALUATED'})
        run.guard()
        memory = read(PR / 'memory_panel.json'); contract = read(PR / 'memory_execution_contract.json')
        cases = load_cases(DATA); index = read(PR / 'memory_case_index.json')
        backend = make_er_backend({ref: cases[n-1] for ref, n in index.items()}, config=run.c3,
            contract_fingerprint=contract['fingerprint'], checkpoint_root=run.run / 'memory_requests', authorized=True)
        # Existing harness owns its stricter per-Episode and 480-request budget.
        # Outer global wall/output/approval guard is also checked every request.
        def wrapped(execution_case, features):
            run.guard()
            active = _BUDGET.get(); original = active.resource_guard
            active.resource_guard = lambda: (original(), run.guard())
            try:
                return backend(execution_case, features)
            finally:
                active.resource_guard = original
        child = {**record, 'scope': 'C4_38_CURRENT_ABLATION_DEVELOPMENT', 'plan_fingerprint': contract['fingerprint'],
                 'parent_final_exp_protocol': protocol['fingerprint']}
        result = run_paired_ablation(contract, memory, run_one=wrapped, execute=True, approval_record=child,
            public_root=PUB / 'final_exp_memory_001', private_root=PR / 'final_exp_memory_001')
        run.budget.used += result['generation_count']
        results['memory'] = result
        write(PUB / 'final_memory_actual_results.json', result)
        for after in (run.run / 'memory_requests').rglob('Request*_after.json'):
            if read(after).get('error_type') in ('ModelIdentityError', 'SharedIntegrityError'):
                raise ValueError('SHARED_MODEL_IDENTITY_FAILURE_STOP_ALL')
        if any(r['result'].get('error_type') in ('ModelIdentityError', 'SharedIntegrityError')
               for r in rows(PR / 'final_exp_memory_001/episodes.jsonl')):
            raise SharedIntegrityError('SHARED_INTEGRITY_FAILURE_STOP_ALL')
        routing = read(PR / 'routing_panel.json'); bindings = read(PR / 'routing_bindings.json')
        registry = read(PUB / 'routing_3b_agent_registry.json')['registry']
        pilot_rows = run.collect('routing_pilot', routing['pilot'], bindings, run.c3)
        gs = complete_groups([p['question'] for p in routing['pilot']], bindings, pilot_rows)
        gate = sum(g['complete_scorable'] for g in gs) >= 2
        differentiated = any(len({r['answer_correct'] for r in pilot_rows if r['question_id'] == g['question_id']}) > 1
                             for g in gs if g['complete_scorable'])
        write(PUB / 'routing_technical_pilot_results.json', {'groups': gs, 'advance': gate and differentiated,
            'states': dict(Counter(r['state'] for r in pilot_rows)), 'actual_requests': sum(r['state'] != 'NOT_ATTEMPTED' for r in pilot_rows)})
        artifact = None; eval_rows = []; train_rows = []; blocker = None
        if gate and differentiated:
            train_rows = run.collect('routing_train', routing['train'], bindings, run.c3)
            try:
                artifact = fit_source_utility([p['question'] for p in routing['train']], bindings,
                                             train_rows, registry, provenance_verified=True)
                write(PR / 'source_qa_development_utility.json', artifact)
                write(PUB / 'routing_utility_artifact_manifest.json', {'status': 'DEVELOPMENT_SOURCE_QA_ONLY',
                    'sha256': sha(PR / 'source_qa_development_utility.json'), 'clinical_gate': 'UNCHANGED_BLOCKED',
                    'registry_fingerprint': digest(registry), 'Core_formula_changed': False})
                eval_rows = run.collect('routing_evaluation', routing['evaluation'], bindings, run.c3)
            except ValueError as exc:
                blocker = str(exc)
        else:
            blocker = 'PILOT_INSUFFICIENT_COMPLETE_SCORABLE_OR_DIFFERENTIATED_GROUPS'
        er3 = []
        if protocol['wall_seconds'] - (time.monotonic()-run.start) >= 2400:
            from app.locdi.answer_grounded_compact import PROMPT
            er_binding = [{'agent_id': 'ER_GENERAL', 'system_prompt': PROMPT}]
            ext1 = run.collect('external_1b5', routing['evaluation'], er_binding, run.c1)
            er3 = run.collect('external_3b', routing['evaluation'], er_binding, run.c3)
            write(PUB / 'external_diagnosis_evaluation_results.json', {'registered_per_model': 20,
                'models': {name: dict(Counter(r['state'] for r in rs)) for name, rs in [('1b5', ext1), ('3b', er3)]},
                'clinical_accuracy': 'NOT_EVALUATED', 'source_scoring': 'frozen Compact lexical scorer'})
        else:
            write(PUB / 'external_diagnosis_evaluation_results.json', {'status': 'SKIPPED_PREDECLARED_REMAINING_TIME_RULE', 'generation_requests': 0})
        evaluation = evaluate_policies([p['question'] for p in routing['evaluation']], bindings, eval_rows, registry,
                                       artifact=artifact, er_rows=er3)
        write(PR / 'routing_actual_evaluation_results.json', evaluation)
        write(PUB / 'routing_actual_evaluation_results.json', {k: v for k, v in evaluation.items() if k != 'traces'})
        write(PUB / 'routing_final_status.json', {'status': 'ROUTING_EVALUATED' if artifact and eval_rows else 'ROUTING_INSUFFICIENT_DATA',
            'blocker_code': blocker, 'clinical_routing': 'BLOCKED', 'source_department_accuracy': 'NOT_APPLICABLE'})
        write(PUB / 'routing_complete_group_report.json', {role: complete_groups([p['question'] for p in routing[role]], bindings, rs)
            for role, rs in [('pilot', pilot_rows), ('train', train_rows), ('evaluation', eval_rows)]})
    except Exception as exc:
        fatal = type(exc).__name__
    finally:
        write(run.run / 'finished.json', {'at': now(), 'fatal_error_type': fatal, 'stage_errors': run.errors,
            'actual_generation_requests': run.budget.used, 'wall_seconds': time.monotonic()-run.start})
        write(PUB / 'actual_execution_summary.json', {'fatal_error_type': fatal, 'stage_errors': run.errors,
            'actual_generation_requests': run.budget.used, 'wall_seconds': time.monotonic()-run.start,
            'clinical_accuracy': 'NOT_EVALUATED', 'existing_protection': check(read(PR / 'protected_input_hashes.json'))})
    print(json.dumps({'fatal_error_type': fatal, 'generation_requests': run.budget.used}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'execute'])
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--allow-llm', action='store_true')
    parser.add_argument('--approval-file')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif not (args.execute and args.allow_llm and args.approval_file):
        parser.error('NEW APPROVAL + --execute + --allow-llm required; no model client constructed')
    else:
        execute(args.approval_file)
