"""FINAL-EXP saved-trace reaggregation and thesis addendum; zero model calls.

Run only after the one authorized live run has finished. All publications are
exclusive. Previous 42 result rows and the original nine thesis files survive.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_final_3b_evaluation import PUB, PR, MODEL3, DIGEST3, csv_file
from scripts.audit_research_gaps import read, rows, sha, write, check
from scripts.finalize_c4_39_results import no_gold
from app.locdi.current_ablation_runner import reaggregate_trace
from app.locdi.final_diagnosis_evaluation import diagnosis_match
from app.locdi.answer_grounded_compact import score, PROMPT
from app.locdi.answer_grounded_routing import model_input
from app.locdi.llm_setup import digest
from app.locdi.dataset_loader import load_cases
from datetime import datetime

DOC = ROOT / 'docs/연구 평가/FINAL_EXP'
LIVE = PR / 'live/final_exp_001'


def md(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(text.rstrip() + '\n')


def table(rs, columns):
    def v(x):
        if x is None: return 'NOT_EVALUATED'
        if isinstance(x, float): return '%.6f' % x
        return str(x).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |'] +
                     ['| ' + ' | '.join(v(r.get(c)) for c in columns) + ' |' for r in rs])


def paired_bootstrap(first, second, repeats=2000):
    """Whole-panel success-yield indicator, not imputed clinical correctness."""
    if len(first) != len(second) or not first:
        raise ValueError('NONEMPTY_MATCHED_PANEL_REQUIRED')
    diffs = [float(b)-float(a) for a, b in zip(first, second)]
    rng = random.Random(42); n = len(diffs)
    samples = sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(repeats))
    return {'B2_minus_B0': sum(diffs)/n, 'patients': n, 'bootstrap_repeats': repeats, 'seed': 42,
            'percentile_95_CI': [samples[int(.025*(repeats-1))], samples[int(.975*(repeats-1))]],
            'unit': 'paired patient, one encounter per registered patient',
            'interpretation': 'descriptive development Source supported-success yield; clinical correctness unknown for failures/unresolved'}


def request_audit():
    audits = []; gold_ok = True
    for after in sorted(LIVE.rglob('Request*_after.json')):
        a = read(after); before = after.with_name(after.name.replace('_after', '_before'))
        b = read(before) if before.exists() else None
        request = b['request'] if b else None
        user = json.loads(request['messages'][1]['content']) if request else {}
        gold_ok = gold_ok and no_gold(user)
        envelope = a.get('response_envelope', {})
        mem = user.get('observation', {}).get('longitudinal_memory', {})
        audits.append({'stage': after.relative_to(LIVE).parts[0],
            'episode_folder': after.parent.name if 'memory_requests' not in after.parts else after.parent.name,
            'after_private_path': str(after.relative_to(ROOT)), 'after_sha256': sha(after),
            'generation_attempted': a.get('generation_attempted', True),
            'HTTP_status': a.get('HTTP_status'), 'status': a.get('status'),
            'error_type': a.get('error_type'), 'wall_seconds': a.get('wall_seconds'),
            'prompt_tokens': envelope.get('prompt_eval_count'), 'output_tokens': envelope.get('eval_count'),
            'done_reason': envelope.get('done_reason'), 'identity_after': a.get('identity_after'),
            'memory_records': len(mem.get('records', [])), 'memory_mode': mem.get('mode'),
            'structured_gold_private_keys_absent': no_gold(user),
            'request_fingerprint_matches': b and b['request_fingerprint'] == digest(request)
                and a.get('request_fingerprint') == digest(request)})
    return audits, gold_ok


def information_boundaries():
    """Check delivered Memory/source times and actual Test receipt; no clinical verdict."""
    cases = load_cases(ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl')
    case_index = read(PR / 'memory_case_index.json'); panel = read(PR / 'memory_panel.json')
    counts = Counter(); failures = []; unknown = []
    def date(value):
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    for folder in sorted((LIVE / 'memory_requests').glob('Episode*')):
        episode = int(folder.name[-4:]); entry = panel[(episode-1) % 20]
        case = cases[case_index[entry['case_ref']]-1]
        source = {r['memory_id']: r for v in case['agent_visible']['past_visits'] for r in v['records']}
        raw_path = folder / 'raw_runner_result.json'
        raw = read(raw_path) if raw_path.exists() else {}
        visits = (raw.get('execution') or {}).get('visits', [])
        events = visits[0]['final_state']['event_log'] if visits else []
        received = [e for e in events if e['event_type'] == 'result_received']
        for before in sorted(folder.glob('Request*_before.json')):
            u = json.loads(read(before)['request']['messages'][1]['content'])
            obs = u['observation']; mem = obs.get('longitudinal_memory', {})
            counts['requests'] += 1
            if not no_gold(u): failures.append({'episode': episode, 'code': 'HIDDEN_PRIVATE_FIELD'})
            records = mem.get('records', [])
            if episode <= 20 and records: failures.append({'episode': episode, 'code': 'B0_MEMORY_PRESENT'})
            if len(records) > 5: failures.append({'episode': episode, 'code': 'TOP5_EXCEEDED'})
            for record in records:
                counts['memory_occurrences'] += 1
                original = source.get(record['memory_id'])
                valid = (original is not None and all(record.get(k) == v for k, v in original.items())
                    and record['patient_id'] == case['patient_id']
                    and record['visit_id'] != case['agent_visible']['index_encounter_id']
                    and date(record['visit_started_at']) < date(mem['as_of'])
                    and date(record['available_at']) <= date(mem['as_of'])
                    and date(record['observed_at']) <= date(mem['as_of']))
                counts['memory_cutoff_and_source_pass'] += bool(valid)
                if not valid: failures.append({'episode': episode, 'code': 'MEMORY_SOURCE_OR_CUTOFF'})
            for finding in obs.get('objective_findings', []):
                counts['objective_finding_occurrences'] += 1
                matching = [e for e in received if finding in (e['payload']['test_result'].get('summary'), e['payload']['test_result'].get('impression'))]
                if not matching:
                    unknown.append({'episode': episode, 'code': 'NO_RECEIVED_RESULT_LINEAGE'})
                    continue
                valid_time = any(e['minute'] <= obs['elapsed_minutes'] for e in matching)
                counts['finding_after_actual_receipt'] += valid_time
                if not valid_time: failures.append({'episode': episode, 'code': 'RESULT_BEFORE_RECEIPT'})
                linked = False
                for result in matching:
                    name = result['payload']['test_result']['test_name']
                    for event in case['environment']['current_encounter_events']:
                        if event['action'] == 'Test' and finding == '%s: %s' % (name, event['evidence']):
                            linked = True
                            if result['minute'] < event['available_minute']:
                                failures.append({'episode': episode, 'code': 'RESULT_BEFORE_SOURCE_PROXY'})
                counts['finding_exact_source_rendering'] += linked
                if not linked: unknown.append({'episode': episode, 'code': 'SOURCE_RENDERING_NOT_LINKED'})
    return {'status': 'FAIL' if failures else 'UNKNOWN' if unknown else 'PASS', 'counts': dict(counts),
        'failures': failures, 'unknown': unknown, 'recorded_time_proxy': 'DEVELOPMENT_ACCEPTED_NOT_CLINICALLY_REVIEWED',
        'B2_NOT_ATTEMPTED': counts['memory_occurrences'] == 0,
        'B2_cutoff_new_execution_evidence': 'NOT_APPLICABLE_IF_NO_B2_REQUESTS',
        'clinical_visibility_review': 'NOT_REVIEWED'}


def failure_analysis():
    details = []
    for folder in sorted((LIVE / 'memory_requests').glob('Episode*')):
        raw_path = folder / 'raw_runner_result.json'
        if not raw_path.exists(): continue
        raw = read(raw_path)
        if raw['status'] != 'failed': continue
        befores = sorted(folder.glob('Request*_before.json'))
        before = read(befores[-1]); after = read(befores[-1].with_name(befores[-1].name.replace('_before', '_after')))
        user = json.loads(before['request']['messages'][1]['content'])
        value = json.loads(after['response_envelope']['message']['content'])
        indices = value.get('evidence_indices', []); eligible = user.get('supported_diagnosis_eligible_indices', [])
        ineligible = (value.get('action') == 'diagnose' and value.get('certainty') == 'supported'
                      and not set(indices).intersection(eligible))
        details.append({'episode_ordinal': int(folder.name[-4:]),
            'original_ordinal': read(PR/'memory_case_index.json')[read(PR/'memory_panel.json')[int(folder.name[-4:])-1]['case_ref']],
            'raw_status': raw['status'], 'HTTP_status': after.get('HTTP_status'), 'JSON_schema': after['status'],
            'action': value.get('action'), 'certainty': value.get('certainty'),
            'citation_indices': indices, 'eligible_objective_indices': eligible,
            'cited_kinds': [user['evidence_records'][i]['kind'] for i in indices if 0 <= i < len(user['evidence_records'])],
            'cause': 'SUPPORTED_DIAGNOSIS_WITHOUT_ELIGIBLE_OBJECTIVE_CITATION' if ineligible else 'REQUIRES_FURTHER_READ_ONLY_ANALYSIS',
            'source_function': 'app/core/hospital_llm_policy.py: nested validate / supported diagnosis eligible objective finding',
            'code_SHA256': sha(ROOT/'app/core/hospital_llm_policy.py'),
            'retry_repair': 0, 'automatic_response_modification': False})
    return {'failures': details, 'Memory_comparison': 'BLOCKED_TECHNICAL_STOP' if details else 'CHECK_MEMORY_METRICS',
            'scope': 'contract failure; no clinical error verdict', 'new_reruns': 0}


def recheck_collections():
    panel = read(PR / 'routing_panel.json'); bindings = read(PR / 'routing_bindings.json')
    checks = []; summaries = {}
    for path in sorted(LIVE.glob('*_outcomes.json')):
        stage = path.name[:-len('_outcomes.json')]
        role = stage.removeprefix('routing_') if stage.startswith('routing_') else 'evaluation'
        pairs = panel[role]; qi = {p['question']['question_id']: p for p in pairs}
        outcomes = read(path)
        for n, row in enumerate(outcomes, 1):
            if row['state'] in ('NOT_ATTEMPTED', 'GENERATION_FAILED'):
                continue
            after_path = LIVE / stage / ('Task%04d' % n) / 'Request0001_after.json'
            a = read(after_path); b = read(after_path.with_name('Request0001_before.json'))
            p = qi[row['question_id']]; req = b['request']
            expected = score(p['question'], p['reference'], a['response_envelope']['message']['content'])
            system = PROMPT if stage.startswith('external_') else next(x['system_prompt'] for x in bindings if x['agent_id'] == row['agent_id'])
            matched = (expected['state'] == row['state'] and expected['answer_correct'] is row['answer_correct']
                and sha(after_path) == row['response_sha256'] and digest(p['reference']) == row['reference_fingerprint']
                and json.loads(req['messages'][1]['content']) == model_input(p['question'])
                and req['messages'][0]['content'] == system and b['request_fingerprint'] == digest(req))
            checks.append({'stage': stage, 'task_ordinal': n, 'match': matched})
        summaries[stage] = dict(Counter(r['state'] for r in outcomes))
    return {'status': 'EXACT_MATCH' if all(x['match'] for x in checks) else 'FAIL',
            'verified_received_outcomes': len(checks), 'checks': checks, 'stage_counts': summaries}


def main():
    if not (LIVE / 'finished.json').exists():
        raise ValueError('AUTHORIZED_RUN_NOT_FINISHED; do not fabricate results')
    if (PUB / 'final_research_results.json').exists():
        raise ValueError('FINAL_PUBLICATION_ALREADY_EXISTS_NO_OVERWRITE')
    finished = read(LIVE / 'finished.json'); protocol = read(PUB / 'final_exp_protocol_final.json')
    protected = check(read(PR / 'protected_input_hashes.json'))
    pinned = check(protocol['pins'])
    audit, gold_ok = request_audit()
    write(PR / 'all_request_audit.json', audit)
    boundary = information_boundaries(); failures = failure_analysis()
    write(PUB / 'actual_information_boundary_check.json', boundary)
    write(PUB / 'timeout_and_error_analysis.json', failures)
    visible_audit = [{k: v for k, v in r.items() if k not in ('after_private_path', 'after_sha256')} for r in audit]
    attempted = sum(r['generation_attempted'] for r in audit)
    write(PUB / 'request_accounting.json', {'checkpoint_count': len(audit), 'generation_attempts_from_after': attempted,
        'reported_budget_used': finished['actual_generation_requests'], 'match': attempted == finished['actual_generation_requests'],
        'stage_counts': dict(Counter(r['stage'] for r in audit if r['generation_attempted'])),
        'request_metadata': visible_audit, 'raw_wire': 'NOT_CAPTURED', 'retry': 0, 'repair': 0})
    oldresults = read(ROOT / 'outputs/research_evaluation/c4_39_v1/final_research_results.json')['rows']
    if len(oldresults) != 42:
        raise ValueError('ORIGINAL_42_ROW_BASELINE_CHANGED')
    all_results = deepcopy(oldresults); additions = []
    def add(group, condition, metric, value, denominator, source, patients=None, visits=None, scope='development Source agreement'):
        additions.append({'group': group, 'condition': condition, 'metric': metric, 'value': value,
            'patients': patients, 'visits': visits, 'denominator': denominator, 'source': source,
            'scope': scope, 'clinical_metrics': 'NOT_EVALUATED'})
    memory_rows = []; memory_table = []; paired = {'status': 'NOT_RUN'}; memory_ok = False; memory_reaggregation_match = None
    trace = PR / 'final_exp_memory_001/episodes.jsonl'
    if trace.exists():
        contract = read(PR / 'memory_execution_contract.json'); panel = read(PR / 'memory_panel.json')
        memory_rows = rows(trace); aggregate = reaggregate_trace(trace, contract, panel)
        match = aggregate == read(PUB / 'final_exp_memory_001/metrics.json')
        memory_reaggregation_match = match
        write(PUB / 'memory_independent_reaggregation.json', {'status': 'EXACT_MATCH' if match else 'FAIL', 'metrics': aggregate})
        references = {r['case_ref']: r['reference'] for r in panel}
        private_matches = []; by_condition = {}
        for condition in ('B0', 'B2'):
            rs = [r for r in memory_rows if r['condition'] == condition]; m = aggregate['conditions'][condition]
            for r in rs:
                pred = r['result'].get('prediction', {}); ref = references[r['case_ref']]
                evidence = diagnosis_match(pred.get('diagnosis'), ref['accepted_diagnoses'], certainty=pred.get('certainty'))
                private_matches.append({'condition': condition, 'case_ref': r['case_ref'], 'status': r['status'], **evidence})
            requests = [a for a in audit if a['stage'] == 'memory_requests'
                        and (int(a['episode_folder'][-4:]) <= 20) == (condition == 'B0')]
            item = {'condition': condition, 'registered': len(rs), 'completed': m['completed'],
                'incomplete': m['incomplete'], 'failed': m['failed'], 'not_attempted': m['not_attempted'],
                'unresolved': m['unresolved'], 'diagnosis_source_exact': m['source_diagnosis_correct_supported_count'],
                'department_source_agreement': m['department_correct_yield_reference_cases']['numerator'],
                'mean_actions': m['mean_executed_actions_registered'], 'generation_count': m['generation_count'],
                'prompt_tokens': sum(a['prompt_tokens'] or 0 for a in requests),
                'output_tokens': sum(a['output_tokens'] or 0 for a in requests),
                'retrieval_nonempty_requests': sum(a['memory_records'] > 0 for a in requests),
                'memory_record_occurrences': sum(a['memory_records'] for a in requests),
                'HTTP_wall_seconds': sum(a['wall_seconds'] or 0 for a in requests), 'action_counts': m['executed_action_counts_recorded_sum']}
            item['supported_answer_denominator'] = m['supported_answer_agreement']['denominator']
            if all(r['status'] == 'not_attempted' for r in rs):
                item.update(diagnosis_source_exact=None, department_source_agreement=None, mean_actions=None)
            memory_table.append(item); by_condition[condition] = {r['case_ref']: r for r in rs}
            for key in ('completed', 'diagnosis_source_exact', 'department_source_agreement'):
                value = item[key]/len(rs) if key == 'completed' or any(r['status'] != 'not_attempted' for r in rs) else None
                add('FINAL_EXP_Memory_3B', condition, key, value, len(rs),
                    'final_exp_v1/final_exp_memory_001/metrics.json', len(rs), len(rs))
        paired = {'status': 'DESCRIPTIVE_DEVELOPMENT_COMPARISON', 'registered_patients': 20,
            'first10_diagnostic_pilot_excluded': True, 'independent_clinical_reference': False,
            'no_combining_old_1b5_panel_with_new_3b_panel': True}
        complete_paired = all(m['completed'] == 20 for m in memory_table)
        if not complete_paired:
            paired.update(status='NOT_ESTIMABLE_TECHNICAL_BATCH_STOP', reason='No fully observed matched B0/B2 cohort; unattempted B2 is not evidence of no benefit',
                          bootstrap='NOT_RUN', Memory_effect='NOT_EVALUATED')
        for metric in (('diagnosis', 'department') if complete_paired else ()):
            indicators = {}
            for condition in ('B0', 'B2'):
                values = []
                for p in panel:
                    r = by_condition[condition][p['case_ref']]; pred = r['result'].get('prediction', {})
                    values.append(r['status'] == 'completed' and pred.get('certainty') == 'supported'
                                  and pred.get(metric) in p['reference']['accepted_diagnoses' if metric == 'diagnosis' else 'accepted_departments'])
                indicators[condition] = values
            paired[metric + '_supported_yield'] = paired_bootstrap(indicators['B0'], indicators['B2'])
        write(PR / 'memory_concept_review.json', private_matches)
        write(PUB / 'final_memory_paired_comparison.json', paired)
        csv_file(PUB / 'final_memory_metrics.csv', memory_table, list(memory_table[0]))
        memory_ok = match and all(m['completed'] == 20 for m in memory_table)
    else:
        write(PUB / 'final_memory_paired_comparison.json', paired)
        csv_file(PUB / 'final_memory_metrics.csv', [], ['condition', 'registered', 'completed'])
    collection_checks = recheck_collections()
    write(PR / 'collection_independent_reaggregation.json', collection_checks)
    write(PUB / 'collection_independent_reaggregation.json', {k: v for k, v in collection_checks.items() if k != 'checks'})
    pilot_rows = read(PR / 'diagnosis_model_pilot_results.json') if (PR / 'diagnosis_model_pilot_results.json').exists() else []
    snapshots = read(PR / 'diagnostic_snapshot_panel.json'); dx_table = []
    for condition in ('B0', 'B2'):
        selected = [p for p in snapshots if p['condition'] == condition]
        new = [r for r in pilot_rows if r.get('condition') == condition]
        for model, rs, is_old in [('1.5B stored original', selected, True), ('3B frozen snapshot', new, False)]:
            exact = sum(diagnosis_match(r['old_prediction']['diagnosis'], r['reference']['accepted_diagnoses'])['original_exact'] for r in rs) if is_old else sum(r.get('match', {}).get('original_exact', False) for r in rs)
            department = sum(r['old_prediction']['department'] in r['reference']['accepted_departments'] for r in rs) if is_old else sum(r.get('prediction', {}).get('action') == 'diagnose' and r.get('prediction', {}).get('department') in snapshots[r['episode']-1]['reference']['accepted_departments'] for r in rs)
            dx_table.append({'model': model, 'condition': condition, 'registered': 10, 'observed': len(rs),
                'source_exact': exact, 'department_match': department,
                'final_diagnose_count': len(rs) if is_old else sum(r.get('prediction', {}).get('action') == 'diagnose' for r in rs),
                'comparison': 'identical frozen final public snapshot; stored old terminal response vs one new action'})
            if not is_old:
                add('FINAL_EXP_FrozenSnapshot_3B', condition, 'diagnosis_source_exact_yield', exact/10, 10,
                    'final_exp_v1/diagnosis_model_pilot_results.json', 10, 10, '20 snapshots of10 prior development cases; not new episode equivalence')
    csv_file(PUB / 'diagnosis_model_comparison.csv', dx_table, list(dx_table[0]))
    external_table = []
    for model, stage in [('1.5B', 'external_1b5'), ('3B', 'external_3b')]:
        path = LIVE / (stage+'_outcomes.json')
        rs = read(path) if path.exists() else []
        counts = Counter(r['state'] for r in rs)
        external_table.append({'dataset': 'AgentClinic-MedQA static Source QA', 'model': model,
            'registered': 20, 'actual_received': sum(v for k, v in counts.items() if k not in ('NOT_ATTEMPTED', 'GENERATION_FAILED')),
            'source_exact': counts['CORRECT_ANSWER'], 'supported_incorrect': counts['INCORRECT_ANSWER'],
            'unresolved': counts['UNRESOLVED'], 'invalid': counts['INVALID_RESPONSE'],
            'failed': counts['GENERATION_FAILED'], 'not_attempted': counts['NOT_ATTEMPTED'] if rs else 20,
            'source_exact_yield': counts['CORRECT_ANSWER']/20 if rs else None,
            'clinical_accuracy': 'NOT_EVALUATED'})
        add('FINAL_EXP_ExternalQA', model, 'source_exact_yield', external_table[-1]['source_exact_yield'], 20,
            'final_exp_v1/external_diagnosis_evaluation_results.json', None, None, 'static all-disclosed Source QA, development exposed')
    csv_file(PUB / 'multi_dataset_evaluation_summary.csv', external_table, list(external_table[0]))
    routing = read(PUB / 'routing_final_status.json') if (PUB / 'routing_final_status.json').exists() else {'status': 'ROUTING_EXECUTION_BLOCKED', 'blocker_code': finished['fatal_error_type']}
    utility = (PR / 'source_qa_development_utility.json').exists()
    routing_metrics = read(PUB / 'routing_actual_evaluation_results.json') if (PUB / 'routing_actual_evaluation_results.json').exists() else {}
    write(PUB / 'routing_utility_eligibility.json', {'development_source_qa_fit_performed': utility,
        'clinical_correctness_gate': 'UNCHANGED_BLOCKED', 'routing_status': routing,
        'unresolved_as_false': False, 'original_formula': 'unchanged Core fit_utility min_samples=2',
        'model_identifier': DIGEST3, 'distinct_specialty_weights': False})
    write(PUB / 'routing_correctness_provenance.json', {'state': 'SOURCE_MATCH_ONLY_NOT_CLINICAL_CORRECTNESS',
        'scorer': 'existing answer_grounded_compact.score + agentclinic exact normalization',
        'response_reaggregation': collection_checks['status'], 'references': 'existing C4_24 private Source manifest',
        'clinical_reference_approved': False, 'independent_reviewer_created': False,
        'unsupported_unresolved_correctness': None, 'clinical_Gate_changed': False})
    if not utility:
        write(PUB / 'routing_utility_artifact_manifest.json', {'status': 'NO_ARTIFACT_CREATED',
            'artifact_path': None, 'fitting_calls': 0, 'reason': routing.get('blocker_code'), 'not_a_utility_artifact': True})
    write(PUB / 'routing_paired_comparison.json', {'status': routing['status'], 'summary': routing_metrics.get('summary', {}),
        'paired_questions': 20, 'shared_disclosed_input_and_candidate_set': True,
        'clinical_Routing_accuracy': 'NOT_EVALUATED', 'agent_selection_scope': 'static source QA; no Consult or Hospital state changes',
        'patient_independence': 'UNKNOWN outside provided distinct source scenario components',
        'hypothesis': 'BLOCKED' if not utility else 'INCONCLUSIVE_UNTIL_BASELINE_COMPARISON',
        'topK_and_department_accuracy': 'NOT_APPLICABLE_NO_VALID_SPECIALTY_REFERENCE'})
    for name, m in routing_metrics.get('summary', {}).items():
        actual_evaluation_available = utility and routing['status'] == 'ROUTING_EVALUATED'
        add('FINAL_EXP_RoutingSourceQA', name, 'correct_supported_yield', m['correct_supported_yield'] if actual_evaluation_available or name == 'EROnly' else None,
            m['registered'], 'final_exp_v1/routing_actual_evaluation_results.json', scope='static answer-grounded agent selection; native abstention retained')
    all_results.extend(additions)
    write(PUB / 'final_research_results.json', {'rows': all_results, 'old42_rows_unchanged': all_results[:42] == oldresults,
        'new_rows': len(additions), 'mixed_matched_B0_B4': False, 'clinical_metrics': 'NOT_EVALUATED'})
    csv_file(PUB / 'final_research_results.csv', all_results, list(oldresults[0]))
    states = {}
    for condition in ('B0', 'B2'):
        rs = [r for r in memory_rows if r['condition'] == condition]
        states[condition] = ('DEVELOPMENT_COMPLETED' if rs and all(r['status'] == 'completed' for r in rs)
            else 'DEVELOPMENT_PARTIAL' if any(r['status'] != 'not_attempted' for r in rs) else 'NOT_RUN_AFTER_BATCH_STOP')
    ready = {'B0': states['B0'], 'B1': 'SKIPPED_FULL_HISTORY_CONTEXT',
        'B2': states['B2'], 'B3': 'BLOCKED', 'B4': 'BLOCKED',
        'official_matched_B0_B4': 'INCOMPLETE', 'development_utility_does_not_unlock_clinical_gate': True}
    write(PUB / 'final_b0_b4_readiness.json', ready)
    write(PUB / 'final_b0_b4_actual_results.json', {'conditions': ready, 'new_memory_registered_patients': 20,
        'old_results_merged_as_matched_conditions': False, 'B3_B4_generation': 0})
    inventory = [
        {'dataset': 'Original50/Memory72590', 'source': 'existing Synthea longitudinal wrappers', 'role': '20 prior snapshots +20 fixed new3B patient comparisons', 'reference': 'Source diagnosis/department policy; independent clinical review absent', 'exposure': 'DEVELOPMENT_EXPOSED'},
        {'dataset': 'AgentClinic-MedQA', 'source': 'official107; preexisting eligible92/excluded15', 'role': 'pilot3, conditionalTrain20, conditionalEval20; external paired20 reuse Eval source cases', 'reference': 'dataset-declared final diagnosis, lexical scoring', 'exposure': 'DEVELOPMENT_EXPOSED; model pretraining UNKNOWN'},
        {'dataset': 'M5 development', 'source': '173 patients/229 encounters', 'role': 'previous Frozen M5 results retained; no refit', 'reference': 'unreviewed Source Conditions', 'exposure': 'DEVELOPMENT_EXPOSED'},
        {'dataset': 'C4_31 VoI', 'source': '63 patients/92 encounters', 'role': 'previous development result retained', 'reference': 'Source agreement', 'exposure': 'DEVELOPMENT_EXPOSED'},
        {'dataset': 'C4_35 fresh synthetic', 'source': '35 patients/71 encounters', 'role': 'previous VoI result retained, H3 extension/H2 original separate', 'reference': 'limited export independence; Source agreement', 'exposure': 'NOW_DEVELOPMENT_EXPOSED'}]
    write(PUB / 'final_dataset_inventory.json', {'datasets': inventory, 'source_families': 2,
        'newly_generated_dataset_count': 0, 'original_MedQA_multiple_choice_raw': 'NOT_AVAILABLE',
        '_medqa_jsonl_identity': 'AgentClinic; not an additional independent MedQA dataset'})
    write(PUB / 'dataset_eligibility_and_exposure.json', {'datasets': inventory, 'routing_split': read(PUB / 'routing_train_eval_split.json'),
        'train_eval_source_component_overlap': 0, 'unverified_real_patient_linkage': True,
        'VoI_train_vs_routing_train_identity': 'different sources, linkage not available; cannot claim complete patient-level independence',
        'development_source_does_not_become_clinical_gold': True})
    mapping = ROOT / 'outputs/research_evaluation/c4_39_v1/final_formula_code_mapping.md'
    md(PUB / 'final_formula_code_mapping.md', '# FINAL EXP formula preservation\n\nExisting11-item mapping is retained verbatim below. New code changes model binding/evaluation only.\n\n' + mapping.read_text(encoding='utf-8'))
    write(PUB / 'final_formula_invariance_verification.json', {'protected': protected, 'protocol_pins': pinned,
        'Core_formula_changes': 0, 'Frozen_M5_SHA256': sha(ROOT / 'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'),
        'M5_refitting': 0, 'new_Memory_Routing_VoI_formula': 0, 'original11_mapping_sha256': sha(mapping)})
    status = ('FINAL_EXP_FULL_DEVELOPMENT_EVALUATED' if memory_ok and utility and routing['status'] == 'ROUTING_EVALUATED'
              else 'FINAL_EXP_MEMORY_AND_DIAGNOSIS_EVALUATED' if memory_ok and len(pilot_rows) == 20
              else 'FINAL_EXP_PARTIAL_EVALUATED' if attempted else 'FINAL_EXP_TECHNICAL_BLOCKED')
    limitations = ('# FINAL EXP research limitations\n\n'
        'Source agreement is not independently verified clinical accuracy. Original50, AgentClinic and all new evaluation groups are development exposed. '
        'The diagnostic pilot compares a stored old final decision to one new action at the same snapshot; a new nonterminal action is not a final diagnosis. '
        'The new Memory panel excludes the previous10 pilot patients but is a fixed20-case development sample; Case030 was previously exposed. '
        'Static external questions disclose the existing eligible vignette at once and do not measure interactive clinical safety. '
        'Source scenario duplicate components are separated, but source patient IDs and model-training overlap are unavailable. '
        'Source-QA Utility, when eligible, does not confer clinical Correctness approval or complete official B3/B4. '
        'No disease synonym or unresolved correctness was invented. Prompt byte guards can block full history and late turns. '
        'Sequential model order/cache affects timing, seed equality does not establish bitwise reproducibility, resource enforcement is between bounded calls.\n\n'
        'Preserved VoI: C4-35 H3 patient-macro AUBC52.0952%, Global EIG45.2381%, +6.8571pp, CI[1.9524,11.9524]pp. '
        'Budget5, Top3/NLL/Brier and Dynamic Horizon limitations remain; H2 is the original method, H3 an extension. '
        'Runtime5/5 agreement within5.551115123125783e-16 verifies numerical integration, not clinical utility; Hospital unresolved5/5.\n')
    md(PUB / 'final_research_limitations.md', limitations)
    md(PUB / 'final_completion_status.md', '# FINAL EXP status\n\n' + status + '\n\n' + json.dumps(ready, ensure_ascii=False, indent=2)
       + '\n\nRouting: ' + routing['status'] + '; blocker=' + str(routing.get('blocker_code')) + '\nActual generations=' + str(attempted) + '. No automatic rerun.\n')
    memory_md = table(memory_table, ['condition', 'registered', 'completed', 'diagnosis_source_exact', 'department_source_agreement', 'mean_actions', 'generation_count', 'prompt_tokens']) if memory_table else 'NOT_RUN'
    dx_md = table(dx_table, ['model', 'condition', 'registered', 'observed', 'source_exact', 'department_match', 'final_diagnose_count'])
    ext_md = table(external_table, ['dataset', 'model', 'registered', 'actual_received', 'source_exact', 'unresolved', 'failed', 'not_attempted'])
    oldtables = (ROOT / 'docs/연구 평가/FINAL_THESIS/FINAL_THESIS_RESULT_TABLES.md').read_text(encoding='utf-8')
    thesis_tables = ('# FINAL EXP thesis result tables\n\nNew results below supplement the preserved42 rows. These panels are not one matched B0–B4 experiment.\n\n'
        '## A Dataset and exposure\n\n' + table(inventory, ['dataset', 'source', 'role', 'reference', 'exposure']) +
        '\n\n## B Identical frozen diagnostic snapshots\n\n' + dx_md +
        '\n\n## C Preserved Memory search and prior results\n\nSee the original tables reproduced in the supplement below; search algorithm unchanged.\n\n'
        '## D New3B Memory comparison\n\n' + memory_md + '\n\nPaired differences (whole registered Source-success yield):\n\n```json\n' + json.dumps(paired, ensure_ascii=False, indent=2) + '\n```\n\n'
        '## E Routing and external Source QA\n\n' + ext_md + '\n\nRouting: ' + json.dumps(routing, ensure_ascii=False) +
        '\n\n## F Probability model and G VoI\n\nFrozen results retained. No refit or policy rerun. H3 extension is distinct from original H2.\n\n'
        '## H Runtime and integrated evaluation\n\nExisting Source-time Runtime5/5 numerical comparison retained. New official B3/B4 remain BLOCKED.\n\n'
        '## Preserved prior tables for supplementary use\n\n' + oldtables)
    md(DOC / 'FINAL_EXP_THESIS_RESULT_TABLES.md', thesis_tables)
    changes = [
        {'section': 'Title', 'change': 'REQUIRES_SUPERVISOR_APPROVAL', 'direction': 'Keep existing title pending approval of a narrower implemented/evaluated scope; do not claim clinically superior full Framework.'},
        {'section': 'Abstract', 'change': 'RESULT_UPDATED / CLAIM_QUALIFIED', 'direction': 'Report actual new denominators and development Source metrics; separate model gain from Memory gain.'},
        {'section': 'Introduction', 'change': 'UNCHANGED_METHOD', 'direction': 'Retain longitudinal information and decision-support motivation; qualify clinical claims.'},
        {'section': 'Related work', 'change': 'DATASET_EXPANDED', 'direction': 'Add actual AgentClinic external Source-QA role, not a second independent original MCQ MedQA dataset.'},
        {'section': '3.1 Dataset', 'change': 'DATASET_EXPANDED / EXPERIMENT_SETTING_UPDATED', 'direction': 'Describe Original50 subsets, fixed20 external cases and all development exposures separately.'},
        {'section': '3.2 Framework', 'change': 'UNCHANGED_METHOD / IMPLEMENTATION_DETAIL_UPDATED', 'direction': 'Keep Top5/Jaccard/Bayesian/Bellman/H2 formulas. State new3B binding and manifest-wrapper verification; Memory is context, not likelihood.'},
        {'section': '4.1 Conditions and metrics', 'change': 'EVALUATION_METRIC_ADDED', 'direction': 'Define raw exact, normalized exact, evidence-only concept match, unresolved and supported-yield denominators; no clinical accuracy.'},
        {'section': '4.2 Results', 'change': 'RESULT_UPDATED / CLAIM_QUALIFIED', 'direction': 'Use new tables B/D/E and preserved Probability/VoI; report both improvements and non-improvements.'},
        {'section': 'Table1', 'change': 'REQUIRES_SUPERVISOR_APPROVAL', 'direction': 'Preserve B0–B4 feature definitions but mark B3/B4 unexecuted in latest matched evaluation.'},
        {'section': 'Table2', 'change': 'RESULT_UPDATED / REQUIRES_SUPERVISOR_APPROVAL', 'direction': 'Replace unsupported integrated outcome claims with separate empirically grounded panels after scope approval; preserve historical values as historical, not reproducible current runs.'},
        {'section': 'Table3', 'change': 'EVALUATION_METRIC_ADDED / CLAIM_QUALIFIED', 'direction': 'Use Source precision/concept coverage trade-off; do not equate with independently reviewed relevance/stale precision.'},
        {'section': 'Conclusion', 'change': 'CLAIM_QUALIFIED', 'direction': 'Limit contributions to implementation, measured development effects and bounded synthetic VoI evidence; official clinical integrated superiority unproved.'},
        {'section': 'References', 'change': 'DATASET_EXPANDED', 'direction': 'Retain verified references and cite actual AgentClinic source/paper from existing C3 evidence; do not invent publication metadata.'}]
    md(DOC / 'FINAL_EXP_THESIS_CHANGE_MATRIX.md', '# FINAL EXP thesis change matrix\n\nThe original paper and nine FINAL_THESIS files remain unchanged. Historical90% and new AUBC52.10% have different metrics/cohorts and cannot replace each other.\n\n' + table(changes, ['section', 'change', 'direction']))
    # A compact working text, not a claim that a PDF has been typeset/reviewed.
    draft = ('# FINAL EXP 논문 수정용 압축 문안\n\n'
        '## 초록 수정\n\n본 연구는 종단 기록 검색, 유사도 기반 전문 역할 선택 및 유한 지평 정보가치 계획을 결합한 LoCDI의 구현과 평가 가능 범위를 조사하였다. '
        '기존 Top5 검색, Jaccard 가중 성공확률, Bayesian posterior 및 Bellman 계산은 유지하고 Qwen2.5-3B의 개발용 진단·Memory 평가를 추가하였다. '
        '실험별 Source와 평가 분모를 분리하였으며, Source 정답 일치를 독립적인 임상 정확도로 해석하지 않았다.\n\n'
        '## 실험 결과에 삽입할 실제 수치\n\n' + memory_md + '\n\n' + ext_md + '\n\n'
        '위 Memory 비교는 동일20환자의 B0/B2 개발 실험이며 기존1.5B10환자 평가와 한 matched 모델 비교로 합치지 않는다. '
        '동일 정보 모델 비교는 별도로 고정한 최종 공개 Snapshot에서 수행하였다.\n\n'
        'Routing 판정은 ' + routing['status'] + '이며, 차단 사유는 ' + str(routing.get('blocker_code')) + '이다. '
        '조건부 개발용 Utility가 생성되더라도 임상 Gate와 공식 B3/B4의 미완료 상태는 유지된다.\n\n'
        '기존 신규 합성35명/71방문의 사전 주 결과에서 H3의 환자 Macro AUBC는52.0952%, Global EIG는45.2381%로 차이6.8571pp였다(95% CI1.9524~11.9524pp). '
        '이는 제한된 합성 Source 일치 평가이며 Budget5 열위와 보조지표의 불확실성은 유지된다. 원래 H2와 확장 H3를 구분한다.\n\n'
        '## 한계와 결론 수정\n\n원본 Source의 임상 검토, 실제 환자 및 모델 학습 노출의 독립성, 전문과 선택의 임상 Reference와 공식 matched B0~B4 평가가 부족하다. '
        '연구의 의의는 구현된 수식과 실행 Trace를 검증 가능한 결과에 연결한 데 있으며, Framework 전체의 임상적 우월성을 입증한 것은 아니다.\n\n'
        '서론과 기존 수식 설명은 유지하고 위 문안을 해당 초록·실험·한계 문단에 반영한다. 제목 변경과 통합 결과표의 기능별 재구성은 교수님 확인이 필요하다. '
        '저자·소속·발표자·사사 정보는 사용자 확인 전 비워 둔다.\n')
    md(DOC / 'FINAL_EXP_THESIS_REVISED_DRAFT.md', draft)
    md(DOC / 'FINAL_EXP_THESIS_REVISION_GUIDE.md', '# FINAL EXP 논문 수정 안내\n\n'
        '기존 원문은 수정하지 않았다. Change Matrix와 실제 결과표에서 초록, §3.1, §3.2의 구현 세부, §4.1/§4.2, 표1~3, 결론에 필요한 부분만 반영한다. '
        '방법론을 다시 쓰거나 역사적90%를 AUBC52.10%로 치환하지 않는다. 모델 변경, 데이터 확대, 지표 추가와 연구 범위 변경을 구분한다.\n\n'
        'Downloads의 DocForm_2.docx XML에서 A4, 위30/아래20/좌우10mm, 제목·저자·요약1단/본문·참고문헌2단을 확인했다. '
        '지침은 참고문헌·부록·사사 포함2~3쪽,9pt 이상, PDF 제출이다. 사사는 해당 시1쪽 하단 각주다. '
        '이번 산출물은 Markdown 수정 자료이며 Word/PDF 조판이나3쪽 적합성을 검증한 최종 제출본이 아니다. '
        '본문에는 Memory/외부QA/VoI의 핵심 표만, 나머지 재현성·코드·상세 Trace는 보충자료로 둔다.\n\n'
        '다양한 데이터 활용 의견은 이번 사용자 진술에 따른 USER_REPORTED 피드백이다. 원문 심사평·교수님 발언을 확보한 것으로 표현하지 않는다. '
        '출판본 제목, 기존 통합 성능 주장 축소 및 평가 구조 변경은 지도교수 확인 사항이다.\n')
    briefing = ('# FINAL EXP 교수님 보고\n\n기존 Top5·Routing 성공확률·Bayesian/Bellman 수식을 유지하면서 실제 모델 실행을 ' + str(attempted) + '회 수행했다. '
        '실제 모델과 할당 Context는 저장된 Identity 및 요청별 Checkpoint로 확인했다.\n\n'
        '## Memory 및 외부 평가\n\n' + memory_md + '\n\n' + ext_md + '\n\n'
        'Memory의 B0/B2 효과는 B0 중 계약 오류에 따른 Batch 중단으로 평가하지 못했다. 미실행 B2를0점으로 취급하지 않는다. '
        'Routing 최종 상태: ' + routing['status'] + '; 사유: ' + str(routing.get('blocker_code')) + '. '
        '독립 임상 Correctness와 공식 B3/B4는 미검증이다. 기존 VoI의 합성 AUBC 결과와 한계는 그대로 보존했다.\n\n'
        '현재 연구 범위를 구현·개발 Source 평가 및 제한된 합성 VoI 근거로 설명하고, 과거 통합 임상 우월성 주장을 축소해도 되는지 확인이 필요하다. '
        '새 결과가 낮거나 불완전한 항목도 그대로 보고한다.\n')
    md(DOC / 'FINAL_EXP_PROFESSOR_BRIEFING.md', briefing)
    md(DOC / 'FINAL_EXP_PROFESSOR_EMAIL_DRAFT.md', '# 교수님 이메일 초안\n\n## 진행 보고 초안\n\n'
        '교수님 안녕하세요. 기존 수식을 유지한 상태에서3B 모델의 Memory 및 외부 Source-QA 평가를 진행하였고, 요청별 실행 기록과 분리된 평가 분모를 확보했습니다. '
        '현재 완료 범위와 Routing의 적격성 제한을 정리해 첨부드릴 예정입니다. 임상 성능과 개발용 Source 일치를 구분하여 논문 수정에 반영하겠습니다.\n\n'
        '## 최종 수정 보고 초안\n\n교수님 안녕하세요. 최종 실제 결과와 연구 한계를 반영한 결과표 및 절별 수정 자료를 정리했습니다. '
        '최종 상태는 ' + status + '이며 실제 Generation은' + str(attempted) + '회입니다. '
        '공식 matched B0~B4와 독립 임상 Routing 검증은 완료되지 않았습니다. '
        '통합 우월성 주장 대신 기능별 실제 평가로 결과 절을 구성하고 제목·연구 범위를 조정하는 방향을 확인 부탁드립니다. '
        '저자·사사 정보를 확인한 뒤 출판 양식에 반영하겠습니다.\n\n두 문안은 초안이며 발송하지 않았다.\n')
    md(DOC / 'FINAL_EXP_MULTI_DATASET_REVIEW_RESPONSE.md', '# 다양한 데이터 활용 의견에 대한 대응\n\n'
        '사용자가 전달한 다양한 데이터 활용 의견에 따라 기존 Original50 종단 Source와 AgentClinic의 별도 정적 Source-QA를 활용하였다. '
        '기존 M5 개발173명/229방문, VoI 개발63명/92방문, 신규 합성35명/71방문은 각각 다른 역할과 분모로 유지했다. '
        '로컬 _medqa.jsonl은 AgentClinic이므로 별도 공식 객관식 MedQA 확보로 중복 집계하지 않았다. '
        '새 Source 생성이나 대규모 획득 없이 이미 보유한 데이터의 실제 평가 범위를 확대했다. 임상 Reference와 데이터 독립성의 제한은 해소됐다고 주장하지 않는다.\n\n'
        '실제 신규 외부 결과:\n\n' + ext_md)
    report = ('# FINAL EXP 3B Memory 및 Routing 평가 보고서\n\n' + status + '\n\n'
        '새 승인 Protocol `' + protocol['fingerprint'] + '`에 따라 실제 실행했으며 자동 재시도/Repair/Resume는 하지 않았다. '
        'Generation ' + str(attempted) + '회, 기록된 전역 예산 ' + str(finished['actual_generation_requests']) + '회, '
        '실제 wall ' + str(finished['wall_seconds']) + '초다. 실패/차단은 실제 원인 그대로 기록했다.\n\n'
        '## 설치 및 호환성\n\n3B Digest `' + DIGEST3 + '`와 실제 요청 후 Context32768을 검증했다. '
        '새 Ollama의 로컬 Manifest 래퍼와 원본 Manifest Digest 차이를 확인해, 원본 blob SHA와 config/layer 내용이 일치할 때만 인식하도록 llm_setup의 최소 호환 수정을 추가했다. '
        '기존 Core와 수식은 바꾸지 않았다. 실제 모델Weight 전체 재Hash나 Raw wire 복원을 주장하지 않는다.\n\n'
        '## 기존20건 및 동일 Snapshot 진단 비교\n\n기존20건은 원문·정규화 exact0건이며 근거 없는 질병 개념 대응은 REVIEW_REQUIRED로 유지했다.\n\n' + dx_md +
        '\n\n3B의 비진단 Action은 그 시점에 최종 진단을 하지 않은 것이며 새로운 전체 Episode 실패/정오로 바꾸지 않았다.\n\n'
        '## 새로운20환자 Memory\n\n' + memory_md + '\n\n' + json.dumps(paired, ensure_ascii=False, indent=2) +
        '\n\n실제 오류 분석:\n\n```json\n' + json.dumps(failures, ensure_ascii=False, indent=2) + '\n```\n\n'
        'B0는9완료/1실패/10미시도, B2는20미시도다. 따라서 Memory 추가 효과는 NOT_EVALUATED이며 paired bootstrap도 수행하지 않았다. '
        'JSON/HTTP 정상과 Core의 행동·Evidence 의미 계약 통과는 별개다. 응답을 자동 수정하거나 추가 호출하지 않았다.\n\n'
        '## Routing 및 외부 평가\n\n' + json.dumps(routing, ensure_ascii=False) + '\n\n' + ext_md +
        '\n\n동일 시나리오의 공개 Observation과5후보를 공유한다. 미해결 응답을 false로 바꾸거나 Agent별 결과를 복제하지 않았다. '
        'Routing 수집 완전성과 Source 정답 채점 가능성, 임상 Correctness는 서로 다른 조건이다.\n\n'
        '## 보존과 재현성\n\n' + str(protected['checked']) + '개 보호 파일 Hash 검사 ' + protected['status'] + ', 승인 전 고정' + str(pinned['checked']) + '개 Pin 검사 ' + pinned['status'] + '. '
        '기존42행과 최종 논문 수정9개를 보존했다. 관련 회귀184 PASS는 전체 저장소 테스트가 아니다. '
        '저장 Trace 재집계는 모델 재생성의 bitwise 재현성을 뜻하지 않는다.\n\n'
        '## 최종 범위와 다음 작업\n\nB0/B2는 위 개발 패널로 구분하며 B1은 FullHistory/Context 제한으로 생략, 공식 B3/B4는 차단 상태다. '
        '기존 VoI/M5/Runtime 결과와 이 실험을 공통 matched Ablation으로 혼합하지 않는다. '
        '새 연구 실행 단계나 자동 재실행은 제안하지 않는다. FINAL_EXP 폴더의 Change Matrix, 실제 표와 압축 문안을 이용해 논문을 수정하고 연구 범위·제목을 교수님께 확인한다.\n')
    answers = [
        ('기존20건 개념 일치', '원문/정규화 exact0; 검증된 동의어0, 나머지 REVIEW_REQUIRED. 임상적 동등성 부재를 단정하지 않음.'),
        ('동일 정보1.5B/3B 차이', '20최종 Snapshot의3B 출력은 test18/diagnose2, Source exact0. 전체 Episode 모델 비교가 아님.'),
        ('3B 진료과 개선', '일부 Snapshot의2개 진단 출력으로 임상/전체 Episode 개선을 확정할 수 없음; comparison CSV 참조.'),
        ('Memory 추가 진단 효과', 'B0 중단으로 B2 미실행, NOT_EVALUATED.'),
        ('Memory 진료과 효과', '동일한 관측완료 쌍이 없어 NOT_EVALUATED.'),
        ('Memory 확대 규모', '20명/40Episode 등록,10명 시도·9완료/1실패; B2 미시도. 계획20명을 완료20명으로 쓰지 않음.'),
        ('데이터 종류', '신규 실행은 Synthea 기반 Original50와 AgentClinic 두 Source 계열; 이전 VoI 패널은 별도 역할로 재사용.'),
        ('외부 실제 규모', '같은 고정20사례 ×2모델=40응답. 역할 Pilot3사례15응답은 다른 고정 Source 그룹.'),
        ('적격 Train 그룹', '실제 Routing Pilot scorable 그룹0. Train 수집 미진입.'),
        ('실제 Utility 추정', '실제 fitting0, Artifact 미생성; TEST_ONLY 수식 검증을 연구 학습으로 계수하지 않음.'),
        ('전문 역할 차이', 'Pilot15응답 Source correct1/incorrect5/unresolved9; 역할 효과를 추정할 적격 Train/Eval 부족.'),
        ('Adaptive 개선', '분리 Eval 수집 및 Utility 차단으로 NOT_EVALUATED. 가상 기준선 성능 없음.'),
        ('Coverage/Abstention', '실제 응답의 supported/unresolved와 Router abstention은 구분. 후자는 적격 Utility 없으므로 미평가.'),
        ('개발/임상 Routing 차이', 'Source-QA의 lexical agreement 계약이며 독립 임상 Reference/공식 Gate는 미승인 상태 유지.'),
        ('기존 VoI 보존', 'C4-35 H3 AUBC+6.8571pp와 CI, Budget5/보조지표/동일 생성기 제한 모두 보존.'),
        ('수식/Core 보존', '보호302개/승인Pin335개 검증. 최소 변경은 llm_setup Manifest 호환과 외부 평가 코드; Core 수식 변경0.'),
        ('B0~B4 완료 범위', '새 B0 부분완료, B2 미시도, B1 생략, B3/B4 차단. 기존 C4-39 완료 결과는 별도 보존.'),
        ('논문 수정 절', '초록·§3.1·§3.2 구현 세부·§4.1/4.2·표1~3·한계/결론. 변화 Matrix 참조.'),
        ('연구 근간', 'Top5/Jaccard/Bayesian/Bellman/H2 유지. 모델/Source/평가 범위 보완이며 새 수식 아님.'),
        ('연구 범위 변경', '통합 임상 우월성 대신 기능별 개발 근거로 주장 축소는 실질적 범위 변경이므로 교수님 확인 필요.'),
        ('새 실제 성과', '3B/Digest/실제 Context 연결,135호출·외부40응답·계약 실패 위치 및 제한된 Source 모델 차이 확보.'),
        ('남은 한계', '새 Memory paired 효과 미평가, Routing 미평가, 임상 Gold 부재, 개발 노출/환자 독립성 및 학습 중복 미확정.'),
        ('최종 논문 표', '원42행 보존 + 신규 관측/미평가 표 제공. 원문 PDF나 최종 조판본을 자동 수정하지 않음.'),
        ('재현성', 'Memory 저장 Trace와 외부55개 응답 재집계 EXACT_MATCH; 모델 재생성의 bitwise 재현성을 주장하지 않음.')]
    report += '\n## 필수 질문별 판정\n\n' + table([{'항목': a, '실제 판정': b} for a, b in answers], ['항목', '실제 판정'])
    report += '\n\n재현 명령은 `python -B scripts/finalize_final_3b_results.py`이며 모델 호출이 없다. 최종 산출물 존재 시 덮어쓰기를 거부한다. 원본 보고서9개와 과거 D7/완료 Artifact의 Code Pin은 소급 갱신하지 않았다.\n'
    md(ROOT / 'docs/연구 평가/FINAL_EXP_3B_MEMORY_ROUTING_EVALUATION_REPORT.md', report)
    # Bind the publication to actual records rather than regenerated predictions.
    private_index = {str(p.relative_to(ROOT)): sha(p) for p in PR.rglob('*') if p.is_file()}
    write(PR / 'final_trace_index.json', private_index)
    write(PUB / 'final_reproducibility_manifest.json', {'protocol_fingerprint': protocol['fingerprint'],
        'model_identifier': DIGEST3, 'original_1b5_identifier': read(PR / 'llm_config_1b5.json')['model_identifier'],
        'code_hashes': {p: sha(ROOT/p) for p in ['scripts/run_final_3b_evaluation.py', 'scripts/finalize_final_3b_results.py',
            'app/locdi/final_diagnosis_evaluation.py', 'app/locdi/final_3b_routing_evaluation.py', 'app/locdi/llm_setup.py']},
        'private_trace_index_sha256': sha(PR / 'final_trace_index.json'), 'original42_preserved': all_results[:42] == oldresults,
        'clinical_reference_not_created': True, 'source_agreement_reaggregation': collection_checks['status'],
        'memory_reaggregation': 'EXACT_MATCH' if memory_reaggregation_match else 'NOT_VERIFIED',
        'aggregate_command': 'python -B scripts/finalize_final_3b_results.py',
        'model_generation_not_bitwise_reproduced': True, 'old_C4_39_EXACT_MATCH_preserved': True,
        'DocForm_2_docx_SHA256': sha(Path.home() / 'Downloads/DocForm_2.docx'),
        'DocForm_2_doc_SHA256': sha(Path.home() / 'Downloads/DocForm_2.doc'),
        'doc_docx_binary_semantic_equivalence': 'NOT_CHECKED; format inspected from docx sibling',
        'original_paper_SHA256': sha(Path.home()/'Downloads/심사용+논문.pdf'),
        'original_paper_matches_preexisting_verified_SHA': sha(Path.home()/'Downloads/심사용+논문.pdf') == 'd28e51965188a22b6a87d9cb8a52d1dfde49d942400f31ef716badafe7eebc99'})
    verification = {'status': 'PASS' if gold_ok and boundary['status'] == 'PASS' and protected['status'] == 'PASS' and pinned['status'] == 'PASS'
        and collection_checks['status'] == 'EXACT_MATCH' and attempted == finished['actual_generation_requests'] else 'FAIL',
        'structured_private_gold_key_scan': gold_ok, 'Raw_wire_checked': False,
        'actual_source_visibility': 'existing Runtime projection; static QA all disclosed Source only',
        'independent_clinical_visibility_review': 'NOT_REVIEWED', 'protected_files': protected,
        'protocol_pins': pinned, 'received_response_reaggregation': collection_checks['status'], 'actual_information_boundaries': boundary,
        'actual_generation_requests': attempted, 'published_result_rows': len(all_results), 'new_rows': len(additions),
        'old42_unchanged': True, 'official_B3_B4': 'BLOCKED', 'final_status': status,
        'verification_scope': 'publication and preservation integrity, not all experimental success',
        'hypotheses': {'Memory_added_effect': 'NOT_EVALUATED', 'Original_snapshot_model': 'NO_GAIN_OBSERVED',
                       'external_source_QA': 'DESCRIPTIVE_GAIN_ONLY', 'Adaptive_Routing': 'BLOCKED'},
        'new_tests': '184 directly related regression PASS; not whole repository'}
    write(PUB / 'verification.json', verification)
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
