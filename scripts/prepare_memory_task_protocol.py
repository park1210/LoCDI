"""C4-04 read-only task inventory and smaller blinded FULL_POOL review cohort.

No retrieval, relevance judgment, LLM, fitting, or partial scoring implementation.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.dataset_freeze import load, sha
from app.locdi.llm_setup import digest
from app.locdi.memory_reference_evaluation import load_bundle, released_query
from app.locdi.routing_data_reuse import BANK, SOURCE, DATABASE, read
from app.locdi.routing_collector import load_preparation

PARENT = ROOT/'outputs/private/research_evaluation/c4_02_v1/review_packets'
RULE = 'FULL_POOL_REVIEW_LOWEST_ELIGIBLE_RECORD_COUNT_THEN_CASE_ORDINAL_V1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def query_inventory(packets):
    rows = []
    for packet in packets:
        query, projection = released_query(packet['released_history'],
            snapshot_minute=packet['decision_minute'], source_event=packet['released_source_event'])
        if query != packet['query'] or projection != packet['query_input_projection']:
            raise ValueError('QUERY_REPRODUCTION_FAILED')
        facts = json.loads(packet['released_source_event']['evidence'])['facts']
        rows.append({'snapshot_id': packet['snapshot_id'], 'ordinal': packet['case_ordinal'],
            'source_status': 'SOURCE_STRUCTURALLY_VALID', 'clinical_status': 'QUERY_NOT_REVIEWED',
            'context_status': 'INSUFFICIENT_CLINICAL_CONTEXT',
            'context_interpretation': 'NO_CHIEF_COMPLAINT_OR_SYMPTOM_NARRATIVE_NOT_AN_EXPERT_INVALIDITY_VERDICT',
            'task_validity': 'UNDETERMINED_PENDING_INDEPENDENT_REVIEW',
            'chief_complaint_present': bool(packet['chief_complaint']),
            'query_fields': list(projection), 'source_fact_count': len(facts),
            'source_code_counts': dict(Counter(f['code'] for f in facts)),
            'decision_minute': packet['decision_minute'], 'query_fingerprint': packet['query_fingerprint'],
            'actual_clinical_release_observed': packet['source_visibility_evidence']['physical_clinical_release_observed'],
            'eligible_records': len(packet['eligible_memory']),
            'source_text_characters': sum(len(r['text']) for r in packet['eligible_memory']),
            'source_link_count': len(packet['memory_provenance'])})
    return rows


def select_cohort(rows, count=5):
    if type(count) is not int or not 5 <= count <= 10:
        raise ValueError('REVIEW_COHORT_MUST_BE_5_TO_10')
    eligible = [r for r in rows if r['source_status'] == 'SOURCE_STRUCTURALLY_VALID'
                and r['eligible_records'] > 0 and r['source_link_count'] == r['eligible_records']]
    if len(eligible) < count:
        raise ValueError('INSUFFICIENT_STRUCTURAL_REVIEW_CANDIDATES')
    chosen = sorted(eligible, key=lambda r: (r['eligible_records'], r['ordinal']))[:count]
    return sorted(chosen, key=lambda r: r['ordinal'])


def metric_protocol():
    return {'version': 'locdi.memory.review_protocol.c4_04.v1',
        'recommended': 'FULL_POOL', 'implemented_evaluator': 'C4_02_FULL_POOL_GATE_UNCHANGED',
        'full_pool': {'precision': '|S intersect R| / |S|', 'recall_at_5': '|S5 intersect R| / |R|',
            'full_history_recall': '|Sfull intersect R| / |R|', 'stale_rate': '|S intersect T| / |S|',
            'requirements': 'Every eligible record reviewed; unresolved annotation blocks cohort'},
        'partial_pool': {'status': 'DESIGN_ONLY_NOT_EXECUTABLE',
            'judged_set_precision': '|S intersect J intersect R| / |S intersect J|',
            'judged_coverage': '|S intersect J| / |S|',
            'known_relevant_retrieval_count': '|S intersect J intersect R|',
            'candidate_pool_agreement': '|SA intersect SB| / |SA union SB|',
            'full_pool_recall': 'FORBIDDEN',
            'sampling': 'After query approval: union of fixed lexical, recency and deterministic non-search samples; freeze before judgments; blind membership and rank',
            'selection_bias': 'Pooled methods and sample size favor represented candidates; cannot estimate full-pool recall'},
        'zero_denominator': 'NOT_APPLICABLE', 'unjudged': 'UNKNOWN_NEVER_FALSE',
        'symbols': 'E eligible; S returned; J evaluably judged; R reviewer relevant; T reviewer stale',
        'clinical_approval': False, 'quality_evaluation': 'NOT_RUN'}


def alternative_inventory():
    original = read(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    dm_path = ROOT/'datasets/versions/simulation_source_replay_v1/dataset_manifest.json'
    dm = load(dm_path)
    if digest({k:v for k,v in dm.items() if k != 'manifest_fingerprint'}) != dm['manifest_fingerprint']:
        raise ValueError('D3_MANIFEST_FINGERPRINT')
    pins = {dm_path.relative_to(ROOT).as_posix(): sha(dm_path)}
    for item in dm['files'].values():
        path = ROOT/item['path']
        if sha(path) != item['sha256']:
            raise ValueError('D3_FILE_HASH')
        pins[item['path']] = item['sha256']
    d3 = read(ROOT/dm['files']['cases.jsonl']['path'])
    initial = read(ROOT/dm['files']['initial_observations.jsonl']['path'])
    events = read(ROOT/dm['files']['action_events.jsonl']['path'])
    memories = read(ROOT/dm['files']['memory.json']['path'])['records']
    from validation.source_replay_validator import validate_source_bundle
    d3_validation = validate_source_bundle({'private': {
        name: read(ROOT/item['path']) for name, item in dm['files'].items()}})
    bank, sources = read(BANK), read(SOURCE)
    prepared = load_preparation('phase24b_candidates_v2')
    for p in [BANK, SOURCE, DATABASE, ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/manifest.json',
              ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json',
              ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/reference_candidates.json']:
        pins[str(p)] = sha(p)
    context_counts = Counter(); context_refs = []
    connection = sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro', uri=True)
    try:
        for row in bank:
            hit = connection.execute('SELECT payload FROM contexts WHERE cid=?', (row['training_case_id'],)).fetchone()
            if hit is None:
                context_counts['missing'] += 1
                continue
            context = json.loads(hit[0]); context_refs.append({'context_ref': row['training_case_id'], 'fingerprint': digest(context)})
            context_counts['linked'] += context['patient_id'] == row['training_patient_id']
            context_counts['with_initial_observations'] += bool(context['initial_observations'])
            context_counts['with_past_encounters'] += bool(context['past_encounters'])
            context_counts['explicit_chief_complaint'] += bool(context.get('chief_complaint'))
    finally:
        connection.close()
    a = {r['patient_id'] for r in original}; b = {r['training_patient_id'] for r in bank}; c = {r['patient_id'] for r in d3}
    split = prepared['split_manifest']['patients']
    train, validation, test = (set(split[k]) for k in ('train', 'validation', 'test'))
    obs = [r['decision']['observation'] for r in prepared['candidates']]
    summary = {
        'original': {'cases': len(original), 'patients': len(a),
            'chief_complaint_cases': sum(bool(r['agent_visible']['current_initial_state']['doctor_visible_info'].get('chief_complaint')) for r in original),
            'past_visits': sum(len(r['agent_visible']['past_visits']) for r in original),
            'memory_records': sum(len(v['records']) for r in original for v in r['agent_visible']['past_visits']),
            'action_counts': dict(Counter(e['action'] for r in original for e in r['environment']['current_encounter_events']))},
        'D3': {'cases': len(d3), 'patients': len(c), 'chief_complaint_cases': sum(bool(r['chief_complaint']) for r in initial),
            'canonical_validation': d3_validation,
            'current_symptom_cases': sum(bool(r['current_symptoms']) for r in initial),
            'action_counts': dict(Counter(e['action'] for e in events)),
            'ask_codes': dict(Counter(e['code'] for e in events if e['action']=='Ask')),
            'cases_with_test_candidate': len({e['case_id'] for e in events if e['action']=='Test'}),
            'value_types': dict(Counter(e['value_type'] for e in events)),
            'actual_release_count': sum(e['actual_available_at'] is not None for e in events),
            'memory_records': len(memories), 'memory_patients': len({r['patient_id'] for r in memories}),
            'status': 'STRUCTURAL_POST_ACTION_CANDIDATES_ONLY_NOT_VALIDATED_QUERY'},
        'bank': {'visits': len(bank), 'patients': len(b), 'numeric_profiles': sum(bool(r['profile']['numeric']) for r in bank),
            'explicit_chief_complaint': sum(bool(r.get('chief_complaint')) for r in bank),
            'source_rows': len(sources), 'source_symptom_kinds': dict(Counter(s['kind'] for r in sources for s in r['symptom_sources'])),
            'source_event_timing': dict(Counter(e['relative_to_decision'] for r in sources for e in r['source_events'])),
            'source_availability': dict(Counter(e['availability_status'] for r in sources for e in r['source_events'])),
            'context_counts': dict(context_counts), 'status': 'NUMERIC_PROFILE_WITH_UNREVIEWED_VISIBILITY_NOT_A_READY_RETRIEVAL_QUERY'},
        'prepared_decisions': {'cases': len(obs), 'chief_complaint': sum(bool(o['chief_complaint']) for o in obs),
            'current_symptoms': sum(bool(o['current_symptoms']) for o in obs),
            'history_cases': sum(bool(o['history']) for o in obs),
            'objective_finding_cases': sum(bool(o['objective_findings']) for o in obs),
            'objective_findings': sum(len(o['objective_findings']) for o in obs),
            'objective_visibility': dict(Counter(v['actual_release_status'] for r in prepared['candidates'] for k,v in r['visibility'].items() if k.startswith('/objective_findings/'))),
            'memory_scope': 'ALREADY_SELECTED_TOP5_NOT_FULL_POOL_REFERENCE', 'status': 'PREVIOUSLY_MODEL_EXPOSED_NUMERIC_OBSERVATIONS_NOT_INDEPENDENT_REFERENCE'},
        'patient_overlap': {'original_bank': len(a&b), 'original_D3': len(a&c), 'bank_D3': len(b&c),
            'routing_train_validation': len(train&validation), 'routing_train_holdout': len(train&test),
            'routing_validation_holdout': len(validation&test)},
        'clinically_validated_alternative_query_count': 0,
        'alternatives_conclusion': 'MORE_OBSERVATION_VALUES_EXIST_BUT_NO_REVIEWED_COMPLAINT_OR_RELEVANCE_QUERY',
        'no_diagnosis_or_future_observation_used_as_query': True}
    return summary, pins, context_refs


def prepare(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace):
        raise ValueError('INVALID_NAMESPACE')
    public = ROOT/'outputs/research_evaluation'/namespace
    private = ROOT/'outputs/private/research_evaluation'/namespace
    if public.exists() or private.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    parent, packets = load_bundle(PARENT)
    rows = query_inventory(packets); chosen = select_cohort(rows)
    selection = {'rule': RULE, 'selected_snapshot_ids': [r['snapshot_id'] for r in chosen],
        'purpose': 'QUERY_SUITABILITY_REVIEW_FIRST_NOT_APPROVED_BENCHMARK',
        'model_predictions_or_retrieval_outputs_used': False,
        'excluded': [{'snapshot_id': r['snapshot_id'], 'reason': 'HIGHER_FULL_POOL_REVIEW_BURDEN_UNDER_FIXED_RULE'}
                     for r in rows if r not in chosen],
        'selected_records': sum(r['eligible_records'] for r in chosen),
        'selected_text_characters': sum(r['source_text_characters'] for r in chosen),
        'original_records': sum(r['eligible_records'] for r in rows),
        'original_text_characters': sum(r['source_text_characters'] for r in rows)}
    public.mkdir(parents=True); private.mkdir(parents=True)
    write(public/'reports/cohort_preselection.json', selection)
    write(public/'reports/query_inventory.json', rows)
    alternative, pins, context_refs = alternative_inventory()
    write(public/'reports/alternative_inventory.json', alternative)
    write(private/'alternative_source_hashes.json', pins)
    write(private/'context_row_fingerprints.json', context_refs)
    write(public/'reports/metric_protocol.json', metric_protocol())
    target = private/'review_packets'; target.mkdir()
    selected_ids = set(selection['selected_snapshot_ids'])
    entries = [deepcopy(e) for e in parent['packets'] if e['snapshot_id'] in selected_ids]
    for e in entries:
        with (target/e['file']).open('xb') as stream:
            stream.write((PARENT/e['file']).read_bytes())
    manifest = deepcopy(parent); manifest.pop('fingerprint')
    manifest.update(packets=entries, cohort_rule=RULE,
        parent_manifest_fingerprint=parent['fingerprint'], review_protocol='C4_04_FULL_POOL_QUERY_REVIEW_FIRST_V1')
    manifest['source_hashes'][str(PARENT/'manifest.json')] = sha(PARENT/'manifest.json')
    manifest['source_hashes'][str(PARENT/'review_results_blank.json')] = sha(PARENT/'review_results_blank.json')
    manifest['fingerprint'] = digest(manifest)
    write(target/'manifest.json', manifest)
    blank = load(PARENT/'review_results_blank.json')
    blank['manifest_fingerprint'] = manifest['fingerprint']
    blank['snapshots'] = [r for r in blank['snapshots'] if r['snapshot_id'] in selected_ids]
    write(target/'review_results_blank.json', blank)
    load_bundle(target)
    write(public/'reports/preparation_result.json', {'status': 'C4_04_REFERENCE_PROTOCOL_PREPARED',
        'review_candidates': len(chosen), 'clinically_approved_queries': 0, 'reviewed_records': 0,
        'full_pool_record_count': selection['selected_records'], 'reference_status': 'UNREVIEWED',
        'quality_evaluation': 'NOT_RUN', 'partial_evaluator': 'NOT_IMPLEMENTED_DESIGN_ONLY',
        'retrieval_calls': 0, 'llm_generations': 0, 'utility_fitting': 0,
        'manifest_fingerprint': manifest['fingerprint']})
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='c4_04_v1')
    args = parser.parse_args()
    def guard(event, values):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):
            raise RuntimeError('C4_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    from unittest.mock import patch
    with patch('app.core.longitudinal_memory.PatientMemoryStore.retrieve', side_effect=AssertionError('NO_RETRIEVAL_DURING_REVIEW_PREPARATION')):
        print(json.dumps(prepare(args.namespace), ensure_ascii=True))
