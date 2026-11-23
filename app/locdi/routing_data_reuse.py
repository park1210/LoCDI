"""Read-only reuse inventory. No labels, inputs, answers or Utilities created.

Only aggregate counts and controlled enums are public. Identity intersections
stay private. Missing review/release evidence can never authorize training.
"""
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import sqlite3

from app.locdi.llm_setup import ROOT, digest
from app.locdi.routing_training_setup import HOSPITAL, HOLDOUT, REGISTRY, training_preflight

SYNTHEA = ROOT.parent / '001_synthea-docker'
EXPERIMENTS = SYNTHEA / 'experiments'
BANK = EXPERIMENTS / '34_live_action_policy/output/live_action_policy_v1/training_bank.jsonl'
SOURCE = EXPERIMENTS / '30_source_enriched_study_50/output/locdi_50_study_v3_qwen/audit/training_sources.jsonl'
DATABASE = EXPERIMENTS / '21_clinical_source_annotations_1000/output/run_v2/annotation_store.sqlite'
HISTORICAL = HOSPITAL / 'outputs/action_replay_pipeline_20261024_004817_4f9741/routing_train.json'
OSCE = HOSPITAL / 'data/raw/_medqa.jsonl'


def read(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    return [json.loads(line) for line in text.splitlines() if line.strip()] if Path(path).suffix == '.jsonl' else json.loads(text)


def overlap(left, right):
    """Aggregate only; never export members or their potentially reversible hashes."""
    a, b = set(left), set(right)
    return {'left_unique': len(a), 'right_unique': len(b), 'intersection': len(a & b), 'union': len(a | b)}


def probe_comparison(versions, holdout):
    output, pairs = {}, []
    for name, rows in versions.items():
        output[name] = {'rows': len(rows), 'patients': len({r['patient_id'] for r in rows}),
                        'decisions': len({r['decision_id'] for r in rows}),
                        'holdout_overlap': len({r['patient_id'] for r in rows} & set(holdout)),
                        'status': 'REQUIRES_REVIEW', 'approved_training_groups': 0}
    names = list(versions)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = versions[first], versions[second]
            pairs.append({'versions': [first, second],
                'patients': overlap([r['patient_id'] for r in a], [r['patient_id'] for r in b]),
                'decisions': overlap([r['decision_id'] for r in a], [r['decision_id'] for r in b]),
                'observations': overlap([digest(r['observation']) for r in a], [digest(r['observation']) for r in b]),
                'full_rows': overlap([digest(r) for r in a], [digest(r) for r in b])})
    return {'versions': output, 'pairs': pairs,
            'union_patients': len({r['patient_id'] for rows in versions.values() for r in rows}),
            'union_decisions': len({r['decision_id'] for rows in versions.values() for r in rows})}


def _milliseconds(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('SOURCE_TIMEZONE_MISSING')
    return int(dt.timestamp() * 1000)


def _feature(code, units):
    return json.dumps([code, units], ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def bank_analysis(bank, *, holdout, sources, contexts, csv_records, taxonomy):
    """Compare recorded bank fields to independent archived source/context rows.

    A recorded interval upper bound is only a release proxy, not actual clinical
    receipt. No output evidence, patient identifier, diagnosis or feature value.
    """
    counts = Counter()
    visits, reviewed_candidates, patients, per_patient = set(), 0, set(), Counter()
    specialties = Counter()
    for row in bank:
        pid, cid = row['training_patient_id'], row['training_case_id']
        patients.add(pid); per_patient[pid] += 1
        source, context = sources.get(cid), contexts.get(cid)
        if source is None or context is None:
            counts['missing_source_or_context_rows'] += 1
            continue
        encounter = source['index_encounter']['encounter_id']
        visits.add(encounter)
        linked = (pid == source['patient_id'] == context['patient_id']
                  and cid == row['source_provenance']['context_id'] == context['case_id']
                  and source['index_encounter'] == context['index_encounter'])
        counts['patient_context_visit_link_matches'] += linked
        condition_ok = bool(row['source_provenance']['condition_rows'])
        for ref in row['source_provenance']['condition_rows']:
            raw = csv_records.get((ref['source_file'], ref['source_row']))
            match = (raw == ref['source_record'] and raw is not None
                     and raw.get('PATIENT') == pid and raw.get('ENCOUNTER') == encounter
                     and raw.get('DESCRIPTION') == row['training_label'])
            counts['condition_csv_row_matches'] += match
            counts['condition_csv_row_references'] += 1
            condition_ok = condition_ok and match
        counts['rows_with_condition_reference_matches'] += condition_ok
        counts['selected_label_matches_source_selection'] += row['training_label'] == source['selected_actual_target_diagnosis']
        dept = taxonomy.get(row['training_label'])
        specialties[dept if dept in {'cardiology', 'neurology', 'nephrology', 'endocrinology', 'pulmonology'} else 'UNMAPPED'] += 1
        event_index = {e['event_id']: e for e in source['source_events']}
        symptom_index = {e['source_id']: e for e in source['symptom_sources']}
        expected, late, missing = {}, False, False
        start = _milliseconds(context['decision_time'])
        for eid in row['source_provenance']['observation_event_ids']:
            event = event_index.get(eid)
            if event is not None:
                counts['test_event_references'] += 1
                identity = event['patient_id'] == pid and event['encounter_id'] == encounter
                counts['event_patient_visit_matches'] += identity
                lower, upper = event.get('timestamp_lower_epoch_ms'), event.get('timestamp_upper_epoch_ms')
                if type(lower) is int and type(upper) is int:
                    counts['recorded_interval_spans_decision'] += lower <= start <= upper
                    counts['upper_bound_after_decision'] += upper > start
                    counts['proxy_available_minute_one'] += max(0, (upper - start + 59999) // 60000) == 1
                    counts['nominal_timestamp_after_decision'] += _milliseconds(event['timestamp']) > start
                    late = late or upper > start
                else:
                    counts['timestamp_bounds_missing'] += 1
                evidence = event['evidence']
                ref = event.get('source_metadata', {})
                raw = csv_records.get((ref.get('source_file'), ref.get('source_row')))
                counts['test_observation_csv_row_matches'] += (raw is not None and raw == ref.get('source_record')
                    and raw.get('PATIENT') == pid and raw.get('ENCOUNTER') == encounter)
                expected[_feature(event['target']['code'], evidence.get('units'))] = float(evidence['value'])
                counts['actual_result_release_not_verified'] += 1
            elif eid in symptom_index:
                symptom = symptom_index[eid]
                raw = symptom['source_metadata']['_source']['source_record']
                ref = symptom['source_metadata']['_source']
                csv_row = csv_records.get((ref.get('source_file'), ref.get('source_row')))
                counts['symptom_observation_csv_row_matches'] += csv_row is not None and csv_row == raw
                counts['symptom_references'] += 1
                counts['symptom_patient_visit_matches'] += raw['PATIENT'] == pid and raw['ENCOUNTER'] == encounter
                counts['symptom_recorded_by_decision'] += _milliseconds(symptom['source_timestamp']) <= start
                expected[_feature(raw['CODE'], raw.get('UNITS'))] = float(raw['VALUE'])
            else:
                missing = True
                counts['unresolved_observation_references'] += 1
        counts['rows_with_later_recorded_upper_bound'] += late
        counts['numeric_profile_matches_source'] += not missing and expected == row['profile']['numeric']
        counts['demographics_match_context'] += (row['profile']['age'] == context['demographics'].get('age_at_index')
                                                 and row['profile']['sex'] == context['demographics'].get('sex'))
        counts['past_visit_references'] += len(context.get('past_encounters', []))
        reviewed_candidates += bool(linked and condition_ok and not missing)
    return {'status': 'REQUIRES_REVIEW', 'row_count': len(bank), 'unique_patients': len(patients),
            'unique_contexts': len({r['training_case_id'] for r in bank}), 'unique_visits': len(visits),
            'patient_with_multiple_contexts_count': sum(n > 1 for n in per_patient.values()),
            'extra_rows_beyond_one_per_patient': len(bank) - len(patients),
            'holdout_overlap_count': len(patients & set(holdout)),
            'specialty_source_label_counts': dict(sorted(specialties.items())), 'source_checks': dict(sorted(counts.items())),
            'source_linked_review_candidate_count': reviewed_candidates,
            'validated_decision_count': 0, 'approved_training_group_count': 0,
            'reference_clinical_review': 'NOT_REVIEWED', 'release_visibility': 'NOT_REVIEWED',
            'decision_reconstruction': 'REQUIRES_TRANSFORMATION'}


def _csv_references(bank, sources=None):
    wanted = {}
    for row in bank:
        for ref in row['source_provenance']['condition_rows']:
            wanted.setdefault(ref['source_file'], set()).add(ref['source_row'])
        if sources is not None:
            source = sources.get(row['training_case_id'], {})
            ids = set(row['source_provenance']['observation_event_ids'])
            refs = [e['source_metadata'] for e in source.get('source_events', []) if e['event_id'] in ids]
            refs += [e['source_metadata']['_source'] for e in source.get('symptom_sources', []) if e['source_id'] in ids]
            for ref in refs:
                wanted.setdefault(ref['source_file'], set()).add(ref['source_row'])
    records = {}
    for name, numbers in wanted.items():
        path = (SYNTHEA / name).resolve()
        if SYNTHEA.resolve() not in path.parents or not path.is_file():
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for index, row in enumerate(csv.DictReader(handle), 2):
                if index in numbers:
                    records[name, index] = row
    return records


def historical_analysis(rows, *, registry, probes, holdout):
    current = {a['department']: a for a in registry['agents'] if a['role'] == 'specialty'}
    # Historical agent -> dept comes from its own archived Registry, not ID edits.
    original = read(EXPERIMENTS / '29_prepared_study_50/output/locdi_50_study_v1/registry.json')
    binding = {a['agent_id']: a for a in original['agents']}
    changes, groups = [], {}
    for agent_id in sorted({r['agent_id'] for r in rows}):
        old = binding.get(agent_id)
        entries = [r for r in rows if r['agent_id'] == agent_id]
        target = current.get(old['department']) if old else None
        changes.append({'department': old['department'] if old else 'NOT_RECORDED',
            'recorded_models': sorted({r['model'] for r in entries}),
            'recorded_versions': sorted({r['version'] for r in entries}),
            'current_model_matches': bool(target) and all(r['model'] == target['model'] for r in entries),
            'current_version_matches': bool(target) and all(r['version'] == target['version'] for r in entries),
            'current_agent_id_matches': bool(target) and target['agent_id'] == agent_id})
    for row in rows:
        groups.setdefault((row['patient_id'], row['decision_id']), []).append(row)
    pindex = {(p['patient_id'], p['decision_id']): p for p in probes}
    return {'status': 'NOT_ELIGIBLE', 'rows': len(rows), 'unique_patients': len({r['patient_id'] for r in rows}),
            'decision_groups': len(groups), 'holdout_overlap': len({r['patient_id'] for r in rows} & set(holdout)),
            'bindings': changes, 'split_train_row_count': sum(r.get('split') == 'train' for r in rows),
            'missing_answer_count': sum(not r.get('answer') for r in rows),
            'error_audit_count': sum(a.get('status') == 'error' for r in rows for a in r.get('audit', [])),
            'same_observation_group_count': sum(len({digest(r['observation']) for r in group}) == 1 for group in groups.values()),
            'original_registry_complete_group_count': sum({r['agent_id'] for r in g} == {a['agent_id'] for a in original['agents'] if a['role'] == 'specialty'} for g in groups.values()),
            'original_probe_observation_matched_group_count': sum(k in pindex and all(r['observation'] == pindex[k]['observation'] for r in g) for k, g in groups.items()),
            'actual_digest_status': 'NOT_RECORDED', 'actual_decoding_status': 'NOT_RECORDED',
            'clinical_review': 'NOT_REVIEWED', 'approved_training_group_count': 0}


def audit_existing_data():
    """Resolve exact local paths, read only, return sanitized analysis."""
    required = {'bank': BANK, 'sources': SOURCE, 'database': DATABASE, 'historical': HISTORICAL,
                'osce': OSCE, 'holdout': HOLDOUT, 'registry': REGISTRY,
                'original_registry': EXPERIMENTS / '29_prepared_study_50/output/locdi_50_study_v1/registry.json',
                'taxonomy': EXPERIMENTS / '31_completed_run_diagnosis/output/run_20261024_030710_analysis/label_departments.json'}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {'status': 'BLOCKED', 'missing': missing, 'classification': 'MISSING', 'approved_training_group_count': 0}
    holdout = {r['patient_id'] for r in read(HOLDOUT)}
    bank, source_rows, registry = read(BANK), read(SOURCE), read(REGISTRY)
    sources = {r['case_id']: r for r in source_rows}
    contexts, memory_counts = {}, Counter()
    # Explicit SQLite read-only URI. Never import/run archived builders.
    connection = sqlite3.connect(DATABASE.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        for row in bank:
            cid = row['training_case_id']
            hit = connection.execute('SELECT payload FROM contexts WHERE cid=?', (cid,)).fetchone()
            if hit:
                context = json.loads(hit[0]); contexts[cid] = context
                for item, in connection.execute('SELECT m.payload FROM case_memory c JOIN memory m ON c.rid=m.rid WHERE c.cid=?', (cid,)):
                    record = json.loads(item)
                    memory_counts['references'] += 1
                    memory_counts['patient_matches'] += record['patient_id'] == row['training_patient_id']
                    memory_counts['past_encounter'] += record['encounter_id'] != context['index_encounter']['encounter_id']
                    memory_counts['recorded_available_by_cutoff'] += _milliseconds(record['available_at']) <= _milliseconds(context['decision_time'])
    finally:
        connection.close()
    taxonomy = read(EXPERIMENTS / '31_completed_run_diagnosis/output/run_20261024_030710_analysis/label_departments.json')
    summary = bank_analysis(bank, holdout=holdout, sources=sources, contexts=contexts,
                            csv_records=_csv_references(bank, sources), taxonomy=taxonomy)
    summary['memory_source_counts'] = dict(memory_counts)
    versions, probe_missing = {}, []
    for name, relative in [('v29', '29_prepared_study_50/output/locdi_50_study_v1'),
                           ('v30_v2', '30_source_enriched_study_50/output/locdi_50_study_v2'),
                           ('v30_qwen', '30_source_enriched_study_50/output/locdi_50_study_v3_qwen')]:
        path = EXPERIMENTS / relative / 'routing_probes.json'
        if path.is_file():
            versions[name] = read(path)
        else:
            probe_missing.append(name)
    probes = probe_comparison(versions, holdout)
    bank_visits = {r['index_encounter']['encounter_id'] for r in sources.values() if r['case_id'] in {b['training_case_id'] for b in bank}}
    probes['probe_patients_in_bank'] = len({p['patient_id'] for v in versions.values() for p in v} & {b['training_patient_id'] for b in bank})
    probes['probe_visit_prefixes_in_bank'] = len({p['decision_id'].split(':')[0] for v in versions.values() for p in v} & bank_visits)
    osce = read(OSCE)
    osce_summary = {'status': 'REQUIRES_REVIEW', 'rows': len(osce), 'unique_full_record_hashes': len({digest(r) for r in osce}),
                    'declared_diagnosis_count': sum(bool(r.get('OSCE_Examination', {}).get('Correct_Diagnosis')) for r in osce),
                    'patient_identity_status': 'NOT_REVIEWED', 'test_overlap_status': 'NOT_REVIEWED',
                    'origin_status': 'NOT_REVIEWED', 'license_status': 'NOT_REVIEWED',
                    'independent_reference_status': 'NOT_REVIEWED', 'approved_training_group_count': 0}
    preflight = training_preflight()
    return {'schema_version': 'locdi.routing_reuse_audit.v1', 'status': 'READ_ONLY_AUDIT',
            'bank': summary, 'probes': probes, 'missing_probe_versions': probe_missing,
            'historical': historical_analysis(read(HISTORICAL), registry=registry, probes=versions.get('v29', []), holdout=holdout),
            'osce': osce_summary, 'holdout_patients': len(holdout),
            'training_preflight': {k: preflight[k] for k in ('training_readiness', 'complete_group_count', 'blocker_codes')},
            'approved_training_group_count': 0, 'llm_generation_count': 0, 'utility_fit_count': 0,
            'clinical_evaluation_count': 0, 'research_readiness': False}
