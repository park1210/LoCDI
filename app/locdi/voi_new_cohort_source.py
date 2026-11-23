"""C4-33 CSV-to-existing-Source adapter. No generation, imputation or fitting."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.locdi.llm_setup import digest
from app.locdi.voi_source_action_inventory import project, unique_reference, SYSTEM
from app.locdi.voi_information_improvement import bucket

VERSION = 'locdi.voi_new_cohort_source.c4_33.v1'


def epoch(text):
    """Portable UTC epoch arithmetic, including pre1970 Source history on Windows.

    Date-only Condition START is a date proxy, never a confirmed diagnosis time.
    """
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int((value.astimezone(timezone.utc) - datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds()*1000)


def load_csv(root):
    root = Path(root).resolve()
    result = {}
    for name, required in {
        'patients': {'Id'}, 'encounters': {'Id', 'PATIENT', 'START'},
        'conditions': {'PATIENT', 'ENCOUNTER', 'START', 'CODE', 'DESCRIPTION'},
        'observations': {'PATIENT', 'ENCOUNTER', 'DATE', 'CODE', 'VALUE', 'UNITS', 'TYPE'}
    }.items():
        path = root / 'csv' / (name + '.csv')
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError('SOURCE_PATH_ESCAPE')
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not required <= set(reader.fieldnames or []):
                raise ValueError('CSV_SCHEMA:' + name)
            result[name] = [(i, r) for i, r in enumerate(reader, 2)]
    return result


def identities(tables):
    pids = [r['Id'] for _, r in tables['patients']]
    visits = [r['Id'] for _, r in tables['encounters']]
    if any(not p for p in pids + visits) or len(set(pids)) != len(pids) or len(set(visits)) != len(visits):
        raise ValueError('DUPLICATE_OR_EMPTY_SOURCE_ID')
    patients = set(pids)
    enc = {r['Id']: r for _, r in tables['encounters']}
    for r in enc.values():
        if r['PATIENT'] not in patients:
            raise ValueError('ENCOUNTER_PATIENT_LINK')
        epoch(r['START'])
    for name in ('conditions', 'observations'):
        for _, r in tables[name]:
            if r['PATIENT'] not in patients:
                raise ValueError('SOURCE_PATIENT_LINK')
            if r['ENCOUNTER'] and (r['ENCOUNTER'] not in enc or enc[r['ENCOUNTER']]['PATIENT'] != r['PATIENT']):
                raise ValueError('SOURCE_ENCOUNTER_LINK')
    return patients, enc


def encounter_fingerprints(tables):
    """Full linked encounter content, excluding IDs/absolute dates; not each isolated BP."""
    _, enc = identities(tables)
    observations, conditions = defaultdict(list), defaultdict(list)
    for _, r in tables['conditions']:
        if r['ENCOUNTER']:
            conditions[r['ENCOUNTER']].append([r.get('SYSTEM'), r['CODE'], r['DESCRIPTION'],
                (epoch(r['START']) - epoch(enc[r['ENCOUNTER']]['START'])) // 60000])
    for _, r in tables['observations']:
        if r['ENCOUNTER']:
            observations[r['ENCOUNTER']].append([r['CODE'], r['UNITS'], r['TYPE'], r['VALUE'],
                (epoch(r['DATE']) - epoch(enc[r['ENCOUNTER']]['START'])) // 60000])
    return {eid: digest({'class': r.get('ENCOUNTERCLASS'),
                        'conditions': sorted(conditions[eid], key=digest),
                        'observations': sorted(observations[eid], key=digest)})
            for eid, r in enc.items() if observations[eid] and conditions[eid]}


def overlap(tables, old_patients, old_encounters, old_fingerprints):
    patients, encounters = identities(tables)
    copies = encounter_fingerprints(tables)
    hits = set(copies.values()) & set(old_fingerprints)
    return {'new_patients': len(patients), 'new_visits': len(encounters),
            'patient_overlap': len(patients & set(old_patients)),
            'encounter_overlap': len(set(encounters) & set(old_encounters)),
            'content_duplicate_encounters': sum(fp in hits for fp in copies.values()),
            'within_new_content_duplicates': len(copies) - len(set(copies.values())),
            'content_definition': 'linked condition/observation values and minute offsets, excluding IDs and absolute dates',
            'scope': 'known indexed and raw inventories; unknown external exposure NOT_CHECKED',
            'status': 'BLOCKED' if (patients & set(old_patients) or set(encounters) & set(old_encounters) or hits or len(copies) != len(set(copies.values()))) else 'PASS'}


def adapt(tables, root, model):
    """Only numeric explicit CSV values, same encounter, 0..45min recorded proxy.

    Pain score 72514-3 is a declared CSV Ask projection; it does not manufacture
    a dialogue answer or verify clinical availability. All other modeled keys
    are Test. Native symptom history is never promoted to an observation.
    """
    patients, enc = identities(tables)
    obs, cond = defaultdict(list), defaultdict(list)
    missing_link = Counter()
    for line, r in tables['conditions']:
        if r['ENCOUNTER']:
            cond[r['ENCOUNTER']].append(r)
        else:
            missing_link['UNLINKED_CONDITION'] += 1
    for line, r in tables['observations']:
        if r['ENCOUNTER']:
            obs[r['ENCOUNTER']].append((line, r))
        else:
            missing_link['UNLINKED_OBSERVATION'] += 1
    visits, ledger = [], []
    menu = set(model['provenance']['action_key_mapping'])
    bydiagnosis = defaultdict(Counter)
    for ordinal, (eid, encounter) in enumerate(sorted(enc.items()), 1):
        pid, start = encounter['PATIENT'], encounter['START']
        s = {'case_id': 'c4_33:' + digest([pid, eid])[:24], 'patient_id': pid,
             'index_encounter': {'encounter_id': eid, 'start_time': start},
             'symptom_sources': [], 'source_events': [],
             'index_conditions': [{'source_record': r} for r in cond[eid]]}
        label = unique_reference(s, model['diagnoses'])
        # All distinct conditions remain available privately for ambiguity review.
        for line, raw in obs[eid]:
            if raw['TYPE'].lower() != 'numeric':
                continue
            # Do not silently alias a newly supplied coding system to the old
            # CSV-unspecified identity used to train M5.
            if raw.get('SYSTEM') and raw['SYSTEM'] != SYSTEM:
                continue
            timestamp = epoch(raw['DATE'])
            ref = {'source_record': raw, 'source_file': str(Path(root).resolve() / 'csv/observations.csv'), 'source_row': line}
            event_id = 'csv:' + digest([str(root), line, raw])
            if raw['CODE'] == '72514-3':
                # Existing Ask projection only accepts explicit reports at/before arrival.
                s['symptom_sources'].append({'kind': 'symptom', 'evidence_grade': 'A',
                    'visibility_allowed': True, 'candidate_reason': 'explicit_patient_report_observation',
                    'source_timestamp': raw['DATE'], 'source_id': event_id, 'source_metadata': {'_source': ref}})
            else:
                s['source_events'].append({'event_type': 'observations', 'action_type_candidate': 'Test',
                    'patient_id': pid, 'encounter_id': eid, 'timestamp_upper_epoch_ms': timestamp,
                    'timestamp': raw['DATE'], 'target': {'system': SYSTEM}, 'event_id': event_id,
                    'source_metadata': ref})
        # Reuse the exact existing projection with portable timestamp parsing;
        # no Core/file change. This adapter is serial, not a concurrent runtime.
        with patch('app.locdi.voi_source_action_inventory.epoch', epoch):
            v = project(s)
        # No categorical conversion, no source labels in the policy-facing values.
        v['values'] = {k: e for k, e in v['values'].items() if k in menu and e['data_type'] == 'numeric'}
        for k, e in v['values'].items():
            bucket(e['value'], model['provenance']['binning_thresholds'][k])
        count = len(v['values'])
        reasons = []
        if label is None:
            reasons.append('NO_SINGLE_UNIVERSE_REFERENCE')
        if count != len(menu):
            reasons.append('INCOMPLETE_ACTION_PANEL')
        cls = label or 'NO_SINGLE_UNIVERSE_REFERENCE'
        bydiagnosis[cls].update(visits=1, any_action=int(count >= 1), at_least3=int(count >= 3), complete=int(count == len(menu)))
        ledger.append({'ordinal': ordinal, 'patient_id': pid, 'visit_id': eid, 'diagnosis': label,
                       'action_count': count, 'missing_keys': sorted(menu - set(v['values'])),
                       'complete': not reasons, 'reasons': reasons,
                       'condition_timing': 'UNREVIEWED_SOURCE_START_NOT_DIAGNOSIS_CONFIRMATION'})
        if label is not None:
            v['diagnosis'] = label
            visits.append(v)
    eligible = [r for r in ledger if r['diagnosis'] is not None]
    return visits, ledger, {'status': 'SOURCE_STRUCTURAL_ONLY', 'generated_patients': len(patients),
        'all_visits': len(enc), 'diagnosis_eligible_patients': len({r['patient_id'] for r in eligible}),
        'diagnosis_eligible_visits': len(eligible),
        'at_least1_action_visits': sum(r['action_count'] >= 1 for r in eligible),
        'at_least3_action_visits': sum(r['action_count'] >= 3 for r in eligible),
        'complete_panel_patients': len({r['patient_id'] for r in eligible if r['complete']}),
        'complete_panel_visits': sum(r['complete'] for r in eligible),
        'by_source_diagnosis': {k: dict(v) for k, v in bydiagnosis.items()},
        'missing_action_counts': dict(Counter(k for r in eligible for k in r['missing_keys'])),
        'excluded_reasons': dict(Counter(x for r in ledger for x in r['reasons'])),
        'unlinked_records': dict(missing_link), 'availability': 'RECORDED_TIME_PROXY',
        'clinical_reference': 'NOT_APPROVED', 'clinical_availability': 'NOT_VERIFIED'}
