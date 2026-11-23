"""C4-07 deterministic private evidence sidecar. No clinical judgments or retrieval."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import timedelta
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.dataset_freeze import load, sha
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.memory_reference_evaluation import reject_hidden
from app.locdi.source_enrichment import validate_candidate, pointer
from app.core.longitudinal_memory import timestamp
from app.locdi.routing_data_reuse import DATABASE, SYNTHEA
from scripts.enrich_full_cohort_sources import DATA, write
from scripts.audit_memory_review_progress import check_hashes

VERSION = 'locdi.deterministic_pre_review.c4_07.v1'
TIME_FIELDS = {'conditions': 'START', 'observations': 'DATE', 'procedures': 'START',
               'medications': 'START', 'allergies': 'START', 'careplans': 'START',
               'imaging_studies': 'DATE', 'immunizations': 'DATE', 'devices': 'START', 'supplies': 'DATE'}


def check(rule, status, evidence=None):
    return {'rule_id': rule, 'result': status, 'evidence': evidence}


def numeric(value):
    if type(value) not in (str, int, float, Decimal) or value == '':
        return None
    try:
        n = Decimal(str(value))
        return n if n.is_finite() else None
    except InvalidOperation:
        return None


def compare_fact(source, fact):
    """Exact fields only. No synonym, unit conversion, normal range or clinical inference."""
    code = bool(source.get('code') and fact.get('code')) and str(source['code']) == str(fact['code'])
    term = bool(source.get('description') and fact.get('description')) and source['description'] == fact['description']
    units = bool(source.get('units') and fact.get('units')) and source['units'] == fact['units']
    system_known = bool(source.get('system') and fact.get('system'))
    system_same = system_known and source['system'] == fact['system']
    same_type = source.get('type') == 'observations' and fact.get('source_type') == 'observations'
    a, b = numeric(source.get('value')), numeric(fact.get('value'))
    missing = []
    if not system_known: missing.append('CODE_NAMESPACE_NOT_RECORDED')
    if not source.get('units') or not fact.get('units'): missing.append('EXPLICIT_UNITS_MISSING')
    if a is None or b is None: missing.append('NUMERIC_PAIR_NOT_AVAILABLE')
    if not code or not units or not same_type or (system_known and not system_same):
        comparable = 'NOT_APPLICABLE'
    else:
        comparable = 'PASS' if system_same and a is not None and b is not None else 'UNKNOWN'
    # Mechanical arithmetic only after all explicit comparison preconditions hold.
    return {'exact_code_text_match': code, 'exact_term_match': term, 'same_explicit_units': units,
            'same_observation_type': same_type, 'code_namespace_match': system_same if system_known else None,
            'comparability': comparable, 'source_value': source.get('value'), 'released_value': fact.get('value'),
            'units': source.get('units') if units else None,
            'released_minus_historical': str(b - a) if comparable == 'PASS' else None,
            'missing_evidence': missing, 'rule_id': 'M12_EXACT_FIELD_COMPARISON',
            'meaning': 'TEXT_AND_ARITHMETIC_ONLY_NOT_RELEVANCE_STALE_OR_CONFLICT'}


def memory_checks(record, identity, original, link, origin, csv_row, raw_digest_ok, by_id):
    checks = []
    checks.append(check('M01_PATIENT', 'PASS' if record['patient_id'] == identity['patient_id'] else 'FAIL'))
    checks.append(check('M02_PREVIOUS_VISIT', 'PASS' if record['visit_id'] != identity['visit_id'] else 'FAIL'))
    try:
        start, observed, available, cutoff = [timestamp(x) for x in
            (record['visit_started_at'], record['observed_at'], record['available_at'], identity['started_at'])]
        checks.append(check('M03_CHRONOLOGY', 'PASS' if start <= observed <= available else 'FAIL'))
        checks.append(check('M04_FIXED_MEMORY_CUTOFF', 'PASS' if start < cutoff and available <= cutoff else 'FAIL'))
    except (ValueError, TypeError):
        checks.extend([check('M03_CHRONOLOGY', 'FAIL'), check('M04_FIXED_MEMORY_CUTOFF', 'UNKNOWN')])
    checks.append(check('M05_ORIGINAL_RECORD', 'PASS' if original == record else 'FAIL'))
    checks.append(check('M06_ORIGINAL_FINGERPRINT', 'MISSING' if link is None else
                        'PASS' if link['record_fingerprint'] == digest(record) else 'FAIL'))
    checks.append(check('M07_SOURCE_RECORD', 'PASS' if origin is not None and csv_row is not None else 'MISSING'))
    if origin is None or csv_row is None:
        checks.append(check('M08_SOURCE_FINGERPRINT', 'MISSING'))
        checks.append(check('M09_SOURCE_TO_MEMORY', 'MISSING'))
    else:
        checks.append(check('M08_SOURCE_FINGERPRINT', 'PASS' if csv_row == origin['_source']['source_record'] and raw_digest_ok else 'FAIL'))
        description = csv_row.get('DESCRIPTION') or csv_row.get('BODYSITE_DESCRIPTION') or csv_row.get('MODALITY_DESCRIPTION', '')
        time_key = TIME_FIELDS.get(origin['type'])
        normalized = ' | '.join(str(v) for v in [csv_row.get(time_key, ''), origin['type'], description,
                                                csv_row.get('VALUE', ''), csv_row.get('UNITS', '')])
        same = (origin['patient_id'] == csv_row.get('PATIENT') == record['patient_id']
                and origin['encounter_id'] == csv_row.get('ENCOUNTER') == record['visit_id']
                and record['text'] == origin['normalized_text'] == normalized
                and record['source_id'] == origin['record_id']
                and origin['code'] == (csv_row.get('CODE') or csv_row.get('PROCEDURE_CODE'))
                and origin['description'] == description and origin['value'] == csv_row.get('VALUE')
                and origin['units'] == csv_row.get('UNITS')
                and record['observed_at'] == origin['observed_at_upper_bound']
                and record['available_at'] == origin['available_at']
                and origin['timestamp'] == csv_row.get(time_key))
        checks.append(check('M09_SOURCE_TO_MEMORY', 'PASS' if same else 'FAIL'))
        # Source time precision upper bound is structural, never clinical receipt time.
        try:
            raw_time = origin['timestamp']
            lower = timestamp(raw_time + 'T00:00:00+00:00' if len(raw_time) == 10 else raw_time)
            upper = lower + (timedelta(days=1, milliseconds=-1) if len(raw_time) == 10 else
                             timedelta(milliseconds=999) if '.' not in raw_time else timedelta(0))
            checks.append(check('M10_SOURCE_TIME_BOUND', 'PASS' if upper == timestamp(record['observed_at']) else 'FAIL'))
        except (TypeError, ValueError): checks.append(check('M10_SOURCE_TIME_BOUND', 'UNKNOWN'))
    refs = record.get('supersedes')
    valid_refs = isinstance(refs, list)
    for rid in refs or []:
        old = by_id.get(rid)
        try:
            valid_refs &= bool(old and old['patient_id'] == record['patient_id']
                               and timestamp(old['available_at']) <= timestamp(record['available_at']))
        except (ValueError, TypeError): valid_refs = False
    checks.append(check('M11_EXPLICIT_SUPERSEDES', 'FAIL' if not valid_refs else 'PASS' if refs else 'UNKNOWN',
                        {'explicit_relation_count': len(refs) if isinstance(refs, list) else None,
                         'empty_does_not_mean_no_change': True}))
    if origin is None: value_state = 'MISSING'
    elif origin.get('value') is None or origin.get('value') == '':
        value_state = 'MISSING' if origin.get('type') == 'observations' else 'NOT_APPLICABLE'
    else: value_state = 'PASS'
    checks.append(check('M13_EXPLICIT_SOURCE_VALUE', value_state))
    return checks


def query_checks(candidate, case):
    checks = []
    projection = candidate['projection']
    for rule, key in (('Q01_CHIEF_COMPLAINT', 'chief_complaint'), ('Q02_SYMPTOM_NARRATIVE', 'symptom_narrative')):
        checks.append(check(rule, 'PASS' if isinstance(projection.get(key), str) and projection[key].strip() else 'MISSING'))
    events = case['environment']['current_encounter_events']
    event_by_id = {e['event_id']: e for e in events}
    links = candidate['source_links']
    for action in ('Ask', 'Test'):
        checks.append(check('Q03_' + action.upper() + '_RELEASE', 'PASS' if any(l['action'] == action for l in links) else 'MISSING'))
    timing_ok = all(type(l.get('observed_minute')) in (int, float)
                    and 0 <= l['observed_minute'] <= candidate['decision_minute']
                    and l['event_id'] in event_by_id
                    and event_by_id[l['event_id']]['available_minute'] <= l['observed_minute'] for l in links)
    checks.append(check('Q04_REPLAY_RELEASE_TIME', 'PASS' if timing_ok else 'FAIL'))
    checks.append(check('Q05_EVENT_FINGERPRINT', 'PASS' if all(l['event_id'] in event_by_id and
                        digest(event_by_id[l['event_id']]) == l['source_fingerprint'] for l in links) else 'FAIL'))
    try:
        validate_candidate(candidate, events, case['agent_visible']['longitudinal'])
        reproduced = True
    except (ValueError, KeyError, TypeError): reproduced = False
    checks.append(check('Q06_SOURCE_VALUE_AND_QUERY_REPRODUCTION', 'PASS' if reproduced else 'FAIL'))
    try:
        reject_hidden(projection)
        safe_keys = set(projection) <= {'history', 'objective_findings'}
    except ValueError: safe_keys = False
    checks.append(check('Q07_HIDDEN_FIELD_WHITELIST', 'PASS' if safe_keys else 'FAIL'))
    actual = [l.get('actual_clinical_available_at') for l in links]
    checks.append(check('Q08_ACTUAL_CLINICAL_AVAILABLE_AT', 'PASS' if actual and all(actual) else 'MISSING'))
    checks.append(check('Q09_RECORDED_TIME_PROXY_DECLARED', 'PASS' if candidate.get('availability') ==
                        'OBSERVED_OFFLINE_REPLAY_WITH_DECLARED_RECORDED_TIME_ASSUMPTION' else 'UNKNOWN'))
    try:
        not_future = all(timestamp(f['recorded_at']) <= timestamp(candidate['decision_time']) for f in candidate['source_facts'])
    except (KeyError, ValueError, TypeError): not_future = False
    checks.append(check('Q10_SOURCE_FACT_NOT_FUTURE', 'PASS' if not_future else 'FAIL'))
    return checks


def classifications(checks):
    # A structural pass can coexist with missing evidence and mandatory clinical review.
    structural = [c for c in checks if c['rule_id'].startswith(('M0', 'M10', 'Q04', 'Q05', 'Q06', 'Q07', 'Q10'))]
    labels = ['REQUIRES_CLINICAL_REVIEW']
    if any(c['result'] == 'FAIL' for c in checks): labels.append('STRUCTURAL_FAIL')
    elif structural and all(c['result'] == 'PASS' for c in structural): labels.append('STRUCTURAL_PASS')
    if any(c['result'] in ('MISSING', 'UNKNOWN') for c in checks): labels.append('MISSING_EVIDENCE')
    if any(c['result'] == 'PASS' for c in checks): labels.append('OBJECTIVE_EVIDENCE_AVAILABLE')
    return labels


def safe_source(origin, permitted):
    """No whole raw row, final STOP/stopReason, or future record values in reviewer sidecar."""
    if origin is None or not permitted: return None
    keys = ('record_id', 'type', 'code', 'system', 'description', 'value', 'units',
            'timestamp', 'timing_precision', 'availability_basis')
    return {k: origin.get(k) for k in keys}


def compare_working(blank, working):
    if working is None: return {'status': 'NOT_FOUND'}
    preserved = (working.get('schema_version') == blank['schema_version']
                 and working.get('manifest_fingerprint') == blank['manifest_fingerprint']
                 and [r.get('snapshot_id') for r in working.get('snapshots', [])] ==
                     [r['snapshot_id'] for r in blank['snapshots']])
    rows_equal = preserved and all([r.get('memory_id') for r in a.get('records', [])] ==
                                  [r['memory_id'] for r in b['records']]
                                  for a, b in zip(working['snapshots'], blank['snapshots']))
    return {'status': 'STRUCTURE_MATCH' if preserved and rows_equal else 'STRUCTURE_DIFFERENT',
            'equals_blank': working == blank, 'clinical_fields_modified_by_script': 0,
            'clinical_approval_inferred': False}


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + '\n')


def run(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace): raise ValueError('INVALID_NAMESPACE')
    public = ROOT / 'outputs/research_evaluation' / namespace
    private = ROOT / 'outputs/private/research_evaluation' / namespace
    if public.exists() or private.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    prior = ROOT / 'outputs/private/research_evaluation/c4_05_v1'
    pins = load(prior / 'baseline_hashes.json')
    for name in ('c4_05_v1', 'c4_06_v1'):
        mp = ROOT / 'outputs/research_evaluation' / name / 'reports/output_manifest.json'
        m = load(mp)
        if digest({k: v for k, v in m.items() if k != 'fingerprint'}) != m['fingerprint']:
            raise ValueError('PRIOR_MANIFEST_FINGERPRINT')
        pins.update(m['files']); pins[mp.relative_to(ROOT).as_posix()] = sha(mp)
        if 'private_index_sha256' in m:
            ip = prior / 'private_index.json'
            if sha(ip) != m['private_index_sha256']: raise ValueError('PRIVATE_INDEX_HASH')
            pins.update(load(ip)['files']); pins[ip.relative_to(ROOT).as_posix()] = sha(ip)
    working_path = ROOT / 'outputs/private/research_evaluation/c4_06_reviews/partial_working_v1.json'
    if working_path.is_file(): pins[working_path.relative_to(ROOT).as_posix()] = sha(working_path)
    protected_count = check_hashes(pins)
    source_pins = load(prior / 'all_source_hashes.json')
    source_pins.update(load(prior / 'additional_source_asset_hashes.json'))
    check_hashes(source_pins)
    original = load_cases(DATA)
    data_hash = sha(DATA)
    if len(original) != 50: raise ValueError('ORIGINAL50_REQUIRED')
    bundle = prior / 'judged_pool_v1'
    manifest = load(bundle / 'manifest.json')
    if digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
        raise ValueError('POOL_MANIFEST_FINGERPRINT')
    packets = []
    for entry in manifest['packets']:
        if Path(entry['file']).name != entry['file'] or sha(bundle / entry['file']) != entry['sha256']:
            raise ValueError('PACKET_HASH')
        p = load(bundle / entry['file'])
        if digest(p) != entry['packet_fingerprint']: raise ValueError('PACKET_FINGERPRINT')
        packets.append(p)
    expected = ['OriginalCase%03d' % i for i in range(1, 51)]
    if [p['snapshot_id'] for p in packets] != expected: raise ValueError('ALL50_ORDER_REQUIRED')
    blank = load(bundle / 'review_results_blank.json')
    working = load(working_path) if working_path.is_file() else None
    working_check = compare_working(blank, working)
    link_map = {}
    wanted_ids = {r['memory_id'] for p in packets for r in p['candidate_memories']}
    for line in (prior / 'original_memory_links.jsonl').open(encoding='utf8'):
        row = json.loads(line)
        if row['memory_id'] in wanted_ids: link_map[row['memory_id']] = row
    db_sha = sha(DATABASE)
    source_pins[str(DATABASE.resolve())] = db_sha
    con = sqlite3.connect(DATABASE.resolve().as_uri() + '?mode=ro', uri=True)
    con.execute('PRAGMA query_only=ON')
    origins = {}; csv_wanted = defaultdict(set); registry_fps = {}
    try:
        for p in packets:
            for r in p['candidate_memories']:
                hit = con.execute('SELECT payload FROM memory WHERE rid=?', (r['source_id'],)).fetchone()
                origin = json.loads(hit[0]) if hit else None
                origins[r['memory_id']] = origin
                if origin:
                    path = (SYNTHEA / origin['_source']['source_file']).resolve()
                    if not path.is_relative_to(SYNTHEA.resolve()): raise ValueError('SOURCE_PATH_ESCAPE')
                    csv_wanted[path].add(origin['_source']['source_row'])
                    hit = con.execute('SELECT digest FROM raw_refs WHERE tab=? AND n=?',
                                      (origin['type'], origin['_source']['source_row'])).fetchone()
                    registry_fps[r['memory_id']] = hit[0] if hit else None
    finally: con.close()
    actual_rows = {}
    for path, numbers in csv_wanted.items():
        if not path.is_file(): continue
        source_pins[str(path)] = sha(path)
        with path.open(encoding='utf-8-sig', newline='') as f:
            for n, row in enumerate(csv.DictReader(f), 2):
                if n in numbers: actual_rows[(str(path), n)] = row
    records_out = []; queries_out = []; rules = defaultdict(Counter)
    categories = Counter(); query_categories = Counter(); compare_counts = Counter()
    for packet, case in zip(packets, original):
        identity = case['agent_visible']['longitudinal']
        qc = query_checks(packet['query_candidate'], case)
        if packet['private_identity'] != identity: qc.append(check('Q11_PACKET_IDENTITY', 'FAIL'))
        else: qc.append(check('Q11_PACKET_IDENTITY', 'PASS'))
        qlabels = classifications(qc)
        q = packet['query_candidate']
        qrow = {'schema_version': VERSION, 'snapshot_id': packet['snapshot_id'],
                'source_location': str(bundle / (packet['snapshot_id'] + '.json')) + '#/query_candidate',
                'query_fingerprint': q['query_fingerprint'], 'timestamp': q['decision_time'],
                'checks': qc, 'classifications': qlabels, 'source_links': q['source_links'],
                'missing_evidence': [x['rule_id'] for x in qc if x['result'] in ('MISSING', 'UNKNOWN')],
                'human_review_required': True}
        qrow['evidence_fingerprint'] = digest(qrow); queries_out.append(qrow)
        query_categories.update(qlabels)
        for c in qc: rules[c['rule_id']][c['result']] += 1
        all_records = [r for v in case['agent_visible']['past_visits'] for r in v['records']]
        by_id = {r['memory_id']: r for r in all_records}
        event_by_id = {e['event_id']: e for e in case['environment']['current_encounter_events']}
        released_facts = []
        query_safe = not any(c['result'] == 'FAIL' for c in qc)
        if query_safe:
            for link in q['source_links']:
                e = event_by_id[link['event_id']]; raw = json.loads(e['evidence'])
                for f in (raw['facts'] if e['action'] == 'Ask' else raw['records']):
                    clean = next((x for x in q['source_facts'] if x['code'] == f['code'] and
                                  x['recorded_at'] == f['recorded_at'] and x['description'] == f['description']), None)
                    if clean:
                        released_facts.append(dict(clean, system=f.get('system'), source_type='observations',
                                                   source_event_reference=link['event_id']))
        for i, r in enumerate(packet['candidate_memories']):
            origin = origins[r['memory_id']]
            source_path = (SYNTHEA / origin['_source']['source_file']).resolve() if origin else None
            csv_row = actual_rows.get((str(source_path), origin['_source']['source_row'])) if origin else None
            # Prior registry uses the same canonical JSON encoder as llm_setup.digest.
            fp_ok = csv_row is not None and digest(csv_row) == registry_fps.get(r['memory_id'])
            link = link_map.get(r['memory_id'])
            native = pointer(original, link['original_record_pointer']) if link else None
            checks = memory_checks(r, identity, native, link, origin, csv_row, fp_ok, by_id)
            labels = classifications(checks)
            permitted = 'STRUCTURAL_PASS' in labels and query_safe
            facts = safe_source(origin, permitted)
            comparisons = []
            if facts:
                for f in released_facts:
                    result = compare_fact(facts, f)
                    if result['exact_code_text_match'] or result['exact_term_match']:
                        result['released_fact_reference'] = {'event_id': f['source_event_reference'],
                                                             'recorded_at': f['recorded_at'], 'code': f['code']}
                        comparisons.append(result); compare_counts[result['comparability']] += 1
            row = {'schema_version': VERSION, 'snapshot_id': packet['snapshot_id'], 'memory_id': r['memory_id'],
                   'checks': checks, 'classifications': labels, 'source_value': facts,
                   'source_location': {'original_jsonl_record_pointer': link['original_record_pointer'] if link else None,
                                       'original_dataset_sha256': data_hash,
                                       'annotation_database': str(DATABASE), 'annotation_payload_fingerprint': digest(origin) if origin else None,
                                       'source_file': str(source_path) if origin else None,
                                       'source_file_sha256': source_pins.get(str(source_path)),
                                       'csv_row_number_header_is_1': origin['_source']['source_row'] if origin else None,
                                       'raw_row_fingerprint': digest(csv_row) if csv_row else None,
                                       'prior_registry_fingerprint': registry_fps.get(r['memory_id'])},
                   'timestamp': {k: r[k] for k in ('visit_started_at', 'observed_at', 'available_at')},
                   'cutoff': identity['started_at'], 'comparison_evidence': comparisons,
                   'comparison_scope': 'EXACT_CODE_OR_TERM_WITH_RELEASED_FACTS_ONLY',
                   'explicit_relation_count': len(r['supersedes']),
                   'missing_evidence': [c['rule_id'] for c in checks if c['result'] in ('MISSING', 'UNKNOWN')],
                   'human_review_required': True, 'rule_id': 'C4_07_RECORD_CLASSIFICATION_V1',
                   'clinical_judgments_generated': False, 'source_values_quarantined': not permitted}
            row['evidence_fingerprint'] = digest(row); records_out.append(row)
            categories.update(labels)
            for c in checks: rules[c['rule_id']][c['result']] += 1
    if len(records_out) != 989 or len(queries_out) != 50: raise ValueError('COHORT_COUNT_MISMATCH')
    if sum(len(v['records']) for c in original for v in c['agent_visible']['past_visits']) != 72590:
        raise ValueError('ORIGINAL_MEMORY_COUNT')
    # Check before any output; the full completed source snapshots/templates remain immutable.
    check_hashes(pins); check_hashes(source_pins)
    summary = {'status': 'C4_07_DETERMINISTIC_FACT_CLASSIFICATION_COMPLETED', 'schema_version': VERSION,
               'registered_queries': 50, 'registered_pool_records': 989, 'original_memory_records': 72590,
               'query_classifications': dict(query_categories), 'record_classifications': dict(categories),
               'rule_results': {k: dict(v) for k, v in sorted(rules.items())},
               'exact_match_comparisons': dict(compare_counts), 'source_csv_files': len(csv_wanted),
               'source_csv_rows_found': len(actual_rows), 'protected_file_count': protected_count,
               'protected_changed_or_missing': 0, 'source_assets_checked': len(source_pins),
               'working_file_comparison': working_check, 'clinical_review_required_records': 989,
               'clinical_fields_modified': 0, 'quality_evaluation': 'NOT_RUN', 'llm_generations': 0,
               'utility_fitting': 0, 'full_pool_recall': 'NOT_COMPUTED',
               'source_boundary': 'Only declared released current facts and cutoff-eligible historical fields; no raw STOP/final-state fields',
               'blinding': 'No selection-method membership, retrieval rank/score or predictions; exact-match hints may still influence reviewers'}
    jsonl(private / 'pre_review_evidence.jsonl', records_out)
    jsonl(private / 'query_evidence.jsonl', queries_out)
    write(private / 'protected_hashes.json', pins)
    write(private / 'source_hashes.json', source_pins)
    write(public / 'reports/classification_summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='c4_07_v1')
    args = parser.parse_args()
    def guard(event, args):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind'): raise RuntimeError('C4_07_OFFLINE_ONLY')
    sys.addaudithook(guard)
    result = run(args.namespace)
    print(json.dumps(result, ensure_ascii=True))
