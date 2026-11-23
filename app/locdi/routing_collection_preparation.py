"""Private, deterministic candidate preparation; never calls a model or fits.

Reference labels only stratify candidates and remain outside task payloads.
Proxy visibility remains unreviewed; explicit execution acknowledgement is not
a clinical/source review and cannot clear the existing training validator.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import sqlite3

from app.core.agent_registry import AgentRegistry
from app.core.longitudinal_memory import PatientMemoryStore, timestamp
from app.core.routing_utility import observed_tokens
from app.llm.prompt_loader import specialty_system_prompt
from app.locdi.llm_setup import ROOT, digest, load_setup, validate_setup
from app.locdi.routing_data_reuse import BANK, SOURCE, DATABASE, EXPERIMENTS, read, _milliseconds, _csv_references
from app.locdi.routing_training_setup import HOLDOUT, REGISTRY
from validation.routing_training_data import _leaves, validate_split_manifest, validate_training_data

DEPARTMENTS = ('cardiology', 'endocrinology', 'nephrology', 'neurology', 'pulmonology')
CATALOG = EXPERIMENTS / '30_source_enriched_study_50/output/locdi_50_study_v3_qwen/diagnosis_choices.json'
TAXONOMY = EXPERIMENTS / '31_completed_run_diagnosis/output/run_20261024_030710_analysis/label_departments.json'
CONFIG = ROOT / 'configs/development_specialty_llm.json'
ASSUMPTIONS = ['RECORDED_UPPER_BOUND_IS_RELEASE_PROXY', 'RECORDED_MEMORY_CUTOFF_IS_AVAILABILITY_PROXY',
               'EXISTING_SHARED_CATALOG_IS_DEVELOPMENT_ONLY', 'REFERENCE_REVIEW_DEFERRED_NO_CORRECTNESS']


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def select_candidates(bank, taxonomy, holdout):
    """Deterministic constrained search, rare departments first, no row cloning."""
    reasons = Counter(); pools = {d: [] for d in DEPARTMENTS}
    seen = set()
    for row in bank:
        key = (row['training_patient_id'], row['training_case_id'])
        if key in seen:
            reasons['DUPLICATE_PATIENT_CONTEXT'] += 1; continue
        seen.add(key)
        if key[0] in holdout:
            reasons['HOLDOUT_PATIENT'] += 1; continue
        dept = taxonomy.get(row['training_label'])
        if dept not in pools:
            reasons['UNMAPPED_SPECIALTY'] += 1; continue
        pools[dept].append(row)
    for pool in pools.values():
        pool.sort(key=lambda r: digest(['routing-pilot-v1', r['training_patient_id'], r['training_case_id']]))
    order = sorted(DEPARTMENTS, key=lambda d: (len({r['training_patient_id'] for r in pools[d]}), d))
    slots = [d for d in order for _ in range(3)]
    chosen = []
    def search(index, used):
        if index == len(slots):
            return True
        dept = slots[index]
        tried = set()
        for row in pools[dept]:
            pid = row['training_patient_id']
            if pid in used or pid in tried:
                continue
            tried.add(pid); chosen.append((dept, row))
            if search(index + 1, used | {pid}):
                return True
            chosen.pop()
        return False
    if not search(0, set()):
        raise ValueError('INSUFFICIENT_DISTINCT_PATIENTS_FOR_STRATIFIED_SPLIT')
    result = []
    for dept in DEPARTMENTS:
        rows = [r for d, r in chosen if d == dept]
        for index, row in enumerate(rows):
            result.append({'department': dept, 'split': 'train' if index < 2 else 'validation', 'bank': deepcopy(row)})
    selected = {r['bank']['training_case_id'] for r in result}
    used = {r['bank']['training_patient_id'] for r in result}
    excluded_visits = []
    for row in bank:
        if row['training_case_id'] not in selected:
            reason = 'OTHER_VISIT_SAME_PATIENT_RESERVED_TO_SAME_SPLIT' if row['training_patient_id'] in used else 'NOT_SELECTED_FIXED_PILOT_QUOTA'
            reasons[reason] += 1
            excluded_visits.append({'patient_id': row['training_patient_id'], 'context_id': row['training_case_id'], 'reason': reason})
    return result, dict(reasons), excluded_visits


def response_schema(catalog):
    return {'type': 'object', 'additionalProperties': False,
            'required': ['action', 'diagnosis', 'certainty', 'rationale'], 'properties': {
                'action': {'const': 'diagnose'}, 'diagnosis': {'enum': list(catalog) + ['Unresolved']},
                'certainty': {'enum': ['supported', 'unresolved']},
                'rationale': {'type': 'string', 'minLength': 1, 'maxLength': 1000}}}


def _snapshot(row, source, context, memory):
    """Only selected current source values. No labels/provenance in observation."""
    observation = {'chief_complaint': '', 'demographics': deepcopy(context['demographics']),
                   'elapsed_minutes': 1, 'current_symptoms': [], 'history': [], 'objective_findings': [],
                   'longitudinal_memory': memory}
    links = {}; excluded = 0
    ids = set(row['source_provenance']['observation_event_ids'])
    start = _milliseconds(context['index_encounter']['start_time'])
    def link(prefix, value, ref, available, basis, metadata):
        for pointer in _leaves(value, prefix):
            links[pointer] = {'source_ref': ref, 'available_minute': available, 'basis': basis,
                              'actual_release_status': 'NOT_RECORDED', 'review_status': 'NOT_REVIEWED', **metadata}
    for event in sorted(source['source_events'], key=lambda e: e['event_id']):
        if event['event_id'] not in ids:
            continue
        upper = event.get('timestamp_upper_epoch_ms')
        if (event['patient_id'] != row['training_patient_id'] or event['encounter_id'] != context['index_encounter']['encounter_id']
                or type(upper) is not int):
            raise ValueError('SOURCE_EVENT_BINDING_INVALID')
        available = max(0, (upper - start + 59999) // 60000)
        if available > 1:
            excluded += 1; continue
        evidence = event['evidence']
        text = '{}: {} {}'.format(event['target']['description'], evidence['value'], evidence.get('units') or '').strip()
        index = len(observation['objective_findings']); observation['objective_findings'].append(text)
        ref = event['source_metadata']
        link('/objective_findings/' + str(index), text, event['event_id'], available, 'RECORDED_UPPER_BOUND_PROXY',
             {'recorded_at': event['timestamp'], 'recorded_lower_epoch_ms': event['timestamp_lower_epoch_ms'],
              'recorded_upper_epoch_ms': upper, 'source_file': ref['source_file'], 'source_row': ref['source_row'],
              'source_record_fingerprint': digest(ref['source_record'])})
    for item in sorted(source['symptom_sources'], key=lambda e: e['source_id']):
        if item['source_id'] not in ids:
            continue
        if _milliseconds(item['source_timestamp']) > start:
            excluded += 1; continue
        raw = item['source_metadata']['_source']['source_record']
        if raw['PATIENT'] != row['training_patient_id'] or raw['ENCOUNTER'] != context['index_encounter']['encounter_id']:
            raise ValueError('SOURCE_REPORT_BINDING_INVALID')
        fact = {'question': raw['DESCRIPTION'], 'answer': str(raw['VALUE']) + ' ' + (raw.get('UNITS') or ''), 'answer_status': 'provided'}
        index = len(observation['history']); observation['history'].append(fact)
        ref = item['source_metadata']['_source']
        link('/history/' + str(index), fact, item['source_id'], 0, 'RECORDED_PATIENT_REPORT_PROXY',
             {'recorded_at': item['source_timestamp'], 'source_file': ref['source_file'], 'source_row': ref['source_row'],
              'source_record_fingerprint': digest(ref['source_record'])})
    for pointer in _leaves(observation):
        if pointer in links:
            continue
        if pointer.startswith('/longitudinal_memory/'):
            basis, ref = 'RECORDED_MEMORY_CUTOFF_PROXY', 'fixed_memory_snapshot'
        elif pointer.startswith('/demographics'):
            basis, ref = 'RECORDED_CONTEXT', row['training_case_id']
        else:
            basis, ref = 'COLLECTION_CONFIGURATION', 'collection-v1'
        links[pointer] = {'source_ref': ref, 'available_minute': 0, 'basis': basis,
                          'actual_release_status': 'NOT_RECORDED' if basis.endswith('PROXY') else 'NOT_APPLICABLE',
                          'review_status': 'NOT_REVIEWED' if basis != 'COLLECTION_CONFIGURATION' else 'NOT_APPLICABLE'}
        parts = pointer.split('/')
        if len(parts) > 3 and parts[1:3] == ['longitudinal_memory', 'records']:
            record = memory['records'][int(parts[3])]
            links[pointer].update(source_ref=record['source_id'], recorded_available_at=record['available_at'],
                                  recorded_at=record['observed_at'], cutoff=context['index_encounter']['start_time'])
    return observation, links, excluded


def _fixed_memory(connection, context, query):
    starts = {v['encounter_id']: v['start_time'] for v in context['past_encounters']}
    records, rejected = [], Counter()
    cursor = connection.execute('SELECT m.payload FROM case_memory c JOIN memory m ON c.rid=m.rid WHERE c.cid=?', (context['case_id'],))
    for raw, in cursor:
        r = json.loads(raw)
        visit = r['encounter_id']; cutoff = timestamp(context['index_encounter']['start_time'])
        if (r['patient_id'] != context['patient_id'] or visit not in starts
                or timestamp(starts[visit]) >= cutoff or timestamp(r['available_at']) > cutoff):
            rejected['IDENTITY_OR_CUTOFF'] += 1; continue
        record = {'memory_id': r['record_id'], 'patient_id': r['patient_id'], 'visit_id': visit,
                  'visit_started_at': starts[visit], 'observed_at': r['observed_at_upper_bound'],
                  'available_at': r['available_at'], 'source_id': r['record_id'],
                  'kind': 'objective_finding' if r['type'] == 'observations' else 'history',
                  'text': r['normalized_text'], 'supersedes': []}
        try:
            PatientMemoryStore([record])
        except ValueError:
            rejected['SOURCE_CHRONOLOGY_UNRESOLVED'] += 1; continue
        records.append(record)
    identity = {'patient_id': context['patient_id'], 'visit_id': context['index_encounter']['encounter_id'],
                'started_at': context['index_encounter']['start_time']}
    return PatientMemoryStore(records).retrieve(identity, query, mode='selective', top_k=5), dict(rejected)


def project_input(observation, catalog):
    """Shared clinical payload only; memory identities/source metadata removed."""
    visible = {k: deepcopy(observation[k]) for k in ('chief_complaint', 'demographics', 'elapsed_minutes',
               'current_symptoms', 'history', 'objective_findings')}
    visible['longitudinal_memory'] = {'mode': 'selective', 'records': [
        {'memory_ref': 'Memory%03d' % (i + 1), 'text': r['text'], 'status_as_of_visit': r['status_as_of_visit']}
        for i, r in enumerate(observation['longitudinal_memory'].get('records', []))]}
    return {'task': 'Single decision development assessment. Return only the requested JSON. Unresolved when evidence is insufficient.',
            'observation': visible, 'allowed_actions': ['diagnose'], 'diagnosis_catalog': list(catalog)}


def build_preparation():
    bank = read(BANK); source_rows = read(SOURCE); sources = {r['case_id']: r for r in source_rows}
    holdout = sorted({r['patient_id'] for r in read(HOLDOUT)})
    selected, reasons, excluded = select_candidates(bank, read(TAXONOMY), set(holdout))
    registry = AgentRegistry(read(REGISTRY)).snapshot(); config = load_setup(CONFIG)['llm']
    config = deepcopy(config); config['max_requests'] = 75  # Separate collection config, original stays unchanged.
    setup = validate_setup(config)
    if setup['static_readiness'] != 'READY':
        raise ValueError('STATIC_MODEL_BINDING_BLOCKED')
    agents = [a for a in registry['agents'] if a['role'] == 'specialty']
    if {a['department'] for a in agents} != set(DEPARTMENTS):
        raise ValueError('SPECIALTY_REGISTRY_SCOPE_MISMATCH')
    catalog = read(CATALOG)
    if not isinstance(catalog, list) or not catalog or len(catalog) != len(set(catalog)):
        raise ValueError('COMMON_CATALOG_INVALID')
    fingerprints = {a['agent_id']: setup['prepared']['prompt_fingerprints'][a['agent_id']] for a in agents}
    generation = {k: config[k] for k in ('model_identifier', 'temperature', 'seed', 'max_output_tokens')}
    generation['prompt_fingerprints'] = fingerprints
    manifest = {'schema_version': 'locdi.routing_split.v1', 'dataset_fingerprint': file_hash(HOLDOUT),
                'holdout_fingerprint': digest(holdout), 'patients': {'train': [], 'validation': [], 'test': holdout}, 'decisions': []}
    candidates, references, reservations = [], [], []
    csv_rows = _csv_references([s['bank'] for s in selected], sources)
    connection = sqlite3.connect(DATABASE.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        for i, entry in enumerate(selected):
            row = entry['bank']; source = sources[row['training_case_id']]
            context = json.loads(connection.execute('SELECT payload FROM contexts WHERE cid=?', (row['training_case_id'],)).fetchone()[0])
            if context['patient_id'] != row['training_patient_id'] or context['index_encounter'] != source['index_encounter']:
                raise ValueError('CONTEXT_BINDING_MISMATCH')
            for ref in row['source_provenance']['condition_rows']:
                if csv_rows.get((ref['source_file'], ref['source_row'])) != ref['source_record']:
                    raise ValueError('CONDITION_SOURCE_CHANGED')
            selected_ids = set(row['source_provenance']['observation_event_ids'])
            observation_refs = [e['source_metadata'] for e in source['source_events'] if e['event_id'] in selected_ids]
            observation_refs += [e['source_metadata']['_source'] for e in source['symptom_sources'] if e['source_id'] in selected_ids]
            for ref in observation_refs:
                if csv_rows.get((ref['source_file'], ref['source_row'])) != ref['source_record']:
                    raise ValueError('OBSERVATION_SOURCE_CHANGED')
            empty = {'mode': 'selective', 'records': []}
            provisional, _, _ = _snapshot(row, source, context, empty)
            query = ' '.join(provisional['objective_findings'] + [r['answer'] for r in provisional['history']])
            memory, rejected = _fixed_memory(connection, context, query)
            observation, ledger, future = _snapshot(row, source, context, memory)
            decision_id = 'routing-v1:' + context['index_encounter']['encounter_id'] + ':minute1'
            decision = {'patient_id': row['training_patient_id'], 'visit_id': context['index_encounter']['encounter_id'],
                'decision_id': decision_id, 'minute': 1, 'split': entry['split'], 'source_ref': row['training_case_id'],
                'observation': observation, 'registry_fingerprint': digest(registry), 'tokens': observed_tokens(observation),
                'generation': deepcopy(generation), 'allowed_actions': ['diagnose']}
            manifest['patients'][entry['split']].append(row['training_patient_id'])
            manifest['decisions'].append({k: decision[k] for k in ('patient_id', 'decision_id', 'split')})
            shared = project_input(observation, catalog)
            candidates.append({'case_ref': 'Candidate%03d' % (i + 1), 'stratum': entry['department'], 'decision': decision,
                'visit_started_at': context['index_encounter']['start_time'], 'visibility': ledger,
                'source_review_status': 'NOT_REVIEWED', 'reference_review_status': 'NOT_REVIEWED',
                'memory_fingerprint': digest(memory), 'memory_query_fingerprint': digest(query), 'memory_rejections': rejected,
                'shared_payload': shared, 'shared_input_fingerprint': digest(shared),
                'excluded_selected_future_events': future,
                'source_fingerprint': digest({'bank': row, 'source': source})})
            references.append({'case_ref': candidates[-1]['case_ref'], 'patient_id': decision['patient_id'], 'decision_id': decision_id,
                'source_condition_rows': deepcopy(row['source_provenance']['condition_rows']), 'declared_label': row['training_label'],
                'clinical_review_status': 'NOT_REVIEWED', 'correctness_status': 'NOT_REVIEWED', 'correct': None})
        owner = {p: s for s, ps in manifest['patients'].items() for p in ps}
        for row in bank:
            if row['training_patient_id'] in owner:
                reservations.append({'patient_id': row['training_patient_id'], 'context_id': row['training_case_id'], 'split': owner[row['training_patient_id']]})
    finally:
        connection.close()
    split = validate_split_manifest(manifest, holdout_patient_ids=holdout, dataset_fingerprint=file_hash(HOLDOUT))
    if split['status'] != 'VALID':
        raise ValueError('PATIENT_SPLIT_INVALID')
    pins = {name: file_hash(path) for name, path in {'bank': BANK, 'sources': SOURCE, 'holdout': HOLDOUT,
            'registry': REGISTRY, 'config': CONFIG, 'catalog': CATALOG, 'taxonomy': TAXONOMY}.items()}
    bundle = {'schema_version': 'locdi.routing_preparation.v1', 'purpose': 'development_routing_collection',
        'registry': registry, 'llm_config': config, 'assumptions': ASSUMPTIONS, 'catalog': catalog,
        'response_schema': response_schema(catalog), 'split_manifest': manifest, 'candidates': candidates,
        'patient_visit_reservations': reservations, 'selection_reasons': reasons,
        'excluded_bank_contexts': excluded, 'source_pins': pins, 'reference_status': 'NOT_REVIEWED', 'ready': False}
    bundle['fingerprint'] = digest(bundle)
    return bundle, references


def preparation_audit(bundle):
    """No original identities, clinical evidence, labels or hashes of evidence."""
    candidates = bundle['candidates']
    collection = {'schema_version': 'locdi.routing_collection.v1', 'purpose': 'routing_training', 'registry': bundle['registry'],
                  'split_manifest': bundle['split_manifest'], 'decisions': [c['decision'] for c in candidates], 'outcomes': [], 'judgments': []}
    sources = [{'source_ref': c['decision']['source_ref'], 'purpose': 'routing_training',
                **{k: deepcopy(c['decision'][k]) for k in ('patient_id', 'visit_id', 'decision_id', 'minute', 'observation')},
                'source_fingerprint': c.get('source_fingerprint'), 'visit_started_at': c.get('visit_started_at'),
                'visibility': {p: row['available_minute'] for p, row in c['visibility'].items()},
                'review': {'status': 'NOT_REVIEWED'}} for c in candidates]
    current = validate_training_data(collection, registry_manifest=bundle['registry'], source_catalog=sources, label_catalog=[],
        holdout_patient_ids=bundle['split_manifest']['patients']['test'], dataset_fingerprint=bundle['split_manifest']['dataset_fingerprint'])
    return {'schema_version': 'locdi.routing_preparation_audit.v1', 'status': 'dry_run', 'candidate_count': len(candidates),
        'train_patients': len(bundle['split_manifest']['patients']['train']), 'validation_patients': len(bundle['split_manifest']['patients']['validation']),
        'holdout_patients': len(bundle['split_manifest']['patients']['test']), 'holdout_overlap': len(set(
            bundle['split_manifest']['patients']['train'] + bundle['split_manifest']['patients']['validation']) & set(bundle['split_manifest']['patients']['test'])),
        'strata': {d: dict(Counter(c['decision']['split'] for c in candidates if c['stratum'] == d)) for d in DEPARTMENTS},
        'candidates': [{'case_ref': c['case_ref'], 'split': c['decision']['split'], 'stratum': c['stratum'],
                        'review_status': 'NOT_REVIEWED', 'memory_count': len(c['decision']['observation']['longitudinal_memory']['records'])} for c in candidates],
        'selection_reasons': bundle['selection_reasons'], 'bundle_fingerprint': bundle['fingerprint'],
        'planned_agent_tasks': len(candidates) * 5, 'global_request_limit': bundle['llm_config']['max_requests'],
        'generation_requests': 0, 'utility_fit_count': 0, 'complete_train_groups': current['complete_group_count'],
        'training_readiness': current['training_readiness'], 'research_readiness': False,
        'execution_blockers': ['EXPLICIT_EXECUTION_APPROVAL_REQUIRED', 'DEVELOPMENT_PROXY_ASSUMPTIONS_NOT_ACCEPTED', 'SERVER_IDENTITY_NOT_CHECKED'],
        'reference_review_status': 'NOT_REVIEWED', 'clinical_metrics': 'NOT_EVALUATED'}
