"""D2 read-only source audit. No cases, clinical labels or probabilities made.

Public data are aggregate structural counts. Identifiable cohort relationships
are returned separately for an explicitly private manifest, never agent input.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from app.locdi.llm_setup import ROOT, digest
from app.locdi.routing_data_reuse import SYNTHEA, BANK, DATABASE, EXPERIMENTS, read
from app.locdi.routing_training_setup import HOLDOUT

COHORT = SYNTHEA / 'experiments/15_symptom_cohort_1000/output/runs/cohort_1000_json_v1'
CLASSES = ('emergency', 'urgentcare', 'outpatient', 'wellness')
PREPARATION = ROOT / 'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def millis(text):
    """Timezone-aware source instant only; date-only reference is separate."""
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('SOURCE_TIMEZONE_MISSING')
    return round(value.timestamp() * 1000)


def reference_timing(condition_start, encounter_start):
    """Compare recorded dates, NEVER infer availability or clinical validity."""
    if not condition_start or not encounter_start:
        return 'UNKNOWN'
    try:
        left = date.fromisoformat(condition_start[:10])
        right = date.fromisoformat(encounter_start[:10])
    except (ValueError, TypeError):
        return 'UNKNOWN'
    return 'AFTER_ENCOUNTER_DATE' if left > right else 'SAME_DATE_TIME_UNRESOLVED' if left == right else 'BEFORE_ENCOUNTER_DATE'


def symptom_asof(symptoms, encounters):
    """Use historical timeInfos values, never final resolved/lastUpdateTime.

    An active encoded symptom is a candidate, not an observed chief complaint,
    patient answer, onset time, or permission to publish it at arrival.
    """
    updates = []
    for name, symptom in symptoms.items():
        for module, source in symptom.get('sources', {}).items():
            for event in source.get('timeInfos', {}).values():
                when, value = event.get('time'), event.get('value')
                if type(when) in (int, float) and type(value) in (int, float):
                    updates.append((when, name, module, value))
    updates.sort()
    states = {}; cursor = 0; results = {}
    for eid, start in sorted(encounters, key=lambda x: (x[1], x[0])):
        while cursor < len(updates) and updates[cursor][0] <= start:
            when, name, module, value = updates[cursor]
            states[name, module] = (when, value); cursor += 1
        active = [(when, name) for (name, module), (when, value) in states.items() if value > 0]
        results[eid] = {'active_symptom_name_count': len({name for _, name in active}),
            'active_updated_at_encounter_count': sum(when == start for when, _ in active),
            'source_updates_after_cutoff': len(updates) - cursor}
    return results, len(updates)


def split_relationships(patients, memberships):
    """All encounters inherit patient membership; no new split is assigned."""
    raw = set(patients)
    excluded = set().union(*(memberships.get(k, set()) for k in ('holdout', 'bank', 'voi_train')))
    known = memberships.get('known_development', set())
    context = memberships.get('prepared_context_pool', set())
    memory = memberships.get('prepared_memory_pool', set())
    candidates = raw - excluded
    pairs = []
    names = sorted(memberships)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            pairs.append({'left': left, 'right': right,
                'intersection_patients': len(memberships[left] & memberships[right])})
    return {'raw_patients': len(raw), 'membership_counts': {k: len(v) for k,v in memberships.items()},
        'overlaps': pairs, 'new_after_holdout_bank_voi_exclusion': len(candidates),
        'new_after_known_development_exclusion': len(candidates-known),
        'new_outside_prepared_context_pool': len(candidates-known-context),
        'new_outside_prepared_context_and_memory_pools': len(candidates-known-context-memory),
        'approved_independent_validation_patients': 0,
        'independent_validation_status': 'NOT_ESTABLISHED_REVIEW_AND_EXPOSURE_HISTORY_REQUIRED'}, candidates


def _identities(value):
    if isinstance(value, dict):
        for k,v in value.items():
            if k in {'patient_id', 'training_patient_id'} and isinstance(v,str):
                yield v
            elif k == 'training_patients' and isinstance(v,list):
                yield from (p for p in v if isinstance(p,str))
            else:
                yield from _identities(v)
    elif isinstance(value,list):
        for v in value:
            yield from _identities(v)


def load_memberships():
    cases = read(HOLDOUT); bank = read(BANK); prep = read(PREPARATION)
    split = prep['split_manifest']['patients']
    memberships = {'holdout': {c['patient_id'] for c in cases},
        'bank': {b['training_patient_id'] for b in bank},
        'routing_train': set(split['train']), 'routing_validation': set(split['validation']),
        'voi_train': set(read(BANK.parent/'voi_model.json')['provenance']['training_patients'])}
    known = set().union(*memberships.values()); exposure_files = []
    versions = ('29_prepared_study_50/output/locdi_50_study_v1',
        '30_source_enriched_study_50/output/locdi_50_study_v2',
        '30_source_enriched_study_50/output/locdi_50_study_v3_qwen',
        '34_live_action_policy/output/live_action_policy_v1')
    for version in versions:
        for name in ('cases.json', 'routing_probes.json', 'voi_training_rows.json', 'voi_model.json'):
            path = EXPERIMENTS/version/name
            if not path.is_file():
                exposure_files.append({'path': 'SYNTHEA/experiments/'+version+'/'+name, 'status': 'NOT_FOUND'})
                continue
            ids = set(_identities(read(path))); known.update(ids)
            exposure_files.append({'path': 'SYNTHEA/experiments/'+version+'/'+name,
                'status': 'FOUND', 'patient_refs': len(ids), 'sha256': sha(path)})
    memberships['known_development'] = known
    with sqlite3.connect(DATABASE.as_uri()+'?mode=ro', uri=True) as connection:
        memberships['prepared_context_pool'] = {json.loads(r[0])['patient_id'] for r in connection.execute('SELECT payload FROM contexts')}
        # Only identities from the pre-existing index, not clinical memory text.
        memberships['prepared_memory_pool'] = {r[0] for r in connection.execute(
            "SELECT DISTINCT json_extract(payload, '$.patient_id') FROM memory") if r[0]}
    return memberships, exposure_files


def audit_raw(cohort, memberships, *, inspect_json=True):
    """Scan original CSVs and patient JSON; no transformations or judgments."""
    cohort = Path(cohort); sources = []; missing = []; counts = Counter()
    def rows(name):
        path = cohort/'csv'/name
        if not path.is_file():
            missing.append('csv/'+name); return
        before = path.stat()
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream); headers = reader.fieldnames; n = 0
            for number, row in enumerate(reader,2):
                n += 1; yield number, row
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('SOURCE_CHANGED_DURING_AUDIT')
        sources.append({'path': 'csv/'+name, 'status': 'FOUND', 'rows': n,
                        'fields': headers, 'bytes': after.st_size, 'sha256': sha(path)})
    patients = {r['Id'] for _,r in rows('patients.csv')}
    encounters = {}; by_patient = defaultdict(list)
    for number,r in rows('encounters.csv'):
        eid = r['Id']; start = millis(r['START'])
        if eid in encounters:
            raise ValueError('DUPLICATE_ENCOUNTER_ID')
        e = {'patient': r['PATIENT'], 'class': r['ENCOUNTERCLASS'].lower(), 'start': start,
             'start_text': r['START'], 'source_row': number, 'count': Counter(), 'observation_times': [],
             'condition_codes': set(), 'categories': set(), 'json_linked': False}
        encounters[eid] = e; by_patient[e['patient']].append(eid)
        counts['encounter_reason_populated'] += bool(r['REASONDESCRIPTION'].strip())
        counts['encounter_referral_word_match'] += any(t in r['DESCRIPTION'].lower() for t in ('referral','consultation','specialist'))
    def link(row):
        e = encounters.get(row.get('ENCOUNTER'))
        if e is None or e['patient'] != row.get('PATIENT'):
            counts['unlinked_or_wrong_patient_rows'] += 1; return None
        return e
    targets = defaultdict(lambda: {'rows': 0, 'patients': set(), 'encounters': set(),
        'candidate_encounters': set(), 'class_encounters': defaultdict(set), 'at_or_before_start': 0,
        'after_start': 0, 'empty_values': 0})
    _, candidates = split_relationships(patients, memberships)
    for _,r in rows('observations.csv'):
        e = link(r)
        if e is None: continue
        category = r['CATEGORY'] or 'UNCATEGORIZED'; e['categories'].add(category)
        c = e['count']; c['observation_rows'] += 1
        present = bool(r['VALUE'].strip())
        when = millis(r['DATE']); e['observation_times'].append(when)
        relation = 'at_or_before_start' if when <= e['start'] else 'after_start'
        c['observation_'+relation] += 1
        if present:
            c['value_'+category] += 1; c['value_'+category+'_'+relation] += 1
        c['reported_label_candidates'] += present and 'reported' in r['DESCRIPTION'].lower()
        c['reported_label_'+relation] += present and 'reported' in r['DESCRIPTION'].lower()
        key = (category, r['CODE'], r['DESCRIPTION'], r['UNITS'], r['TYPE'])
        t = targets[key]; t['rows'] += 1; t['patients'].add(e['patient']); t['encounters'].add(r['ENCOUNTER'])
        t['class_encounters'][e['class']].add(r['ENCOUNTER']); t[relation] += 1; t['empty_values'] += not present
        if e['patient'] in candidates: t['candidate_encounters'].add(r['ENCOUNTER'])
    condition_timing = Counter()
    for _,r in rows('conditions.csv'):
        e = link(r)
        if e is None: continue
        status = reference_timing(r['START'],e['start_text'])
        condition_timing[status] += 1; e['count']['condition_'+status] += 1
        e['count']['condition_rows'] += 1; e['condition_codes'].add((r['SYSTEM'], r['CODE']))
    referrals = Counter(); referral_types = Counter()
    for table, field in (('procedures.csv','DESCRIPTION'), ('careplans.csv','DESCRIPTION'),
                          ('imaging_studies.csv','MODALITY_DESCRIPTION')):
        for _,r in rows(table):
            e = link(r)
            if e is None: continue
            name = table[:-4]; e['count'][name+'_rows'] += 1
            if any(word in r[field].lower() for word in ('referral', 'consultation', 'specialist')):
                referrals[name] += 1; referral_types[name,r.get('CODE','')] += 1
    provider_specialties = Counter(r['SPECIALITY'] for _,r in rows('providers.csv'))
    claim_referring = 0
    for _,r in rows('claims.csv'):
        claim_referring += bool(r['REFERRINGPROVIDERID'].strip())
    # History counts require a prior encounter AND a recorded observation cutoff.
    for patient, ids in by_patient.items():
        starts = sorted(encounters[eid]['start'] for eid in ids)
        for eid in ids: encounters[eid]['observation_times'].sort()
        for eid in ids:
            e = encounters[eid]; e['count']['past_encounters'] = bisect_left(starts, e['start'])
            e['count']['past_recorded_observations'] = sum(
                bisect_right(encounters[old]['observation_times'],e['start']) for old in ids
                if encounters[old]['start'] < e['start'])
    json_files = []; json_counts = Counter(); json_keys = Counter(); symptom_names = set()
    for path in sorted((cohort/'json').glob('*.json')) if inspect_json else []:
        raw = path.read_bytes(); d = json.loads(raw)
        json_files.append({'file': 'json/'+path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        es = d.get('record',{}).get('encounters',[]); eligible = []
        pid = d.get('attributes',{}).get('id')
        for entry in es:
            e = encounters.get(entry.get('uuid'))
            if e is None:
                json_counts['json_encounters_not_in_csv'] += 1; continue
            if pid != e['patient']:
                raise ValueError('JSON_PATIENT_LINK_MISMATCH')
            e['json_linked'] = True; eligible.append((entry['uuid'],entry['start']))
            json_counts['encounter_start_precision_difference'] += entry['start'] != e['start']
            json_keys.update(entry.keys())
            json_counts['diagnostic_report_objects'] += len(entry.get('reports',[]))
            json_counts['explicit_consult_or_note_fields'] += sum(bool(entry.get(k)) for k in
                ('consults','consultations','consultNotes','referrals','notes'))
            json_counts['explicit_chief_complaint_fields'] += sum(bool(entry.get(k)) for k in
                ('chiefComplaint','chief_complaint'))
        syms = d.get('symptoms',{}); symptom_names.update(syms)
        asof, updates = symptom_asof(syms,eligible)
        json_counts['timestamped_symptom_updates'] += updates
        json_counts['patients_with_symptom_history'] += updates > 0
        for eid,s in asof.items():
            encounters[eid]['count'].update(s)
    if inspect_json and not json_files: missing.append('json/*.json')
    scopes = {'all': patients, 'new_after_train_holdout_exclusion': candidates,
              'new_outside_prepared_context_pool': candidates-memberships.get('known_development',set())-memberships.get('prepared_context_pool',set()),
              'new_outside_context_and_memory': candidates-memberships.get('known_development',set())-memberships.get('prepared_context_pool',set())-memberships.get('prepared_memory_pool',set())}
    groups = []
    for scope, selected in scopes.items():
        for cls in ('ALL',)+CLASSES:
            es = [e for e in encounters.values() if e['patient'] in selected and (cls=='ALL' or e['class']==cls)]
            totals = Counter(); conditions = set()
            for e in es: totals.update(e['count']); conditions.update(e['condition_codes'])
            count_keys = ('observation_rows','condition_rows','value_vital-signs','value_laboratory','value_survey',
                'value_imaging','value_exam','reported_label_candidates','observation_after_start','past_encounters',
                'past_recorded_observations','procedures_rows','careplans_rows','imaging_studies_rows',
                'condition_AFTER_ENCOUNTER_DATE')
            groups.append({'scope':scope, 'encounter_class':cls, 'patients':len({e['patient'] for e in es}),
                'encounters':len(es), 'json_linked_encounters':sum(e['json_linked'] for e in es),
                'distinct_condition_codes':len(conditions),
                'totals':{k:totals[k] for k in count_keys},
                'encounter_coverage':{k:sum(e['count'][k]>0 for e in es) for k in
                    ('value_vital-signs','value_laboratory','value_exam','value_imaging','value_survey',
                     'value_vital-signs_at_or_before_start','value_laboratory_at_or_before_start',
                     'reported_label_candidates','reported_label_at_or_before_start','active_symptom_name_count',
                     'active_updated_at_encounter_count','past_encounters','past_recorded_observations',
                     'condition_rows','condition_AFTER_ENCOUNTER_DATE')},
                'actual_doctor_release_verified':'NOT_REVIEWED', 'clinical_reference_approved':0,
                'source_chief_complaint_verified':0, 'research_ready':False})
    split, _ = split_relationships(patients,memberships)
    public = {'schema_version':'locdi.dataset_suitability.D2.v1','scope':'READ_ONLY_SOURCE_AUDIT',
        'status':'PARTIAL_SOURCE_MISSING' if missing else 'AUDIT_COMPLETE_NOT_CLINICALLY_REVIEWED',
        'sources':sources, 'missing_sources':missing, 'counts':dict(counts), 'raw_patients':len(patients),
        'raw_encounters':len(encounters),'encounter_classes':dict(Counter(e['class'] for e in encounters.values())),
        'groups':groups,'split':split, 'condition_date_relations':dict(condition_timing),
        'provider_specialties':dict(provider_specialties), 'referral_keyword_rows':dict(referrals),
        'nonempty_claim_referring_provider_rows':claim_referring,
        'json':{'reviewed_files':len(json_files),'bytes':sum(x['bytes'] for x in json_files),
            'file_manifest_fingerprint':digest(json_files),'counts':dict(json_counts),
            'encounter_field_counts':dict(json_keys),'distinct_symptom_names':len(symptom_names),
            'symptom_scope':'ASOF_ENCODED_STATE_NOT_CHIEF_COMPLAINT_OR_RELEASE_PROOF'},
        'test_targets':[{'category':key[0],'code':key[1],'description':key[2],'units':key[3],'value_type':key[4],
            **{k:len(t[k]) if isinstance(t[k],set) else t[k] for k in ('rows','patients','encounters','candidate_encounters','at_or_before_start','after_start','empty_values')},
            'encounters_by_class':{k:len(v) for k,v in t['class_encounters'].items()}}
            for key,t in sorted(targets.items())],
        'new_llm_generations':0,'utility_fit_count':0,'new_probability_estimates':0,
        'clinical_verdicts_generated':0,'research_readiness':False}
    private = {'schema_version':'locdi.dataset_split_relations.D2.v1','private':True,
        'not_agent_input':True,'assigned_new_splits':False,'json_files':json_files,
        'patients':[{'patient_id':pid,'memberships':sorted(k for k,v in memberships.items() if pid in v),
            'new_candidate_after_exclusion':pid in candidates,'assigned_new_split':None,
            'independent_review_status':'NOT_REVIEWED',
            'encounters':[{'encounter_id':eid,'source_row':encounters[eid]['source_row'],
                           'class':encounters[eid]['class']} for eid in sorted(by_patient.get(pid,[]))]}
            for pid in sorted(patients)]}
    return public, private


def audit_suitability():
    memberships, exposures = load_memberships()
    public, private = audit_raw(COHORT,memberships)
    public['exposure_source_files'] = exposures
    # Existing findings are cited as existing measurements, not new clinical clearance.
    for name in ('D1_inventory_v2.json','D1_inventory_v2_coverage.csv','D1_boundaries_v2.json'):
        path=ROOT/'outputs/reports'/name
        public.setdefault('prior_audit_sources',[]).append({'file':name,'status':'FOUND' if path.is_file() else 'NOT_FOUND',
                                                        'sha256':sha(path) if path.is_file() else None})
    private['public_summary_fingerprint'] = digest(public)
    return public, private
