"""C4-30 read-only recorded-source projection; never infer missing outcomes."""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sqlite3

from app.locdi.voi_probability_validation import epoch, compact
from app.locdi.llm_setup import digest
from app.locdi.routing_data_reuse import DATABASE, SOURCE, BANK
from app.locdi.reference_readiness import source_path, COHORT

SYSTEM='CSV_OBSERVATIONS_SYSTEM_UNSPECIFIED'


def action_key(kind, code, unit, system=SYSTEM):
    return compact([kind,system,str(code),unit,'Observation'])


def finite(value):
    if isinstance(value,bool):return None
    try:x=float(value)
    except (TypeError,ValueError):return None
    return x if math.isfinite(x) else None


def select_first(entries):
    """Keep only earliest unambiguous recorded value; no averaging/imputation."""
    groups=defaultdict(list)
    for e in entries:groups[e['key']].append(e)
    selected={};reasons=Counter()
    for k,es in groups.items():
        t=min(e['available_minute'] for e in es);tie=[e for e in es if e['available_minute']==t]
        if len({compact(e['value']) for e in tie})!=1:
            reasons['CONFLICT_AT_FIRST_PROXY_TIME']+=1;continue
        selected[k]=min(tie,key=lambda e:e['event_id']);reasons['LATER_OR_DUPLICATE_MEASUREMENTS']+=len(es)-1
    return selected,dict(reasons)


def project(s, minute_limit=45):
    pid=s['patient_id'];eid=s['index_encounter']['encounter_id'];start=epoch(s['index_encounter']['start_time'])
    entries=[];rejected=Counter()
    def add(raw,kind,event_id,minute,timestamp,system=None):
        if raw.get('PATIENT')!=pid or raw.get('ENCOUNTER')!=eid:raise ValueError('SOURCE_IDENTITY_MISMATCH')
        value=finite(raw.get('VALUE'));category='numeric' if value is not None else 'categorical'
        if value is None:value=raw.get('VALUE')
        if value in (None,''):
            rejected['EMPTY_SOURCE_VALUE']+=1;return
        if minute<0 or minute>minute_limit:
            rejected['OUTSIDE_PROXY_WINDOW']+=1;return
        confirmed=s.get('diagnosis_confirmed_at')
        if confirmed is not None and epoch(timestamp)>epoch(confirmed):
            rejected['AFTER_EXPLICIT_DIAGNOSIS_CONFIRMATION']+=1;return
        key=action_key(kind,raw['CODE'],raw.get('UNITS'),system or SYSTEM)
        entries.append({'key':key,'value':value,'data_type':category,'event_id':event_id,'available_minute':minute,
            'recorded_at':timestamp,'source_record_fingerprint':digest(raw),'source_record':raw,
            'actual_available_at':None,'availability_basis':'RECORDED_TIME_PROXY','diagnosis_confirmation_time_known':confirmed is not None})
    for item in s.get('symptom_sources',[]):
        if not (item['kind']=='symptom' and item['evidence_grade']=='A' and item['visibility_allowed']
                and item['candidate_reason']=='explicit_patient_report_observation' and epoch(item['source_timestamp'])<=start):continue
        ref=item['source_metadata']['_source'];raw=ref['source_record']
        add(raw,'Ask',item['source_id'],0,item['source_timestamp'])
        if entries and entries[-1]['event_id']==item['source_id']:
            entries[-1].update(source_file=ref['source_file'],source_row=ref['source_row'])
    for e in s.get('source_events',[]):
        if e['event_type']!='observations' or e['action_type_candidate']!='Test':continue
        if e['patient_id']!=pid or e['encounter_id']!=eid:raise ValueError('SOURCE_IDENTITY_MISMATCH')
        upper=e['timestamp_upper_epoch_ms']
        if upper is None:
            rejected['UNKNOWN_RECORDED_TIME']+=1;continue
        if upper<start:
            rejected['PRE_ENCOUNTER_TEST_NOT_NEW_INFORMATION']+=1;continue
        ref=e['source_metadata'];raw=ref['source_record']
        add(raw,'Test',e['event_id'],max(0,(upper-start+59999)//60000),e['timestamp'],e['target'].get('system'))
        if entries and entries[-1]['event_id']==e['event_id']:
            entries[-1].update(source_file=ref['source_file'],source_row=ref['source_row'])
    values,repeats=select_first(entries);rejected.update(repeats)
    return {'patient_id':pid,'visit_id':eid,'case_id':s['case_id'],'values':values,
        'decision_time':s['index_encounter']['start_time'],'diagnosis_confirmed_at':s.get('diagnosis_confirmed_at'),
        'rejections':dict(rejected),'clinical_timeline':'NOT_CLAIMED; source-window projection, not executed clinical requests'}


def database_sources():
    """Only read the existing indexed contexts/events/source DB; no data creation."""
    db=sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        contexts=[json.loads(r[0]) for r in db.execute('SELECT payload FROM contexts ORDER BY cid')]
        events=defaultdict(list)
        for r, in db.execute('SELECT payload FROM events'):
            e=json.loads(r);events[e['encounter_id']].append(e)
        sources=defaultdict(list)
        for cid,text in db.execute('SELECT cid,payload FROM sources'):sources[cid].append(json.loads(text))
        for c in contexts:
            yield {'case_id':c['case_id'],'patient_id':c['patient_id'],'index_encounter':c['index_encounter'],
                'source_events':events[c['index_encounter']['encounter_id']], 'symptom_sources':sources[c['case_id']],
                'decision_time':c['decision_time'],'index_conditions':c['evaluation_only']['index_conditions']}
    finally:db.close()


def unique_reference(s, universe):
    labels=set()
    for r in s['index_conditions']:
        raw=r['source_record']
        if raw['PATIENT']!=s['patient_id'] or raw['ENCOUNTER']!=s['index_encounter']['encounter_id']:
            raise ValueError('REFERENCE_IDENTITY_MISMATCH')
        if raw['DESCRIPTION'] in universe:labels.add(raw['DESCRIPTION'])
    return next(iter(labels)) if len(labels)==1 else None


def csv_check(visits):
    """Validate every selected projected source row against actual raw CSV."""
    needed=defaultdict(dict)
    for v in visits:
        for e in v['values'].values():
            key=e['source_file'];row=e['source_row'];expected=e['source_record_fingerprint']
            if row in needed[key] and needed[key][row]!=expected:raise ValueError('CONFLICTING_SOURCE_ROW')
            needed[key][row]=expected
    checked=0;paths=[]
    for ref,expected in needed.items():
        path=source_path(ref);paths.append(path)
        with path.open(encoding='utf-8-sig',newline='') as f:
            for n,r in enumerate(csv.DictReader(f),2):
                if n in expected:
                    if digest(r)!=expected.pop(n):raise ValueError('RAW_CSV_SOURCE_MISMATCH')
                    checked+=1
        if expected:raise ValueError('RAW_CSV_ROW_MISSING')
    return checked,paths


def support(visits, keys, diagnoses):
    result=[]
    for key in keys:
        rows=[v for v in visits if key in v['values']];numeric=[v for v in rows if v['values'][key]['data_type']=='numeric']
        result.append({'key':key,'visits':len(rows),'patients':len({v['patient_id'] for v in rows}),
            'diagnosis_coverage':sum(any(v['diagnosis']==d for v in rows) for d in diagnoses),
            'diagnosis_visits':[sum(v['diagnosis']==d for v in rows) for d in diagnoses],
            'diagnosis_patients':[len({v['patient_id'] for v in rows if v['diagnosis']==d}) for d in diagnoses],
            'missing_rate':1-len(rows)/len(visits) if visits else None,'numeric_rows':len(numeric),
            'unique_numeric_values':len({v['values'][key]['value'] for v in numeric}),
            'A_source_exists':bool(rows),'B_count_estimation':'DEVELOPMENT_PROXY_ONLY' if len(numeric)==len(rows) and rows else 'NOT_ELIGIBLE_CURRENT_NUMERIC_PROTOCOL',
            'C_sequential_reveal':'OBSERVED_PANEL_PROXY_ONLY; no result for missing visits; clinical availability NOT_VERIFIED'})
    return result
