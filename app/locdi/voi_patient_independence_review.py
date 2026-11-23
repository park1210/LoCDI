"""C4-35 structural Source independence review. No policy scores or fitting."""
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import csv
import json
import sqlite3
from unittest.mock import patch

from app.locdi.llm_setup import digest
from app.locdi.voi_new_cohort_source import epoch,identities,encounter_fingerprints
from app.locdi.voi_source_action_inventory import project,SYSTEM
from app.locdi.voi_information_improvement import bucket

VERSION='locdi.voi_patient_independence_review.c4_35.v1'
REQUIRED={'patients':{'Id'},'encounters':{'Id','PATIENT','START'},
    'conditions':{'PATIENT','ENCOUNTER','START','CODE','DESCRIPTION'},
    'observations':{'PATIENT','ENCOUNTER','DATE','CODE','VALUE','UNITS','TYPE'}}


class SourceIndex:
    """Private lossless CSV index; existing caches always opened read-only."""
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-16384')
        for name in REQUIRED:
            if {r[1] for r in self.db.execute('PRAGMA table_info('+name+')')}!={'line','pid','eid','data'}:
                raise ValueError('SOURCE_INDEX_SCHEMA')
    def close(self):self.db.close()
    def patients(self):return [r[0] for r in self.db.execute('SELECT pid FROM patients ORDER BY pid')]
    def tables(self,pid):
        t={name:[(line,json.loads(data)) for line,data in self.db.execute('SELECT line,data FROM '+name+' WHERE pid=? ORDER BY line',(pid,))] for name in REQUIRED}
        for name,rows in t.items():
            if any(not REQUIRED[name]<=set(r) for _,r in rows):raise ValueError('SOURCE_FIELDS_MISSING')
        identities(t)
        return t
    def encounter_map(self):return {eid:pid for eid,pid in self.db.execute('SELECT eid,pid FROM encounters')}


def build_index(root,path,progress=None):
    root=Path(root).resolve();path=Path(path)
    if path.exists():raise ValueError('NO_INDEX_OVERWRITE')
    db=sqlite3.connect(str(path));db.execute('PRAGMA cache_size=-16384')
    try:
        for name,required in REQUIRED.items():
            db.execute('CREATE TABLE '+name+' (line INTEGER PRIMARY KEY,pid TEXT NOT NULL,eid TEXT,data TEXT NOT NULL)')
            with (root/'csv'/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                if not required<=set(reader.fieldnames or []):raise ValueError('CSV_SCHEMA:'+name)
                batch=[];count=0
                for line,r in enumerate(reader,2):
                    pid=r['Id'] if name=='patients' else r['PATIENT']
                    eid=r['Id'] if name=='encounters' else r.get('ENCOUNTER')
                    batch.append((line,pid,eid,json.dumps(r,ensure_ascii=False,separators=(',',':'))))
                    if len(batch)>=2000:
                        db.executemany('INSERT INTO '+name+' VALUES (?,?,?,?)',batch);db.commit();count+=len(batch);batch=[]
                if batch:db.executemany('INSERT INTO '+name+' VALUES (?,?,?,?)',batch);db.commit();count+=len(batch)
            db.execute('CREATE INDEX '+name+'_pid ON '+name+' (pid)');db.commit()
            if progress:progress({'table':name,'rows':count})
        db.execute('CREATE UNIQUE INDEX unique_pid ON patients(pid)')
        db.execute('CREATE UNIQUE INDEX unique_eid ON encounters(eid)');db.commit()
        for name in ('encounters','conditions','observations'):
            if db.execute('SELECT count(*) FROM '+name+' s LEFT JOIN patients p ON s.pid=p.pid WHERE p.pid IS NULL').fetchone()[0]:raise ValueError('PATIENT_LINK')
        for name in ('conditions','observations'):
            if db.execute('SELECT count(*) FROM '+name+' s LEFT JOIN encounters e ON s.eid=e.eid WHERE s.eid IS NOT NULL AND s.eid!=\'\' AND (e.eid IS NULL OR e.pid!=s.pid)').fetchone()[0]:raise ValueError('ENCOUNTER_LINK')
    finally:db.close()


def patient_sets(cross_groups,within_groups,ledger,encounters):
    cross={eid for g in cross_groups for eid in g['new_visit_ids']}
    within={eid for g in within_groups for eid in g['new_visit_ids']}
    if not (cross|within)<=set(encounters):raise ValueError('COLLISION_ENCOUNTER_UNLINKED')
    A={encounters[e] for e in cross};B={encounters[e] for e in within}
    D={r['patient_id'] for r in ledger if r['diagnosis'] is not None}
    E={r['patient_id'] for r in ledger if r['complete']}
    sets={'A':A,'B':B,'C':A|B,'D':D,'E':E}
    intersections={a+'_and_'+b:sorted(sets[a]&sets[b]) for a in ('A','B','C') for b in ('D','E')}
    return sets,intersections,{'cross_encounters':cross,'within_encounters':within}


def full_history(tables):
    """ID-free complete *exported CSV* Condition/Observation/Encounter history.

    No match is not a proof of independence across different export windows.
    Does not claim all medications/procedures/native Person state were compared.
    """
    _,enc=identities(tables)
    patient=tables['patients'][0][1]
    birth_status='MISSING'
    try:
        anchor=epoch(patient['BIRTHDATE']) if patient.get('BIRTHDATE') else None
        if anchor is not None:birth_status='VALID_SOURCE_DATE'
    except (ValueError,TypeError,OverflowError,OSError):
        anchor=None;birth_status='INVALID_SOURCE_DATE_NO_CONVERSION'
    basis='BIRTHDATE_PROXY' if anchor is not None else 'FIRST_EXPORTED_ENCOUNTER'
    if anchor is None:anchor=min((epoch(r['START']) for r in enc.values()),default=0)
    items=[]
    for eid,r in enc.items():
        clinical={k:r.get(k) for k in ('ENCOUNTERCLASS','CODE','DESCRIPTION','REASONCODE','REASONDESCRIPTION')}
        items.append(['Encounter',(epoch(r['START'])-anchor)//60000,
                      (epoch(r['STOP'])-epoch(r['START']))//60000 if r.get('STOP') else None,clinical])
    for kind,timekey in [('conditions','START'),('observations','DATE')]:
        for _,r in tables[kind]:
            eid=r.get('ENCOUNTER')
            clinical={k:v for k,v in r.items() if k not in ('PATIENT','ENCOUNTER',timekey,'STOP')}
            if r.get('STOP'):clinical['stop_relative_minute']=(epoch(r['STOP'])-anchor)//60000
            items.append([kind,(epoch(r[timekey])-anchor)//60000,
                (epoch(enc[eid]['START'])-anchor)//60000 if eid else None,clinical])
    demo={k:patient.get(k) for k in ('GENDER','RACE','ETHNICITY')}
    fp=digest({'scope':'exported_encounter_condition_observation_history.v1','anchor_basis':basis,'demographics':demo,'items':sorted(items,key=digest)})
    return {'fingerprint':fp,'encounters':len(enc),'conditions':len(tables['conditions']),
        'observations':len(tables['observations']),'anchor':basis,'birthdate_status':birth_status,
        'scope':'all exported CSV encounters/conditions/observations only; other resource types not compared'}


def project_patient(tables,root,model):
    """All visits projected with original project; references audit-only, not scoring."""
    _,enc=identities(tables);obs=defaultdict(list);cond=defaultdict(list)
    for line,r in tables['observations']:
        if r['ENCOUNTER']:obs[r['ENCOUNTER']].append((line,r))
    for _,r in tables['conditions']:
        if r['ENCOUNTER']:cond[r['ENCOUNTER']].append(r)
    root=Path(root);source_file=str(root.resolve()/'csv/observations.csv')
    menu=model['provenance']['action_key_mapping'];out=[]
    for eid,r in sorted(enc.items()):
        pid=r['PATIENT'];start=r['START'];s={'patient_id':pid,'case_id':'c4_33:'+digest([pid,eid])[:24],
            'index_encounter':{'encounter_id':eid,'start_time':start},'source_events':[],'symptom_sources':[]}
        for line,raw in obs[eid]:
            if raw['TYPE'].lower()!='numeric' or (raw.get('SYSTEM') and raw['SYSTEM']!=SYSTEM):continue
            ref={'source_record':raw,'source_file':source_file,'source_row':line};event='csv:'+digest([str(root),line,raw])
            if raw['CODE']=='72514-3':
                s['symptom_sources'].append({'kind':'symptom','evidence_grade':'A','visibility_allowed':True,
                    'candidate_reason':'explicit_patient_report_observation','source_timestamp':raw['DATE'],'source_id':event,
                    'source_metadata':{'_source':ref}})
            else:
                s['source_events'].append({'event_type':'observations','action_type_candidate':'Test',
                    'patient_id':pid,'encounter_id':eid,'timestamp_upper_epoch_ms':epoch(raw['DATE']),
                    'timestamp':raw['DATE'],'target':{'system':SYSTEM},'event_id':event,'source_metadata':ref})
        with patch('app.locdi.voi_source_action_inventory.epoch',new=epoch):v=project(s)
        v['values']={k:e for k,e in v['values'].items() if k in menu and e['data_type']=='numeric'}
        labels={x['DESCRIPTION'] for x in cond[eid] if x['DESCRIPTION'] in model['diagnoses']}
        v['reference_candidates']=sorted(labels);v['diagnosis']=next(iter(labels)) if len(labels)==1 else None
        out.append(v)
    return out


def input_signatures(visit,model):
    keys=sorted(model['provenance']['action_key_mapping'])
    if not set(keys)<=set(visit['values']):return None
    raw=[];numeric=[];schedule=[];encoded=[]
    for k in keys:
        e=visit['values'][k];r=e['source_record'];bounds=model['provenance']['binning_thresholds'][k]
        raw.append([k,r.get('VALUE'),r.get('TYPE')]);numeric.append([k,e['value']])
        schedule.append([k,e['available_minute'],e.get('availability_basis')])
        encoded.append([k,bucket(e['value'],bounds)])
    return {'raw8':digest(raw),'numeric8':digest(numeric),'availability':digest(schedule),
        'raw8_and_availability':digest([raw,schedule]),'ternary':digest(encoded),
        'reference_and_raw8':digest([visit.get('diagnosis'),raw]) if visit.get('diagnosis') else None,
        'reference_and_ternary':digest([visit.get('diagnosis'),encoded]) if visit.get('diagnosis') else None,
        'actual_policy_disclosure_order':'NOT_RUN; availability signature is not a realized policy trace'}


def conservative_proposal(candidates,collision_patients,reuse_patients,input_duplicate_patients,global_blockers=()):
    """Returns a simulation of a proposed Gate, never an admission approval."""
    reasons=defaultdict(list)
    for pid in collision_patients:reasons[pid].append('ANY_ENCOUNTER_CONTENT_COLLISION_PATIENT_REVIEW_REQUIRED')
    for pid in reuse_patients:reasons[pid].append('FULL_EXPORTED_HISTORY_MATCH_REUSE_SUSPECTED')
    for pid in input_duplicate_patients:reasons[pid].append('EXACT_EVALUATION_RAW_INPUT_DUPLICATE')
    registered={r['patient_id'] for r in candidates}
    excluded=registered&set(reasons)
    proposed=[] if global_blockers else [r for r in candidates if r['patient_id'] not in excluded]
    return {'status':'PROPOSAL_ONLY_NOT_APPROVED','registered_patients':len(registered),'registered_visits':len(candidates),
        'proposed_patients':len({r['patient_id'] for r in proposed}),'proposed_visits':len(proposed),
        'excluded_patients':sorted(excluded),'excluded_visits':sum(r['patient_id'] in excluded for r in candidates),
        'reasons':dict(reasons),'global_blockers':list(global_blockers),'proposed':proposed,
        'admitted_patients':0,'admitted_visits':0,'approval_required':True}
