"""Exact existing C4-35 CSV -> Native wrapper; no invented symptoms/labels."""
from copy import deepcopy
from datetime import datetime,timezone
import math,csv
from pathlib import Path
from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest
from app.locdi.source_replay import project_wrapper
from app.locdi.m5_common_source import bind_case,verify_csv
from app.locdi.case_adapter import adapt_case
from app.locdi.original_source_inventory import sha_file
from app.locdi.frozen_m5_runtime_adapter import bind_value

def wrapper(candidate,model):
    v=candidate['visit'];pid=v['patient_id'];vid=v['visit_id']
    if pid!=candidate['patient_id'] or vid!=candidate['visit_id']:raise ValueError('C435_SOURCE_IDENTITY')
    c={'case_id':'C435Native:'+v['case_id'],'patient_id':pid,'visit_id':vid,'decision_time':v['decision_time']}
    directory=Path(next(iter(v['values'].values()))['source_file']).parent
    with (directory/'encounters.csv').open(encoding='utf-8-sig',newline='') as stream:
        enc=next((row for row in csv.DictReader(stream) if row['Id']==vid),None)
    with (directory/'patients.csv').open(encoding='utf-8-sig',newline='') as stream:
        patient=next((row for row in csv.DictReader(stream) if row['Id']==pid),None)
    if enc is None or patient is None or enc['PATIENT']!=pid or timestamp(enc['START'])!=timestamp(v['decision_time']):raise ValueError('SOURCE_PATIENT_ENCOUNTER_LINK')
    born=datetime.fromisoformat(patient['BIRTHDATE']).date();day=timestamp(v['decision_time']).date()
    age=day.year-born.year-((day.month,day.day)<(born.month,born.day))
    if age<0:raise ValueError('SOURCE_AGE_CHRONOLOGY')
    case=project_wrapper(c,{'encounter_class':enc['ENCOUNTERCLASS'],'demographics':{'age_at_index':age,'sex':patient['GENDER']}},[],[],{})
    events=[];files={str(directory/f):sha_file(directory/f) for f in ('encounters.csv','patients.csv')}
    for key,value in v['values'].items():
        bind_value(model,v,key)
        raw=value['source_record'];ms=int(timestamp(raw['DATE']).timestamp()*1000)
        meta={'source_file':value['source_file'],'source_row':value['source_row'],'source_record':deepcopy(raw)}
        if value['source_file'] not in files:files[value['source_file']]=sha_file(value['source_file'])
        events.append({'event_id':value['event_id'],'case_id':c['case_id'],'patient_id':pid,'encounter_id':vid,
            'event_type':'observations','timestamp':raw['DATE'],'timing_precision':'second',
            'timestamp_lower_epoch_ms':ms,'timestamp_upper_epoch_ms':ms+999,
            'target':{'code':raw['CODE'],'description':raw['DESCRIPTION'],'system':None},
            'evidence':{'category':raw['CATEGORY'],'kind':'recorded_observation','value':raw['VALUE'],'units':raw['UNITS']},
            'source_metadata':meta,'availability_status':'UNKNOWN_EXPLICIT_DEVELOPMENT_PROXY','relative_to_decision':'same_precision_interval_uncertain',
            'reveal_status':'ACTION_GATED','action_type_candidate':'Ask' if raw['CODE']=='72514-3' else 'Test'})
    case['environment']['source_event_store']=events
    case['provenance']['source_package']='C435_NATIVE_VALIDATION_RECORDED_TIME_PROXY.v1'
    root=Path(__file__).resolve().parents[2]
    case['provenance']['source_sha256']={Path(f).resolve().relative_to(root).as_posix():h for f,h in files.items()}
    b=bind_case(case,model)
    projected=[]
    for row in b['rows']:
        e=row['source_event'];raw=e['source_metadata']['source_record']
        if row['kind']=='Ask':
            fact={'active_status':None,'code':raw['CODE'],'description':raw['DESCRIPTION'],'duration':None,'onset':None,
                  'recorded_at':raw['DATE'],'units':raw['UNITS'],'value':raw['VALUE']}
            payload={'kind':'recorded_patient_report','replay_bin':'UNBINNED_SOURCE_VALUE_NO_VOI','facts':[fact]}
            case['provenance']['source_case_audit']['symptom_sources'].append({'source_id':e['event_id'],'source_timestamp':raw['DATE'],
                'kind':'recorded_patient_report','evidence_grade':'SOURCE_LINKED_DEVELOPMENT_ONLY','source_metadata':{'_source':e['source_metadata']},
                'visible_value':{'symptom_name':raw['DESCRIPTION'],'value':raw['VALUE']},'visibility_allowed':True})
            target=row['target']
        else:
            payload={'replay_bin':'UNBINNED_SOURCE_VALUE_NO_VOI','records':[{'code':raw['CODE'],'description':raw['DESCRIPTION'],
                'event_type':'observations','recorded_at':raw['DATE'],'recorded_evidence':deepcopy(e['evidence']),'system':None,'timing_precision':'second'}]}
            target='{} [{}]'.format(raw['DESCRIPTION'],raw['CODE'])
        projected.append({'event_id':e['event_id'],'action':row['kind'],'target':target,'available_minute':row['available_minute'],'evidence':__import__('json').dumps(payload,sort_keys=True)})
    case['environment']['current_encounter_events']=projected
    # Source labels remain outside the wrapper/model input entirely.
    if case['evaluation']['diagnosis_labels'] or len(b['rows'])!=len(v['values']):raise ValueError('C435_BINDING_COVERAGE')
    adapt_case(case)
    return case,b

def reference(candidate,conditions_path):
    v=candidate['visit'];rows=candidate['conditions']
    if not rows or v['diagnosis'] is None:raise ValueError('CURRENT_SOURCE_REFERENCE_REQUIRED')
    for r in rows:
        raw=r['source_record']
        if raw['PATIENT']!=v['patient_id'] or raw['ENCOUNTER']!=v['visit_id'] or raw['START'][:10]>v['decision_time'][:10]:raise ValueError('REFERENCE_IDENTITY_DATE')
    verify_csv([{'source_metadata':{'source_file':str(conditions_path),'source_row':r['source_row'],'source_record':r['source_record']}} for r in rows])
    return {'label':v['diagnosis'],'source_condition_rows':deepcopy(rows),'source_file':str(conditions_path),
            'timing_status':'CURRENT_ENCOUNTER_SOURCE_DATE_PROXY_INTRADAY_UNREVIEWED',
            'clinical_reference_approved':False,'past_VoI_development_exposure':True}
