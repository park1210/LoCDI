"""Observation-only past-source projection into unchanged ten-field Memory schema."""
from collections import Counter,defaultdict
from copy import deepcopy
import json
from app.core.longitudinal_memory import PatientMemoryStore,timestamp
from app.locdi.llm_setup import digest


def project_past_observations(identity,encounters,observations,*,accept_recorded_time_proxy=False):
    """No Condition/Reference, no clinical summary, no current visit, no future row.

    Input rows are (original CSV line, raw dict); all excluded rows are accounted
    for, and full raw lineage stays in the separate private sidecar.
    """
    if accept_recorded_time_proxy is not True:raise ValueError('EXPLICIT_PROXY_SCOPE_REQUIRED')
    cutoff=timestamp(identity['started_at']);seen={};reasons=Counter();visits=defaultdict(list);lineage=[]
    for _,raw in encounters:
        if raw['PATIENT']!=identity['patient_id']:raise ValueError('CROSS_PATIENT_ENCOUNTER')
        if raw['Id'] in seen:raise ValueError('DUPLICATE_ENCOUNTER_ID')
        seen[raw['Id']]=raw
    for line,raw in observations:
        if raw['PATIENT']!=identity['patient_id']:raise ValueError('CROSS_PATIENT_MEMORY')
        encounter=seen.get(raw['ENCOUNTER'])
        if encounter is None:reasons['MISSING_ENCOUNTER']+=1;continue
        if raw['ENCOUNTER']==identity['visit_id']:reasons['CURRENT_VISIT']+=1;continue
        try:
            start=timestamp(encounter['START']);observed=timestamp(raw['DATE'])
            end=timestamp(encounter['STOP']) if encounter.get('STOP') else None
        except ValueError:reasons['UNPARSEABLE_SOURCE_TIME']+=1;continue
        if start>=cutoff:reasons['NONPAST_VISIT']+=1;continue
        if end is None:reasons['VISIT_END_UNKNOWN']+=1;continue
        if end<start:reasons['VISIT_TIME_REVERSED']+=1;continue
        if end>cutoff:reasons['PAST_START_BUT_VISIT_NOT_ENDED']+=1;continue
        if observed<start:reasons['RECORD_BEFORE_VISIT']+=1;continue
        if observed>end:reasons['RECORD_AFTER_VISIT_END']+=1;continue
        if observed>cutoff:reasons['FUTURE_RECORD']+=1;continue
        # Whitelist Source facts only, preserving value representation. Do not
        # synthesize medical summaries, units, coded system or update links.
        facts={k:raw.get(k) for k in ['DATE','CATEGORY','CODE','DESCRIPTION','VALUE','UNITS','TYPE']}
        if not raw.get('CODE') or raw.get('VALUE') in (None,''):
            reasons['MISSING_SOURCE_VALUE_OR_CODE']+=1;continue
        mid='C437-memory:'+digest([identity['patient_id'],raw['ENCOUNTER'],line,raw])
        record={'memory_id':mid,'patient_id':identity['patient_id'],'visit_id':raw['ENCOUNTER'],
            'visit_started_at':encounter['START'],'observed_at':raw['DATE'],'available_at':raw['DATE'],
            'source_id':'C434:csv/observations.csv:row:'+str(line)+':'+digest(raw),
            'kind':'objective_finding','text':json.dumps(facts,sort_keys=True,ensure_ascii=False),'supersedes':[]}
        visits[raw['ENCOUNTER']].append(record)
        lineage.append({'memory_id':mid,'source_row':line,'source_record':deepcopy(raw),
            'source_record_fingerprint':digest(raw),'available_at_basis':'RECORDED_TIME_PROXY',
            'actual_clinical_available_at':None,'review_status':'UNREVIEWED','conditions_or_reference_used':False})
    result=[{'visit_id':vid,'started_at':seen[vid]['START'],'ended_at':seen[vid]['STOP'],'records':rows}
            for vid,rows in sorted(visits.items(),key=lambda x:(seen[x[0]]['START'],x[0]))]
    records=[r for v in result for r in v['records']]
    PatientMemoryStore(records)
    return {'past_visits':result,'lineage':lineage,'summary':{
        'eligible_past_visits':len(result),'records':len(records),'excluded_rows':dict(reasons),
        'input_observation_rows':len(observations),'schema':'PatientMemoryStore ten fields PASS',
        'source_scope':'CSV observations only; no Condition or evaluation Reference',
        'availability':'RECORDED_TIME_PROXY_NOT_CLINICALLY_VERIFIED','actual_retrieval_run':False,
        'bayesian_memory_evidence':False,'concept_diverse_deployment_changed':False}}
