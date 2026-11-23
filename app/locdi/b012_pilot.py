"""Bounded B0/B1/B2 collection/metrics. No retrieval or clinical policy."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path

from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_validate_result,features_for
from app.locdi.llm_setup import digest
from app.locdi.memory_paired_recovery import SharedRecoveryStop,classify_failure
from app.locdi.synthea_diagnosis_audit import matches

CONDITIONS=('B0','B1','B2')


def schedule(panel):
    if len(panel)!=10 or len({x['patient_ref'] for x in panel})!=10 or len({x['case_ref'] for x in panel})!=10:
        raise ValueError('EXACT_FIXED_TEN_UNIQUE_PATIENTS_REQUIRED')
    return [{'episode_ordinal':3*i+j+1,'patient_ordinal':i+1,'case_ref':p['case_ref'],'condition':c}
        for i,p in enumerate(panel) for j,c in enumerate(CONDITIONS)]


def collect(panel,backend,*,trace_path,preflight,guard,total_cap=360,per_episode_cap=12,audit_failure=None):
    tasks=schedule(panel);budget=GenerationBudget(total_cap,per_episode_cap,resource_guard=guard)
    if total_cap>360 or per_episode_cap>12 or total_cap<1 or per_episode_cap<1:
        raise ValueError('B012_FIXED_CAPS_REQUIRED')
    checks={(r['patient_ordinal'],r['condition']):r for r in preflight}
    if set(checks)!={(t['patient_ordinal'],t['condition']) for t in tasks} or len(preflight)!=30:
        raise ValueError('ALL30_PREFLIGHT_ROWS_REQUIRED')
    rows=[];stopped=None;attempted=0;path=Path(trace_path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        for t in tasks:
            row={**t,'status':'NOT_ATTEMPTED','request_count':0,'result':None,'error_type':None,'failure_audit':None,
                 'backend_episode_ordinal':None}
            ready=checks[t['patient_ordinal'],t['condition']]
            if ready['status']=='CONTEXT_BLOCKED':
                row.update(status='PREFLIGHT_CONTEXT_BLOCKED',error_type='UNTRUNCATED_CONTEXT_LIMIT')
            elif ready['status']!='READY':
                raise ValueError('UNSUPPORTED_PREFLIGHT_STATE')
            elif stopped is None:
                before=budget.used;budget.episode_used=0;token=_BUDGET.set(budget)
                try:
                    guard()
                    attempted+=1;row['backend_episode_ordinal']=attempted
                    result=backend({'case_ref':t['case_ref']},features_for(t['condition']))
                    _validate_result(result,budget.used-before)
                    row['result']=deepcopy(result)
                    row['status']={'completed':'COMPLETED','incomplete':'INCOMPLETE','failed':'CONTRACT_FAILED'}[result['status']]
                    guard()
                    if row['status']=='CONTRACT_FAILED':stopped='FIRST_TECHNICAL_FAILURE'
                except Exception as exc:
                    row['status']=classify_failure([type(exc).__name__]);row['error_type']=type(exc).__name__
                    stopped='FIRST_TECHNICAL_FAILURE:'+type(exc).__name__
                    if audit_failure is not None:row['failure_audit']=audit_failure(row,type(exc).__name__)
                finally:
                    _BUDGET.reset(token);row['request_count']=budget.used-before
            row['global_stop_reason']=stopped;row['row_fingerprint']=digest(row)
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());rows.append(row)
    return {'rows':rows,'generation_count':budget.used,'global_stop_reason':stopped,'registered_episodes':30}


def aggregate(panel,rows):
    tasks=schedule(panel)
    if len(rows)!=30:raise ValueError('FIXED_REGISTERED_DENOMINATOR30_REQUIRED')
    ref={p['case_ref']:p['reference'] for p in panel};out=[]
    statuses={'COMPLETED','INCOMPLETE','CONTRACT_FAILED','TIMEOUT','CONTEXT_BLOCKED','NOT_ATTEMPTED','PREFLIGHT_CONTEXT_BLOCKED'}
    for task,r in zip(tasks,rows):
        if any(r.get(k)!=v for k,v in task.items()) or r['status'] not in statuses:
            raise ValueError('SCHEDULE_OR_STATUS_MISMATCH')
        if r.get('row_fingerprint')!=digest({k:v for k,v in r.items() if k!='row_fingerprint'}):
            raise ValueError('ROW_FINGERPRINT_MISMATCH')
        prediction=(r.get('result') or {}).get('prediction') or {};done=r['status']=='COMPLETED'
        supported=done and prediction.get('certainty')=='supported'
        m=matches(prediction.get('diagnosis') if supported else None,ref[task['case_ref']]['accepted_diagnoses'])
        out.append({**task,'status':r['status'],'certainty':prediction.get('certainty') if done else None,
            'source_exact':m['source_exact'] if done else None,'normalized':m['normalized_match'] if done else None,
            'source_department':prediction['department'] in ref[task['case_ref']]['accepted_departments']
                if done and isinstance(prediction.get('department'),str) else None,'request_count':r['request_count']})
    summary={}
    for c in CONDITIONS:
        rs=[r for r in out if r['condition']==c];supported=[r for r in rs if type(r['source_exact']) is bool]
        exact=sum(r['source_exact'] is True for r in rs);dept=sum(r['source_department'] is True for r in rs)
        summary[c]={'registered':10,'states':dict(Counter(r['status'] for r in rs)),
            'Source_exact_registered_success_count':exact,'Source_department_registered_success_count':dept,
            'supported_scorable':len(supported),'unresolved':sum(r['certainty']=='unresolved' for r in rs),
            'generation':sum(r['request_count'] for r in rs),'threshold_2_of_10_reached':exact>=2 or dept>=2,
            'completed_all10':sum(r['status']=='COMPLETED' for r in rs)==10}
    pairs={}
    for first,second in (('B0','B1'),('B1','B2'),('B0','B2')):
        ps=[]
        for i in range(10):
            a=out[3*i+CONDITIONS.index(first)];b=out[3*i+CONDITIONS.index(second)]
            complete=a['status']==b['status']=='COMPLETED'
            scorable=type(a['source_exact']) is bool and type(b['source_exact']) is bool
            ps.append({'patient_ordinal':i+1,'first_status':a['status'],'second_status':b['status'],
                'complete_pair':complete,'Source_scorable_pair':scorable,
                'Source_difference':int(b['source_exact'])-int(a['source_exact']) if scorable else None})
        known=[r['Source_difference'] for r in ps if r['Source_scorable_pair']]
        pairs[second+'-'+first]={'complete_pairs':sum(r['complete_pair'] for r in ps),'scorable_pairs':len(known),
            'difference':sum(known)/len(known) if known else None,'rows':ps}
    return {'conditions':summary,'paired':pairs,'episodes':out,'registered_episodes':30,'registered_patients':10,
        'freeze_threshold':{'minimum_successes':2,'registered_denominator':10,'metric':'Source diagnosis OR Source department',
            'any_condition_reached':any(r['threshold_2_of_10_reached'] for r in summary.values()),
            'clinical_approval':False,'does_not_unlock_B3_B4':True},'clinical_accuracy':'NOT_EVALUATED'}
