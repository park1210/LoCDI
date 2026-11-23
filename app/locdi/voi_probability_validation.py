"""C4-29 development probability measurement; no runtime or Core modification."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from statistics import median

from app.core.diagnostic_belief import distribution, entropy, posterior
from app.core.information_outcomes import InformationOutcomes
from app.core.voi_evaluation import fit_outcomes
from app.locdi.dynamic_horizon import DynamicHorizonPolicy
from app.locdi.llm_setup import digest

VERSION = 'locdi.voi_probability_validation.c4_29.v1'
BINS = ['at_or_below_train_median', 'above_train_median']


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def epoch(text):
    return int(datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()*1000)


def numeric(event):
    """Same single numeric code/unit rule as the pinned source build_study."""
    raw=json.loads(event['evidence'])
    if event['action']=='Ask':
        pairs=[(r['code'],r.get('units'),r.get('value')) for r in raw.get('facts',[])]
    elif event['action']=='Test':
        records=raw.get('records',[]) or ([raw] if raw.get('event_type')=='observations' else [])
        pairs=[(r['code'],r['recorded_evidence'].get('units'),r['recorded_evidence'].get('value'))
               for r in records if r['event_type']=='observations']
    else:return None
    if len(pairs)!=1:return None
    code,unit,value=pairs[0]
    try:value=float(value)
    except (TypeError,ValueError):return None
    if not math.isfinite(value):return None
    return compact([event['action'],code,unit]),value


def first_values(events, keys, minute_limit=45):
    groups=defaultdict(list)
    for e in events:
        if e['available_minute']>minute_limit:continue
        result=numeric(e)
        if result and result[0] in keys:groups[result[0]].append((e['available_minute'],result[1],e['event_id']))
    result={}
    for key,items in groups.items():
        early=min(x[0] for x in items);ties=[x for x in items if x[0]==early]
        if len({x[1] for x in ties})==1:
            x=min(ties,key=lambda x:x[2]);result[key]={'value':x[1],'event_id':x[2],'available_minute':x[0]}
    return result


def source_training(source, keys, minute_limit):
    """Reconstruct only the existing first-value contract; IDs stay private."""
    out=[]
    for s in source:
        pid=s['patient_id'];eid=s['index_encounter']['encounter_id'];start=epoch(s['index_encounter']['start_time']);events=[]
        for item in s['symptom_sources']:
            if not (item['kind']=='symptom' and item['evidence_grade']=='A' and item['visibility_allowed']
                    and item['candidate_reason']=='explicit_patient_report_observation' and epoch(item['source_timestamp'])<=start):continue
            raw=item['source_metadata']['_source']['source_record']
            if raw['PATIENT']!=pid or raw['ENCOUNTER']!=eid:raise ValueError('SOURCE_PATIENT_VISIT_MISMATCH')
            events.append({'event_id':item['source_id'],'action':'Ask','available_minute':0,
                'evidence':compact({'facts':[{'code':raw['CODE'],'units':raw['UNITS'],'value':raw['VALUE']}]})})
        for e in s['source_events']:
            if e['event_type']!='observations' or e['action_type_candidate']!='Test':continue
            if e['patient_id']!=pid or e['encounter_id']!=eid:raise ValueError('SOURCE_PATIENT_VISIT_MISMATCH')
            upper=e['timestamp_upper_epoch_ms']
            if upper is None or upper<start:continue
            events.append({'event_id':e['event_id'],'action':'Test','available_minute':max(0,(upper-start+59999)//60000),
                'evidence':compact({'records':[{'event_type':'observations','code':e['target']['code'],'recorded_evidence':e['evidence']}]})})
        values=first_values(events,keys,minute_limit)
        out.append({'patient_id':pid,'visit_id':eid,'case_id':s['case_id'],
            'diagnosis':s['selected_actual_target_diagnosis'],'values':values,'clinical_timeline':'NOT_INFERRED'})
    return out


def encode(value, threshold):
    if type(value) not in (float,int) or not math.isfinite(value):raise ValueError('FINITE_NUMERIC_REQUIRED')
    return BINS[int(value>threshold)]


def encode_rows(visits, mapping, thresholds):
    rows=[]
    for v in visits:
        for key,aid in mapping.items():
            if key in v['values']:
                rows.append({k:v[k] for k in ('patient_id','visit_id','diagnosis')})
                rows[-1].update(split='train',action_id=aid,outcome_id=encode(v['values'][key]['value'],thresholds[key]))
    return rows


def grouped_folds(visits, n=5, seed=42):
    if not 2<=n<=10:raise ValueError('FOLD_RANGE')
    groups=defaultdict(list)
    seen=set()
    for v in visits:
        key=(v['patient_id'],v['visit_id'])
        if key in seen:raise ValueError('DUPLICATE_VISIT')
        seen.add(key);groups[v['patient_id']].append(v)
    if len(groups)<n:raise ValueError('TOO_FEW_PATIENTS')
    support=Counter(d for vv in groups.values() for d in {v['diagnosis'] for v in vv})
    order=sorted(groups,key=lambda p:(min(support[v['diagnosis']] for v in groups[p]),
        hashlib.sha256((str(seed)+':'+p).encode()).hexdigest()))
    counts=[Counter() for _ in range(n)];patients=[[] for _ in range(n)];sizes=[0]*n
    for p in order:
        labels={v['diagnosis'] for v in groups[p]}
        f=min(range(n),key=lambda i:(sum(counts[i][d] for d in labels),sizes[i],len(patients[i]),i))
        patients[f].append(p);sizes[f]+=len(groups[p]);counts[f].update(labels)
    if any(not p for p in patients):raise ValueError('EMPTY_FOLD')
    return {p:i for i,ps in enumerate(patients) for p in ps}


def fit_fold(template, visits, mapping, evaluation_patients):
    """Existing fit_outcomes permits zero-cell smoothing, unlike strict spec gate.

    No clinical/readiness promotion; unsupported cells are explicitly counted.
    """
    if {v['patient_id'] for v in visits}&set(evaluation_patients):raise ValueError('VALIDATION_LEAKAGE')
    if any(v['diagnosis'] not in template['diagnoses'] for v in visits):raise ValueError('OOV_TRAIN_DIAGNOSIS')
    thresholds={k:median(v['values'][k]['value'] for v in visits if k in v['values']) for k in mapping}
    rows=encode_rows(visits,mapping,thresholds)
    model=fit_outcomes(template,rows,alpha=1.0)
    support={a['id']:{d:sum(r['action_id']==a['id'] and r['diagnosis']==d for r in rows) for d in model['diagnoses']} for a in model['actions']}
    model['model_id']='C4_29_DEVELOPMENT_FOLD_ONLY';model['provenance'].update(binning_thresholds=thresholds,
        empirical_support=support,minimum_observations=0,coverage_policy='development_existing_fit_outcomes_zero_cell_smoothing_not_strict_fit_outcome_spec',
        clinical_support='not_inferred',evaluation_exposure='fixed_development_diagnosis_universe')
    InformationOutcomes(model)
    return model,rows


def independent_outcomes(model, rows):
    return {a['id']:{o['id']:(1+sum(r['action_id']==a['id'] and r['outcome_id']==o['id'] for r in rows)) /
        (len(a['outcomes'])+sum(r['action_id']==a['id'] for r in rows)) for o in a['outcomes']} for a in model['actions']}


def probability_score(probabilities, truth, topk=3):
    distribution(probabilities)
    if truth not in probabilities:raise ValueError('OOV_REFERENCE')
    ranked=sorted(probabilities,key=lambda d:(-probabilities[d],d));p=probabilities[truth]
    return {'nll':-math.log(p) if p>0 else None,'zero_probability':p==0,
        'brier':sum((v-int(d==truth))**2 for d,v in probabilities.items()),
        'top1':int(ranked[0]==truth),'top3':int(truth in ranked[:topk]),'reference_probability':p,
        'max_probability':probabilities[ranked[0]],'normalized_entropy':entropy(probabilities)/math.log(len(probabilities)),
        'margin':probabilities[ranked[0]]-probabilities[ranked[1]]}


def summary(scores):
    n=len(scores)
    if not n:return {'n':0,'status':'NOT_APPLICABLE'}
    out={'n':n,'status':'COMPUTED','zero_probability_count':sum(s['zero_probability'] for s in scores)}
    for key in ('brier','top1','top3','reference_probability','max_probability','normalized_entropy','margin'):
        out[key]=sum(s[key] for s in scores)/n
    out['nll']=None if out['zero_probability_count'] else sum(s['nll'] for s in scores)/n
    out['nll_status']='POSITIVE_INFINITY' if out['zero_probability_count'] else 'FINITE'
    return out


def calibration(scores):
    bins=[];n=len(scores)
    for i in range(10):
        seq=[s for s in scores if min(int(s['max_probability']*10),9)==i]
        bins.append({'lower':i/10,'upper':(i+1)/10,'count':len(seq),'confidence':sum(s['max_probability'] for s in seq)/len(seq) if seq else None,
            'source_agreement':sum(s['top1'] for s in seq)/len(seq) if seq else None})
    return {'n':n,'bins':bins,'ece':sum(b['count']*abs(b['confidence']-b['source_agreement']) for b in bins if b['count'])/n if n else None,
        'scope':'Source agreement;10 equal-width bins fixed before evaluation, small-sample descriptive only'}


def apply_outcome(belief, action, outcome_id):
    outcomes=action['outcomes'];match=next((o for o in outcomes if o['id']==outcome_id),None)
    if match is None:raise ValueError('UNKNOWN_OUTCOME')
    pred={o['id']:sum(belief[d]*o['likelihood'][d] for d in belief) for o in outcomes}
    updated,mass=posterior(belief,match['likelihood'])
    if not math.isclose(pred[outcome_id],mass,rel_tol=0,abs_tol=1e-12):raise ValueError('MASS_MISMATCH')
    return updated,pred


def unique_observations(events, decision_minute):
    seen={};result=[]
    for e in sorted(events,key=lambda e:(e['minute'],e['event_index'])):
        if e['minute']>decision_minute:raise ValueError('FUTURE_OBSERVATION')
        a=e['action_id']
        if a in seen:
            if seen[a]!=e['outcome_id']:raise ValueError('CONFLICTING_REPEATED_ACTION')
            continue
        seen[a]=e['outcome_id'];result.append(e)
    return result


def trajectory(model, events):
    belief=dict(model['prior']);actions={a['id']:a for a in model['actions']};steps=[]
    for e in unique_observations(events,max((e['minute'] for e in events),default=0)):
        before=dict(belief);belief,pred=apply_outcome(belief,actions[e['action_id']],e['outcome_id'])
        steps.append({'event':e,'before':before,'outcome_prediction':pred,'after':dict(belief)})
    return belief,steps


def horizon(belief, model, used=()):
    actions=[{'action_id':a['id'],'type':'Ask' if a['type']=='ask_patient' else 'Test','action':a['type'],
        'target':a['target'],'outcome_model':a['id']} for a in model['actions'] if a['id'] not in used]
    return DynamicHorizonPolicy().select(belief=belief,available_actions=actions,observable_context={})


def discriminability(model):
    labels=list(model['diagnoses']);per=[]
    for i,a in enumerate(model['actions'],1):
        tv=[];js=[]
        for j,d in enumerate(labels):
            p=[o['likelihood'][d] for o in a['outcomes']]
            for e in labels[j+1:]:
                q=[o['likelihood'][e] for o in a['outcomes']];m=[(x+y)/2 for x,y in zip(p,q)]
                tv.append(sum(abs(x-y) for x,y in zip(p,q))/2)
                js.append(sum(x*math.log(x/z)/2 if x else 0 for x,z in zip(p,m))+sum(x*math.log(x/z)/2 if x else 0 for x,z in zip(q,m)))
        prior=model['prior'];pred=[sum(prior[d]*o['likelihood'][d] for d in labels) for o in a['outcomes']]
        ig=-sum(p*math.log(p) for p in pred if p)
        ig-=sum(prior[d]*(-sum(o['likelihood'][d]*math.log(o['likelihood'][d]) for o in a['outcomes'] if o['likelihood'][d])) for d in labels)
        per.append({'action_ordinal':i,'pairs':len(tv),'tv_mean':sum(tv)/len(tv),'tv_min':min(tv),'tv_max':max(tv),'js_nats_mean':sum(js)/len(js),'prior_expected_information_gain_nats':ig})
    return {'actions':per,'scope':'Internal model discriminability, not clinical test efficiency'}
