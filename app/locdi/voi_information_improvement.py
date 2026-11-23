"""C4-30 fixed development representations; existing discrete VoI mathematics."""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import random

from app.core.diagnostic_belief import posterior, entropy
from app.core.information_outcomes import InformationOutcomes
from app.core.voi_evaluation import fit_outcomes
from app.locdi.voi_probability_validation import probability_score, summary, calibration, discriminability, compact
from app.locdi.voi_source_action_inventory import support
from app.locdi.llm_setup import digest

VERSION='locdi.voi_information_improvement.c4_30.v2'
MAX_ACTIONS=8


class EstimationBudget:
    def __init__(self):self.count=0;self.by_model=Counter();self.seen=set()
    def reserve(self,name,fold):
        if name not in {'M2','M3','M4','M5','M6'} or not 0<=fold<5:raise ValueError('ESTIMATION_BINDING')
        if (name,fold) in self.seen:raise ValueError('DUPLICATE_ESTIMATION')
        if self.count>=25 or self.by_model[name]>=5:raise ValueError('MODEL_ESTIMATION_CAP25_OR_PER_MODEL5')
        self.count+=1;self.by_model[name]+=1;self.seen.add((name,fold))
        return self.count


def quantile(values,p):
    if not values:raise ValueError('NO_TRAIN_VALUES')
    values=sorted(values);x=(len(values)-1)*p;i=int(x);f=x-i
    return values[i]*(1-f)+values[min(i+1,len(values)-1)]*f


def thresholds(visits,keys,encoding):
    if encoding not in ('binary','ternary'):raise ValueError('ENCODING')
    ps=[.5] if encoding=='binary' else [1/3,2/3]
    return {k:[quantile([v['values'][k]['value'] for v in visits if k in v['values'] and v['values'][k]['data_type']=='numeric'],p) for p in ps] for k in keys}


def bucket(value,bounds):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('NUMERIC_OUTCOME_REQUIRED')
    return str(sum(value>x for x in bounds))


def pearson(pairs):
    if len(pairs)<3:return None
    a=sum(x for x,y in pairs)/len(pairs);b=sum(y for x,y in pairs)/len(pairs)
    den=math.sqrt(sum((x-a)**2 for x,y in pairs)*sum((y-b)**2 for x,y in pairs))
    return sum((x-a)*(y-b) for x,y in pairs)/den if den else None


def information_stat(visits,key,prior):
    """Train-only descriptive count statistic, not another artifact fit."""
    seq=[v for v in visits if key in v['values'] and v['values'][key]['data_type']=='numeric']
    if not seq:return None
    cut=quantile([v['values'][key]['value'] for v in seq],.5)
    cnt=Counter((v['diagnosis'],int(v['values'][key]['value']>cut)) for v in seq)
    probs={d:[(cnt[d,o]+1)/(cnt[d,0]+cnt[d,1]+2) for o in (0,1)] for d in prior}
    mixture=[sum(prior[d]*probs[d][o] for d in prior) for o in (0,1)]
    h=lambda ps:-sum(x*math.log(x) for x in ps if x)
    eig=h(mixture)-sum(prior[d]*h(probs[d]) for d in prior)
    tv=[abs(probs[d][0]-probs[e][0]) for i,d in enumerate(prior) for e in list(prior)[i+1:]]
    js=[]
    for i,d in enumerate(prior):
        for e in list(prior)[i+1:]:
            p,q=probs[d],probs[e];m=[(x+y)/2 for x,y in zip(p,q)]
            js.append(sum(x*math.log(x/z)/2 for x,z in zip(p,m))+sum(x*math.log(x/z)/2 for x,z in zip(q,m)))
    return {'train_eig_nats':eig,'train_tv_mean':sum(tv)/len(tv),'train_js_mean':sum(js)/len(js),'threshold':cut,
        'outcome_counts':[sum(cnt[d,o] for d in prior) for o in (0,1)],'zero_outcome_cells':sum(cnt[d,o]==0 for d in prior for o in (0,1)),
        'interpretation':'Train-only count diagnostic, no validation labels and no additional model fit'}


def select_actions(train,base_keys,candidate_keys,prior):
    """Fixed Train-only support/EIG/coverage ranking; max3 extra, no tuning."""
    if len(base_keys)!=5:raise ValueError('BASE5_REQUIRED')
    rows=support(train,candidate_keys,list(prior));selected=list(base_keys);ledger=[]
    for r in rows:
        stat=information_stat(train,r['key'],prior) if r['numeric_rows']==r['visits'] and r['visits'] else None
        r.update(stat or {});r['eligible_extra']=(r['key'] not in base_keys and r['patients']>=20 and r['diagnosis_coverage']>=10
            and r['missing_rate']<=.5 and r['unique_numeric_values']>=3 and r['numeric_rows']==r['visits'])
        r['ranking_score']=(r.get('train_eig_nats',0)*(1-r['missing_rate'])) if r['missing_rate'] is not None else 0
    for r in sorted([r for r in rows if r['eligible_extra']],key=lambda r:(-r['ranking_score'],-r['patients'],r['key'])):
        redundant=False;comparisons=[]
        for key in selected:
            pairs=[(v['values'][r['key']]['value'],v['values'][key]['value']) for v in train if r['key'] in v['values'] and key in v['values']]
            corr=pearson(pairs) if len(pairs)>=20 else None
            comparisons.append({'other_key':key,'n':len(pairs),'r':corr})
            redundant|=corr is not None and abs(corr)>=.95
        chosen=not redundant and len(selected)<MAX_ACTIONS
        ledger.append({'key':r['key'],'selected':chosen,'reason':'CORRELATION_GE_095' if redundant else 'SELECTED' if chosen else 'ACTION_CAP8','comparisons':comparisons})
        if chosen:selected.append(r['key'])
    return selected,rows,ledger


def estimate(template,train,keys,encoding,evaluation_patients,fixed_bounds=None):
    if {v['patient_id'] for v in train}&set(evaluation_patients):raise ValueError('VALIDATION_LEAKAGE')
    if not 1<=len(keys)<=MAX_ACTIONS:raise ValueError('ACTION_CAP8')
    bounds=deepcopy(fixed_bounds) if fixed_bounds is not None else thresholds(train,keys,encoding)
    if set(bounds)!=set(keys):raise ValueError('BOUNDS_KEYS')
    nout=2 if encoding=='binary' else 3
    model=deepcopy(template);model['model_id']='C4_30_DEVELOPMENT_ONLY';model['revision']=VERSION+':'+encoding
    model['actions']=[];ids={k:'source:'+digest(k)[:24] for k in keys}
    for k in keys:
        kind=__import__('json').loads(k)[0];target=('Source observation ['+digest(k)[:16]+']')
        if kind=='Ask':target='What is the recorded source response ['+digest(k)[:16]+']?'
        model['actions'].append({'id':ids[k],'type':'ask_patient' if kind=='Ask' else 'request_test','target':target,
            'outcomes':[{'id':str(i),'likelihood':{d:1/nout for d in model['diagnoses']},
                'observation':{'source':'history' if kind=='Ask' else 'objective_findings','text':target+': outcome'+str(i)},'supports_diagnoses':[]} for i in range(nout)]})
    rows=[]
    for v in train:
        for k in keys:
            if k not in v['values']:continue
            if v['values'][k]['data_type']!='numeric':raise ValueError('NONNUMERIC_NOT_SELECTED')
            rows.append({x:v[x] for x in ('patient_id','visit_id','diagnosis')})
            rows[-1].update(split='train',action_id=ids[k],outcome_id=bucket(v['values'][k]['value'],bounds[k]))
    fitted=fit_outcomes(model,rows,alpha=1)
    support_counts={a['id']:{d:sum(r['action_id']==a['id'] and r['diagnosis']==d for r in rows) for d in model['diagnoses']} for a in fitted['actions']}
    fitted['provenance'].update(binning_thresholds=bounds,action_key_mapping=ids,observation_encoding='c4_30_source_discrete_v2',
        empirical_support=support_counts,minimum_observations=0,coverage_policy='development_zero_cells_reported_not_clinical_admission',
        encoding=encoding,training_scope='OUTER_TRAIN_ONLY',actual_value_source='EXISTING_RECORDED_SOURCE',clinical_support='not_inferred',
        outcome_semantics='Train-only bounds; equal ternary cutpoints keep empty middle bin with alpha1, no bin merging',
        assumptions=list(model['provenance']['assumptions'])+['C4-30 development only, recorded-time proxy, no clinical availability attestation'])
    InformationOutcomes(fitted)
    return fitted,rows


def discrete_belief(model,values,exclude=(),only=None):
    belief=dict(model['prior']);mapping=model['provenance']['action_key_mapping'];bounds=model['provenance']['binning_thresholds'];used=[]
    for a in model['actions']:
        key=next(k for k,v in mapping.items() if v==a['id'])
        if key not in values or key in exclude or (only is not None and key not in only):continue
        oid=bucket(values[key]['value'],bounds[key]);outcome=next(o for o in a['outcomes'] if o['id']==oid)
        belief=posterior(belief,outcome['likelihood'])[0];used.append(key)
    return belief,used


def gaussian_estimate(train,keys,prior,evaluation_patients):
    if {v['patient_id'] for v in train}&set(evaluation_patients):raise ValueError('VALIDATION_LEAKAGE')
    stats={}
    for k in keys:
        allvalues=[v['values'][k]['value'] for v in train if k in v['values']];mu=sum(allvalues)/len(allvalues)
        variance=sum((x-mu)**2 for x in allvalues)/max(1,len(allvalues)-1);floor=1e-6*max(1,abs(mu))**2;variance=max(variance,floor)
        means={};counts={}
        for d in prior:
            xs=[v['values'][k]['value'] for v in train if v['diagnosis']==d and k in v['values']];n=len(xs);counts[d]=n
            means[d]=(sum(xs)+5*mu)/(n+5) if n>=3 else mu
        stats[k]={'pooled_mean':mu,'shared_variance':variance,'class_means':means,'class_counts':counts,'variance_floor':floor}
    return {'model_type':'DIAGNOSTIC_PROBE_ONLY_GAUSSIAN','prior':dict(prior),'keys':list(keys),'stats':stats,'shrinkage':5,'minimum_class_support':3,
        'training_patients':sorted({v['patient_id'] for v in train}),'core_discrete_compatible':False}


def logsumexp(values):
    m=max(values);return m+math.log(sum(math.exp(x-m) for x in values))


def normal_log_density(x,mu,var):return -.5*(math.log(2*math.pi*var)+(x-mu)**2/var)


def gaussian_log_posterior(model,values):
    logs={d:math.log(p) for d,p in model['prior'].items()}
    for k in model['keys']:
        if k not in values:continue
        s=model['stats'][k];x=values[k]['value']
        for d in logs:logs[d]+=normal_log_density(x,s['class_means'][d],s['shared_variance'])
    z=logsumexp(list(logs.values()));return {d:x-z for d,x in logs.items()}


def gaussian_belief(model,values):return {d:math.exp(x) for d,x in gaussian_log_posterior(model,values).items()}


def score_results(model,visits,kind='discrete'):
    results=[]
    for v in visits:
        if kind=='gaussian':
            if v['patient_id'] in model['training_patients']:raise ValueError('VALIDATION_LEAKAGE')
            logb=gaussian_log_posterior(model,v['values']);b={d:math.exp(x) for d,x in logb.items()};used=[k for k in model['keys'] if k in v['values']]
        else:
            if v['patient_id'] in model['provenance']['training_patients']:raise ValueError('VALIDATION_LEAKAGE')
            b,used=discrete_belief(model,v['values'])
        s=probability_score(b,v['diagnosis'])
        if kind=='gaussian':
            s.update(nll=-logb[v['diagnosis']],zero_probability=False,probability_underflow=b[v['diagnosis']]==0,nll_computation='finite log posterior; no probability clipping')
        results.append({'patient_id':v['patient_id'],'visit_id':v['visit_id'],'diagnosis':v['diagnosis'],'belief':b,'used_keys':used,
            'score':s,'prior_score':probability_score(model['prior'],v['diagnosis'])})
    return results


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['patient_id']].append(r['score'])
    ss=[summary(v) for v in groups.values()]
    metrics=['nll','brier','top1','top3','reference_probability','max_probability','normalized_entropy','margin']
    return {'visits':len(rows),'patients':len(groups),'visit_macro':summary([r['score'] for r in rows]),
        'patient_macro':{k:sum(s[k] for s in ss)/len(ss) if ss and all(s[k] is not None for s in ss) else None for k in metrics},
        'calibration':calibration([r['score'] for r in rows]),
        'reference_probability_increased':sum(r['score']['reference_probability']>r['prior_score']['reference_probability'] for r in rows),
        'reference_probability_decreased':sum(r['score']['reference_probability']<r['prior_score']['reference_probability'] for r in rows),
        'observed_actions':{'total':sum(len(r['used_keys']) for r in rows),'minimum':min((len(r['used_keys']) for r in rows),default=None),'maximum':max((len(r['used_keys']) for r in rows),default=None)}}


def paired(left,right,repetitions=2000,seed=42):
    a={(r['patient_id'],r['visit_id']):r for r in left};b={(r['patient_id'],r['visit_id']):r for r in right}
    if set(a)!=set(b):raise ValueError('PAIRED_DENOMINATOR_MISMATCH')
    groups=defaultdict(list)
    for key in sorted(a):groups[key[0]].append({k:b[key]['score'][k]-a[key]['score'][k] for k in ('nll','brier','top1','top3')})
    ps=sorted(groups);means=[{k:sum(r[k] for r in groups[p])/len(groups[p]) for k in ('nll','brier','top1','top3')} for p in ps]
    out={'patients':len(ps),'visits':len(a),'difference':'right minus left','patient_macro':{},'visit_macro':{k:sum(r[k] for rs in groups.values() for r in rs)/len(a) if a else None for k in ('nll','brier','top1','top3')},
        'bootstrap':'2000 patient-cluster resamples seed42; percentile95%; descriptive development, no multiplicity correction'}
    rng=random.Random(seed);rep={k:[] for k in ('nll','brier','top1','top3')}
    if not ps:return out
    for _ in range(repetitions):
        draw=[means[rng.randrange(len(ps))] for _ in ps]
        for k in rep:rep[k].append(sum(r[k] for r in draw)/len(draw))
    for k in rep:out['patient_macro'][k]={'difference':sum(r[k] for r in means)/len(means),'ci95':[quantile(rep[k],.025),quantile(rep[k],.975)]}
    return out


class SourceOutcomePanel:
    """C4-31 preparation only: require actual source and reveal after selection."""
    def __init__(self,model,visit):
        self.model=InformationOutcomes(model).artifact;self.values=deepcopy(visit['values']);self.used=set()
        self.mapping=model['provenance']['action_key_mapping'];self.bounds=model['provenance']['binning_thresholds']
    def menu(self):return [a['id'] for a in self.model['actions'] if a['id'] not in self.used]
    def acquire(self,action_id,*,selected,budget_remaining):
        if selected!=action_id:raise ValueError('OUTCOME_BEFORE_SELECTION')
        if budget_remaining<=0:raise ValueError('BUDGET_EXHAUSTED')
        if action_id in self.used:raise ValueError('REPEATED_ACTION')
        key=next((k for k,a in self.mapping.items() if a==action_id),None)
        if key is None:raise ValueError('UNKNOWN_ACTION')
        if key not in self.values:raise ValueError('MISSING_SOURCE_OUTCOME')
        self.used.add(action_id)
        return {'action_id':action_id,'outcome_id':bucket(self.values[key]['value'],self.bounds[key]),
            'available_minute':self.values[key]['available_minute'],'availability_basis':'RECORDED_TIME_PROXY','source_event_id':self.values[key]['event_id']}
