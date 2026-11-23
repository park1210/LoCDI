"""C4-30 v2 offline information representation experiment, max25 fits, no resume."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import csv
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,rows,sha,write,check
from app.locdi.voi_source_action_inventory import *
from app.locdi.voi_information_improvement import *
from app.locdi.voi_probability_validation import horizon
from app.locdi.dynamic_horizon import policy_spec,DynamicHorizonPolicy
from app.locdi.source_replay import PRIVATE as D3_PRIVATE
from app.locdi.routing_data_reuse import DATABASE,SOURCE,BANK

NS='c4_30_v2'
CODE=['app/locdi/voi_source_action_inventory.py','app/locdi/voi_information_improvement.py','scripts/evaluate_voi_information_improvement.py','tests/locdi/test_voi_information_improvement.py']
OLD=ROOT/'outputs/private/research_evaluation/c4_29_v1'
PUB29=ROOT/'outputs/research_evaluation/c4_29_v1'


def places():return ROOT/'outputs/research_evaluation'/NS,ROOT/'outputs/private/research_evaluation'/NS


def inventory():
    pub,pr=places();pub.mkdir(parents=True,exist_ok=False);pr.mkdir(parents=True,exist_ok=False)
    inherited=read(OLD/'protected_hashes.json');cm=read(PUB29/'completion_manifest.json');index=read(OLD/'private_index.json')
    if cm['fingerprint']!=digest(cm['files']) or index['fingerprint']!=digest(index['files']):raise ValueError('C4_29_MANIFEST')
    inherited.update(cm['files']);inherited.update(index['files']);inherited[str((PUB29/'completion_manifest.json').relative_to(ROOT))]=sha(PUB29/'completion_manifest.json')
    write(pr/'protected_hashes.json',inherited);pres=check(inherited);write(pr/'initial_preservation.json',pres)
    if pres['changed_or_missing']:raise ValueError('PROTECTED_ASSET_CHANGED')
    model=read(BANK.parent/'voi_model.json');InformationOutcomes(model)
    paths=[DATABASE,SOURCE,BANK,BANK.parent/'voi_model.json',OLD/'patient_grouped_split_manifest.json',D3_PRIVATE/'cases.jsonl']
    sourcepins={str(p):sha(p) for p in paths}
    original=rows(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl');hold={r['patient_id'] for r in original};d3={r['patient_id'] for r in rows(D3_PRIVATE/'cases.jsonl')}
    forbidden=hold|d3;base=[];source=rows(SOURCE);oldvisits=read(OLD/'source_visits.json');bycase={r['case_id']:r for r in oldvisits}
    basepairs={(v['patient_id'],v['visit_id']) for v in oldvisits};oldp={v['patient_id'] for v in oldvisits}
    if oldp&forbidden:raise ValueError('EXISTING_TRAIN_EVALUATION_OVERLAP')
    for s in source:
        v=project(s);v['diagnosis']=s['selected_actual_target_diagnosis'];v['origin']='EXISTING229';old=bycase[v['case_id']]
        if (v['patient_id'],v['visit_id'],v['diagnosis'])!=(old['patient_id'],old['visit_id'],old['diagnosis']):raise ValueError('BASE_IDENTITY')
        for k,e in old['values'].items():
            kind,code,unit=json.loads(k);newkey=action_key(kind,code,unit)
            if newkey not in v['values'] or v['values'][newkey]['value']!=e['value']:raise ValueError('BASE_NUMERIC_RECONSTRUCTION')
        base.append(v)
    extra=[];rejected=Counter();dbp=set();dbpairs=set();diaglinkedp=set();unusedpairs=set()
    for s in database_sources():
        pid=s['patient_id'];pair=(pid,s['index_encounter']['encounter_id']);dbp.add(pid)
        if pair in dbpairs:raise ValueError('DUPLICATE_DB_VISIT')
        dbpairs.add(pair)
        if pair not in basepairs:unusedpairs.add(pair)
        label=unique_reference(s,model['diagnoses'])
        if label is not None:diaglinkedp.add(pid)
        if pid in forbidden:rejected['EVALUATION_PATIENT_EXCLUDED']+=1;continue
        if pair in basepairs:rejected['EXISTING229_VISIT_NOT_DUPLICATED']+=1;continue
        if s['decision_time']!=s['index_encounter']['start_time']:rejected['NONARRIVAL_CONTEXT']+=1;continue
        if label is None:rejected['NO_SINGLE_UNAMBIGUOUS_UNIVERSE_REFERENCE']+=1;continue
        v=project(s)
        if not v['values']:rejected['NO_ADMISSIBLE_SOURCE_OBSERVATIONS']+=1;continue
        v.update(diagnosis=label,origin='ADDITIONAL_RECORDED_SOURCE');extra.append(v)
    checked,csvpaths=csv_check(base+extra)
    for p in csvpaths:sourcepins[str(p)]=sha(p)
    patient_csv=COHORT/'csv/patients.csv'
    with patient_csv.open(encoding='utf-8-sig',newline='') as f:rawpatients={r['Id'] for r in csv.DictReader(f)}
    sourcepins[str(patient_csv)]=sha(patient_csv)
    # Inspect actual native schema and one existing linked native observation UUID.
    ref=source[0]['symptom_sources'][0]['source_metadata']['_source'];native=source_path(ref['raw_json_file'])
    native_doc=read(native);native_text=native.read_text(encoding='utf-8')
    if ref['raw_observation_uuid'] not in native_text:raise ValueError('LINKED_NATIVE_OBSERVATION_NOT_FOUND')
    sourcepins[str(native)]=sha(native)
    if sha(DATABASE)!=sourcepins[str(DATABASE)] or sha(SOURCE)!=sourcepins[str(SOURCE)]:raise ValueError('SOURCE_CHANGED_DURING_READ')
    allkeys=sorted({k for v in base+extra for k in v['values']})
    oldmapping=read(OLD/'action_mapping.json');basekeys=[]
    for a in model['actions']:
        key=next(k for k,aid in oldmapping.items() if aid==a['id']);kind,code,unit=json.loads(key);basekeys.append(action_key(kind,code,unit))
    write(pr/'base_visits.json',base);write(pr/'additional_visits.json',extra);write(pr/'source_hashes.json',sourcepins)
    write(pr/'action_keys.json',{'base':basekeys,'candidates':allkeys,'legacy_mapping':oldmapping})
    write(pr/'excluded_evaluation_patients.json',sorted(forbidden))
    inv={'source_cohort_csv_patients':len(rawpatients),'native_json_files':len(list((COHORT/'json').glob('*.json'))),
        'native_sample_verified':1,'native_all_files_content_verified':False,'native_sample_schema_keys':list(native_doc),
        'database_patients':len(dbp),'database_visits':len(dbpairs),'database_train_patient_overlap':len(dbp&oldp),'database_original50_overlap':len(dbp&hold),'database_d3_overlap':len(dbp&d3),
        'existing_training_patients':len(oldp),'existing_training_visits':len(base),'raw_patients_not_old_training_or_existing_evaluation':len(rawpatients-oldp-forbidden),
        'db_patients_not_old_training_or_existing_evaluation':len(dbp-oldp-forbidden),'db_visits_not_existing229':len(unusedpairs),
        'universe_linked_db_patients':len(diaglinkedp),'additional_eligible_patients':len({v['patient_id'] for v in extra}),
        'additional_new_patients':len({v['patient_id'] for v in extra}-oldp),'additional_visits':len(extra),
        'additional_existing_patient_visits':sum(v['patient_id'] in oldp for v in extra),'additional_evaluation_patient_overlap':len({v['patient_id'] for v in extra}&forbidden),
        'raw_csv_rows_verified':checked,'rejection_reasons':dict(rejected),'clinical_reference_review':'NOT_REVIEWED',
        'diagnosis_confirmation_times':'NOT_AVAILABLE; Condition START not assumed diagnostic confirmation','availability':'RECORDED_TIME_PROXY_DEVELOPMENT_ONLY','new_independent_test':'NOT_AVAILABLE'}
    write(pub/'source_expansion_inventory.json',inv)
    write(pub/'unused_patient_eligibility.json',{k:v for k,v in inv.items() if k not in ('native_sample_schema_keys',)})
    inventory_rows=support(base,allkeys,list(model['diagnoses']))
    for i,r in enumerate(inventory_rows,1):r['action_ordinal']=i;r['is_existing5']=r['key'] in basekeys
    write(pub/'candidate_action_inventory.json',{'candidates':inventory_rows,'system_status':'CSV system unspecified; exact source/code/unit/resource grouping, no invented LOINC binding',
        'categorical_policy':'Inventoried but NOT_ELIGIBLE in this numeric-only bounded comparison','memory_policy':'NOT_SUPPORTED as new information outcome; existing initial Memory not turned into likelihood',
        'excluded_resources':['Condition labels','administrative codes','unrecorded outcomes','post explicit diagnosis-confirmation observations'],
        'clinical_availability':'NOT_VERIFIED; A source existence != B development estimation != C observed complete-panel proxy reveal'})
    write(pub/'partial_observation_feasibility.json',{'additional_visits':len(extra),'visits_missing_some_base5':sum(not set(basekeys)<=set(v['values']) for v in extra),
        'visits_with_any_base5':sum(bool(set(basekeys)&set(v['values'])) for v in extra),'missing_is_negative':False,'core_fit_contract':'Action-specific counts; visits with >=1 selected observation count once in prior',
        'zero_selected_observation_visits':'not fitted; retained in source inventory','new_patients':'augmentation training only in every fold; never called independent validation',
        'existing173_patients':'additional visits follow unchanged C4-29 patient fold, never train on an outer validation patient',
        'sequential_complete_panel_distinct':True})
    print(json.dumps({'phase':'INVENTORY','base':len(base),'extra':len(extra),'new_patients':inv['additional_new_patients'],'candidate_keys':len(allkeys),'fit_calls':0}),flush=True)


def prepare():
    pub,pr=places();keys=read(pr/'action_keys.json');sourcepins=read(pr/'source_hashes.json')
    spec={'protocol':VERSION,'namespace':NS,'prepared_at':datetime.now(timezone.utc).isoformat(),'diagnoses':15,
        'fold_manifest_sha256':sha(OLD/'patient_grouped_split_manifest.json'),'source_hash_manifest_sha256':sha(pr/'source_hashes.json'),
        'code_hashes':{n:sha(ROOT/n) for n in CODE},'base_visits_sha256':sha(pr/'base_visits.json'),'additional_visits_sha256':sha(pr/'additional_visits.json'),
        'candidate_keys_sha256':sha(pr/'action_keys.json'),'maximum_new_model_estimations':25,'models':['M0 prior','M1 reused binary5','M2 ternary5','M3 Gaussian5 PROBE_ONLY','M4 selected binary','M5 selected ternary','M6 M4 frozen bounds expanded counts'],
        'max_per_model':5,'alpha':1,'utility':'unchanged 1/0','max_actions':8,'core_limits':{'max_actions':12,'max_states':20000,'horizon':3},
        'branch_state_upper_bound_at8_ternary_h3':9600,'primary':['diagnosis_nll','diagnosis_brier'],'secondary':['top1','top3','reference_probability','ECE10','entropy','margin','patient_macro'],
        'encoding':{'binary':'Train median linear R7','ternary':'Train quantile1/3,2/3 R7; retain3 bins even if cutpoints equal, empty middle smoothed alpha1'},
        'gaussian':{'scope':'DIAGNOSTIC_PROBE_ONLY','shared_variance':'Train pooled unbiased variance per action','floor':'1e-6*max(1,abs(pooled_mean))^2','mean_shrinkage':5,'minimum_diagnosis_n':3,'sparse_fallback':'pooled mean','posterior':'stable log-density normalization, no continuous artifact injection'},
        'selection':{'retained_base_actions':5,'maximum_extras':3,'extra_min_patient_support':20,'extra_min_diagnosis_coverage':10,'extra_max_missing':.5,'extra_min_distinct_values':3,
            'ranking':'Outer Train binary EIG * observed visit fraction; then patient support, canonical key','redundancy':'reject |Pearson r|>=.95 on >=20 pair-complete train visits versus retained actions',
            'no_validation_labels':True,'statistics_passes':5,'candidate_count_summaries':'Part of the five outer-train selection analyses, no extra estimator/grid-search calls'},
        'M6':'Same M4 per-fold action set AND frozen M4 median thresholds; add eligible recorded visits only, exclude evaluation patients and same-fold validation patients; no zero-observation imputation',
        'evaluation':'All registered229 visits for each model, plus same per-fold complete selected-action panel and action-specific denominators; missing action contributes no likelihood',
        'outcome_comparison':'Only compare each encoding with its own nonconditional baseline. Never compare binary/ternary/density NLL directly',
        'bootstrap':{'patient_cluster_resamples':2000,'seed':42,'CI':'percentile95% of patient-macro paired differences','multiple_comparisons':'descriptive, unadjusted'},
        'decision':'A: discrete lower NLL AND Brier than M0 AND M1 in all229 and common complete panel, plus Top1>=M1; choose lowest NLL then Brier then model ID among qualifying only. Inspect class regressions, no clinical admission. B: only Gaussian meets primary improvement. C: M4 improves M1 but no A/B. D: no eligible extras. Else E.',
        'stop_conditions':['source/code/hash mismatch','patient leakage','>25 estimation reservations','invalid probability/artifact','core contract failure'],
        'horizon_policy_fingerprint':digest(policy_spec()),'c4_31':'Preparation only; seven separate policies, same patient/prior/menu/max budget, existing native STOP, no outcome sampling or full sequential comparison now',
        'clinical_readiness':False,'threshold_or_alpha_tuning':False,'LLM_generation':0,'routing_utility_fit':0,'v1_results_created':False}
    spec['fingerprint']=digest(spec);write(pub/'information_improvement_protocol.json',spec);write(pub/'action_selection_protocol.json',spec['selection'])
    print(json.dumps({'phase':'PREPARED','protocol_fingerprint':spec['fingerprint'],'fit_calls':0}),flush=True)


def baseline_view(original,keys,legacy):
    view=deepcopy(original);mapping={}
    for k in keys:
        kind,system,code,unit,resource=json.loads(k);old=compact([kind,code,unit]);mapping[k]=legacy[old]
    bounds={k:[original['provenance']['binning_thresholds'][compact([json.loads(k)[0],json.loads(k)[2],json.loads(k)[3]])]] for k in keys}
    for a in view['actions']:
        for i,o in enumerate(a['outcomes']):o['id']=str(i)
    view['provenance'].update(action_key_mapping=mapping,binning_thresholds=bounds,encoding='binary',observation_encoding='c4_30_readonly_alias_view')
    InformationOutcomes(view);return view


def outcome_metrics(model,train_rows,visits):
    mapping=model['provenance']['action_key_mapping'];bounds=model['provenance']['binning_thresholds'];keys=list(mapping);result=[]
    for key in keys:
        a=next(a for a in model['actions'] if a['id']==mapping[key]);n=sum(r['action_id']==a['id'] for r in train_rows)
        base={o['id']:(1+sum(r['action_id']==a['id'] and r['outcome_id']==o['id'] for r in train_rows))/(len(a['outcomes'])+n) for o in a['outcomes']}
        pred={o['id']:sum(model['prior'][d]*o['likelihood'][d] for d in model['prior']) for o in a['outcomes']}
        ms=[];bs=[]
        for v in visits:
            if key in v['values']:
                oid=bucket(v['values'][key]['value'],bounds[key]);ms.append(probability_score(pred,oid));bs.append(probability_score(base,oid))
        result.append({'key':key,'model':summary(ms),'nonconditional':summary(bs),'outcome_count':len(a['outcomes'])})
    return result


def train_rows_for_view(model,train):
    rows=[];mapping=model['provenance']['action_key_mapping'];bounds=model['provenance']['binning_thresholds']
    for v in train:
        for key,aid in mapping.items():
            if key in v['values']:rows.append({'patient_id':v['patient_id'],'visit_id':v['visit_id'],'diagnosis':v['diagnosis'],'action_id':aid,'outcome_id':bucket(v['values'][key]['value'],bounds[key]),'split':'train'})
    return rows


def run():
    pub,pr=places();spec=read(pub/'information_improvement_protocol.json')
    if spec['fingerprint']!=digest({k:v for k,v in spec.items() if k!='fingerprint'}):raise ValueError('PROTOCOL_CHANGED')
    for n,h in {**spec['code_hashes'],**read(pr/'source_hashes.json')}.items():
        if sha(Path(n) if Path(n).is_absolute() else ROOT/n)!=h:raise ValueError('INPUT_OR_CODE_CHANGED')
    for name,field in [('base_visits.json','base_visits_sha256'),('additional_visits.json','additional_visits_sha256'),('action_keys.json','candidate_keys_sha256')]:
        if sha(pr/name)!=spec[field]:raise ValueError('SOURCE_SNAPSHOT_CHANGED')
    write(pr/'started.json',{'protocol_fingerprint':spec['fingerprint'],'started_at':datetime.now(timezone.utc).isoformat()})
    folds=read(OLD/'patient_grouped_split_manifest.json');base=read(pr/'base_visits.json');extra=read(pr/'additional_visits.json');keys=read(pr/'action_keys.json');template=read(BANK.parent/'voi_model.json')
    result={m:[] for m in ['M0','M1','M2','M3','M4','M5','M6']};common={m:[] for m in result};fits=[];selection=[];outcomes=[];deps=[];discr=[];horizons=defaultdict(list);diagnostics=[];panels=[]
    reservations=0;budget=EstimationBudget()
    def reserve(name,fold):
        nonlocal reservations
        reservations=budget.reserve(name,fold);write(pr/('Estimation%02d_before.json'%reservations),{'model':name,'fold':fold+1,'estimation_ordinal':reservations,'retry':0})
    for fold in range(5):
        tr=[v for v in base if folds[v['patient_id']]!=fold];va=[v for v in base if folds[v['patient_id']]==fold];vp={v['patient_id'] for v in va}
        old=read(OLD/('fold%02d_probability_artifact.json'%(fold+1)));m1=baseline_view(old,keys['base'],keys['legacy_mapping']);rr1=train_rows_for_view(m1,tr)
        selected,stats,ledger=select_actions(tr,keys['base'],keys['candidates'],m1['prior'])
        selection.append({'fold':fold+1,'selected_keys':selected,'candidate_statistics':stats,'selection_ledger':ledger,'train_patients':len({v['patient_id'] for v in tr}),'validation_patients':len(vp)})
        models={'M1':m1};train_by_model={'M1':rr1};complete={(v['patient_id'],v['visit_id']) for v in va if set(selected)<=set(v['values'])}
        for name,encoding,k in [('M2','ternary',keys['base']),('M4','binary',selected),('M5','ternary',selected)]:
            reserve(name,fold);models[name],train_by_model[name]=estimate(template,tr,k,encoding,vp)
        reserve('M3',fold);models['M3']=gaussian_estimate(tr,keys['base'],m1['prior'],vp)
        augmented=tr+[v for v in extra if v['patient_id'] not in vp]
        if extra:
            reserve('M6',fold);models['M6'],train_by_model['M6']=estimate(template,augmented,selected,'binary',vp,fixed_bounds=models['M4']['provenance']['binning_thresholds'])
        foldresults={}
        for name,model in models.items():
            kind='gaussian' if name=='M3' else 'discrete';r=score_results(model,va,kind);foldresults[name]=r
            result[name].extend(r);common[name].extend(x for x in r if (x['patient_id'],x['visit_id']) in complete)
            path=pr/('fold%02d_%s_model.json'%(fold+1,name));write(path,model);write(pr/('fold%02d_%s_results.json'%(fold+1,name)),r)
            if name!='M1':fits.append({'model':name,'fold':fold+1,'artifact_sha256':sha(path),'scope':'DIAGNOSTIC_PROBE_ONLY' if name=='M3' else 'DEVELOPMENT_DISCRETE_ONLY','training_patients':len(model['training_patients']) if name=='M3' else len(model['provenance']['training_patients']),
                'training_visits':len(tr) if name=='M3' else model['provenance']['training_visits'],'training_rows':None if name=='M3' else model['provenance']['training_rows'],'action_count':len(model['keys']) if name=='M3' else len(model['actions'])})
            if name!='M3':
                outcomes.append({'fold':fold+1,'model':name,'actions':outcome_metrics(model,train_by_model[name],va)})
                disc=discriminability(model);discr.append({'fold':fold+1,'model':name,**disc})
            else:
                density=[]
                for key,s in model['stats'].items():
                    xs=[v['values'][key]['value'] for v in va if key in v['values']]
                    mn=[-logsumexp([math.log(model['prior'][d])+normal_log_density(x,s['class_means'][d],s['shared_variance']) for d in model['prior']]) for x in xs]
                    bn=[-normal_log_density(x,s['pooled_mean'],s['shared_variance']) for x in xs]
                    density.append({'key':key,'n':len(xs),'model_density_nll':sum(mn)/len(mn) if mn else None,'nonconditional_density_nll':sum(bn)/len(bn) if bn else None})
                outcomes.append({'fold':fold+1,'model':'M3','metric_kind':'CONTINUOUS_DENSITY_NLL_NOT_COMPARABLE_WITH_DISCRETE','actions':density})
            for v,pred in zip(va,r):
                chosen_keys=keys['base'] if name=='M3' else list(model['provenance']['action_key_mapping'])
                belief=pred['belief'];maxp=pred['score']['max_probability'];en=pred['score']['normalized_entropy'];margin=pred['score']['margin']
                remaining=sum(k not in pred['used_keys'] for k in chosen_keys)
                horizons[name].append({'fold':fold+1,'entropy':en,'margin':margin,'entropy_requested':1 if en<=1/3 else 2 if en<=2/3 else 3,
                    'margin_requested':1 if margin>=2/3 else 2 if margin>=1/3 else 3,'remaining_actions':remaining,'actual_planner_execution':False})
        # M0 uses exactly the unchanged M1/C4-29 prior.
        z=[{'patient_id':v['patient_id'],'visit_id':v['visit_id'],'diagnosis':v['diagnosis'],'belief':dict(m1['prior']),'score':probability_score(m1['prior'],v['diagnosis']),
            'prior_score':probability_score(m1['prior'],v['diagnosis']),'used_keys':[]} for v in va]
        result['M0'].extend(z);common['M0'].extend(x for x in z if (x['patient_id'],x['visit_id']) in complete)
        # Raw/outcome dependency uses Train only, never selects using validation.
        pairs=[]
        for i,k in enumerate(selected):
            for j,e in enumerate(selected[i+1:],i+1):
                vv=[v for v in tr if k in v['values'] and e in v['values']];xy=[(v['values'][k]['value'],v['values'][e]['value']) for v in vv]
                bnd=models['M4']['provenance']['binning_thresholds'];bo=[(int(bucket(x,bnd[k])),int(bucket(y,bnd[e]))) for x,y in xy]
                pairs.append({'action_indices':[i+1,j+1],'pair_visits':len(xy),'raw_pearson':pearson(xy),'binary_phi':pearson(bo),
                    'by_diagnosis':[{'diagnosis_ordinal':di+1,'n':len([(x['values'][k]['value'],x['values'][e]['value']) for x in vv if x['diagnosis']==d]),
                        'r':pearson([(x['values'][k]['value'],x['values'][e]['value']) for x in vv if x['diagnosis']==d])} for di,d in enumerate(template['diagnoses'])]})
        deps.append({'fold':fold+1,'scope':'OUTER_TRAIN_ONLY; correlations are not clinical redundancy verdicts','pairs':pairs})
        # Single-action and leave-one-out probes reuse fitted probabilities; zero new fitting.
        for name in ['M1','M4']:
            mm=models[name];ks=list(mm['provenance']['action_key_mapping']);probes=[]
            for key in ks:
                for mode in ['single','leave_one_out']:
                    out=[]
                    for v in va:
                        bb,used=discrete_belief(mm,v['values'],exclude=[key] if mode=='leave_one_out' else (),only=[key] if mode=='single' else None)
                        out.append(probability_score(bb,v['diagnosis']))
                    probes.append({'key':key,'mode':mode,'all_registered_visit_scores':summary(out),'actual_action_observed_visits':sum(key in v['values'] for v in va)})
            diagnostics.append({'fold':fold+1,'model':name,'probes':probes})
        panels.append({'fold':fold+1,'registered':len(va),'complete_selected_action_visits':len(complete),'action_count':len(selected),'selected_keys':selected,
            'fixed_order':sorted(selected),'random_seed':42,'source_snapshot_sha256':sha(pr/'base_visits.json')})
        print('FOLD_COMPLETED',fold+1,'ESTIMATIONS',reservations,'ACTIONS',len(selected),'COMPLETE_PANEL',len(complete),flush=True)
    write(pr/'all_model_results.json',result);write(pr/'common_panel_results.json',common);write(pr/'fold_selection.json',selection)
    metrics={name:aggregate(rs) if rs else {'status':'NOT_RUN_NO_ADDITIONAL_SOURCE'} for name,rs in result.items()};cmp={name:aggregate(rs) if rs else {'status':'NOT_APPLICABLE_NO_COMMON_PANEL'} for name,rs in common.items()}
    write(pub/'grouped_model_comparison.json',{'registered_visits':229,'registered_patients':173,'all_registered':metrics,'same_complete_selected_action_panel':cmp,'no_unobserved_outcome_imputation':True})
    write(pub/'diagnosis_probability_metrics.json',{'models':metrics,'source_reference':'UNREVIEWED_DIAGNOSIS; not clinical accuracy','multiple_comparisons':'development-exposed seven conditions, no independent held-out model selection'})
    oldresult=read(OLD/'cv_patient_results.json');oldidx={(v['patient_id'],v['visit_id']):v for v in oldresult};err=0
    for r in result['M1']:
        o=oldidx[r['patient_id'],r['visit_id']]
        err=max(err,max(abs(r['belief'][d]-o['final_belief'][d]) for d in r['belief']))
        if any(abs(r['score'][k]-o['final'][k])>1e-12 for k in ['nll','brier','top1','top3']):raise ValueError('M1_REPRODUCTION_FAILED')
    write(pub/'binary_baseline_reproduction.json',{'status':'PASS','visits':len(result['M1']),'maximum_belief_difference':err,'new_M1_estimations':0,'original_artifacts_changed':False,'view':'Read-only copies alias original outcome IDs to0/1 and map exact source keys; probabilities reused unchanged'})
    comparisons={}
    for name in result:
        if name in ('M0','M1') or not result[name]:continue
        comparisons[name]={baseline:paired(result[baseline],result[name]) for baseline in ('M0','M1')}
    write(pub/'patient_level_paired_comparison.json',{'models':comparisons,'bootstrap_resamples':2000,'seed':42,'scope':'Patient-macro percentile intervals, exploratory unadjusted multi-model comparison'})
    write(pr/'visit_level_paired_differences.json',{name:[{'patient_id':r['patient_id'],'visit_id':r['visit_id'],'deltas':{b:{k:r['score'][k]-next(x for x in result[b] if (x['patient_id'],x['visit_id'])==(r['patient_id'],r['visit_id']))['score'][k] for k in ['nll','brier','top1','top3']} for b in ['M0','M1']}} for r in rs] for name,rs in result.items() if name not in ('M0','M1')})
    write(pub/'calibration_comparison.json',{name:m.get('calibration',{'status':'NOT_RUN'}) for name,m in metrics.items()})
    write(pub/'outcome_encoding_comparison.json',{'fold_action_metrics':outcomes,'cross_encoding_outcome_nll_comparison':'NOT_COMPARABLE; use Source Diagnosis NLL/Brier','degenerate_ternary_policy':spec['encoding']['ternary'],'M3':'Density NLL separately in diagnostic probe only; never compared with discrete NLL'})
    write(pub/'action_dependency_analysis.json',{'folds':deps,'condition_independence_changed':False,'small_diagnosis_samples':'n<3 => r=null; small n remains descriptive only'})
    write(pub/'action_discriminability_analysis.json',{'fitted_model_discriminability':discr,'single_action_and_leave_one_out':diagnostics,'extra_estimator_calls':0,'internal_EIG_is_not_predictive_improvement':True})
    write(pub/'action_support_analysis.json',{'folds':[{'fold':s['fold'],'candidate_support_and_train_statistics':s['candidate_statistics'],'selected':s['selected_keys']} for s in selection],
        'missing_rate_selection_bias':'Actual measured subset counts only; presence may depend on diagnosis, never interpreted as random missingness'})
    write(pub/'fold_model_manifest.json',{'actual_new_model_estimations':reservations,'maximum':25,'by_model':dict(Counter(f['model'] for f in fits)),
        'fold_models':fits,'M1_fit_calls':0,'train_only_action_selection_passes':5,'candidate_descriptive_count_statistics':'Recorded separately, no search or independent artifact fit','LLM_generation':0,'routing_utility_fitting':0})
    hs={}
    for name,rs in horizons.items():
        hs[name]={'states':len(rs),'requested_horizon':dict(Counter(str(max(r['entropy_requested'],r['margin_requested'])) for r in rs)),
            'feasibility_capped_horizon':dict(Counter(str(min(max(r['entropy_requested'],r['margin_requested']),r['remaining_actions'],3)) if r['remaining_actions'] else 'TERMINAL_ONLY' for r in rs)),
            'mean_entropy':sum(r['entropy'] for r in rs)/len(rs),'mean_margin':sum(r['margin'] for r in rs)/len(rs),'actual_planner_executions':0}
    write(pub/'horizon_feature_comparison.json',{'models_final_states':hs,'existing_policy_spec_fingerprint':digest(policy_spec()),'thresholds_changed':False,'M3':'Features only; no continuous Planner Artifact'})
    # Decide using predeclared joint criteria, preserving all model results.
    def improves(name,baseline,common_required=True):
        a=metrics[name].get('visit_macro');b=metrics[baseline].get('visit_macro')
        if not a or not b or not(a['nll']<b['nll'] and a['brier']<b['brier']):return False
        if common_required:
            a=cmp[name].get('visit_macro');b=cmp[baseline].get('visit_macro')
            return bool(a and b and a['nll']<b['nll'] and a['brier']<b['brier'])
        return True
    qualified=[n for n in ['M2','M4','M5','M6'] if result[n] and improves(n,'M0') and improves(n,'M1') and metrics[n]['visit_macro']['top1']>=metrics['M1']['visit_macro']['top1']]
    winner=min(qualified,key=lambda n:(metrics[n]['visit_macro']['nll'],metrics[n]['visit_macro']['brier'],n)) if qualified else None
    extra_signal=improves('M4','M1');gaussian=improves('M3','M0') and improves('M3','M1')
    decision='A_IMPROVED_DISCRETE_PROBABILITY_MODEL' if winner else 'B_CONTINUOUS_SIGNAL_ONLY' if gaussian else 'C_ADDITIONAL_ACTION_SIGNAL_SUPPORTED' if extra_signal else 'D_INSUFFICIENT_SUPPORT' if not any(len(s['selected_keys'])>5 for s in selection) else 'E_NO_PREDICTIVE_IMPROVEMENT_OBSERVED'
    ready={'decision':decision,'selected_development_model':winner,'qualifying_models':qualified,'additional_action_signal':extra_signal,'continuous_primary_signal':gaussian,
        'discrete_artifacts_structurally_valid':True,'clinical_research_gate_approved':False,'official_runtime_applied':False,'independent_clinical_validation':False,
        'c4_31_recommendation':'Separate preapproved development sequential policy protocol on real complete panels' if winner else 'Blocked as improved-model study; existing-source structural preparation retained',
        'all_comparison_results_preserved':True,'no_extra_full_cohort_refit':True,'development_artifacts':'Per-fold only, no extra fitting beyond budget'}
    write(pub/'development_voi_artifact_readiness.json',ready)
    write(pub/'sequential_policy_evaluation_feasibility.json',{'fold_panels':panels,'total_complete_panel_visits':sum(p['complete_selected_action_visits'] for p in panels),
        'registered':229,'complete_panel_scope':'Declared source-presence subset, not whole229 or independent holdout','future_outcome_visibility':'SourceOutcomePanel acquire requires explicit action selection, remaining budget and actual source; raw outcomes private',
        'incomplete_panel':'Missing source => blocked, never sampled from likelihood','full_policy_comparison':'NOT_RUN_C4_31_ONLY','new_core_state_limit':False,'max_actions':8,
        'branch_state_bound_h3_ternary':9600,'core_state_limit':20000,'artifact_winner':winner})
    with (pub/'next_research_decision.md').open('x',encoding='utf-8') as f:f.write('# Decision\n\n'+decision+'\n\nSelected development model: '+str(winner)+'. All results retained. No independent clinical validation, no runtime injection, no tuning. Continuous models are probes only. C4-31 must use a fresh protocol and real complete source panels; no outcome sampling or native STOP override.\n')
    with (pub/'c4_31_policy_evaluation_protocol.md').open('x',encoding='utf-8') as f:f.write('''# C4-31 proposed development protocol (NOT EXECUTED)

Freeze one qualifying discrete model family, per-fold artifacts, actual source complete-panel membership and hashes before policy execution. Keep unchanged outer-patient split. If no qualifying discrete family exists, improved-model evaluation is BLOCKED; this file is a structural plan only.

Policies: Prior-only, Uniform Random(seed42), Fixed Order(canonical source-key lexical order frozen per train-selected fold), Information Gain H1, existing Bayesian VoI H1/H2/H3. Each patient/visit/policy gets an isolated Episode, identical fold prior/menu and maximum acquisition budget k in0..min(5,menu size). Action count is not clinical cost, time or risk. Use existing select_action/plan_voi and validated model actions. SourceOutcomePanel returns the actual recorded outcome only after the selected action; missing source blocks, no model sampling.

A. Budget-capped native policies keep STOP/epsilon/tie/repeat/feasibility unchanged; fewer acquisitions are allowed and reported. B. An optional fixed-k ranking probe is a separately labeled policy, never reported as native STOP execution. Never force native continuation.

Report Source agreement, NLL/Brier by acquired count, actual acquisition count, STOP, terminal agreement and patient-cluster paired differences. Keep registered and complete-panel denominators plus exclusions. No gold, reference label or future value in policy input. Source availability is a recorded-time development proxy, not actual clinical turnaround. No full sequential comparison was executed in C4-30.
''')
    print(json.dumps({'phase':'COMPLETE','actual_model_estimations':reservations,'decision':decision,'winner':winner,'complete_panel_visits':sum(p['complete_selected_action_visits'] for p in panels),'LLM_generation':0}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['inventory','prepare','run']);a=p.parse_args()
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    {'inventory':inventory,'prepare':prepare,'run':run}[a.phase]()


if __name__=='__main__':main()
