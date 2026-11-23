"""C4-29 offline train-only probability evaluation. Fresh namespace, no resume."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read, rows, sha, write, check
from app.locdi.routing_data_reuse import BANK, SOURCE
from app.locdi.voi_probability_validation import *
from app.locdi.voi_empirical_evaluation import released_observation
from app.locdi.dynamic_horizon import policy_spec

CODE=['app/locdi/voi_probability_validation.py','scripts/evaluate_voi_probability_model.py','tests/locdi/test_voi_probability_validation.py']


def locations(namespace):
    if not namespace.startswith('c4_29_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in namespace):raise ValueError('NAMESPACE')
    return ROOT/'outputs/research_evaluation'/namespace,ROOT/'outputs/private/research_evaluation'/namespace


def sources():
    folder=BANK.parent;study=SOURCE.parent.parent
    return {'bank':BANK,'training_source':SOURCE,'training_rows':study/'voi_training_rows.json','model':folder/'voi_model.json',
        'cases':folder/'cases.json','labels':folder/'evaluation_labels.json','original50':ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl',
        'config':study/'study_config.json','generator':study.parent.parent/'build_study.py',
        'replay':ROOT/'outputs/private/research_evaluation/c4_18_v1/decision_trace.jsonl',
        'registered_reports':ROOT/'outputs/private/research_evaluation/c4_18_v1/registered_historical_reports.json',
        'horizon_policy':ROOT/'app/locdi/dynamic_horizon.py','core_outcomes':ROOT/'app/core/information_outcomes.py',
        'core_belief':ROOT/'app/core/diagnostic_belief.py','core_fit':ROOT/'app/core/voi_evaluation.py',
        'core_planner':ROOT/'app/core/voi_planning.py','core_selector':ROOT/'app/core/information_value.py',
        'artifact_gate':ROOT/'validation/artifact_readiness.py','research_gate':ROOT/'validation/research_artifact_gate.py'}


def prepare(namespace):
    pub,pr=locations(namespace);pub.mkdir(parents=True,exist_ok=False);pr.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'outputs/private/research_evaluation/c4_28_v1'
    pins={**read(previous/'protected_hashes.json'),**read(previous/'source_hashes.json')}
    cm=read(ROOT/'outputs/research_evaluation/c4_28_v1/completion_manifest.json')
    if cm['fingerprint']!=digest(cm['files']):raise ValueError('PARENT_MANIFEST')
    pins.update(cm['files']);pins.update(read(previous/'private_index.json')['files'])
    pins['outputs/research_evaluation/c4_28_v1/completion_manifest.json']=sha(ROOT/'outputs/research_evaluation/c4_28_v1/completion_manifest.json')
    for folder in ['app/core','configs','prompts']:
        pins.update({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    original=check(pins);write(pr/'protected_hashes.json',pins);write(pr/'initial_preservation.json',original)
    if original['changed_or_missing']:raise ValueError('OLD_ASSET_CHANGED')
    paths=sources();srcpins={str(p):sha(p) for p in paths.values()}
    spec18=read(ROOT/'outputs/research_evaluation/c4_18_v1/evaluation_protocol.json')
    for entry in read(paths['registered_reports']):
        p=BANK.parent/'runs'/spec18['historical_run']/entry['report_file'];srcpins[str(p)]=sha(p)
    for name in ['C4_18_VOI_EMPIRICAL_EVALUATION_REPORT.md','C4_19_DYNAMIC_HORIZON_AND_RESEARCH_ALIGNMENT_REPORT.md',
                 'C4_20_RESEARCH_GAPS_AND_REMEDIATION_PLAN.md','C4_28_WEIGHT_DIFFERENTIATED_AGENT_REPORT.md']:
        p=ROOT/'docs'/('\uC5F0\uAD6C \uD3C9\uAC00')/name;srcpins[str(p)]=sha(p)
    write(pr/'source_hashes.json',srcpins)
    model=read(paths['model']);InformationOutcomes(model)
    if sha(paths['model'])!=spec18['artifact_sha256']:raise ValueError('FROZEN_MODEL_CHANGED')
    keys=model['provenance']['binning_thresholds'];visits=source_training(rows(paths['training_source']),keys,read(paths['config'])['minute_limit'])
    bank={x['training_case_id']:x for x in rows(paths['bank'])}
    if len(bank)!=len(visits):raise ValueError('BANK_SOURCE_COUNT')
    for v in visits:
        bv=bank[v['case_id']]
        if bv['training_patient_id']!=v['patient_id'] or bv['training_label']!=v['diagnosis']:raise ValueError('BANK_SOURCE_BINDING')
    cases=read(paths['cases']);mapping={}
    for key in keys:
        targets={e['target'] for c in cases for e in c['current_encounter_events'] if numeric(e) and numeric(e)[0]==key}
        if len(targets)!=1:raise ValueError('SOURCE_ACTION_MAPPING')
        matches=[a for a in model['actions'] if a['target'] in targets]
        if len(matches)!=1:raise ValueError('SOURCE_ACTION_NOT_UNIQUE')
        mapping[key]=matches[0]['id']
    if len(mapping)!=len(model['actions']) or any(set(v['values'])!=set(keys) for v in visits):raise ValueError('SOURCE_ACTION_MISSING')
    expected=encode_rows(visits,mapping,keys);stored=read(paths['training_rows'])
    signature=lambda r:tuple(r[k] for k in ('patient_id','visit_id','diagnosis','action_id','outcome_id','split'))
    if sorted(map(signature,expected))!=sorted(map(signature,stored)):raise ValueError('RAW_SOURCE_ENCODED_ROWS_MISMATCH')
    if any(median(v['values'][k]['value'] for v in visits)!=value for k,value in keys.items()):raise ValueError('FROZEN_TRAIN_MEDIAN_MISMATCH')
    folds=grouped_folds(visits,5,42)
    write(pr/'source_visits.json',visits);write(pr/'action_mapping.json',mapping);write(pr/'patient_grouped_split_manifest.json',folds)
    write(pub/'patient_grouped_split_manifest.json',{'seed':42,'folds':5,'rule':'Patient groups, rare diagnosis first; minimize existing per-diagnosis patient count then visits/patients/fold index; hashed seed42 tie order',
        'fixed_universe':'15 development-exposed diagnoses; not reselected using validation',
        'zero_support_policy':'Existing fit_outcomes alpha1 smoothing permits zero cells for development prediction; strict fit_outcome_spec minimum1 gate not promoted',
        'split_fingerprint':digest(folds),'folds_summary':[{'fold':i+1,'validation_patients':sum(f==i for f in folds.values()),'validation_visits':sum(folds[v['patient_id']]==i for v in visits)} for i in range(5)]})
    protocol={'version':VERSION,'prepared_at':datetime.now(timezone.utc).isoformat(),'namespace':namespace,
        'frozen_artifact_sha256':sha(paths['model']),'source_hash_manifest_sha256':sha(pr/'source_hashes.json'),'split_fingerprint':digest(folds),
        'code_hashes':{n:sha(ROOT/n) for n in CODE},'alpha':1,'seed':42,'folds':5,'count_fit_calls_planned':5,
        'fold_contract':'Train-only raw medians, prior and likelihood; original universe and utility unchanged; zero-support cells explicitly flagged, no clinical Artifact gate admission',
        'frozen_scope':'Actual delivered Ask/Test events in one pinned historical run, first unambiguous modeled source per action, all50 retained including zero observations',
        'cv_outcome_scope':'Prior-marginal prediction per observed action; row order is NOT a clinical timeline; final posterior is order-invariant unique action set',
        'baselines':['prior_only_diagnosis','sequential_bayes_for_frozen_or_unordered_final_for_cv','train_action_outcome_counts_plus_alpha1'],
        'metrics':'Natural-log NLL, sum-of-class-squares Brier (binary range0..2), lexical diagnosis tie Top1/Top3, entropy/logK, margin',
        'calibration':'10 equal-width [i/10,(i+1)/10), confidence1 in final bin; ECE count-weighted; source agreement only',
        'zero_rule':'No clipping; probability0 gives undefined numeric NLL plus POSITIVE_INFINITY status/count; denominator0 NOT_APPLICABLE',
        'repeats':'Same action same outcome counted once; conflicting outcomes rejected as Core observe does; no invented tests',
        'policy_spec_fingerprint':digest(policy_spec()),'new_independent_patients':'NOT_AVAILABLE','source_clinical_review':'UNREVIEWED',
        'llm_generation':0,'routing_utility_fit':0,'clinical_gold_created':0,'threshold_tuning':False,'selection_before_metrics':True}
    protocol['fingerprint']=digest(protocol);write(pub/'probability_validation_protocol.json',protocol)
    print(json.dumps({'phase':'PREPARED','visits':len(visits),'patients':len(folds),'actions':len(mapping),'folds':5,'generation':0}),flush=True)


def evaluate_visits(model,visits,mapping,train_rows,timeline=False):
    ind=independent_outcomes(model,train_rows);inverse={v:k for k,v in mapping.items()};aa={a['id']:a for a in model['actions']};out=[]
    for v in visits:
        if v['diagnosis'] not in model['diagnoses']:
            out.append({**v,'status':'NOT_EVALUATED_OOV'});continue
        if v['patient_id'] in model['provenance']['training_patients']:raise ValueError('EVALUATION_PATIENT_IN_TRAIN')
        ev=v['events'] if timeline else [{'action_id':a['id'],'outcome_id':encode(v['values'][inverse[a['id']]]['value'],model['provenance']['binning_thresholds'][inverse[a['id']]]),
            'minute':0,'event_index':i} for i,a in enumerate(model['actions']) if inverse[a['id']] in v['values']]
        final,steps=trajectory(model,ev);observations=[]
        for step in steps:
            aid=step['event']['action_id'];oid=step['event']['outcome_id']
            pred=step['outcome_prediction'] if timeline else {o['id']:sum(model['prior'][d]*o['likelihood'][d] for d in model['prior']) for o in aa[aid]['outcomes']}
            observations.append({'action_id':aid,'outcome_id':oid,'model':probability_score(pred,oid),'independent':probability_score(ind[aid],oid),
                'before':step['before'],'after':step['after'],'predicted_distribution':pred,'actual_outcome_probability':pred[oid],
                'source_event':step['event']})
        progression=[{'step':0,'score':probability_score(model['prior'],v['diagnosis']),'horizon':horizon(model['prior'],model)}]
        if timeline:
            used=[]
            for i,s in enumerate(steps,1):
                used.append(s['event']['action_id']);progression.append({'step':i,'score':probability_score(s['after'],v['diagnosis']),'horizon':horizon(s['after'],model,used)})
        else:progression.append({'step':'unordered_final','score':probability_score(final,v['diagnosis']),'horizon':horizon(final,model,[s['event']['action_id'] for s in steps])})
        out.append({'patient_id':v['patient_id'],'visit_id':v['visit_id'],'diagnosis':v['diagnosis'],'status':'EVALUATED',
            'prior':probability_score(model['prior'],v['diagnosis']),'final':probability_score(final,v['diagnosis']),
            'outcomes':observations,'progression':progression,'final_belief':final,'timeline':'ACTUAL_DELIVERED_PROXY' if timeline else 'UNORDERED_SET_NOT_CLINICAL_TIMELINE'})
    return out


def macro(values,key):
    groups=defaultdict(list)
    for r in values:groups[r['patient_id']].append(r[key])
    patient=[summary(v) for v in groups.values()]
    metrics=('nll','brier','top1','top3','reference_probability','max_probability','normalized_entropy','margin')
    return {'patients':len(groups),'visits':len(values),'patient_macro':{k:sum(p[k] for p in patient)/len(patient) if patient and all(p[k] is not None for p in patient) else None for k in metrics},
        'visit_macro':summary([v[key] for v in values])}


def aggregate(results):
    good=[v for v in results if v['status']=='EVALUATED'];obs=[o for v in good for o in v['outcomes']]
    per_patient=defaultdict(list);per_visit=[]
    for v in good:
        if v['outcomes']:
            vs={k:summary([o[k] for o in v['outcomes']]) for k in ('model','independent')};per_visit.append(vs);per_patient[v['patient_id']].extend(v['outcomes'])
    om={k:summary([o[k] for o in obs]) for k in ('model','independent')}
    return {'registered_visits':len(results),'evaluated_visits':len(good),'oov_visits':len(results)-len(good),
        'diagnosis':{k:macro(good,k) for k in ('prior','final')},'outcomes':om,'outcome_observations':len(obs),
        'outcome_macros':{k:{'patient_macro_nll':sum(summary([o[k] for o in v])['nll'] for v in per_patient.values())/len(per_patient) if per_patient else None,
            'visit_macro_nll':sum(v[k]['nll'] for v in per_visit)/len(per_visit) if per_visit else None} for k in ('model','independent')},
        'calibration':{k:calibration([v[k] for v in good]) for k in ('prior','final')},
        'outcome_calibration':{k:calibration([o[k] for o in obs]) for k in ('model','independent')},
        'reference_probability_increased':sum(v['final']['reference_probability']>v['prior']['reference_probability'] for v in good),
        'reference_probability_decreased':sum(v['final']['reference_probability']<v['prior']['reference_probability'] for v in good),
        'entropy_decreased_reference_probability_decreased':sum(v['final']['normalized_entropy']<v['prior']['normalized_entropy'] and v['final']['reference_probability']<v['prior']['reference_probability'] for v in good)}


def run(namespace):
    pub,pr=locations(namespace);spec=read(pub/'probability_validation_protocol.json')
    if spec['fingerprint']!=digest({k:v for k,v in spec.items() if k!='fingerprint'}):raise ValueError('PROTOCOL_CHANGED')
    for n,h in {**read(pr/'source_hashes.json'),**spec['code_hashes']}.items():
        if sha(Path(n) if Path(n).is_absolute() else ROOT/n)!=h:raise ValueError('INPUT_CODE_CHANGED')
    write(pr/'started.json',{'protocol_fingerprint':spec['fingerprint'],'started_at':datetime.now(timezone.utc).isoformat()})
    p=sources();model=read(p['model']);InformationOutcomes(model);visits=read(pr/'source_visits.json');mapping=read(pr/'action_mapping.json');folds=read(pr/'patient_grouped_split_manifest.json');train_rows=read(p['training_rows'])
    cases=read(p['cases']);labels=read(p['labels']);original=rows(p['original50']);cp={c['case_id']:c for c in cases};orig={c['patient_id']:c for c in original}
    trainp={v['patient_id'] for v in visits};evalp={c['longitudinal']['patient_id'] for c in cases}
    if trainp&evalp or evalp!=set(orig):raise ValueError('FROZEN_PATIENT_OVERLAP_OR_COHORT')
    frozen=[];registered=read(p['registered_reports']);oldspec=read(ROOT/'outputs/research_evaluation/c4_18_v1/evaluation_protocol.json');checked_events=0
    for entry in registered:
        c=cp[entry['case_id']];pid=c['longitudinal']['patient_id'];vid=c['index_encounter_id'];key=json.dumps([pid,vid],ensure_ascii=False)
        truth=labels[key]
        if len(truth)!=1 or truth!=orig[pid]['evaluation']['diagnosis_labels']:raise ValueError('SOURCE_LABEL_BINDING')
        report=read(BANK.parent/'runs'/oldspec['historical_run']/entry['report_file']);visit=report['visits'][0]
        if visit['identity']['patient_id']!=pid or visit['identity']['visit_id']!=vid:raise ValueError('REPORT_SOURCE_IDENTITY')
        events=visit['final_state']['event_log'];source={(e['action'],e['target']):e for e in c['current_encounter_events']};delivered=[]
        for index,e in enumerate(events):
            if e['case_id']!=c['case_id'] or e['event_type'] not in ('result_received','question_asked'):continue
            released_observation(c,events,e['minute'],{'mode':'none','records':[]},before_event_index=index+1)
            kind='Test' if e['event_type']=='result_received' else 'Ask';target=e['payload']['requested_test' if kind=='Test' else 'question'];src=source[kind,target]
            key2,value=numeric(src);aid=mapping[key2];oid=encode(value,model['provenance']['binning_thresholds'][key2])
            if json.loads(src['evidence'])['replay_bin']!=oid:raise ValueError('FROZEN_SOURCE_BIN_MISMATCH')
            delivered.append({'action_id':aid,'outcome_id':oid,'minute':e['minute'],'event_index':index,'source_event_id':src['event_id']});checked_events+=1
        frozen.append({'patient_id':pid,'visit_id':vid,'diagnosis':truth[0],'events':delivered})
    fresults=evaluate_visits(model,frozen,mapping,train_rows,True);write(pr/'frozen_patient_results.json',fresults);fs=aggregate(fresults)
    write(pub/'frozen_artifact_evaluation.json',{'status':'EXECUTED_DEVELOPMENT_SOURCE','patients':len(evalp),**fs})
    all_cv=[];fold_summaries=[];artifacts=[];fit_calls=0
    for fold in range(5):
        tr=[v for v in visits if folds[v['patient_id']]!=fold];va=[v for v in visits if folds[v['patient_id']]==fold]
        artifact,rr=fit_fold(model,tr,mapping,{v['patient_id'] for v in va});fit_calls+=1
        path=pr/('fold%02d_probability_artifact.json'%(fold+1));write(path,artifact)
        support=artifact['provenance']['empirical_support'];zero=sum(n==0 for cell in support.values() for n in cell.values())
        res=evaluate_visits(artifact,va,mapping,rr);all_cv.extend(res);write(pr/('fold%02d_results.json'%(fold+1)),res)
        fold_summaries.append({'fold':fold+1,'train_patients':len({v['patient_id'] for v in tr}),'validation_patients':len({v['patient_id'] for v in va}),
            'train_visits':len(tr),'validation_visits':len(va),'zero_support_action_diagnosis_cells':zero,
            'strict_fit_outcome_spec_min1_gate':'BLOCKED_ZERO_CELLS' if zero else 'COUNT_SUPPORT_ONLY_NOT_CLINICAL_READY',
            'train_diagnosis_visit_counts':[sum(v['diagnosis']==d for v in tr) for d in model['diagnoses']],
            'validation_diagnosis_visit_counts':[sum(v['diagnosis']==d for v in va) for d in model['diagnoses']],**aggregate(res)})
        artifacts.append({'fold':fold+1,'sha256':sha(path),'prior_train_visits':len(tr),'outcome_train_rows':len(rr),'artifact_fingerprint':InformationOutcomes(artifact).fingerprint,'zero_support_cells':zero})
    write(pr/'cv_patient_results.json',all_cv);cs=aggregate(all_cv)
    write(pub/'grouped_cross_validation_results.json',{'status':'DEVELOPMENT_5_FOLD_EXECUTED','folds':fold_summaries,'pooled':cs,'independent_generalization':False})
    write(pub/'fold_artifact_manifest.json',{'actual_count_estimation_calls':fit_calls,'function':'app.core.voi_evaluation.fit_outcomes','alpha':1,'artifacts':artifacts,'routing_utility_fit':0,'clinical_artifact_admission':False})
    write(pub/'diagnosis_probability_metrics.json',{'frozen':fs['diagnosis'],'grouped_cv':cs['diagnosis']})
    outcomes={}
    for name,result in [('frozen',fresults),('grouped_cv',all_cv)]:
        outcomes[name]={'overall':aggregate(result)['outcomes'],'per_action':[{'action_ordinal':i+1,**{k:summary([o[k] for v in result if v['status']=='EVALUATED' for o in v['outcomes'] if o['action_id']==a['id']]) for k in ('model','independent')}} for i,a in enumerate(model['actions'])]}
    write(pub/'outcome_probability_metrics.json',outcomes)
    write(pub/'calibration_analysis.json',{'frozen':fs['calibration'],'grouped_cv':cs['calibration'],'outcomes_frozen':fs['outcome_calibration'],'outcomes_cv':cs['outcome_calibration'],'clinical_calibration':'NOT_VALIDATED'})
    progression={}
    for stage in (0,1,2,3):progression[str(stage)]=summary([s['score'] for v in fresults if v['status']=='EVALUATED' for s in v['progression'] if s['step']==stage])
    progression['final']=summary([v['final'] for v in fresults if v['status']=='EVALUATED'])
    write(pub/'posterior_progression_analysis.json',{'frozen_actual_delivered_stages':progression,'frozen_reference_probability_increased':fs['reference_probability_increased'],
        'frozen_reference_probability_decreased':fs['reference_probability_decreased'],'frozen_entropy_and_reference_probability_both_decreased':fs['entropy_decreased_reference_probability_decreased'],
        'cv_reference_probability_increased':cs['reference_probability_increased'],'cv_reference_probability_decreased':cs['reference_probability_decreased'],
        'cv_intermediate_timeline':'NOT_AVAILABLE; only prior and unordered unique-action final, no invented clinical order'})
    write(pub/'patient_level_metrics.json',{'frozen':{'diagnosis':fs['diagnosis'],'outcomes':fs['outcome_macros']},'grouped_cv':{'diagnosis':cs['diagnosis'],'outcomes':cs['outcome_macros']},'independence_unit':'Patient; visit macro and outcome micro are not independent patient counts'})
    support=[]
    for i,d in enumerate(model['diagnoses'],1):
        cell=[r for r in train_rows if r['diagnosis']==d];n=len({(r['patient_id'],r['visit_id']) for r in cell})
        support.append({'diagnosis_ordinal':i,'patients':len({r['patient_id'] for r in cell}),'visits':n,'prior':model['prior'][d],
            'actions':[{'action_ordinal':j+1,'support':sum(r['action_id']==a['id'] for r in cell),'outcome_counts':[sum(r['action_id']==a['id'] and r['outcome_id']==o['id'] for r in cell) for o in a['outcomes']],
                'smoothing_pseudocount_fraction':2/(sum(r['action_id']==a['id'] for r in cell)+2)} for j,a in enumerate(model['actions'])]})
    write(pub/'diagnosis_and_action_support.json',{'diagnoses':support,'prior_unit':'unique patient/visit pairs, not unique patients','missing_policy':'missing actions add no likelihood count; visits with any observed action enter prior once; current complete229x5 cohort',
        'repeated_patient_weighting':'Each visit counted independently, patient with multiple visits has more prior/count weight','zero_outcome_cells':sum(x==0 for s in support for a in s['actions'] for x in a['outcome_counts']),
        'minimum_action_diagnosis_support':min(a['support'] for s in support for a in s['actions']),'maximum_action_diagnosis_support':max(a['support'] for s in support for a in s['actions'])})
    write(pub/'likelihood_discriminability.json',discriminability(model))
    historical=rows(p['replay']);hrows=[];aa={a['id']:a for a in model['actions']}
    for h in historical:
        belief,matched=InformationOutcomes(model).observe(h['model_observation'])
        if belief!=h['belief'] or matched!=h['matched_outcomes']:raise ValueError('REPLAY_BELIEF_CHANGED')
        ids={x['action_id'] for x in h['horizons']['3']['plan']['action_values']}
        actionmenu=[{'action_id':a['id'],'type':'Ask' if a['type']=='ask_patient' else 'Test','action':a['type'],'target':a['target'],'outcome_model':a['id']} for a in model['actions'] if a['id'] in ids]
        hrows.append(DynamicHorizonPolicy().select(belief=belief,available_actions=actionmenu,observable_context={}))
    def hs(seq):
        return {'states':len(seq),'selected_horizons':dict(Counter(str(s['selected_horizon']) for s in seq)),
            'uncertainty_requested_horizons':dict(Counter(str(s['observable_feature_summary'].get('uncertainty_requested_horizon')) for s in seq)),
            'entropy_min':min((s['observable_feature_summary']['normalized_entropy'] for s in seq),default=None),'entropy_max':max((s['observable_feature_summary']['normalized_entropy'] for s in seq),default=None)}
    write(pub/'horizon_saturation_analysis.json',{'historical_263_revalidated':hs(hrows),'frozen_actual_progression':hs([s['horizon'] for v in fresults if v['status']=='EVALUATED' for s in v['progression']]),
        'grouped_cv_prior_and_unordered_final':hs([s['horizon'] for v in all_cv if v['status']=='EVALUATED' for s in v['progression']]),'thresholds_unchanged':True,'capped_selected_vs_uncertainty_requested_are_distinct':True})
    write(pub/'probability_model_inventory.json',{'diagnoses':len(model['diagnoses']),'training_patients':len(trainp),'training_visits':len(visits),'observations':len(train_rows),
        'actions':len(model['actions']),'types':dict(Counter(a['type'] for a in model['actions'])),'outcomes_per_action':[len(a['outcomes']) for a in model['actions']],
        'alpha':1,'prior_cells':len(model['prior']),'synthetic':model['provenance']['synthetic'],'review_status':model['provenance']['review_status'],'utility_values':sorted({v for row in model['terminal_utility'].values() for v in row.values()}),
        'model_sha256':sha(p['model']),'raw_source_encoded_row_match':True,'train_median_match':True,'universe_evaluation_informed':True,
        'signatures':{name:str(inspect.signature(fn)) for name,fn in [('fit_outcomes',fit_outcomes),('posterior',posterior),('InformationOutcomes.observe',InformationOutcomes.observe)]}})
    write(pub/'source_eligibility.json',{'training_patients':len(trainp),'training_visits':len(visits),'frozen_patients':len(evalp),'frozen_visits':len(frozen),
        'actually_delivered_source_observations':checked_events,'scored_unique_observations':fs['outcome_observations'],'patients_with_observations':sum(bool(v['outcomes']) for v in fresults if v['status']=='EVALUATED'),
        'zero_observation_visits_retained':sum(not v['outcomes'] for v in fresults if v['status']=='EVALUATED'),'oov':fs['oov_visits'],'new_independent_holdout':'NOT_AVAILABLE','clinical_reference':'UNREVIEWED_SOURCE_DIAGNOSIS'})
    write(pub/'leakage_and_exposure_audit.json',{'frozen_train_patient_overlap':len(trainp&evalp),'fold_patient_overlap':0,'fold_visit_overlap':0,'all_original50_retained':True,
        'train_only_medians_prior_likelihood':True,'actual_fold_estimation_calls':fit_calls,'per_query_diagnosis_used_to_initialize_belief':False,'source_diagnosis_used_by_evaluator_and_train_counts_only':True,
        'future_release_block':'Delivered event prefix with recorded order/result/source checked; no later events in earlier update',
        'duplicate_actions':'Same outcome collapsed once; conflicting outcomes rejected','actual_clinical_available_at':'UNKNOWN; recorded-time proxy',
        'diagnosis_universe_development_exposure':True,'original50_development_exposure':True,'training_selection_complete_action_bias':True,'independent_clinical_validation':False,
        'routing_validation_voi_train_overlap':5,'cross_component_split_issue_preserved':True,'raw_llm_wire_reconstructed':False})
    # Predeclared support criterion; no threshold sweep or fitted calibration.
    supported=all(s['diagnosis']['final']['visit_macro']['nll']<s['diagnosis']['prior']['visit_macro']['nll'] and s['outcomes']['model']['nll']<s['outcomes']['independent']['nll'] for s in [fs,cs])
    decision='A_PROBABILITY_MODEL_HAS_DEVELOPMENT_PREDICTIVE_SUPPORT' if supported else 'B_LIKELIHOOD_HAS_LIMITED_DISCRIMINABILITY'
    with (pub/'next_research_decision.md').open('x',encoding='utf-8') as f:f.write('# Decision\n\n'+decision+'\n\nBoth frozen and grouped development comparisons must improve diagnosis NLL and outcome NLL for A. No independent clinical validation, no clinical utility claim, no threshold/alpha tuning. If A fails, inspect existing outcome encoding and information-action discriminability under a separate protocol; do not automatically refit the official model. Small-sample ECE is descriptive.\n')
    print(json.dumps({'phase':'EVALUATED','frozen_patients':len(evalp),'frozen_observations':fs['outcome_observations'],'cv_patients':len(trainp),'cv_visits':len(all_cv),'cv_observations':cs['outcome_observations'],'fit_calls':fit_calls,'decision':decision}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','run']);p.add_argument('--namespace',default='c4_29_v1');a=p.parse_args()
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('OFFLINE_NO_NETWORK_OR_MODEL_CALLS')
    sys.addaudithook(guard)
    (prepare if a.phase=='prepare' else run)(a.namespace)


if __name__=='__main__':main()
