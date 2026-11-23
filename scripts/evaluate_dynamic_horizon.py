"""C4-19 preregistered opt-in development experiment; no live execution/fitting."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import csv
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core.diagnostic_belief import terminal_choice
from app.core.information_outcomes import InformationOutcomes
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.routing_data_reuse import read,BANK
from app.locdi.routing_two_decision_collection import verify_pins,verify_fingerprint
from app.locdi.voi_empirical_evaluation import decision_key,distribution_stats
from app.locdi.dynamic_horizon import DynamicHorizonPolicy,policy_spec
from app.locdi.dynamic_horizon_evaluation import action_interface,run_planner,run_dynamic,common_value,q_diagnostics,patient_summaries,restore_model_belief,complete_identity_recall
from scripts.enrich_full_cohort_sources import write,rows

PARENT=ROOT/'outputs/private/research_evaluation/c4_18_v1'
PARENT_PUBLIC=ROOT/'outputs/research_evaluation/c4_18_v1'


def locations(namespace):
    if not re.fullmatch(r'c4_19_[A-Za-z0-9_-]{1,48}',namespace):raise ValueError('C4_19_NAMESPACE_REQUIRED')
    return ROOT/'outputs/research_evaluation'/namespace,ROOT/'outputs/private/research_evaluation'/namespace


def freeze(namespace):
    pub,pri=locations(namespace)
    if pub.exists() or pri.exists():raise ValueError('UNUSED_NAMESPACE_REQUIRED')
    m=load(PARENT_PUBLIC/'output_manifest.json');verify_fingerprint(m);ip=ROOT/m['private_index_path']
    if sha(ip)!=m['private_index_sha256']:raise ValueError('PARENT_PRIVATE_INDEX_HASH')
    pins=load(PARENT/'protected_hashes.json');pins.update(m['files']);pins.update(load(ip)['files'])
    for p in [PARENT_PUBLIC/'output_manifest.json',ip]:pins[p.relative_to(ROOT).as_posix()]=sha(p)
    sources=load(PARENT/'source_hashes.json');verify_pins(pins);verify_pins(sources)
    pub.mkdir(parents=True);pri.mkdir(parents=True)
    write(pri/'protected_hashes.json',pins);write(pri/'source_hashes.json',sources)
    spec=policy_spec()
    protocol={'version':'locdi.dynamic_horizon.evaluation.c4_19.v1','frozen_at':datetime.now(timezone.utc).isoformat(),
        'spec':spec,'registered_source':'C4-18 sealed263 decisions from49 patients within registered50, retain failed zero-decision case',
        'source_trace_sha256':sha(PARENT/'decision_trace.jsonl'),'primary':'entropy_margin_feasibility',
        'fixed_baselines':[1,2,3],'common_evaluation_horizon':3,
        'common_value':'Q3 of selected first information action, with optimal continuation at depth2; STOP gets same terminalT, not V3. Model internal descriptive difference only, never clinical regret',
        'timing':'One passive plan_voi observer per select_action call; count returned evaluated_states, separate policy selection wall time; primary execution order cyclically rotated across decisions; repeat measurements excluded from main timing',
        'near_tie':'Preserve Core exact lexical tie and root stop epsilon; annotate gaps<=1e-12 only for audit, never change actions',
        'development_exposure':'Policy designed after viewing C4-18; exploratory even though thresholds/spec frozen before C4-19 outcomes',
        'independent_validation':'NOT_RUN','llm_generation':0,'utility_fitting':0,'new_clinical_labels':0}
    write(pub/'dynamic_horizon_policy_spec.json',spec);write(pub/'evaluation_protocol.json',protocol)
    write(pri/'initial_preservation.json',{'status':'PASS','protected_files':len(pins),'source_files':len(sources),
        'new_policy_outcomes_computed':False,'protocol_fingerprint':digest(protocol)})
    print('FROZEN',namespace,flush=True)


def without_timing(x):
    if isinstance(x,dict):return {k:without_timing(v) for k,v in x.items() if k not in {'wall_seconds','selection_wall_seconds','total_wall_seconds'}}
    if isinstance(x,list):return [without_timing(v) for v in x]
    return x


def run(namespace,opt_in):
    if opt_in is not True:raise ValueError('EXPLICIT_DEVELOPMENT_OPT_IN_REQUIRED')
    pub,pri=locations(namespace)
    if (pri/'execution_started.json').exists():raise ValueError('NO_RESUME_OR_OVERWRITE')
    spec=load(pub/'dynamic_horizon_policy_spec.json');protocol=load(pub/'evaluation_protocol.json')
    if spec!=policy_spec() or protocol['spec']!=spec:raise ValueError('FROZEN_POLICY_SPEC_MISMATCH')
    if digest(protocol)!=load(pri/'initial_preservation.json')['protocol_fingerprint']:raise ValueError('FROZEN_PROTOCOL_CHANGED')
    if sha(PARENT/'decision_trace.jsonl')!=protocol['source_trace_sha256']:raise ValueError('SOURCE_TRACE_CHANGED')
    pins=load(pri/'protected_hashes.json');sources=load(pri/'source_hashes.json');verify_pins(pins);verify_pins(sources)
    codefiles=[ROOT/'app/locdi/dynamic_horizon.py',ROOT/'app/locdi/dynamic_horizon_evaluation.py',Path(__file__).resolve()]
    write(pri/'execution_started.json',{'started_at':datetime.now(timezone.utc).isoformat(),
        'protocol_fingerprint':digest(protocol),'code_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in codefiles},
        'explicit_development_opt_in':True})
    artifact=load(BANK.parent/'voi_model.json');model=InformationOutcomes(artifact)
    trace=read(PARENT/'decision_trace.jsonl');registered=load(PARENT/'registered_historical_reports.json')
    if len(trace)!=263 or len({t['case_ref'] for t in trace})!=49 or len(registered)!=50:raise ValueError('FULL_PARENT_COHORT_REQUIRED')
    runpath=BANK.parent/'runs/live_20261024_090220_e77410'
    contexts={};reports={}
    for i,e in enumerate(registered,1):
        ref='HistoricalCase%03d'%i;path=runpath/e['report_file'];report=load(path);reports[ref]=(report,sha(path))
        state=next(iter(report['visits'][0]['final_state']['cases'].values()))
        for a in state['llm_audit']:
            if 'specialist_proposal' in a and a.get('status')=='success':
                key=(ref,a['input_sha256'])
                contexts.setdefault(key,[]).append(a)
    full=[];public=[];feature_rows=[]
    for ordinal,t in enumerate(trace):
        if t['source_report_sha256']!=reports[t['case_ref']][1] or t['artifact_fingerprint']!=model.fingerprint:raise ValueError('SOURCE_REPORT_OR_MODEL_FINGERPRINT')
        actual_belief,matched=restore_model_belief(model,t['model_observation'],t['belief'],t['matched_outcomes'])
        audits=[a for a in contexts[(t['case_ref'],t['recorded_core_input_sha256'])] if a['selector_status']['values']==t['horizons']['2']['selection']['values'] or
            [{k:v for k,v in c.items() if k!='cost'} for c in a['selector_status']['values']]==t['horizons']['2']['selection']['values']]
        if len(audits)!=1:raise ValueError('DECISION_CONTEXT_NOT_UNIQUE')
        audit=audits[0];belief=actual_belief;interface=action_interface(t['horizons']['2']['selection']['values'])
        utility=model.artifact['terminal_utility'];scores={c:sum(belief[d]*u[d] for d in belief) for c,u in utility.items()}
        best=max(scores.values())
        context={'decision_minute':t['minute'],
            'remaining_minutes':audit.get('decision_validation',{}).get('remaining_visit_minutes'),
            'resources':None,'terminal_utility':best,'terminal_tie_count':sum(s==best for s in scores.values()),
            'terminal_action_available':any(c['type']=='Diagnose' for c in interface),
            'used_action_count':len(matched),'time_source':'stored decision_validation.remaining_visit_minutes',
            'resource_source':'NOT_RECORDED_AT_DECISION; final scheduler resources not used',
            'terminal_support_status':'NOT_CLINICALLY_REVIEWED'}
        results={};order=['H1','H2','H3','dynamic'];shift=ordinal%4;order=order[shift:]+order[:shift]
        for condition in order:
            results[condition]=run_dynamic(belief,interface,model.artifact,context,development_opt_in=True) if condition=='dynamic' else run_planner(belief,interface,model.artifact,int(condition[1]))
        for h in ('1','2','3'):
            if results['H'+h]['selection']!=t['horizons'][h]['selection'] or results['H'+h]['plan']!=t['horizons'][h]['plan']:
                raise ValueError('C4_18_FIXED_BASELINE_NOT_REPRODUCED')
        repeat=run_dynamic(belief,interface,model.artifact,context,development_opt_in=True)
        if without_timing(repeat)!=without_timing(results['dynamic']):raise ValueError('DYNAMIC_REPRODUCTION_MISMATCH')
        for variant in spec['ablations']:
            results[variant]=run_dynamic(belief,interface,model.artifact,context,variant=variant,development_opt_in=True)
        h3=results['H3']['plan'];primary=results['dynamic'];features=primary['horizon_policy']['observable_feature_summary']
        derived={}
        for name,result in results.items():
            choice=result['selection']['selected_action']
            derived[name]={'binding':decision_key(choice),'action':choice['action'],'stop':result['plan']['selected_action_id'] is None,
                'common_H3_action_value':common_value(choice,h3),'self_horizon_selected_q':choice['q_value'],
                'q_diagnostics':q_diagnostics(result['plan'])}
        row={'case_ref':t['case_ref'],'decision_ordinal':t['decision_ordinal'],'minute':t['minute'],
            'parent_input_fingerprint':t['input_fingerprint'],'candidate_fingerprint':digest(interface),
            'artifact_fingerprint':model.fingerprint,'context':context,'horizon_input_belief':belief,
            'information_action_interface':[c for c in interface if c['type']!='Diagnose'],
            'results':results,'derived':derived,'repeat_equal_excluding_timing':True,'primary_execution_order':order}
        full.append(row);feature_rows.append(features)
        public.append({'case_ref':t['case_ref'],'decision_ordinal':t['decision_ordinal'],'minute':t['minute'],
            'source_input_fingerprint':t['input_fingerprint'],'candidate_fingerprint':digest(interface),
            'artifact_fingerprint':model.fingerprint,'horizon_policy':primary['horizon_policy'],
            'selected_action_type':derived['dynamic']['action'],'stop':derived['dynamic']['stop'],
            'changed_vs_H2':derived['dynamic']['binding']!=derived['H2']['binding'],
            'common_H3_value_delta_vs_H2':derived['dynamic']['common_H3_action_value']-derived['H2']['common_H3_action_value'],
            'q_diagnostics':derived['dynamic']['q_diagnostics']})
        if (ordinal+1)%50==0:print('DYNAMIC_REPLAY',ordinal+1,flush=True)
    rows(pri/'dynamic_execution_trace.jsonl',full)
    write(pub/'dynamic_horizon_execution_trace.json',{'version':spec['policy_version'],'scope':'OPT_IN_DEVELOPMENT_SOURCE_REPLAY',
        'registered_cases':50,'cases_with_decisions':49,'decisions':263,'rows':public,'repeat_equal':True,
        'unexecuted_case':{'case_ref':'HistoricalCase030','historical_status':'failed','voi_decisions':0,'new_outcomes_generated':False}})
    summarize(pub,full,feature_rows,spec)
    consolidate(pub)
    verify_pins(pins);verify_pins(sources)
    write(pub/'verification.json',{'status':'PASS','final_status':'C4_19_DYNAMIC_HORIZON_IMPLEMENTED_AND_EVALUATED',
        'protected_files':len(pins),'source_files':len(sources),'changed_or_missing':0,'decisions':263,'patients_with_decisions':49,
        'registered_patients':50,'fixed_baseline_full_return_matches':789,'dynamic_repeats_equal':263,
        'core_or_existing_artifact_changed':False,'new_llm_generation':0,'utility_fitting':0,'artifact_retraining':0,
        'clinical_verdicts':0,'independent_validation':'NOT_RUN','manuscript_alignment':'NOT_CHECKED_NO_MANUSCRIPT_IN_SEARCH_SCOPE'})
    print('C4_19_DYNAMIC_HORIZON_IMPLEMENTED_AND_EVALUATED',flush=True)


def summarize(pub,full,features,spec):
    n=len(full);names=['H1','H2','H3','dynamic']+spec['ablations']
    def condition(name,group=full):
        rs=[r['results'][name] for r in group];ds=[r['derived'][name] for r in group]
        return {'decisions':len(group),'action_types':dict(Counter(d['action'] for d in ds)),
            'stop_count':sum(d['stop'] for d in ds),'continue_count':sum(not d['stop'] for d in ds),
            'horizon_selection':dict(Counter(str(r.get('horizon_policy',{}).get('selected_horizon',r['plan']['horizon'])) for r in rs)),
            'fallback_count':sum(bool(r.get('horizon_policy',{}).get('fallback_reason')) for r in rs),
            'common_H3_action_value':distribution_stats([d['common_H3_action_value'] for d in ds]),
            'self_horizon_selected_q':distribution_stats([d['self_horizon_selected_q'] for d in ds]),
            'planner_states_total':sum(r['planner_evaluated_states'] for r in rs),
            'planner_states':distribution_stats([r['planner_evaluated_states'] for r in rs]),
            'selector_planner_wall_seconds_total':sum(r['wall_seconds'] for r in rs),
            'selector_planner_wall_seconds':distribution_stats([r['wall_seconds'] for r in rs]),
            'horizon_selection_wall_seconds_total':sum(r.get('horizon_policy',{}).get('selection_wall_seconds',0) for r in rs)}
    def pair(a,b,group=full):
        return {'decisions':len(group),'action_binding_changes':sum(r['derived'][a]['binding']!=r['derived'][b]['binding'] for r in group),
            'action_agreement':sum(r['derived'][a]['binding']==r['derived'][b]['binding'] for r in group)/len(group) if group else None,
            'stop_continue_changes':sum(r['derived'][a]['stop']!=r['derived'][b]['stop'] for r in group),
            'common_H3_action_value_delta_a_minus_b':distribution_stats([r['derived'][a]['common_H3_action_value']-r['derived'][b]['common_H3_action_value'] for r in group])}
    summaries={name:condition(name) for name in names};comparisons={h:pair('dynamic',h) for h in ['H1','H2','H3']}
    write(pub/'fixed_vs_dynamic_horizon.json',{'conditions':{k:summaries[k] for k in ['H1','H2','H3','dynamic']},
        'dynamic_comparisons':comparisons,'common_evaluation_horizon':3,'comparison_scope':'Model internal same-H3 selected-first-action value, not actual regret or clinical benefit',
        'primary_policy':spec['primary_policy'],'independent_validation':'NOT_RUN'})
    write(pub/'dynamic_horizon_ablation.json',{'conditions':{k:summaries[k] for k in ['H2','entropy_only','dynamic','margin_only','feasibility_removed_fixed_fallback']},
        'differences_vs_primary':{k:pair(k,'dynamic') for k in spec['ablations']},
        'feature_semantics':{'entropy_only':'Remove margin; mandatory safety cap retained','margin_only':'Remove entropy; mandatory safety cap retained',
            'feasibility_removed_fixed_fallback':'Action feasibility is required for safe bounded selection; remove feature -> explicit fixedH2 fallback, safety cap stays'},
        'clinical_or_official_ablation':False})
    counts=Counter(str(r['results']['dynamic']['horizon_policy']['selected_horizon']) for r in full)
    demands=Counter(f.get('uncertainty_requested_horizon') for f in features if f['remaining_information_actions'])
    write(pub/'dynamic_horizon_feature_audit.json',{'decisions':n,'features':{k:distribution_stats([f[k] for f in features]) for k in ['entropy_nats','normalized_entropy','max_posterior','top1_top2_margin','terminal_utility','remaining_information_actions']},
        'horizon_distribution':dict(counts),'uncertainty_requested_horizon_distribution':dict(demands),
        'action_cap_active':sum(f.get('action_cap_active',False) for f in features),
        'terminal_only':sum(f['remaining_information_actions']==0 for f in features),
        'time_observed':sum(f['time_status']=='OBSERVED' for f in features),'resource_observed':sum(f['resource_status']!='UNKNOWN' for f in features),
        'time_resource_used_to_select':False,'uncertainty_band_single_value':len(demands)==1,
        'same_actions_as_fixedH3':comparisons['H3']['action_binding_changes']==0,
        'same_horizon_for_all_nonterminal_decisions':len({k for k in counts if k!='None'})<=1,
        'clinical_calibration':'NOT_VALIDATED','diagnosis_coding_values_exposed_publicly':False})
    write(pub/'patient_level_horizon_analysis.json',patient_summaries(full,['HistoricalCase%03d'%i for i in range(1,51)]))
    oldchanged=[r for r in full if r['derived']['H2']['binding']!=r['derived']['H3']['binding']]
    tie_summary={name:{'near_information_ties':sum(r['derived'][name]['q_diagnostics']['near_information_tie'] for r in full),
        'exact_information_ties':sum(r['derived'][name]['q_diagnostics']['exact_information_tie'] for r in full),
        'near_stop_boundary':sum(r['derived'][name]['q_diagnostics']['near_stop_boundary'] for r in full)} for name in ['H1','H2','H3','dynamic']}
    write(pub/'near_tie_sensitivity.json',{'tolerance_annotation_only':1e-12,'core_tie_breaking_changed':False,'conditions':tie_summary,
        'parent_H2_H3_changed_decisions':len(oldchanged),
        'parent_changed_with_near_tie_in_at_least_one':sum(r['derived']['H2']['q_diagnostics']['near_information_tie'] or r['derived']['H3']['q_diagnostics']['near_information_tie'] for r in oldchanged),
        'parent_changed_with_near_tie_in_both':sum(r['derived']['H2']['q_diagnostics']['near_information_tie'] and r['derived']['H3']['q_diagnostics']['near_information_tie'] for r in oldchanged),
        'common_H3_value_difference_H3_minus_H2':distribution_stats([r['derived']['H3']['common_H3_action_value']-r['derived']['H2']['common_H3_action_value'] for r in oldchanged]),
        'changed_parent_pairs_common_H3_gap_at_most_epsilon':sum(abs(r['derived']['H3']['common_H3_action_value']-r['derived']['H2']['common_H3_action_value'])<=1e-12 for r in oldchanged),
        'clinical_meaning_inferred':False})
    write(pub/'computation_cost_analysis.json',{'conditions':{k:{f:v for f,v in summaries[k].items() if 'wall_seconds' in f or 'planner_states' in f} for k in names},
        'primary_selector_invocations':4*n,'main_dynamic_horizon_selections':n,'additional_primary_repeat_invocations':n,
        'ablation_invocations':3*n,'common_H3_scoring_additional_planner_invocations':0,
        'planner_state_metric':'Existing returned evaluated_states counts cached recursive v calls, not root/action enumeration operations',
        'execution_order':'Cyclic rotation of H1,H2,H3,dynamic by decision; ablations afterward',
        'warmup':'NONE','timing_scope':'Instrumented select_action call (including passive observer copy/patch); artifact-view deepcopy preprocessing excluded; dynamic selection overhead separate',
        'speedup_or_clinical_efficiency_claimed':False})


def csv_file(path,records):
    if not records:raise ValueError('CSV_RECORDS_REQUIRED')
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def consolidate(pub):
    memory=load(ROOT/'outputs/research_evaluation/c4_10_v1/csv_export_aligned/retrieval_evaluation.json')
    selective=next(r for r in memory['summary'] if r['mode']=='selective')
    labs=Counter()
    for c in memory['cases']:labs.update(c['labels'])
    recall=complete_identity_recall(memory)
    recall_num,recall_den=recall['numerator'],recall['denominator']
    base=list((ROOT/'outputs/private/routing_collection/phase24b_collect_001').glob('Candidate*_Agent*.json'))
    baseline_counts=Counter(load(p)['response']['certainty'] for p in base)
    new=load(ROOT/'outputs/research_evaluation/c4_14c_v1/response_support_matrix.json')
    routing=load(ROOT/'outputs/research_evaluation/c4_17_v1/domain_agreement_metrics.json')
    old=load(PARENT_PUBLIC/'off_on_comparison.json');dyn=load(pub/'fixed_vs_dynamic_horizon.json')
    table=[{'area':'Memory','evidence':'C4-10 source measurement identity','metric':'positive/negative/unknown','value':str(dict(labs)),'denominator':'72590 records','scope':'Not clinical relevance'},
        {'area':'Memory','evidence':'C4-10 selective5','metric':'judged identity precision','value':str(selective['verified_identity_judged_precision']),'denominator':'243/250 judged returned','scope':'Development source-code identity only'},
        {'area':'Memory','evidence':'C4-10 complete query subset','metric':'micro recall','value':str(recall_num/recall_den),'denominator':str(recall_num)+'/'+str(recall_den)+';40 queries','scope':'No full50 clinical recall'},
        {'area':'Routing','evidence':'Original collection75','metric':'response certainty','value':str(dict(baseline_counts)),'denominator':'75 responses','scope':'No independent correctness'},
        {'area':'Routing','evidence':'C4-14C collection10','metric':'supported/unresolved','value':'4/6','denominator':'10 separate responses','scope':'Stored matrix read, no merging'},
        {'area':'Routing','evidence':'C4-17 source domain','metric':'confirmed positive matches static/rule/development','value':'/'.join(str(routing['conditions'][k]['confirmed_positive_tag_match_count']) for k in ['A','B','C']),'denominator':'15 snapshots;11 positive-tag cases','scope':'Not clinical routing accuracy'},
        {'area':'Routing','evidence':'Current Gate','metric':'verified adaptive utility','value':'0','denominator':'independent reviewed utility','scope':'BLOCKED'},
        {'area':'VoI','evidence':'C4-18 frozen snapshot replay','metric':'OFF/ON action changes','value':str(old['changed_actions']),'denominator':'263 decisions in49 patients','scope':'Unreviewed model sensitivity, not clinical utility'},
        {'area':'VoI','evidence':'C4-19 new development','metric':'dynamic vsH2 action changes','value':str(dyn['dynamic_comparisons']['H2']['action_binding_changes']),'denominator':'263 decisions in49 patients','scope':'New after C4-18, exploratory only'}]
    csv_file(pub/'consolidated_results_table.csv',table)
    claims=[{'claim':'Source-grounded memory identity retrieval','status':'IMPLEMENTED_AND_EMPIRICALLY_EVALUATED','evidence':'C4-10 final csv_export_aligned','boundary':'No clinical relevance/diagnostic benefit'},
        {'claim':'Memory clinical relevance','status':'EVALUATION_BLOCKED','evidence':'Independent review missing','boundary':'Identity labels are not clinical labels'},
        {'claim':'Adaptive Routing scoring and abstention','status':'IMPLEMENTED_STRUCTURALLY_TESTED','evidence':'Existing Core/tests','boundary':'Verified accuracy Utility0'},
        {'claim':'Clinical Routing accuracy','status':'EVALUATION_BLOCKED','evidence':'C4-17 partial source tags only','boundary':'Clinical reference0'},
        {'claim':'Cost-free finite-horizon Bayesian VoI','status':'IMPLEMENTED_NOT_CLINICALLY_VALIDATED','evidence':'C4-03 numerical repair,C4-18 source replay','boundary':'Unreviewed likelihood and validation1/0 U'},
        {'claim':'Adaptive Dynamic Horizon','status':'NEW_DEVELOPMENT_EXPERIMENT','evidence':'C4-19 opt-in263 replay','boundary':'Not implemented in historical C4-18; no clinical validation'},
        {'claim':'Sequential Routing+VoI runtime links','status':'IMPLEMENTED_STRUCTURALLY_TESTED','evidence':'C4-18 existing TEST_ONLY static/adaptive','boundary':'Live reviewed integrated executor blocked'},
        {'claim':'Official B0-B4 evaluation','status':'NOT_IMPLEMENTED','evidence':'experiments/run_locdi_ablation.py raises NotImplementedError','boundary':'No dynamic injection or official result'},
        {'claim':'Graduation manuscript paragraph/equation alignment','status':'NOT_CHECKED','evidence':'No manuscript located in repository/nondata document search or immediate parent files','boundary':'No invented paragraphs/equation numbers'}]
    csv_file(pub/'research_claim_evidence_matrix.csv',claims)
    write(pub/'implementation_status.json',{'new_policy_version':policy_spec()['policy_version'],'new_development_modules':['app/locdi/dynamic_horizon.py','app/locdi/dynamic_horizon_evaluation.py'],
        'evaluation_script':'scripts/evaluate_dynamic_horizon.py','policy_opt_in_required_for_execution':True,
        'existing_core_changed':False,'live_hook_installed':False,'independent_clinical_validation':False,
        'manuscript_alignment':'NOT_CHECKED','historical_claims_rewritten':False,
        'baseline_routing_response_counts':dict(baseline_counts),'new10_matrix_fingerprint':digest(new)})
    write(pub/'evaluation_eligibility.json',{'development_replay':'EXECUTED','independent_validation':'NOT_RUN',
        'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False,'raw_reference_created':False,
        'clinical_utility':'NOT_APPROVED','source_catalog_evaluation_exposure':'KNOWN_C4_18',
        'routing_validation_voi_train_overlap':5,'new_holdout_acquired':False})
    write(pub/'ablation_readiness.json',{'dynamic_feature_ablation':'EXECUTED_DEVELOPMENT_ONLY','official_B0_B4':'BLOCKED',
        'official_runner':'NotImplementedError after gate; official ready false and artifacts unresolved',
        'official_configuration_or_prompt_changed':False,'proposed_extension':'Separate future protocol/version needed; no automatic B4 migration'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','run']);p.add_argument('--namespace',default='c4_19_v1')
    p.add_argument('--enable-development-dynamic-horizon',action='store_true');a=p.parse_args()
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind'}:raise RuntimeError('OFFLINE_DEVELOPMENT_NO_MODEL_CALLS')
    sys.addaudithook(guard)
    if a.command=='freeze':freeze(a.namespace)
    else:run(a.namespace,a.enable_development_dynamic_horizon)


if __name__=='__main__':main()
