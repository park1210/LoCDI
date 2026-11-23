"""C4-32 frozen Dynamic Source evaluation; no generation/fitting/Core changes."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import inspect
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,rows,sha,write,check
from scripts.evaluate_voi_sequential_policies import load_panel, dump_line, PREP, PARENT as SOURCE_PARENT
from scripts.analyze_voi_sequential_policies import aubc_rows,paired_comparison
from app.locdi.dynamic_horizon import DynamicHorizonPolicy,policy_spec
from app.locdi.dynamic_horizon_source_evaluation import *
from app.locdi.voi_sequential_policy_evaluation import mean_metrics,SCORE_KEYS

OLD=ROOT/'outputs/private/research_evaluation/c4_31_v1'
OLDPUB=ROOT/'outputs/research_evaluation/c4_31_v1'
CODE=['app/locdi/dynamic_horizon_source_evaluation.py','scripts/evaluate_dynamic_horizon_source_policy.py','tests/locdi/test_dynamic_horizon_source_evaluation.py']


def places(ns):
    if not ns or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in ns):raise ValueError('INVALID_NAMESPACE')
    return ROOT/'outputs/research_evaluation'/ns,ROOT/'outputs/private/research_evaluation'/ns


def merge_pins(target,extra):
    for n,h in extra.items():
        if n in target and target[n]!=h:raise ValueError('CONFLICTING_INHERITED_PIN')
        target[n]=h


def prepare(ns):
    pub,pr=places(ns)
    if pub.exists() or pr.exists():raise ValueError('NO_OVERWRITE_OR_RESUME')
    pins=read(OLD/'protected_hashes.json')
    for p in (OLDPUB/'completion_manifest.json',OLD/'private_index.json'):
        m=read(p)
        if digest(m['files'])!=m['fingerprint']:raise ValueError('PARENT_MANIFEST')
        merge_pins(pins,m['files']);merge_pins(pins,{p.relative_to(ROOT).as_posix():sha(p)})
    sources=read(OLD/'source_hashes.json')
    if check(pins)['changed_or_missing'] or check(sources)['changed_or_missing']:raise ValueError('PARENT_PRESERVATION_FAILED')
    inputs,ident,truth,models,values,verification=load_panel()
    saved=read(ROOT/'outputs/research_evaluation/c4_19_v1/dynamic_horizon_policy_spec.json')
    if saved!=policy_spec():raise ValueError('C4_19_POLICY_SPEC_CHANGED')
    policy=DynamicHorizonPolicy();expected=digest({'spec':saved,'variant':'entropy_margin_feasibility'})
    if policy.fingerprint!=expected:raise ValueError('POLICY_FINGERPRINT')
    chosen=[p['scenario_ordinal'] for fold in range(1,6) for p in [r for r in inputs if r['fold']==fold][:2]]
    extra=['app/locdi/dynamic_horizon.py','app/locdi/dynamic_horizon_evaluation.py',
        'app/locdi/voi_sequential_policy_evaluation.py','app/locdi/voi_source_replay_environment.py',
        'app/core/voi_planning.py','app/core/diagnostic_belief.py','app/core/information_value.py',
        'scripts/evaluate_voi_sequential_policies.py','scripts/analyze_voi_sequential_policies.py']
    spec={'protocol':VERSION,'namespace':ns,'prepared_at':datetime.now(timezone.utc).isoformat(),
        'policy_version':saved['policy_version'],'policy_fingerprint':expected,'policy_spec':saved,
        'parent_protocol_fingerprint':read(OLDPUB/'sequential_policy_protocol.json')['fingerprint'],
        'code_hashes':{n:sha(ROOT/n) for n in CODE+extra},
        'input_hashes':{**read(OLDPUB/'frozen_artifact_verification.json')['files'],
            **{p.relative_to(ROOT).as_posix():sha(p) for p in (OLD/'episode_trace.jsonl',OLD/'planner_results.jsonl',OLD/'score_rows.json',OLDPUB/'final_test_results.json')}},
        'cohort':{'patients':63,'visits':92,'outcomes':736,'fold_visits':[21,12,23,19,17],'excluded_incomplete':137,'diagnosis_universe':15,'observed_reference_classes':11},
        'main_dynamic_slots':552,'budgets':list(range(6)),'comparators':['voi_h1','voi_h2','voi_h3','global_eig','information_gain'],
        'primary':['dynamic minus voi_h3 AUBC','dynamic minus voi_h1 AUBC'],'major_budget':5,
        'horizon':'unchanged Dynamic selected=min(requested,actions,3); wrapper effective=min(selected,remaining_budget); zero => no planner',
        'stop':'unchanged Core rootSTOP epsilon1e-12; exact Q/action tie; no forced acquisition',
        'bootstrap':{'repetitions':2000,'seed':42,'unit':'patient cluster','estimator':'patient macro','CI':'linear percentile95','shared_draws':True,'multiple_comparisons':'unadjusted additional development experiment','noninferiority_margin':None},
        'compute':{'full_panel':'logical states per decision from exact actual Core outputs, no C4-31 mixed-policy wall comparison',
            'controlled_ordinals':chosen,'selection':'first2 input-order visits in each fold before new Dynamic results',
            'controlled_slots':480,'passes':['A_no_external_cache','B_per_policy_independent_cache'],
            'repetitions_per_pass':1,'controlled_budgets':list(range(6)),'policies':POLICIES,
            'order':'cyclic rotation by (visit ordinal+budget) mod4, same schedule both passes, no concurrent tests',
            'B_initial_cache':'empty per policy at start of controlled pass; no main/parent/other-policy cache sharing',
            'timing':'Core call and episode timers, controlled trace serialization after timers; internal Core LRU unchanged; external OS workloads NOT_CONTROLLED'},
        'diagnostics':{'states':'root and first actual acquired-outcome posterior in Dynamic budget5; all H1/H2/H3 on same belief/menu; max552 queries',
            'no_new_policy_rollouts':True,'reference':'scorer only after calls','C4_31_fixed_k':'reused read-only, not rerun'},
        'limits':{'maximum_actual_source_episodes':1032,'maximum_actual_core_calls':15000,'maximum_executed_states':50000000,'maximum_wall_seconds':3600,'core_max_states':20000,'core_max_actions':12,'max_horizon':3},
        'failures':'first Source/hash/split/selection/planner/resource mismatch stops; append partial traces; no resume/imputation',
        'decision':'A requires lower full-panel logical states than H3 and positive Visit+Patient AUBC vsH3 with patient CI lower>0. C if H3 request>=90% in states with budget>=3 and actions>=3 AND >=90% same normalized trajectories asH3. D if H1 Visit AUBC>=Dynamic and logical states lower (descriptive, not superiority proof). B if lower states but no A; else E. Flags can coexist. No equivalence claim.',
        'independent_source':'read-only existing inventory/DB, exclude all known fitting/model-selection/evaluation patients; no artifact assigned to new patients and no new holdout scores',
        'LLM_generation':0,'model_fit':0,'new_gold':0,'Core_changes':0,'clinical_readiness':False}
    spec['fingerprint']=digest(spec)
    pub.mkdir(parents=True);pr.mkdir(parents=True)
    write(pr/'protected_hashes.json',pins);write(pr/'source_hashes.json',sources)
    write(pub/'dynamic_source_protocol.json',spec);write(pub/'computation_measurement_protocol.json',spec['compute'])
    write(pub/'source_panel_verification.json',verification)
    write(pub/'frozen_policy_verification.json',{'status':'PASS','version':saved['policy_version'],'policy_fingerprint':expected,'C4_19_spec_identical':True,
        'C4_19_semantics':'selected horizon is uncertainty+action cap; old empty-menu h1 sentinel terminal calculation, this Source wrapper bypasses planner at depth0',
        'additional_cap':'remaining acquisition budget only; selection function unchanged','function_signatures':{
            'DynamicHorizonPolicy.select':str(inspect.signature(DynamicHorizonPolicy.select)),'run_episode':str(inspect.signature(run_episode))},'code_hashes':spec['code_hashes']})
    inventory=[]
    for prefix in ['C4_19','C4_29','C4_30','C4_31']:
        p=next((ROOT/'docs').rglob(prefix+'*.md'));t=p.read_text(encoding='utf-8-sig');inventory.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'lines':len(t.splitlines())})
    for n in ['fixed_k_probe_metrics.json','planner_resource_usage.json','paired_policy_comparison.json','patient_bootstrap_results.json']:read(OLDPUB/n)
    write(pub/'parent_review_inventory.json',inventory)
    print(json.dumps({'prepared':True,'fingerprint':spec['fingerprint'],'policy':expected,'protected':len(pins),'controlled_ordinals':chosen}),flush=True)


def verify(ns):
    pub,pr=places(ns);s=read(pub/'dynamic_source_protocol.json')
    if digest({k:v for k,v in s.items() if k!='fingerprint'})!=s['fingerprint']:raise ValueError('PROTOCOL_CHANGED')
    if check({**s['code_hashes'],**s['input_hashes']})['changed_or_missing']:raise ValueError('CODE_OR_INPUT_CHANGED')
    if DynamicHorizonPolicy().fingerprint!=s['policy_fingerprint']:raise ValueError('POLICY_CHANGED')
    return pub,pr,s


def run(ns):
    pub,pr,s=verify(ns)
    if (pr/'started.json').exists():raise ValueError('NO_RESUME_OR_DUPLICATE_EXECUTION')
    if check(read(pr/'protected_hashes.json'))['changed_or_missing'] or check(read(pr/'source_hashes.json'))['changed_or_missing']:raise ValueError('PRESERVATION_BEFORE_RUN')
    write(pr/'started.json',{'protocol':s['fingerprint'],'started_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'logical_cpus':os.cpu_count(),'external_cpu_workload':'NOT_CONTROLLED','concurrent_tests_by_this_runner':False,'GPU_used':False})
    inputs,identity,truth,models,values,_=load_panel();budget=ComputeBudget();allrows=[];measure=[]
    parenttraces=[r for r in rows(OLD/'episode_trace.jsonl') if r['section']=='native' and r['policy'] in POLICIES[:3]]
    parentplans={r['cache_key']:r for r in rows(OLD/'planner_results.jsonl')}
    costs=[{'scenario_ordinal':r['scenario_ordinal'],'policy':r['policy'],'budget':r['budget'],**parent_plan_resources(r,models[r['fold'],'M5'],parentplans)} for r in parenttraces]
    write(pr/'fixed_full_panel_resources.json',costs)
    try:
        with (pr/'dynamic_trace.jsonl').open('x',encoding='utf-8') as f,(pr/'new_plans.jsonl').open('x',encoding='utf-8') as pf:
            meter=PlannerMeter(cached=True,budget=budget,sink=lambda r:dump_line(pf,r),label='MAIN_DYNAMIC')
            for p in inputs:
                i=p['scenario_ordinal'];m=models[p['fold'],'M5']
                for k in range(6):
                    r=run_episode(m,values[i],'dynamic',k,meter);score_episode(r,truth[i]);r.update(scenario_ordinal=i,fold=p['fold'],stage='main')
                    dump_line(f,r);allrows.append(r)
                print(json.dumps({'main_visit':i,'slots':len(allrows),'executed_states':budget.states}),flush=True)
        write(pr/'main_score_rows.json',[{k:v for k,v in r.items() if k not in ('decisions','releases','belief','terminal_choice')} for r in allrows])
        expected={(r['scenario_ordinal'],r['policy'],r['budget']):r for r in parenttraces+allrows}
        seen={};selected=set(s['compute']['controlled_ordinals'])
        with (pr/'controlled_measurement.jsonl').open('x',encoding='utf-8') as f:
            for stage,cached in [('A_no_external_cache',False),('B_per_policy_independent_cache',True)]:
                meters={p:PlannerMeter(cached=cached,budget=budget,label=stage+':'+p) for p in POLICIES}
                for p in inputs:
                    i=p['scenario_ordinal']
                    if i not in selected:continue
                    for k in range(6):
                        offset=(i+k)%4;order=POLICIES[offset:]+POLICIES[:offset]
                        for policy in order:
                            r=run_episode(models[p['fold'],'M5'],values[i],policy,k,meters[policy]);score_episode(r,truth[i]);r.update(scenario_ordinal=i,fold=p['fold'],stage=stage)
                            key=(i,policy,k)
                            if normalized_trace(r)!=normalized_trace(expected[key]):raise ValueError('REPLAY_MISMATCH_WITH_FIXED_PARENT_OR_DYNAMIC_MAIN')
                            if stage.startswith('A'):seen[key]=normalized_trace(r)
                            elif seen[key]!=normalized_trace(r):raise ValueError('CACHE_CHANGES_TRAJECTORY')
                            dump_line(f,r);measure.append(r)
                    print(json.dumps({'controlled_stage':stage,'visit':i,'completed':len(measure),'executed_states':budget.states}),flush=True)
        write(pub/'execution_status.json',{'status':'C4_32_DYNAMIC_SOURCE_EVALUATION_COMPLETE','main_slots':len(allrows),'controlled_episodes':len(measure),'actual_source_episodes':len(allrows)+len(measure),
            'generation':0,'fitting':0,'failures':0,'actual_core_calls':budget.calls,'actual_executed_states':budget.states,'wall_seconds':time.perf_counter()-budget.start})
    except Exception as e:
        write(pr/'failure.json',{'error_type':type(e).__name__,'code':str(e),'main_completed':len(allrows),'controlled_completed':len(measure)})
        write(pub/'execution_status.json',{'status':'F_EXECUTION_BLOCKED','main_completed':len(allrows),'controlled_completed':len(measure),'error_type':type(e).__name__})
        raise


def curve(rs,identity):
    return {str(k):mean_metrics([r for r in rs if r['budget']==k],identity) for k in range(6)}


def analyze(ns):
    pub,pr,s=verify(ns)
    if read(pub/'execution_status.json')['failures']:raise ValueError('PARTIAL_CANNOT_CLAIM_COMPLETE')
    inputs,identity,truth,models,values,_=load_panel();dynamic=rows(pr/'dynamic_trace.jsonl');old=rows(OLD/'episode_trace.jsonl')
    old=[r for r in old if r['section']=='native' and r['policy'] in s['comparators']]
    metrics={'dynamic':curve(dynamic,identity)}
    for policy in s['comparators']:metrics[policy]=curve([r for r in old if r['policy']==policy],identity)
    write(pub/'dynamic_native_policy_metrics.json',metrics['dynamic']);write(pub/'fixed_policy_comparison.json',metrics)
    pairs={}
    for policy in s['comparators']:
        rr=dynamic+[r for r in old if r['policy']==policy]
        pairs['dynamic_minus_'+policy]={'AUBC':paired_comparison(aubc_rows(rr,'dynamic'),aubc_rows(rr,policy),identity),
            'budget5':paired_comparison([r for r in dynamic if r['budget']==5],[r for r in old if r['policy']==policy and r['budget']==5],identity)}
    write(pub/'patient_paired_comparison.json',pairs)
    write(pub/'patient_bootstrap_results.json',{k:{n:v[n]['patient_macro'] for n in ('AUBC','budget5')} for k,v in pairs.items()})
    requests=Counter();selected=Counter();effective=Counter();joint=Counter();sufficient=Counter();cap=Counter();features=[]
    for r in dynamic:
        if not r['decisions']:cap['budget_zero_terminal']+=1
        for d in r['decisions']:
            h=d['horizon_selection'];requests[str(h['requested_horizon'])]+=1;selected[str(h['selected_horizon'])]+=1;effective[str(h['effective_horizon'])]+=1
            joint['%s/%s/%s'%(h['requested_horizon'],h['selected_horizon'],h['effective_horizon'])]+=1
            cap['action_cap']+=h['action_cap'];cap['budget_cap']+=h['budget_cap'];cap['fallback']+=bool(h['selection']['fallback_reason'])
            if h['remaining_actions']>=3 and h['remaining_budget']>=3:sufficient[str(h['requested_horizon'])]+=1
            features.append({k:h[k] for k in ('normalized_entropy','margin','entropy_horizon','margin_horizon','requested_horizon','selected_horizon','effective_horizon')})
    write(pub/'horizon_request_distribution.json',{'requests':dict(requests),'selected_after_action_cap':dict(selected),'total_decisions':sum(requests.values()),'features_summary':{k:{'min':min(x[k] for x in features),'max':max(x[k] for x in features),'mean':sum(x[k] for x in features)/len(features)} for k in ('normalized_entropy','margin')}})
    write(pub/'horizon_effective_distribution.json',{'effective':dict(effective),'requested_selected_effective_joint':dict(joint),'caps_and_terminal':dict(cap),'interpretation':'Budget-capped H1 != uncertainty-selected H1; terminal Budget0 invokes no planner'})
    oldidx={(r['scenario_ordinal'],r['policy'],r['budget']):r for r in old};matches=sum(normalized_trace(r)==normalized_trace(oldidx[r['scenario_ordinal'],'voi_h3',r['budget']]) for r in dynamic)
    saturation={'requests_H3_rate':requests['3']/sum(requests.values()),'sufficient_actions_and_budget_request_counts':dict(sufficient),
        'sufficient_H3_rate':sufficient['3']/sum(sufficient.values()) if sufficient else None,'same_normalized_trajectory_vsH3':matches,'registered_episodes':552,
        'C4_19':'all232 nonempty historical replay states requested H3; now different M5 source-policy cohort, descriptive contrast only',
        'selection_policy_fingerprint':s['policy_fingerprint']}
    write(pub/'horizon_saturation_analysis.json',saturation)
    controlled=rows(pr/'controlled_measurement.jsonl');costs=read(pr/'fixed_full_panel_resources.json');resources={}
    for policy in POLICIES:
        if policy=='dynamic':full={'calls':sum(r['resources']['calls'] for r in dynamic),'logical_states':sum(r['resources']['logical_states'] for r in dynamic),'maximum_states':max(r['resources']['maximum_states'] for r in dynamic)}
        else:
            cs=[r for r in costs if r['policy']==policy];full={'calls':sum(r['calls'] for r in cs),'logical_states':sum(r['logical_states'] for r in cs),'maximum_states':max(r['maximum_states'] for r in cs)}
        resources[policy]={'full_panel':{**full,'registered_slots':552,'mean_states_per_slot':full['logical_states']/552,'mean_states_per_visit_all6budgets':full['logical_states']/92,
            'wall_comparison':'NOT_AVAILABLE_FULL_PANEL; fixed counts reuse sealed per-input Core results, no mixed C4-31 timing comparison'},
            'controlled':{stage:resource_summary([r for r in controlled if r['policy']==policy and r['stage']==stage]) for stage in ('A_no_external_cache','B_per_policy_independent_cache')}}
    write(pub/'policy_resource_comparison.json',{'scope':'full-panel identical logical-state definition + independently executed matched subset A/B; no cross-policy caches',
        'policies':resources,'controlled_visits':10,'controlled_slots_per_policy_per_pass':60,'Core_internal_cache':'unchanged; hits not exposed','wall_caveat':'one repeat per pass; external CPU/OS workload not isolated; controlled serialization after timer; do not claim clean speedup'})
    trade={}
    for policy in POLICIES:
        m=metrics[policy];trade[policy]={'AUBC':sum(m[str(k)]['visit_macro']['top1'] for k in range(1,6))/5,'budget5':m['5']['visit_macro'],**resources[policy]['full_panel']}
    ratio=1-trade['dynamic']['logical_states']/trade['voi_h3']['logical_states'];extra=trade['dynamic']['logical_states']-trade['voi_h1']['logical_states']
    flags=[]
    primary=pairs['dynamic_minus_voi_h3']['AUBC'];g=primary['visit_macro']['top1']>0 and primary['patient_macro']['top1']['ci95'][0]>0
    if ratio>0 and g:flags.append('A_DYNAMIC_DEVELOPMENT_GAIN')
    if saturation['sufficient_H3_rate']>=.9 and matches/552>=.9:flags.append('C_DYNAMIC_H3_SATURATION')
    if trade['voi_h1']['AUBC']>=trade['dynamic']['AUBC'] and trade['voi_h1']['logical_states']<trade['dynamic']['logical_states']:flags.append('D_FIXED_H1_FAVORED')
    if ratio>0 and not g:flags.append('B_DYNAMIC_COMPUTATION_ONLY')
    if not flags:flags.append('E_MIXED_OR_INCONCLUSIVE')
    write(pub/'accuracy_computation_tradeoff.json',{'policies':trade,'logical_state_reduction_vsH3':ratio,'additional_states_vsH1':extra,'decision_flags':flags,
        'Pareto':'descriptive point-estimate AUBC maximize / identical-scope logical states minimize; no new weighted Utility','clinical_readiness':False,'equivalence':'NOT_TESTED; no margin'})
    write(pub/'stop_and_budget_analysis.json',{p:{str(k):dict(Counter(r['stop_reason'] for r in dynamic+old if r['policy']==p and r['budget']==k)) for k in range(6)} for p in metrics})
    labels=sorted(set(truth.values()));sub=[]
    for n,label in enumerate(labels,1):
        ids={i for i,d in truth.items() if d==label};sub.append({'diagnosis_ordinal':n,'visits':len(ids),
            'metrics':{p:mean_metrics([r for r in dynamic+old if r['policy']==p and r['budget']==5 and r['scenario_ordinal'] in ids],identity) for p in metrics}})
    write(pub/'diagnosis_subgroup_analysis.json',{'budget':5,'groups':sub,'diagnosis_names':'PRIVATE_ONLY','clinical_reference':'NOT_REVIEWED'})
    limitations(pub,pr,dynamic,old,models,truth)
    independent_sources(pub,pr,models,identity)
    with (pub/'next_research_decision.md').open('x',encoding='utf-8') as f:
        f.write('# C4-32 decision\n\n'+', '.join(flags)+'\n\n'+json.dumps({'state_reduction_vsH3':ratio,'extra_states_vsH1':extra,'saturation':saturation,'paired_primary':primary},indent=2)+
            '\n\nDevelopment Source evidence only. Preserve current frozen policy; do not retune thresholds on these92 visits. No clinical superiority/equivalence/official B4 claim. Independent reference and prospective artifact assignment remain required.\n')
    plots(pub,metrics,resources,trade,requests,effective)
    print(json.dumps({'analyzed':True,'flags':flags,'state_reduction_vsH3':ratio,'extra_states_vsH1':extra}),flush=True)


def limitations(pub,pr,dynamic,old,models,truth):
    oldidx={(r['scenario_ordinal'],r['policy'],r['budget']):r for r in old};comparisons={}
    for p in ['voi_h2','voi_h3','dynamic']:
        cs=Counter()
        for r in dynamic if p=='dynamic' else [r for r in old if r['policy']==p]:
            b=oldidx[r['scenario_ordinal'],'voi_h1',r['budget']];x=r['score']['top1'];y=b['score']['top1']
            cs['left_only_correct']+=int(x and not y);cs['H1_only_correct']+=int(y and not x)
            cs['first_action_different']+=int([d['selected_action'] for d in r['decisions'][:1]]!=[d['selected_action'] for d in b['decisions'][:1]])
        comparisons[p]=dict(cs)
    update=Counter();support=Counter()
    for r in dynamic:
        m=models[r['fold'],'M5']
        for d in r['decisions']:
            if d['selected_action'] is None:continue
            update['acquired']+=1;update['reference_probability_decreased']+=d['reference_probability_delta']<0
            before=sorted(d['belief_before'],key=lambda k:(-d['belief_before'][k],k))[0];after=sorted(d['belief_after'],key=lambda k:(-d['belief_after'][k],k))[0]
            update['one_step_true_to_nonmatching']+=before==truth[r['scenario_ordinal']] and after!=truth[r['scenario_ordinal']]
            counts=m['provenance']['empirical_support'][d['selected_action']]
            support['any_diagnosis_zero_support_acquisitions']+=any(n==0 for n in counts.values());support['reference_class_zero_support_acquisitions']+=counts[truth[r['scenario_ordinal']]]==0
    # Same actual root and first released-outcome belief/menu across H1/H2/H3.
    execution=read(pub/'execution_status.json')
    diagnostic_budget=ComputeBudget(max_calls=15000-execution['actual_core_calls'],max_states=50000000-execution['actual_executed_states'],max_seconds=max(0,3600-execution['wall_seconds']))
    shared=[];meter=PlannerMeter(cached=True,budget=diagnostic_budget);calls=0
    for r in dynamic:
        if r['budget']!=5:continue
        m=models[r['fold'],'M5'];states=[(r['decisions'][0]['belief_before'],[a['id'] for a in m['actions']])]
        first=next((d for d in r['decisions'] if 'belief_after' in d),None)
        if first:states.append((first['belief_after'],[a['id'] for a in m['actions'] if a['id']!=first['selected_action']]))
        for ordinal,(belief,remaining) in enumerate(states):
            actions=[a for a in m['actions'] if a['id'] in remaining];ps={}
            for h in (1,2,3):ps[str(h)]=meter(belief,actions,m['terminal_utility'],horizon=h);calls+=1
            shared.append({'scenario_ordinal':r['scenario_ordinal'],'state_ordinal':ordinal,'belief':belief,'remaining':remaining,'plans':ps})
    write(pr/'same_belief_horizon_diagnostics.json',shared)
    rank=Counter()
    for r in shared:
        for h in ('2','3'):
            rank['H'+h+'_vsH1_states']+=1;rank['H'+h+'_vsH1_selected_diff']+=r['plans'][h]['selected_action_id']!=r['plans']['1']['selected_action_id']
            rank['H'+h+'_vsH1_ranking_diff'] += [v['action_id'] for v in r['plans'][h]['action_values']]!=[v['action_id'] for v in r['plans']['1']['action_values']]
    write(pub/'long_horizon_limitation_analysis.json',{'paired_vsH1_all6budgets':comparisons,'dynamic_observation_changes':dict(update),'dynamic_support':dict(support),
        'same_belief_ranking':dict(rank),'diagnostic_queries':calls,'diagnostic_actual_core_calls':meter.budget.calls,'diagnostic_states':meter.budget.states,
        'combined_actual_core_calls':execution['actual_core_calls']+meter.budget.calls,'combined_executed_states':execution['actual_executed_states']+meter.budget.states,
        'C4_31_H3_budget5':{'native_correct':40,'fixed_k_correct':40,'denominator':92},'native_vs_fixed_k':read(OLDPUB/'fixed_k_probe_metrics.json'),
        'causality':'descriptive association only; STOP alone does not explain H3 limitations; multistep Q not one-step calibration error'})


def independent_sources(pub,pr,models,identity):
    import csv
    from app.locdi.voi_source_action_inventory import database_sources,unique_reference,project
    from app.locdi.reference_readiness import COHORT
    from app.locdi.source_replay import PRIVATE as D3
    base=read(SOURCE_PARENT/'base_visits.json');additional=read(SOURCE_PARENT/'additional_visits.json')
    oldpatients={v['patient_id'] for v in base};selection={v['patient_id'] for v in additional}|oldpatients
    fitted=set()
    for p in SOURCE_PARENT.glob('fold??_M?_model.json'):
        m=read(p);fitted.update(m.get('training_patients',m.get('provenance',{}).get('training_patients',[])))
    hold={v['patient_id'] for v in rows(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')};d3={v['patient_id'] for v in rows(D3/'cases.jsonl')};current={r['patient_id'] for r in identity.values()}
    forbidden=fitted|selection|hold|d3|current;allids=set();candidateids=set();byclass=defaultdict(set);counts=Counter();private=[];complete=Counter()
    universe=list(models[1,'M5']['prior'])
    for source in database_sources():
        pid=source['patient_id'];allids.add(pid)
        if pid in forbidden:counts['excluded_known_fitting_selection_evaluation_visits']+=1;continue
        candidateids.add(pid);counts['not_in_known_use_visits']+=1
        label=unique_reference(source,universe)
        if label is None:counts['no_single_source_reference_in_universe']+=1;continue
        byclass[label].add(pid);v=project(source);counts['single_source_reference_visits']+=1
        if not v['values']:counts['single_reference_but_no_projected_observation']+=1
        per_fold={str(f):set(models[f,'M5']['provenance']['action_key_mapping'])<=set(v['values']) for f in range(1,6)}
        for f,ok in per_fold.items():complete[f]+=ok
        private.append({'patient_id':pid,'visit_id':v['visit_id'],'reference':label,'per_fold_complete':per_fold,'source_keys':list(v['values']),'availability_basis':'RECORDED_TIME_PROXY'})
    with (COHORT/'csv/patients.csv').open(encoding='utf-8-sig',newline='') as f:rawids={r['Id'] for r in csv.DictReader(f)}
    write(pr/'independent_source_candidates.json',private)
    write(pub/'independent_source_feasibility.json',{'status':'NOT_AVAILABLE_FOR_INDEPENDENT_EVALUATION','raw_patients':len(rawids),'indexed_patients':len(allids),
        'C4_29_CV_patients':len(oldpatients),'C4_30_any_fitting_patients':len(fitted),'C4_30_source_selection_patients':len(selection),
        'C4_31_patients':len(current),'Original50_patients':len(hold),'D3_patients':len(d3),'known_exclusion_union':len(forbidden),
        'raw_not_in_known_use':len(rawids-forbidden),'indexed_not_in_known_use':len(candidateids),'visit_counts':dict(counts),
        'single_reference_patient_count':len(set().union(*byclass.values())) if byclass else 0,
        'reference_class_patient_support_by_ordinal':{str(i+1):len(byclass[d]) for i,d in enumerate(sorted(universe))},'complete_visits_per_frozen_fold_menu':dict(complete),
        'earlier_source_inventory':read(ROOT/'outputs/research_evaluation/c4_30_v2/source_expansion_inventory.json'),
        'exposure':'All indexed contexts were already inventoried for C4-30 model/data selection; not-in-known-fitting is not never-exposed. Raw-only patient clinical contents NOT_REVIEWED here.',
        'artifact_assignment':'NOT_SELECTED; fold-specific M5 models/menus cannot be assigned ad hoc to new patients','clinical_reference':'UNREVIEWED_SOURCE_CONDITION',
        'availability':'RECORDED_TIME_PROXY; actual clinical release not established','new_holdout_scores':0,'new_fits':0,
        'next':'external/prospectively quarantined source and reviewed references, frozen final artifact assignment or separately approved final-fit protocol required'})


def plots(pub,metrics,resources,trade,requests,effective):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for p in metrics:ax.plot(range(6),[metrics[p][str(k)]['visit_macro']['top1'] for k in range(6)],marker='o',label=p)
    ax.set(xlabel='Maximum acquisitions',ylabel='Top1 Source agreement',title='Frozen Dynamic vs fixed policies | 92 development visits');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(pub/'dynamic_budget_accuracy_curve.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.bar(POLICIES,[resources[p]['full_panel']['logical_states'] for p in POLICIES]);ax.set(ylabel='Total logical evaluated_states',title='Identical 552-slot scope | cache-free-equivalent work');fig.tight_layout();fig.savefig(pub/'policy_computation_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));xs=list(range(1,4));ax.bar([x-.18 for x in xs],[requests[str(x)] for x in xs],width=.36,label='uncertainty requested');ax.bar([x+.18 for x in xs],[effective[str(x)] for x in xs],width=.36,label='executed after caps');ax.set(xticks=xs,xlabel='Horizon',ylabel='Decision count',title='Requested vs effective (Budget cap distinguished)');ax.legend();fig.tight_layout();fig.savefig(pub/'horizon_selection_distribution.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for p in POLICIES:
        x=trade[p]['mean_states_per_slot'];y=trade[p]['AUBC'];ax.scatter([x],[y]);ax.annotate(p,(x,y),xytext=(5,7),textcoords='offset points')
    ax.set(xlabel='Mean logical evaluated_states per registered slot',ylabel='Visit AUBC Top1',title='Accuracy-computation point estimates | not equivalence evidence');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(pub/'accuracy_computation_pareto.png',dpi=160);plt.close(fig)


def regression(ns):
    import unittest
    pub,pr,s=verify(ns);patterns=read(OLDPUB/'final_test_results.json')['patterns'];suite=unittest.TestSuite();counts={}
    for pattern in list(patterns)+['test_dynamic_horizon_source_evaluation.py']:
        if pattern.endswith('.py'):ss=unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'),pattern=pattern)
        else:ss=unittest.defaultTestLoader.loadTestsFromName(pattern)
        counts[pattern]=ss.countTestCases();suite.addTests(ss)
    start=time.perf_counter()
    with (pr/'regression.log').open('x',encoding='utf-8') as f:
        with patch.object(socket.socket,'connect',side_effect=RuntimeError('NETWORK_DISABLED')),patch.object(socket.socket,'bind',side_effect=RuntimeError('NETWORK_DISABLED')):result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    out={'total':result.testsRun,'passed':result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped),
        'seconds':time.perf_counter()-start,'patterns':counts,'scope':'RELATED_REGRESSION; full repository suite NOT_RUN','network':'DISABLED','new_source_model_fits':0,'LLM_generation':0}
    write(pub/'test_results.json',out);print(json.dumps(out),flush=True)
    if not result.wasSuccessful():raise ValueError('REGRESSION_FAILED')


def seal(ns):
    pub,pr,s=verify(ns);protected=check(read(pr/'protected_hashes.json'));sources=check(read(pr/'source_hashes.json'))
    if protected['changed_or_missing'] or sources['changed_or_missing']:raise ValueError('PRESERVATION_FAILED')
    text='\n'.join(p.read_text(encoding='utf-8') for p in pub.glob('*.json'))
    ids=read(PREP/'identity_membership.json');refs=read(PREP/'scoring_keys.json');secrets={r[k] for r in ids for k in ('patient_id','visit_id')}|{r['source_diagnosis_reference'] for r in refs}
    if any(v in text for v in secrets):raise ValueError('PUBLIC_LEAKAGE')
    tests=read(pub/'test_results.json')
    if tests['errors'] or tests['failures']:raise ValueError('TESTS_FAILED')
    write(pub/'verification.json',{'status':'PASS','protected':protected,'sources':sources,'code':check(s['code_hashes']),
        'policy_fingerprint_unchanged':True,'M5_unchanged':True,'Source_unchanged':True,'Core_changes':0,'threshold_changes':0,'new_probability_fits':0,'LLM_generation':0,'new_gold':0,
        'new_source_outcomes':0,'official_B0_B4':'UNCHANGED_NOT_RUN','manuscript_changed':False,'git_commit_push':False,'clinical_readiness':False,
        'public_exact_private_identifier_reference_scan':'PASS','D7':'historical code-version mismatch not retroactively updated'})
    with (pub/'actual_execution_command.md').open('x',encoding='utf-8') as f:
        f.write('# Actual offline execution\n\n```powershell\n'+'\n'.join('python -B scripts/evaluate_dynamic_horizon_source_policy.py '+stage+' --namespace '+ns for stage in ['prepare','run','analyze','regression','seal'])+'\n```\n\nNo concurrent tests during controlled measurement. Seal follows report writing. Completed namespace rejects resume/overwrite; reproduction requires a fresh namespace and identical frozen inputs/code.\n')
    index={'files':{p.relative_to(ROOT).as_posix():sha(p) for p in pr.rglob('*') if p.is_file()}};index['fingerprint']=digest(index['files']);write(pr/'private_index.json',index)
    report=next((ROOT/'docs').rglob('C4_32_DYNAMIC_HORIZON_SOURCE_EVALUATION_REPORT.md'));files=[p for p in pub.iterdir() if p.is_file()]+[pr/'private_index.json',report]+[ROOT/n for n in CODE]
    manifest={'status':read(pub/'execution_status.json')['status'],'files':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'protocol_fingerprint':s['fingerprint']};manifest['fingerprint']=digest(manifest['files']);write(pub/'completion_manifest.json',manifest)
    print(json.dumps({'sealed':True,'protected':protected['checked'],'sources':sources['checked']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run','analyze','regression','seal']);parser.add_argument('--namespace',default='c4_32_v1');a=parser.parse_args()
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('NETWORK_DISABLED_C4_32')):globals()[a.stage](a.namespace)
