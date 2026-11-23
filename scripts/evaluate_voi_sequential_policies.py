"""C4-31: preregister, run frozen Source replay, analyze, test, seal. No network/fits."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read, rows, sha, write, check
from scripts.evaluate_voi_information_improvement import baseline_view
from app.core.information_outcomes import InformationOutcomes
from app.core.diagnostic_belief import posterior
from app.locdi.voi_information_improvement import bucket
from app.locdi.voi_sequential_policy_evaluation import *

PARENT=ROOT/'outputs/private/research_evaluation/c4_30_v2'
PUBLIC_PARENT=ROOT/'outputs/research_evaluation/c4_30_v2'
OLD=ROOT/'outputs/private/research_evaluation/c4_29_v1'
PREP=PARENT/'c4_31_preparation'
CODE=['app/locdi/voi_source_replay_environment.py','app/locdi/voi_sequential_policy_evaluation.py',
      'scripts/evaluate_voi_sequential_policies.py','tests/locdi/test_voi_sequential_policy_evaluation.py']
NS='c4_31_v1'


def places(namespace):
    if not namespace or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in namespace):raise ValueError('NAMESPACE')
    return ROOT/'outputs/research_evaluation'/namespace, ROOT/'outputs/private/research_evaluation'/namespace


def dump_line(f,row):
    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    f.flush()


def load_panel():
    inputs=read(PREP/'policy_inputs.json');identities=read(PREP/'identity_membership.json')
    outcomes=read(PREP/'source_outcomes.json');scoring=read(PREP/'scoring_keys.json')
    expected=set(range(1,93))
    for rs in (inputs,identities,outcomes,scoring):
        if len(rs)!=92 or {r['scenario_ordinal'] for r in rs}!=expected:raise ValueError('PANEL_92_ORDINALS')
    identity={r['scenario_ordinal']:r for r in identities};truth={r['scenario_ordinal']:r['source_diagnosis_reference'] for r in scoring}
    source={r['scenario_ordinal']:r['outcomes'] for r in outcomes}
    visits={(v['patient_id'],v['visit_id']):v for v in read(PARENT/'base_visits.json')}
    split=read(OLD/'patient_grouped_split_manifest.json');keys=read(PARENT/'action_keys.json')
    models={};values={};checks=Counter();foldcounts=Counter()
    for f in range(1,6):
        for m in ('M4','M5'):
            models[f,m]=InformationOutcomes(read(PARENT/('fold%02d_%s_model.json'%(f,m)))).artifact
        original=read(OLD/('fold%02d_probability_artifact.json'%f))
        models[f,'M1']=baseline_view(original,keys['base'],keys['legacy_mapping'])
        if models[f,'M1']!=read(PARENT/('fold%02d_M1_model.json'%f)):raise ValueError('M1_READONLY_ALIAS_CHANGED')
    for p in inputs:
        i=p['scenario_ordinal'];ident=identity[i];f=p['fold'];foldcounts[f]+=1
        v=visits[ident['patient_id'],ident['visit_id']];values[i]=deepcopy(v['values'])
        if split[ident['patient_id']]!=f-1 or ident['fold']!=f:raise ValueError('PATIENT_FOLD')
        if v['diagnosis']!=truth[i]:raise ValueError('REFERENCE_BINDING')
        model=models[f,'M5'];mapping=model['provenance']['action_key_mapping']
        if p['initial_prior']!=model['prior'] or p['information_action_ids']!=[a['id'] for a in model['actions']]:raise ValueError('POLICY_MODEL_BINDING')
        if p['fixed_action_order']!=canonical_order(model):raise ValueError('FIXED_ORDER')
        if sha(PARENT/p['artifact_file'])!=p['artifact_sha256']:raise ValueError('PREP_ARTIFACT_HASH')
        if len(source[i])!=8 or len({e['action_id'] for e in source[i]})!=8:raise ValueError('SOURCE8')
        for m in ('M1','M4','M5'):
            if ident['patient_id'] in models[f,m]['provenance']['training_patients']:raise ValueError('TRAIN_VALIDATION_LEAKAGE')
        for e in source[i]:
            k=next(k for k,a in mapping.items() if a==e['action_id']);val=values[i][k]
            if (bucket(val['value'],model['provenance']['binning_thresholds'][k])!=e['outcome_id'] or
                val['value']!=e['actual_source_value'] or val['event_id']!=e['source_event_id'] or
                val['source_file']!=e['source_file'] or val['source_row']!=e['source_row'] or
                val['source_record_fingerprint']!=e['source_record_fingerprint'] or
                val['available_minute']!=e['recorded_available_minute']):raise ValueError('SOURCE_OUTCOME_BINDING')
            if digest(val['source_record'])!=val['source_record_fingerprint']:raise ValueError('SOURCE_RECORD_HASH')
            if val['source_record']['PATIENT']!=ident['patient_id'] or val['source_record']['ENCOUNTER']!=ident['visit_id']:raise ValueError('SOURCE_IDENTITY')
            if val['availability_basis']!='RECORDED_TIME_PROXY':raise ValueError('PROXY_BINDING')
            checks['verified_source_outcomes']+=1
    if sorted(foldcounts.values())!=sorted([21,12,23,19,17]) or len({r['patient_id'] for r in identities})!=63:raise ValueError('COHORT_COUNTS')
    return inputs,identity,truth,models,values,{'status':'PASS','visits':92,'patients':63,'fold_visits':dict(foldcounts),**checks,
        'excluded_incomplete_visits':137,'total_original_visits':229,'actual_availability_review':'NOT_APPROVED',
        'quantiles_reestimated':False,'actual_source_values_sampled':False,'patient_group_split_matches':True}


def prepare(namespace):
    pub,pr=places(namespace)
    if pub.exists() or pr.exists():raise ValueError('NAMESPACE_ALREADY_USED_NO_RESUME')
    inherited=read(PARENT/'protected_hashes.json')
    for mp in (PUBLIC_PARENT/'completion_manifest.json',PARENT/'private_index.json'):
        doc=read(mp)
        if digest(doc['files'])!=doc['fingerprint']:raise ValueError('PARENT_MANIFEST')
        inherited.update(doc['files']);inherited[mp.relative_to(ROOT).as_posix()]=sha(mp)
    # Pin every current Core and existing reports in addition to inherited chains.
    for folder in ('app/core','docs','configs','prompts','experiments/configs'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:inherited[p.relative_to(ROOT).as_posix()]=sha(p)
    if check(inherited)['changed_or_missing']:raise ValueError('INHERITED_HASH_MISMATCH')
    sourcepins={}
    for n in ('source_hashes.json',):sourcepins.update(read(PARENT/n))
    sourcepins.update(read(OLD/'source_hashes.json'))
    if check(sourcepins)['changed_or_missing']:raise ValueError('SOURCE_HASH_MISMATCH')
    inputs,identity,truth,models,values,panel=load_panel()
    pub.mkdir(parents=True);pr.mkdir(parents=True)
    write(pr/'protected_hashes.json',inherited);write(pr/'source_hashes.json',sourcepins)
    docs=[]
    for prefix in ('C4_18','C4_19','C4_29','C4_30'):
        p=next((ROOT/'docs').rglob(prefix+'*.md'));text=p.read_text(encoding='utf-8-sig')
        docs.append({'file':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'lines':len(text.splitlines())})
    for n in ('information_improvement_protocol.json','grouped_model_comparison.json','fold_model_manifest.json',
              'development_voi_artifact_readiness.json','sequential_policy_evaluation_feasibility.json','c4_31_policy_evaluation_protocol.json'):
        read(PUBLIC_PARENT/n)
    (PUBLIC_PARENT/'c4_31_policy_evaluation_protocol.md').read_text(encoding='utf-8-sig')
    write(pub/'parent_review_inventory.json',docs)
    write(pub/'source_panel_verification.json',panel)
    # Eligibility of subset enumeration checked without scoring/reference access.
    from itertools import combinations
    from app.core.diagnostic_belief import terminal_choice
    mismatch=0;count=0;max_delta=0
    for p in inputs:
        m=models[p['fold'],'M5'];lookup={a['id']:a for a in m['actions']}
        lik={aid:next(o['likelihood'] for o in lookup[aid]['outcomes'] if o['id']==bucket(values[p['scenario_ordinal']][k]['value'],m['provenance']['binning_thresholds'][k])) for k,aid in m['provenance']['action_key_mapping'].items()}
        for k in range(6):
            for subset in combinations(sorted(lookup),k):
                a=dict(m['prior']);b=dict(a)
                for aid in subset:a=posterior(a,lik[aid])[0]
                for aid in reversed(subset):b=posterior(b,lik[aid])[0]
                max_delta=max(max_delta,max(abs(a[d]-b[d]) for d in a));count+=1
                mismatch+=terminal_choice(a,m['terminal_utility'])[0]!=terminal_choice(b,m['terminal_utility'])[0]
    write(pub/'random_order_invariance_precheck.json',{'subset_pairs':count,'reverse_order_terminal_disagreements':mismatch,'maximum_probability_delta':max_delta,
        'exact_subset_expectation':'NOT_USED; floating-point terminal-tie order sensitivity','reference_used':False,'fallback':'32 fixed seed-derived permutations per visit and budget'})
    pins={str(p.relative_to(ROOT)):sha(p) for p in PREP.glob('*.json')}
    for f in range(1,6):
        for m in ('M4','M5','M1'):
            p=PARENT/('fold%02d_%s_model.json'%(f,m));pins[str(p.relative_to(ROOT))]=sha(p)
        p=OLD/('fold%02d_probability_artifact.json'%f);pins[str(p.relative_to(ROOT))]=sha(p)
    for p in (OLD/'patient_grouped_split_manifest.json',PARENT/'base_visits.json',PUBLIC_PARENT/'information_improvement_protocol.json',PUBLIC_PARENT/'c4_31_policy_evaluation_protocol.json'):
        pins[str(p.relative_to(ROOT))]=sha(p)
    core=['app/core/diagnostic_belief.py','app/core/information_outcomes.py','app/core/information_value.py','app/core/voi_planning.py','app/core/voi_evaluation.py','app/locdi/voi_information_improvement.py','app/locdi/dynamic_horizon.py']
    spec={'protocol':VERSION,'namespace':namespace,'prepared_at':datetime.now(timezone.utc).isoformat(),
        'parent_protocol_fingerprint':read(PUBLIC_PARENT/'information_improvement_protocol.json')['fingerprint'],
        'input_hashes':pins,'code_hashes':{n:sha(ROOT/n) for n in CODE+core},
        'patients':63,'visits':92,'source_outcomes':736,'excluded_original_incomplete':137,
        'model':'M5 frozen ternary8, exact serialized boundaries; no fitting','policies':POLICIES,'budgets':list(range(6)),
        'horizon':'min(configured,remaining_budget,remaining_feasible_actions); zero=>terminal',
        'native_stop':'Core VoI STOP bestQ > T+1e-12, Q ties action_id; IG existing selector with Diagnose tie preference; random/fixed/global exhaust cap; prior immediate',
        'fixed':'canonical Source key lexical order from C4-30 preparation',
        'global_eig':'frozen fold Train prior/likelihood H1 EIG descending, tie action_id; no reordering after observations',
        'random':{'method':'FIXED_32_SEED_REPETITIONS','seed':42,'repetitions':32,'seed_use':'first16hex digest({seed42,scenario_ordinal,repeat0..31}) integer',
                  'order':'Random(seed).shuffle(sorted action IDs), same registered seeds but fresh state for every budget','exact_precheck':'20148 reversed subset pairs:19 terminal-tie changes at <=5.56e-16; finite fallback fixed before scoring',
                  'metrics':'average32 scores, not mean posterior; fractional discordance; cluster CI conditional on these fixed paths; MC error separately reported'},
        'primary':'native M5 voi_h3 minus global_eig; visit-macro Top1 AUBC = mean budgets1..5',
        'primary_inference':'patient-macro AUBC paired bootstrap; gain only if visit and patient deltas positive AND patient CI lower>0; developmental unadjusted',
        'major_budget':5,'secondary':SCORE_KEYS+['actual_actions','stop','budget_exhausted'],
        'pairs':PAIRS,'bootstrap':{'repetitions':2000,'seed':42,'unit':'patient cluster','estimator':'patient macro','CI':'percentile95 linear interpolation','shared_draws':'same sorted patient set and seed for every comparison','multiplicity':'unadjusted exploratory'},
        'fixed_k_probe':{'policies':['voi_h1','voi_h2','voi_h3'],'budgets':list(range(6)),'visits':92,'slots':1656,'rule':'read existing exposed Q ranking; ignore rootSTOP ONLY in separately labelled probe'},
        'auxiliary':{'M1':['fixed','voi_h3'],'M4':['global_eig','voi_h3'],'same_visits':92,'budgets':list(range(6)),'slots':2208,'no_fitting':True},
        'limits':{'main_slots':4416,'all_slots':8280,'maximum_environment_paths':25392,'maximum_planner_cache_misses':20000,'maximum_expanded_states':50000000,
                  'max_wall_seconds':3600,'core_max_states':20000,'core_max_actions':12,'maximum_horizon':3},
        'failures':'preserve append-only partial traces; stop on first source/integrity/planner/resource failure; no imputation/resume',
        'research_decision':'A primary CI-positive; C H2/H3 vsH1 AUBC CI-positive; D no H3/global gain but H3>random or canonical fixed; otherwise E; F partial. B descriptive fewer actions only, never equivalence without margin.',
        'clinical_readiness':False,'utility':'unchanged dimensionless validation1/0','dynamic_horizon_changed':False,
        'LLM_generation':0,'new_probability_fits':0,'routing_fit':0,'new_gold':0}
    spec['fingerprint']=digest(spec);write(pub/'sequential_policy_protocol.json',spec)
    write(pub/'policy_registry.json',{'policies':POLICIES,'native_stop':spec['native_stop'],'fixed':spec['fixed'],'global_eig':spec['global_eig'],'random':spec['random']})
    write(pub/'evaluation_budget_manifest.json',spec['limits'])
    write(pub/'frozen_artifact_verification.json',{'status':'PASS','files':pins,'M1':'read-only aliases identical to C4-30 views; original C4-29 prior/likelihood unchanged',
        'serialized_boundaries_reused':True,'clinical_support':'NOT_APPROVED; zero-support smoothed cells retained'})
    print(json.dumps({'prepare':'PASS','protected':len(inherited),'source':len(sourcepins),'protocol':spec['fingerprint']}),flush=True)


def verify_protocol(pub):
    spec=read(pub/'sequential_policy_protocol.json')
    if digest({k:v for k,v in spec.items() if k!='fingerprint'})!=spec['fingerprint']:raise ValueError('PROTOCOL_HASH')
    if check({**spec['code_hashes'],**spec['input_hashes']})['changed_or_missing']:raise ValueError('CODE_OR_INPUT_CHANGED')
    return spec


def run(namespace):
    pub,pr=places(namespace);spec=verify_protocol(pub)
    if (pr/'started.json').exists():raise ValueError('NO_RESUME_OR_DUPLICATE_RUN')
    if check(read(pr/'protected_hashes.json'))['changed_or_missing'] or check(read(pr/'source_hashes.json'))['changed_or_missing']:raise ValueError('PRESERVATION_BEFORE_RUN')
    write(pr/'started.json',{'protocol':spec['fingerprint'],'started_at':datetime.now(timezone.utc).isoformat()})
    inputs,identity,truth,models,values,_=load_panel();records=[];paths=0;start=time.perf_counter()
    with (pr/'planner_results.jsonl').open('x',encoding='utf-8') as pf, (pr/'episode_trace.jsonl').open('x',encoding='utf-8') as ef, (pr/'random_paths.jsonl').open('x',encoding='utf-8') as rf:
        planner=PlannerCache(sink=lambda r:dump_line(pf,r))
        orders={k:global_eig_order(m,planner) for k,m in models.items()}
        write(pr/'frozen_global_eig_orders.json',[{'fold':f,'model':m,'order':o} for (f,m),o in orders.items()])
        try:
            for p in inputs:
                i=p['scenario_ordinal'];f=p['fold']
                configs=[('native','M5',POLICIES,False),('fixed_k','M5',['voi_h1','voi_h2','voi_h3'],True),
                         ('M1_aux','M1',['fixed','voi_h3'],False),('M4_aux','M4',['global_eig','voi_h3'],False)]
                for section,m,policies,fixedk in configs:
                    model=models[f,m]
                    for policy in policies:
                        for budget in range(6):
                            if time.perf_counter()-start>spec['limits']['max_wall_seconds']:raise ValueError('TOTAL_WALL_LIMIT')
                            def sink(row):
                                dump_line(rf,{'scenario_ordinal':i,'budget':budget,**row})
                            if policy=='random':
                                result=random_expectation(model,values[i],budget,truth[i],planner,orders[f,m],sink,seed_context=i)
                                paths+=result['enumeration_paths']
                            else:
                                result=episode(model,values[i],policy,budget,planner,orders[f,m],fixed_k=fixedk)
                                score_episode(result,truth[i]);paths+=1
                            result.update(scenario_ordinal=i,fold=f,model=m,section=section)
                            dump_line(ef,result)
                            # Compact score rows have no raw outcomes/diagnosis but stay private.
                            records.append({k:v for k,v in result.items() if k not in ('decisions','releases','belief','terminal_choice')})
                            if paths>spec['limits']['maximum_environment_paths']:raise ValueError('ENVIRONMENT_PATH_LIMIT')
                print(json.dumps({'visit_completed':i,'slots_completed':len(records),'paths':paths,'cache_misses':planner.misses,'cache_hits':planner.hits,'seconds':round(time.perf_counter()-start,2)}),flush=True)
        except Exception as exc:
            write(pr/'failure.json',{'error_type':type(exc).__name__,'code':str(exc),'completed_slots':len(records),'paths':paths})
            write(pub/'execution_status.json',{'status':'C4_31_SEQUENTIAL_POLICY_EVALUATION_PARTIAL','completed_slots':len(records),'registered_slots':8280,'failure_type':type(exc).__name__})
            raise
        finally:
            write(pr/'score_rows.json',records)
            write(pub/'planner_resource_usage.json',planner.resource_report())
    write(pub/'execution_status.json',{'status':'C4_31_SEQUENTIAL_POLICY_EVALUATION_COMPLETE','completed_slots':len(records),'registered_slots':8280,
        'main_slots':sum(r['section']=='native' for r in records),'environment_paths':paths,'random_repetition_paths':sum(r.get('enumeration_paths',0) for r in records),'seconds':time.perf_counter()-start})


def aubc_rows(records,policy):
    groups=defaultdict(list)
    for r in records:
        if r['policy']==policy and 1<=r['budget']<=5:groups[r['scenario_ordinal']].append(r)
    out=[]
    for i,rs in groups.items():
        if len(rs)!=5 or {r['budget'] for r in rs}!=set(range(1,6)):raise ValueError('AUBC_MISSING_BUDGET')
        out.append({'scenario_ordinal':i,'status':'completed','actual_actions':sum(r['actual_actions'] for r in rs)/5,
                    'score':{k:sum(r['score'][k] for r in rs)/5 for k in SCORE_KEYS}})
    return out


def analyze(namespace):
    pub,pr=places(namespace);verify_protocol(pub)
    if read(pub/'execution_status.json')['status']!='C4_31_SEQUENTIAL_POLICY_EVALUATION_COMPLETE':raise ValueError('PARTIAL_NO_COMPLETE_INFERENCE')
    records=read(pr/'score_rows.json');identity={r['scenario_ordinal']:r for r in read(PREP/'identity_membership.json')}
    native=[r for r in records if r['section']=='native']
    def curves(rs):
        return {p:{str(k):mean_metrics([r for r in rs if r['policy']==p and r['budget']==k],identity) for k in range(6)} for p in sorted({r['policy'] for r in rs})}
    metrics=curves(native);probe=curves([r for r in records if r['section']=='fixed_k'])
    write(pub/'random_monte_carlo_uncertainty.json',{'repetitions':32,'budgets':{str(k):{'conditional_visit_mean_top1_MC_SE':math.sqrt(sum(r['random_top1_mean_variance'] for r in native if r['policy']=='random' and r['budget']==k))/92} for k in range(6)},
        'interpretation':'simulation variability conditional on cohort; cluster bootstrap conditional on fixed paths, not a clinical confidence interval'})
    write(pub/'native_policy_metrics.json',metrics);write(pub/'budget_curve_metrics.json',metrics)
    write(pub/'fixed_k_probe_metrics.json',{'status':'EXECUTED','scope':'Q_RANKING_ONLY_NOT_NATIVE_STOP','metrics':probe})
    for sec,name in [('M1_aux','m1_auxiliary_comparison.json'),('M4_aux','m4_sensitivity_comparison.json')]:
        write(pub/name,{'status':'EXECUTED','metrics':curves([r for r in records if r['section']==sec]),
            'main_model_replaced':False,'interpretation':'M1 action-universe/encoding confounded; M4 encoding sensitivity. No refit.'})
    paired={};boot={}
    for a,b in PAIRS:
        key=a+'_minus_'+b
        aubc=paired_comparison(aubc_rows(native,a),aubc_rows(native,b),identity)
        budget=paired_comparison([r for r in native if r['policy']==a and r['budget']==5], [r for r in native if r['policy']==b and r['budget']==5],identity)
        # Per-budget descriptive comparisons use same registered pairs (no bootstrap repetition here).
        each={}
        for k in range(6):
            each[str(k)]=paired_comparison([r for r in native if r['policy']==a and r['budget']==k], [r for r in native if r['policy']==b and r['budget']==k],identity,repetitions=1)
            each[str(k)].pop('patient_macro');each[str(k)]['bootstrap']='NOT_COMPUTED_PER_BUDGET; see AUBC and preregistered k5'
        paired[key]={'AUBC':aubc,'budget5':budget,'per_budget':each};boot[key]={'AUBC':aubc['patient_macro'],'budget5':budget['patient_macro']}
    write(pub/'paired_policy_comparison.json',paired);write(pub/'patient_bootstrap_results.json',boot)
    primary=paired['voi_h3_minus_global_eig']['AUBC']
    primary.update(primary_metric='mean Top1 budgets1..5',policies=['voi_h3','global_eig'],
        baseline_AUBC=sum(metrics['global_eig'][str(k)]['visit_macro']['top1'] for k in range(1,6))/5,
        voi_AUBC=sum(metrics['voi_h3'][str(k)]['visit_macro']['top1'] for k in range(1,6))/5)
    write(pub/'primary_comparison.json',primary)
    flags=[]
    def gain(row):return row['visit_macro']['top1']>0 and row['patient_macro']['top1']['ci95'][0]>0
    if gain(primary):flags.append('A_VOI_POLICY_GAIN_SUPPORTED')
    if any(gain(paired[x]['AUBC']) for x in ('voi_h2_minus_voi_h1','voi_h3_minus_voi_h1')):flags.append('C_HORIZON_GAIN_SUPPORTED')
    if not gain(primary):
        flags.append('D_STRONG_FIXED_BASELINE_MATCH' if any(paired[x]['AUBC']['visit_macro']['top1']>0 for x in ('voi_h3_minus_random','voi_h3_minus_fixed')) else 'E_NO_VOI_POLICY_GAIN_OBSERVED')
    text='# C4-31 research decision\n\n'+', '.join(flags)+'\n\n'+json.dumps(primary,indent=2)+'\n\nDevelopment Source agreement only. No clinical efficacy, equivalence, safety, official B4 or independent holdout claim.\n'
    with (pub/'research_decision.md').open('x',encoding='utf-8') as f:f.write(text)
    analyze_traces(pub,pr,identity,paired)
    plots(pub,metrics)


def analyze_traces(pub,pr,identity,paired):
    traces=rows(pr/'episode_trace.jsonl');native=[r for r in traces if r['section']=='native' and r['policy']!='random']
    keys=read(PARENT/'action_keys.json');base=set(keys['base']);actions=defaultdict(Counter);positions=defaultdict(Counter)
    sourcekind=defaultdict(Counter);replanning=defaultdict(Counter);values=defaultdict(list);violations=Counter();support=defaultdict(Counter)
    models={(f,m):read(PARENT/('fold%02d_%s_model.json'%(f,m))) for f in range(1,6) for m in ('M5','M4')}
    for r in native:
        p=r['policy'];model=models[r['fold'],'M5'];mapping=model['provenance']['action_key_mapping'];reverse={a:k for k,a in mapping.items()}
        previous_next=None
        for pos,d in enumerate(r['decisions'],1):
            selected=d['selected_action']
            if d.get('effective_horizon',0)>min(3,d['remaining_budget'],8-len([z for z in r['decisions'][:pos-1] if z['selected_action']])):violations['horizon']+=1
            if 'action_values' in d:
                ranked=[x['action_id'] for x in sorted(d['action_values'],key=lambda x:(-x['q_value'],x['action_id']))]
                if previous_next is not None:replanning[p]['next_choice_comparisons']+=1;replanning[p]['changed_from_previous_Q_ranking']+=int(selected!=previous_next)
                previous_next=next((a for a in ranked if a!=selected),None)
                if selected is not None:
                    q=next(x for x in d['action_values'] if x['action_id']==selected)
                    values[p].append({'predicted_voi':q['voi'],'predicted_Q':q['q_value'],'T_before':d['T_before'],
                        'realized_T_delta':d['T_after']-d['T_before'],'reference_probability_delta':d['reference_probability_delta'],
                        'final_agreement':r['score']['top1']})
            if selected is None:continue
            actions[p][selected]+=1;positions[p][str(pos)+':'+selected]+=1
            sourcekind[p]['base5' if reverse[selected] in base else 'new3']+=1
            cnt=model['provenance']['empirical_support'][selected]
            support[p]['acquisitions_with_any_zero_diagnosis_support']+=int(any(n==0 for n in cnt.values()))
            support[p]['zero_support_diagnosis_cells_encountered']+=sum(n==0 for n in cnt.values())
        if len({x['action_id'] for x in r['releases']})!=r['actual_actions']:violations['repeat']+=1
        for e in r['releases']:
            if e['released_at_minute']<e['available_minute'] or e['released_at_minute']<e['selected_at_minute']:violations['future_release']+=1
    # Exact random expected positional distribution, uniform without replacement.
    random_distribution={str(k):{'theoretical_each_action_inclusion_probability':k/8,'theoretical_each_position_each_action_probability':1/8,'subsets':math.comb(8,k),'actual_repetitions':32} for k in range(6)}
    write(pub/'action_selection_distribution.json',{'deterministic_native_counts':{p:dict(c) for p,c in actions.items()},
        'position_action_counts':{p:dict(c) for p,c in positions.items()},'base5_vs_new3':{p:dict(c) for p,c in sourcekind.items()},
        'scope':'pooled independent budget episodes, not unique visits','random_theoretical_uniform_marginals':random_distribution})
    indexed={(r['scenario_ordinal'],r['policy'],r['budget']):r for r in native}
    comparisons={}
    for p,b in [('voi_h3','global_eig'),('voi_h2','voi_h1'),('voi_h3','voi_h1'),('voi_h3','voi_h2')]:
        counts=Counter()
        for i in range(1,93):
            for k in range(1,6):
                a=indexed[i,p,k];c=indexed[i,b,k];x=[z['action_id'] for z in a['releases']];y=[z['action_id'] for z in c['releases']]
                counts['episodes']+=1;counts['different_full_path']+=int(x!=y);counts['different_first_action']+=int(x[:1]!=y[:1]);counts['different_action_set']+=int(set(x)!=set(y))
        comparisons[p+'_vs_'+b]=dict(counts)
    write(pub/'adaptive_replanning_analysis.json',{'replan_vs_previous_remaining_Q_order':{p:dict(c) for p,c in replanning.items()},
        'path_comparison':comparisons,'interpretation':'changing relative Q ranking after Source update is descriptive; no causal clinical benefit inferred'})
    write(pub/'horizon_comparison.json',{'paired':{k:v for k,v in paired.items() if k in ('voi_h2_minus_voi_h1','voi_h3_minus_voi_h1','voi_h3_minus_voi_h2')},'paths':comparisons})
    stops={}
    for p in POLICIES:
        stops[p]={str(k):dict(Counter(r['stop_reason'] for r in traces if r['section']=='native' and r['policy']==p and r['budget']==k)) for k in range(6)}
    write(pub/'stop_and_budget_analysis.json',{'counts':stops,'budget0':'terminal due to zero cap, not native utility STOP','prior':'immediate prior-only, not counted as nativeSTOP'})
    write(pub/'predicted_vs_realized_value.json',{p:{'decisions':len(rs),'means':{k:sum(r[k] for r in rs)/len(rs) for k in rs[0]},
        'predicted_positive_actual_reference_decrease':sum(r['predicted_voi']>0 and r['reference_probability_delta']<0 for r in rs),
        'predicted_positive_actual_T_decrease':sum(r['predicted_voi']>0 and r['realized_T_delta']<0 for r in rs),
        'interpretation':'model expectation across possible outcomes vs one realized Source observation; final agreement is episode level'} for p,rs in values.items()})
    truth={r['scenario_ordinal']:r['source_diagnosis_reference'] for r in read(PREP/'scoring_keys.json')};labels=sorted(set(truth.values()))
    sub=[]
    for ordinal,label in enumerate(labels,1):
        ids={i for i,d in truth.items() if d==label};rs=[r for r in traces if r['section']=='native' and r['budget']==5 and r['scenario_ordinal'] in ids]
        sub.append({'diagnosis_ordinal':ordinal,'visits':len(ids),'patients':len({identity[i]['patient_id'] for i in ids}),
            'metrics':{p:mean_metrics([r for r in rs if r['policy']==p],identity) for p in POLICIES}})
    write(pr/'diagnosis_ordinal_mapping.json',{str(i+1):d for i,d in enumerate(labels)})
    write(pub/'diagnosis_subgroup_analysis.json',{'budget':5,'groups':sub,'zero_support_use':{p:dict(c) for p,c in support.items()},
        'strict_clinical_gate':'NOT_APPROVED; zero-support smoothed cells retained','rare_subgroups':'small denominators descriptive, no clinical subgroup accuracy claim'})
    if violations:raise ValueError('OBSERVED_TRACE_BOUNDARY_FAILURE')
    write(pub/'leakage_and_exposure_audit.json',{'trace_boundary_violations':dict(violations),'cohort':{'patients':63,'visits':92},
        'policy_API':'belief/frozen model/remaining IDs/cap/order only, no environment/scorer/reference/identity',
        'source':'selected only; max(clock,recorded minute) before posterior','gold_in_policy':False,
        'exposure':['development Source complete panel; selection bias excludes137','shared diagnosis universe originally development-exposed',
                    'recorded-time proxy not clinically approved','conditional independence and Utility1/0 unvalidated','Routing validation5 overlaps historical VoI train5 remains'],
        'clinical_evaluation_ready':False,'new_generation':0,'new_probability_fits':0,'routing_fit':0,'new_gold':0})


def plots(pub,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,name,title in [('top1','budget_accuracy_curve.png','Top-1 Source agreement'),('nll','budget_nll_curve.png','Diagnosis NLL'),
        ('brier','budget_brier_curve.png','Multiclass Brier (sum)'),('actual_actions','actual_action_count_curve.png','Mean acquired actions'),('stop','budget_stop_curve.png','Native STOP rate')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for p in POLICIES:ax.plot(range(6),[metrics[p][str(k)]['visit_macro'][metric] for k in range(6)],marker='o',label=p)
        ax.set(xlabel='Maximum acquisitions',ylabel=title,title=title+' | 92 development Source visits');ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(pub/name,dpi=160);plt.close(fig)
    for name,policies,title in [('policy_comparison.png',POLICIES,'Mean budget1-5 Top1 (AUBC)'),('horizon_comparison.png',['voi_h1','voi_h2','voi_h3'],'Native horizon comparison (AUBC)')]:
        fig,ax=plt.subplots(figsize=(9,5));ys=[sum(metrics[p][str(k)]['visit_macro']['top1'] for k in range(1,6))/5 for p in policies]
        ax.bar(policies,ys);ax.set(ylabel='Source agreement',title=title);ax.tick_params(axis='x',labelrotation=25);fig.tight_layout();fig.savefig(pub/name,dpi=160);plt.close(fig)


def regression(namespace):
    import unittest
    pub,pr=places(namespace);verify_protocol(pub)
    old=read(PUBLIC_PARENT/'test_results.json')['patterns'];patterns=[n for n in old if n.endswith('.py')]+['test_voi_sequential_policy_evaluation.py']
    suite=unittest.TestSuite();counts={}
    for pattern in patterns:
        ss=unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'),pattern=pattern);counts[pattern]=ss.countTestCases();suite.addTests(ss)
    mod=str(ROOT/'tests/locdi')
    if mod not in sys.path:sys.path.insert(0,mod)
    ss=unittest.defaultTestLoader.loadTestsFromName('test_phase211_alignment.DecisionMathAlignmentTests');counts['test_phase211_alignment.DecisionMathAlignmentTests']=ss.countTestCases();suite.addTests(ss)
    start=time.perf_counter()
    with (pr/'regression.log').open('x',encoding='utf-8') as f:
        with patch.object(socket.socket,'connect',side_effect=RuntimeError('NETWORK_DISABLED_C4_31')),patch.object(socket.socket,'bind',side_effect=RuntimeError('NETWORK_DISABLED_C4_31')):
            result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    report={'total':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'seconds':time.perf_counter()-start,
        'patterns':counts,'scope':'RELATED_REGRESSION; full repository suite NOT_RUN','network':'DISABLED','generation':0,'source_fits':0}
    write(pub/'test_results.json',report);print(json.dumps(report),flush=True)
    if not result.wasSuccessful():raise ValueError('REGRESSION_FAILED')


def seal(namespace):
    pub,pr=places(namespace);verify_protocol(pub)
    protected=check(read(pr/'protected_hashes.json'));sources=check(read(pr/'source_hashes.json'))
    if protected['changed_or_missing'] or sources['changed_or_missing']:raise ValueError('PRESERVATION_FAILED')
    code=read(pub/'sequential_policy_protocol.json')['code_hashes']
    public='\n'.join(p.read_text(encoding='utf-8') for p in pub.glob('*.json'))
    identities=read(PREP/'identity_membership.json');refs=read(PREP/'scoring_keys.json')
    sensitive={v for r in identities for k,v in r.items() if k in ('patient_id','visit_id')}|{r['source_diagnosis_reference'] for r in refs}
    if any(s in public for s in sensitive):raise ValueError('PUBLIC_IDENTIFIER_OR_REFERENCE_LEAKAGE')
    write(pub/'verification.json',{'status':'PASS','protected':protected,'sources':sources,'code':check(code),
        'Core_changes':0,'source_changes':0,'old_artifact_changes':0,'old_report_changes':0,'LLM_generation':0,'new_probability_fit':0,
        'routing_fit':0,'new_gold':0,'source_sampling':0,'manuscript_changes':0,'git_commit_push':False,
        'public_identity_and_reference_scan':'PASS','clinical_evaluation_ready':False,'official_B0_B4':'NOT_RUN',
        'D7':'existing code-version mismatch remains; not rewritten or re-sealed'})
    command='python -B scripts/evaluate_voi_sequential_policies.py {prepare,run,analyze,regression,seal} --namespace '+namespace
    with (pub/'actual_execution_command.md').open('x',encoding='utf-8') as f:
        f.write('# Execution\n\nEach stage executed separately, in listed order. Seal follows report writing.\n\n```powershell\n'+
            '\n'.join('python -B scripts/evaluate_voi_sequential_policies.py '+s+' --namespace '+namespace for s in ('prepare','run','analyze','regression','seal'))+
            '\n```\n\nCompleted namespaces reject duplicate execution; no resume. Independent reproduction requires a fresh namespace, frozen identical code/source and the same staged commands.\n')
    index={'files':{p.relative_to(ROOT).as_posix():sha(p) for p in pr.rglob('*') if p.is_file()}}
    index['fingerprint']=digest(index['files']);write(pr/'private_index.json',index)
    report=next((ROOT/'docs').rglob('C4_31_VOI_SEQUENTIAL_POLICY_EVALUATION_REPORT.md'))
    paths=[p for p in pub.rglob('*') if p.is_file()]+[pr/'private_index.json',report]+[ROOT/n for n in CODE]
    manifest={'status':read(pub/'execution_status.json')['status'],'protocol_fingerprint':read(pub/'sequential_policy_protocol.json')['fingerprint'],
        'files':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    manifest['fingerprint']=digest(manifest['files']);write(pub/'completion_manifest.json',manifest)
    print(json.dumps({'seal':'PASS','protected':protected['checked'],'sources':sources['checked']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','run','analyze','regression','seal']);ap.add_argument('--namespace',default=NS)
    args=ap.parse_args()
    with patch.object(socket.socket,'connect',side_effect=RuntimeError('NETWORK_DISABLED_C4_31')):
        globals()[args.stage](args.namespace)
