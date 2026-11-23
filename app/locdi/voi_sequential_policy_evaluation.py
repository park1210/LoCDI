"""Frozen-model, non-generative C4-31 evaluation. Core mathematics is reused."""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import math
import random
import time
from unittest.mock import patch

from app.core.diagnostic_belief import posterior, terminal_choice
from app.core.voi_planning import plan_voi
from app.core.information_value import select_action
from app.locdi.llm_setup import digest
from app.locdi.voi_information_improvement import quantile
from app.locdi.voi_probability_validation import probability_score
from app.locdi.voi_source_replay_environment import SourceReplayEnvironment

VERSION = 'locdi.voi_sequential_policy_evaluation.c4_31.v1'
POLICIES = ['prior', 'random', 'fixed', 'global_eig', 'information_gain', 'voi_h1', 'voi_h2', 'voi_h3']
SCORE_KEYS = ['top1','top3','nll','brier','reference_probability','max_probability','normalized_entropy','margin']
PAIRS = [('voi_h1','random'),('voi_h2','random'),('voi_h3','random'),
         ('voi_h3','fixed'),('voi_h3','global_eig'),('voi_h3','information_gain'),
         ('voi_h2','voi_h1'),('voi_h3','voi_h1'),('voi_h3','voi_h2')]


class PlannerCache:
    """Exact external memoization; key contains no source outcomes or reference."""
    def __init__(self, max_calls=20000, max_expansions=50000000, max_seconds=3600, sink=None):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.expansions = 0
        self.max_calls, self.max_expansions, self.max_seconds = max_calls, max_expansions, max_seconds
        self.start = time.perf_counter()
        self.by_horizon = defaultdict(lambda: {'calls':0,'hits':0,'misses':0,'seconds':0.0,'evaluated_states':0,'maximum_states':0})
        self.sink = sink
        self.failures = 0

    def __call__(self, belief, actions, utility, *, horizon=1, max_states=20000):
        key = digest({'belief':belief,'actions':actions,'utility':utility,'horizon':horizon,'max_states':max_states})
        stats = self.by_horizon[horizon]
        stats['calls'] += 1
        if key in self.cache:
            self.hits += 1
            stats['hits'] += 1
            return deepcopy(self.cache[key])
        if self.misses >= self.max_calls or self.expansions >= self.max_expansions or time.perf_counter()-self.start > self.max_seconds:
            raise ValueError('PREREGISTERED_PLANNER_RESOURCE_LIMIT')
        start = time.perf_counter()
        try:
            result = plan_voi(belief, actions, utility, horizon=horizon, max_states=max_states)
        except Exception:
            self.failures += 1
            raise
        elapsed = time.perf_counter()-start
        self.misses += 1
        self.expansions += result['evaluated_states']
        stats['misses'] += 1
        stats['seconds'] += elapsed
        stats['evaluated_states'] += result['evaluated_states']
        stats['maximum_states'] = max(stats['maximum_states'], result['evaluated_states'])
        self.cache[key] = deepcopy(result)
        if self.sink:
            self.sink({'cache_key':key,'belief':belief,'actions':actions,'horizon':horizon,'result':result,'seconds':elapsed})
        return deepcopy(result)

    def resource_report(self):
        return {'external_cache_hits':self.hits,'external_cache_misses':self.misses,
                'evaluated_states_sum_on_misses':self.expansions,'by_effective_horizon':dict(self.by_horizon),
                'planner_failures':self.failures,'internal_lru_hits':'NOT_EXPOSED_BY_CORE',
                'wall_seconds':time.perf_counter()-self.start,'max_calls':self.max_calls,
                'max_expansions':self.max_expansions,'max_seconds':self.max_seconds,
                'cache_key':'exact belief/actions/likelihood/utility/horizon/max_states; no Source/reference/IDs',
                'wall_scope':'instrumented offline execution incl serialization; not isolated planner speed benchmark'}


def canonical_order(model):
    return [a for k,a in sorted(model['provenance']['action_key_mapping'].items())]


def global_eig_order(model, planner):
    plan = planner(model['prior'], model['actions'], model['terminal_utility'], horizon=1)
    return [r['action_id'] for r in sorted(plan['action_values'], key=lambda r:(-r['information_gain_nats'],r['action_id']))]


def choose(policy, belief, model, remaining, budget, fixed_order, eig_order, planner, fixed_k=False):
    """Only public belief/model/menu/budget; environment and scorer are absent."""
    if budget <= 0 or not remaining or policy == 'prior':
        return None, None, 0
    if policy in ('fixed','global_eig'):
        order = fixed_order if policy == 'fixed' else eig_order
        return next(a for a in order if a in remaining), None, 0
    if policy not in POLICIES or policy == 'random':
        raise ValueError('UNSUPPORTED_POLICY_DECISION')
    configured = 1 if policy == 'information_gain' else int(policy[-1])
    effective = min(configured, budget, len(remaining))
    actions = [a for a in model['actions'] if a['id'] in remaining]
    if policy == 'information_gain':
        interface = [{'action_id':a['id'],'type':'Ask' if a['type']=='ask_patient' else 'Test',
                      'action':a['type'],'target':a['target'],'outcome_model':a['id']} for a in actions]
        interface.append({'action_id':'diagnose','type':'Diagnose','action':'diagnose','target':None,'outcome_model':None})
        # Scoped injection instruments the existing selector's planner call, no Core change.
        with patch('app.core.information_value.plan_voi', planner):
            result = select_action(interface, selector='information_gain', belief=belief, artifact=model)
        selected = result['selected_action']
        return (None if selected['type']=='Diagnose' else selected['action_id']), None, effective
    plan = planner(belief, actions, model['terminal_utility'], horizon=effective)
    selected = plan['action_values'][0]['action_id'] if fixed_k else plan['selected_action_id']
    return selected, plan, effective


def episode(model, values, policy, budget, planner, eig_order, *, fixed_k=False, forced_order=None):
    env = SourceReplayEnvironment(model, values, budget)
    belief = dict(model['prior'])
    remaining = [a['id'] for a in model['actions']]
    fixed = canonical_order(model)
    decisions = []
    reason = 'PRIOR_ONLY' if policy=='prior' else 'BUDGET_EXHAUSTED'
    while len(env.releases) < budget and remaining and policy != 'prior':
        left = budget-len(env.releases)
        if forced_order is not None:
            selected, plan, effective = forced_order[len(env.releases)], None, 0
        else:
            selected, plan, effective = choose(policy, belief, model, remaining, left, fixed, eig_order, planner, fixed_k)
        decision = {'decision_ordinal':len(decisions)+1,'belief_before':dict(belief),
                    'requested_horizon':None if not policy.startswith('voi_') else int(policy[-1]),
                    'effective_horizon':effective,'remaining_budget':left,'selected_action':selected,
                    'T_before':terminal_choice(belief,model['terminal_utility'])[1]}
        if plan is not None:
            decision.update(root_stop=plan['selected_action_id'] is None,
                predicted_expected_terminal_value=plan['optimal_value'],
                action_values=[{k:r[k] for k in ('action_id','q_value','voi','information_gain_nats')} for r in plan['action_values']])
        decisions.append(decision)
        if selected is None:
            reason = 'NATIVE_STOP'
            break
        released = env.acquire(selected)
        action = next(a for a in model['actions'] if a['id']==selected)
        outcome = next(o for o in action['outcomes'] if o['id']==released['outcome_id'])
        belief = posterior(belief,outcome['likelihood'])[0]
        remaining.remove(selected)
        decision.update(released_outcome=released,belief_after=dict(belief),
                        T_after=terminal_choice(belief,model['terminal_utility'])[1])
    if not remaining and len(env.releases)<budget:
        reason='NO_ACTIONS'
    return {'status':'completed','policy':policy,'budget':budget,'mode':'FIXED_K_Q_RANKING_PROBE' if fixed_k else 'NATIVE',
            'belief':belief,'terminal_choice':terminal_choice(belief,model['terminal_utility'])[0],
            'actual_actions':len(env.releases),'stop':int(reason=='NATIVE_STOP'),
            'budget_exhausted':int(reason=='BUDGET_EXHAUSTED'), 'stop_reason':reason,
            'decisions':decisions,'releases':env.private_trace(),'elapsed_proxy_minutes':env.clock}


def score_episode(result, reference):
    """Called only after policy/environment execution. No score feeds back."""
    result['score'] = probability_score(result['belief'], reference)
    for d in result['decisions']:
        if 'belief_after' in d:
            d['reference_probability_delta'] = d['belief_after'][reference]-d['belief_before'][reference]
        d['final_source_agreement'] = result['score']['top1']
    return result


def random_expectation(model, values, budget, reference, planner, eig_order, path_sink=None, *, seed_context=1, repetitions=32):
    """Preregistered finite random permutations; fresh replay for every path/cap."""
    ids=sorted(a['id'] for a in model['actions'])
    paths=[]
    for repeat in range(repetitions):
        seed=int(digest({'seed':42,'scenario_ordinal':seed_context,'repeat':repeat})[:16],16)
        permutation=list(ids);random.Random(seed).shuffle(permutation)
        result=episode(model,values,'random',budget,planner,eig_order,forced_order=permutation[:budget])
        result.update(random_repeat=repeat,random_seed=seed)
        score_episode(result,reference)
        if path_sink:path_sink(result)
        paths.append(result)
    n=len(paths)
    result={'status':'completed','policy':'random','budget':budget,'mode':'NATIVE_FIXED_32_RANDOM_REPETITIONS',
            'score':{k:sum(p['score'][k] for p in paths)/n for k in SCORE_KEYS},
            'actual_actions':min(budget,len(ids)),'stop':0,'budget_exhausted':1,'stop_reason':'BUDGET_EXHAUSTED',
            'enumeration_paths':n,'decisions':[],'releases':[],
            'random_top1_mean_variance':sum((p['score']['top1']-sum(z['score']['top1'] for z in paths)/n)**2 for p in paths)/max(n-1,1)/n,
            'interpretation':'mean scores of fixed32 ordered paths, not exact expectation or score of average posterior'}
    return result


def mean_metrics(results, membership):
    valid=[r for r in results if r['status']=='completed']
    keys=SCORE_KEYS+['actual_actions','stop','budget_exhausted']
    def value(r,k):return r['score'][k] if k in SCORE_KEYS else r[k]
    groups=defaultdict(list)
    for r in valid:groups[membership[r['scenario_ordinal']]['patient_id']].append(r)
    return {'registered_visits':len(results),'completed_visits':len(valid),'failed_visits':len(results)-len(valid),
        'visit_macro':{k:sum(value(r,k) for r in valid)/len(valid) if valid else None for k in keys},
        'patient_macro':{k:sum(sum(value(r,k) for r in rs)/len(rs) for rs in groups.values())/len(groups) if groups else None for k in keys},
        'patients':len(groups),'accuracy_denominator':'registered92 only when all92 complete; failures never imputed'}


def paired_comparison(left, right, membership, repetitions=2000, seed=42):
    """Left minus right; expected fractional random correctness is explicit."""
    a={r['scenario_ordinal']:r for r in left if r['status']=='completed'}
    b={r['scenario_ordinal']:r for r in right if r['status']=='completed'}
    if set(a)!=set(b) or not a:
        raise ValueError('INCOMPLETE_PAIRED_DENOMINATOR')
    keys=['top1','nll','brier','actual_actions']
    def val(r,k):return r['actual_actions'] if k=='actual_actions' else r['score'][k]
    groups=defaultdict(list)
    for i in sorted(a):groups[membership[i]['patient_id']].append({k:val(a[i],k)-val(b[i],k) for k in keys})
    means=[{k:sum(r[k] for r in groups[p])/len(groups[p]) for k in keys} for p in sorted(groups)]
    rng=random.Random(seed);boot={k:[] for k in keys}
    for _ in range(repetitions):
        draw=[means[rng.randrange(len(means))] for _ in means]
        for k in keys:boot[k].append(sum(r[k] for r in draw)/len(draw))
    discord={'left_only':0.0,'right_only':0.0,'both':0.0,'neither':0.0}
    for i in a:
        x,y=a[i]['score']['top1'],b[i]['score']['top1']
        # At least one side is deterministic in all registered comparisons.
        discord['both']+=x*y;discord['left_only']+=x*(1-y);discord['right_only']+=(1-x)*y;discord['neither']+=(1-x)*(1-y)
    return {'registered_visits':len(left),'paired_visits':len(a),'patients':len(groups),
        'difference':'left minus right','visit_macro':{k:sum(val(a[i],k)-val(b[i],k) for i in a)/len(a) for k in keys},
        'patient_macro':{k:{'difference':sum(r[k] for r in means)/len(means),'ci95':[quantile(boot[k],.025),quantile(boot[k],.975)]} for k in keys},
        'correctness_joint_counts':discord,'random_counts':'fractional average over preregistered32 random repetitions when compared with random',
        'bootstrap':{'repetitions':repetitions,'seed':seed,'unit':'patient cluster, patient macro','CI':'linear percentile95','multiplicity':'unadjusted development'}}
