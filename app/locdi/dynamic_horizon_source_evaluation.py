"""C4-32 adapter: unchanged frozen Horizon policy and C4-31 Source runner."""
from collections import Counter
from copy import deepcopy
import time
from unittest.mock import patch

from app.core.voi_planning import plan_voi
from app.locdi.dynamic_horizon import DynamicHorizonPolicy, policy_spec
from app.locdi.dynamic_horizon_evaluation import action_interface, choose_dynamic
from app.locdi.llm_setup import digest
from app.locdi.voi_sequential_policy_evaluation import episode, score_episode

VERSION='locdi.dynamic_horizon.source_policy_evaluation.c4_32.v1'
POLICIES=['voi_h1','voi_h2','voi_h3','dynamic']


def plan_key(belief,actions,utility,horizon,max_states=20000):
    return digest({'belief':belief,'actions':actions,'utility':utility,'horizon':horizon,'max_states':max_states})


class ComputeBudget:
    def __init__(self,max_calls=15000,max_states=50000000,max_seconds=3600):
        self.calls=0;self.states=0;self.start=time.perf_counter()
        self.max_calls=max_calls;self.max_states=max_states;self.max_seconds=max_seconds
    def check(self):
        if self.calls>=self.max_calls or self.states>=self.max_states or time.perf_counter()-self.start>self.max_seconds:
            raise ValueError('REGISTERED_COMPUTE_LIMIT')


class PlannerMeter:
    """Per-policy independent cache; logical states and executed states are separate."""
    def __init__(self,*,cached=False,budget=None,sink=None,label=None):
        self.cached=cached;self.cache={};self.budget=budget or ComputeBudget();self.sink=sink;self.label=label;self.state_history=[]
        self.stats={'calls':0,'hits':0,'misses':0,'logical_states':0,'executed_states':0,
                    'maximum_states':0,'planner_seconds':0.0,'meter_seconds':0.0}
    def snapshot(self):return dict(self.stats)
    def __call__(self,belief,actions,utility,*,horizon=1,max_states=20000):
        if max_states!=20000:raise ValueError('CORE_STATE_LIMIT_MUST_REMAIN_20000')
        begin=time.perf_counter();key=plan_key(belief,actions,utility,horizon,max_states) if self.cached or self.sink else None
        self.stats['calls']+=1
        if self.cached and key in self.cache:
            self.stats['hits']+=1;result=deepcopy(self.cache[key])
        else:
            self.budget.check();start=time.perf_counter()
            result=plan_voi(belief,actions,utility,horizon=horizon,max_states=max_states)
            elapsed=time.perf_counter()-start;self.budget.calls+=1;self.budget.states+=result['evaluated_states']
            self.stats['misses']+=1;self.stats['executed_states']+=result['evaluated_states'];self.stats['planner_seconds']+=elapsed
            if self.cached:self.cache[key]=deepcopy(result)
            if self.sink:self.sink({'label':self.label,'key':key,'belief':belief,'actions':actions,'horizon':horizon,'plan':result,'planner_seconds':elapsed})
        self.stats['logical_states']+=result['evaluated_states']
        self.state_history.append(result['evaluated_states'])
        self.stats['maximum_states']=max(self.stats['maximum_states'],result['evaluated_states'])
        self.stats['meter_seconds']+=time.perf_counter()-begin
        return result


def horizon_selection(belief,model,remaining,remaining_budget):
    """Strict public input projection; no Source/Reference/provenance in selector."""
    if type(remaining_budget) is not int or remaining_budget<0:raise ValueError('INVALID_REMAINING_BUDGET')
    lookup={a['id']:a for a in model['actions']}
    if len(remaining)!=len(set(remaining)) or any(a not in lookup for a in remaining):raise ValueError('INVALID_REMAINING_ACTIONS')
    start=time.perf_counter()
    actions=[{'action_id':a['id'],'type':'Ask' if a['type']=='ask_patient' else 'Test',
              'action':a['type'],'target':a['target'],'outcome_model':a['id']} for a in model['actions'] if a['id'] in remaining]
    selected=choose_dynamic(belief,action_interface(actions),
        {'remaining_minutes':None,'resources':None,'used_action_count':len(lookup)-len(remaining)},development_opt_in=True)
    if selected['eligibility_status']=='BLOCKED':raise ValueError('FROZEN_DYNAMIC_POLICY_BLOCKED')
    summary=selected['observable_feature_summary'];h=selected['selected_horizon']
    effective=min(h,remaining_budget) if h is not None and remaining_budget else 0
    return {'requested_horizon':summary.get('uncertainty_requested_horizon',2 if selected['fallback_reason'] else None),
        'selected_horizon':h,'effective_horizon':effective,'remaining_budget':remaining_budget,
        'remaining_actions':len(remaining),'entropy':summary.get('entropy_nats'),'normalized_entropy':summary.get('normalized_entropy'),
        'margin':summary.get('top1_top2_margin'),'entropy_horizon':summary.get('entropy_horizon'),'margin_horizon':summary.get('margin_horizon'),
        'action_cap':bool(h is not None and summary.get('uncertainty_requested_horizon',h)>h),
        'budget_cap':bool(h is not None and effective<h),'selector_seconds':time.perf_counter()-start,
        'selection':selected,'actual_planner_execution':effective>0}


def run_episode(model,values,policy,budget,planner):
    """Source acquisition/posterior/STOP loop is C4-31 episode(), not a new runner."""
    before=planner.snapshot();history_start=len(planner.state_history);selectors=[];start=time.perf_counter()
    if policy=='dynamic':
        def choose(_policy,belief,view,remaining,left,_fixed,_eig,meter,fixed_k=False):
            if fixed_k:raise ValueError('NATIVE_ONLY')
            selected=horizon_selection(belief,view,remaining,left);selectors.append(selected)
            if not selected['effective_horizon']:return None,None,0
            actions=[a for a in view['actions'] if a['id'] in remaining]
            plan=meter(belief,actions,view['terminal_utility'],horizon=selected['effective_horizon'])
            return plan['selected_action_id'],plan,selected['effective_horizon']
        with patch('app.locdi.voi_sequential_policy_evaluation.choose',choose):
            result=episode(model,values,'voi_h3',budget,planner,[])
        result['policy']='dynamic'
        for d,s in zip(result['decisions'],selectors):
            d['requested_horizon']=s['requested_horizon'];d['horizon_selection']=s
        if len(result['decisions'])!=len(selectors):raise ValueError('HORIZON_TRACE_ALIGNMENT')
    elif policy in POLICIES:
        result=episode(model,values,policy,budget,planner,[])
    else:raise ValueError('UNSUPPORTED_MEASUREMENT_POLICY')
    elapsed=time.perf_counter()-start;after=planner.snapshot()
    cost={k:after[k]-before[k] for k in after if k!='maximum_states'}
    cost['maximum_states']=max(planner.state_history[history_start:],default=0)
    result['resources']={**cost,'episode_wall_seconds':elapsed,'selector_seconds':sum(s['selector_seconds'] for s in selectors)}
    result['terminal_only_reason']='BUDGET_ZERO' if budget==0 else 'NO_ACTIONS' if not model['actions'] else None
    return result


def normalized_trace(result):
    return {'belief':result['belief'],'terminal':result['terminal_choice'],'stop_reason':result['stop_reason'],
        'actions':[(r['action_id'],r['outcome_id'],r['released_at_minute']) for r in result['releases']],
        'horizons':[d['effective_horizon'] for d in result['decisions']]}


def resource_summary(results):
    keys=['calls','hits','misses','logical_states','executed_states','planner_seconds','meter_seconds','episode_wall_seconds','selector_seconds']
    n=len(results)
    out={'episodes':n,'sums':{k:sum(r['resources'][k] for r in results) for k in keys}}
    out['means']={k:v/n if n else None for k,v in out['sums'].items()}
    out['maximum_states']=max((r['resources']['maximum_states'] for r in results),default=0)
    return out


def parent_plan_resources(trace,model,plans):
    """Fair full-panel logical work, from each exact sealed Core call result."""
    remaining=[a['id'] for a in model['actions']];states=0;maxstate=0;calls=0
    for d in trace['decisions']:
        actions=[a for a in model['actions'] if a['id'] in remaining]
        key=plan_key(d['belief_before'],actions,model['terminal_utility'],d['effective_horizon'])
        p=plans[key]['result']
        if p['selected_action_id']!=d['selected_action']:raise ValueError('SEALED_PLAN_ACTION_MISMATCH')
        if [{k:r[k] for k in ('action_id','q_value','voi','information_gain_nats')} for r in p['action_values']]!=d['action_values']:raise ValueError('SEALED_PLAN_VALUES_MISMATCH')
        calls+=1;states+=p['evaluated_states'];maxstate=max(maxstate,p['evaluated_states'])
        if d['selected_action'] is not None:remaining.remove(d['selected_action'])
    return {'calls':calls,'logical_states':states,'maximum_states':maxstate,
            'wall_time':'NOT_MEASURED_COMPARABLY; C4-31 shared-cache timing not used',
            'basis':'per-input actual sealed Core evaluated_states counted once per logical decision, external-cache-free-equivalent count'}
