"""Opt-in offline adapter. Reuses unmodified selector/planner; no live hook."""
from copy import deepcopy
from time import perf_counter
from unittest.mock import patch
from collections import Counter,defaultdict

from app.core.diagnostic_belief import distribution
from app.core.information_value import select_action
from app.core.voi_planning import plan_voi
from app.locdi.dynamic_horizon import DynamicHorizonPolicy,ACTION_FIELDS


def action_interface(candidates):
    return [{k:c[k] for k in ACTION_FIELDS} for c in candidates]


def restore_model_belief(model,observation,stored_belief,stored_matched):
    """Restore original observe() iteration order, not JSONL's sorted-key order.

    Values and matched evidence must remain exactly equal to the sealed trace.
    This preserves original floating-point sum order without changing Core.
    """
    belief,matched=model.observe(observation)
    if belief!=stored_belief or matched!=stored_matched:
        raise ValueError('SOURCE_OBSERVATION_BELIEF_CHANGED')
    return belief,matched


def run_planner(belief,interface,artifact,horizon):
    """Passively capture the one real planning invocation inside select_action."""
    calls=[]
    def observe(*args,**kwargs):
        value=plan_voi(*args,**kwargs);calls.append(deepcopy(value));return value
    view=deepcopy(artifact);view['horizon']=horizon
    begin=perf_counter()
    with patch('app.core.information_value.plan_voi',side_effect=observe):
        selected=select_action(deepcopy(interface),selector='bayesian_voi',belief=deepcopy(belief),artifact=view)
    seconds=perf_counter()-begin
    if len(calls)!=1 or selected.get('status')!='available' or selected.get('selected_action') is None:
        raise ValueError('PLANNER_SELECTOR_EXECUTION_UNAVAILABLE')
    p=calls[0];choice=selected['selected_action']
    if ((p['selected_action_id'] is None)!=(choice['type']=='Diagnose')
        or p['selected_action_id'] is not None and p['selected_action_id']!=choice['action_id']):
        raise ValueError('PLANNER_SELECTOR_DISAGREEMENT')
    return {'selection':selected,'plan':p,'wall_seconds':seconds,'planner_calls':1,
        'planner_evaluated_states':p['evaluated_states']}


def choose_dynamic(belief,interface,context,*,variant='entropy_margin_feasibility',development_opt_in=False):
    if development_opt_in is not True: raise ValueError('DEVELOPMENT_DYNAMIC_HORIZON_OPT_IN_REQUIRED')
    info=[c for c in interface if c['type']!='Diagnose']
    begin=perf_counter()
    selection=DynamicHorizonPolicy(variant).select(belief=deepcopy(belief),
        available_actions=deepcopy(info),observable_context=deepcopy(context))
    selection['selection_wall_seconds']=perf_counter()-begin
    return selection


def run_dynamic(belief,interface,artifact,context,*,variant='entropy_margin_feasibility',development_opt_in=False):
    selection=choose_dynamic(belief,interface,context,variant=variant,development_opt_in=development_opt_in)
    if selection['eligibility_status']=='BLOCKED':
        raise ValueError('HORIZON_POLICY_INPUT_BLOCKED')
    distribution(belief)  # Fallback does not make an invalid probability model executable.
    h=selection['selected_horizon']
    # Empty-action terminal calculation requires Core's h1..3 API, but performs depth0 work.
    result=run_planner(belief,interface,artifact,1 if h is None else h)
    result['horizon_policy']=selection
    result['total_wall_seconds']=result['wall_seconds']+selection['selection_wall_seconds']
    result['terminal_only_h1_api_sentinel']=h is None
    return result


def common_value(choice,common_plan):
    """Same predeclared H3 evaluator; STOP is T, never counterfactual V3."""
    if choice['type']=='Diagnose':return common_plan['baseline_terminal_utility']
    return next(v['q_value'] for v in common_plan['action_values'] if v['action_id']==choice['action_id'])


def q_diagnostics(plan):
    vals=plan['action_values'];t=plan['baseline_terminal_utility']
    gap=vals[0]['q_value']-vals[1]['q_value'] if len(vals)>1 else None
    stopgap=vals[0]['q_value']-t if vals else None
    return {'top_two_information_q_gap':gap,'exact_information_tie':gap==0 if gap is not None else False,
        'near_information_tie':gap is not None and abs(gap)<=1e-12,
        'best_information_vs_terminal':stopgap,
        'near_stop_boundary':stopgap is not None and abs(stopgap)<=1e-12}


def patient_summaries(rows,registered_case_refs):
    if len(set(registered_case_refs))!=len(registered_case_refs):raise ValueError('DUPLICATE_REGISTERED_CASE')
    grouped=defaultdict(list)
    for r in rows:
        if r['case_ref'] not in registered_case_refs:raise ValueError('UNREGISTERED_PATIENT')
        grouped[r['case_ref']].append(r)
    patients=[]
    for ref in registered_case_refs:
        rs=grouped[ref];n=len(rs)
        changed=sum(r['derived']['dynamic']['binding']!=r['derived']['H2']['binding'] for r in rs)
        near=sum(r['derived']['dynamic']['q_diagnostics']['near_information_tie'] for r in rs)
        patients.append({'case_ref':ref,'decisions':n,
            'horizon_distribution':dict(Counter(str(r['results']['dynamic']['horizon_policy']['selected_horizon']) for r in rs)),
            'changed_actions_vs_H2':changed,'action_change_rate':changed/n if n else None,
            'stop_continue_changes_vs_H2':sum(r['derived']['dynamic']['stop']!=r['derived']['H2']['stop'] for r in rs),
            'fallback_count':sum(bool(r['results']['dynamic']['horizon_policy']['fallback_reason']) for r in rs),
            'near_information_tie_count':near,'near_information_tie_rate':near/n if n else None,
            'no_decision_status':'NOT_APPLICABLE_HISTORICAL_FAILURE' if not n else None})
    active=[p for p in patients if p['decisions']]
    return {'registered_patients':len(registered_case_refs),'patients_with_decisions':len(active),'rows':patients,
        'patient_macro_action_change_rate_over_active':sum(p['action_change_rate'] for p in active)/len(active) if active else None,
        'decision_micro_action_change_rate':sum(p['changed_actions_vs_H2'] for p in patients)/len(rows) if rows else None,
        'independent_decisions_assumed':False,'significance_test':'NOT_RUN'}


def complete_identity_recall(memory_report):
    """Query fields present != complete full-pool reference: use saved metric status."""
    complete=[c for c in memory_report['cases']
              if c['modes']['selective']['recall_status']=='COMPUTED_COMPLETE_POOL']
    numerator=sum(c['modes']['selective']['positive_returned'] for c in complete)
    denominator=sum(c['modes']['selective']['full_pool_positive'] for c in complete)
    return {'complete_queries':len(complete),'numerator':numerator,'denominator':denominator,
            'value':numerator/denominator if denominator else None}
