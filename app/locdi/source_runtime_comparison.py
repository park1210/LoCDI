"""Identity-keyed audit only; never changes Planner inputs/order/ties/math."""
import math
from app.core.diagnostic_belief import terminal_choice

TOLERANCE=1e-12


class ComparisonMismatch(ValueError):
    def __init__(self,code,detail=None):
        super().__init__(code);self.code=code;self.detail=detail


def require(ok,code,detail=None):
    if not ok:raise ComparisonMismatch(code,detail)


def indexed(rows):
    require(isinstance(rows,list),'ACTION_ROWS_REQUIRED')
    ids=[r.get('action_id') for r in rows]
    require(all(isinstance(i,str) and i for i in ids),'ACTION_ID_MISSING')
    require(len(ids)==len(set(ids)),'DUPLICATE_ACTION_ID')
    return {r['action_id']:r for r in rows}


def number(a,b,code):
    require(type(a) in (int,float) and type(b) in (int,float)
            and math.isfinite(a) and math.isfinite(b),'NONFINITE_'+code)
    delta=abs(a-b);require(delta<=TOLERANCE,code,{'absolute_difference':delta});return delta


def belief(a,b):
    require(set(a)==set(b),'BELIEF_UNIVERSE_MISMATCH')
    return max((number(a[d],b[d],'POSTERIOR_MISMATCH') for d in a),default=0)


def compare_plan(actual,reference):
    left=indexed(actual['action_values']);right=indexed(reference['action_values'])
    require(set(left)==set(right),'ACTION_ID_SET_MISMATCH')
    require(actual['horizon']==reference['horizon'],'HORIZON_MISMATCH')
    q=voi=outcome=0
    for aid,a in left.items():
        b=right[aid]
        q=max(q,number(a['q_value'],b['q_value'],'Q_MISMATCH'))
        voi=max(voi,number(a['voi'],b['voi'],'VOI_MISMATCH'))
        number(a['information_gain_nats'],b['information_gain_nats'],'EIG_MISMATCH')
        ao=a.get('outcomes',[]);bo=b.get('outcomes',[])
        require(len(ao)==len(bo),'OUTCOME_COUNT_MISMATCH')
        require(len({r['outcome_id'] for r in ao})==len(ao)
                and len({r['outcome_id'] for r in bo})==len(bo),'DUPLICATE_OUTCOME_ID')
        bm={r['outcome_id']:r for r in bo}
        require({r['outcome_id'] for r in ao}==set(bm),'OUTCOME_ID_SET_MISMATCH')
        for row in ao:
            other=bm[row['outcome_id']]
            outcome=max(outcome,number(row['probability'],other['probability'],'OUTCOME_MASS_MISMATCH'))
            if row['posterior'] is None or other['posterior'] is None:
                require(row['posterior'] is None and other['posterior'] is None,'OUTCOME_POSTERIOR_MISMATCH')
            else:outcome=max(outcome,belief(row['posterior'],other['posterior']))
    # Raw order is recorded, never fed back into policy or used to align values.
    require((actual['selected_action_id'] is None)==(reference['selected_action_id'] is None),'STOP_MISMATCH')
    require(actual['selected_action_id']==reference['selected_action_id'],'SELECTED_ACTION_MISMATCH')
    t=number(actual['baseline_terminal_utility'],reference['baseline_terminal_utility'],'T_MISMATCH')
    number(actual['optimal_value'],reference['optimal_value'],'OPTIMAL_VALUE_MISMATCH')
    return {'action_ID_set_match':True,'raw_order_equal':list(left)==list(right),
        'action_count':len(left),'maximum_Q_difference':q,'maximum_VoI_difference':voi,
        'maximum_T_difference':t,'maximum_outcome_posterior_difference':outcome,
        'selected_action_match':True,'STOP_match':True}


def compare_episode(actual,reference,plans,model,*,budget=5,audit=None):
    audit=audit if audit is not None else {}
    audit.update(action_identity_comparison=[],posterior_comparison=[],decision_comparison=[])
    require(len(plans)==len(reference['decisions']),'REFERENCE_PLAN_DECISION_COUNT')
    require(len(actual['decisions'])>=len(plans),'RUNTIME_DECISION_COUNT')
    for i,(plan,decision) in enumerate(zip(plans,reference['decisions'])):
        a=actual['decisions'][i];require(len(a['plans'])==1,'RUNTIME_PLAN_COUNT')
        require(a['remaining_budget']==decision['remaining_budget']==budget-i,'BUDGET_MISMATCH')
        require(plan['horizon']==decision['effective_horizon'],'REFERENCE_HORIZON_MISMATCH')
        row=compare_plan(a['plans'][0],plan);audit['action_identity_comparison'].append(row)
        number(a['T_before'],decision['T_before'],'DECISION_T_BEFORE_MISMATCH')
        if 'T_after' in decision:number(a['T_after'],decision['T_after'],'DECISION_T_AFTER_MISMATCH')
        before=belief(a['belief_before'],decision['belief_before'])
        after=belief(a['belief_after'],decision.get('belief_after',decision['belief_before']))
        audit['posterior_comparison'].append({'decision':i+1,'before_max_abs':before,'after_max_abs':after})
        audit['decision_comparison'].append({'decision':i+1,'selected_and_STOP_match':True,
            'effective_horizon':plan['horizon'],'remaining_budget':a['remaining_budget']})
    extras=actual['decisions'][len(plans):]
    if extras:
        require(len(extras)==1 and reference['stop_reason']=='BUDGET_EXHAUSTED'
                and reference['actual_actions']==budget,'UNMATCHED_RUNTIME_DECISION')
        extra=extras[0];require(extra['remaining_budget']==0,'TERMINAL_BUDGET_MISMATCH')
        require(len(extra['plans'])==1,'TERMINAL_PLAN_COUNT')
        p=extra['plans'][0]
        require(not p['action_values'] and p['selected_action_id'] is None
                and extra['proposal']['action']=='diagnose','EXTRA_DECISION_NOT_TERMINAL')
        belief(extra['belief_before'],reference['belief']);belief(extra['belief_after'],reference['belief'])
        number(p['baseline_terminal_utility'],terminal_choice(reference['belief'],model['terminal_utility'])[1],'TERMINAL_T_MISMATCH')
    got=[(e['action_id'],e['outcome_id'],e['released_at_minute']) for e in actual['actions'] if e.get('action_id') is not None]
    wanted=[(e['action_id'],e['outcome_id'],e['released_at_minute']) for e in reference['releases']]
    require(got==wanted,'SOURCE_RELEASE_MISMATCH')
    final=belief(actual['belief'],reference['belief'])
    require(actual['algorithmic_choice']==reference['terminal_choice'],'TERMINAL_CHOICE_MISMATCH')
    t=number(actual['terminal_value'],terminal_choice(reference['belief'],model['terminal_utility'])[1],'TERMINAL_T_MISMATCH')
    audit.update(status='MATCH',matched_decisions=len(plans),runtime_decisions=len(actual['decisions']),
        reference_decisions=len(plans),extra_terminal_decisions=len(extras),
        maximum_final_posterior_difference=final,maximum_terminal_value_difference=t,
        terminal_choice_match=True,source_releases_match=True,tolerance=TOLERANCE)
    return audit
