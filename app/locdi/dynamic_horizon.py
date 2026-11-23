"""Deterministic development Horizon selection, independent of VoI Core.

No likelihood, action Q, gold, future observation, fitting, or runtime injection.
Thresholds are operational assumptions, not calibrated clinical thresholds.
"""
from collections import Counter
from copy import deepcopy
import json
import math

from app.core.diagnostic_belief import distribution, entropy
from app.locdi.llm_setup import digest

VERSION = 'locdi.dynamic_horizon.development.v1'
VARIANTS = {'entropy_margin_feasibility', 'entropy_only', 'margin_only',
            'feasibility_removed_fixed_fallback'}
ACTION_FIELDS = {'action_id', 'type', 'action', 'target', 'outcome_model'}
INFORMATION_KINDS = {'ask_patient', 'request_test', 'consult_specialty', 'retrieve_memory'}
ACTION_TYPES = {'ask_patient':'Ask','request_test':'Test','consult_specialty':'Consult','retrieve_memory':'Memory'}
CONTEXT_FIELDS = {'decision_minute', 'remaining_minutes', 'resources', 'terminal_utility',
                  'terminal_tie_count', 'terminal_action_available', 'used_action_count',
                  'time_source', 'resource_source', 'terminal_support_status'}
FORBIDDEN = {'gold', 'gold_department', 'gold_diagnosis', 'correct_diagnosis', 'evaluation',
    'task_annotations', 'source_condition', 'source_conditions', 'source_event_store',
    'future_outcome', 'future_outcomes', 'future_observation', 'future_observations',
    'outcomes', 'likelihood', 'q_value', 'voi', 'optimal_value', 'correctness',
    'reference_diagnosis', 'accepted_specialties'}


def policy_spec():
    return {'policy_name':'Adaptive State-Conditioned Horizon Selection','policy_version':VERSION,
        'thresholds':{'low':1/3,'high':2/3},
        'threshold_basis':'Equal thirds of unit interval, declared operational assumptions; no sweep/calibration or selection using263 measured Q/action outcomes',
        'entropy_demand':'H1 if normalized_entropy<=1/3; H2 if <=2/3; H3 otherwise',
        'margin_demand':'H1 if top1_top2_margin>=2/3; H2 if >=1/3; H3 otherwise',
        'uncertainty_horizon':'max(entropy_demand, margin_demand) for enabled uncertainty features',
        'primary_policy':'entropy_margin_feasibility',
        'feasibility':'min(uncertainty_horizon, remaining_information_actions,3); all nonempty cases H1..3',
        'zero_information_actions':'selected_horizon=null, eligibility=TERMINAL_ONLY, effective_information_depth=0; unchanged Core invoked at h1 solely for empty-action terminal calculation',
        'fallback':'Fixed H2 capped by known actual information action count; invalid belief/menu cannot be executed; unsafe gold/future input BLOCKED, never fallback',
        'unknown_time_resources':'Explicit null/UNKNOWN, not filled from final state, not used to select depth; no duration model',
        'ablations':['entropy_only','margin_only','feasibility_removed_fixed_fallback'],
        'safety_cap_not_ablatable':True,'live_injection':False,'clinical_thresholds':False}


def reject_unsafe(value):
    if isinstance(value,dict):
        if {str(k).casefold() for k in value} & FORBIDDEN:
            raise ValueError('GOLD_FUTURE_OR_ACTION_VALUE_IN_HORIZON_INPUT')
        for v in value.values(): reject_unsafe(v)
    elif isinstance(value,list):
        for v in value: reject_unsafe(v)
    elif isinstance(value,str) and value.lstrip().startswith(('{','[')):
        try: parsed=json.loads(value)
        except ValueError: return
        reject_unsafe(parsed)


class DynamicHorizonPolicy:
    """Pure selection only. Actual execution requires a separate opt-in adapter."""
    def __init__(self, variant='entropy_margin_feasibility'):
        if variant not in VARIANTS: raise ValueError('UNKNOWN_DEVELOPMENT_VARIANT')
        self.variant=variant
        self.fingerprint=digest({'spec':policy_spec(),'variant':variant})

    def select(self, *, belief, available_actions, observable_context):
        result={'selected_horizon':None,'policy_version':VERSION,'variant':self.variant,
            'selection_reason':None,'observable_feature_summary':{},'input_fingerprint':None,
            'policy_fingerprint':self.fingerprint,'fallback_reason':None,
            'eligibility_status':'BLOCKED','effective_information_depth':None}
        try:
            reject_unsafe([belief,available_actions,observable_context])
            result['input_fingerprint']=digest({'belief':belief,'actions':available_actions,'context':observable_context})
        except (ValueError,TypeError):
            result['selection_reason']='UNSAFE_OR_UNSERIALIZABLE_INPUT';return result
        if (not isinstance(available_actions,list) or len(available_actions)>12
            or any(not isinstance(a,dict) or set(a)!=ACTION_FIELDS or a['action'] not in INFORMATION_KINDS
                   or a['type']!=ACTION_TYPES[a['action']]
                   or any(not isinstance(a[k],str) or not a[k] for k in ['action_id','target','outcome_model']) for a in available_actions)
            or len({a['action_id'] for a in available_actions})!=len(available_actions)):
            result['selection_reason']='INVALID_INFORMATION_ACTION_MENU';return result
        n=len(available_actions)
        summary={'remaining_information_actions':n,
                 'action_counts':dict(Counter(a['action'] for a in available_actions))}
        result['observable_feature_summary']=summary

        def fallback(reason):
            result.update(selected_horizon=min(2,n) if n else None,
                effective_information_depth=min(2,n),fallback_reason=reason,
                selection_reason='FIXED_H2_FALLBACK_WITH_MANDATORY_ACTION_CAP',
                eligibility_status='FALLBACK' if n else 'TERMINAL_ONLY')
            return result

        if not isinstance(observable_context,dict): return fallback('OBSERVABLE_CONTEXT_MISSING')
        if set(observable_context)-CONTEXT_FIELDS:
            result['selection_reason']='CONTEXT_WHITELIST_VIOLATION';return result
        try:
            distribution(belief)
            if not 2<=len(belief)<=32: raise ValueError()
            values=sorted(belief.values(),reverse=True)
            h=entropy(belief);hn=min(1.0,max(0.0,h/math.log(len(values))))
            margin=values[0]-values[1]
        except (ValueError,TypeError): return fallback('BELIEF_FEATURES_INVALID')
        low,high=1/3,2/3
        eh=1 if hn<=low else 2 if hn<=high else 3
        mh=1 if margin>=high else 2 if margin>=low else 3
        minutes=observable_context.get('remaining_minutes')
        if minutes is not None and (type(minutes) not in (int,float) or not math.isfinite(minutes) or minutes<0):
            return fallback('REMAINING_TIME_INVALID')
        summary.update(entropy_nats=h,normalized_entropy=hn,max_posterior=values[0],
            top1_top2_margin=margin,entropy_horizon=eh,margin_horizon=mh,
            terminal_utility=observable_context.get('terminal_utility'),
            terminal_tie_count=observable_context.get('terminal_tie_count'),
            terminal_action_available=observable_context.get('terminal_action_available'),
            used_action_count=observable_context.get('used_action_count'),
            remaining_minutes=minutes,time_status='UNKNOWN' if minutes is None else 'OBSERVED',
            resource_status='UNKNOWN' if observable_context.get('resources') is None else 'OBSERVED_UNUSED',
            time_or_resources_used_in_rule=False)
        if n==0:
            result.update(selection_reason='NO_INFORMATION_ACTIONS_TERMINAL_ONLY',
                eligibility_status='TERMINAL_ONLY',effective_information_depth=0)
            return result
        if self.variant=='feasibility_removed_fixed_fallback':
            return fallback('ACTION_FEASIBILITY_FEATURE_REMOVED_SAFETY_CAP_RETAINED')
        requested=eh if self.variant=='entropy_only' else mh if self.variant=='margin_only' else max(eh,mh)
        selected=min(requested,n,3)
        summary.update(uncertainty_requested_horizon=requested,action_cap_active=selected<requested)
        result.update(selected_horizon=selected,effective_information_depth=selected,
            eligibility_status='ELIGIBLE',selection_reason='UNCERTAINTY_BAND_CAPPED_BY_AVAILABLE_ACTIONS')
        return result
