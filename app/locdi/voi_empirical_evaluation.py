"""Read-only selector replay of recorded decisions, not a new runtime/policy."""
from copy import deepcopy
import json
import math

from app.core.information_outcomes import InformationOutcomes
from app.core.information_value import select_action
from app.core.voi_planning import plan_voi
from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest

VERSION='locdi.voi_empirical.c4_18.v1'


def reject_labels(value):
    forbidden={'gold','evaluation','task_annotations','source_event_store','correct_diagnosis',
               'gold_department','source_condition','source_conditions','correctness','reference_diagnosis'}
    if isinstance(value,dict):
        if {str(k).casefold() for k in value}&forbidden:raise ValueError('PRIVATE_LABEL_IN_OBSERVATION')
        for x in value.values():reject_labels(x)
    elif isinstance(value,list):
        for x in value:reject_labels(x)
    elif isinstance(value,str) and value.lstrip().startswith(('{','[')):
        try:parsed=json.loads(value)
        except ValueError:return
        reject_labels(parsed)


def released_observation(case,event_log,minute,memory,*,before_event_index):
    """Rebuild VoI-consumed fields from delivered events, never hidden lookup alone.

    This is a declared source-event replay projection, not the saved raw LLM wire.
    """
    if type(minute) not in (int,float) or not math.isfinite(minute):raise ValueError('DECISION_MINUTE_REQUIRED')
    if type(before_event_index) is not int or not 0<=before_event_index<=len(event_log):raise ValueError('EVENT_BOUNDARY_REQUIRED')
    identity=case['longitudinal'];cutoff=timestamp(identity['started_at'])
    allowed={r['memory_id']:r for p in case['past_visits'] for r in p['records']}
    for r in memory.get('records',[]):
        if (r['memory_id'] not in allowed or r['text']!=allowed[r['memory_id']]['text']
            or r['patient_id']!=identity['patient_id'] or r['visit_id']==identity['visit_id']
            or timestamp(r['visit_started_at'])>=cutoff or timestamp(r['available_at'])>cutoff):
            raise ValueError('MEMORY_IDENTITY_OR_CUTOFF')
    observation={'objective_findings':[],'history':[],'longitudinal_memory':deepcopy(memory),'elapsed_minutes':minute}
    targets={}
    for src in case['current_encounter_events']:
        key=(src['action'],src['target'])
        if key in targets:raise ValueError('AMBIGUOUS_SOURCE_ACTION')
        targets[key]=src
    for e in event_log[:before_event_index]:
        if e['minute']>minute or e['case_id']!=case['case_id']:continue
        p=e['payload'];kind=e['event_type']
        if kind not in {'result_received','question_asked'}:continue
        target=p['requested_test'] if kind=='result_received' else p['question']
        src=targets.get(('Test' if kind=='result_received' else 'Ask',target))
        if src is None or e['minute']<src['available_minute']:raise ValueError('SOURCE_RELEASE_BINDING')
        if kind=='result_received':
            # A source value is admitted only after a recorded order and result.
            if not any(x['case_id']==case['case_id'] and x['event_type']=='test_ordered'
                       and x['minute']<=e['minute'] and x['payload'].get('requested_test')==target for x in event_log):
                raise ValueError('RESULT_WITHOUT_RECORDED_ORDER')
            finding=target+': '+src['evidence']
            if p['test_result']['summary']!=finding:raise ValueError('RELEASED_RESULT_SOURCE_MISMATCH')
            observation['objective_findings'].append(finding)
        else:
            if p['answer']!=src['evidence']:raise ValueError('RELEASED_ANSWER_SOURCE_MISMATCH')
            observation['history'].append({'question':target,'answer':p['answer'],'answer_status':'provided'})
    reject_labels(observation)
    return observation


def decision_key(row):
    if row is None:return None
    kind=row.get('action')
    # STOP comparison does not grade or equate predicted diagnosis strings.
    return (kind,None if kind=='diagnose' else row.get('target'))


def replay(artifact,observation,candidates,specialist_proposal):
    """Existing OFF branch and fixed-depth planners; no fitting or clinical gate."""
    reject_labels(observation)
    model=InformationOutcomes(artifact);belief,matched=model.observe(observation)
    required={'action_id','type','action','target','outcome_model'}
    interface=[{k:c[k] for k in required} for c in candidates]
    if len({c['action_id'] for c in interface})!=len(interface):raise ValueError('DUPLICATE_CANDIDATES')
    modeled={a['id']:a for a in model.artifact['actions']}
    info=[modeled[c['outcome_model']] for c in interface if c['type']!='Diagnose']
    off=select_action(deepcopy(interface),selector='specialist_heuristic',specialist_proposal=deepcopy(specialist_proposal))
    if off['selected_action'] is None:raise ValueError('EXISTING_OFF_PROPOSAL_REQUIRED')
    out={'input_fingerprint':digest(observation),'candidate_fingerprint':digest(interface),
        'artifact_fingerprint':model.fingerprint,'belief':belief,'matched_outcomes':matched,'off':off,'horizons':{}}
    for h in (1,2,3):
        # Local sensitivity parameter, not a persisted/retrained artifact.
        view=deepcopy(model.artifact);view['horizon']=h
        selected=select_action(deepcopy(interface),selector='bayesian_voi',belief=belief,artifact=view)
        plan=plan_voi(belief,info,view['terminal_utility'],horizon=h)
        choice=selected['selected_action']
        if (plan['selected_action_id'] is None and (choice is None or choice['type']!='Diagnose')
            or plan['selected_action_id'] is not None and (choice is None or choice['action_id']!=plan['selected_action_id'])):
            raise ValueError('PLANNER_SELECTOR_MISMATCH')
        out['horizons'][str(h)]={'selection':selected,'plan':plan}
    none=deepcopy(observation);none['longitudinal_memory']={'mode':'none','records':[]}
    b0,m0=model.observe(none)
    if b0!=belief or m0!=matched:
        out['memory_direct_belief_changed']=True
    else:out['memory_direct_belief_changed']=False
    out['memory_action_count']=sum(a['type']=='retrieve_memory' for a in model.artifact['actions'])
    out['memory_off_input_fingerprint']=digest(none)
    out['memory_none_belief']=b0
    out['off_on_action_changed']=decision_key(off['selected_action'])!=decision_key(out['horizons'][str(artifact.get('horizon',1))]['selection']['selected_action'])
    out['horizon_action_changed']=len({decision_key(x['selection']['selected_action']) for x in out['horizons'].values()})>1
    return out


def distribution_stats(values):
    if not values:return {'count':0,'minimum':None,'maximum':None,'mean':None,'status':'NOT_APPLICABLE'}
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('NONFINITE_METRIC')
    return {'count':len(values),'minimum':min(values),'maximum':max(values),'mean':sum(values)/len(values),'status':'COMPUTED'}
