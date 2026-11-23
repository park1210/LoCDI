"""C3-14 saved-file analysis only. No model/runtime invocation or alias creation."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import sha
from app.locdi.external_reference_readiness import pinned_private_json
from app.locdi.agentclinic_benchmark_scoring import _score,SCORING_VERSION


def scoring_plan(scenarios,episodes,keys):
    return {'schema_version':'agentclinic.saved_lexical_plan.v1','scoring_version':SCORING_VERSION,
        'cohort':[s.scenario_id for s in scenarios],
        'source_records':{s.scenario_id:s.source['record_fingerprint'] for s in scenarios},
        'episode_fingerprints':[digest(e) for e in episodes],
        'scoring_keys_fingerprint':digest([{'scenario_ref':k.scenario_id,'record_fingerprint':k.source_record_fingerprint,
                                         'correct_diagnosis':k.correct_diagnosis} for k in keys]),
        'implementation_sha256':sha(Path(__file__)),'lexical_implementation_sha256':sha(ROOT/'app/locdi/agentclinic_benchmark_scoring.py'),
        'scope':'OFFLINE_AGENTCLINIC_SAVED_PILOT_LEXICAL_ONLY','generation_budget':0,'grading_budget':0}


class PinnedSavedScoringApproval:
    def __init__(self,path,expected_sha256):self.path,self.sha256=path,expected_sha256

    def verify(self,plan):
        a=pinned_private_json(self.path,self.sha256)
        if (a.get('schema_version')!='agentclinic.saved_scoring_approval.v1'
                or a.get('scope')!=plan['scope'] or a.get('plan_fingerprint')!=digest(plan)
                or a.get('approval_source')!='CURRENT_USER_C3_14_ANALYSIS_REQUEST'
                or a.get('approved_by')!='USER_REQUESTING_OFFLINE_ANALYSIS'
                or a.get('permitted_metric')!='EXACT_MATCH_LEXICAL_ONLY'
                or a.get('clinical_equivalence_approved') is not False
                or a.get('alias_approved') is not False
                or a.get('generation_budget')!=0 or a.get('grading_budget')!=0
                or not a.get('request_evidence')):
            raise ValueError('EXPLICIT_SAVED_SCORING_APPROVAL_REQUIRED')
        return a


def score_saved(scenarios,episodes,keys,*,authority=None):
    """Separate actual saved-response entry; old evaluate_test_only stays locked."""
    if not isinstance(authority,PinnedSavedScoringApproval):raise ValueError('SAVED_SCORING_AUTHORITY_REQUIRED')
    plan=scoring_plan(scenarios,episodes,keys);authority.verify(plan)
    registered=[s.scenario_id for s in scenarios]
    if not registered or len(set(registered))!=len(registered):raise ValueError('REGISTERED_COHORT_REQUIRED')
    refs={k.scenario_id:k for k in keys};records={e['scenario_ref']:e for e in episodes}
    if len(refs)!=len(keys) or len(records)!=len(episodes) or set(refs)!=set(registered) or set(records)-set(registered):
        raise ValueError('SCORING_COHORT_OR_KEY_COMPLETENESS')
    rows=[]
    for ordinal,s in enumerate(scenarios,1):
        if refs[s.scenario_id].source_record_fingerprint!=s.source['record_fingerprint']:
            raise ValueError('SCORING_REFERENCE_SOURCE')
        if s.scenario_id not in records:
            row={'state':'not_executed','exact_match':None,'review_required':False}
        else:row=_score(records[s.scenario_id],refs[s.scenario_id],None)
        rows.append(dict(row,ordinal=ordinal))
    n=len(registered);states=Counter(r['state'] for r in rows)
    def rate(num,den):return {'numerator':num,'denominator':den,'value':num/den if den else None,
                            'status':'MEASURED_SAVED_PILOT' if den else 'NOT_APPLICABLE'}
    def distribution(field):
        values=[e['result'][field] for e in episodes if type(e['result'].get(field)) is int]
        return {'sum':sum(values),'denominator':len(values),'mean':statistics.mean(values) if values else None,
                'median':statistics.median(values) if values else None,'missing_planned':n-len(values)}
    trace=[t for e in episodes for t in e['trace']];actions=Counter(t['action'] for t in trace)
    metrics={'exact_match_accuracy':rate(sum(r['exact_match'] is True for r in rows),n),
        'diagnosis_completion_rate':rate(states['completed'],n),'unresolved_rate':rate(states['diagnosis_unresolved'],n),
        'incomplete_rate':rate(states['incomplete'],n),'failed_rate':rate(states['failed'],n),
        'not_executed_rate':rate(states['not_executed'],n),'invalid_prediction_rate':rate(states['invalid_prediction'],n),
        'review_required_count':sum(r['review_required'] for r in rows),'turns':distribution('turn_count'),
        'action_requests':distribution('action_count'),'action_counts':{k:actions[k] for k in ('ASK','EXAMINE','TEST','DIAGNOSE')},
        'invalid_unavailable_action_rate':rate(sum(t['status'] in {'INVALID','UNAVAILABLE','UNSUPPORTED'} for t in trace),len(trace)),
        'repeated_action_rate':rate(sum(bool(t['repeat_of']) for t in trace),sum(t['action'] in {'ASK','EXAMINE','TEST'} for t in trace)),
        'repeated_all_request_rate':rate(sum(bool(t['repeat_of']) for t in trace),len(trace))}
    return {'status':'SAVED_PILOT_LEXICAL_SCORED','scoring_version':SCORING_VERSION,
        'plan_fingerprint':digest(plan),'planned_denominator':n,'metrics':metrics,'private_rows':rows,
        'alias_match':'NOT_EVALUATED_NO_APPROVED_ALIASES','clinical_equivalence':'CLINICAL_EQUIVALENCE_NOT_VERIFIED',
        'clinical_metrics':'NOT_EVALUATED','generation_calls':0,'grader_calls':0}


def trace_analysis(scenario,episode,before_rows):
    """Verify actual prefix visibility, target kind and repeated observations."""
    targets={kind:{t.target_id:t for t in values} for kind,values in
        [('ASK',scenario.ask_targets),('EXAMINE',scenario.exam_targets),('TEST',scenario.test_targets)]}
    if len(before_rows)!=len(episode['trace']):raise ValueError('MISSING_REQUEST_OR_EVENT')
    released=[];seen={};result=[]
    for event,before in zip(episode['trace'],before_rows):
        payload=before['payload']
        if payload['observations']!=released:raise ValueError('OBSERVATION_PREFIX_MISMATCH')
        kind,target=event['action'],event['target'];t=targets.get(kind,{}).get(target)
        key=(kind,target);repeat=key in seen
        if kind in targets and event['status']=='EXECUTED':
            if t is None or event['observation']['value']!=t.value:raise ValueError('SOURCE_OBSERVATION_MISMATCH')
            if repeat:
                if event['repeat_of']!=seen[key]['request_id'] or event['observation']!=seen[key]:
                    raise ValueError('REPEAT_LINK_MISMATCH')
            else:
                if event['repeat_of'] is not None:raise ValueError('FIRST_RELEASE_MARKED_REPEAT')
                seen[key]=event['observation'];released.append(event['observation'])
        result.append({'request_id':before['request_id'],'turn':event['turn_index'],'action':kind,'target':target,
            'node_kind':t.node_kind if t else 'NOT_APPLICABLE','source_path':list(t.source_path) if t else [],
            'remaining_scenario_requests':payload['remaining_requests'],
            'remaining_pilot_requests':49-before['counters']['doctor_request_attempts'],
            'prior_observation_count':len(payload['observations']),'repeated_target':repeat,
            'prior_observation_provided':repeat and seen[key] in payload['observations'],
            'new_observation':bool(t and not repeat),'released_container':isinstance(t.value,(dict,list)) if t else False,
            'released_value_fingerprint':digest(t.value) if t else None,
            'validation':event['validation'],'status':event['status']})
    return result
