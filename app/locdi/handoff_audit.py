"""Versioned audit projections; never alter saved episodes or infer wire text."""
from copy import deepcopy
from pathlib import Path
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import load,sha

RUN_ID='20261030000000000000000000002903'


def project_audit(episode,requests,handoffs,routing,*,actual_requests=None,identity_checks=None):
    decisions=episode['workflow']['decisions'];links=[]
    for r in requests:
        decision=next((d for d in decisions if d['sequence']==r.get('core_decision_sequence')),None)
        links.append({'request_id':r['request_id'],'decision_id':r['decision_id'],'stage':r['stage'],
            'core_call_index':r.get('core_call_index'),'agent_id':r.get('agent_id'),'role':r.get('role'),
            'model':r.get('model'),'model_version':r.get('version'),'prompt_fingerprint':r.get('prompt_fingerprint'),
            'input_fingerprint':r.get('input_fingerprint'),'schema_fingerprint':r.get('schema_fingerprint'),
            'response_fingerprint':r.get('response_fingerprint'),'request_status':r.get('status'),
            'error_code':r.get('error_code'),'pinned_model_digest':r.get('pinned_model_digest'),
            'observed_server_digest':r.get('observed_server_digest'),
            'identity_validation':('NOT_CHECKED' if r.get('observed_server_digest') in (None,'NOT_RECORDED') else
                'MATCHED' if r.get('pinned_model_digest')==r.get('observed_server_digest') else 'MISMATCH'),
            'minute':r['minute'],'kind':'action_proposal' if r.get('core_decision_sequence') is not None else 'logical_judgment',
            'core_decision_sequence':r.get('core_decision_sequence'),'action_id':r.get('action_id'),
            'execution_event_sequences':deepcopy(r.get('execution_event_sequences',[])),
            'model_proposal_action':r.get('model_proposal_action'),'actual_selected_action':decision['selected_action'] if decision else None,
            'selection_relation':('proposal_replaced_by_voi' if decision and r.get('model_proposal_action') and
                decision['actual_selector']=='explicit_outcome_voi' and r['model_proposal_action']!=decision['selected_action']
                else 'voi_selected_after_proposal' if decision and decision['actual_selector']=='explicit_outcome_voi'
                else 'core_action_proposal' if decision else 'no_action_execution'),
            'repair':r.get('repair',False),'repair_kind':r.get('repair_kind'),'repair_of_request_id':r.get('repair_of_request_id'),
            'retry':r.get('retry',False),'schema_validation':r.get('schema_validation'),
            'semantic_validation':deepcopy(r.get('specialty_evidence_validation'))})
    normalized=[]
    for h in handoffs:
        returned=next((r for r in requests if r['decision_id']==h.get('return_to_er_decision_id')),None)
        status=('completed' if h.get('specialty_evidence_validation',{}).get('semantic_validation')=='VALID' and
                returned and returned.get('core_validation')=='success' else
                'not_performed_'+h['route_not_selected'] if h.get('route_not_selected') in {'disabled','abstained'} else 'failed')
        normalized.append({'handoff_id':h['handoff_id'],'status':status,'legacy_status':h['handoff_status'],
            'source_decision_id':h['source_decision_id'],'specialty_response_id':h.get('specialty_response_id'),
            'return_to_er_decision_id':h.get('return_to_er_decision_id'),'request_minute':h['request_minute'],
            'response_minute':h.get('response_minute'),'specialty_evidence_validation':deepcopy(h.get('specialty_evidence_validation'))})
    return {'schema_version':'locdi.handoff_audit.v2','episode_id':episode['run_id'],
        'configuration_fingerprint':episode['configuration_fingerprint'],'episode_status':episode['result']['status'],
        'handoff_status':normalized[-1]['status'] if normalized else 'not_requested','handoffs':normalized,
        'request_attempt_count':len(requests),'actual_generation_request_count':actual_requests,
        'core_action_decision_count':len(decisions),'executed_action_decision_count':sum(bool(d['execution_event_sequences']) for d in decisions),
        'logical_judgment_request_count':sum(r['kind']=='logical_judgment' for r in links),
        'identity_check_count':identity_checks,'specialty_semantic_repair_count':sum(r.get('repair_kind')=='SPECIALTY_SEMANTIC' for r in requests),
        'core_repair_count':sum(r.get('repair_kind')=='CORE_ACTION_CONTRACT' for r in requests),'retry_count':sum(bool(r.get('retry')) for r in requests),
        'routing':deepcopy(routing),'core_routing_status':'disabled' if episode['configuration']['routing_mode']=='er_only' else 'configured',
        'legacy_core_routing_abstention_interpretation':'Do not equate selected_specialty=None under er_only with adaptive abstention',
        'requests':links,'memory':deepcopy(episode['memory']),'workflow':deepcopy(episode['workflow']),
        'terminal':deepcopy(episode['result']),'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}


def audit_saved_success(root=None):
    root=Path(root) if root is not None else ROOT/'outputs/development_handoff'/RUN_ID
    names=['reports/'+n+'.json' for n in ('preflight','started','pilot','handoff','runtime')]+['episodes/Case001.json','metrics/summary.jsonl']
    docs={n:load(root/n) for n in names};pilot=docs['reports/pilot.json'];a=docs['reports/handoff.json']
    ep=docs['episodes/Case001.json'];runtime=docs['reports/runtime.json'];summary=docs['metrics/summary.jsonl']['episodes'][0]
    pre=docs['reports/preflight.json'];plan=pre['plan'];checks={};rows=a['requests']
    checks['four_independent_requests']=(len(rows)==len({r['request_id'] for r in rows})==len({r['decision_id'] for r in rows})==4
        and [r['stage'] for r in rows]==['ER_INITIAL_DECISION','SPECIALTY_ASSESSMENT','ER_REASSESSMENT','ER_REASSESSMENT'])
    checks['counters']=(pilot['llm_execution']['request_count']==a['request_count']==ep['llm_execution']['request_count']==summary['llm_execution']['request_count']==4
        and a['identity_check_count']==pilot['llm_execution']['identity_check_count']==ep['llm_execution']['identity_check_count']==4
        and [r['generation_count_after'] for r in rows]==[1,2,3,4] and [r['identity_check_count_after'] for r in rows]==[1,2,3,4])
    checks['plan_fingerprint']=digest(plan)==pre['plan_fingerprint']==pilot['plan_fingerprint']==a['plan_fingerprint']
    checks['started_record']=docs['reports/started.json']['plan_fingerprint']==pre['plan_fingerprint']
    checks['registry_and_prompts']=all(all(r[k]==plan['stages'][r['stage']][k] for k in ('agent_id','model','version'))
        and r['prompt_fingerprint']==plan['stages'][r['stage']]['system_prompt_fingerprint'] for r in rows)
    checks['pinned_observed_digests']=all(r['pinned_model_digest']==r['observed_server_digest']==plan['model_identifier'] for r in rows)
    stable=('request_id','decision_id','stage','minute','input_fingerprint','prompt_fingerprint','schema_fingerprint','response_fingerprint','generation_count_after','identity_check_count_after')
    checkpoints=[];checkpoint_matches=True
    for r in rows:
        before=load(root/('private/'+r['request_id']+'_before.json'));after=load(root/('private/'+r['request_id']+'_after.json'))
        checkpoint_matches &= all(after['request'][k]==r[k] for k in stable)
        checkpoint_matches &= all(before['request'][k]==r[k] for k in ('request_id','decision_id','stage','minute','input_fingerprint','prompt_fingerprint','schema_fingerprint'))
        checkpoints.extend([r['request_id']+'_before',r['request_id']+'_after'])
        if 'specialty_evidence_validation' in r:
            semantic=load(root/('private/'+r['request_id']+'_semantic.json'))
            checkpoint_matches &= semantic['request']['specialty_evidence_validation']==r['specialty_evidence_validation']
            checkpoints.append(r['request_id']+'_semantic')
    checks['checkpoint_correspondence']=bool(checkpoint_matches)
    decisions=ep['workflow']['decisions'];events={e['sequence']:e for e in ep['workflow']['events']}
    checks['request_action_event_links']=len(decisions)==2 and all(
        any(r.get('core_decision_sequence')==d['sequence'] and r.get('execution_event_sequences')==d['execution_event_sequences'] and
            r.get('action_id')=='Action%04d'%d['sequence'] for r in rows) and
        d['execution_success'] is True and all(i in events and events[i]['minute']==d['minute'] for i in d['execution_event_sequences']) for d in decisions)
    checks['logical_judgments_have_no_execution']=all(r['action_id'] is None and not r['execution_event_sequences'] for r in rows[:2])
    checks['exports_correspond']=(runtime['workflow']==ep['workflow'] and runtime['terminal']==ep['result'] and runtime['memory']==ep['memory']
        and a['configuration_fingerprint']==ep['configuration_fingerprint']==summary['configuration_fingerprint']
        and summary['action_counts']==ep['result']['action_counts'])
    checks['terminal']=pilot['status']==pilot['core_status']==ep['result']['status']==summary['status']=='completed' and pilot['export_status']=='EXPORTED'
    checks['handoff_completed']=pilot['handoff_outcome']=='handoff_completed' and pilot['sequential_handoff_verified'] is True
    checks['memory_cutoff']=all(r['cutoff_compliant'] is True for r in ep['memory']['retrievals'])
    checks['empty_evidence_valid']=all(rows[1]['specialty_evidence_validation'][k]==v for k,v in
        {'semantic_validation':'VALID','reported_certainty':'unresolved','evidence_count':0,'citation_count':0}.items())
    checks['no_failure_repair_retry']=not pilot['blocker_codes'] and not any(r['retry'] or r['repair'] or r['status']!='response_received' for r in rows)
    timeline=[{'event':e['event_type'],'minute':e['minute'],'sequence':e['sequence']} for e in events.values()
        if e['event_type'] in {'test_ordered','test_started','result_received','case_completed'}]
    checks['recorded_result_order']=[e['minute'] for e in timeline]==[3,4,9,13] and rows[-1]['minute']>=9
    overlay=project_audit(ep,rows,a['handoffs'],a['routing'],actual_requests=a['request_count'],identity_checks=a['identity_check_count'])
    hashes={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
    return {'schema_version':'locdi.phase210.saved_success_audit.v1','run_id':RUN_ID,'status':'PASS' if all(checks.values()) else 'FAIL',
        'checks':checks,'file_count':len(hashes),'files_fingerprint':digest(hashes),'audit_v2':overlay,'source_release_event_timeline':timeline,
        'elapsed_simulation_minutes':ep['result']['elapsed_simulation_minutes'],'finalization_event_minute':timeline[-1]['minute'],
        'diagnosis_unresolved':ep['result']['diagnosis_unresolved'],'raw_wire_payload_validation':'NOT_CHECKED_NOT_SAVED',
        'clinical_accuracy':'NOT_EVALUATED','new_model_generations':0,'utility_fitting':0,'clinical_verdicts_generated':0}
