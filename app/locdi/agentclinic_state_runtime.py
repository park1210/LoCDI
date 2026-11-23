"""External action-availability orchestration over the unmodified Source executor.

Rejected proposals never enter Source replay. Their decision turns and model
requests are retained separately, never converted to another action or repaired.
"""
from copy import deepcopy
import hashlib,json
from app.locdi.llm_setup import digest
from app.locdi.agentclinic_state_actions import ActionAvailability,VERSION,KINDS
from app.locdi.agentclinic_scenario_adapter import action_catalog,load_scenarios
from app.locdi.agentclinic_turn_runtime import run_scenario,TurnOptions
from app.locdi.agentclinic_doctor_runtime import _Replay
from app.locdi.agentclinic_doctor_client import FakeDoctorClient,LiveDoctorClient,build_request,action_schema,validate_action,PROMPT_SHA256
from app.locdi.agentclinic_live_preflight import SOURCE,preflight
from app.locdi.agentclinic_live_pilot_runner import _Capture
from app.locdi.run_simulation import _offline_network_guard

RUNTIME_VERSION='agentclinic.state_action_orchestration.v1'


def source_prefix(scenario,actions,options):
    replay=_Replay(actions);episode=run_scenario(scenario,replay,options=options)
    waiting=replay.observation is not None
    if waiting:
        # _Replay's private pause is not a real failed model/source event.
        episode['trace']=episode['trace'][:-1]
        episode['result'].update(status='incomplete',completed=False,stop_reason='AWAITING_POLICY_DECISION',error_type=None,
            turn_count=len(episode['trace']),action_count=len(episode['trace']),request_count=len(actions))
        episode['result']['invalid_action_count']=sum(t['status'] in {'INVALID','UNAVAILABLE','UNSUPPORTED'} for t in episode['trace'])
    return episode,deepcopy(replay.observation)


def verify_full_source_binding(doctor,observation):
    if not isinstance(doctor,LiveDoctorClient):return
    current=preflight(doctor.definition)
    if current['static_blockers'] or digest(current['plan'])!=doctor.plan_fingerprint:raise ValueError('LIVE_BINDING_CHANGED')
    doctor.authority.verify(current['plan'])
    binding=current['plan']['scenarios'].get(observation['scenario_ref'])
    if not binding or digest(observation['objective'])!=binding['objective_fingerprint'] or digest(observation['catalog'])!=binding['catalog_fingerprint']:
        raise ValueError('LIVE_ORIGINAL_CATALOG_BINDING')


def run_state_episode(scenario,doctor,*,enabled,max_decisions=10,counter=None,checkpoint=None,context_observer=None):
    if not isinstance(doctor,(LiveDoctorClient,FakeDoctorClient)):raise TypeError('GATED_DOCTOR_REQUIRED')
    if type(max_decisions) is not int or not 1<=max_decisions<=10:raise ValueError('POLICY_DECISION_LIMIT')
    live=isinstance(doctor,LiveDoctorClient)
    if getattr(doctor,'prompt_profile',None) is not None:raise ValueError('STATE_POLICY_REQUIRES_UNCHANGED_V1_PROMPT')
    if live:
        canonical=next((s for s in load_scenarios(SOURCE).scenarios if s.scenario_id==scenario.scenario_id),None)
        if scenario!=canonical:raise ValueError('LIVE_SOURCE_PROJECTION_CHANGED')
        if counter is None or context_observer is None or checkpoint is None:raise ValueError('LIVE_POLICY_OBSERVABILITY_REQUIRED')
    policy=ActionAvailability(scenario.scenario_id,action_catalog(scenario),enabled=enabled)
    options=TurnOptions(10,10);actions=[];policy_trace=[];history=[]
    request_start=doctor.request_count;audit_start=len(doctor.audit)
    original=doctor.transport;capture=_Capture(original);doctor.transport=capture
    generation_start=getattr(original,'request_count',0) if live else 0
    episode,_=source_prefix(scenario,actions,options);stop=None;error=None
    try:
        for decision in range(1,max_decisions+1):
            episode,observation=source_prefix(scenario,actions,options)
            if observation is None:break
            if doctor.request_count>=doctor.request_budget:
                stop='GLOBAL_REQUEST_BUDGET_EXHAUSTED';break
            observation['turn_index']=decision-1
            observation['history']=deepcopy(history)
            observation['remaining_requests']=min(max_decisions-decision+1,doctor.request_budget-doctor.request_count)
            projected=deepcopy(observation);projected['catalog']=policy.available_catalog()
            before_consumed=policy.snapshot();capture.reset();proposal=None;failure=None;ctx=None
            row={'turn_index':decision,'source_turn_index':len(actions)+1,'proposed_action':None,'proposed_target':None,
                 'action_available':False,'executed_action':None,'action_status':'PENDING',
                 'consumed_before':before_consumed,'consumed_after':before_consumed,'new_observation_count':0,
                 'rejection_reason':None,'repeated_proposal':False,'source_event_index':None,
                 'remaining_case_requests_before':observation['remaining_requests']}
            before={'turn_index':decision,'source_turn_index':len(actions)+1,'scenario_ref':scenario.scenario_id,
                    'consumed_before':before_consumed,'original_catalog_fingerprint':digest(observation['catalog'])}
            calls_before=capture.calls;schema_rejected=False
            try:
                verify_full_source_binding(doctor,observation)
                system,payload,schema=build_request(projected)
                user=json.dumps(payload,ensure_ascii=False,sort_keys=True)
                count=counter.count(system,user) if counter else None
                before.update(payload=payload,request_sha256=hashlib.sha256(user.encode()).hexdigest(),
                    system_sha256=hashlib.sha256(system.encode()).hexdigest(),input_tokens=count,
                    utf8_bytes=len(system.encode())+len(user.encode()))
                if checkpoint:checkpoint('before',before)
                if before['utf8_bytes']>doctor.max_prompt_bytes or (count is not None and count+1024+512>32768):
                    raise ValueError('CONTEXT_LIMIT_EXCEEDED')
                if live:
                    # Same old v1 client and pinned Doctor gate, with the original
                    # full source binding verified BEFORE this external projection.
                    proposal=doctor._choose(projected)
                else:
                    with _offline_network_guard():proposal=doctor._choose(projected)
            except Exception as exc:
                failure=exc
                # A consumed-ID proposal can be rejected by the available-target
                # JSON schema first. Preserve that audit error, explicitly classify
                # the original valid source action, and continue as a NEW decision.
                try:
                    raw=validate_action(capture.response,action_schema(policy.catalog))
                    available,reason=policy.classify(raw)
                    if enabled and not available and reason=='ALREADY_OBSERVED':
                        proposal=raw;failure=None;schema_rejected=True
                except (ValueError,TypeError):pass
            if context_observer and capture.calls>calls_before:
                try:
                    ctx=context_observer()
                    if ctx['status']!='VALID':raise ValueError('ACTUAL_CONTEXT_MISMATCH')
                except Exception as exc:failure=exc;stop='ACTUAL_CONTEXT_UNVERIFIED_OR_MISMATCH'
            if proposal is not None:
                kind,target=proposal.get('action'),proposal.get('target')
                row.update(proposed_action=kind,proposed_target=target if target in policy.ids.get(kind,set()) else None,
                    repeated_proposal=(kind,target) in policy.consumed)
                available,reason=policy.classify(proposal);row['action_available']=available
                if failure is None and not available:
                    row.update(action_status='REJECTED',rejection_reason=reason)
                elif failure is None:
                    prior_refs={e['observation_ref'] for e in episode['trace'] if e['observation_ref']}
                    actions.append(deepcopy(proposal));episode,_=source_prefix(scenario,actions,options)
                    event=episode['trace'][-1];policy.observe(event)
                    row.update(executed_action=kind if event['status']=='EXECUTED' else None,
                        action_status=event['status'],source_event_index=len(episode['trace'])-1,
                        consumed_after=policy.snapshot(),
                        new_observation_count=int(bool(event['observation_ref']) and event['observation_ref'] not in prior_refs),
                        rejection_reason=event['error_code'])
            if failure is not None:
                error=type(failure).__name__;stop=stop or ('EXPORT_FAILED' if isinstance(failure,OSError) else 'DOCTOR_CLIENT_OR_CONTRACT_FAILED')
                row.update(action_status='FAILED',rejection_reason=stop)
            row['schema_rejected_consumed_proposal']=schema_rejected
            policy_trace.append(row)
            history.append({'request_id':'P%04d'%decision,'turn_index':decision,'action':row['proposed_action'],
                'target':row['proposed_target'],'validation':'VALID' if row['action_available'] else 'INVALID',
                'status':row['action_status'],'error_code':row['rejection_reason'],
                'observation_ref':episode['trace'][-1]['observation_ref'] if row['source_event_index'] is not None else None,
                'repeat_of':episode['trace'][-1]['repeat_of'] if row['source_event_index'] is not None else None})
            # With no rejection, baseline history preserves the old source IDs.
            if row['source_event_index'] is not None:
                event=episode['trace'][row['source_event_index']]
                history[-1]={k:deepcopy(event[k]) for k in ('request_id','turn_index','action','target','validation','status','error_code','observation_ref','repeat_of')}
            after={'turn_index':decision,'policy_event':row,'response_content':capture.response,
                'response_sha256':capture.response_hash,'error_type':error,'context_observation':ctx,
                'doctor_request_count':doctor.request_count,'generation_count':getattr(original,'request_count',0) if live else 0,
                'identity_count':getattr(original,'identity_check_count',0),'schema_rejected_consumed_proposal':schema_rejected}
            if checkpoint:
                try:checkpoint('after',after)
                except Exception:stop='EXPORT_FAILED';error='PolicyExportError';break
            if failure is not None or episode['result']['status']=='completed':break
        if error or episode['result']['status']!='completed':
            natural=episode['result']['stop_reason']=='TURN_BUDGET_EXHAUSTED'
            episode['result'].update(status='failed' if error else 'incomplete',completed=False,
                stop_reason=stop or ('TURN_BUDGET_EXHAUSTED' if natural else 'POLICY_DECISION_BUDGET_EXHAUSTED'),error_type=error)
    finally:doctor.transport=original
    episode.update(mock_client=not live,prompt_fingerprint=PROMPT_SHA256,
        doctor_audit=deepcopy(doctor.audit[audit_start:]),doctor_request_attempts=doctor.request_count-request_start,
        real_generation_calls=getattr(original,'request_count',0)-generation_start if live else 0)
    episode['config'].update(mode='GATED_LIVE_DOCTOR' if live else 'OFFLINE_JSON_DOCTOR',
        runtime_version=RUNTIME_VERSION,action_availability_policy=VERSION if enabled else 'baseline.original_catalog.v1',
        max_policy_decisions=max_decisions,doctor_global_request_budget=doctor.request_budget)
    episode['config_fingerprint']=digest(episode['config'])
    if live:episode['model_metadata']=original.public_metadata()
    sidecar={'schema_version':'agentclinic.action_policy_trace.v1','policy_version':VERSION if enabled else 'baseline.original_catalog.v1',
        'scenario_ref':scenario.scenario_id,'decision_turn_count':len(policy_trace),'events':policy_trace,
        'episode_fingerprint':digest(episode),'prompt_sha256':PROMPT_SHA256,
        'source_runtime_unchanged':True,'model_internal_reasoning':'NOT_OBSERVED'}
    return episode,sidecar
