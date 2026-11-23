"""Sequential five-case pilot with one client, immutable checkpoints and no resume."""
from copy import deepcopy
import hashlib
import json
from app.locdi.llm_setup import ROOT,digest
from app.locdi.episode_exporter import _write
from app.locdi.agentclinic_doctor_client import FakeDoctorClient,LiveDoctorClient,create_live_client,build_request
from app.locdi.agentclinic_doctor_runtime import run_doctor_scenario
from app.locdi.agentclinic_doctor_exporter import export_doctor_episode
from app.locdi.agentclinic_turn_runtime import TurnOptions
from app.locdi.agentclinic_scenario_adapter import load_scenarios
from app.locdi.agentclinic_live_preflight import SOURCE,proposal
from app.locdi.agentclinic_pilot_readiness import output_paths,readiness,PinnedPilotApproval,prepare_plan


class PilotStop(RuntimeError):
    def __init__(self,code):
        self.code=code
        super().__init__(code)


class _Capture:
    def __init__(self,transport):
        self.original=transport
        self.calls=0
        self.response=None
        self.response_hash=None
        self.error_type=None

    def __getattr__(self,name):
        return getattr(self.original,name)

    def reset(self):
        self.response=self.response_hash=self.error_type=None

    def chat_json(self,system,user,model=None,*,schema=None):
        self.calls+=1
        try:
            result=self.original.chat_json(system,user,model=model,schema=schema)
            self.response=result
            if isinstance(result,str):self.response_hash=hashlib.sha256(result.encode('utf8')).hexdigest()
            return result
        except Exception as exc:
            self.error_type=type(exc).__name__
            raise
        finally:
            raw=getattr(self.original,'last_response_content',None)
            if self.response is None and isinstance(raw,str):
                self.response=raw;self.response_hash=hashlib.sha256(raw.encode('utf8')).hexdigest()


class _Monitor:
    def __init__(self,client,public,private,plan,counter=None,context_tokens=32768,authority=None):
        self.client,self.public,self.private,self.plan=client,public,private,plan
        self.counter,self.context_tokens=counter,context_tokens
        self.authority=authority
        self.capture=_Capture(client.transport);client.transport=self.capture
        self.original_choose=client.choose_action
        client.choose_action=self.choose
        self.ordinal=0;self.attempts=0;self.total_input_tokens=0
        self.last_code=None;self.current=[]
        self.context_observer=None;self.prior_generation_count=0
        if isinstance(client,LiveDoctorClient) and authority is not None:
            from app.locdi.external_reference_readiness import pinned_private_json
            record=pinned_private_json(authority.path,authority.sha256)
            if 'ablation_evidence_path' in record:
                from app.locdi.agentclinic_context_bootstrap import observe_context
                self.context_observer=lambda:observe_context(self.plan['doctor_plan']['definition']['legacy_config'])
            if 'approval_basis_type' in record:
                from app.locdi.agentclinic_context_bootstrap import observe_context,verify_bootstrap_receipt
                receipt=verify_bootstrap_receipt(record['bootstrap_receipt_path'],record['bootstrap_receipt_sha256'])
                self.prior_generation_count=receipt['generation_count']
                self.context_observer=lambda:observe_context(self.plan['doctor_plan']['definition']['legacy_config'])
                client.request_budget=min(client.request_budget,50-self.prior_generation_count)

    def counts(self):
        return {'doctor_request_attempts':self.client.request_count,
            'generation_count':getattr(self.capture.original,'request_count',0) if isinstance(self.client,LiveDoctorClient) else 0,
            'identity_count':getattr(self.capture.original,'identity_check_count',0),
            'fake_transport_calls':self.capture.calls if isinstance(self.client,FakeDoctorClient) else 0}

    def model_identity(self):
        return {'expected':self.plan['doctor_plan']['definition']['legacy_config']['model_identifier'],
            'observed':getattr(self.capture.original,'last_observed_model_identifier',None),
            'mode':'LIVE_PER_REQUEST_ADAPTER_CHECK' if isinstance(self.client,LiveDoctorClient) else 'TEST_ONLY_NO_SERVER'}

    def _code(self,exc,payload):
        if isinstance(exc,PilotStop):return exc.code
        if self.capture.error_type=='ModelIdentityError':return 'IDENTITY_MISMATCH'
        if self.capture.error_type and 'timeout' in self.capture.error_type.casefold():return 'FAILED_TIMEOUT'
        if self.capture.response is not None:
            try:
                a=json.loads(self.capture.response)
                if isinstance(a,dict) and a.get('action') in ('ASK','EXAMINE','TEST'):
                    if a.get('target') not in [t['target'] for t in payload['catalog'][a['action']]]:
                        return 'SOURCE_ACTION_ERROR'
            except Exception:pass
            return 'JSON_RESPONSE_INVALID'
        if self.capture.error_type=='LLMResponseError':return 'JSON_RESPONSE_INVALID'
        return 'FAILED_CLIENT_OR_CONTRACT'

    def choose(self,observation):
        self.attempts+=1;self.capture.reset();self.last_code=None
        request_id='Request%04d'%self.attempts
        self.current.append((request_id,observation['turn_index']+1))
        before={'request_id':request_id,'ordinal':self.ordinal,'turn_index':observation['turn_index'],
            'state':'BEFORE','plan_fingerprint':digest(self.plan),'prompt_sha256':self.plan['doctor_plan']['definition']['prompt_sha256'],
            'config_fingerprint':digest(self.plan['doctor_plan']['definition']['legacy_config']),
            'model_identity':self.model_identity(),'counters':self.counts()}
        payload={};action=None;failure=None;result=None;input_tokens=None;context_observation=None
        generation_before=self.counts()['generation_count']
        try:
            system,payload,_=build_request(observation,prompt_profile=self.plan['doctor_plan']['definition'].get('prompt_profile'))
            user=json.dumps(payload,ensure_ascii=False,sort_keys=True)
            before.update(payload=payload,request_sha256=hashlib.sha256(user.encode('utf8')).hexdigest(),
                          system_sha256=hashlib.sha256(system.encode('utf8')).hexdigest(),
                          utf8_bytes=len(system.encode('utf8'))+len(user.encode('utf8')))
            if self.counter is not None:input_tokens=self.counter.count(system,user)
            before['input_tokens']=input_tokens
            # Persist BEFORE even a denied request; inability to persist prevents the call.
            _write(before,'checkpoints/'+request_id+'_before.json',self.private,'checkpoints')
            if isinstance(self.client,LiveDoctorClient):
                if not isinstance(self.authority,PinnedPilotApproval):raise PilotStop('PILOT_APPROVAL_REQUIRED')
                self.authority.verify(prepare_plan(self.plan['namespace'],self.plan['doctor_plan']['definition']),load_tokenizer=False)
                if self.plan['doctor_plan']['definition'].get('prompt_profile') is not None:
                    from app.locdi.agentclinic_prompt_ablation import verify_execution_order
                    verify_execution_order(self.authority,self.plan)
            if self.client.request_count>=self.client.request_budget:raise PilotStop('BUDGET_EXCEEDED')
            if before['utf8_bytes']>self.client.max_prompt_bytes:raise PilotStop('CONTEXT_LIMIT_EXCEEDED')
            if isinstance(self.client,LiveDoctorClient) and input_tokens is None:raise PilotStop('CONTEXT_UNVERIFIED')
            if input_tokens is not None:
                if type(input_tokens) is not int or input_tokens<0:raise PilotStop('CONTEXT_UNVERIFIED')
                if input_tokens+1024+512>self.context_tokens:raise PilotStop('CONTEXT_LIMIT_EXCEEDED')
                if self.total_input_tokens+input_tokens>self.plan['max_total_input_tokens']:raise PilotStop('BUDGET_EXCEEDED')
                self.total_input_tokens+=input_tokens
            action=self.original_choose(observation)
            result=action
        except Exception as exc:
            failure=exc
            self.last_code='EXPORT_FAILED' if isinstance(exc,OSError) else self._code(exc,payload)
        # Observe synchronously even after a failed generation. Never execute the
        # selected source action or allow another request on a context mismatch.
        if self.context_observer and self.counts()['generation_count']>generation_before:
            try:
                context_observation=self.context_observer()
                if context_observation['status']!='VALID':raise PilotStop('ACTUAL_CONTEXT_MISMATCH')
            except Exception as exc:
                failure=exc;result=None;self.last_code='ACTUAL_CONTEXT_UNVERIFIED_OR_MISMATCH'
        after={'request_id':request_id,'ordinal':self.ordinal,'turn_index':observation['turn_index'],
            'state':'AFTER_SUCCESS' if failure is None else 'AFTER_FAILED','error_code':self.last_code,
            'plan_fingerprint':digest(self.plan),'prompt_sha256':before['prompt_sha256'],
            'config_fingerprint':before['config_fingerprint'],'request_sha256':before.get('request_sha256'),
            'response_sha256':self.capture.response_hash,
            'response_hash_scope':'ADAPTER_RETURNED_CONTENT_NOT_RAW_HTTP; null if adapter did not return content',
            'action':deepcopy(action),'model_identity':self.model_identity(),'counters':self.counts(),
            'input_tokens':input_tokens,'total_measured_input_tokens':self.total_input_tokens,
            'response_content':self.capture.response,'context_observation':context_observation,
            'prior_bootstrap_generation_count':self.prior_generation_count,
            'experiment_generation_count':self.prior_generation_count+self.counts()['generation_count']}
        try:_write(after,'checkpoints/'+request_id+'_after.json',self.private,'checkpoints')
        except Exception:
            self.last_code='EXPORT_FAILED'
            raise PilotStop('EXPORT_FAILED') from None
        if failure is not None:raise PilotStop(self.last_code) from None
        return result

    def link_events(self,episode):
        for request_id,turn in self.current:
            event=next((e for e in episode['trace'] if e['turn_index']==turn),None)
            row={'request_id':request_id,'ordinal':self.ordinal,'turn_index':turn,
                 'event':deepcopy(event),'episode_fingerprint':digest(episode)}
            _write(row,'checkpoints/'+request_id+'_event.json',self.private,'checkpoints')
            public={'request_id':request_id,'ordinal':self.ordinal,'turn_index':turn,
                    'event_status':event['status'] if event else 'NOT_EXECUTED',
                    'action':event['action'] if event else None,'target':event['target'] if event else None,
                    'observation_fingerprint':digest(event['observation']) if event and event['observation'] is not None else None,
                    'private_link_fingerprint':digest(row)}
            _write(public,'checkpoints/'+request_id+'_link.json',self.public,'checkpoints')


def _create_outputs(namespace):
    public,private=output_paths(namespace)
    if public.exists() or private.exists():raise ValueError('OUTPUT_NAMESPACE_EXISTS_NO_RESUME')
    # Exclusive mkdir is a run lock. Partial directory creation is deliberately retained.
    public.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    return public,private


def _run(namespace,client,plan,*,counter=None,max_turns=10,context_tokens=32768,authority=None):
    public,private=_create_outputs(namespace)
    scenarios=load_scenarios(SOURCE).scenarios[:5]
    _write(plan,'reports/plan.json',public,'reports')
    _write({'ordinals':[1,2,3,4,5],'scenario_refs':[s.scenario_id for s in scenarios],
            'selection_rule':plan['selection_rule'],'scoring_status':'NOT_EVALUATED'},
           'reports/cohort.json',private,'reports')
    _write({'status':'STARTED','mode':'LIVE' if isinstance(client,LiveDoctorClient) else 'TEST_ONLY_FAKE',
            'plan_fingerprint':digest(plan),'resume_supported':False},'reports/started.json',public,'reports')
    monitor=_Monitor(client,public,private,plan,counter,context_tokens,authority)
    rows=[{'ordinal':i,'scenario_ref':s.scenario_id,'status':'NOT_EXECUTED','episode_export':'NOT_ATTEMPTED'}
          for i,s in enumerate(scenarios,1)]
    stop=None
    try:
        for index,s in enumerate(scenarios):
            row=rows[index];monitor.ordinal=index+1;monitor.current=[];monitor.last_code=None
            episode=None
            try:
                episode=run_doctor_scenario(s,client,options=TurnOptions(max_turns,max_turns))
                r=episode['result']
                state=('UNRESOLVED' if r['diagnosis_unresolved'] else 'COMPLETED') if r['status']=='completed' else r['status'].upper()
                row.update(status=monitor.last_code or state,terminal_status=r['status'],
                    diagnosis_unresolved=r['diagnosis_unresolved'],turn_count=r['turn_count'],action_count=r['action_count'],
                    doctor_request_attempts=episode['doctor_request_attempts'],
                    action_counts={k:sum(t['action']==k for t in episode['trace']) for k in ('ASK','EXAMINE','TEST','DIAGNOSE')})
                monitor.link_events(episode)
                export_doctor_episode(episode,public_root=public,private_root=private,name='Scenario%03d'%(index+1))
                row['episode_export']='EXPORTED'
            except Exception as exc:
                row.update(status='EXPORT_FAILED' if isinstance(exc,OSError) or monitor.last_code=='EXPORT_FAILED' else 'FAILED',
                           episode_export='FAILED_OR_PARTIAL',error_code='EXPORT_FAILED' if episode is not None else 'RUNTIME_FAILED')
            _write({'row':row,'counters':monitor.counts()},'reports/Scenario%03d_status.json'%(index+1),public,'reports')
            if row['status'] not in {'COMPLETED','UNRESOLVED'}:
                stop=row['status'];break
    finally:
        if isinstance(client,LiveDoctorClient):client.close()
    summary={'schema_version':'agentclinic.pilot_summary.v1','mode':'LIVE' if isinstance(client,LiveDoctorClient) else 'TEST_ONLY_FAKE',
        'status':stop or ('UNRESOLVED' if any(r['status']=='UNRESOLVED' for r in rows) else 'COMPLETED'),
        'planned_denominator':5,'executed_scenarios':sum(r['status']!='NOT_EXECUTED' for r in rows),
        'not_executed':sum(r['status']=='NOT_EXECUTED' for r in rows),'rows':rows,
        'counters':monitor.counts(),'checkpoint_attempts':monitor.attempts,
        'measured_input_tokens':monitor.total_input_tokens if counter is not None else None,
        'retry_count':0,'repair_count':0,'resume_supported':False,
        'scoring_policy':'agentclinic.lexical_scoring.v1','benchmark_scores':'NOT_EVALUATED',
        'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}
    _write(summary,'metrics/summary.json',public,'metrics')
    return summary


def run_fake(namespace,transport,*,generation_budget=50,max_turns=10,max_prompt_bytes=16384,
             token_counter=None,context_tokens=32768):
    """Explicit Fake-only API; no config option can substitute Fake for Live."""
    if type(max_turns) is not int or not 1<=max_turns<=10:raise ValueError('FAKE_TURN_LIMIT')
    plan=prepare_plan(namespace)
    plan=dict(plan,mode='TEST_ONLY_FAKE',fixture_limits={'generation_budget':generation_budget,
        'max_turns':max_turns,'max_prompt_bytes':max_prompt_bytes,'context_tokens':context_tokens})
    client=FakeDoctorClient(transport,request_budget=generation_budget,max_prompt_bytes=max_prompt_bytes)
    return _run(namespace,client,plan,counter=token_counter,max_turns=max_turns,context_tokens=context_tokens)


def run_live(namespace,*,authority=None,execute=False,allow_llm=False,definition=None):
    if execute is not True or allow_llm is not True or not isinstance(authority,PinnedPilotApproval):
        raise ValueError('EXPLICIT_PINNED_PILOT_APPROVAL_REQUIRED')
    definition=definition or proposal()
    report=readiness(namespace,authority,definition=definition)
    if not report['LIVE_PILOT_READY']:raise ValueError('LIVE_PILOT_PREFLIGHT_BLOCKED')
    doctor_authority,counter,_=authority.verify(report['plan'])
    client=create_live_client(definition,execute=True,allow_llm=True,authority=doctor_authority)
    try:return _run(namespace,client,report['plan'],counter=counter,authority=authority)
    finally:client.close()
