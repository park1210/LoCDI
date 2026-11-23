"""Offline sequential judgments around the real ER policy, never a physical Consult.

Core owns observations, clock, action validation and execution. This boundary
adds no source facts, clinical delays, routing formula or real-model approval.
"""
from copy import deepcopy
from dataclasses import replace
import json
from jsonschema import Draft202012Validator
from app.core.agent_registry import AgentRegistry
from app.core.longitudinal_memory import timestamp
from app.llm.prompt_loader import specialty_system_prompt
from app.locdi.llm_setup import digest
from app.locdi.run_simulation import OfflineClient,run_case,_configuration
from app.locdi.case_adapter import adapt_case
from app.locdi.episode_exporter import ExportSession,build_episode,export_summary,_write
from app.locdi.specialty_evidence_contract import projection,validate_response,EvidenceScopeError,PARENT_ERROR,REPAIRABLE_CODES,REPAIR_VERSION,repair_payload

FORBIDDEN={'gold','gold_department','evaluation','task_annotations','provenance','source_event_store',
           'current_encounter_events','data_replay','reference','reference_candidates','logical_specialty_assessment'}


class HandoffStopped(RuntimeError):
    pass


def inspect_scope(value):
    if isinstance(value,dict):
        if set(value)&FORBIDDEN:raise ValueError('INFORMATION_SCOPE_VIOLATION')
        for v in value.values():inspect_scope(v)
    elif isinstance(value,list):
        for v in value:inspect_scope(v)


def strict_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('DUPLICATE_RESPONSE_KEY')
            result[k]=v
        return result
    def invalid(value):raise ValueError('NONFINITE_RESPONSE')
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


class SequentialEngine:
    """Shared orchestration only; concrete clients own transport authorization."""
    def __init__(self,transport,registry,*,episode_id,patient_id,visit_id,cutoff,budget,
                 routing='disabled',static_department=None,specialty_semantic_repairs=0):
        if type(budget) is not int or not 1<=budget<=100:raise ValueError('INVALID_GLOBAL_BUDGET')
        if routing not in {'disabled','static_development','adaptive'}:raise ValueError('INVALID_HANDOFF_ROUTING')
        if type(specialty_semantic_repairs) is not int or specialty_semantic_repairs not in (0,1):raise ValueError('INVALID_SPECIALTY_REPAIR_LIMIT')
        if routing=='adaptive' and registry.utility is None:raise ValueError('ROUTING_UTILITY_MISSING')
        if routing=='static_development':
            try:registry.get('specialty',static_department)
            except ValueError:raise ValueError('SPECIALTY_UNAVAILABLE') from None
        elif static_department is not None:raise ValueError('STATIC_TARGET_SCOPE_MISMATCH')
        self.transport=transport;self.registry=registry;self.episode_id=episode_id
        self.patient_id=patient_id;self.visit_id=visit_id;self.cutoff=timestamp(cutoff)
        self.budget=budget;self.routing=routing;self.static_department=static_department
        self.specialty_semantic_repairs=specialty_semantic_repairs
        self.requests=[];self.handoffs=[];self.core_calls=[];self.routing_records=[]
        self.failure=None;self.initial_done=False;self._completed=None

    def _stop(self,code):
        self.failure=code
        if self.handoffs and self.handoffs[-1]['handoff_status'] in {'handoff_requested','specialty_completed'}:
            self.handoffs[-1]['handoff_status']='handoff_failed'
            self.handoffs[-1]['failure_code']=code
        raise HandoffStopped(code)

    def _dispatch(self,stage,payload,system,binding,schema):
        return self.transport.chat_json(system,json.dumps(payload,sort_keys=True,ensure_ascii=False),model=binding['model'],schema=schema)

    def _failure_code(self,exc):
        return 'TIMEOUT' if isinstance(exc,TimeoutError) else 'SCHEMA_INVALID' if isinstance(exc,(ValueError,TypeError)) or type(exc).__name__=='ValidationError' else 'CLIENT_ERROR'

    def _record_specialty_evidence(self,row,summary):
        if row is not None:row['specialty_evidence_validation']=deepcopy(summary)
        if self.handoffs:self.handoffs[-1]['specialty_evidence_validation']=deepcopy(summary)

    def _request(self,stage,payload,system,binding,schema,*,core_index,repair=False,validate=True,
                 repair_kind=None,repair_of_request_id=None):
        if self.failure:raise HandoffStopped(self.failure)
        if len(self.requests)>=self.budget:self._stop('GLOBAL_REQUEST_BUDGET_EXHAUSTED')
        row={'request_id':'Request{:04d}'.format(len(self.requests)+1),
             'decision_id':'Decision{:04d}'.format(len(self.requests)+1),'stage':stage,
             'episode_id':self.episode_id,'core_call_index':core_index,'minute':payload['observation']['elapsed_minutes'],
             'agent_id':binding['agent_id'],'role':binding['role'],'model':binding['model'],'version':binding['version'],
             'prompt_fingerprint':digest(system),'input_fingerprint':digest(payload),'schema_fingerprint':digest(schema),'status':'started',
             'pinned_model_digest':'NOT_RECORDED','observed_server_digest':'NOT_RECORDED',
             'retry':False,'repair':repair,'response_fingerprint':None,'core_decision_sequence':None,
             'action_id':None,'execution_event_sequences':[]}
        row['repair_of_request_id']=self.core_calls[-2].get('action_request_id') if repair and len(self.core_calls)>1 else None
        row['repair_kind']=repair_kind or ('CORE_ACTION_CONTRACT' if repair else None)
        if repair_of_request_id is not None:row['repair_of_request_id']=repair_of_request_id
        self.requests.append(row)
        try:
            raw=self._dispatch(stage,payload,system,binding,schema)
            if not isinstance(raw,str):raise ValueError('RESPONSE_NOT_JSON_STRING')
            row['response_fingerprint']=digest(raw)
            if validate:
                parsed=strict_json(raw);Draft202012Validator(schema).validate(parsed)
            else:parsed=raw
            row['status']='response_received';row['schema_validation']='VALID' if validate else 'CORE_PENDING'
            return parsed,row
        except Exception as exc:
            row['status']='failed'
            code=self._failure_code(exc)
            row['error_code']=code;row['schema_validation']='INVALID' if code=='SCHEMA_INVALID' else 'NOT_CHECKED';self._stop(code)

    def _scope(self,payload):
        inspect_scope(payload)
        observed=payload['observation'];minute=observed['elapsed_minutes']
        if type(minute) is not int or minute<0:raise ValueError('INVALID_SCHEDULER_MINUTE')
        for record in observed.get('longitudinal_memory',{}).get('records',[]):
            if (record['patient_id']!=self.patient_id or record['visit_id']==self.visit_id or
                    not timestamp(record['visit_started_at'])<=timestamp(record['observed_at'])<=timestamp(record['available_at'])<=self.cutoff
                    or timestamp(record['visit_started_at'])>=self.cutoff):
                raise ValueError('MEMORY_BOUNDARY_VIOLATION')
        return digest(observed),digest(observed.get('longitudinal_memory',{}))

    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        if self.failure:raise HandoffStopped(self.failure)
        payload=strict_json(user_prompt)
        try:obs_fp,mem_fp=self._scope(payload)
        except Exception:self._stop('INFORMATION_BOUNDARY_VIOLATION')
        er=self.registry.get('er','')
        if model!=er['model']:self._stop('UNEXPECTED_CORE_ROLE_BINDING')
        core_index=len(self.core_calls)+1;repair='output_contract_repair' in payload
        entry={'core_call_index':core_index,'request_ids':[]};self.core_calls.append(entry)
        if not self.initial_done:
            self.initial_done=True
            initial_schema={'type':'object','required':['handoff_required','request_reason','decision'],
                'additionalProperties':False,'properties':{'handoff_required':{'type':'boolean'},
                'request_reason':{'type':'string','minLength':1,'maxLength':240},
                'decision':{'anyOf':[schema,{'type':'null'}]}}}
            initial=deepcopy(payload);initial['handoff_stage']='ER_INITIAL_DECISION'
            initial['instruction']='Decide whether logical specialty review is needed. Return handoff_required, request_reason and decision. If handoff_required use decision=null; otherwise return a valid original action decision. No physical consult is performed.'
            value,source=self._request('ER_INITIAL_DECISION',initial,system_prompt,er,initial_schema,core_index=core_index)
            entry['request_ids'].append(source['request_id'])
            if not value['handoff_required']:
                if value['decision'] is None:self._stop('SCHEMA_INVALID')
                entry['action_request_id']=source['request_id'];return json.dumps(value['decision'])
            if value['decision'] is not None:self._stop('SCHEMA_INVALID')
            if self.routing=='disabled':
                route={'decision_status':'disabled','selected_specialty':None}
            elif self.routing=='static_development':
                route={'decision_status':'selected','selected_specialty':self.static_department}
            else:
                depts=[a['department'] for a in self.registry.snapshot()['agents'] if a['role']=='specialty']
                route=self.registry.select_specialty(payload['observation'],payload['observation'].get('longitudinal_memory',{}),depts,router='adaptive')
            self.routing_records.append({'mode':self.routing,'status':route['decision_status'],'selected_specialty':route.get('selected_specialty')})
            handoff={'handoff_id':'Handoff0001','episode_id':self.episode_id,'source_decision_id':source['decision_id'],
                'requesting_role':'er','target_department':route.get('selected_specialty'),
                'request_reason':'ER_REQUESTED_LOGICAL_REVIEW','request_reason_fingerprint':digest(value['request_reason']),
                'visible_observation_fingerprint':obs_fp,'memory_snapshot_fingerprint':mem_fp,
                'information_scope':'REQUEST_TIME_RELEASED_OBSERVATION_AND_FIXED_MEMORY_ONLY',
                'request_minute':source['minute'],'specialty_response_id':None,'response_minute':None,
                'return_to_er_decision_id':None,'handoff_status':'handoff_requested'}
            self.handoffs.append(handoff)
            if route['decision_status']!='selected':self._stop('ROUTING_'+route['decision_status'].upper())
            try:binding=self.registry.get('specialty',route['selected_specialty'])
            except ValueError:self._stop('SPECIALTY_UNAVAILABLE')
            specialty_schema={'type':'object','required':['assessment','certainty','evidence_indices','completed'],
                'additionalProperties':False,'properties':{'assessment':{'type':'string','minLength':1,'maxLength':160},
                'certainty':{'enum':['supported','unresolved']},'completed':{'const':True},
                'evidence_indices':{'type':'array','maxItems':4,'uniqueItems':True,'items':{'type':'integer','minimum':0}}}}
            snapshot={'observation':deepcopy(payload['observation']),'evidence':deepcopy(payload.get('evidence',[])),
                'diagnosis_catalog':deepcopy(payload.get('diagnosis_catalog',[])),
                'handoff_stage':'SPECIALTY_ASSESSMENT','information_scope':handoff['information_scope'],
                'instruction':'Return a bounded independent assessment of released information, not a source Consult note. Do not invent facts. completed must be true. Opinions are not objective confirmation.'}
            try:snapshot.update(projection(payload))
            except EvidenceScopeError:
                self._record_specialty_evidence(None,{'semantic_validation':'INVALID','error_code':PARENT_ERROR,
                    'detail_code':'SPECIALTY_EVIDENCE_SCOPE_MISMATCH'})
                self._stop(PARENT_ERROR)
            snapshot['instruction']+=' Cite zero-based positions in evidence only; use evidence_index_map. A supported assessment needs a cited supported_diagnosis_eligible_indices entry. Memory positions and diagnosis catalog positions are not evidence indices. If the evidence list or eligible list is empty, supported is not permitted; return an honest unresolved assessment with evidence_indices=[] when appropriate.'
            value,response=self._request('SPECIALTY_ASSESSMENT',snapshot,specialty_system_prompt(binding['department']),binding,specialty_schema,core_index=core_index)
            entry['request_ids'].append(response['request_id'])
            summary=validate_response(value,snapshot);self._record_specialty_evidence(response,summary)
            if summary['semantic_validation']!='VALID' and self.specialty_semantic_repairs and summary['detail_code'] in REPAIRABLE_CODES:
                original=response
                corrected_input=repair_payload(snapshot,summary)
                # No added observation, Memory, private data or previous answer.
                # _request checks the same global counter before any transport.
                try:
                    repaired_obs,repaired_mem=self._scope(corrected_input)
                    if (repaired_obs,repaired_mem)!=(obs_fp,mem_fp):raise ValueError('REPAIR_SNAPSHOT_CHANGED')
                except Exception:self._stop('INFORMATION_BOUNDARY_VIOLATION')
                value,response=self._request('SPECIALTY_ASSESSMENT',corrected_input,specialty_system_prompt(binding['department']),
                    binding,specialty_schema,core_index=core_index,repair=True,repair_kind='SPECIALTY_SEMANTIC',
                    repair_of_request_id=original['request_id'])
                entry['request_ids'].append(response['request_id'])
                summary=validate_response(value,snapshot);self._record_specialty_evidence(response,summary)
            if summary['semantic_validation']!='VALID':self._stop(PARENT_ERROR)
            handoff.update(specialty_response_id=response['decision_id'],response_minute=response['minute'],handoff_status='specialty_completed')
            self._completed={'observation_fingerprint':obs_fp,'memory_fingerprint':mem_fp,'minute':response['minute'],
                'handoff_id':handoff['handoff_id'],'response_id':response['decision_id'],
                'summary':{k:value[k] for k in ('assessment','certainty','evidence_indices')}}
        returned=deepcopy(payload)
        if self._completed is not None:
            c=self._completed
            # Only same-snapshot assessment may be inserted; later decisions use
            # their new Core observation without replaying an outdated opinion.
            if obs_fp==c['observation_fingerprint'] and mem_fp==c['memory_fingerprint']:
                if c['minute']>payload['observation']['elapsed_minutes']:self._stop('FUTURE_SPECIALTY_RESPONSE')
                returned['logical_specialty_assessment']={'handoff_id':c['handoff_id'],'response_id':c['response_id'],
                    'response_minute':c['minute'],'kind':'MODEL_OPINION_NOT_SOURCE_RESULT',**deepcopy(c['summary'])}
        returned['handoff_stage']='ER_REASSESSMENT'
        returned['instruction']+=' A logical specialty assessment is an unverified model opinion, never objective confirmation or a source Consult result. Select only an original allowed action.'
        raw,response=self._request('ER_REASSESSMENT',returned,system_prompt,er,schema,core_index=core_index,repair=repair,validate=False)
        entry['request_ids'].append(response['request_id']);entry['action_request_id']=response['request_id']
        if self.handoffs and self.handoffs[-1]['handoff_status']=='specialty_completed' and (self.handoffs[-1]['return_to_er_decision_id'] is None or repair):
            self.handoffs[-1]['return_to_er_decision_id']=response['decision_id']
        return raw


class SequentialClient(SequentialEngine,OfflineClient):
    """Original offline entry; live transports remain forbidden here."""
    def __init__(self,transport,registry,**kwargs):
        if not isinstance(transport,OfflineClient):raise ValueError('OFFLINE_CLIENT_REQUIRED')
        super().__init__(transport,registry,**kwargs)


def link_handoff_requests(client,episode):
    """Link actual dispatches to the existing Core decision/event sequences."""
    for call,decision in zip(client.core_calls,episode['workflow']['decisions']):
        for row in client.requests:
            if row['request_id']==call.get('action_request_id'):
                row.update(core_decision_sequence=decision['sequence'],
                    action_id='Action{:04d}'.format(decision['sequence']) if decision['execution_event_sequences'] else None,
                    execution_event_sequences=decision['execution_event_sequences'],core_validation=decision['call_status'],
                    schema_validation='CORE_ACCEPTED' if decision['call_status']=='success' else 'CORE_REJECTED')
    for handoff in client.handoffs:
        returns=[r for r in client.requests if r['decision_id']==handoff['return_to_er_decision_id']]
        if handoff['handoff_status']=='specialty_completed':
            handoff['handoff_status']='handoff_completed' if returns and returns[0].get('core_validation')=='success' else 'handoff_failed'


def run_sequential_handoff(case,*,options,budget=10,routing='disabled',static_department=None,
                           routing_utility=None,test_only=False,output_root=None,export_id=None,specialty_semantic_repairs=0):
    """Offline-only entry. No real client can be admitted by this implementation."""
    session=ExportSession();result={'schema_version':'locdi.sequential_handoff.v1','status':'blocked',
        'blocker_codes':[],'episode':None,'audit':None,'raw':None,'new_llm_generations':0,
        'utility_fit_count':0,'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}
    try:
        if test_only is not True or options.allow_llm is not False or options.purpose!='development_smoke':raise ValueError('TEST_ONLY_OFFLINE_REQUIRED')
        _configuration(options)
        if options.runner_options['routing_mode']!='er_only' or options.runner_options['voi_enabled'] or options.max_retries!=0:raise ValueError('ER_CORE_NO_VOI_NO_RETRY_REQUIRED')
        if any(not a['version'].startswith('TEST_ONLY') for a in options.registry_manifest['agents']):raise ValueError('TEST_ONLY_BINDINGS_REQUIRED')
        context=adapt_case(case,registry_manifest=options.registry_manifest)
        registry=AgentRegistry(options.registry_manifest,routing_utility)
        client=SequentialClient(options.client,registry,episode_id=session.run_id,patient_id=case['patient_id'],
            visit_id=case['agent_visible']['index_encounter_id'],cutoff=case['agent_visible']['index_encounter_start'],
            budget=budget,routing=routing,static_department=static_department,specialty_semantic_repairs=specialty_semantic_repairs)
    except Exception as exc:
        safe={'ROUTING_UTILITY_MISSING','SPECIALTY_UNAVAILABLE','TEST_ONLY_OFFLINE_REQUIRED','ER_CORE_NO_VOI_NO_RETRY_REQUIRED','TEST_ONLY_BINDINGS_REQUIRED'}
        result['blocker_codes']=[str(exc) if str(exc) in safe else 'HANDOFF_INPUT_CONTRACT_BLOCKED'];return result
    raw=run_case(case,options=replace(options,client=client));result.update(raw=raw,status=raw['status'])
    if client.failure:result['blocker_codes'].append(client.failure)
    episode=build_episode(raw,context=context,session=session)
    link_handoff_requests(client,episode)
    audit={'schema_version':'locdi.sequential_handoff_audit.v1','episode_id':session.run_id,'test_only':True,
        'requests':client.requests,'handoffs':client.handoffs,'routing':client.routing_records,
        'routing_mode':routing,'routing_status':client.routing_records[-1]['status'] if client.routing_records else 'not_requested',
        'global_budget':budget,'request_count':len(client.requests),'runtime_status':raw['status'],
        'specialty_semantic_repair_count':sum(r.get('repair_kind')=='SPECIALTY_SEMANTIC' for r in client.requests),
        'retry_count':sum(r['retry'] for r in client.requests),
        'export_status':'NOT_REQUESTED','specialty_transition_kind':'LOGICAL_JUDGMENT_EXCHANGE',
        'source_consult_count':sum(e['event_type']=='consult_note_added' for e in episode['workflow']['events']),
        'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False,
        'configuration_fingerprint':episode['configuration_fingerprint']}
    audit['handoff_configuration']={'version':'locdi.sequential_handoff.v1','budget':budget,'routing':routing,
        'static_department':static_department,'registry_fingerprint':digest(registry.snapshot()),
        'routing_utility_fingerprint':digest(routing_utility) if routing_utility is not None else None,
        'one_handoff_per_episode':True,'logical_delay_minutes':0,'test_only':True}
    audit['handoff_configuration']['specialty_semantic_repair']={'version':REPAIR_VERSION,'max_attempts':specialty_semantic_repairs}
    audit['handoff_configuration_fingerprint']=digest(audit['handoff_configuration'])
    result.update(episode=episode,audit=audit)
    if output_root is not None:
        try:
            name=export_id or session.run_id
            _write(episode,'episodes/'+name+'.json',output_root,'episodes')
            export_summary({'episodes':[episode]},'metrics/'+name+'.jsonl',output_root=output_root)
            audit['export_status']='EXPORTED'
            _write(audit,'reports/'+name+'_handoff.json',output_root,'reports')
        except Exception:
            audit['export_status']='EXPORT_FAILED';result['status']='export_failed';result['blocker_codes'].append('EXPORT_FAILED')
    return result
