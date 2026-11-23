"""TEST_ONLY sequential handoff around unchanged Core routing/Bayes execution."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
from app.core.agent_registry import AgentRegistry
from app.core.information_value import InformationPolicy
from app.locdi.run_simulation import OfflineClient,_configuration,run_case,_SOCKET_LOCK
from app.locdi.sequential_handoff import SequentialClient,HandoffStopped,strict_json,link_handoff_requests
from app.locdi.specialty_evidence_contract import projection
from app.locdi.case_adapter import adapt_case
from app.locdi.episode_exporter import ExportSession,build_episode,export_summary,_write,ACTIONS
from app.locdi.handoff_audit import project_audit
from app.locdi.llm_setup import digest


class RoutingNotSelected(HandoffStopped):
    pass


class IntegratedSequentialClient(SequentialClient):
    """Offline-only subclass; no Live adapter inheritance or registry switch."""
    def _stop(self,code):
        if code in {'ROUTING_ABSTAINED','ROUTING_DISABLED'}:
            status=code.removeprefix('ROUTING_').lower()
            self.handoffs[-1].update(handoff_status='not_performed',route_not_selected=status)
            raise RoutingNotSelected(code)
        super()._stop(code)

    def _dispatch(self,stage,payload,system,binding,schema):
        raw=super()._dispatch(stage,payload,system,binding,schema)
        if stage=='ER_REASSESSMENT':
            try:value=strict_json(raw)
            except (ValueError,TypeError):value={}
            action=value.get('action') if isinstance(value,dict) else None
            if action in ACTIONS:self.requests[-1]['model_proposal_action']=action
        return raw

    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        try:
            p=strict_json(user_prompt);self._scope(p);projection(p)
        except Exception:self._stop('INFORMATION_BOUNDARY_VIOLATION')
        try:return super().chat_json(system_prompt,user_prompt,model,schema=schema)
        except RoutingNotSelected:
            # Resume ER, with no Specialty opinion. The initial logical request
            # and ER proposal belong to the SAME original Core decision.
            original=self.core_calls.pop()
            try:return super().chat_json(system_prompt,user_prompt,model,schema=schema)
            finally:
                self.core_calls[-1]['request_ids']=original['request_ids']+self.core_calls[-1]['request_ids']


def run_handoff_integrated(case,*,options,test_only=False,routing='static_development',static_department='neurology',
                           budget=10,specialty_semantic_repairs=1,output_root=None,export_id=None):
    result={'schema_version':'locdi.handoff_integration.v2','status':'blocked','test_only':test_only,'blocker_codes':[],
        'episode':None,'audit':None,'raw':None,'export_status':'NOT_REQUESTED','integration_contract_ready':False,
        'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False,'new_model_generations':0,'utility_fitting':0}
    try:
        if (test_only is not True or options.purpose!='development_smoke' or options.allow_llm is not False
                or not isinstance(options.client,OfflineClient)):raise ValueError('TEST_ONLY_OFFLINE_REQUIRED')
        if routing not in {'static_development','adaptive_test_only','disabled'}:raise ValueError('HANDOFF_ROUTING_MODE_INVALID')
        if type(budget) is not int or not 1<=budget<=10:raise ValueError('GLOBAL_BUDGET_INVALID')
        if options.max_retries!=0:raise ValueError('RUNNER_RETRY_DISABLED')
        if options.runner_options['routing_mode']!='er_only':raise ValueError('CORE_ER_COORDINATOR_REQUIRED')
        if any(not a['version'].startswith('TEST_ONLY') or not a['model'].startswith('TEST_ONLY')
               for a in options.registry_manifest['agents']):raise ValueError('TEST_ONLY_BINDINGS_REQUIRED')
        if routing=='adaptive_test_only':
            if options.routing_utility is None:raise ValueError('ROUTING_UTILITY_MISSING')
            if any(not r['patient_id'].startswith('TEST_ONLY') for r in options.routing_utility['samples']):raise ValueError('TEST_ONLY_ROUTING_REQUIRED')
        elif options.routing_utility is not None:raise ValueError('INACTIVE_ROUTING_ARTIFACT')
        enabled=options.runner_options['voi_enabled']
        if enabled:
            if options.voi_model is None:raise ValueError('VOI_ARTIFACT_MISSING')
            model=InformationPolicy(options.voi_model).model
            if (options.voi_model['provenance']['synthetic'] is not True or not options.voi_model['model_id'].startswith('TEST_ONLY')
                    or any(not d.startswith('TEST_ONLY') for d in model.artifact['diagnoses'])
                    or list(options.diagnosis_catalog or [])!=list(model.artifact['diagnoses'])
                    or options.diagnosis_departments!=model.artifact['diagnoses']):raise ValueError('TEST_ONLY_VOI_CATALOG_REQUIRED')
        _configuration(options)
        context=adapt_case(case,registry_manifest=options.registry_manifest)
        registry=AgentRegistry(options.registry_manifest,options.routing_utility)
        client=IntegratedSequentialClient(options.client,registry,episode_id=ExportSession().run_id,
            patient_id=case['patient_id'],visit_id=case['agent_visible']['index_encounter_id'],
            cutoff=case['agent_visible']['index_encounter_start'],budget=budget,
            routing='adaptive' if routing=='adaptive_test_only' else routing,
            static_department=static_department if routing=='static_development' else None,
            specialty_semantic_repairs=specialty_semantic_repairs)
    except Exception as exc:
        safe={'TEST_ONLY_OFFLINE_REQUIRED','HANDOFF_ROUTING_MODE_INVALID','GLOBAL_BUDGET_INVALID','RUNNER_RETRY_DISABLED',
            'CORE_ER_COORDINATOR_REQUIRED','TEST_ONLY_BINDINGS_REQUIRED','ROUTING_UTILITY_MISSING','TEST_ONLY_ROUTING_REQUIRED',
            'INACTIVE_ROUTING_ARTIFACT','VOI_ARTIFACT_MISSING','TEST_ONLY_VOI_CATALOG_REQUIRED'}
        result['blocker_codes']=[str(exc) if str(exc) in safe else 'INTEGRATION_INPUT_CONTRACT_BLOCKED'];return result
    traces=[];original=InformationPolicy.propose
    def observe(policy,runtime,observed,**kwargs):
        client._scope({'observation':observed})
        if (observed.get('objective_findings',[])!=runtime.objective_findings
                or observed.get('history',[])!=runtime.question_history or 'logical_specialty_assessment' in observed):
            raise RuntimeError('SOURCE_OBSERVATION_BOUNDARY_VIOLATION')
        before=deepcopy(runtime.voi_state.get('belief',policy.model.artifact['prior']))
        value=original(policy,runtime,observed,**kwargs)
        state=runtime.voi_state;trace=state.get('history',[{}])[-1]
        traces.append({'sequence':len(traces)+1,'minute':runtime.current_time_min,
            'preceding_request_id':client.requests[-1]['request_id'] if client.requests else None,
            'preceding_request_stage':client.requests[-1]['stage'] if client.requests else None,
            'model_fingerprint':policy.model.fingerprint,'horizon':policy.model.artifact['horizon'],
            'belief_before_fingerprint':digest(before),'belief_after_fingerprint':digest(state['belief']) if 'belief' in state else None,
            'belief_changed':before!=state.get('belief') if 'belief' in state else None,
            'matched_outcome_count':len(state.get('matched_outcomes',[])),
            'released_objective_count':len(observed.get('objective_findings',[])),'released_history_count':len(observed.get('history',[])),
            'observation_fingerprint':digest(observed),'specialty_opinion_used_as_source':False,
            'status':trace.get('status'),'selected_action':value.get('action') if value else None,
            'fallback':value is None,'unavailable_reason':trace.get('reason') if trace.get('reason') in
                {'missing_action_outcome_model','unsupported_observation','diagnostic_gate_requires_evidence','no_selectable_action'} else None})
        return value
    with _SOCKET_LOCK,patch.object(InformationPolicy,'propose',observe):
        raw=run_case(case,options=replace(options,client=client))
    result.update(raw=raw,status=raw['status'])
    if client.failure:result['blocker_codes'].append(client.failure)
    session=ExportSession(run_id=client.episode_id)
    try:
        episode=build_episode(raw,context=context,session=session);link_handoff_requests(client,episode)
        audit=project_audit(episode,client.requests,client.handoffs,client.routing_records,actual_requests=0,identity_checks=0)
        rows={r['request_id']:r for r in client.requests}
        for trace in traces:
            linked=rows.get(trace['preceding_request_id'],{})
            trace['core_decision_sequence']=linked.get('core_decision_sequence')
            trace['execution_event_sequences']=deepcopy(linked.get('execution_event_sequences',[]))
        audit.update(test_only=True,offline_dispatch_count=len(client.requests),routing_mode=routing,
            routing_performance_evaluated=False,voi=traces,voi_enabled=enabled,
            artifacts={'routing_fingerprint':registry.utility.fingerprint if registry.utility is not None else None,
                'voi_fingerprint':model.fingerprint if enabled else None,'scope':'TEST_ONLY_NOT_RESEARCH'},
            source_consult_count=sum(e['event_type']=='consult_note_added' for e in episode['workflow']['events']))
        audit['fingerprint']=digest(audit)
        result.update(episode=episode,audit=audit)
        if any(t['fallback'] for t in traces):result['blocker_codes'].append('VOI_FALLBACK_OBSERVED')
        result['integration_contract_ready']=(raw['status']=='completed' and not result['blocker_codes']
            and all(d['execution_success'] is True for d in episode['workflow']['decisions'])
            and (not enabled or bool(traces) and all(t['core_decision_sequence'] and t['execution_event_sequences'] for t in traces)))
        if output_root is not None:
            name=export_id or session.run_id
            _write(episode,'episodes/'+name+'.json',output_root,'episodes')
            export_summary({'episodes':[episode]},'metrics/'+name+'.jsonl',output_root=output_root)
            _write(audit,'reports/'+name+'_handoff_v2.json',output_root,'reports')
            result['export_status']='EXPORTED'
    except Exception:
        result['export_status']='FAILED_OR_PARTIAL';result['integration_contract_ready']=False
        result['blocker_codes'].append('EXPORT_OR_AUDIT_CONTRACT_ERROR')
    return result
