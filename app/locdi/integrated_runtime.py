"""Offline integration entry over the existing Runner/Scheduler/policies.

No new clinical algorithm, model backend, training or evaluation path.
Toy Routing/VoI injection is restricted to explicit TEST_ONLY execution.
"""
from copy import deepcopy
from unittest.mock import patch

from app.core.agent_registry import AgentRegistry
from app.core.information_value import InformationPolicy
from app.locdi.llm_setup import digest
from app.locdi.run_simulation import (run_case, _configuration, _SOCKET_LOCK,
                                     OfflineClient, SimulationConfigurationError)
from app.locdi.case_adapter import adapt_case
from app.locdi.episode_exporter import (ExportSession, build_episode, export_episode, export_summary, _write)
from app.locdi.routing_collector import _id


def run_integrated(case, *, options, test_only=False, expected_registry_fingerprint=None,
                   expected_voi_fingerprint=None, output_root=None, export_id=None):
    """Run one validated case, observe actual calls, optionally export safely.

    Private raw result remains in memory. Only existing redacted Episode and
    whitelist Runtime audit are saved. Source-backed Consult is never invented.
    """
    readiness = {'structural_contract_ready': False, 'runtime_integration_ready': False,
                 'clinical_reference_ready': False, 'research_evaluation_ready': False}
    result = {'schema_version':'locdi.integrated_runtime.v1', 'test_only':test_only,
              'status':'blocked', 'readiness':readiness, 'blocker_codes':[], 'episode':None, 'audit':None,
              'raw':None, 'clinical_metrics':'NOT_EVALUATED', 'correctness':None,
              'new_llm_generation_count':0, 'utility_fit_count':0}
    try:
        if type(test_only) is not bool or options.purpose != 'development_smoke' or options.allow_llm is not False or not isinstance(options.client, OfflineClient):
            raise SimulationConfigurationError('OFFLINE_DEVELOPMENT_ONLY')
        features = options.runner_options or {}
        adaptive = features.get('routing_mode') != 'er_only' and (features.get('routing_mode') == 'adaptive' or features.get('primary_router') == 'adaptive')
        if adaptive and options.routing_utility is None:
            result['blocker_codes']=['ROUTING_UTILITY_MISSING']; return result
        if (adaptive or features.get('voi_enabled')) and not test_only:
            result['blocker_codes']=['REVIEWED_RUNTIME_ARTIFACT_NOT_AVAILABLE']; return result
        if test_only and any(not a['version'].startswith('TEST_ONLY') for a in (options.registry_manifest or {}).get('agents', [])):
            result['blocker_codes']=['TEST_ONLY_BINDINGS_REQUIRED']; return result
        core = _configuration(options)
        if expected_registry_fingerprint is not None and expected_registry_fingerprint != digest(core['registry_manifest']):
            result['blocker_codes']=['REGISTRY_FINGERPRINT_MISMATCH']; return result
        if options.voi_model is not None:
            model = InformationPolicy(options.voi_model).model
            if (options.voi_model['provenance']['synthetic'] is not True
                    or list(options.diagnosis_catalog or []) != list(model.artifact['diagnoses'])
                    or options.diagnosis_departments != model.artifact['diagnoses']):
                result['blocker_codes']=['VOI_CATALOG_OR_TEST_ONLY_PROVENANCE_MISMATCH']; return result
            if expected_voi_fingerprint is not None and expected_voi_fingerprint != model.fingerprint:
                result['blocker_codes']=['VOI_FINGERPRINT_MISMATCH']; return result
        context = adapt_case(case, registry_manifest=options.registry_manifest)
        readiness['structural_contract_ready']=True
    except Exception:
        result['blocker_codes']=['RUNTIME_INPUT_CONTRACT_BLOCKED']; return result
    routing, voi = [], []
    select = AgentRegistry.select_specialty; propose = InformationPolicy.propose

    def observed_select(registry, *args, **kwargs):
        value=select(registry,*args,**kwargs)
        routing.append(deepcopy(value))
        return value

    def observed_propose(policy, runtime, observed, **kwargs):
        before=deepcopy(runtime.voi_state.get('belief',policy.model.artifact['prior']))
        value=propose(policy,runtime,observed,**kwargs)
        state=runtime.voi_state
        voi.append({'minute':runtime.current_time_min, 'model_fingerprint':policy.model.fingerprint,
            'horizon':policy.model.artifact['horizon'], 'belief_before_fingerprint':digest(before),
            'belief_after_fingerprint':digest(state['belief']) if 'belief' in state else None,
            'belief_changed':before != state.get('belief') if 'belief' in state else None,
            'matched_outcome_count':len(state.get('matched_outcomes',[])),
            'trace':deepcopy(state.get('history',[{}])[-1]), 'proposal':deepcopy(value)})
        return value
    # Passive observers forward every original call. The existing Runner owns
    # Memory, clocks, roles, Bayes updates, actions and finalization.
    with _SOCKET_LOCK, patch.object(AgentRegistry,'select_specialty',observed_select), patch.object(InformationPolicy,'propose',observed_propose):
        raw=run_case(case,options=options)
    result['raw']=raw; result['status']=raw['status']
    session=ExportSession()
    try:
        episode=build_episode(raw,context=context,session=session)
        traces=[]
        for i,v in enumerate(voi,1):
            t=v['trace']; proposal=v['proposal']; links=[]
            if proposal:
                for d in episode['workflow']['decisions']:
                    if d['minute']==v['minute'] and d['selected_action']==proposal['action']:
                        links.append(d['sequence'])
            traces.append({k:v[k] for k in ('minute','model_fingerprint','horizon','belief_before_fingerprint',
                'belief_after_fingerprint','belief_changed','matched_outcome_count')})
            traces[-1].update(sequence=i,status=t.get('status') if t.get('status') in {'selected','unavailable','evaluating'} else None,
                selected_action=proposal.get('action') if proposal else None,
                candidate_actions=[c.get('action') for c in t.get('candidates',[])],
                decision_sequences=links, fallback=proposal is None,
                unavailable_reason=t.get('reason') if t.get('reason') in {'missing_action_outcome_model','unsupported_observation','diagnostic_gate_requires_evidence','no_selectable_action'} else None)
        audit={'schema_version':'locdi.runtime_integration_audit.v1','episode_id':session.run_id,
            'configuration_fingerprint':episode['configuration_fingerprint'],'test_only':test_only,
            'role_bindings':deepcopy(episode['decision_audit']['records']),
            'memory':deepcopy(episode['memory']),
            'routing':[{'sequence':i,'status':r.get('decision_status') if r.get('decision_status') in {'selected','abstained','unavailable','disabled'} else None,
                        'selected_specialty':r.get('selected_specialty'), 'artifact_fingerprint':r.get('utility_artifact_sha256')}
                       for i,r in enumerate(routing,1)],
            'specialty_transition_kind':'logical_role_binding_not_source_consult',
            'source_backed_consult_event_count':sum(e['event_type']=='consult_note_added' for e in episode['workflow']['events']),
            'voi':traces,'workflow':deepcopy(episode['workflow']), 'terminal':deepcopy(episode['result']),
            'clinical_metrics':'NOT_EVALUATED','correctness':None,
            'evaluation_policy_status':{'diagnosis_correctness':'NOT_REVIEWED','abstention_appropriateness':'NOT_REVIEWED','routing_success':'NOT_REVIEWED'}}
        audit['fingerprint']=digest(audit)
        readiness['runtime_integration_ready']=(raw['status']=='completed' and bool(episode['workflow']['decisions'])
            and all(d['execution_success'] is True for d in episode['workflow']['decisions'])
            and (not features['voi_enabled'] or
            bool(traces) and all(t['status']=='selected' and t['decision_sequences'] for t in traces)))
        result.update(episode=episode,audit=audit)
        if features['voi_enabled'] and any(t['fallback'] for t in traces):
            result['blocker_codes'].append('VOI_FALLBACK_OBSERVED')
        if output_root is not None:
            name=_id(export_id or session.run_id)
            export_episode(raw,'episodes/'+name+'.json',context=context,session=session,output_root=output_root)
            export_summary({'episodes':[episode]},'metrics/'+name+'.jsonl',output_root=output_root)
            _write(audit,'reports/'+name+'_runtime.json',output_root,'reports')
    except Exception:
        readiness['runtime_integration_ready']=False
        result['blocker_codes'].append('EXPORT_OR_AUDIT_CONTRACT_ERROR')
    return result
