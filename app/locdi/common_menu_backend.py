"""New development-only Native B0..B4 connection; no legacy Gate changes.

Adaptive execution requires BOTH existing correctness validators and exact fit
lineage. Clinical research admission remains unchanged/blocked. No fitting here.
"""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import observed_tokens
from app.core.diagnostic_belief import terminal_choice
from app.core.hospital_llm_policy import HospitalLLMPolicy
from app.core.voi_planning import plan_voi as original_plan_voi
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient
from app.locdi.current_ablation_runner import _atomic_json,features_for
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.run_simulation import SimulationOptions,run_case,_SOCKET_LOCK,_digest
from app.locdi.case_adapter import adapt_case
from app.locdi.episode_exporter import build_episode
from app.locdi.m5_common_source import CommonM5SourceView,NativeM5InformationPolicy
from validation.routing_correctness_gate import to_policy_fit_rows
from validation.artifact_readiness import validate_routing_artifact
from app.locdi.development_routing_proxy import MARKERS,admit_development_proxy
from app.locdi.common_prediction_contract import extract_core_prediction,shared_selection
from app.locdi.common_runtime_recovery import released_measurement_view,StopThenLLMInformationPolicy
from app.locdi.medical_knowledge import MedicalKnowledgeIndex,historical_labels
from app.locdi.literature_rule_router import LiteratureRuleRegistry


def routing_admission(evidence,registry,patients,config):
    """Never accept a supplied ready/report flag or impute unresolved labels."""
    if not isinstance(evidence,dict):raise ValueError('REVIEWED_ROUTING_EVIDENCE_REQUIRED')
    data=evidence['training_data'];options=evidence['validator_options']
    if options.get('registry_manifest')!=registry or options.get('allow_test_only') is True:
        raise ValueError('CURRENT_REGISTRY_REAL_REVIEW_REQUIRED')
    converted=to_policy_fit_rows(data,**options)
    if not converted['report']['utility_fit_ready'] or converted['rows'] is None:
        raise ValueError('EXISTING_CORRECTNESS_AND_LEGACY_GATES_BLOCKED')
    artifact=evidence['routing_artifact']
    if not evidence.get('origin_files'):raise ValueError('ACTUAL_COLLECTION_REVIEW_LINEAGE_REQUIRED')
    for name,expected_hash in evidence['origin_files'].items():
        h=hashlib.sha256()
        with Path(name).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected_hash:raise ValueError('ORIGINAL_COLLECTION_OR_REVIEW_CHANGED')
    if artifact.get('min_samples')!=2:raise ValueError('FROZEN_MIN_SAMPLES_TWO_REQUIRED')
    expected=[]
    for row in converted['rows']:
        expected.append({**{k:row[k] for k in ('patient_id','decision_id','agent_id','correct')},
                         'tokens':observed_tokens(row['observation'])})
    if sorted(map(digest,artifact['samples']))!=sorted(map(digest,expected)):
        raise ValueError('EXACT_REVIEWED_FIT_LINEAGE_REQUIRED')
    manifest=AgentRegistry(registry).snapshot()
    if any(a['model']!=config['model'] or a['version']!=config['model_version'] for a in manifest['agents']):
        raise ValueError('CURRENT_MODEL_VERSION_MISMATCH')
    if evidence.get('model_identifier')!=config['model_identifier']:
        raise ValueError('CURRENT_ROUTING_DIGEST_PROOF_REQUIRED')
    prepared=validate_setup(config)
    if prepared['static_readiness']!='READY':raise ValueError('CURRENT_CONFIG_INVALID')
    if evidence.get('prompt_fingerprints')!=prepared['prepared']['prompt_fingerprints']:
        raise ValueError('CURRENT_PROMPT_PROOF_REQUIRED')
    result=validate_routing_artifact(registry,artifact,evaluation_patient_ids=patients,
                                     expected_bindings=registry['agents'])
    if result['development_status']!='DEVELOPMENT_CONTRACT_READY':
        raise ValueError('EXISTING_ROUTING_ARTIFACT_CONTRACT_BLOCKED')
    train=set(artifact_row['patient_id'] for artifact_row in artifact['samples'])
    if train&set(patients):raise ValueError('ROUTING_TRAIN_EVALUATION_PATIENT_OVERLAP')
    return {'correctness':converted['report'],'artifact':artifact,'review_authentication':'NOT_INSTALLED',
            'clinical_research_admission':False,'scope':'INDEPENDENT_REVIEW_CONTRACT_DEVELOPMENT_ONLY'}


class CommonMenuNativeClient(BudgetedNativeERClient):
    def __init__(self,*args,source_binding,selection_taxonomy=None,runtime_recovery=False,
                 knowledge_pack=None,memory_time_labels=False,evidence_set=False,**kwargs):
        self.source_binding=deepcopy(source_binding)
        self.selection_taxonomy=deepcopy(selection_taxonomy)
        self.runtime_recovery=runtime_recovery;self.last_terminal_candidate=None
        self.knowledge_pack=deepcopy(knowledge_pack);self.memory_time_labels=memory_time_labels;self.evidence_set=evidence_set
        super().__init__(*args,**kwargs)

    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        self.last_terminal_candidate=None
        payload=json.loads(user_prompt)
        if 'allowed_actions' in payload:
            # Source descriptions/code/unit only; no outcome, value or category.
            payload['source_action_metadata']=[{k:r[k] for k in ('kind','target','description','code','unit')}
                                                for r in self.source_binding['rows']]
            if self.selection_taxonomy is not None:
                payload,schema=shared_selection(payload,schema,self.selection_taxonomy)
            if getattr(self,'runtime_recovery',False):payload=released_measurement_view(payload,self.source_binding)
            if getattr(self,'knowledge_pack',None) is not None:
                payload['external_medical_knowledge']=deepcopy(self.knowledge_pack)
            if getattr(self,'memory_time_labels',False):payload['memory_scope_annotations']=historical_labels(payload['observation'])
        core_schema=deepcopy(schema)
        if getattr(self,'evidence_set',False):
            from app.locdi.evidence_selection_wire import encode
            payload,schema=encode(payload,schema)
        content=super().chat_json(system_prompt,json.dumps(payload,ensure_ascii=False,sort_keys=True),model,schema=schema)
        if getattr(self,'evidence_set',False):
            from app.locdi.evidence_selection_wire import decode,VERSION
            decoded=decode(json.loads(content),core_schema);content=json.dumps(decoded,ensure_ascii=False,sort_keys=True)
            _atomic_json(self.checkpoint_root/('Request%04d_core_response.json'%self.request_count),
                {'wire_contract':VERSION,'decoded_response':decoded,'core_schema_fingerprint':digest(core_schema),
                 'decoded_response_fingerprint':digest(decoded)})
        if getattr(self,'runtime_recovery',False):
            self.last_terminal_candidate={'observation_fingerprint':digest(payload['observation']),'decision':json.loads(content)}
        return content


def make_common_backend(cases,*,config,bindings,model,authorization,plan_fingerprint,
                        checkpoint_root,routing_evidence=None,history_policy=None,
                        experiment_kind='CLINICAL_EVALUATION',catalog_selection=False,runtime_recovery=False,
                        knowledge_corpus=None,memory_time_labels=False,rule_based=False,evidence_set=False):
    """Called only inside a bounded/approved development CLI budget context.

    Reuses the actual Core primary-specialty execution path, not a new router or
    a claim of three-stage ER-Specialty-ER handoff. All Source/Model guards stay on.
    """
    if experiment_kind!='DEVELOPMENT_EXPERIMENT':
        raise ValueError('CLINICAL_EVALUATION_REQUIRES_UNCHANGED_OFFICIAL_GATES')
    if runtime_recovery and not catalog_selection:raise ValueError('RECOVERY_REQUIRES_SHARED_CATALOG_SELECTION')
    knowledge_pack=MedicalKnowledgeIndex(knowledge_corpus).common() if knowledge_corpus is not None else None
    if (authorization.get('accept_source_label_proxy') is not True
        or any(authorization.get(k)!=MARKERS[k] for k in ('experiment_kind','source_label_proxy','independent_clinical_review','clinical_accuracy'))):
        raise ValueError('EXPLICIT_SOURCE_LABEL_PROXY_DEVELOPMENT_REQUIRED')
    if (authorization.get('plan_fingerprint')!=plan_fingerprint
        or authorization.get('scope')!='B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY'
        or authorization.get('operator_approved') is not True
        or authorization.get('no_other_ollama_work') is not True
        or authorization.get('clinical_release_approved') is not False
        or authorization.get('accept_common_m5_menu') is not True
        or authorization.get('accept_recorded_time_proxy') is not True
        or not authorization.get('user_statement') or not authorization.get('operator_ref')):
        raise ValueError('PINNED_PLAN_ACTUAL_OPERATOR_APPROVAL_REQUIRED')
    issued=datetime.fromisoformat(authorization['recorded_at']);expires=datetime.fromisoformat(authorization['expires_at'])
    if issued.tzinfo is None or expires.tzinfo is None or not issued<=datetime.now(timezone.utc)<expires or not 0<(expires-issued).total_seconds()<=86400:
        raise ValueError('EXPLICIT_OPERATOR_APPROVAL_TIME_INVALID')
    ready=validate_setup(config)
    if ready['static_readiness']!='READY':raise ValueError('MODEL_REGISTRY_PROMPT_NOT_READY')
    registry=ready['prepared']['registry'];patients={c['patient_id'] for c in cases.values()}
    count=[0]
    def execute_case(execution_case,features):
        condition=next((c for c in ('B0','B1','B2','B3','B4') if features==features_for(c)),None)
        if condition is None:raise ValueError('COMMON_MENU_FEATURES_INVALID')
        if condition not in authorization.get('conditions',[]):raise ValueError('CONDITION_NOT_AUTHORIZED')
        ref=execution_case['case_ref'];case=deepcopy(cases[ref]);binding=bindings[ref]
        artifact=None
        if condition in ('B3','B4'):
            artifact=admit_development_proxy(routing_evidence,registry,patients,config)['artifact']
        if condition=='B4':
            from app.locdi.development_routing_proxy import read as proxy_read
            tp=proxy_read(routing_evidence['collection_plan'])
            val={c['decision']['patient_id'] for c in tp['candidates'] if c['decision']['split']=='validation'}
            if val&set(model['provenance']['training_patients']):
                raise ValueError('ROUTING_VALIDATION_VOI_TRAIN_OVERLAP')
            if patients&set(model['provenance']['training_patients']):raise ValueError('M5_TRAIN_EVALUATION_PATIENT_OVERLAP')
        count[0]+=1;folder=Path(checkpoint_root)/('Episode%04d'%count[0])
        source_auth={**authorization,'source_binding_fingerprint':digest(binding)}
        view=CommonM5SourceView(binding,source_auth)
        client=CommonMenuNativeClient(config,authorized=True,contract_fingerprint=plan_fingerprint,
            checkpoint_root=folder,response_contract='memory_citation_subset_code.v1',source_binding=binding,
            selection_taxonomy=model['diagnoses'] if catalog_selection else None,runtime_recovery=runtime_recovery,
            knowledge_pack=knowledge_pack,memory_time_labels=memory_time_labels,evidence_set=evidence_set)
        decisions=[];policies=[];runtimes=[];plans=[];start=time.monotonic()
        original=HospitalLLMPolicy._decision
        def observed(policy,runtime,role,department=''):
            if not any(r is runtime for r in runtimes):runtimes.append(runtime)
            result=original(policy,runtime,role,department)
            history=runtime.voi_state.get('history',[])
            if (runtime_recovery and condition=='B4' and history
                and history[-1].get('minute')==runtime.current_time_min
                and history[-1].get('terminal_origin')=='CURRENT_LLM_DIAGNOSE_PENDING_CORE_REVALIDATION'):
                history[-1]['terminal_origin']='CURRENT_LLM_DIAGNOSE_CORE_REVALIDATED'
                history[-1]['hospital_terminal_certainty']=result.get('certainty')
                runtime.llm_audit[-1]['terminal_policy_contract']={
                    'origin':'LLM_AFTER_FROZEN_M5_STOP','core_revalidated':True,
                    'm5_clinical_support':False,'clinical_support_verified':False}
            if isinstance(result,dict):decisions.append({'minute':runtime.current_time_min,'entry_role':role,**deepcopy(result)})
            return result
        def information_policy(artifact,selector='bayesian_voi'):
            if digest(artifact)!=digest(model):raise ValueError('IMMUTABLE_M5_REQUIRED')
            p=(StopThenLLMInformationPolicy(artifact,binding,selector,lambda:client.last_terminal_candidate)
               if runtime_recovery else NativeM5InformationPolicy(artifact,binding,selector))
            policies.append(p);return p
        def measured_plan(*args,**kwargs):
            if len(plans)>=12:raise RuntimeError('EPISODE_PLANNER_CALL_BUDGET')
            result=original_plan_voi(*args,**kwargs);plans.append(deepcopy(result))
            if sum(r['evaluated_states'] for r in plans)>240000:raise RuntimeError('EPISODE_PLANNER_STATE_BUDGET')
            return result
        def registry_adapter(manifest,utility_artifact=None):
            if rule_based and condition in ('B3','B4'):
                return LiteratureRuleRegistry(manifest,utility_artifact,source_binding=binding)
            return AgentRegistry(manifest,utility_artifact)
        try:
            options=SimulationOptions(purpose='development_llm',allow_llm=True,dry_run=False,
                client=client,llm_config=config,minute_limit=config['minute_limit'],registry_manifest=registry,
                runner_options=features,accept_recorded_release_assumption=True,max_retries=0,
                routing_utility=artifact,voi_model=model if condition=='B4' else None,
                allow_unfitted_adaptive_development=condition in ('B3','B4') and artifact is None,
                diagnosis_catalog=list(model['diagnoses']),sequential_source_view=view,
                diagnosis_departments=deepcopy(model['diagnoses']) if catalog_selection else None,
                history_policy=history_policy if condition=='B1' else None)
            with _SOCKET_LOCK,patch.object(HospitalLLMPolicy,'_decision',new=observed),\
                    patch('app.core.longitudinal_runner.AgentRegistry',new=registry_adapter),\
                    patch('app.core.longitudinal_runner.InformationPolicy',new=information_policy),\
                    patch('app.core.information_value.plan_voi',new=measured_plan):
                raw=run_case(case,options=options)
            _atomic_json(folder/'raw_runner_result.json',raw)
            _atomic_json(folder/'actual_core_decisions.json',decisions)
            _atomic_json(folder/'planner_trace.json',plans)
            if runtimes:_atomic_json(folder/'runtime_snapshot_private.json',runtimes[-1].to_summary())
            # Old public serializer has a closed policy enum. Keep the original
            # raw/config intact; publish an explicitly linked compatibility view.
            export_raw=deepcopy(raw)
            cfg=export_raw['execution_metadata']['configuration'];cfg.pop('sequential_source_view',None)
            cfg.pop('development_unfitted_adaptive',None)
            export_raw['execution_metadata']['configuration_fingerprint']=_digest(cfg)
            context=adapt_case(case,registry_manifest=registry)
            episode=build_episode(export_raw,context=context)
            linkage={'native_policy':view.metadata(),'raw_fingerprint':digest(raw),'episode_fingerprint':digest(episode),
                'export_projection':'old serializer omits unsupported source-policy/development-abstention metadata; raw kept unchanged and linked here',
                'clinical_research_admission':False,**MARKERS,
                'routing_utility_status':'ABSTENTION_NO_PROXY_GROUPS' if condition in ('B3','B4') and artifact is None else 'FITTED_PROXY_OR_ROUTING_DISABLED'}
            linkage['runtime_recovery_contract']='released_measurements_llm_at_voi_stop.v1' if runtime_recovery else None
            linkage['knowledge_experiment']={'common_knowledge':knowledge_pack is not None,
                'knowledge_fingerprint':knowledge_pack['corpus_fingerprint'] if knowledge_pack else None,
                'memory_time_labels':memory_time_labels,'rule_based':rule_based and condition in ('B3','B4'),
                'evidence_index_set':evidence_set,'knowledge_as_patient_evidence':False}
            _atomic_json(folder/'episode_and_policy.json',{'episode':episode,'source_policy_linkage':linkage})
            prediction=extract_core_prediction(decisions,raw['status'])
            if policies and policies[-1].model.runtime is not None:
                belief=policies[-1].model.runtime.voi_state.get('belief')
                if belief:prediction['algorithmic_diagnosis_choice']=terminal_choice(belief,model['terminal_utility'])[0]
            routing=runtimes[-1].voi_state.get('primary_routing_history',[]) if runtimes else []
            voi=runtimes[-1].voi_state.get('history',[]) if runtimes else []
            return {'status':raw['status'],'prediction':prediction,'request_count':client.request_count,
                'identity_check_count':client.identity_check_count,'action_counts':episode['result']['action_counts'],
                'routing_call_count':len(routing),'routing_selected_count':sum(r.get('decision_status')=='selected' for r in routing),
                'routing_abstention_count':sum(r.get('decision_status')=='abstained' for r in routing),
                'voi_information_action_count':sum(t.get('selected_action') in ('request_test','ask_patient') for t in voi),
                'voi_STOP_count':sum(t.get('selected_action')=='diagnose' for t in voi),
                'client_metadata':client.public_metadata(),'computation':{'wall_seconds':time.monotonic()-start,
                    'planner_calls':len(plans),'evaluated_states':sum(t['evaluated_states'] for t in plans)},
                'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False,**MARKERS}
        finally:client.close()
    return execute_case
