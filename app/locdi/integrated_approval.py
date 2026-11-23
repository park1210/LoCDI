"""Fail-closed integrated LIVE plan. No adapter, server, fitting or executor.

Existing ER-only/static Live entrypoints remain authoritative and untouched.
Even future valid research artifacts need a separately reviewed live executor.
"""
from copy import deepcopy
import re
from app.locdi.llm_setup import ROOT, digest
from app.locdi.live_handoff import load_config, prepare, sha
from app.locdi.dataset_freeze import load
from validation.research_artifact_gate import assess_artifacts

MODES = {'adaptive_routing','bayesian_voi','adaptive_and_voi'}
OUTPUT_ROOT = ROOT/'outputs/development_integrated'


def preflight_integrated(*, run_id, mode, routing_artifact=None, voi_artifact=None,
                         correctness_inputs=None, config=None, budget=10, source_actions=None,
                         diagnosis_candidates=None, evaluation_patient_ids=None):
    result = {'schema_version':'locdi.integrated_approval.v1','status':'dry_run',
        'plan':None,'plan_fingerprint':None,'blocker_codes':[],
        'actual_generation_calls':0,'identity_check_count':0,'research_evaluation_ready':False}
    if not isinstance(run_id,str) or not re.fullmatch('[0-9a-f]{32}',run_id):
        result['blocker_codes']=['EXPLICIT_NEW_RUN_ID_REQUIRED'];return result
    destination=OUTPUT_ROOT/run_id
    if (destination.resolve().parent!=OUTPUT_ROOT.resolve() or not OUTPUT_ROOT.resolve().is_relative_to((ROOT/'outputs').resolve())
            or destination.exists() or (ROOT/'outputs/development_handoff'/run_id).exists()
            or (ROOT/'outputs/development_pilot'/run_id).exists()):
        result['blocker_codes']=['OUTPUT_NAMESPACE_INVALID_OR_USED'];return result
    if mode not in MODES or type(budget) is not int or not 1<=budget<=10:
        result['blocker_codes']=['MODE_OR_BUDGET_INVALID'];return result
    try:
        definition=deepcopy(load_config() if config is None else config)
        definition['max_requests']=budget
        setup=prepare(definition);inputs=setup['plan_inputs']
        case=setup['case'];events=case['environment']['current_encounter_events']
        action_catalog=sorted(set((e['action'],e['target']) for e in events))
        memory={'source':'fixed','mode':'selective','top_k':5,'cutoff':'same_patient_past_visit_available_at_index'}
        binding={'registry_fingerprint':inputs['registry_fingerprint'], 'model_identifier':inputs['model_identifier'],
            'prompt_fingerprint':digest(inputs['stages']), 'action_catalog_fingerprint':digest(action_catalog),
            'memory_fingerprint':digest(memory)}
        readiness=assess_artifacts(setup['prepared']['registry'],routing_artifact=routing_artifact,
            voi_artifact=voi_artifact,correctness_inputs=correctness_inputs,binding=binding,
            evaluation_patient_ids=evaluation_patient_ids,source_actions=source_actions,
            diagnosis_candidates=diagnosis_candidates)
        release=ROOT/'datasets/releases/locdi_dataset_v1.0/freeze_manifest.json'
        plan={'schema_version':'locdi.integrated_live_plan.v1','run_id':run_id,'mode':mode,
            'output_namespace':'outputs/development_integrated/'+run_id,
            'model_identifier':inputs['model_identifier'],'stages':inputs['stages'],
            'registry_fingerprint':inputs['registry_fingerprint'],'registry_file_sha256':inputs['registry_file_sha256'],
            'dataset_source_sha256':inputs['source_fingerprint'],'case_fingerprint':inputs['case_fingerprint'],
            'utility_artifact_hash':digest(routing_artifact) if routing_artifact is not None else None,
            'voi_artifact_hash':digest(voi_artifact) if voi_artifact is not None else None,
            'artifact_readiness':readiness,'current_binding':binding,
            'runtime_configuration':{'mode':mode,'generation':inputs['generation'],'memory':memory,
                'core_coordinator':'er_only','outer_handoff_routing':'adaptive' if mode!='bayesian_voi' else 'static_development',
                'voi_enabled':mode!='adaptive_routing','pre_decision_hook':False},
            'global_request_budget':budget,'budget_includes':['all_stages','semantic_repair','core_repair','retry'],
            'retry_limit':0,'semantic_repair_limit':definition.get('specialty_semantic_repairs',0),
            'action_catalog_fingerprint':digest(action_catalog),
            'release_contract':{'file_sha256':sha(release),'manifest_fingerprint':load(release)['manifest_fingerprint'],
                'availability':'DEVELOPMENT_PROXY_REQUIRES_EXPLICIT_ACCEPTANCE','clinical_release_validation':'NOT_CHECKED'},
            'research_evaluation_ready':False,'clinical_metrics':'NOT_EVALUATED',
            'code_sha256':{**inputs['code_sha256'],**{p:sha(ROOT/p) for p in (
                'app/locdi/integrated_approval.py','validation/research_artifact_gate.py','app/locdi/handoff_integration.py',
                'app/core/information_value.py','app/core/information_outcomes.py','app/core/voi_planning.py',
                'app/core/diagnostic_belief.py','app/core/agent_registry.py','app/core/routing_utility.py','app/core/longitudinal_memory.py')}}}
        required=['routing'] if mode=='adaptive_routing' else ['voi'] if mode=='bayesian_voi' else ['routing','voi']
        result['blocker_codes']=[kind.upper()+'_RESEARCH_ARTIFACT_BLOCKED' for kind in required if readiness['artifacts'][kind]['status']!='READY']
        result['blocker_codes'].append('LIVE_INTEGRATED_EXECUTOR_NOT_IMPLEMENTED')
        result.update(plan=plan,plan_fingerprint=digest(plan))
    except Exception:
        result['blocker_codes']=['INTEGRATED_PLAN_INPUT_CONTRACT_INVALID']
    return result


def request_integrated_approval(*, execute=False, allow_llm=False, accept_release_assumption=False,
                                expected_plan_fingerprint=None, **inputs):
    """Recompute all local inputs before approval; NEVER constructs a client."""
    report=preflight_integrated(**inputs)
    if execute is False:return report
    report['status']='blocked'
    if any(type(v) is not bool for v in (execute,allow_llm,accept_release_assumption)) or not (execute and allow_llm and accept_release_assumption):
        report['blocker_codes'].append('EXPLICIT_LIVE_APPROVAL_REQUIRED')
    if expected_plan_fingerprint is None or expected_plan_fingerprint!=report['plan_fingerprint']:
        report['blocker_codes'].append('APPROVED_PLAN_FINGERPRINT_MISMATCH')
    return report
