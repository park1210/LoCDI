"""Fixed A/B prompt experiment authorization; no automatic approval or generation."""
from datetime import datetime,timezone,timedelta
from app.locdi.llm_setup import digest
from app.locdi.external_reference_readiness import pinned_private_json


def verify_experiment(path,expected_sha256):
    e=pinned_private_json(path,expected_sha256)
    now=datetime.now(timezone.utc)
    if (e.get('schema_version')!='agentclinic.prompt_ablation_authorization.v1'
            or e.get('approval_source')!='EXPLICIT_CURRENT_USER_C3_15_APPROVAL'
            or e.get('approved_by')!='USER_LOCAL_SERVER_OPERATOR'
            or e.get('execute_authorized') is not True or not e.get('user_statement')
            or e.get('order')!=['A','B'] or e.get('cohort')!=[1,2,3,4,5]
            or e.get('condition_budget')!=50 or e.get('combined_generation_budget')!=100
            or e.get('retry_budget')!=0 or e.get('repair_budget')!=0
            or e.get('scoring_scope')!='OFFLINE_EXACT_MATCH_LEXICAL_ONLY'
            or e.get('exposure_policy')!='DEVELOPMENT_EXPOSED_NOT_INDEPENDENT_TEST'):
        raise ValueError('PINNED_NEW_ABLATION_AUTHORIZATION_REQUIRED')
    issued=datetime.fromisoformat(e['issued_at']);expires=datetime.fromisoformat(e['expires_at'])
    if issued.tzinfo is None or expires.tzinfo is None or not issued<=now<expires or expires-issued>timedelta(hours=24):
        raise ValueError('ABLATION_AUTHORIZATION_EXPIRED')
    if set(e.get('conditions',{}))!={'A','B'}:raise ValueError('PAIRED_CONDITIONS_REQUIRED')
    return e


def verify_condition_authorization(record,plan,*,doctor=False):
    if record.get('approved_by')!='USER_LOCAL_SERVER_OPERATOR':raise ValueError('ABLATION_OPERATOR_REQUIRED')
    e=verify_experiment(record['ablation_evidence_path'],record['ablation_evidence_sha256'])
    d=plan['definition'] if doctor else plan['doctor_plan']['definition']
    condition=d.get('prompt_profile')
    if condition not in ('A','B'):raise ValueError('ABLATION_CONDITION_REQUIRED')
    binding=e['conditions'][condition]
    if (binding.get('doctor_plan_fingerprint' if doctor else 'runner_plan_fingerprint')!=digest(plan)
            or binding.get('prompt_sha256')!=d['prompt_sha256']
            or (not doctor and binding.get('namespace')!=plan['namespace'])):
        raise ValueError('ABLATION_PLAN_OR_NAMESPACE_BINDING')
    return e


def verify_execution_order(authority,plan):
    """B may generate only after this experiment's A has a saved terminal summary."""
    r=pinned_private_json(authority.path,authority.sha256)
    e=verify_condition_authorization(r,plan)
    if plan['doctor_plan']['definition']['prompt_profile']=='B':
        from app.locdi.agentclinic_pilot_readiness import output_paths
        from app.locdi.dataset_freeze import load
        public,_=output_paths(e['conditions']['A']['namespace'])
        saved=load(public/'reports/plan.json');summary=load(public/'metrics/summary.json')
        if digest(saved)!=e['conditions']['A']['runner_plan_fingerprint'] or summary.get('mode')!='LIVE':
            raise ValueError('PAIRED_A_COMPLETION_REQUIRED_BEFORE_B')
