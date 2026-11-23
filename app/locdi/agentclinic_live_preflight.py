"""Static external pilot binding plus private pinned execution authority. No server probe."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from app.locdi.llm_setup import ROOT, digest, validate_setup
from app.locdi.dataset_freeze import sha
from app.locdi.external_reference_readiness import pinned_private_json
from app.locdi.agentclinic_scenario_adapter import load_scenarios, action_catalog
from app.locdi.agentclinic_scenario_schema import REVISION, SOURCE_SHA256, GROUP, POLICY_VERSION
from app.locdi.agentclinic_doctor_client import (system_prompt, PROMPT_VERSION, PROMPT_SHA256,
                                               CLIENT_VERSION, CATALOG_POLICY)

SOURCE = ROOT/'outputs/private/research_evaluation/c3_10_v1/upstream_snapshot/agentclinic_medqa.jsonl'
RUNTIME_VERSION = 'agentclinic.prefix_source_runtime.v2'


def proposal(prompt_profile=None):
    import json
    legacy = json.loads((ROOT/'configs/development_specialty_llm.json').read_text(encoding='utf8'))['llm']
    legacy.update(max_requests=50, max_output_tokens=1024, request_retries=0)
    from app.locdi.agentclinic_doctor_client import prompt_binding
    version,_,prompt_sha=prompt_binding(prompt_profile)
    result={'schema_version': 'agentclinic.live_plan.v1', 'source_revision': REVISION,
            'source_sha256': SOURCE_SHA256, 'scenario_ordinals': [1, 2, 3, 4, 5],
            'evaluation_group': GROUP, 'scenario_schema_version': 'agentclinic.osce.v1',
            'prompt_version': version, 'prompt_sha256': prompt_sha,
            'client_version': CLIENT_VERSION, 'runtime_version': RUNTIME_VERSION,
            'catalog_version': POLICY_VERSION, 'catalog_policy': CATALOG_POLICY,
            'legacy_config': legacy, 'max_turns_per_case': 10, 'total_generation_budget': 50,
            'retry_budget': 0, 'repair_budget': 0, 'max_prompt_bytes': 16384,
            'context_tokens': 32768, 'exposure_policy': 'DEVELOPMENT_EXPOSED_NOT_INDEPENDENT_TEST'}
    if prompt_profile is not None:result['prompt_profile']=prompt_profile
    return result


def preflight(definition):
    blockers, plan = [], None
    try:
        d = deepcopy(definition)
        expected = proposal(d.get('prompt_profile'))
        if set(d) != set(expected) or any(d[k] != expected[k] for k in (
                'schema_version','source_revision','source_sha256','evaluation_group','scenario_schema_version',
                'prompt_version','prompt_sha256','client_version','runtime_version','catalog_version',
                'catalog_policy','exposure_policy')):
            raise ValueError('FIXED_BINDING')
        ordinals = d['scenario_ordinals']
        if (not isinstance(ordinals,list) or not 1 <= len(ordinals) <= 5
                or any(type(i) is not int or not 1 <= i <= 107 for i in ordinals)
                or len(set(ordinals)) != len(ordinals)):
            raise ValueError('SCENARIO_SELECTION')
        if (type(d['max_turns_per_case']) is not int or not 1 <= d['max_turns_per_case'] <= 20
                or type(d['total_generation_budget']) is not int or not 1 <= d['total_generation_budget'] <= 100
                or type(d['max_prompt_bytes']) is not int or not 1024 <= d['max_prompt_bytes'] <= 131072
                or type(d['context_tokens']) is not int or not 4096 <= d['context_tokens'] <= 262144
                or type(d['retry_budget']) is not int or d['retry_budget'] != 0
                or type(d['repair_budget']) is not int or d['repair_budget'] != 0):
            raise ValueError('BUDGET')
        cfg = d['legacy_config']
        if cfg['max_requests'] != d['total_generation_budget'] or cfg['request_retries'] != 0:
            raise ValueError('LEGACY_BUDGET_BINDING')
        if d['max_prompt_bytes'] + cfg['max_output_tokens'] + 512 > d['context_tokens']:
            raise ValueError('CONSERVATIVE_CONTEXT_BUDGET')
        system_prompt(d.get('prompt_profile'))
        bundle = load_scenarios(SOURCE, revision=d['source_revision'])
        selected = [bundle.scenarios[i-1] for i in ordinals]
        setup = validate_setup(cfg)
        if setup['static_readiness'] != 'READY':
            blockers.append('LEGACY_MODEL_REGISTRY_SETUP_NOT_READY')
        versions = ['app/locdi/agentclinic_doctor_client.py','app/locdi/agentclinic_live_preflight.py',
                    'app/locdi/agentclinic_doctor_runtime.py','app/locdi/agentclinic_scenario_adapter.py',
                    'app/locdi/agentclinic_turn_runtime.py','app/locdi/agentclinic_scenario_schema.py',
                    'app/locdi/agentclinic_doctor_exporter.py','app/locdi/agentclinic_benchmark_scoring.py',
                    'app/locdi/ollama_adapter.py','app/llm/role_call.py']
        plan = {'definition': d, 'scenarios': {s.scenario_id: {
            'record_fingerprint': s.source['record_fingerprint'],
            'objective_fingerprint': digest(s.objective), 'catalog_fingerprint': digest(action_catalog(s))}
            for s in selected}, 'legacy_setup_fingerprint': (setup.get('prepared') or {}).get('fingerprint'),
            'implementation_hashes': {n: sha(ROOT/n) for n in versions},
            'input_budget_method': 'UTF8_BYTES_CONSERVATIVE_UNIT_NOT_ACTUAL_TOKENIZER',
            'server_context_enforcement': 'REQUIRES_PINNED_OPERATOR_REVIEW; adapter does not set num_ctx'}
    except Exception:
        blockers.append('SOURCE_MODEL_PROMPT_OR_BUDGET_BINDING_INVALID')
    return {'schema_version':'agentclinic.live_preflight.v1','mode':'DRY_RUN',
            'static_blockers':sorted(set(blockers)), 'plan':plan,
            'plan_fingerprint':digest(plan) if plan else None,
            'observed_server_digest':'UNKNOWN_NOT_CHECKED','identity_checks':0,
            'execution_approval':'NOT_PROVIDED', 'live_execution_ready':False,
            'blocker_codes':sorted(set(blockers+['PINNED_EXECUTION_APPROVAL_REQUIRED',
                'SERVER_IDENTITY_NOT_CHECKED','CONTEXT_CAPACITY_REVIEW_REQUIRED'])),
            'real_generation_calls':0,'research_evaluation_ready':False}


class PinnedDoctorApproval:
    def __init__(self, path, expected_sha256):
        self.path, self.sha256 = path, expected_sha256

    def verify(self, plan, now=None):
        a = pinned_private_json(self.path, self.sha256)
        if plan['definition'].get('prompt_profile') is not None:
            from app.locdi.agentclinic_prompt_ablation import verify_condition_authorization
            verify_condition_authorization(a,plan,doctor=True)
        if 'approval_basis_type' in a:
            from app.locdi.agentclinic_context_bootstrap import verify_basis
            verify_basis(a,'LIVE_AGENTCLINIC_DOCTOR_ONLY')
        if (a.get('schema_version') != 'agentclinic.doctor_approval.v1'
                or a.get('scope') != 'LIVE_AGENTCLINIC_DOCTOR_ONLY'
                or a.get('plan_fingerprint') != digest(plan) or not a.get('approved_by')):
            raise ValueError('DOCTOR_APPROVAL_SCOPE')
        now = now or datetime.now(timezone.utc)
        issued, expires = datetime.fromisoformat(a['issued_at']), datetime.fromisoformat(a['expires_at'])
        if issued.tzinfo is None or expires.tzinfo is None or not issued <= now < expires:
            raise ValueError('DOCTOR_APPROVAL_VALIDITY')
        review = pinned_private_json(a['review_path'], a['review_sha256'])
        if (review.get('plan_fingerprint') != digest(plan) or not review.get('reviewed_by')
                or review.get('review_status') != 'APPROVED'
                or review.get('exposure_policy') != plan['definition']['exposure_policy']
                or review.get('verified_server_context_tokens') != plan['definition']['context_tokens']
                or not review.get('server_context_configuration_reference')):
            raise ValueError('EXPOSURE_OR_CONTEXT_REVIEW_REQUIRED')
        return {'approval_sha256':self.sha256,'clinical_reference_approved':False}
