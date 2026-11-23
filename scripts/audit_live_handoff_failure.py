"""Read-only audit of the specified failed run. No client or model calls."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.llm_setup import ROOT,digest

RUN_ID='20261029000000000000000000002901'
REQUIRED=('reports/preflight.json','reports/started.json','reports/pilot.json','reports/runtime.json','reports/handoff.json',
    'private/Request0001_before.json','private/Request0001_after.json','private/Request0002_before.json','private/Request0002_after.json',
    'episodes/Case001.json','metrics/summary.jsonl')


def inspect():
    root=ROOT/'outputs/development_handoff'/RUN_ID
    present={name:json.loads((root/name).read_text(encoding='utf-8')) for name in REQUIRED if (root/name).is_file()}
    missing=[name for name in REQUIRED if name not in present]
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    result={'schema_version':'locdi.phase29b.saved_failure_audit.v1','run_id':RUN_ID,'files_present':list(present),
        'files_missing':missing,'file_sha256':hashes,'new_model_generations':0,'utility_fit_count':0,'clinical_verdicts_generated':0}
    if missing:result['status']='INCOMPLETE_SAVED_EVIDENCE';return result
    pre=present['reports/preflight.json'];pilot=present['reports/pilot.json'];audit=present['reports/handoff.json']
    episode=present['episodes/Case001.json'];summary=present['metrics/summary.jsonl'];runtime=present['reports/runtime.json']
    requests=audit['requests'];stages=[r['stage'] for r in requests]
    stable=('request_id','stage','input_fingerprint','prompt_fingerprint','schema_fingerprint','response_fingerprint',
        'schema_validation','status','generation_count_after','identity_check_count_after')
    plan=pre['plan'];baseline=json.loads((ROOT/'outputs/private/phase29b_baseline_hashes_v1.json').read_text(encoding='utf-8'))
    result.update(status='FAILED_AND_EXPORTED',schema_response_statuses=[{'stage':r['stage'],'status':r['status'],
        'schema_validation':r['schema_validation']} for r in requests],
        request_count=pilot['llm_execution']['request_count'],identity_check_count=pilot['llm_execution']['identity_check_count'],
        counts_match=(audit['request_count']==summary['episodes'][0]['llm_execution']['request_count']==episode['llm_execution']['request_count']==2
            and audit['identity_check_count']==summary['episodes'][0]['llm_execution']['identity_check_count']==episode['llm_execution']['identity_check_count']==2),
        schema_transport_success=all(r['status']=='response_received' and r['schema_validation']=='VALID' for r in requests),
        digest_match=all(r['pinned_model_digest']==r['observed_server_digest']==plan['model_identifier'] for r in requests),
        stage_binding_matches=all(all(r[k]==plan['stages'][r['stage']][k] for k in ('agent_id','model','version'))
            and r['prompt_fingerprint']==plan['stages'][r['stage']]['system_prompt_fingerprint'] for r in requests),
        approved_code_matches_pre_repair_baseline=all(baseline.get(p)==value for p,value in plan['code_sha256'].items()),
        plan_fingerprint_valid=digest(plan)==pre['plan_fingerprint']==pilot['plan_fingerprint']==audit['plan_fingerprint'],
        checkpoint_matches_final_requests=all(all(present['private/'+r['request_id']+'_after.json']['request'][k]==r[k] for k in stable) for r in requests),
        handoff_requested_recorded=bool(audit['handoffs'] and audit['handoffs'][0]['source_decision_id']==requests[0]['decision_id']),
        target_department=audit['handoffs'][0]['target_department'],request_stages=stages,
        er_return_requests=sum(stage=='ER_REASSESSMENT' for stage in stages),
        failure_code=pilot['blocker_codes'],core_status=pilot['core_status'],export_status=pilot['export_status'],
        sequential_handoff_verified=pilot['sequential_handoff_verified'],
        workflow_matches=runtime['workflow']==episode['workflow'],terminal_matches=runtime['terminal']==episode['result'],
        configuration_fingerprint_matches=audit['configuration_fingerprint']==summary['episodes'][0]['configuration_fingerprint']==episode['configuration_fingerprint'],
        terminal_status_matches=pilot['core_status']==episode['result']['status']==summary['episodes'][0]['status']=='failed',
        action_counts=episode['result']['action_counts'],finalized=episode['result']['finalized'],
        elapsed_simulation_minutes=episode['result']['elapsed_simulation_minutes'],
        memory_cutoff_compliant=all(r['cutoff_compliant'] for r in episode['memory']['retrievals']),
        raw_response_availability='NOT_SAVED_IN_LISTED_RUN_FILES',
        exact_returned_indices='UNKNOWN_WITHOUT_RAW_RESPONSE',exact_returned_certainty='UNKNOWN_WITHOUT_RAW_RESPONSE',
        evidence_level='CONFIRMED_FROM_SAVED_EVIDENCE')
    return result


if __name__=='__main__':print(json.dumps(inspect(),ensure_ascii=True,indent=2))
