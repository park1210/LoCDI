"""One separately approved, episode-isolated B0/B2 recovery; no retries.

Existing native backend, Core, Evidence Validator and prompts are read-only.
Default prepare is non-generating. Completed namespaces are never resumed.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_research_gaps import read, rows, sha, write, check
from scripts.run_final_3b_evaluation import metadata, DIGEST3, MODEL3, csv_file
from app.locdi.llm_setup import digest, validate_setup
from app.locdi.dataset_loader import load_cases
from app.locdi.current_ablation_llm_backend import make_er_backend
from app.locdi.current_ablation_runner import _process_private_bytes
from app.locdi.memory_paired_recovery import collect, aggregate, schedule, citation_audit, SharedRecoveryStop

PUB = ROOT/'outputs/research_evaluation/final_exp_recovery_v1'
PR = ROOT/'outputs/private/research_evaluation/final_exp_recovery_v1'
PARENT = ROOT/'outputs/private/research_evaluation/final_exp_v1'
SCOPE = 'FINAL_EXP_RECOVERY_FROZEN20_B0_B2_ONLY'
RUN = 'memory_recovery_001'
DATA = ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'


def now(): return datetime.now(timezone.utc).isoformat()


def load_protocol():
    final=PUB/'final_memory_protocol_final.json'
    return read(final if final.exists() else PUB/'final_memory_protocol.json')


def prepare():
    if PUB.exists() or PR.exists(): raise ValueError('FRESH_RECOVERY_NAMESPACE_REQUIRED')
    # Preserve every parent artifact, not just selected successful responses.
    protected = list((ROOT/'app/core').glob('*.py'))
    for folder in ['outputs/research_evaluation/final_exp_v1','outputs/private/research_evaluation/final_exp_v1',
                   'docs/연구 평가/FINAL_EXP','docs/연구 평가/FINAL_THESIS',
                   'outputs/research_evaluation/c4_35_v1','outputs/research_evaluation/c4_39_v1']:
        protected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    protected += [DATA, ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json',
        ROOT/'docs/연구 평가/FINAL_EXP_3B_MEMORY_ROUTING_EVALUATION_REPORT.md']
    protected += list((ROOT/'prompts').rglob('*.txt')) + list((ROOT/'configs').rglob('*.json'))
    protected += [ROOT/p for p in ['app/locdi/current_ablation_runner.py','app/locdi/current_ablation_llm_backend.py',
        'app/locdi/run_simulation.py','app/locdi/case_adapter.py','app/locdi/llm_setup.py']]
    pins = {str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    panel = read(PARENT/'memory_panel.json'); index = read(PARENT/'memory_case_index.json')
    if len(panel)!=20 or list(index.values())!=list(range(11,31)):
        raise ValueError('ORIGINAL_FIXED20_PANEL_CHANGED')
    schedule(panel)
    config = read(PARENT/'llm_config_3b.json')
    if config['model']!=MODEL3 or config['model_identifier']!=DIGEST3 or validate_setup(config)['static_readiness']!='READY':
        raise ValueError('UNCHANGED_3B_BINDING_REQUIRED')
    model = [m for m in metadata('tags')['models'] if m['name']==MODEL3]
    if len(model)!=1 or 'sha256:'+model[0]['digest']!=DIGEST3: raise ValueError('SERVER_DIGEST_MISMATCH')
    old = PARENT/'live/final_exp_001/memory_requests/Episode0010'
    before = read(old/'Request0006_before.json'); after = read(old/'Request0006_after.json')
    audit = citation_audit(before['request'],after['response_envelope'])
    if audit['error_code']!='SUPPORTED_DIAGNOSIS_WITHOUT_ELIGIBLE_OBJECTIVE_CITATION':
        raise ValueError('PARENT_FAILURE_EVIDENCE_CHANGED')
    audit.update(prompt_changed=False, validator_changed=False,
        previous_before_SHA256=sha(old/'Request0006_before.json'),previous_after_SHA256=sha(old/'Request0006_after.json'),
        zero_based_instruction_present='zero-based indices' in json.loads(before['request']['messages'][1]['content'])['instruction'],
        ambiguity='One-based-index intention cannot be inferred; no automatic remap. Objective entries exist; clinical sufficiency is not established.')
    lat=[read(p)['wall_seconds'] for p in (PARENT/'live/final_exp_001/memory_requests').glob('Episode*/Request*_after.json')]
    old_rows=rows(PARENT/'final_exp_memory_001/episodes.jsonl')
    walls=[r['result']['computation']['wall_seconds'] for r in old_rows if r['status']=='completed']
    write(PR/'protected_input_hashes.json',pins);write(PR/'panel.json',panel);write(PR/'case_index.json',index);write(PR/'llm_config.json',config)
    write(PUB/'prior_failure_evidence_audit.json',audit)
    write(PUB/'latency_and_generation_estimate.json',{'observed_requests':len(lat), 'mean_HTTP_seconds':statistics.mean(lat),
        'median_HTTP_seconds':statistics.median(lat),'max_HTTP_seconds':max(lat),
        'prior_requests_per_attempted_episode':len(lat)/10,
        'planning_estimate_requests':240,'HTTP_seconds_at_prior_mean_240':240*statistics.mean(lat),
        'maximum_requests':480,'HTTP_seconds_at_prior_mean_480':480*statistics.mean(lat),
        'completed_episode_mean_wall_seconds':statistics.mean(walls),'40_episode_extrapolation_seconds':40*statistics.mean(walls),
        'extrapolation_limit':'Prior B0 only, B2 longer context/cold loading differ; no completion guarantee.',
        'sum_of_request_timeouts_seconds':480*120,'global_wall_seconds':5400,'generation_requests':0})
    write(PUB/'read_only_environment.json',{'at':now(),'server':metadata('version'),'loaded':metadata('ps'),'model':model,
        'applied_context':'NOT_CHECKED_WITHOUT_NEW_GENERATION','requested_context':32768,'generation_requests':0})
    protocol={'version':'final_exp_recovery.v1','scope':SCOPE,'created_at':now(),'run_id':RUN,
        'purpose':'development_source_evaluation','registered_patients':20,'registered_episodes':40,
        'source_panel_fingerprint':digest(panel),'original_ordinals':list(range(11,31)),
        'order':'patient input order; each B0 then B2; register both even if one fails',
        'schedule':schedule(panel),'model':MODEL3,'model_identifier':DIGEST3,'context':32768,
        'temperature':0,'seed':42,'max_output_tokens':1024,'timeout_seconds':120,
        'max_generations':480,'per_episode_generations':12,'retry':0,'repair':0,'warmup':0,
        'wall_seconds':5400,'process_memory_bytes':4*1024**3,'output_bytes':1024**3,
        'minute_limit':config['minute_limit'],'Memory':{'B0':'none','B2':'existing selective Top5'},'routing':'er_only','VoI':False,
        'Prompt':'unchanged original ER/patient/dynamic Citation instruction; exact existing binding',
        'Evidence_Validator':'unchanged Core; never rewrite citation or certainty',
        'failure_isolation':'Episode-local contract/timeout/context failure preserved; continue every registered B0 and B2 once',
        'global_stop':'identity/source/approval/global-resource/global-request boundary; preserve remaining NOT_ATTEMPTED',
        'resource_enforcement':'before/after Episodes, before each POST reservation; one call/write can transiently exceed resource caps',
        'no_previous_results_merged':True,'resume':False,'server_restart_download_unload':False,
        'recorded_time_proxy':'requires new operator acceptance for development generation only',
        'concept_agreement':'NOT_EVALUATED; no pre-frozen independent equivalences',
        'clinical_accuracy':'NOT_EVALUATED','routing_generation':0,'utility_fitting':0,'VoI_refitting_or_evaluation':0,
        'source_reference_only_in_private_scorer':True,'only_collection_policy_changed':True,
        'approval_validity_max_hours':24}
    inputs=list(PR.glob('*.json'))+list(PUB.glob('*.json'))
    code=['scripts/run_memory_paired_recovery.py','app/locdi/memory_paired_recovery.py',
          'scripts/finalize_memory_paired_recovery.py','tests/locdi/test_memory_paired_recovery.py']
    protocol['input_hashes']={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    protocol['input_hashes'].update({p:sha(ROOT/p) for p in code})
    protocol['protected_manifest_SHA256']=sha(PR/'protected_input_hashes.json')
    protocol['fingerprint']=digest(protocol)
    write(PUB/'final_memory_protocol.json',protocol)
    write(PUB/'approval_status.json',{'status':'NEW_EXPLICIT_APPROVAL_PENDING','protocol_fingerprint':protocol['fingerprint'],'generation_calls':0})
    print(json.dumps({'status':'PREPARED_NOT_EXECUTED','protocol':protocol['fingerprint'],'protected_files':len(pins),'maximum_generation':480}))


def verify_approval(record,p):
    if (record.get('approved') is not True or record.get('scope')!=SCOPE or record.get('plan_fingerprint')!=p['fingerprint']
        or record.get('evidence_type')!='USER_EXPLICIT' or not record.get('user_statement')):
        raise SharedRecoveryStop('NEW_EXPLICIT_RECOVERY_APPROVAL_REQUIRED')
    issued=datetime.fromisoformat(record['recorded_at']);expires=datetime.fromisoformat(record['expires_at'])
    if issued.tzinfo is None or expires.tzinfo is None or not issued<=datetime.now(timezone.utc)<expires or not 0<(expires-issued).total_seconds()<=86400:
        raise SharedRecoveryStop('APPROVAL_NOT_CURRENT')


def inspect_failed_folder(folder,error_type):
    errors=[error_type] if error_type else []; citations=[]
    for after in sorted(Path(folder).glob('Request*_after.json')):
        a=read(after)
        if a.get('error_type'): errors.append(a['error_type'])
        before=after.with_name(after.name.replace('_after','_before'))
        if a.get('response_envelope') and before.exists():
            try: citations.append(citation_audit(read(before)['request'],a['response_envelope']))
            except (ValueError,KeyError,TypeError): pass
    raw=Path(folder)/'raw_runner_result.json'
    if raw.exists():
        r=read(raw);errors += (r.get('llm_execution') or {}).get('error_types',[])
    return {'error_types':[e for e in errors if e], 'citations':citations,
        'shared_integrity_failure':any('ModelIdentity' in str(e) or 'SharedRecoveryStop' in str(e) for e in errors),
        'original_checkpoints_preserved':True,'response_rewrite':False}


def execute(approval_file):
    p=load_protocol();approval=read(approval_file);verify_approval(approval,p)
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}): raise ValueError('PROTOCOL_HASH')
    protected=read(PR/'protected_input_hashes.json')
    if check(p['input_hashes'])['changed_or_missing'] or check(protected)['changed_or_missing']:
        raise SharedRecoveryStop('PINNED_SOURCE_OR_CODE_CHANGED')
    folder=PR/RUN;folder.mkdir(parents=True,exist_ok=False)
    write(folder/'started.json',{'at':now(),'approval':approval,'protocol_fingerprint':p['fingerprint']})
    panel=read(PR/'panel.json');index=read(PR/'case_index.json');config=read(PR/'llm_config.json')
    cases=load_cases(DATA);begin=time.monotonic()
    watched={**protected,**p['input_hashes']}
    stats={f:((ROOT/f).stat().st_size,(ROOT/f).stat().st_mtime_ns) for f in watched}
    def guard():
        verify_approval(approval,p)
        for f,s in stats.items():
            try:st=(ROOT/f).stat()
            except OSError:raise SharedRecoveryStop('PINNED_SOURCE_DISAPPEARED') from None
            if (st.st_size,st.st_mtime_ns)!=s:raise SharedRecoveryStop('PINNED_SOURCE_CHANGED')
        if time.monotonic()-begin>=p['wall_seconds']:raise SharedRecoveryStop('GLOBAL_WALL_LIMIT')
        if _process_private_bytes()>=p['process_memory_bytes']:raise SharedRecoveryStop('GLOBAL_PROCESS_MEMORY_LIMIT')
        size=sum(x.stat().st_size for root in (PR,PUB) for x in root.rglob('*') if x.is_file())
        if size>=p['output_bytes']:raise SharedRecoveryStop('GLOBAL_OUTPUT_LIMIT')
    native=make_er_backend({r:cases[n-1] for r,n in index.items()},config=config,
        contract_fingerprint=p['fingerprint'],checkpoint_root=folder/'requests',authorized=True)
    execution_count=[0]
    def backend(execution_case,features):
        execution_count[0]+=1
        return native(execution_case,features)
    def failure(task,error):
        request_folder=folder/'requests'/('Episode%04d'%execution_count[0])
        detail=inspect_failed_folder(request_folder,error)
        # Export what Core actually recorded, including a failure, if the
        # existing exporter supports it. No synthetic success or invented event.
        raw_path=request_folder/'raw_runner_result.json'
        if raw_path.exists():
            try:
                from app.locdi.case_adapter import adapt_case
                from app.locdi.episode_exporter import build_episode
                raw=read(raw_path)
                episode=build_episode(raw,context=adapt_case(cases[index[task['case_ref']]-1],registry_manifest=read(ROOT/config['registry_path'])))
                write(request_folder/'partial_episode.json',episode)
                detail['partial_export']='EXPORTED_RECORDED_STATE'
            except Exception as exc:
                detail['partial_export']='RAW_ONLY';detail['partial_export_error_type']=type(exc).__name__
        return detail
    result=collect(panel,backend,trace_path=folder/'episodes.jsonl',total_cap=p['max_generations'],
        per_episode_cap=p['per_episode_generations'],guard=guard,audit_failure=failure)
    metrics=aggregate(panel,result['rows'])
    write(folder/'aggregate.json',metrics)
    write(folder/'finished.json',{k:v for k,v in result.items() if k!='rows'} | {'wall_seconds':time.monotonic()-begin,'protocol_fingerprint':p['fingerprint']})
    print(json.dumps({'generation_count':result['generation_count'],'complete_pairs':metrics['complete_pairs'],
        'states':{c:m['states'] for c,m in metrics['conditions'].items()},'global_stop':result['global_stop_reason']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','execute'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true');parser.add_argument('--approval-file')
    a=parser.parse_args()
    if a.mode=='prepare':prepare()
    elif not (a.execute and a.allow_llm and a.approval_file):parser.error('new exact approval + --execute + --allow-llm required')
    else:execute(a.approval_file)
