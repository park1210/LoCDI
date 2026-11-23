# -*- coding: utf-8 -*-
"""Static files/hashes and saved-result inventory only; stdlib, no model/test imports.

Does not run a Router, Planner, Validator, test, fitting, transport or experiment.
Stored gate results are evidence, never a newly granted clinical admission.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/research_evaluation/FINAL_B012_MEMORY_STABILITY_RECOVERY_v1/retry_002'
PRIVATE=ROOT/'outputs/private/research_evaluation/FINAL_B012_MEMORY_STABILITY_RECOVERY_v1/retry_002'
MANUAL=ROOT/'outputs/research_evaluation/FINAL_B012_FAILURE_RECOVERY_AND_MANUAL_EXECUTION_v1'
MP=ROOT/'outputs/private/research_evaluation/FINAL_B012_FAILURE_RECOVERY_AND_MANUAL_EXECUTION_v1'
FINAL=ROOT/'outputs/research_evaluation/final_exp_v1'
M5=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
M5_SHA='7409b58581d02fad68565e8a9ac311647c07ec968ccc5070b314f26a762c722c'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def pin_report(p):
    # Inspect file contents, not imported project modules or algorithm behavior.
    failed=[];count=0
    for group in ('input_hashes','binding_hashes','code_hashes'):
        for name,expected in p[group].items():
            count+=1;path=ROOT/name
            if not path.is_file() or sha(path)!=expected:failed.append(name)
    return {'checked_files':count,'changed_or_missing_count':len(failed),
            'changed_or_missing_paths':failed,'file_integrity':'MATCH' if not failed else 'MISMATCH'}


def inspect():
    p=read(BASE/'B012_EXECUTION_PROTOCOL.json');source=read(PRIVATE/'source_rows.json')
    selected=set(p['original20_ordinals']);chosen=[r for r in source if r['patient_ordinal'] in selected]
    m=read(M5);matrix=[]
    for key,aid in m['provenance']['action_key_mapping'].items():
        kind,system,code,unit,resource=json.loads(key)
        exact=[r for r in chosen if r['source_event']['target']['code']==code and r['source_event']['evidence'].get('units')==unit]
        code_only=[r for r in chosen if r['source_event']['target']['code']==code]
        action=next(a for a in m['actions'] if a['id']==aid)
        matrix.append({'kind':kind,'code':code,'unit':unit,'M5_system':system,'resource':resource,'M5_action_id':aid,
            'M5_target':action['target'],'additional_rows_code_only':len(code_only),
            'additional_rows_code_and_unit':len(exact),'selected_patients_code_and_unit':len({r['patient_ordinal'] for r in exact}),
            'additional_units':sorted({str(r['source_event']['evidence'].get('units')) for r in code_only}),
            'native_existing_projection':('facts/replay_bin' if kind=='Ask' else 'records/replay_bin') if code in {'72514-3','8462-4','8480-6','8867-4','9279-1'} else 'additional raw numeric',
            'admitted_binding':False,'reason':'CODE_UNIT_ONLY_NOT_EXACT_BINDING; no Source row/single-value/bin/release proof'})
    utility=read(FINAL/'routing_utility_artifact_manifest.json')
    registry=read(FINAL/'routing_3b_agent_registry.json')
    actual=read(BASE/'actual_results_v1/B012_ACTUAL_METRICS.json')
    result={'inspection':'STATIC_FILES_ONLY_NOT_DYNAMICALLY_TESTED','generation':0,'tests_run':0,'fitting_calls':0,
        'B012_pins':pin_report(p),'source_rows_total':len(source),'selected_source_rows':len(chosen),
        'selected_additional_code_unit_types':len({(r['source_event']['target']['code'],r['source_event']['evidence'].get('units')) for r in chosen}),
        'M5_SHA256':sha(M5),'M5_expected_SHA256':M5_SHA,'M5_hash_matches':sha(M5)==M5_SHA,
        'M5_encoding':m['provenance']['observation_encoding'],'M5_horizon':m['horizon'],
        'M5_actions':len(m['actions']),'M5_diagnoses':len(m['diagnoses']),
        'M5_diagnostic_support_outcomes':sum(bool(o['supports_diagnoses']) for a in m['actions'] for o in a['outcomes']),
        'M5_source_mapping':matrix,'utility_manifest':utility,
        'stored_3b_routing_status':read(FINAL/'routing_final_status.json'),
        'stored_3b_technical_pilot':{'requests':read(FINAL/'routing_technical_pilot_results.json')['actual_requests'],
            'complete_scorable_groups':sum(g['complete_scorable'] for g in read(FINAL/'routing_technical_pilot_results.json')['groups'])},
        'registry_specialties':[a['department'] for a in registry['registry']['agents'] if a['role']=='specialty'],
        'registry_agents_share_model_weights':not registry['distinct_weight_models'],
        'current_ER_registry_specialties':sum(a['role']=='specialty' for a in read(FINAL/'er_registry_3b.json')['agents']),
        'states':{'B0':'STATIC_ENTRYPOINT_AVAILABLE_OPERATOR_GATES_REQUIRED','B1_CONTEXT_BUDGETED_HISTORY':'STATIC_ENTRYPOINT_AVAILABLE_OPERATOR_GATES_REQUIRED',
                  'B2':'STATIC_ENTRYPOINT_AVAILABLE_OPERATOR_GATES_REQUIRED','B3':'BLOCKED','B4':'BLOCKED'},
        'clinical_research_evaluation_ready':False,
        'B3_blockers':['NO_ADMITTED_CURRENT_REGISTRY_UTILITY','INDEPENDENT_REFERENCE_MISSING','CURRENT_NATIVE_BACKEND_ER_ONLY','NO_ADAPTIVE_3B_LIVE_HANDOFF_BINDING'],
        'B4_blockers':['B3_BLOCKED','NATIVE_ACTION_OUTCOME_BINDING_MISSING','FROZEN_ENCODING_NOT_ORDINARY_GATE_ACCEPTED','NO_REVIEWED_FINALIZATION_SUPPORT'],
        'manual_run_id':read(MANUAL/'B012_EXECUTION_PROTOCOL.json')['run_id'],
        'manual_run_already_exists':(MP/read(MANUAL/'B012_EXECUTION_PROTOCOL.json')['run_id']).exists(),
        'stored_Run002_conditions':actual['summary']['conditions'],
        'stored_Run002_generation':actual['summary']['generation_attempts'],
        'stored_Source_time_probe':read(ROOT/'outputs/research_evaluation/c4_39_v1/final_source_runtime_verification.json'),
        'scope_note':'Source-time probe uses another admitted cohort, no LLM/scheduler-resource comparison; not B4 on selected10'}
    return result


def comparison(summary):
    rows=[]
    for c in ('B0','B1','B2','B3','B4'):
        t=summary.get('conditions',{}).get(c)
        row={'condition':c,'status':'BLOCKED_NOT_EVALUATED' if t is None else 'SAVED_SOURCE_RESULTS',
            'registered':10,'completed':None,'failed':None,'not_attempted':None,
            'source_diagnosis_success':None,'source_department_success':None,
            'completion_rate_registered':None,'technical_failure_rate_started':None,
            'source_diagnosis_registered_yield':None,'source_diagnosis_supported_rate':None,
            'source_department_registered_yield':None,'source_department_completed_rate':None,
            'supported_scorable':None,'unresolved':None,
            'generation':None,'routing_selection':'NOT_EVALUATED','routing_abstention':'NOT_EVALUATED',
            'VoI_action_STOP':'NOT_EVALUATED','source_time_violations':'NOT_CHECKED_BY_THIS_STATIC_SCRIPT',
            'clinical_accuracy':'NOT_EVALUATED'}
        if t:
            states=t['states'];done=states.get('COMPLETED',0)
            failed=sum(states.get(s,0) for s in ('CONTRACT_FAILED','TIMEOUT','CONTEXT_BLOCKED'))
            started=done+failed+states.get('INCOMPLETE',0)
            diag=t['Source_exact_registered_success_count'];dept=t['Source_department_registered_success_count']
            supported=t['supported_scorable']
            row.update(completed=done,failed=failed,
                not_attempted=states.get('NOT_ATTEMPTED',0),
                source_diagnosis_success=diag,source_department_success=dept,generation=t['generation'],
                completion_rate_registered=done/10,technical_failure_rate_started=failed/started if started else None,
                source_diagnosis_registered_yield=diag/10,source_diagnosis_supported_rate=diag/supported if supported else None,
                source_department_registered_yield=dept/10,source_department_completed_rate=dept/done if done else None,
                supported_scorable=supported,unresolved=t['unresolved'],
                routing_selection='DISABLED_ER_ONLY',routing_abstention='NOT_APPLICABLE_DISABLED',VoI_action_STOP='DISABLED')
        rows.append(row)
    return rows


def saved_workflow(root):
    """Saved-file arithmetic only. Never generate, replay, validate or repair."""
    collection=read(root/'collection.json');rows=[];violations=[]
    for task in collection['rows']:
        count=Counter();n=task.get('backend_episode_ordinal')
        folder=root/'requests'/('Episode%04d'%n) if n else None
        raw=read(folder/'raw_runner_result.json') if folder and (folder/'raw_runner_result.json').exists() else None
        added={};receipts={};starts={};orders={}
        if raw:
            audits=raw.get('synthetic_source_view_audit') or []
            added={a['test_name']:a for a in (audits[0].get('added_tests',[]) if audits else [])}
            for visit in (raw.get('execution') or {}).get('visits',[]):
                for e in visit['final_state']['event_log']:
                    count['event_'+e['event_type']]+=1
                    name=e.get('payload',{}).get('requested_test')
                    if name not in added:continue
                    for kind,target in [('test_ordered',orders),('test_started',starts),('result_received',receipts)]:
                        if e['event_type']==kind:target.setdefault(name,[]).append(e['minute'])
            for name,minutes in receipts.items():
                if (not orders.get(name) or not starts.get(name)
                    or min(starts[name])<min(orders[name])
                    or min(minutes)<max(min(starts[name])+5,added[name]['available_minute'])):
                    violations.append({'episode_ordinal':task['episode_ordinal'],'rule':'ORDER_START_RELEASE_BOUNDARY'})
            count['added_test_orders']=sum(map(len,orders.values()))
            count['added_results_received']=sum(map(len,receipts.values()))
        for bp in sorted(folder.glob('Request*_before.json')) if folder else []:
            b=read(bp);ap=bp.with_name(bp.name.replace('_before','_after'))
            count['before_checkpoints']+=1
            if not ap.exists():count['missing_after']+=1;continue
            a=read(ap);env=a.get('response_envelope') or {}
            count['after_checkpoints']+=1;count['HTTP200']+=a.get('HTTP_status')==200
            count['response_received']+=a.get('response_received') is True
            count['native_contract_VALID']+=a.get('status')=='VALID'
            count['prompt_tokens_reported']+=env.get('prompt_eval_count',0)
            count['output_tokens_reported']+=env.get('eval_count',0)
            count['HTTP_wall_seconds']+=a.get('wall_seconds',0)
            payload=json.loads(b['request']['messages'][1]['content']);observed=payload['observation']
            count['memory_records_provided_occurrences']+=len(observed.get('longitudinal_memory',{}).get('records',[]))
            count['memory_ids_cited_valid_occurrences']+=len((a.get('internal_response') or {}).get('memory_ids',[]))
            count['evidence_indices_valid_occurrences']+=len((a.get('internal_response') or {}).get('evidence_indices',[]))
            for finding in observed.get('objective_findings',[]):
                for name in added:
                    if finding.startswith(name+': ') and (not receipts.get(name) or observed['elapsed_minutes']<min(receipts[name])):
                        violations.append({'episode_ordinal':task['episode_ordinal'],'rule':'ADDITIONAL_OBSERVATION_BEFORE_RESULT_RECEIVED'})
        rows.append({'episode_ordinal':task['episode_ordinal'],'patient_ordinal':task['patient_ordinal'],
            'condition':task['condition'],'status':task['status'],'registered_request_count':task['request_count'],
            'raw_present':raw is not None,'export_present':bool(folder and (folder/'episode_and_raw.json').exists()),
            'observed_counts':dict(count)})
    return {'run_id':root.name,'generation_count_as_saved':collection['generation_count'],
            'global_stop_reason':collection['global_stop_reason'],'rows':rows,
            'source_release_violations':violations,
            'source_release_check_scope':'additional projected objective findings/order/start/receipt only; actual clinical release and other unrecorded payload scope NOT_CHECKED',
            'memory_cutoff_revalidation':'NOT_CHECKED_BY_THIS_SCRIPT; retain existing strict runtime Validator and saved audit',
            'peak_resources':'NOT_RECORDED; token and HTTP wall only, no peak inference',
            'routing':'DISABLED_ER_ONLY','VoI':'DISABLED; no adaptive/VoI counts inferred'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Fresh directory strictly within outputs/research_evaluation')
    parser.add_argument('--new-manual-summary',action='store_true',help='Read already analyzed manual003; never execute it')
    a=parser.parse_args();data=inspect()
    summary=read(MANUAL/'manual_actual_results_v1/summary.json') if a.new_manual_summary else read(BASE/'actual_results_v1/B012_ACTUAL_METRICS.json')['summary']
    expected=read(MANUAL/'B012_EXECUTION_PROTOCOL.json')['run_id'] if a.new_manual_summary else 'synthea_b012_subset_stable_002'
    if summary.get('run_id')!=expected:raise ValueError('SAVED_RUN_ID_MISMATCH')
    data['comparison_run_id']=expected;data['saved_paired_comparisons']=summary.get('paired')
    rows=comparison(summary)
    run_root=MP/expected if a.new_manual_summary else PRIVATE/expected
    workflow=saved_workflow(run_root)
    if a.output:
        path=a.output.resolve();allowed=(ROOT/'outputs/research_evaluation').resolve()
        if allowed not in path.parents:raise ValueError('OUTPUT_PATH_OUTSIDE_RESEARCH_EVALUATION')
        path.mkdir(parents=True,exist_ok=False)
        with (path/'static_readiness.json').open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        with (path/'B0_B4_saved_comparison.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with (path/'source_action_mapping.json').open('x',encoding='utf-8') as f:json.dump(data['M5_source_mapping'],f,ensure_ascii=False,indent=2)
        with (path/'saved_workflow_counts.json').open('x',encoding='utf-8') as f:json.dump(workflow,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'static_readiness':data,'comparison':rows,'saved_workflow':workflow},ensure_ascii=True,allow_nan=False))


if __name__=='__main__':main()
