"""C4-18 offline replay over one pinned historical run. No HTTP/fit/new policy."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/locdi'))
from app.locdi.dataset_freeze import load,sha
from app.locdi.routing_data_reuse import BANK,SOURCE,read
from app.locdi.llm_setup import digest
from app.locdi.routing_two_decision_collection import verify_pins
from app.locdi.voi_empirical_evaluation import released_observation,replay,decision_key,distribution_stats,reject_labels
from app.core.information_outcomes import InformationOutcomes
from scripts.enrich_full_cohort_sources import write,rows


def compact_sha(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def boundary(events,case_id,h):
    kind={'request_test':'test_ordered','ask_patient':'question_asked','diagnose':'case_completed'}.get(h.get('selected_action'))
    if kind is None:raise ValueError('NO_RECORDED_SELECTED_ACTION')
    matched=[]
    for i,e in enumerate(events):
        if e['case_id']!=case_id or e['minute']!=h['minute'] or e['event_type']!=kind:continue
        if kind=='test_ordered' and e['payload']['requested_test']!=h['target']:continue
        if kind=='question_asked' and e['payload']['question']!=h['target']:continue
        matched.append(i)
    if len(matched)!=1:raise ValueError('EXECUTION_EVENT_BOUNDARY_NOT_UNIQUE')
    return matched[0]


def run():
    pub=ROOT/'outputs/research_evaluation/c4_18_v1';pri=ROOT/'outputs/private/research_evaluation/c4_18_v1'
    if (pri/'execution_started.json').exists():raise ValueError('NO_RESUME_OR_OVERWRITE')
    spec=load(pub/'evaluation_protocol.json');pins=load(pri/'protected_hashes.json');sources=load(pri/'source_hashes.json')
    verify_pins(pins);verify_pins(sources)
    write(pri/'execution_started.json',{'protocol_fingerprint':digest(spec),'script_sha256':sha(__file__),
        'measurement_module_sha256':sha(ROOT/'app/locdi/voi_empirical_evaluation.py')})
    folder=BANK.parent;runpath=folder/'runs'/spec['historical_run'];artifact=load(folder/'voi_model.json')
    if sha(folder/'voi_model.json')!=spec['artifact_sha256']:raise ValueError('ARTIFACT_HASH_MISMATCH')
    model=InformationOutcomes(artifact);cases=load(folder/'cases.json');bycase={c['case_id']:c for c in cases}
    original=read(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    prep=load(ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json')
    training=read(BANK);training_rows=load(SOURCE.parent.parent/'voi_training_rows.json')
    vt=set(artifact['provenance']['training_patients']);hold={c['patient_id'] for c in original}
    rp={c['longitudinal']['patient_id'] for c in cases}
    split=prep['split_manifest']['patients'];rt=set(split['train']);rv=set(split['validation'])
    if vt&rp or rp!=hold or rt&rv:raise ValueError('PATIENT_SPLIT_OR_COHORT_MISMATCH')
    api={}
    with (runpath/'actual_api_calls.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            x=json.loads(line)
            if x['condition']!='B4' or x['purpose']!='decision':continue
            if compact_sha(x['request_body'])!=x['request_sha256']:raise ValueError('HISTORICAL_REQUEST_HASH_MISMATCH')
            api.setdefault(x['request_sha256'],[]).append(x)
    selected=load(pri/'registered_historical_reports.json');full=[];blocked=[];casecounts=[];public=[]
    memory_records=0
    for ordinal,e in enumerate(selected,1):
        case=bycase[e['case_id']];report=load(runpath/e['report_file']);ref='HistoricalCase%03d'%ordinal
        if report['voi_model']!=artifact or len(report['visits'])!=1:raise ValueError('HISTORICAL_MODEL_OR_VISIT_MISMATCH')
        visit=report['visits'][0];identity=visit['identity']
        if identity['patient_id']!=case['longitudinal']['patient_id'] or identity['visit_id']!=case['index_encounter_id']:raise ValueError('PATIENT_ENCOUNTER_MISMATCH')
        fs=visit['final_state'];state=fs['cases'][case['case_id']];histories=state.get('voi_state',{}).get('history',[])
        audits=[a for a in state['llm_audit'] if a.get('status')=='success' and 'specialist_proposal' in a and a.get('selector_status',{}).get('status')=='available']
        used=set();valid=0
        for seq,h in enumerate(histories,1):
            ident={'case_ref':ref,'decision_ordinal':seq,'minute':h['minute']}
            try:
                matches=[(i,a) for i,a in enumerate(audits) if i not in used and a['selector_status'].get('values')==h.get('candidates')]
                if len(matches)!=1:raise ValueError('AUDIT_CANDIDATE_BINDING_NOT_UNIQUE')
                ai,a=matches[0];used.add(ai)
                request_sha=a.get('response_metadata',{}).get('actual_request_sha256')
                wires=[x for x in api.get(request_sha,[]) if x['case_id']==case['case_id'] and x['status']=='http_success']
                if len(wires)!=1:raise ValueError('SAVED_REQUEST_LINK_NOT_UNIQUE')
                wire=wires[0];wire_payload=json.loads(wire['request_body']['messages'][-1]['content'])
                reject_labels(wire_payload.get('observation',{}))
                event_index=boundary(fs['event_log'],case['case_id'],h)
                obs=released_observation(case,fs['event_log'],h['minute'],a['memory_context'],before_event_index=event_index)
                result=replay(artifact,obs,h['candidates'],a['specialist_proposal'])
                if result['belief']!=h['belief'] or result['matched_outcomes']!=h['matched_outcomes']:raise ValueError('OBSERVED_BELIEF_OR_MATCHED_OUTCOME_MISMATCH')
                repeat=replay(artifact,deepcopy(obs),deepcopy(h['candidates']),deepcopy(a['specialist_proposal']))
                if result!=repeat:raise ValueError('POLICY_REPRODUCTION_MISMATCH')
                r2=result['horizons'][str(artifact['horizon'])]['selection']
                saved=a['selector_status']['selected_action']
                result.update(ident,source_report_sha256=sha(runpath/e['report_file']),
                    historical_request_sha256=request_sha,recorded_core_input_sha256=a['input_sha256'],
                    model_observation=obs,existing_specialist_proposal=deepcopy(a['specialist_proposal']),
                    execution_event_index=event_index,source_execution_status=e['engine_status'],
                    saved_selector_reproduced=r2=={k:v for k,v in a['selector_status'].items() if k!='executed_selector'},
                    saved_action_binding_reproduced=decision_key(r2['selected_action'])==decision_key(saved))
                full.append(result);valid+=1;memory_records+=len(a['memory_context']['records'])
                public.append({**ident,'input_fingerprint':result['input_fingerprint'],'candidate_fingerprint':result['candidate_fingerprint'],
                    'artifact_fingerprint':result['artifact_fingerprint'],'matched_outcomes':len(result['matched_outcomes']),
                    'off_action':result['off']['selected_action']['action'],
                    'on_action':r2['selected_action']['action'],'action_changed':result['off_on_action_changed'],
                    'horizon_actions':{k:v['selection']['selected_action']['action'] for k,v in result['horizons'].items()},
                    'horizon_changed':result['horizon_action_changed'],
                    'horizon_optimal_value':{k:v['plan']['optimal_value'] for k,v in result['horizons'].items()},
                    'saved_action_binding_reproduced':result['saved_action_binding_reproduced']})
            except (ValueError,KeyError,TypeError) as exc:
                # Retain the case/decision denominator and exact contract failure.
                reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
                if len(reason)>120 or not reason.replace('_','').isalnum():reason='REPLAY_CONTRACT_ERROR_'+type(exc).__name__
                blocked.append({**ident,'reason':reason})
        casecounts.append({'case_ref':ref,'historical_status':e['engine_status'],'registered_voi_decisions':len(histories),'replayed':valid,'blocked':len(histories)-valid})
        if ordinal%10==0:print('HISTORICAL_REPORTS_REPLAYED',ordinal,'VALID',len(full),'BLOCKED',len(blocked),flush=True)
    rows(pri/'decision_trace.jsonl',full)
    write(pub/'decision_trace.json',{'scope':'HISTORICAL_SNAPSHOT_SELECTOR_REPLAY_NOT_NEW_EPISODE','rows':public,'blocked':blocked,'cases':casecounts,
        'repeat_equal':True,'original_report_count':50,'independent_evaluation':False})
    n=len(full);registered=sum(x['registered_voi_decisions'] for x in casecounts)
    changes=sum(x['off_on_action_changed'] for x in full)
    write(pub/'off_on_comparison.json',{'status':'EXECUTED_SAME_B4_SNAPSHOT_SELECTOR_REPLAY','registered_decisions':registered,'valid_decisions':n,
        'blocked_decisions':len(blocked),'execution_success_rate':n/registered if registered else None,
        'off_policy':'existing specialist_heuristic(saved validated specialist_proposal)','on_policy':'existing bayesian_voi horizon2',
        'changed_actions':changes,'action_change_rate':changes/n if n else None,
        'off_action_counts':dict(Counter(x['off']['selected_action']['action'] for x in full)),
        'on_action_counts':dict(Counter(x['horizons']['2']['selection']['selected_action']['action'] for x in full)),
        'diagnose_is_not_router_abstention':True,'abstention_appropriateness':'NOT_EVALUATED',
        'full_off_on_episode_comparison':'NOT_RUN_REQUIRES_NEW_MODEL_PROPOSALS_AND_REVIEWED_EXECUTION_CONTRACT',
        'invalid_or_blocked_reasons':dict(Counter(x['reason'] for x in blocked)),
        'saved_action_binding_reproduced':sum(x['saved_action_binding_reproduced'] for x in full),
        'saved_selector_full_return_reproduced':sum(x['saved_selector_reproduced'] for x in full),
        'clinical_improvement':'NOT_EVALUATED'})
    summaries={}
    for h in ('1','2','3'):
        ps=[x['horizons'][h]['plan'] for x in full]
        summaries[h]={'decisions':n,'selected_actions':dict(Counter(x['horizons'][h]['selection']['selected_action']['action'] for x in full)),
            'stop_count':sum(p['selected_action_id'] is None for p in ps),
            'candidate_voi':distribution_stats([v['voi'] for p in ps for v in p['action_values']]),
            'candidate_q':distribution_stats([v['q_value'] for p in ps for v in p['action_values']]),
            'terminal_utility':distribution_stats([p['baseline_terminal_utility'] for p in ps]),
            'optimal_value':distribution_stats([p['optimal_value'] for p in ps]),
            'evaluated_states':distribution_stats([p['evaluated_states'] for p in ps])}
    pairs={a+'_'+b:{'action_binding_changes':sum(decision_key(x['horizons'][a]['selection']['selected_action'])!=decision_key(x['horizons'][b]['selection']['selected_action']) for x in full),
        'ranking_changes':sum([v['action_id'] for v in x['horizons'][a]['plan']['action_values']]!=[v['action_id'] for v in x['horizons'][b]['plan']['action_values']] for x in full),
        'optimal_value_changes':sum(x['horizons'][a]['plan']['optimal_value']!=x['horizons'][b]['plan']['optimal_value'] for x in full)} for a,b in [('1','2'),('1','3'),('2','3')]}
    write(pub/'horizon_comparison.json',{'fixed_horizon_sensitivity':'EXECUTED','dynamic_horizon_comparison':'NOT_IMPLEMENTED',
        'dynamic_programming_is_not_adaptive_horizon':True,'horizon_uses':'Artifact scalar only; recursive depth decreases, not minutes/deterioration/future truth',
        'conditions':summaries,'pairs':pairs,'any_action_binding_changed':sum(x['horizon_action_changed'] for x in full),
        'clinical_time_or_risk_effect':'NOT_EVALUATED'})
    write(pub/'memory_impact.json',{'status':'EXECUTED_SAME_RELEASED_SNAPSHOT_DIRECT_MODEL_SENSITIVITY','decisions':n,
        'recorded_memory_mode_counts':dict(Counter(x['model_observation']['longitudinal_memory']['mode'] for x in full)),
        'record_occurrences_identity_cutoff_checked':memory_records,'model_retrieve_memory_action_count':sum(a['type']=='retrieve_memory' for a in artifact['actions']),
        'direct_belief_changes_off_vs_recorded':sum(x['memory_direct_belief_changed'] for x in full),
        'memory_can_affect_existing_specialist_proposal_or_routing':True,
        'memory_off_episode':'NOT_RUN_NEW_MODEL_PROPOSALS_REQUIRED','unsupported_memory_text_not_injected_into_likelihood':True,
        'clinical_memory_usefulness':'NOT_EVALUATED'})
    split_audit={'replay_patients':len(rp),'voi_training_patients':len(vt),'training_bank_patients':len({r['training_patient_id'] for r in training}),
        'training_bank_visits':len(training),'voi_training_rows':len(training_rows),'training_row_patient_count':len({r['patient_id'] for r in training_rows}),
        'voi_train_replay_overlap':len(vt&rp),'replay_original50_overlap':len(rp&hold),
        'routing_train_voi_train_overlap':len(rt&vt),'routing_validation_voi_train_overlap':len(rv&vt),
        'routing_train_validation_overlap':len(rt&rv),'original50_development_exposed':True,
        'query_evaluation_labels_read_into_policy':False,'source_conditions_read_into_policy':False,
        'historical_wire_labeled_training_demonstrations':True,'historical_doctor_proposals_not_independent_reference':True,
        'source_release_proxy_not_clinical_available_at':True,'gold_key_check':'PASS',
        'result_received_and_question_event_order_boundary':'CHECKED_FOR_EACH_REPLAYED_DECISION',
        'same_minute_current_action_result_excluded':True,'raw_llm_wire_reconstruction_claimed':False,
        'derived_projection':'Only model-consumed fields rebuilt from actual saved delivered events and recorded Memory; observe() belief and matches must equal saved trace',
        'patient_disjoint_does_not_mean_blind_holdout':True}
    write(pub/'split_leakage_audit.json',split_audit)
    counts=Counter(r['action_id'] for r in training_rows)
    write(pub/'evaluation_data_inventory.json',{'source_package':'adjacent Synthea experiment34/live_action_policy_v1 (read-only)',
        'historical_run':spec['historical_run'],'historical_all_conditions':dict(Counter(r['condition'] for r in read(runpath/'episodes.jsonl'))),
        'selected_B4_reports':50,'historical_B4_status':dict(Counter(r['engine_status'] for r in selected)),
        'registered_voi_decisions':registered,'valid_source_linked_replays':n,'excluded_decisions':blocked,
        'patients':split_audit,'diagnosis_candidates':len(artifact['diagnoses']),'information_actions':len(artifact['actions']),
        'action_types':dict(Counter(a['type'] for a in artifact['actions'])),'categorical_outcomes_per_action':[len(a['outcomes']) for a in artifact['actions']],
        'train_action_row_counts':sorted(counts.values()),'prior_sum':sum(artifact['prior'].values()),
        'utility_values':sorted({u for v in artifact['terminal_utility'].values() for u in v.values()}),
        'utility_scope':artifact['provenance']['utility_scope'],'horizon':artifact['horizon'],
        'artifact_sha256':sha(folder/'voi_model.json'),'artifact_fingerprint':model.fingerprint,
        'synthetic':artifact['provenance']['synthetic'],'review_status':artifact['provenance']['review_status'],
        'fit_method':artifact['provenance']['fit_method'],'smoothing_alpha':artifact['provenance']['smoothing_alpha'],
        'assumptions':artifact['provenance']['assumptions'],'independent_clinical_reference':0,'actual_clinical_release':'UNKNOWN',
        'new_training_or_probability_estimation':False})
    integration(pub,pri)
    clinical=['Correct Decision Rate','Expected Regret','Actual Regret','Time-to-Correct-Department','Time-to-Final-Diagnosis','Resource Efficiency','Patient Safety Outcomes']
    write(pub/'metric_eligibility.json',{'clinical_metrics':{k:{'status':'NOT_EVALUATED','value':None,'reason':'Independent task/outcome reference, clinically justified utility and evaluation execution contract absent'} for k in clinical},
        'execution_metrics':'Source-linked selector replay only, with registered/replayed/excluded denominators',
        'action_changes':'Not clinical improvement; no counterfactual Episode generated',
        'voi_values':'Unreviewed fitted finite-universe model, dimensionless validation utility',
        'research_evaluation_ready':False,'new_generation':0,'fitting':0})
    verify_pins(pins);verify_pins(sources)
    write(pub/'verification.json',{'status':'PASS','final_status':'C4_18_PARTIAL_EVALUATION' if n else 'C4_18_REFERENCE_BLOCKED',
        'protected_files':len(pins),'source_files':len(sources),'changed_or_missing':0,'replayed_decisions':n,'registered_decisions':registered,
        'blocked_decisions':len(blocked),'new_llm_generation':0,'fitting':0,'clinical_verdicts':0,'core_or_existing_policy_changed':False,
        'research_evaluation_ready':False})
    print('C4_18_REPLAY_DONE',registered,n,Counter(x['reason'] for x in blocked),flush=True)


def integration(pub,pri):
    # Existing TEST_ONLY helpers/algorithm parameters; no new probability fixture.
    from handoff_integration_support import integrated_options
    from app.locdi.handoff_integration import run_handoff_integrated
    from app.locdi.dataset_loader import load_cases
    from app.core.longitudinal_runner import run_with_recovery
    from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
    from app.core.voi_planning import plan_voi
    case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0];out=[]
    for name,routing in [('static_development',False),('adaptive_test_only',True)]:
        options=integrated_options(case,routing=routing,voi=True)
        with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner,patch('app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler,patch('app.core.information_value.plan_voi',wraps=plan_voi) as planner:
            result=run_handoff_integrated(case,options=options,test_only=True,routing=name,output_root=pub/'test_only_integration',export_id='TEST_ONLY_'+name)
        out.append({'mode':name,'scope':'TEST_ONLY_EXISTING_TOY_NOT_EMPIRICAL_MODEL','status':result['status'],
            'blocker_codes':result['blocker_codes'],'export_status':result['export_status'],
            'actual_calls':{'runner':runner.call_count,'scheduler':scheduler.call_count,'plan_voi':planner.call_count},
            'integration_contract_ready':result['integration_contract_ready'],'clinical_metrics':'NOT_EVALUATED'})
    write(pub/'routing_voi_integration.json',{'test_only_actual_executions':out,
        'research_combinations':{'static_plus_voi':'BLOCKED_REVIEWED_ARTIFACT_AND_LIVE_EXECUTOR_MISSING',
        'rule_plus_voi':'NOT_RUN_EXISTING_CORE_ROLE_ROUTING_PRESENT_BUT_NO_REVIEWED_LIVE_INTEGRATED_EXECUTOR',
        'adaptive_plus_voi':'MISSING_UTILITY','abstention_aware_development_plus_voi':'NOT_IMPLEMENTED'},
        'handoff_supported_routing_modes':['static_development','adaptive_test_only','disabled'],
        'new_runtime_link_implemented':False,'existing_adaptive_test_only_is_not_verified_utility':True})


if __name__=='__main__':
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind'}:raise RuntimeError('OFFLINE_NO_GENERATION')
    sys.addaudithook(guard);run()
