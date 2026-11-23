"""Saved-file-only analysis. No inference, fitting, policy rerun or old writes.

Recover the existing Core terminal contract in a separate report; preserve the
historical summary which erroneously searched for pre-normalization 'diagnose'.
Clinical references and diagnosis text remain in the Private analysis output.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.run_simulation import _digest
from app.locdi.synthea_diagnosis_audit import matches
from app.core.information_outcomes import InformationOutcomes
from app.core.longitudinal_memory import timestamp
from jsonschema import Draft202012Validator

NS='B34_COMPLETED_RUN_ANALYSIS_20261121_v3'
NAMES=('recovery_llm_b3_001','recovery_llm_b4_001','recovery_llm_all_001')
PUB=ROOT/'outputs/research_evaluation'/NS
PRIVATE=ROOT/'outputs/private/research_evaluation'/NS
DOC=ROOT/'docs'/'\uc5f0\uad6c \ud3c9\uac00'/NS
ACTION={'ask_patient':'ask_history','request_test':'order_test','consult_specialty':'request_specialty','diagnose':'er_finalize'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=True,indent=2,allow_nan=False)
def write_csv(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(xs):return statistics.mean(xs) if xs else None
def key_hits(obj,keys):
    if isinstance(obj,dict):return sum(k.casefold() in keys for k in obj)+sum(key_hits(v,keys) for v in obj.values())
    if isinstance(obj,list):return sum(key_hits(v,keys) for v in obj)
    return 0


def main():
    for p in (PUB,PRIVATE,DOC):p.mkdir(parents=True,exist_ok=False)
    inputs=set()
    for name in NAMES:
        for prefix in ('outputs/private/research_evaluation/B34_CONNECTION_v1','outputs/research_evaluation/B34_CONNECTION_v1'):
            inputs.update(p for p in (ROOT/prefix/name).rglob('*') if p.is_file())
    m5path=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
    inputs.add(m5path);model=read(m5path);engine=InformationOutcomes(model)
    before={str(p.resolve()):sha(p) for p in inputs};write(PRIVATE/'input_hashes_before.json',before)
    rows=[];private_rows=[];runs={};issues=[];counters=Counter();memory=defaultdict(Counter);route=defaultdict(Counter)
    proposal_actions=defaultdict(Counter);selected_actions=defaultdict(Counter);trace_fp={};payload_fp={};response_fp={}
    ids=set();payload_id_hits=Counter();preview_signals={};request_trace=[]
    for name in NAMES:
        pr=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1'/name;run=pr/('run_'+name);pub=ROOT/'outputs/research_evaluation/B34_CONNECTION_v1'/name
        plan=read(pub/'plan.json');config=read(pr/'llm_config.json');panel=read(pr/'panel.json');bindings=read(pr/'bindings.json');collection=read(run/'collection.json');saved=read(pub/'saved_results.json')
        ids.update(b[k] for b in bindings.values() for k in ('patient_id','visit_id','case_id'))
        line_rows=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if s]
        assert line_rows==collection['rows'];assert collection['plan_fingerprint']==plan['fingerprint']
        assert plan['fingerprint']==digest({k:v for k,v in plan.items() if k!='fingerprint'})
        assert collection['generation']==sum(r['request_count'] for r in line_rows)==saved['actual_generation']
        pin_bad=[n for n,h in {**plan['code_hashes'],**plan['input_hashes']}.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
        if pin_bad:issues.append({'run':name,'code':'CURRENT_PLAN_PIN_DRIFT','count':len(pin_bad)})
        info={'generation':collection['generation'],'states':dict(Counter(r['status'] for r in line_rows)),
            'registered':len(line_rows),'wall_seconds':collection['wall_seconds'],'stop_reason':collection['stop_reason'],
            'plan_pin_mismatches':len(pin_bad),'request_budget':plan['generation_total'],'wall_budget_seconds':plan['wall_seconds']}
        runs[name]=info
        for r in line_rows:
            co=r['condition'];n=r['episode_ordinal'];ordinal=r['patient_ordinal'];folder=run/'requests'/('Episode%04d'%n)
            raw=read(folder/'raw_runner_result.json');decisions=read(folder/'actual_core_decisions.json');rt=read(folder/'runtime_snapshot_private.json');linked=read(folder/'episode_and_policy.json');episode=linked['episode'];plans=read(folder/'planner_trace.json')
            reference=panel[ordinal-1]['reference'];binding=bindings[r['case_ref']];identity=raw['execution']['visits'][0]['identity']
            log=raw['execution']['visits'][0]['final_state']['event_log'];end=[e for e in log if e['event_type']=='case_completed']
            terminal=[d for d in decisions if d['action'] in ('er_finalize','specialty_finalize','specialist_finalize')]
            assert len(terminal)==len(end)==1;terminal=terminal[0];end=end[0]
            supported=terminal.get('certainty')=='supported';unresolved=episode['result']['diagnosis_unresolved']
            assert supported==not_bool(unresolved)
            # Scheduler strips terminal whitespace when assigning its runtime.
            assert terminal['target'].strip()==rt['assessment_summary']==end['payload']['final_diagnosis']
            assert terminal['target_department']==rt['target_department']==end['payload']['final_department']
            assert rt['is_completed'] and raw['status']==episode['result']['status']=='completed'
            assert end['payload']['diagnosis_status']==('reported' if supported else 'unresolved')
            old=r['result']['prediction'];score=matches(terminal['target'] if supported else None,reference['accepted_diagnoses'])
            dept=terminal['target_department'] in reference['accepted_departments']
            # Code documents this compatibility fingerprint projection. Check
            # raw and public-format Episode separately; do not rewrite either.
            cfg=raw['execution_metadata']['configuration'];assert raw['execution_metadata']['configuration_fingerprint']==_digest(cfg)
            export_cfg=deepcopy(cfg);export_cfg.pop('sequential_source_view',None);export_cfg.pop('development_unfitted_adaptive',None)
            assert episode['configuration_fingerprint']==_digest(export_cfg)
            assert linked['source_policy_linkage']['raw_fingerprint']==digest(raw)
            assert linked['source_policy_linkage']['episode_fingerprint']==digest(episode)
            actions=Counter(d['action'] for d in decisions);assert dict(actions)==episode['result']['action_counts']==r['result']['action_counts']
            assert [e['minute'] for e in log]==sorted(e['minute'] for e in log)
            assert end['minute']<=episode['result']['elapsed_simulation_minutes']<=config['minute_limit']
            audit=rt['llm_audit'];assert all(a['status']=='success' for a in audit)
            if any(a.get('repair_requested') or a.get('repair_of_audit_index') is not None for a in audit):issues.append({'run':name,'episode':n,'code':'RECORDED_REPAIR'})
            cutoffs=raw['telemetry']['retrievals'];assert all(x['cutoff_compliant'] for x in cutoffs)
            before_files=sorted(folder.glob('Request*_before.json'));after_files=sorted(folder.glob('Request*_after.json'))
            assert len(before_files)==len(after_files)==r['request_count']==r['result']['request_count']==len(decisions)
            assert r['result']['identity_check_count']==2*len(before_files)
            observations=[];mem_counts=[];tokens=Counter();sample_payloads=[];sample_responses=[];overrides=0;observed_roles=Counter();evidence_checks=0;ask_checks=0
            leak_keys={'gold','evaluation','task_annotations','source_event_store','reference_diagnosis','accepted_diagnoses','accepted_departments'}
            for i,(bp,ap) in enumerate(zip(before_files,after_files)):
                b=read(bp);a=read(ap);req=b['request'];envelope=a['response_envelope'];payload=json.loads(req['messages'][1]['content']);obs=payload['observation'];observations.append(obs)
                assert b['contract']==plan['fingerprint'] and b['request_fingerprint']==a['request_fingerprint']==digest(req)
                assert a['status']=='VALID' and a['HTTP_status']==200 and a['response_received'] is True
                assert normalize_model_identifier(a['identity_after']['digest'])==normalize_model_identifier(b['identity_before']['digest'])==config['model_identifier']
                assert a['identity_after']['context_length']==32768 and envelope['model']==config['model'] and envelope['done'] is True and envelope.get('done_reason')!='length'
                wire=json.loads(envelope['message']['content']);Draft202012Validator(req['format']).validate(wire)
                response=a['internal_response'];assert obs['elapsed_minutes']==decisions[i]['minute']
                proposal_actions[name+':'+co][response['action']]+=1;selected_actions[name+':'+co][decisions[i]['action']]+=1
                if ACTION.get(response['action'],response['action'])!=decisions[i]['action'] or response['target']!=decisions[i]['target']:overrides+=1
                assert key_hits(payload,leak_keys)==0
                payload_id_hits[name+':'+co]+=sum(v in json.dumps(payload,ensure_ascii=False) for v in (identity['patient_id'],identity['visit_id']))
                observed_roles[payload['role']+':'+str(payload.get('department') or '')]+=1
                records=obs.get('longitudinal_memory',{}).get('records',[]);mem_counts.append(len(records))
                if co=='B0':assert not records
                if co in ('B2','B3','B4'):assert len(records)<=5
                for rec in records:
                    assert rec['patient_id']==identity['patient_id'] and rec['visit_id']!=identity['visit_id']
                    assert timestamp(rec['visit_started_at'])<timestamp(identity['started_at']) and timestamp(rec['available_at'])<=timestamp(identity['started_at'])
                for br in binding['rows']:
                    if br['kind']=='Test':
                        starts=[e['minute'] for e in log if e['event_type']=='test_started' and e['payload'].get('requested_test')==br['target']]
                        releases=[e['minute'] for e in log if e['event_type']=='result_received' and e['payload'].get('requested_test')==br['target']]
                        if releases:assert starts and releases[0]>=max(starts[0]+5,br['available_minute'])
                        if br['target']+': '+br['evidence'] in obs['objective_findings']:
                            assert releases and obs['elapsed_minutes']>=releases[0];evidence_checks+=1
                    else:
                        if any(h.get('question')==br['target'] and h.get('answer')==br['evidence'] and h.get('answer_status')!='unknown' for h in obs['history']):
                            events=[e for e in log if e['event_type']=='question_asked' and e['payload'].get('question')==br['target']]
                            assert events and events[0]['minute']<=obs['elapsed_minutes'] and obs['elapsed_minutes']>=br['available_minute'];ask_checks+=1
                tokens['prompt']+=envelope['prompt_eval_count'];tokens['output']+=envelope['eval_count'];tokens['HTTP_seconds']+=a['wall_seconds']
                sample_payloads.append(b['request_fingerprint']);sample_responses.append(digest(response))
                counters['valid_responses']+=1;counters['identity_snapshots']+=2
            fp_key=(name,co,ordinal)
            trace_fp[fp_key]=digest([{k:d.get(k) for k in ('minute','action','target','certainty','target_department')} for d in decisions])
            payload_fp[fp_key]=sample_payloads;response_fp[fp_key]=sample_responses
            routing=rt['voi_state'].get('primary_routing_history',[]);voi=rt['voi_state'].get('history',[])
            for q in routing:route[name+':'+co][q.get('selected_specialty') or q.get('decision_status','unknown')]+=1
            max_diff=0.0;effective_horizons=Counter();info_steps=0;stops=0
            if co=='B4':
                assert len(plans)==len(voi)==len(observations)
                for i,(t,obs,planner) in enumerate(zip(voi,observations,plans)):
                    canonical={'history':[],'objective_findings':[]}
                    for br in binding['rows']:
                        act=next(x for x in model['actions'] if x['id']==br['action_id']);out=next(x for x in act['outcomes'] if x['id']==br['outcome_id'])
                        if br['kind']=='Test' and br['target']+': '+br['evidence'] in obs['objective_findings']:canonical['objective_findings'].append(out['observation']['text'])
                        if br['kind']=='Ask' and any(h.get('question')==br['target'] and h.get('answer')==br['evidence'] and h.get('answer_status')!='unknown' for h in obs['history']):canonical['history'].append({'question':br['target'],'answer':out['observation']['text'],'answer_status':'known'})
                    belief,matched=engine.observe(canonical);max_diff=max(max_diff,max(abs(belief[d]-t['belief'][d]) for d in belief));assert digest(matched)==digest(t['matched_outcomes'])
                    assert t['selected_action'] in ACTION and ACTION[t['selected_action']]==decisions[i]['action']
                    if t['selected_action']=='diagnose':assert planner['selected_action_id'] is None;stops+=1
                    else:assert planner['selected_action_id']==t['action_id'];info_steps+=1
                    effective_horizons[planner['horizon']]+=1
                assert max_diff<=1e-12
            elapsed=episode['result']['elapsed_simulation_minutes']
            alg=old.get('algorithmic_diagnosis_choice');alg_match=matches(alg,reference['accepted_diagnoses'])['source_exact'] if co=='B4' else None
            tax_match=terminal['target_department']==model['diagnoses'].get(terminal['target']) if supported else None
            result={'plan_id':name,'condition':co,'patient_ordinal':ordinal,'episode_ordinal':n,'status':r['status'],
                'certainty':terminal['certainty'],'source_exact':score['source_exact'],'normalized_source_match':score['normalized_match'],
                'final_department_source_match':dept,'supported_department_source_match':dept if supported else None,
                'catalog_department_consistent':tax_match,'algorithmic_source_match':alg_match,
                'generation':r['request_count'],'identity_checks':r['result']['identity_check_count'],
                'simulation_minutes':elapsed,'terminal_event_minute':end['minute'],
                'export_elapsed_minus_terminal_minute':elapsed-end['minute'],'episode_wall_seconds':r['result']['computation']['wall_seconds'],
                'Ask':actions['ask_history'],'Test':actions['order_test'],'Consult':actions['request_specialty'],
                'Diagnose':actions['er_finalize']+actions['specialty_finalize']+actions['specialist_finalize'],
                'prompt_tokens':tokens['prompt'],'output_tokens':tokens['output'],'HTTP_wall_seconds':tokens['HTTP_seconds'],
                'memory_records_min':min(mem_counts),'memory_records_max':max(mem_counts),'memory_records_mean':mean(mem_counts),
                'retrieval_log_entries':len(cutoffs),'routing_calls':len(routing),'routing_abstentions':sum(q.get('decision_status')=='abstained' for q in routing),
                'planner_calls':len(plans),'planner_states':sum(x['evaluated_states'] for x in plans),
                'VoI_information_actions':info_steps,'VoI_STOP':stops,'posterior_max_difference':max_diff,
                'model_proposal_vs_core_differences':overrides,'visible_test_result_checks':evidence_checks,'visible_Ask_checks':ask_checks,
                'initial_chief_complaint_empty':not bool(str(observations[0].get('chief_complaint','')).strip()),
                'initial_symptoms_empty':not observations[0].get('current_symptoms'),
                'historical_prediction_missing':old.get('certainty')=='not_available' and supported,
                'effective_horizons':dict(effective_horizons)}
            rows.append(result)
            private_rows.append({**result,'actual_core_terminal':terminal,'accepted_reference':reference,
                'historical_prediction':old,'recovered_diagnosis':terminal['target'] if supported else None,
                'recovered_department':terminal['target_department'],'algorithmic_choice':alg,'reference_case_ref':r['case_ref']})
    aggregate={}
    for co in ('B0','B1','B2','B3','B4'):
        rs=[x for x in rows if x['plan_id']=='recovery_llm_all_001' and x['condition']==co]
        supported=[x for x in rs if x['certainty']=='supported'];alg=[x for x in rs if x['algorithmic_source_match'] is not None]
        sums=['generation','identity_checks','Ask','Test','Consult','Diagnose','prompt_tokens','output_tokens','HTTP_wall_seconds','retrieval_log_entries','routing_calls','routing_abstentions','planner_calls','planner_states','VoI_information_actions','VoI_STOP','model_proposal_vs_core_differences','visible_test_result_checks','visible_Ask_checks']
        aggregate[co]={'registered':len(rs),'completed':sum(x['status']=='COMPLETED' for x in rs),'supported':len(supported),'unresolved':len(rs)-len(supported),
            'source_exact_success':sum(x['source_exact'] is True for x in rs),'normalized_source_success':sum(x['normalized_source_match'] is True for x in rs),
            'source_supported_denominator':len(supported),'final_department_source_success':sum(x['final_department_source_match'] for x in rs),
            'supported_department_source_success':sum(x['supported_department_source_match'] is True for x in rs),
            'algorithmic_source_success':sum(x['algorithmic_source_match'] is True for x in alg),'algorithmic_denominator':len(alg),
            'catalog_department_mismatch':sum(x['catalog_department_consistent'] is False for x in rs),
            'mean_simulation_minutes':mean([x['simulation_minutes'] for x in rs]),'min_simulation_minutes':min(x['simulation_minutes'] for x in rs),'max_simulation_minutes':max(x['simulation_minutes'] for x in rs),
            'mean_actions':mean([x['Ask']+x['Test']+x['Consult']+x['Diagnose'] for x in rs]),
            'mean_episode_wall_seconds':mean([x['episode_wall_seconds'] for x in rs]),
            'memory_min':min(x['memory_records_min'] for x in rs),'memory_max':max(x['memory_records_max'] for x in rs),
            'initial_complaint_empty_cases':sum(x['initial_chief_complaint_empty'] for x in rs),
            'initial_symptoms_empty_cases':sum(x['initial_symptoms_empty'] for x in rs),
            'routing_selection_counts':dict(route['recovery_llm_all_001:'+co]),
            'LLM_proposed_actions':dict(proposal_actions['recovery_llm_all_001:'+co]),
            'executed_core_actions':dict(selected_actions['recovery_llm_all_001:'+co]),
            **{k:sum(x[k] for x in rs) for k in sums}}
    comparisons={}
    for a,b in (('B0','B1'),('B1','B2'),('B0','B2'),('B2','B3'),('B3','B4')):
        aa=aggregate[a];bb=aggregate[b]
        pairs=[]
        for i in range(1,11):
            x=next(x for x in rows if x['plan_id']=='recovery_llm_all_001' and x['condition']==a and x['patient_ordinal']==i)
            y=next(x for x in rows if x['plan_id']=='recovery_llm_all_001' and x['condition']==b and x['patient_ordinal']==i)
            pairs.append({'patient_ordinal':i,'jointly_supported':x['certainty']==y['certainty']=='supported',
                'registered_exact_yield_difference':int(y['source_exact'] is True)-int(x['source_exact'] is True),
                'final_department_source_difference':int(y['final_department_source_match'])-int(x['final_department_source_match'])})
        comparisons[b+'-'+a]={'completed_pairs':10,'jointly_supported_pairs':sum(x['jointly_supported'] for x in pairs),
            'registered_exact_yield_difference_pp':10*sum(x['registered_exact_yield_difference'] for x in pairs),
            'final_department_source_difference_pp':10*sum(x['final_department_source_difference'] for x in pairs),
            'generation_difference':bb['generation']-aa['generation'],
            'generation_change_percent':100*(bb['generation']/aa['generation']-1),'rows':pairs}
    repeats={}
    for co,name in (('B3','recovery_llm_b3_001'),('B4','recovery_llm_b4_001')):
        repeats[co]={'normalized_action_trace_equal':trace_fp[(name,co,1)]==trace_fp[('recovery_llm_all_001',co,1)],
            'request_payload_fingerprints_equal':payload_fp[(name,co,1)]==payload_fp[('recovery_llm_all_001',co,1)],
            'decoded_response_fingerprints_equal':response_fp[(name,co,1)]==response_fp[('recovery_llm_all_001',co,1)],
            'scope':'saved first-patient comparison only; no new regeneration or general reproducibility claim'}
    public_files=[p for name in NAMES for p in (ROOT/'outputs/research_evaluation/B34_CONNECTION_v1'/name).glob('*.json')]
    public_id_leaks=[]
    for p in public_files:
        text=p.read_text(encoding='utf-8-sig')
        if any(x in text for x in ids):public_id_leaks.append(str(p.relative_to(ROOT)))
        assert key_hits(read(p),{'gold','evaluation','task_annotations','source_event_store'})==0
    assert not public_id_leaks
    unchanged=all(sha(p)==h for p,h in before.items());assert unchanged
    result={'analysis_version':NS,'runs':runs,'conditions':aggregate,'paired':comparisons,'minimum_vs_batch_first_case':repeats,
        'recorded_predictions_missing_despite_supported_core_terminal':sum(x['historical_prediction_missing'] for x in rows),
        'all_saved_request_responses_validated':counters['valid_responses'],'identity_snapshots_checked':counters['identity_snapshots'],
        'public_original_id_leaks':public_id_leaks,'request_payload_original_id_occurrences':dict(payload_id_hits),
        'current_input_hash_mismatches':issues,'input_files_hashed':len(before),'input_files_unchanged':unchanged,
        'actual_new_LLM_generation':0,'new_fitting_calls':0,'original_results_rewritten':False,
        'source_label_proxy':True,'independent_clinical_review':False,'clinical_accuracy':'NOT_EVALUATED','research_evaluation_ready':False}
    write(PUB/'analysis.json',result);write(PRIVATE/'recovered_predictions_and_references.json',private_rows)
    write(PRIVATE/'input_hashes_after.json',{p:sha(p) for p in before})
    flat=[{k:v for k,v in x.items() if k!='effective_horizons'} for x in rows]
    write_csv(DOC/'EPISODE_RESULTS.csv',flat)
    table=[{'condition':co,**{k:a[k] for k in ('registered','completed','supported','unresolved','source_exact_success','normalized_source_success','final_department_source_success','algorithmic_source_success','generation','prompt_tokens','output_tokens','mean_actions','mean_simulation_minutes','routing_calls','routing_abstentions','planner_calls','planner_states')}} for co,a in aggregate.items()]
    write_csv(DOC/'CONDITION_RESULTS.csv',table);write(DOC/'RESULTS.json',result)
    print(json.dumps({'namespace':NS,'rows_analyzed':len(rows),'whole_run_generation':runs['recovery_llm_all_001']['generation'],
        'historical_supported_predictions_missing':result['recorded_predictions_missing_despite_supported_core_terminal'],
        'conditions':table,'preserved_files':len(before),'new_LLM_generation':0},ensure_ascii=True))


def not_bool(value):return value is False


if __name__=='__main__':main()
