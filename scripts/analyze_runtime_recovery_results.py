"""Analyze saved actual runtime_recovery_all_001 only. No inference or fitting."""
from collections import Counter,defaultdict
from copy import deepcopy
import csv,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.run_simulation import _digest
from app.locdi.common_prediction_contract import extract_core_prediction
from app.locdi.synthea_diagnosis_audit import matches
from app.core.longitudinal_memory import timestamp
from jsonschema import Draft202012Validator
NS='B34_RUNTIME_RESULTS_ANALYSIS_20261121_v1'
PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS;DOC=ROOT/'docs'/NS
BASE=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001'
RUN=BASE/'run_runtime_recovery_all_001';OLD=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/recovery_llm_all_001/run_recovery_llm_all_001'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=True,indent=2,allow_nan=False)
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(x):return statistics.mean(x) if x else None
def main():
    for p in (PUB,PR,DOC):p.mkdir(parents=True,exist_ok=False)
    srcpub=ROOT/'outputs/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001'
    inputs=set(p for folder in (BASE,srcpub,OLD) for p in folder.rglob('*') if p.is_file())
    pmodel=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json';inputs.add(pmodel)
    hashes={str(p.resolve()):sha(p) for p in inputs};write(PR/'input_hashes_before.json',hashes)
    plan=read(srcpub/'plan.json');config=read(BASE/'llm_config.json');panel=read(BASE/'panel.json');bind=read(BASE/'bindings.json')
    c=read(RUN/'collection.json');saved=read(srcpub/'saved_results.json');oldc=read(OLD/'collection.json')
    assert c['plan_fingerprint']==plan['fingerprint']==digest({k:v for k,v in plan.items() if k!='fingerprint'})
    assert c['rows']==[json.loads(s) for s in (RUN/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if s]
    assert c['generation']==sum(r['request_count'] for r in c['rows'])==saved['actual_generation']==295
    pin_drift=[str(p) for p,h in {**plan['code_hashes'],**plan['input_hashes']}.items() if not Path(p if Path(p).is_absolute() else ROOT/p).is_file() or sha(p if Path(p).is_absolute() else ROOT/p)!=h]
    rows=[];private=[];failures=[];trace=[];counts=Counter();routes=defaultdict(Counter);belief_max_diff=0.;old_comparison=[]
    savedrows={x['episode_ordinal']:x for x in saved['rows']}
    for r in c['rows']:
        n=r['episode_ordinal'];co=r['condition'];i=r['patient_ordinal'];f=RUN/'requests'/('Episode%04d'%n)
        out={'episode':n,'patient_ordinal':i,'condition':co,'status':r['status'],'generation':r['request_count'],
            'certainty':'not_available','source_exact':None,'normalized':None,'department_match':None,'algorithmic_source':None,
            'Ask':0,'Test':0,'Finalize':0,'simulation_minutes':None,'terminal_minute':None,
            'prompt_tokens':0,'output_tokens':0,'HTTP_wall_seconds':0.,'valid_responses':0,'failed_responses':0,
            'routing_calls':0,'routing_abstentions':0,'planner_calls':0,'planner_states':0,'VoI_information':0,'VoI_STOP':0,
            'LLM_terminal_revalidated':False,'M5_clinical_support':None,'memory_min':None,'memory_max':None}
        if r['status']=='NOT_ATTEMPTED':
            assert not f.exists() and not r['request_count'];rows.append(out);continue
        raw=read(f/'raw_runner_result.json');dec=read(f/'actual_core_decisions.json');rt=read(f/'runtime_snapshot_private.json');linked=read(f/'episode_and_policy.json');ep=linked['episode'];plans=read(f/'planner_trace.json')
        assert raw['status']==ep['result']['status']==r['status'].lower()
        assert linked['source_policy_linkage']['raw_fingerprint']==digest(raw) and linked['source_policy_linkage']['episode_fingerprint']==digest(ep)
        cfg=raw['execution_metadata']['configuration'];assert raw['execution_metadata']['configuration_fingerprint']==_digest(cfg)
        projected=deepcopy(cfg);projected.pop('sequential_source_view',None);projected.pop('development_unfitted_adaptive',None)
        assert ep['configuration_fingerprint']==_digest(projected)
        assert linked['source_policy_linkage']['runtime_recovery_contract']=='released_measurements_llm_at_voi_stop.v1'
        prediction=extract_core_prediction(dec,raw['status']);expected=deepcopy(r['result']['prediction']);expected['algorithmic_diagnosis_choice']=None
        assert prediction==expected
        ref=panel[i-1]['reference'];match=matches(prediction['diagnosis'],ref['accepted_diagnoses']);department=prediction['department'] in ref['accepted_departments'] if prediction['department'] is not None else None
        out.update(certainty=prediction['certainty'],source_exact=match['source_exact'],normalized=match['normalized_match'],department_match=department)
        alg=r['result']['prediction'].get('algorithmic_diagnosis_choice')
        if co=='B4':out['algorithmic_source']=matches(alg,ref['accepted_diagnoses'])['source_exact']
        sr=savedrows[n];assert sr['source_exact']==out['source_exact'] and sr['source_department']==department
        actions=Counter(d['action'] for d in dec);assert dict(actions)==ep['result']['action_counts']==r['result']['action_counts']
        out.update(Ask=actions['ask_history'],Test=actions['order_test'],Finalize=actions['er_finalize'],simulation_minutes=ep['result']['elapsed_simulation_minutes'])
        log=raw['execution']['visits'][0]['final_state']['event_log'];assert [x['minute'] for x in log]==sorted(x['minute'] for x in log)
        if r['status']=='COMPLETED':
            end=next(x for x in log if x['event_type']=='case_completed');out['terminal_minute']=end['minute']
            assert dec[-1]['target'].strip()==rt['assessment_summary']==end['payload']['final_diagnosis']
            assert prediction['department']==rt['target_department']==end['payload']['final_department']
            assert ep['result']['diagnosis_unresolved']==(prediction['certainty']=='unresolved')
        identity=raw['execution']['visits'][0]['identity'];requests=sorted(f.glob('Request*_before.json'));afters=sorted(f.glob('Request*_after.json'))
        assert len(requests)==len(afters)==r['request_count'];assert r['result']['identity_check_count']==2*len(requests)
        memory=[]
        for bp,ap in zip(requests,afters):
            b=read(bp);a=read(ap);req=b['request'];payload=json.loads(req['messages'][1]['content']);obs=payload['observation'];envelope=a['response_envelope']
            assert b['request_fingerprint']==a['request_fingerprint']==digest(req) and a['HTTP_status']==200 and a['response_received']
            assert normalize_model_identifier(b['identity_before']['digest'])==normalize_model_identifier(a['identity_after']['digest'])==config['model_identifier']
            assert a['identity_after']['context_length']==32768 and envelope['done'] and envelope['done_reason']!='length'
            value=json.loads(envelope['message']['content']);errors=list(Draft202012Validator(req['format']).iter_errors(value))
            if a['status']=='VALID':assert not errors;out['valid_responses']+=1
            else:
                assert errors;out['failed_responses']+=1
                actual_branch=next(x for x in req['format'].get('anyOf',[req['format']]) if x['properties']['action']['enum']==[value['action']])
                relevant=[{'validator':e.validator,'path':list(e.path),'message':e.message} for e in Draft202012Validator(actual_branch).iter_errors(value)]
                failures.append({'episode':n,'patient_ordinal':i,'condition':co,'request':a['request_number'],'HTTP':a['HTTP_status'],
                    'JSON_valid':True,'Schema_valid':False,'error_type':a['error_type'],'action':value['action'],
                    'evidence_indices':value.get('evidence_indices'),'errors':relevant,'last_request_wall_seconds':a['wall_seconds']})
            assert not ({'gold','evaluation','source_event_store','reference_diagnosis'}&set(payload))
            for x in payload['released_measurements']:
                j=x['evidence_index'];text=payload['evidence'][j];assert str(x['value']) in text
                br=next(br for br in bind[r['case_ref']]['rows'] if br['code']==x['code'] and br['description']==x['description'])
                assert json.loads(br['evidence'])['value']==x['value'] and br['unit']==x['unit'] and obs['elapsed_minutes']>=br['available_minute']
                if br['kind']=='Test':
                    assert br['target']+': '+br['evidence'] in obs['objective_findings']
                    received=[e['minute'] for e in log if e['event_type']=='result_received' and e['payload'].get('requested_test')==br['target']]
                    starts=[e['minute'] for e in log if e['event_type']=='test_started' and e['payload'].get('requested_test')==br['target']]
                    assert starts and received and received[0]>=max(starts[0]+5,br['available_minute']) and obs['elapsed_minutes']>=received[0]
                else:assert any(h.get('question')==br['target'] and h.get('answer')==br['evidence'] and h.get('answer_status')!='unknown' for h in obs['history'])
                counts['released_measurements_checked']+=1
            rec=obs.get('longitudinal_memory',{}).get('records',[]);memory.append(len(rec))
            for m in rec:assert m['patient_id']==identity['patient_id'] and m['visit_id']!=identity['visit_id'] and timestamp(m['available_at'])<=timestamp(identity['started_at']) and timestamp(m['visit_started_at'])<timestamp(identity['started_at'])
            out['prompt_tokens']+=envelope['prompt_eval_count'];out['output_tokens']+=envelope['eval_count'];out['HTTP_wall_seconds']+=a['wall_seconds'];counts['requests']+=1
        out['memory_min']=min(memory);out['memory_max']=max(memory)
        route=rt['voi_state'].get('primary_routing_history',[]);out['routing_calls']=len(route);out['routing_abstentions']=sum(x['decision_status']=='abstained' for x in route)
        for t in route:routes[co][t.get('selected_specialty') or t['decision_status']]+=1
        hist=rt['voi_state'].get('history',[]);out.update(planner_calls=len(plans),planner_states=sum(p['evaluated_states'] for p in plans),VoI_information=sum(t['selected_action']!='diagnose' for t in hist),VoI_STOP=sum(t['selected_action']=='diagnose' for t in hist))
        if co=='B4':
            t=hist[-1];out['LLM_terminal_revalidated']=t.get('terminal_origin')=='CURRENT_LLM_DIAGNOSE_CORE_REVALIDATED';out['M5_clinical_support']=t['m5_clinical_support']
            oldrt=read(OLD/'requests'/('Episode%04d'%n)/'runtime_snapshot_private.json');oldhist=oldrt['voi_state']['history'];assert len(hist)==len(oldhist)==len(plans)
            for z,oldz,p in zip(hist,oldhist,plans):
                assert z['selected_action']==oldz['selected_action'] and z.get('action_id')==oldz.get('action_id')
                belief_max_diff=max(belief_max_diff,max(abs(z['belief'][k]-oldz['belief'][k]) for k in z['belief']))
                trace.append({'patient_ordinal':i,'minute':z['minute'],'action':z['selected_action'],'action_id':z.get('action_id'),'T':p['baseline_terminal_utility'],'V':p['optimal_value'],'STOP':z['selected_action']=='diagnose','remaining_information_candidates':len(p['action_values']),'terminal_origin':z.get('terminal_origin')})
            assert belief_max_diff<=1e-12
        private.append({'episode':n,'patient_ordinal':i,'condition':co,'prediction':r['result']['prediction'],'reference':ref,'status':r['status']})
        rows.append(out)
    aggregate={}
    for co in ('B0','B1','B2','B3','B4'):
        rs=[x for x in rows if x['condition']==co];done=[x for x in rs if x['status']=='COMPLETED'];support=[x for x in done if x['certainty']=='supported']
        sums=('generation','Ask','Test','Finalize','prompt_tokens','output_tokens','HTTP_wall_seconds','valid_responses','failed_responses','routing_calls','routing_abstentions','planner_calls','planner_states','VoI_information','VoI_STOP')
        aggregate[co]={'registered':len(rs),'states':dict(Counter(x['status'] for x in rs)),'completed':len(done),'supported':len(support),'unresolved':sum(x['certainty']=='unresolved' for x in done),
            'Source_exact_success':sum(x['source_exact'] is True for x in rs),'normalized_success':sum(x['normalized'] is True for x in rs),'department_success':sum(x['department_match'] is True for x in rs),'algorithmic_success':sum(x['algorithmic_source'] is True for x in rs),
            'mean_completed_actions':mean([x['Ask']+x['Test']+x['Finalize'] for x in done]),'mean_completed_simulation_minutes':mean([x['simulation_minutes'] for x in done]),
            'completed_prompt_tokens':sum(x['prompt_tokens'] for x in done),'memory_min':min(x['memory_min'] for x in rs if x['memory_min'] is not None),'memory_max':max(x['memory_max'] for x in rs if x['memory_max'] is not None),
            'routing_selection':dict(routes[co]),'LLM_terminal_revalidated':sum(x['LLM_terminal_revalidated'] for x in rs),**{k:sum(x[k] for x in rs) for k in sums}}
        assert aggregate[co]['Source_exact_success']==saved['conditions'][co]['Source_exact_success'] and aggregate[co]['generation']==saved['conditions'][co]['generation']
    previous=read(ROOT/'outputs/private/research_evaluation/B34_COMPLETED_RUN_ANALYSIS_20261121_v3/recovered_predictions_and_references.json')
    comparisons={}
    for co in aggregate:
        new=[x for x in rows if x['condition']==co and x['patient_ordinal']<=9 and x['status']=='COMPLETED'];old=[x for x in previous if x['plan_id']=='recovery_llm_all_001' and x['condition']==co and x['patient_ordinal']<=9]
        comparisons[co]={'paired_completed_patients':9,'old_supported':sum(x['certainty']=='supported' for x in old),'new_supported':sum(x['certainty']=='supported' for x in new),
            'old_source_success':sum(x['source_exact'] is True for x in old),'new_source_success':sum(x['source_exact'] is True for x in new),
            'old_department_success':sum(x['final_department_source_match'] is True for x in old),'new_department_success':sum(x['department_match'] is True for x in new),
            'old_generation':sum(x['generation'] for x in old),'new_generation':sum(x['generation'] for x in new),
            'newly_matching_ordinals':[x['patient_ordinal'] for x in new if x['source_exact'] is True and not next(o for o in old if o['patient_ordinal']==x['patient_ordinal'])['source_exact']],
            'lost_matching_ordinals':[o['patient_ordinal'] for o in old if o['source_exact'] is True and not next(x for x in new if x['patient_ordinal']==o['patient_ordinal'])['source_exact']]}
    assert all(sha(p)==h for p,h in hashes.items())
    report={'namespace':NS,'run_id':'run_runtime_recovery_all_001','status':'PARTIAL_FAILURE_WITH_SAVED_EXPORT','registered':50,'states':dict(Counter(r['status'] for r in rows)),
        'generation':c['generation'],'wall_seconds':c['wall_seconds'],'stop_reason':c['stop_reason'],'conditions':aggregate,'failure':failures,
        'matched_previous_first9':comparisons,'B4_saved_belief_vs_previous_max_abs_diff':belief_max_diff,
        'counts':dict(counts),'current_pin_mismatches':pin_drift,'preserved_input_files':len(hashes),'preserved_input_changes':0,
        'new_LLM_generation':0,'new_fit_calls':0,'independent_clinical_review':False,'clinical_accuracy':'NOT_EVALUATED','research_evaluation_ready':False}
    write(PUB/'analysis.json',report);write(PR/'prediction_reference_details.json',private);write(DOC/'ANALYSIS.json',report)
    csvwrite(DOC/'EPISODE_RESULTS.csv',rows);csvwrite(DOC/'VOI_TRACE.csv',trace)
    flat=[{'condition':k,**{a:v for a,v in value.items() if not isinstance(v,dict)}} for k,value in aggregate.items()];csvwrite(DOC/'CONDITION_RESULTS.csv',flat)
    print(json.dumps(report,ensure_ascii=True))
if __name__=='__main__':main()
