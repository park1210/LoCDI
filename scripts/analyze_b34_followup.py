"""Read-only saved-trace/Source audit; never constructs an inference client."""
from collections import Counter,defaultdict
from copy import deepcopy
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import RoutingUtility,observed_tokens,jaccard_similarity
from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case
from app.locdi.llm_setup import digest
from app.locdi.common_prediction_contract import shared_selection
from app.locdi.current_ablation_llm_backend import input_guard,InputContextBudgetError
from app.locdi.routing_collector import load_preparation
from app.locdi.source_paths import resolve_source_file
from app.locdi.routing_data_reuse import BANK,read as read_bank

NS='B34_FOLLOWUP_ANALYSIS_20261121_v1'
DOC=ROOT/'docs'/NS;PUB=ROOT/'outputs/research_evaluation'/NS;PRIVATE=ROOT/'outputs/private/research_evaluation'/NS
BASE=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/recovery_llm_all_001'
MODEL=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=True,indent=2,allow_nan=False)
def table(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(v):return sum(v)/len(v) if v else None

def main():
    for p in (DOC,PUB,PRIVATE):p.mkdir(parents=True,exist_ok=False)
    model=read(MODEL);panel=read(BASE/'panel.json');bindings=read(BASE/'bindings.json');index=read(BASE/'case_index.json')
    cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl');collection=read(BASE/'run_recovery_llm_all_001/collection.json')
    evidence=read(ROOT/'outputs/private/research_evaluation/B34_LLM_TEST_READY_v1/development_proxy_fit_001/development_routing_evidence.json')
    artifact=evidence['routing_artifact'];registry=AgentRegistry(artifact['registry']);utility=RoutingUtility(artifact,registry)
    groups=list(dict.fromkeys(x['decision_id'] for x in artifact['samples']));trainsets=[next(x['tokens'] for x in artifact['samples'] if x['decision_id']==g) for g in groups]
    prior=read(ROOT/'outputs/private/research_evaluation/B34_COMPLETED_RUN_ANALYSIS_20261121_v3/input_hashes_before.json')
    assert all(sha(p)==h for p,h in prior.items());source_pins={str(MODEL):sha(MODEL)}
    routing_rows=[];voi_rows=[];stops=[];private_steps=[];guards=[];source_rows=[]
    for row in collection['rows']:
        co=row['condition'];ordinal=row['patient_ordinal'];f=BASE/'run_recovery_llm_all_001/requests'/('Episode%04d'%row['episode_ordinal'])
        rt=read(f/'runtime_snapshot_private.json');routing=rt['voi_state'].get('primary_routing_history',[])
        hist=rt['voi_state'].get('history',[]);plans=read(f/'planner_trace.json');dec=read(f/'actual_core_decisions.json')
        before=sorted(f.glob('Request*_before.json'));after=sorted(f.glob('Request*_after.json'));binding=bindings[row['case_ref']]
        previous=None
        for i,(bp,ap) in enumerate(zip(before,after)):
            b=read(bp);a=read(ap);req=b['request'];payload=json.loads(req['messages'][1]['content']);obs=payload['observation']
            q,w=shared_selection(payload,req['format'],model['diagnoses'])
            try:g=input_guard(req['messages'][0]['content'],json.dumps(q,ensure_ascii=False,sort_keys=True),w);bound=g['conservative_input_token_bound'];status='PASS'
            except InputContextBudgetError:bound=None;status='BLOCKED'
            guards.append({'condition':co,'patient_ordinal':ordinal,'request':i+1,'status':status,'conservative_bound':bound})
            if co in ('B3','B4'):
                scores={};similarities=[jaccard_similarity(observed_tokens(obs),t) for t in trainsets]
                for agent in artifact['registry']['agents']:
                    if agent['role']!='specialty':continue
                    estimate=utility.score(obs,agent);scores[agent['department']]=estimate['score'] if estimate else None
                    assert estimate==routing[i]['utility_estimates'][agent['department']]
                pieces={
                    'symptom_only':{'chief_complaint':obs['chief_complaint'],'current_symptoms':obs['current_symptoms']},
                    'objective_only':{'objective_findings':obs['objective_findings']},
                    'memory_only':{'longitudinal_memory':obs.get('longitudinal_memory',{})},
                    'without_memory':{k:v for k,v in obs.items() if k!='longitudinal_memory'}}
                rs={'condition':co,'patient_ordinal':ordinal,'decision':i+1,'minute':obs['elapsed_minutes'],
                    'source_group':panel[ordinal-1]['reference']['accepted_departments'][0],
                    'jaccard_train1':similarities[0],'jaccard_train2':similarities[1],
                    **{'utility_'+k:v for k,v in scores.items()},
                    'selected':routing[i]['selected_specialty'],'observed_tokens':len(observed_tokens(obs)),
                    **{k+'_tokens':len(observed_tokens(v)) for k,v in pieces.items()},
                    'memory_new_tokens':len(set(observed_tokens(obs))-set(observed_tokens(pieces['without_memory']))),
                    'objective_count':len(obs['objective_findings'])}
                routing_rows.append(rs)
            if co=='B4':
                t=hist[i];p=plans[i];belief=t['belief'];choice=max(belief,key=belief.get)
                delta=sum(abs(belief[k]-previous[k]) for k in belief) if previous is not None else 0.
                proposal=a['internal_response'];actual=dec[i]
                vr={'patient_ordinal':ordinal,'decision':i+1,'minute':t['minute'],'selected_action':t['selected_action'],
                    'action_id':t.get('action_id'),'matched_outcomes':len(t['matched_outcomes']),
                    'posterior_L1_change':delta,'posterior_max':max(belief.values()),'entropy_nats':t['entropy_nats'],
                    'terminal_utility':p['baseline_terminal_utility'],'optimal_value':p['optimal_value'],
                    'information_candidates':len(p['action_values']),'max_candidate_voi':max((a['voi'] for a in p['action_values']),default=None),
                    'selected_voi':t.get('voi'),'STOP':t['selected_action']=='diagnose','clinical_support':t.get('clinical_support'),
                    'supports_entries':sum(len(m['supports_diagnoses']) for m in t['matched_outcomes']),
                    'LLM_proposed_action':proposal['action'],'LLM_certainty':proposal['certainty'],
                    'actual_action':actual['action'],'received_tests':len(obs['objective_findings'])}
                voi_rows.append(vr);previous=belief
                private_steps.append({**vr,'belief':belief,'model_choice':choice,'proposal':proposal,'actual':actual})
                if vr['STOP']:stops.append(vr)
    # Source-to-adapter-to-payload audit, without exposing any new information.
    descriptors=[json.loads(s) for s in (ROOT/'outputs/private/research_evaluation/c4_05_v1/case_source_index.jsonl').read_text(encoding='utf-8').splitlines()]
    for ordinal,p in enumerate(panel,1):
        case=cases[index[p['case_ref']]-1];pid=case['patient_id'];visit=case['agent_visible']['index_encounter_id'];d=next(d for d in descriptors if d['patient_id']==pid and d['visit_id']==visit)
        path=Path(d['raw_path']);document=read(path);source_pins[str(path)]=sha(path)
        assert source_pins[str(path)]==d['declared_raw_sha256']
        enc=next(e for e in document['record']['encounters'] if e['uuid']==visit)
        runtime=adapt_case(case).runtime_case;initial=case['agent_visible']['current_initial_state']
        sources=case['provenance']['source_case_audit']['symptom_sources']
        symptom_values=[x.get('visible_value',{}) for x in sources]
        case_events=case['environment']['source_event_store'];boundids={r['source_event']['event_id'] for r in bindings[p['case_ref']]['rows']}
        sr={'patient_ordinal':ordinal,
            'raw_chief_complaint_present':bool(enc.get('chief_complaint')),
            'raw_narrative_present':bool(enc.get('patient_statement') or enc.get('symptom_narrative')),
            'raw_reason_present':bool(enc.get('reason') or enc.get('reasons')),
            'raw_document_encoded_symptom_types':len(document.get('symptoms',{})),
            'initial_doctor_complaint_present':bool(initial['doctor_visible_info'].get('chief_complaint')),
            'initial_patient_complaint_present':bool(initial['patient_visible_info'].get('chief_complaint')),
            'adapted_doctor_complaint_present':bool(runtime['doctor_visible_info'].get('chief_complaint')),
            'adapter_doctor_fields_equal':runtime['doctor_visible_info']==initial['doctor_visible_info'],
            'adapter_patient_fields_equal':runtime['patient_visible_info']==initial['patient_visible_info'],
            'linked_symptom_sources':len(symptom_values),'linked_symptom_value_present':sum(v.get('value') is not None for v in symptom_values),
            'source_observation_events':sum(x.get('event_type')=='observations' for x in case_events),
            'common_menu_rows':len(bindings[p['case_ref']]['rows']),
            'other_source_observation_events':sum(x.get('event_type')=='observations' and x['event_id'] not in boundids for x in case_events)}
        assert sr['adapter_doctor_fields_equal'] and sr['adapter_patient_fields_equal'];source_rows.append(sr)
    # Candidate availability, preserving existing train/validation assignment.
    preparation=load_preparation('phase24b_candidates_v2');all50={c['patient_id'] for c in cases}
    used=set()
    for path in [ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/source_ready_002/routing_training/plan.json',Path(evidence['collection_plan'])]:
        old=read(path);used.update(c['decision']['patient_id'] for c in old['candidates'])
    candidate_rows=[]
    refs=read(ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/reference_candidates.json')
    byref={x['case_ref']:x for x in refs}
    for c in preparation['candidates']:
        d=c['decision'];r=byref[c['case_ref']];raws=[x['source_record'] for x in r.get('source_condition_rows',[])]
        labels={x.get('DESCRIPTION') for x in raws}
        simple=bool(raws) and len(labels)==1 and None not in labels and next(iter(labels)) in model['diagnoses']
        reason='READY_FOR_NEW_TRAIN_COLLECTION_REVIEW' if d['split']=='train' and d['patient_id'] not in all50|used else 'EXISTING_SPLIT_OR_EXPOSURE_EXCLUDED'
        candidate_rows.append({'case_ref':c['case_ref'],'split':d['split'],'patient_id':d['patient_id'],
            'previous_collection_exposure':d['patient_id'] in used,'original50_overlap':d['patient_id'] in all50,
            'single_catalog_source_label':simple,'source_department':model['diagnoses'].get(next(iter(labels))) if simple else None,
            'readiness':reason,'candidate':c,'reference_candidates':r})
    fresh=[x for x in candidate_rows if x['readiness']=='READY_FOR_NEW_TRAIN_COLLECTION_REVIEW']
    write(PRIVATE/'routing_candidates.json',candidate_rows);write(PRIVATE/'voi_step_details.json',private_steps)
    write(PRIVATE/'routing_score_details.json',routing_rows)
    table(DOC/'ROUTING_SCORES.csv',[{k:v for k,v in r.items() if k!='source_group'} for r in routing_rows])
    table(DOC/'VOI_TRACE.csv',voi_rows);table(DOC/'SOURCE_CHAIN.csv',source_rows)
    group_scores={}
    for dept in ('endocrinology','neurology'):
        rr=[r for r in routing_rows if r['condition']=='B3' and r['source_group']==dept and r['decision']==1]
        group_scores[dept]={'cases':len(rr),'mean_jaccard1':mean([x['jaccard_train1'] for x in rr]),'mean_jaccard2':mean([x['jaccard_train2'] for x in rr]),
            'mean_neurology':mean([x['utility_neurology'] for x in rr]),'mean_other':mean([x['utility_cardiology'] for x in rr])}
    result={'namespace':NS,'routing_recomputed_decisions':len(routing_rows),'routing_exact_saved_estimates_match':True,
        'B3_initial_source_group_scores':group_scores,'B3_constant_neurology_reference_agreement':5,'denominator':10,
        'STOP_no_candidates':sum(x['information_candidates']==0 for x in stops),'STOP_with_candidates':sum(x['information_candidates']>0 for x in stops),
        'STOP_rows':stops,'posterior_updates_nonzero':sum(x['posterior_L1_change']>1e-12 for x in voi_rows),
        'all_M5_supports_empty':all(not o.get('supports_diagnoses') for a in model['actions'] for o in a['outcomes']),
        'source_audit_cases':len(source_rows),'raw_complaints_present':sum(x['raw_chief_complaint_present'] for x in source_rows),
        'raw_narratives_present':sum(x['raw_narrative_present'] for x in source_rows),'raw_reasons_present':sum(x['raw_reason_present'] for x in source_rows),
        'fresh_existing_train_candidates':len(fresh),'fresh_single_catalog_source_candidates':sum(x['single_catalog_source_label'] for x in fresh),
        'fresh_source_department_counts':dict(Counter(x['source_department'] for x in fresh)),
        'new_selection_saved_payload_guard':dict(Counter(x['status'] for x in guards)),
        'new_selection_max_saved_input_bound':max((x['conservative_bound'] or 0 for x in guards)),
        'LLM_generation':0,'fitting_calls':0,'source_label_proxy':True,'independent_clinical_review':False,'clinical_accuracy':'NOT_EVALUATED','research_evaluation_ready':False}
    table(DOC/'SELECTION_INPUT_GUARD.csv',guards);write(PUB/'analysis.json',result);write(DOC/'ANALYSIS.json',result)
    write(PRIVATE/'source_hashes.json',source_pins)
    assert all(sha(p)==h for p,h in prior.items()) and all(sha(p)==h for p,h in source_pins.items())
    write(PUB/'preservation.json',{'previous_files_checked':len(prior),'new_source_files_checked':len(source_pins),'mismatches':0,'new_generation':0})
    print(json.dumps({k:v for k,v in result.items() if k!='STOP_rows'},ensure_ascii=True))
if __name__=='__main__':main()
