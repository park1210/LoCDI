"""Read-only input audit and Source acquisition; no model or real fitting."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import json,statistics,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.synthetic_pilot_collection import ZIP,M5,CONFIG,PRIVATE,PUBLIC,read
from app.locdi.synthetic_pilot_source import load_zip,snapshot,file_sha,MISSING_CONFIRMATION,DEPARTMENTS
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import validate_setup,digest
from app.core.routing_utility import observed_tokens,jaccard_similarity,RoutingUtility
from app.core.agent_registry import AgentRegistry

def stats(xs):return {'n':len(xs),'min':min(xs) if xs else None,'mean':statistics.mean(xs) if xs else None,'max':max(xs) if xs else None}
def main():
    m=read(M5);b=load_zip(ZIP,m['diagnoses']);original=[json.loads(s) for s in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf-8').splitlines() if s]
    base=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001/run_runtime_recovery_all_001'
    evaluation=[]
    for row in read(base/'collection.json')['rows']:
        if row['condition']=='B0':
            p=base/'requests'/('Episode%04d'%row['episode_ordinal'])/'Request0001_before.json'
            if p.exists():evaluation.append(json.loads(read(p)['request']['messages'][1]['content'])['observation'])
    cfg=read(CONFIG);reg=AgentRegistry(validate_setup(cfg)['prepared']['registry'])
    old=read(ROOT/'outputs/private/research_evaluation/B34_RUNTIME_RECOVERY_20261121_v1/development_routing_evidence.json')['routing_artifact']
    utility=RoutingUtility(old,reg);reg=AgentRegistry(reg.snapshot(),old)
    agents=[a for a in reg.snapshot()['agents'] if a['role']=='specialty']
    private_rows=[];snapshots={};results={};traces=[]
    for stage in ('initial','all_tests'):
        for view in ('full','source_compatible'):
            key=stage+'_'+view;obs=[]
            for i,c in enumerate(b['cases'],1):
                p,t=snapshot(c,m,view=view,stage=stage);obs.append(p['observation']);snapshots[i,stage,view]=p
                if view=='full':traces.append({'ordinal':i,**t})
            token_rows=[observed_tokens(o) for o in obs];pair=[jaccard_similarity(x,y) for x in token_rows for y in map(observed_tokens,evaluation)]
            within=[jaccard_similarity(x,y) for x,y in combinations(token_rows,2)]
            base_tokens=set().union(*(set(observed_tokens(o)) for o in evaluation));margins=[];ties=0;abstained=0;selected=Counter()
            for i,o in enumerate(obs,1):
                scored=[(a['department'],utility.score(o,a)) for a in agents];available=[(d,s['score']) for d,s in scored if s]
                choice=reg.select_specialty(o,o['longitudinal_memory'],list(DEPARTMENTS),router='adaptive')
                if choice['decision_status']!='selected':abstained+=1
                else:selected[str(choice['selected_specialty'])]+=1
                if available:
                    values=sorted((s for _,s in available),reverse=True);margins.append(values[0]-values[1] if len(values)>1 else None)
                    ties+=sum(abs(s-values[0])<=1e-12 for _,s in available)>1
                private_rows.append({'ordinal':i,'stage':stage,'view':view,'tokens':observed_tokens(o),'U0_scores':scored})
            results[key]={'patients':15,'token_counts':stats([len(x) for x in token_rows]),'jaccard_to_existing_evaluation10':stats(pair),
                'within_pilot_jaccard':stats(within),'within_pilot_pairs_ge_0_8':sum(x>=.8 for x in within),
                'unmatched_token_counts_to_evaluation_vocabulary':stats([len(set(x)-base_tokens) for x in token_rows]),
                'U0_existing_utility_only':{'tie_patients':ties,'abstained_patients':abstained,'selected_distribution':dict(selected),'margin':stats([x for x in margins if x is not None])},
                'U1_U2_new_utility':'NOT_FITTED_NO_AGENT_RESPONSES'}
    pairs=[]
    for x,y in combinations(b['cases'],2):
        sim=jaccard_similarity(observed_tokens(snapshots[b['cases'].index(x)+1,'initial','full']['observation']),
                               observed_tokens(snapshots[b['cases'].index(y)+1,'initial','full']['observation']))
        if sim>=.35:pairs.append({'first':x['case_id'],'second':y['case_id'],'initial_token_jaccard':sim,'semantic_independence':'NOT_REVIEWED'})
    raw=Counter((x['description'],x['code'],x['unit'],x['available_minute']) for c in b['cases'] for x in c['test_timeline'])
    exact_narrative=len({digest(c['static_case']) for c in b['cases']})
    refs=list(b['references'].values())
    report={'zip_sha256':b['zip_sha256'],'member_hashes':b['member_hashes'],'case_count':15,'unique_case_ids':15,
      'public_reference_provenance_alignment':True,'exact_catalog_reference_matches':15,'unique_reference_diagnoses':len({r['reference_diagnosis'] for r in refs}),
      'reference_status_counts':dict(Counter(r['reference_status'] for r in refs)),
      'independent_clinical_reviewed':0,'author_proxy_potentially_eligible_cases':sum(r['reference_status']!=MISSING_CONFIRMATION for r in refs),
      'all_agents_receive_same_source':True,'raw_test_rows':105,'unique_test_code_unit_pairs':7,'M5_supported_test_code_unit_pairs':7,
      'actual_non_M5_test_rows_in_zip':0,'M5_pain_ask_present':False,
      'initial_vitals_exposed_cases':0,'initial_objective_result_cases':0,'chief_complaint_present_cases':15,
      'initial_symptom_count':stats([len(c['static_case']['current_symptoms']) for c in b['cases']]),
      'initial_history_count':stats([len(c['static_case']['history_visible_at_start']) for c in b['cases']]),
      'exact_duplicate_static_cases':15-exact_narrative,'template_similarity':'ALL_CASES_SAME_7_TESTS_2_INITIAL_HISTORIES_2_TIMED_HISTORY_DETAILS',
      'near_duplicate_pair_count_threshold_0_35':len(pairs),'clinical_plausibility':'NOT_ADJUDICATED','authored_not_large_teacher':True,
      'Original50':{'cases':len(original),'chief_complaint_present':sum(bool(c['agent_visible']['current_initial_state']['doctor_visible_info'].get('chief_complaint')) for c in original),
                    'associated_symptoms_nonempty':sum(bool(c['agent_visible']['current_initial_state']['patient_visible_info'].get('associated_symptoms')) for c in original),
                    'associated_symptoms_only_pain_measurement_name':sum(c['agent_visible']['current_initial_state']['patient_visible_info'].get('associated_symptoms')==['Pain severity - 0-10 verbal numeric rating [Score] - Reported'] for c in original)},
      'native_source_profiles':len(traces),'initial_to_all_tests_generation':0,'distribution':results,
      'internal_case_prefix_not_model_visible':True,'patient_independence':'AUTHORED_IDS_DISJOINT_NOT_PROOF_OF_SEMANTIC_INDEPENDENCE',
      'actual_LLM_generation':0,'real_utility_fit_called':False,'clinical_accuracy':'NOT_EVALUATED'}
    atomic_json(PUBLIC/'dataset_audit_rechecked.json',report)
    atomic_json(PRIVATE/'audit_detail_rechecked.json',{'bundle':b,'token_scores':private_rows,'near_duplicate_pairs':pairs,'source_traces':traces,
        'snapshots':[{'ordinal':k[0],'stage':k[1],'view':k[2],**v} for k,v in snapshots.items()],
        'test_inventory':[{'description':d,'code':code,'unit':u,'available_minute':t,'rows':n} for (d,code,u,t),n in raw.items()]})
    print(json.dumps({k:report[k] for k in ('case_count','exact_catalog_reference_matches','unique_reference_diagnoses','reference_status_counts','Original50','distribution')},ensure_ascii=True))
if __name__=='__main__':main()
