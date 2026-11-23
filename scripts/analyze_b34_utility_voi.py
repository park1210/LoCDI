"""Non-generating U0/U1 replay and B4 STOP audit. Never fits Utility."""
from pathlib import Path
from collections import Counter
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.authored_utility_transfer import verify,EXPECTED_SHA
from app.locdi.llm_setup import validate_setup
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import observed_tokens,jaccard_similarity
from app.locdi.synthea_diagnosis_audit import matches

NS='B34_UTILITY_VOI_IMPROVEMENT_v1';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
U1=ROOT/'outputs/private/research_evaluation/SYNTHETIC_PILOT_UTILITY_20261121_v1/syn15_initial_full_005/development_utility.json'
U0=ROOT/'outputs/private/research_evaluation/B34_RUNTIME_RECOVERY_20261121_v1/development_routing_evidence.json'
BP=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/original50_d0d2_full_003_source'
RUN=ROOT/'outputs/private/research_evaluation/ORIGINAL50_D0_D2_EXPERIMENT_v1/original50_d0d2_full_003/run_original50_d0d2_full_003'

def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def main():
    PUB.mkdir(exist_ok=False);PR.mkdir(exist_ok=False);cfg=read(BP/'llm_config.json');registry=validate_setup(cfg)['prepared']['registry'];analysis=read(RUN/'analysis_private.json');panel=read(BP/'panel.json');cases=read(BP/'case_index.json')
    from app.locdi.original_source_inventory import original_cases
    originals=original_cases();patients=[originals[n-1]['patient_id'] for n in cases.values()]
    v=verify(U1,registry,patients,cfg);arts={'U0':read(U0)['routing_artifact'],'U1':v['artifact']};routers={k:AgentRegistry(registry,a) for k,a in arts.items()}
    protected={str(f.resolve()):sha_file(f) for f in RUN.rglob('*') if f.is_file()};protected.update(v['origin_files']);protected[str(U0.resolve())]=sha_file(U0);write(PR/'preservation_before.json',protected)
    routing=[];full_routes=[];summaries={};departments=sorted(a['department'] for a in registry['agents'] if a['role']=='specialty')
    for row in analysis['rows']:
        if row['condition'] not in ('B3','B4'):continue
        folder=RUN/'requests'/row['dataset_view']/('Episode%04d'%row['view_episode_ordinal'])
        for i,bf in enumerate(sorted(folder.glob('Request*_before.json')),1):
            payload=json.loads(read(bf)['request']['messages'][1]['content']);obs=payload['observation'];tokens=set(observed_tokens(obs))
            for key,router in routers.items():
                decision=router.route(obs,departments,mode='adaptive');scores=sorted((x for x in decision['specialty_scores'].values() if x is not None),reverse=True)
                # Replay uses the preselection candidate five, never available_departments=[] after handoff.
                support={}
                for agent in departments:
                    aid=router.get('specialty',agent)['agent_id'];sims=[jaccard_similarity(tokens,s['tokens']) for s in arts[key]['samples'] if s['agent_id']==aid]
                    support[agent]={'positive_similarity_count':sum(x>0 for x in sims),'max_jaccard':max(sims),'mean_jaccard':sum(sims)/len(sims)}
                item={'panel_ordinal':row['patient_ordinal'],'view':row['dataset_view'],'condition':row['condition'],'request_ordinal':i,'utility':key,
                    'token_count':len(tokens),'selected':decision['selected_specialty'],'status':decision['decision_status'],'scores':decision['specialty_scores'],
                    'margin':scores[0]-scores[1] if len(scores)>1 else None,'support':support,'utility_estimates':decision['utility_estimates']}
                routing.append(item);full_routes.append({**item,'tokens':sorted(tokens)})
    for view in ('D0','D2'):
        for cond in ('B3','B4'):
            for key in arts:
                rows=[r for r in routing if r['view']==view and r['condition']==cond and r['utility']==key and r['request_ordinal']==1]
                summaries[view+'_'+cond+'_'+key]={'registered':10,'selection':dict(Counter(r['selected'] or 'ABSTAIN' for r in rows)),
                    'abstention_rate':sum(r['status']=='abstained' for r in rows)/10,'mean_token_count':sum(r['token_count'] for r in rows)/10,
                    'matched_sample_range':[min(x['positive_similarity_count'] for r in rows for x in r['support'].values()),max(x['positive_similarity_count'] for r in rows for x in r['support'].values())]}
    # STOP and overwritten proposals: compare actual core responses, not rationale speculation.
    stop=[];public_regression=[];private_regression=[];lookup={(r['patient_ordinal'],r['condition'],r['dataset_view']):r for r in analysis['rows']}
    for view in ('D0','D2'):
        for i in range(1,11):
            row=lookup[i,'B4',view];f=row['facts'];hist=f['voi_trace'];folder=RUN/'requests'/view/('Episode%04d'%row['view_episode_ordinal']);proposals=[]
            for j,cf in enumerate(sorted(folder.glob('Request*_core_response.json'))):
                candidate=read(cf)['decoded_response'];h=hist[j];correct=candidate['action']=='diagnose' and matches(candidate['target'],f['reference']['accepted_diagnoses'])['source_exact'] is True
                proposals.append({'minute':h['minute'],'candidate':candidate,'source_exact_diagnose':correct,'selected':h['selected_action'],
                    'selected_target':h.get('target'),'action_overridden':candidate['action']!=('ask_question' if h['selected_action']=='ask_patient' else h['selected_action'])})
            b=read(sorted(folder.glob('Request*_before.json'))[-1]);body=json.loads(b['request']['messages'][1]['content']);prior=hist[0]['belief'];last=hist[-1];gold=f['reference']['accepted_diagnoses']
            # Reference comparison is private analysis only, never sent back to a model or planner.
            source_mass_before=sum(prior.get(d,0) for d in gold);source_mass_after=sum(last['belief'].get(d,0) for d in gold)
            stop.append({'view':view,'panel_ordinal':i,'STOP_minute':last['minute'],'STOP_reason':last['stop_reason'],'posterior':last['belief'],
                'source_reference':f['reference'],'source_mass_before':source_mass_before,'source_mass_after':source_mass_after,
                'algorithmic_diagnosis':row['result']['prediction']['algorithmic_diagnosis_choice'],'hospital_prediction':f['prediction'],'final_snapshot':body,
                'actual_proposals':proposals,'correct_proposals_overridden':sum(p['source_exact_diagnose'] and p['selected']!='diagnose' for p in proposals)})
    for i in range(1,11):
        b2=lookup[i,'B2','D2']
        if b2['facts']['source_exact'] is not True:continue
        for cond in ('B3','B4'):
            row=lookup[i,cond,'D2'];last=next((r for r in stop if r['panel_ordinal']==i and r['view']=='D2'),None)
            public_regression.append({'panel_ordinal':i,'comparison':'B2_TO_'+cond,'transition':'CORRECT_TO_INCORRECT','selected_specialty':row['facts']['routing_trace'][0]['selected_specialty'],
                'B2_generation':b2['request_count'],'target_generation':row['request_count'],'diagnoses_and_reference':'PRIVATE_ONLY'})
            private_regression.append({'panel_ordinal':i,'condition':cond,'B2':b2,'target':row,'STOP':last if cond=='B4' else None})
    voi={view:{'correct_proposals_overridden':sum(r['correct_proposals_overridden'] for r in stop if r['view']==view),
        'patients_with_correct_proposal_overridden':sum(r['correct_proposals_overridden']>0 for r in stop if r['view']==view),
        'source_posterior_increased_patients':sum(r['source_mass_after']>r['source_mass_before'] for r in stop if r['view']==view),
        'source_posterior_decreased_patients':sum(r['source_mass_after']<r['source_mass_before'] for r in stop if r['view']==view)} for view in ('D0','D2')}
    train_tokens=[len(s['tokens']) for s in arts['U1']['samples']];label_counts={a:dict(Counter(str(s['correct']) for s in arts['U1']['samples'] if s['agent_id']==a)) for a in sorted({s['agent_id'] for s in arts['U1']['samples']})}
    m=read(ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json')
    result={'U1_sha256':EXPECTED_SHA,'U1_complete_train_groups':11,'U1_samples':55,'U1_label_counts':label_counts,'routing_initial':summaries,'all_frozen_trace_replay_count':len(routing),
        'U1_training_token_range':[min(train_tokens),max(train_tokens)],'voi':voi,'D2_B2_correct_patients':6,'D2_B2_to_B34_lost_pairs':len(public_regression),
        'M5_actions':len(m['actions']),'M5_symptom_likelihood':False,'M5_review_status':m['provenance']['review_status'],
        'Native_existing_admission':'NOT_SUPPORTED_BY_EXISTING_SYNTHETIC_ARTIFACT_SCHEMA','new_transfer_admission':'EXPLICIT_DEVELOPMENT_ONLY_NOT_CLINICAL',
        'routing_improvement_confirmed':False,'U1_retraining_performed':False,'actual_LLM_generation':0,'clinical_accuracy':'NOT_EVALUATED'}
    write(PUB/'analysis_summary.json',result);write(PR/'routing_scores.json',full_routes);write(PR/'STOP_case_analysis.json',stop);write(PR/'B2_regression.json',private_regression)
    with (PUB/'B2_TO_B34_CORRECTNESS_REGRESSION.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(public_regression[0]));w.writeheader();w.writerows(public_regression)
    changed=[f for f,h in protected.items() if sha_file(f)!=h];assert not changed
    write(PUB/'preservation.json',{'files':len(protected),'changed':0,'LLM_generation':0,'fitting':0})
    print(json.dumps(result))

if __name__=='__main__':main()
