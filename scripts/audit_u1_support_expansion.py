"""Per-case Support/tie audit on frozen traces; never estimates new labels."""
from pathlib import Path
from collections import Counter,defaultdict
import json,csv,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.llm_setup import validate_setup
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import observed_tokens,jaccard_similarity
from scripts.analyze_b34_utility_voi import U1,RUN,BP
from scripts.run_u1_voi_expansion import PUB,PR

def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def main():
    pub,pr=PUB/'support_audit',PR/'support_audit';pub.mkdir(parents=True,exist_ok=False);pr.mkdir(parents=True,exist_ok=False)
    before={str(f.resolve()):sha_file(f) for f in RUN.rglob('*') if f.is_file()};before[str(U1.resolve())]=sha_file(U1);write(pr/'preservation_before.json',before)
    artifact=read(U1)['artifact'];reg=validate_setup(read(BP/'llm_config.json'))['prepared']['registry'];router=AgentRegistry(reg,artifact);departments=sorted(a['department'] for a in reg['agents'] if a['role']=='specialty')
    train=defaultdict(dict)
    for s in artifact['samples']:train[s['patient_id'],s['decision_id']][s['agent_id']]=s
    aid_card=router.get('specialty','cardiology')['agent_id'];aid_neuro=router.get('specialty','neurology')['agent_id']
    discordant=[g for g in train.values() if g[aid_card]['correct']!=g[aid_neuro]['correct']]
    rows=[];agent_rows=[];private=[]
    for episode in read(RUN/'collection.json')['rows']:
        if episode['condition'] not in ('B3','B4'):continue
        folder=RUN/'requests'/episode['dataset_view']/('Episode%04d'%episode['view_episode_ordinal']);trajectory=[]
        for ordinal,file in enumerate(sorted(folder.glob('Request*_before.json')),1):
            payload=json.loads(read(file)['request']['messages'][1]['content']);obs=payload['observation'];tokens=observed_tokens(obs);d=router.route(obs,departments,mode='adaptive');estimates=d['utility_estimates']
            scores=sorted((e['score'] for e in estimates.values() if isinstance(e,dict)),reverse=True)
            supported_discriminators=sum(jaccard_similarity(tokens,g[aid_card]['tokens'])>0 for g in discordant)
            step={'request_ordinal':ordinal,'decision_minute':obs['elapsed_minutes'],'selected':d['selected_specialty'],'status':d['decision_status'],'ties':d['tied_departments'],
                'expected_role_if_U1_applied':'er' if d['selected_specialty'] is None else 'specialty','margin':scores[0]-scores[1] if len(scores)>1 else None,
                'cardio_neuro_discordant_training_groups_with_nonzero_similarity':supported_discriminators}
            trajectory.append(step)
            for department in departments:
                aid=router.get('specialty',department)['agent_id'];sims=[jaccard_similarity(tokens,s['tokens']) for s in artifact['samples'] if s['agent_id']==aid];estimate=estimates.get(department) or {}
                agent_rows.append({'panel_ordinal':episode['patient_ordinal'],'view':episode['dataset_view'],'condition':episode['condition'],'request_ordinal':ordinal,'agent':department,
                    'matched_samples':sum(s>0 for s in sims),'Jaccard_min':min(sims),'Jaccard_max':max(sims),'Jaccard_mean':sum(sims)/len(sims),'score':estimate.get('score'),'weight':estimate.get('effective_weight'),'margin':step['margin'],'selection':d['selected_specialty']})
            private.append({'panel_ordinal':episode['patient_ordinal'],'view':episode['dataset_view'],'condition':episode['condition'],'step':step,'tokens':tokens,'routing':d})
        initial,final=trajectory[0],trajectory[-1];rows.append({'panel_ordinal':episode['patient_ordinal'],'view':episode['dataset_view'],'condition':episode['condition'],
            'trace_request_count':len(trajectory),'initial_status':initial['status'],'initial_selected':initial['selected'],'initial_ties':'|'.join(initial['ties']),
            'initial_margin':initial['margin'],'initial_nonzero_discriminating_groups':initial['cardio_neuro_discordant_training_groups_with_nonzero_similarity'],
            'final_status':final['status'],'final_selected':final['selected'],'hypothetical_ER_decisions':sum(t['expected_role_if_U1_applied']=='er' for t in trajectory),
            'selected_distribution':dict(Counter(t['selected'] or 'ABSTAIN' for t in trajectory)),'interpretation':'COUNTERFACTUAL_ON_FROZEN_U0_PATH_NOT_ACTUAL_U1_ROLLOUT'})
    summary={'cases':rows,'cardio_neuro_discordant_train_groups':len(discordant),'saved_snapshot_count':len(private),'initial_case_count_per_view_condition':10,
        'all_scores_are_existing_Core_calculations':True,'model_calls':0,'fitting':0,'clinical_accuracy':'NOT_EVALUATED'}
    write(pub/'support_summary.json',summary);write(pr/'support_details.json',private)
    for name,data in (('case_support.csv',rows),('agent_support.csv',agent_rows)):
        with (pub/name).open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    changed=[f for f,h in before.items() if sha_file(f)!=h];assert not changed;write(pub/'preservation.json',{'files':len(before),'changed':0})
    print(json.dumps({'cases':len(rows),'snapshots':len(private),'agent_rows':len(agent_rows),'discordant_groups':len(discordant),'changed':0,'new_generation':0}))
if __name__=='__main__':main()
