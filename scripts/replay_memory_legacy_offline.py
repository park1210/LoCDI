"""Read saved checkpoints, recompute original Memory only, never regenerate."""
from collections import Counter
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.clinical_observation import memory_query
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.original_source_inventory import read,DATA
from app.locdi.routing_collector import atomic_json
NS='MEMORY_RAG_LEGACY_RECOVERY_20261121_v1'

def main():
    cases=load_cases(DATA);rows=[];counts=Counter();stored={}
    for name in ('recovery_llm_all_001','runtime_recovery_all_001'):
        pr=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1'/name
        p=read(ROOT/'outputs/research_evaluation/B34_CONNECTION_v1'/name/'plan.json');index=read(pr/'case_index.json')
        run=pr/p['run_id'];episodes=read(run/'collection.json')['rows']
        for episode in episodes:
            c=cases[index[episode['case_ref']]-1];identity=c['agent_visible']['longitudinal']
            store=PatientMemoryStore([r for v in c['agent_visible']['past_visits'] for r in v['records']])
            folder=run/'requests'/('Episode%04d'%episode['episode_ordinal'])
            for before in sorted(folder.glob('Request*_before.json')):
                b=read(before);req=b['request'];payload=json.loads(req['messages'][1]['content']);o=payload['observation'];memory=o.get('longitudinal_memory',{})
                wire_ok=digest(req)==b['request_fingerprint'];assert wire_ok,'SAVED_REQUEST_HASH'
                row={'plan_id':name,'condition':episode['condition'],'patient_ordinal':episode['patient_ordinal'],
                    'checkpoint':str(before.relative_to(ROOT)),'saved_request_fingerprint':b['request_fingerprint'],
                    'saved_payload_fingerprint':digest(payload),'request_hash_valid':wire_ok,'memory_contract':memory.get('mode'),
                    'memory_source_records_preserved':all({k:r[k] for k in store._by_id[r['memory_id']]}==store._by_id[r['memory_id']] for r in memory.get('records',[]))}
                assert row['memory_source_records_preserved']
                if memory.get('mode')=='selective':
                    got=store.retrieve(identity,memory_query(o),'selective',5,max_chars=memory.get('max_chars'))
                    row.update(recomputed_record_ids_match=[r['memory_id'] for r in got['records']]==[r['memory_id'] for r in memory['records']],
                        recomputed_record_scores_order_times_exact=got['records']==memory['records'],
                        recomputed_memory_context_exact=got==memory)
                    assert row['recomputed_record_scores_order_times_exact'],'LEGACY_MEMORY_REPLAY_MISMATCH'
                    counts['selective_requests']+=1
                    counts['selective_context_exact']+=int(got==memory)
                rows.append(row);counts['saved_requests']+=1
        stored[name]={'plan_fingerprint':p['fingerprint'],'source_binding':p['source_binding_fingerprint'],
            'code_hashes':p['code_hashes'],'evidence_index_set':p.get('evidence_index_set'),
            'runtime_recovery':p.get('runtime_recovery'),'catalog_selection':p.get('catalog_selection')}
    pr=ROOT/'outputs/private/research_evaluation'/NS;pub=ROOT/'outputs/research_evaluation'/NS
    atomic_json(pr/'legacy_replay_details.json',rows)
    contracts={'same_source_binding':len({v['source_binding'] for v in stored.values()})==1,
        'same_code_pins':len({digest(v['code_hashes']) for v in stored.values()})==1,
        'wire_flags':{n:{k:v[k] for k in ('evidence_index_set','runtime_recovery','catalog_selection')} for n,v in stored.items()}}
    atomic_json(pr/'historical_contract_comparison.json',{'plans':stored,'comparison':contracts})
    report={'status':'PASS','counts':dict(counts),'actual_LLM_generation':0,'clinical_accuracy':'NOT_EVALUATED',
        'saved_request_hash_scope':'FULL_STORED_REQUEST_OBJECT','legacy_replay_scope':'FIXED_SOURCE_RECORDS + ORIGINAL_MEMORY_QUERY + CURRENT_UNMODIFIED_RETRIEVE',
        'does_not_claim_historical_llm_regeneration':True,'contract_comparison':contracts}
    atomic_json(pub/'legacy_replay_report.json',report);print(json.dumps(report))
if __name__=='__main__':main()
