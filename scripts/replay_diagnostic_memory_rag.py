"""Offline Source-checked M2/K2 projection; never scores diagnostic correctness."""
from collections import Counter
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,memory_source_map,original_cases
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.clinical_observation import memory_query
from app.locdi.diagnostic_memory_rag import DiagnosticMemoryRetriever,public_projection,knowledge_projection
from app.locdi.medical_knowledge import MedicalKnowledgeIndex
from app.locdi.medical_knowledge_catalog import corpus
from app.locdi.routing_collector import atomic_json
NS='MEMORY_RAG_LEGACY_RECOVERY_20261121_v1'

def main():
    cases=original_cases();base=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001'
    plan=read(ROOT/'outputs/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001/plan.json')
    index=read(base/'case_index.json');selected=[cases[n-1] for n in index.values()];sources,sr=memory_source_map(selected)
    private=ROOT/'outputs/private/research_evaluation'/NS;public=ROOT/'outputs/research_evaluation'/NS
    atomic_json(private/'panel10_memory_sources.json',sources);atomic_json(private/'panel10_source_verification.json',sr)
    bindings=read(base/'bindings.json');collection=read(base/plan['run_id']/'collection.json');retriever=DiagnosticMemoryRetriever(sources);knowledge=MedicalKnowledgeIndex(corpus());rows=[];counts=Counter()
    for episode in collection['rows']:
        if episode['condition']!='B2':continue
        ref=episode['case_ref'];c=cases[index[ref]-1];identity=c['agent_visible']['longitudinal']
        store=PatientMemoryStore([r for v in c['agent_visible']['past_visits'] for r in v['records']])
        folder=base/plan['run_id']/'requests'/('Episode%04d'%episode['episode_ordinal'])
        before=sorted(folder.glob('Request*_before.json'))
        for bp in before:
            p=json.loads(read(bp)['request']['messages'][1]['content']);obs=p['observation'];legacy=obs['longitudinal_memory']
            _,q=public_projection(obs,bindings[ref]);pool=store.retrieve(identity,memory_query(obs),'full_history')['records']
            m2,trace=retriever.retrieve(legacy,pool,identity,q['query'],max_chars=legacy.get('max_chars'))
            rerank,reranktrace=retriever.retrieve(legacy,legacy['records'],identity,q['query'],max_chars=legacy.get('max_chars'))
            k2,_=knowledge_projection(knowledge,obs,bindings[ref])
            row={'panel_ordinal':episode['patient_ordinal'],'checkpoint':str(bp.relative_to(ROOT)),
                'last_saved_request':bp==before[-1],'legacy':legacy,'M2_full_pool':m2,'M2_trace':trace,
                'TEST_ONLY_DIAGNOSTIC_NOT_REGISTERED_TOP5_RERANK':rerank,'K2':k2,'query_source':q}
            rows.append(row);counts['saved_B2_requests']+=1;counts['K2_available_requests']+=k2['status']=='AVAILABLE'
            counts['M2_empty_requests']+=not bool(m2['records']);counts['full_pool_vs_rerank_selection_diff_requests']+=([r['memory_id'] for r in m2['records']]!=[r['memory_id'] for r in rerank['records']])
            if row['last_saved_request']:
                counts['completed_or_attempted_patients_with_last_request']+=1
                for name,records in [('legacy',legacy['records']),('M2',m2['records'])]:
                    counts[name+'_final_records']+=len(records);counts[name+'_final_pain_records']+=sum(str(sources[r['memory_id']]['code'])=='72514-3' for r in records)
    atomic_json(private/'offline_M2_K2_payload_projection.json',rows)
    report={'status':'OFFLINE_PROJECTION_ONLY','counts':dict(counts),'original_reference_consulted':False,
        'clinical_relevance_or_accuracy':'NOT_EVALUATED','LLM_generation':0,'fitting':0,
        'warning':'Changed inputs do not establish changed LLM actions, diagnosis or causal performance. Full-pool and Top5 rerank are different candidate scopes; live C2 uses only full-pool.'}
    atomic_json(public/'offline_projection_summary.json',report);print(json.dumps(report))
if __name__=='__main__':main()
