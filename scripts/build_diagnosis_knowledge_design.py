"""Saved 45-Episode audit, patient-independent corpus/index and research plan.

No LLM, fitting, source edits or historical output writes.
"""
from collections import Counter,defaultdict
from datetime import datetime
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.medical_knowledge_catalog import corpus
from app.locdi.medical_knowledge import MedicalKnowledgeIndex,tokens,historical_labels
from app.locdi.llm_setup import digest
from app.locdi.common_prediction_contract import shared_selection
from app.locdi.common_runtime_recovery import released_measurement_view
from app.locdi.evidence_selection_wire import encode
from app.locdi.memory_subset_wire import encode as memory_encode
from app.locdi.current_ablation_llm_backend import input_guard,InputContextBudgetError
from app.core.clinical_observation import memory_query
NS='DIAGNOSIS_KNOWLEDGE_DESIGN_20261121_v1'
DOC=ROOT/'docs'/'B0_B4_\uc2e4\ud5d8'/'10_\uc9c4\ub2e8\uc131\ub2a5_\uac1c\uc120\uc124\uacc4'
PUB=ROOT/'outputs/research_evaluation'/NS;PRIVATE=ROOT/'outputs/private/research_evaluation'/NS
BASE=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=True,indent=2,allow_nan=False)
def tab(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stamp(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def main():
    global DOC
    if DOC.exists():
        i=2
        while DOC.with_name(DOC.name+'_v'+str(i)).exists():i+=1
        DOC=DOC.with_name(DOC.name+'_v'+str(i))
    for p in (DOC,PUB,PRIVATE):p.mkdir(parents=True,exist_ok=False)
    files=set()
    for directory in (BASE,ROOT/'outputs/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001',ROOT/'app/core',ROOT/'prompts',ROOT/'configs',ROOT/'outputs/private/research_evaluation/B34_RUNTIME_RECOVERY_20261121_v1'):
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    model_path=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json';files.add(model_path)
    files.add(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    protection={str(p.resolve()):sha(p) for p in files};write(PRIVATE/'protection_before.json',protection)
    catalog=corpus();model=read(model_path);assert {c['disease'] for c in catalog['cards']}==set(model['diagnoses'])
    ix=MedicalKnowledgeIndex(catalog);write(PUB/'medical_knowledge_corpus.json',catalog)
    write(PUB/'bm25_index.json',{'version':'bm25_k1_1.2_b0.75','corpus_fingerprint':ix.fingerprint,
        'documents':[{'id':c['concept_id'],'terms':dict(t)} for c,t in zip(ix.cards,ix.tf)],'document_frequency':dict(ix.df),'average_length':ix.average_length})
    write(PUB/'common_knowledge_packet.json',ix.common())
    bindings=read(BASE/'bindings.json');panel=read(BASE/'panel.json');coll=read(BASE/'run_runtime_recovery_all_001/collection.json')
    result_rows=[];details=[];memory_rows=[];snapshots={};citation_counts=Counter();kinds=defaultdict(Counter);actions=defaultdict(Counter);guards=[]
    for row in coll['rows']:
        if row['status']!='COMPLETED':continue
        n=row['episode_ordinal'];co=row['condition'];i=row['patient_ordinal'];folder=BASE/'run_runtime_recovery_all_001/requests'/('Episode%04d'%n)
        rt=read(folder/'runtime_snapshot_private.json');ds=read(folder/'actual_core_decisions.json');after=read(sorted(folder.glob('Request*_after.json'))[-1])
        b=read(sorted(folder.glob('Request*_before.json'))[-1]);payload=json.loads(b['request']['messages'][1]['content']);obs=payload['observation'];prediction=row['result']['prediction'];reference=panel[i-1]['reference']
        records=obs.get('longitudinal_memory',{}).get('records',[]);cited=after.get('internal_response',{}).get('memory_ids',[])
        audit=rt['llm_audit'][-1];validation=audit.get('decision_validation',{});citations=validation.get('cited_evidence',[])
        exact=prediction['diagnosis'] in reference['accepted_diagnoses'] if prediction.get('diagnosis') else None
        public={'patient_ordinal':i,'condition':co,'source_exact':exact,'certainty':prediction['certainty'],
            'actions':len(ds),'final_decision_minute':ds[-1]['minute'],
            'terminal_elapsed_minutes':read(folder/'episode_and_policy.json')['episode']['result']['elapsed_simulation_minutes'],
            'tests_observed_before_final':len(obs['objective_findings']),
            'questions_answered_before_final':sum(h.get('answer_status')!='unknown' for h in obs['history']),
            'memory_records_before_final':len(records),'memory_citations':len(cited),'eligible_citations':sum(c.get('supports_finalization') is True for c in citations),
            'generation':row['request_count'],'prompt_tokens':sum(read(p)['response_envelope'].get('prompt_eval_count',0) for p in folder.glob('Request*_after.json')),
            'external_knowledge_present':'external_medical_knowledge' in payload}
        result_rows.append(public);snapshots[i,co]=obs;kinds[co].update(r['kind'] for r in records);actions[co].update(d['action'] for d in ds)
        full={'summary':public,'reference':reference,'prediction':prediction,'LLM_last_proposal':after.get('internal_response'),
            'Core_terminal':ds[-1],'current_observation_before_terminal':obs,'decision_validation':validation,
            'cause_status':'OBSERVED_DIFFERENCES_ONLY_RATIONALE_NOT_CAUSAL_PROOF'}
        details.append(full)
        query=set(tokens(memory_query(obs)));cutoff=stamp(bindings[row['case_ref']]['started_at'])
        for index,r in enumerate(records):
            assert r['patient_id']==bindings[row['case_ref']]['patient_id'] and r['visit_id']!=bindings[row['case_ref']]['visit_id']
            assert stamp(r['available_at'])<=cutoff and stamp(r['visit_started_at'])<cutoff
            memory_rows.append({'patient_ordinal':i,'condition':co,'memory_index':index,'kind':r['kind'],
                'age_days':(cutoff-stamp(r['observed_at'])).total_seconds()/86400,'query_token_overlap':len(query&set(tokens(r['text']))),
                'cited':r['memory_id'] in cited,'source_status':r.get('status_as_of_visit'),
                'clinical_relevance':'NOT_REVIEWED','clinical_stale':'NOT_REVIEWED','clinical_conflict':'NOT_REVIEWED'})
    matched=[r['patient_ordinal'] for r in result_rows if r['condition']=='B0' and r['source_exact']]
    comparisons=[]
    for i in matched:
        baseline=snapshots[i,'B0'];a=set(baseline['objective_findings'])
        for co in ('B0','B1','B2','B3','B4'):
            o=snapshots[i,co];summary=next(r for r in result_rows if r['patient_ordinal']==i and r['condition']==co)
            comparisons.append({**summary,'current_results_equal_B0':set(o['objective_findings'])==a,
                'current_results_added_vs_B0':len(set(o['objective_findings'])-a),'current_results_missing_vs_B0':len(a-set(o['objective_findings']))})
    # Apply candidate interventions to saved inputs only, never regenerate a response.
    for row in coll['rows']:
        if row['status']=='NOT_ATTEMPTED':continue
        folder=BASE/'run_runtime_recovery_all_001/requests'/('Episode%04d'%row['episode_ordinal'])
        for bp in sorted(folder.glob('Request*_before.json')):
            req=read(bp)['request'];p=json.loads(req['messages'][1]['content']);s=req['format']
            p['instruction']=p.get('instruction','').split(' Output wire memory_citation_subset_code.v1:',1)[0]
            # Reconstruct pre-memory wire schema for fair size check.
            from app.locdi.memory_citation_wire import _decision_nodes
            for node in _decision_nodes(s):
                props=node['properties'];props.pop('memory_subset_code',None)
                node['required']=[k for k in node['required'] if k!='memory_subset_code']
                ids=[r['memory_id'] for r in p['observation'].get('longitudinal_memory',{}).get('records',[])]
                if ids:
                    props['memory_ids']={'type':'array','uniqueItems':True,'maxItems':min(4,len(ids)),'items':{'enum':ids}}
                    node['required'].append('memory_ids')
            for experiment in ('E0R','E1','E2','E3','E4'):
                q=json.loads(json.dumps(p));schema=json.loads(json.dumps(s))
                if experiment in ('E1','E4'):q['external_medical_knowledge']=ix.common()
                if experiment=='E2':q['memory_scope_annotations']=historical_labels(q['observation'])
                q,schema=encode(q,schema);q,schema,_=memory_encode(q,schema)
                try:g=input_guard(req['messages'][0]['content'],json.dumps(q,ensure_ascii=False,sort_keys=True),schema);status='PASS';bound=g['conservative_input_token_bound']
                except InputContextBudgetError:status='BLOCKED';bound=None
                guards.append({'experiment':experiment,'condition':row['condition'],'patient_ordinal':row['patient_ordinal'],'request':bp.name,'status':status,'bound':bound})
    # Existing synthetic source cohort is a candidate, not declared an unseen holdout.
    admitted=read(ROOT/'outputs/private/research_evaluation/c4_35_v1/approved_admission.json')['panel']
    data=[json.loads(s) for s in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf-8').splitlines() if s]
    excluded={c['patient_id'] for c in data}|set(model['provenance']['training_patients'])
    for path in (ROOT/'outputs/private/research_evaluation/B34_COLLECTION_RECOVERY_v1/source_collection_recovery_001/plan.json',ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/followup_catalog_all_001/routing_training/plan.json',ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/source_ready_002/routing_training/plan.json'):
        excluded.update(c['decision']['patient_id'] for c in read(path)['candidates'])
    candidates={v['patient_id']:v for v in admitted if v['patient_id'] not in excluded}
    chosen=sorted(candidates,key=lambda pid:digest(['knowledge-holdout-20261121',pid]))[:10]
    write(PRIVATE/'holdout_candidate_manifest.json',{'status':'CANDIDATES_NOT_EXECUTION_READY','patients':[candidates[p] for p in chosen],
        'selection':'patient hash seed 20261121; no diagnosis/outcome/Agent response selection',
        'prior_exposure':'C4-35 Source/VoI evaluation; not a completely untouched cohort',
        'blockers':['NATIVE_CASE_ADAPTER_NOT_PREPARED','INPUT_CONTENT_DUPLICATE_AUDIT_NOT_DONE','REFERENCE_TASK_VALIDITY_NOT_REVIEWED','LIVE_BUDGET_APPROVAL_NOT_PROVIDED']})
    write(PRIVATE/'trace_comparison_details.json',details)
    tab(DOC/'TRACE_FACTS.csv',result_rows);tab(DOC/'B0_MATCHED_CASE_COMPARISON.csv',comparisons);tab(DOC/'MEMORY_FACTS.csv',memory_rows);tab(DOC/'SAVED_INPUT_GUARDS.csv',guards)
    stats={'namespace':NS,'docs_folder':str(DOC.relative_to(ROOT)),'completed_episode_count':len(result_rows),'B0_matching_ordinals':matched,
        'B0_matching_lost_all_other_conditions':all(not r['source_exact'] for r in comparisons if r['condition']!='B0'),
        'memory_kind_before_terminal':{k:dict(v) for k,v in kinds.items()},'memory_fact_count':len(memory_rows),
        'source_cutoff_checks':'PASS_FOR_SUPPLIED_RECORDS','clinical_relevance_review':'NOT_REVIEWED',
        'common_pack_bytes':len(json.dumps(ix.common()).encode()),'corpus_topics':len(ix.cards),'sources':len(catalog['sources']),
        'guard_results':{e:dict(Counter(r['status'] for r in guards if r['experiment']==e)) for e in ('E0R','E1','E2','E3','E4')},
        'guard_max':{e:max((r['bound'] or 0 for r in guards if r['experiment']==e),default=0) for e in ('E0R','E1','E2','E3','E4')},
        'disjoint_existing_source_holdout_candidates':len(candidates),'proposed_candidate_patients':len(chosen),'holdout_ready':False,
        'new_LLM_generation':0,'new_fitting':0,'independent_clinical_review':False,'clinical_accuracy':'NOT_EVALUATED'}
    write(PUB/'analysis.json',stats);write(DOC/'ANALYSIS.json',stats)
    assert all(sha(p)==h for p,h in protection.items());print(json.dumps(stats))
if __name__=='__main__':main()
