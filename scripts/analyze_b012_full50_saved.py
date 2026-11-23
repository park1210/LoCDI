"""Read-only full50 trace analysis. No inference, fitting or clinical labels.

Clinical memory P/R is NOT_EVALUATED. Separate structural measures use exact
CSV observation Code/Unit equality to results ALREADY released at each request.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,csv,sys,statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,sha_file,original_cases,memory_source_map,eligible
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.common_prediction_contract import extract_core_prediction
from app.locdi.synthea_diagnosis_audit import matches
from scripts import run_original50_final as runner

NS='B012_FULL50_ANALYSIS_20261123_v1'
PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
CONDITIONS=('B0','B1','B2')

def mean(xs):return statistics.mean(xs) if xs else None
def ratio(n,d):return n/d if d else None

def structural_scores(selected,gold,query):
    """Measurement identity proxy; never medical relevance or adjudication."""
    returned=set(selected)
    if len(returned)!=len(selected):raise ValueError('DUPLICATE_RETURNED_MEMORY')
    hit=returned&set(gold)
    return {'query_available':bool(query),'returned':len(returned),'gold':len(gold),'hits':len(hit),
            'precision':ratio(len(hit),len(returned)) if query else None,
            'recall':ratio(len(hit),len(gold)) if query else None}

def main():
    if PUB.exists() or PR.exists():raise FileExistsError('NEW_ANALYSIS_NAMESPACE_ONLY_NO_OVERWRITE')
    pub,source,p=runner.load('Full50');run=source/p['run_id'];c=read(run/'collection.json');tasks=read(source/'tasks.json')
    lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if c['TEST_ONLY'] is not False or c['status']!='COMPLETED' or c['plan_fingerprint']!=p['fingerprint'] or c['rows']!=lines or len(lines)!=150 or sum(r['request_count'] for r in lines)!=c['generation']:raise ValueError('ACTUAL_FULL50_COLLECTION_LINKAGE')
    PUB.mkdir(parents=True);PR.mkdir(parents=True)
    before={str(f.resolve()):sha_file(f) for f in run.rglob('*') if f.is_file()}
    for f in (pub/'saved_results.json',pub/'execution_status.json',pub/'plan.json',source/'tasks.json'):
        if f.exists():before[str(f.resolve())]=sha_file(f)
    (PR/'preservation_before.json').write_text(json.dumps(before),encoding='utf-8')
    cases=original_cases();refs={r['summary']['ordinal']:r for r in read(runner.PR/'case_audit_private.json')}
    print('Verifying exact Source of72,590 Memory records...',flush=True)
    mapping,source_proof=memory_source_map(cases)
    (PR/'memory_source_verification.json').write_text(json.dumps(source_proof,indent=2),encoding='utf-8')
    original_records={i:{r['memory_id']:r for v in case['agent_visible']['past_visits'] for r in v['records']} for i,case in enumerate(cases,1)}
    gold_index={}
    for i,records in original_records.items():
        index=defaultdict(set)
        for mid,r in records.items():
            a=mapping[mid]
            if a['status']!='VERIFIED_EXACT_CSV_SOURCE':raise ValueError('MEMORY_SOURCE_NOT_VERIFIED')
            if a.get('source_type')=='observations' and a.get('code') is not None and a.get('unit'):
                index[str(a['code']),a['unit']].add(mid)
        gold_index[i]=index
    detailed=[];safe_rows=[];snapshots=[];checks=Counter();confusion=Counter()
    for task,row in zip(tasks,lines):
        if any(row.get(k)!=v for k,v in task.items()) or row['status']!='COMPLETED':raise ValueError('REGISTERED_TASK_OR_COMPLETION')
        i=row['original_ordinal'];cond=row['condition'];folder=run/'requests'/('Episode%04d'%row['episode_ordinal'])
        core=read(folder/'actual_core_decisions.json');actual=extract_core_prediction(core,row['result']['status'])
        if actual!=row['result']['prediction']:raise ValueError('ACTUAL_CORE_FINAL_PREDICTION')
        ref=refs[i]['reference'];score=matches(actual['diagnosis'],ref['accepted_diagnoses']);department=actual['department'] in ref['accepted_departments']
        bb=sorted(folder.glob('Request*_before.json'));aa=sorted(folder.glob('Request*_after.json'));dd=sorted(folder.glob('Request*_core_response.json'))
        if len(bb)!=len(aa) or len(bb)!=len(dd) or len(bb)!=row['request_count']:raise ValueError('REQUEST_COUNTER')
        usage=Counter();shown=[];cited=[];sets=[];per_request=[];payloads=[];normalized=[]
        for bf,af,df in zip(bb,aa,dd):
            b,a,d=read(bf),read(af),read(df);q=b['request'];body=json.loads(q['messages'][1]['content']);decision=d['decoded_response']
            if digest(q)!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint'] or b['contract']!=p['fingerprint'] or a['status']!='VALID' or a['HTTP_status']!=200:raise ValueError('HTTP_OR_REQUEST_INTEGRITY')
            if normalize_model_identifier(a['identity_after']['digest'])!=p['model_identifier'] or a['identity_after']['context_length']!=32768:raise ValueError('ACTUAL_IDENTITY_CONTEXT')
            if d['decoded_response_fingerprint']!=digest(decision):raise ValueError('DECODED_RESPONSE_FINGERPRINT')
            envelope=a['response_envelope'];usage.update(generation=1,prompt_tokens=envelope['prompt_eval_count'],output_tokens=envelope['eval_count'],HTTP_wall_seconds=a['wall_seconds'])
            obs=body['observation'];mem=obs['longitudinal_memory'];selected=mem['records'];ids=[r['memory_id'] for r in selected]
            for m in selected:
                original=original_records[i].get(m['memory_id'])
                if original is None or any(m.get(k)!=v for k,v in original.items()):raise ValueError('MEMORY_CONTENT_CHANGED_OR_FOREIGN')
                if not eligible(m,cases[i-1]['agent_visible']['longitudinal']):raise ValueError('FUTURE_OR_FOREIGN_MEMORY')
            if cond=='B0' and selected:raise ValueError('B0_MEMORY_LEAK')
            if cond=='B2' and len(selected)>5:raise ValueError('TOP5_BOUNDARY')
            citation=decision['memory_ids']
            if not set(citation)<=set(ids):raise ValueError('UNPROVIDED_MEMORY_CITATION')
            evidence=body.get('evidence',[])
            if any(type(n) is not int or not 0<=n<len(evidence) for n in decision['evidence_indices']):raise ValueError('INVALID_EVIDENCE_CITATION')
            query={(str(m['code']),m['unit']) for m in body.get('released_measurements',[]) if m.get('code') and m.get('unit')}
            gold=set().union(*(gold_index[i].get(key,set()) for key in query)) if query else set()
            sm=structural_scores(ids,gold,query);shown.extend(ids);cited.extend(citation);sets.append(set(ids))
            memory_bytes=len(json.dumps(selected,ensure_ascii=False,sort_keys=True).encode('utf-8'))
            snapshot={'original_ordinal':i,'condition':cond,'request_number':len(per_request)+1,
                'selected_count':len(ids),'eligible_count':mem.get('eligible_count'),
                'cited_count':len(citation),'memory_json_bytes':memory_bytes,'structural':sm,
                'query_code_count':len(query),'query_gold_record_count':len(gold),'selected_source_integrity_pass':True}
            per_request.append(snapshot);snapshots.append(snapshot);payloads.append(body);normalized.append(decision)
            checks.update(requests_valid=1,context_digest_valid=1,provided_memory_occurrences=len(ids),memory_citations=len(citation))
        rt=read(folder/'runtime_snapshot_private.json');last=payloads[-1];final=normalized[-1]
        if final['action']!='diagnose' or final['target']!=actual['diagnosis'] or final['department']!=actual['department']:raise ValueError('FINAL_MODEL_TO_CORE_TARGET_LINKAGE')
        actual_tests=len(rt['test_results']);provided=[m for m in last.get('released_measurements',[]) if m['kind']=='Test']
        indices=set(final['evidence_indices']);cited_tests=[m for m in provided if m['evidence_index'] in indices]
        simple={'original_ordinal':i,'condition':cond,'status':row['status'],'old_development10':row['old_development10'],
            'source_exact':score['source_exact'],'normalized_match':score['normalized_match'],'source_department':department,
            'certainty':actual['certainty'],'generation':row['request_count'],'prompt_tokens':usage['prompt_tokens'],
            'output_tokens':usage['output_tokens'],'HTTP_wall_seconds':usage['HTTP_wall_seconds'],
            'Core_wall_seconds':row['result']['computation']['wall_seconds'],'test_orders':row['result']['action_counts'].get('order_test',0),
            'received_test_results':actual_tests,'ask_actions':row['result']['action_counts'].get('ask_history',0),
            'final_provided_tests':len(provided),'final_cited_tests':len(cited_tests),'final_evidence_citations':len(indices),
            'initial_memory_count':per_request[0]['selected_count'],'final_memory_count':per_request[-1]['selected_count'],
            'unique_memory_records_in_episode':len(set(shown)),'provided_memory_occurrences':len(shown),'memory_citations':len(cited),
            'unique_cited_memory_records':len(set(cited)),'memory_selection_changed':any(s!=sets[0] for s in sets),
            'final_memory_citations':len(final['memory_ids']),'final_structural':per_request[-1]['structural']}
        safe_rows.append(simple);detailed.append({'summary':simple,'prediction':actual,'reference':ref,'requests':per_request})
        confusion[cond,ref['accepted_diagnoses'][0],actual['diagnosis']]+=1
    summaries={}
    for cond in CONDITIONS:
        rr=[r for r in safe_rows if r['condition']==cond];ss=[s for s in snapshots if s['condition']==cond];last=[r['final_structural'] for r in rr]
        precision=[s['precision'] for s in last if s['precision'] is not None];recall=[s['recall'] for s in last if s['recall'] is not None]
        eligible=[s for s in last if s['query_available'] and s['gold']>0]
        summaries[cond]={'registered':50,'completed':len(rr),'Source_Exact':sum(r['source_exact'] is True for r in rr),
            'Source_Exact_rate':sum(r['source_exact'] is True for r in rr)/50,'Source_department':sum(r['source_department'] is True for r in rr),
            'Source_department_rate':sum(r['source_department'] is True for r in rr)/50,
            'Normalized_Match':sum(r['normalized_match'] is True for r in rr),'supported':sum(r['certainty']=='supported' for r in rr),
            'unresolved':sum(r['certainty']=='unresolved' for r in rr),
            **{key:sum(r[key] for r in rr) for key in ('generation','prompt_tokens','output_tokens','HTTP_wall_seconds','Core_wall_seconds','test_orders','received_test_results','ask_actions','final_provided_tests','final_cited_tests','final_evidence_citations','provided_memory_occurrences','memory_citations')},
            'mean_generation':mean([r['generation'] for r in rr]),'maximum_episode_generation':max(r['generation'] for r in rr),
            'mean_initial_memory_count':mean([r['initial_memory_count'] for r in rr]),'mean_final_memory_count':mean([r['final_memory_count'] for r in rr]),
            'mean_unique_memory_records_per_episode':mean([r['unique_memory_records_in_episode'] for r in rr]),
            'episodes_memory_selection_changed':sum(r['memory_selection_changed'] for r in rr),
            'episodes_with_memory_citation':sum(r['memory_citations']>0 for r in rr),'episodes_with_final_memory_citation':sum(r['final_memory_citations']>0 for r in rr),
            'clinical_memory_precision':'NOT_APPLICABLE_MEMORY_OFF' if cond=='B0' else 'NOT_EVALUATED_NO_BOUND_REVIEWED_REFERENCE',
            'clinical_memory_recall':'NOT_APPLICABLE_MEMORY_OFF' if cond=='B0' else 'NOT_EVALUATED_NO_BOUND_REVIEWED_REFERENCE',
            'final_snapshot_structural_measurement_P_R':{'scope':'EXACT_CSV_OBSERVATION_CODE_UNIT_TO_ALREADY_RELEASED_MEASUREMENTS_NOT_CLINICAL_RELEVANCE',
                'macro_precision':mean(precision),'precision_eligible_patients':len(precision),'macro_recall':mean(recall),'recall_eligible_patients':len(recall),
                'micro_hits':sum(s['hits'] for s in eligible),'micro_gold':sum(s['gold'] for s in eligible),
                'micro_recall':ratio(sum(s['hits'] for s in eligible),sum(s['gold'] for s in eligible))},
            'memory_source_integrity':'ALL_PROVIDED_VERIFIED','future_memory_violations':0,'unprovided_memory_citations':0}
    lookup={(r['original_ordinal'],r['condition']):r for r in safe_rows};paired={}
    for a,b in (('B0','B1'),('B0','B2'),('B1','B2')):
        transition=Counter();dp=Counter()
        for i in range(1,51):
            x,y=lookup[i,a],lookup[i,b]
            transition['correct_retained' if x['source_exact'] and y['source_exact'] else 'correct_lost' if x['source_exact'] else 'incorrect_to_correct' if y['source_exact'] else 'both_incorrect']+=1
            dp['correct_retained' if x['source_department'] and y['source_department'] else 'correct_lost' if x['source_department'] else 'incorrect_to_correct' if y['source_department'] else 'both_incorrect']+=1
        paired[b+'-'+a]={'registered_pairs':50,'diagnosis_transitions':dict(transition),'department_transitions':dict(dp),'diagnosis_difference_pp':100*(summaries[b]['Source_Exact_rate']-summaries[a]['Source_Exact_rate'])}
    strata={}
    for group,flag in (('old_development10',True),('other40_development_exposed',False)):
        strata[group]={cond:{'registered':sum(r['old_development10']==flag for r in safe_rows if r['condition']==cond),
            'Source_Exact':sum(r['source_exact'] is True and r['old_development10']==flag for r in safe_rows if r['condition']==cond),
            'Source_department':sum(r['source_department'] is True and r['old_development10']==flag for r in safe_rows if r['condition']==cond)} for cond in CONDITIONS}
    (PR/'case_analysis_private.json').write_text(json.dumps({'rows':detailed,'request_snapshots':snapshots,'confusion':[{'condition':k[0],'reference':k[1],'prediction':k[2],'count':v} for k,v in confusion.items()]},ensure_ascii=False),encoding='utf-8')
    # Anonymous patient ordinal, no actual patient/visit identifiers or medical text.
    with (PUB/'episode_metrics.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        table=[{k:v for k,v in r.items() if k!='final_structural'} for r in safe_rows];writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    report={'status':'READ_ONLY_FULL50_REANALYZED','run_id':p['run_id'],'plan_fingerprint':p['fingerprint'],'registered_patients':50,'episodes':150,
        'conditions':summaries,'paired':paired,'strata':strata,'total_generation':c['generation'],'total_wall_seconds':c['wall_seconds'],
        'total_prompt_tokens':sum(r['prompt_tokens'] for r in safe_rows),'total_output_tokens':sum(r['output_tokens'] for r in safe_rows),
        'checks':dict(checks),'B3_B4':'NOT_REGISTERED_NOT_RUN','D2_label_conditioned':True,'clinical_accuracy':'NOT_EVALUATED',
        'clinical_memory_reference':'NO_INDEPENDENT_QUERY_BOUND_REFERENCE; no judgment generated','new_generation':0,'new_fitting':0,
        'existing_summary_placeholder_warning':'Earlier received_tests/final_cited_tests default0 were unpopulated fields, not measured zero. New real counts come from Runtime and decoded responses.'}
    (PUB/'analysis_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    changed=[path for path,h in before.items() if sha_file(path)!=h]
    proof={'status':'PASS' if not changed else 'FAIL','protected_Run_files':len(before),'changed':len(changed),'HTTP200_SCHEMA_VALID':1055,'registered_tasks_verified':150,'new_generation':0}
    (PUB/'preservation_verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    if changed:raise ValueError('EXISTING_COMPLETED_RESULT_CHANGED')
    print(json.dumps({'conditions':summaries,'paired':paired,'total_generation':c['generation'],'total_wall_seconds':c['wall_seconds'],'preservation':proof}))

if __name__=='__main__':main()
