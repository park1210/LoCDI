"""Read-only C4-05 artifact verification; creates new audit files only."""
import argparse
from collections import Counter,defaultdict
import json
import re
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.source_enrichment import pointer,validate_candidate,make_pool
from app.locdi.memory_reference_evaluation import load_bundle,evaluate
from scripts.enrich_full_cohort_sources import DATA,write,partial_evaluate


def leaves(value,path):
    if isinstance(value,dict):
        for key,child in value.items():yield from leaves(child,path+'/'+key.replace('~','~0').replace('/','~1'))
    elif isinstance(value,list):
        for i,child in enumerate(value):yield from leaves(child,path+'/'+str(i))
    elif value is not None and value!='':yield path


def verify(namespace,stage='initial'):
    public=ROOT/'outputs/research_evaluation'/namespace;private=ROOT/'outputs/private/research_evaluation'/namespace
    original=load_cases(DATA);assert len(original)==50
    raw_pins=load(private/'all_source_hashes.json')
    for name,h in raw_pins.items():
        if sha(name)!=h:raise ValueError('SOURCE_HASH_MISMATCH')
    counts=Counter();unique_fields=defaultdict(set);value_fields=defaultdict(set);schema_fields=defaultdict(set);unique_ids=set();roles=Counter();direct_available=0
    files=['source_facts.jsonl']+(['historical_context_source_facts.jsonl'] if stage=='final' else [])
    for filename in files:
        for line in (private/filename).open(encoding='utf8'):
            r=json.loads(line);key=(r['dataset'],r['case_ref']);counts[r['dataset']]+=1;roles[r['projection_role']]+=1
            assert r['source_file_sha256']==raw_pins[r['source_file']]
            assert digest(r['source_value'])==r['source_value_fingerprint']
            assert digest([r['source_file'],r['source_file_sha256'],r['source_field'],digest(r['source_value'])])==r['provenance_fingerprint']
            for path in leaves(r['source_value'],r['source_field']):
                unique_fields[key].add(path)
                schema_fields[r['dataset']].add(r['source_resource_type']+re.sub(r'/[0-9]+(?=/|$)','/[]',path[len(r['source_field']):]))
                if path.endswith('/value'):value_fields[key].add(path)
            if r['source_record_id']:unique_ids.add((r['dataset'],r['source_file'],r['source_record_id']))
            direct_available+=r['available_at'] is not None
    field_counts={'scope':'Unique populated scalar JSON-pointer positions per case visit, including source metadata; not newly invented facts or clinically useful facts',
        'by_dataset':{ds:{'resource_occurrences':counts[ds],
            'unique_populated_field_positions':sum(len(v) for (d,_),v in unique_fields.items() if d==ds),
            'unique_value_field_positions':sum(len(v) for (d,_),v in value_fields.items() if d==ds),
            'source_schema_field_signatures':len(schema_fields[ds])} for ds in counts},
        'unique_resource_ids':len(unique_ids),'explicit_available_at_records':direct_available,'private_projection_roles':dict(roles),
        'legacy_link_counter_limit':'Build-time outside-legacy counters test UUID string presence in the recorded provenance only; not proof that a fact was unknown throughout the repository'}
    suffix='_final' if stage=='final' else ''
    write(public/('reports/distinct_source_field_counts'+suffix+'.json'),field_counts)
    memories={r['memory_id']:r for c in original for v in c['agent_visible']['past_visits'] for r in v['records']}
    seen=set();dataset_sha=sha(DATA)
    for line in (private/'original_memory_links.jsonl').open(encoding='utf8'):
        r=json.loads(line);source=pointer(original,r['original_record_pointer'])
        assert r['dataset_sha256']==dataset_sha
        assert source==memories[r['memory_id']] and digest(source)==r['record_fingerprint']
        if r['memory_id'] in seen:raise ValueError('DUPLICATE_ORIGINAL_MEMORY_LINK')
        seen.add(r['memory_id'])
    assert len(seen)==len(memories)==72590
    candidate_count=0
    for line in (private/'query_candidates.jsonl').open(encoding='utf8'):
        c=json.loads(line);ordinal=int(c['case_ref'][-3:]);source=original[ordinal-1]
        validate_candidate(c,source['environment']['current_encounter_events'],source['agent_visible']['longitudinal']);candidate_count+=1
    assert candidate_count==250
    full_m,full_packets=load_bundle(private/'full_pool_ask_v1')
    assert len(full_packets)==50 and sum(len(p['eligible_memory']) for p in full_packets)==72590
    parent=ROOT/'outputs/private/research_evaluation/c4_02_v1/review_packets'
    old=load(parent/'manifest.json');by_id={p['snapshot_id']:p for p in full_packets}
    assert all(load(parent/e['file'])['query_fingerprint']==by_id[e['snapshot_id']]['query_fingerprint'] for e in old['packets'])
    pool_m=load(private/'judged_pool_v1/manifest.json');ledger=load(private/'pool_method_ledger.json');pool_total=0
    for entry,record,c in zip(pool_m['packets'],ledger,original):
        packet=load(private/'judged_pool_v1'/entry['file']);candidate=packet['query_candidate']
        records=[r for v in c['agent_visible']['past_visits'] for r in v['records']]
        values,methods,counts_again=make_pool(records,c['agent_visible']['longitudinal'],candidate['query'],candidate['source_facts'],int(packet['snapshot_id'][-3:]))
        assert methods==record['methods'] and counts_again==record['counts']
        assert [r['memory_id'] for r in values]==[r['memory_id'] for r in packet['candidate_memories']]
        assert not any('retrieval_score' in r or 'status_as_of_visit' in r for r in packet['candidate_memories'])
        pool_total+=len(values)
    new_struct=load(public/'reports/structural_150_replay.json');old_struct=load(ROOT/'outputs/research_evaluation/c4_01_v1/reports/memory_benchmark.json')
    structural_equal=digest(new_struct)==digest(old_struct)
    assert structural_equal
    checks={}
    with patch('app.core.longitudinal_memory.PatientMemoryStore.retrieve') as retrieve:
        try:evaluate(private/'full_pool_ask_v1',None,None)
        except ValueError as e:checks['missing_full_reference']=str(e)
        ref=private/'full_pool_ask_v1/review_results_blank.json'
        try:evaluate(private/'full_pool_ask_v1',ref,sha(ref))
        except ValueError as e:checks['unreviewed_full_reference']=str(e)
        ref=private/'judged_pool_v1/review_results_blank.json'
        try:partial_evaluate(private/'judged_pool_v1',ref,sha(ref),public/'UNUSED_QUALITY_OUTPUT')
        except ValueError as e:checks['unreviewed_partial_reference']=str(e)
        assert retrieve.call_count==0
    assert checks=={'missing_full_reference':'REVIEWED_REFERENCE_REQUIRED','unreviewed_full_reference':'INDEPENDENT_COMPLETE_REVIEW_REQUIRED',
                    'unreviewed_partial_reference':'INDEPENDENT_PARTIAL_REVIEW_REQUIRED'}
    result={'status':'PASS_STRUCTURAL_AND_REFERENCE_BOUNDARY_ONLY','original_patients':50,'original_memory_verified':72590,
        'source_asset_hashes_verified':len(raw_pins),'source_fact_rows_verified':sum(counts.values()),'source_query_candidates_reproduced':candidate_count,
        'full_pool_cases':len(full_packets),'judged_pool_cases':len(pool_m['packets']),'judged_pool_records':pool_total,
        'pool_replay_exact_cases':50,'old_150_structural_result_identical':structural_equal,'parent15_ask_queries_unchanged':True,
        'reference_gate_checks':checks,'quality_evaluation_retrievals':0,'clinical_quality':'NOT_RUN','actual_generations':0}
    write(public/('reports/artifact_verification'+suffix+'.json'),result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--namespace',default='c4_05_v1');parser.add_argument('--audit-stage',choices=['initial','final'],default='final');args=parser.parse_args()
    if not args.namespace or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.namespace):raise ValueError('INVALID_NAMESPACE')
    def guard(event,values):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('C4_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    print(json.dumps(verify(args.namespace,args.audit_stage),ensure_ascii=True))
