"""C4-21 freeze/run/seal of saved Source concept metrics. No retrieval/model calls."""
import argparse
from collections import Counter
import csv
from itertools import groupby
import json
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.locdi.memory_concept_evaluation import (VERSION,MODES,concept_key,reference_key,
    validate_reference,evaluate_query,public_query,aggregate,ratio)
from app.locdi.llm_setup import digest
from app.locdi.source_enrichment import validate_candidate,pointer
from scripts.audit_research_gaps import read,sha,write,check,locations
from scripts.evaluate_measurement_identity import load_packets,metrics as original_metrics

BASE=ROOT/'outputs/private/research_evaluation/c4_10_v1/csv_export_aligned'
PB=ROOT/'outputs/research_evaluation/c4_10_v1/csv_export_aligned'
CODE=['app/locdi/memory_concept_evaluation.py','scripts/evaluate_memory_concepts.py',
      'tests/locdi/test_memory_concept_evaluation.py']
# Resolve the actual existing Korean directory without replacing any old report.
REPORT=next((ROOT/'docs').glob('*/C4_20_RESEARCH_GAPS_AND_REMEDIATION_PLAN.md')).parent.relative_to(ROOT).as_posix()+'/C4_21_MEMORY_CONCEPT_LEVEL_EVALUATION_REPORT.md'


def jsonlines(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():yield json.loads(line)


def offline(*args,**kwargs):raise RuntimeError('C4_21_NETWORK_DISABLED')


def freeze(ns):
    pub,private=locations(ns)
    if pub.exists() or private.exists():raise ValueError('OUTPUT_EXISTS_NO_RESUME')
    parent=ROOT/'outputs/private/research_evaluation/c4_20_v1'
    pins=read(parent/'protected_hashes.json');sources=read(parent/'source_hashes.json')
    mp=ROOT/'outputs/research_evaluation/c4_20_v1/output_manifest.json';m=read(mp)
    assert m['fingerprint']==digest({k:v for k,v in m.items() if k!='fingerprint'})
    ip=ROOT/m['private_index_path'];assert sha(ip)==m['private_index_sha256']
    for name,h in {**m['files'],**read(ip)['files']}.items():
        if name in pins:assert pins[name]==h
        pins[name]=h
    pins[mp.relative_to(ROOT).as_posix()]=sha(mp);pins[ip.relative_to(ROOT).as_posix()]=sha(ip)
    a,b=check(pins),check(sources)
    assert a['status']==b['status']=='PASS'
    write(private/'protected_hashes.json',pins);write(private/'source_hashes.json',sources)
    write(private/'initial_preservation.json',{'protected':a,'sources':b})
    input_files=[ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl',
        BASE/'verified_identity_reference.jsonl',BASE/'query_source_alignment.jsonl',BASE/'retrieval_trace.json',
        PB/'evaluation_task_manifest.json',PB/'retrieval_evaluation.json',
        ROOT/'outputs/research_evaluation/c4_10_v1/output_manifest.json',
        ROOT/'outputs/research_evaluation/c4_19_v1/consolidated_results_table.csv',mp]
    input_pins={p.relative_to(ROOT).as_posix():sha(p) for p in input_files}
    for name,h in input_pins.items():assert pins[name]==h
    write(private/'input_hashes.json',input_pins)
    codepins={n:sha(ROOT/n) for n in CODE};write(private/'code_hashes.json',codepins)
    protocol={'version':VERSION,'population':'Original50 all50 registered queries/all72590 records',
        'reference':'Final C4-10 csv_export_aligned only; never mix intermediate labels',
        'concept_key':['canonical_source_system_uri','exact_source_code','exact_recorded_unit','verified_native_resource_type'],
        'observation_type':'NativeSyntheaObservation, verified by native /record/encounters/.../observations location; not the native type field or a clinical datatype',
        'normalization':'Only existing C4-10 LOINC->http://loinc.org mapping; code/unit strings otherwise exact; no case/space folding, synonym or unit conversion',
        'requested_concepts':'Exact five released query-source alignments retained independently of past-memory availability',
        'gold_verified_concepts':'Patient/query-scoped distinct concepts of existing POSITIVE memory records; must be a subset of public query concepts; Source identity only, not clinical Gold',
        'complete_eligibility':'Primary concept Hit/Coverage only if query mapping complete, no original UNKNOWN, all POSITIVE records concept-mapped and positive concept set nonempty',
        'partial_queries':'Known-positive hit and known coverage descriptive only; never full recall; all50 rows remain',
        'positive0':'Gold concept set empty => primary hit/coverage/record recall null, NOT_APPLICABLE; no denominator1',
        'original_unknown':'Preserve104 UNKNOWN labels; any additional unverified concept field is separate concept mapping missingness, never relabel original records',
        'concept_hit':'Binary any(gold intersection retrieved), numerator/denominator over complete-positive eligible queries',
        'concept_coverage':'Unique intersection / unique recoverable gold within query; micro=sum intersections/sum gold; macro=mean defined query rates',
        'unique_concepts':'All exact verified returned concepts incl non-target observations; positive-only counts separately',
        'duplicate_ratio':'(mapped returned records - distinct mapped returned concepts) / all returned records; unmapped/non-observation separately reported; 0 returns=>null',
        'record_baseline':'Reproduce original metrics on each of150 saved trace rows; no retrieval calls or tuning',
        'record_ceiling':'For complete-positive40 only: sum min(5,positive_records) /sum positive_records',
        'ceiling_normalized':'Selective positive records /sum min(5,positive_records) over same eligible40; diagnostic only',
        'modes':list(MODES),'selective_budget':5,'full_history_budget':'All existing eligible records; not equal budget',
        'cutoff':'Existing identity.started_at; Query minute45 unchanged; actual clinical release unknown recorded-time proxy',
        'statistics':'Descriptive patient macro/record or query-concept micro; no significance/generalization claims',
        'failure_categories':['VERIFIED_CONCEPT_RETRIEVAL_MISS','RECORD_CAPACITY_LIMITATION','UNKNOWN_NAMESPACE',
            'AMBIGUOUS_NATIVE_SOURCE_JOIN','DUPLICATE_CONCEPT_SATURATION','SOURCE_TIME_CUTOFF_UNCERTAINTY','MISSING_POSITIVE_REFERENCE'],
        'source_hashes_fingerprint':digest(sources),'input_hashes':input_pins,'code_hashes':codepins,
        'retrieval_algorithm_changes':0,'generation':0,'fitting':0,'new_clinical_labels':0}
    protocol['fingerprint']=digest(protocol);write(pub/'concept_protocol.json',protocol)
    print(json.dumps({'freeze':'PASS','protected_files':len(pins),'source_files':len(sources),'protocol_fingerprint':protocol['fingerprint']}),flush=True)


def native_check(alignment,document,expected_encounter=None):
    n=alignment.get('native')
    if not n:return False
    actual=pointer(document,n['pointer'])
    encounter=pointer(document,n['pointer'].split('/observations/')[0])
    if encounter['start']!=n['encounter_start'] or (expected_encounter is not None and encounter['uuid']!=expected_encounter):
        raise ValueError('NATIVE_ENCOUNTER_BINDING_MISMATCH')
    if (actual.get('value')!=n['native_value'] or actual.get('start')!=n['start']
        or (actual.get('unit') or '')!=n['unit'] or actual.get('codes')!=n['all_codings']
        or actual.get('type')!=n['native_type']):raise ValueError('NATIVE_SOURCE_POINTER_VALUE_MISMATCH')
    return True


def run(ns):
    pub,private=locations(ns)
    if (private/'execution_started.json').exists():raise ValueError('NO_RESUME_OR_DUPLICATE_EXECUTION')
    protocol=read(pub/'concept_protocol.json')
    assert protocol['fingerprint']==digest({k:v for k,v in protocol.items() if k!='fingerprint'})
    assert check(read(private/'code_hashes.json'))['status']=='PASS'
    assert check(read(private/'input_hashes.json'))['status']=='PASS'
    write(private/'execution_started.json',{'protocol_fingerprint':protocol['fingerprint'],
        'code_hashes':read(private/'code_hashes.json'),'generation':0,'retrieval_calls':0})
    cases=list(jsonlines(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'));packets=load_packets()
    queries={r['snapshot_id']:r for r in jsonlines(BASE/'query_source_alignment.jsonl')}
    traces=read(BASE/'retrieval_trace.json');trace={(r['snapshot_id'],r['mode']):r for r in traces}
    old=read(PB/'retrieval_evaluation.json');oldcases={r['snapshot_id']:r for r in old['cases']}
    assert len(cases)==len(packets)==len(queries)==50 and len(trace)==len(traces)==150
    grouped=iter(groupby(jsonlines(BASE/'verified_identity_reference.jsonl'),key=lambda r:r['snapshot_id']))
    totals=Counter();reasons=Counter();mapping=Counter();allresults=[];patient_public=[];patient_private=[]
    native_links=0;source_checks=0;labels_checked=0;baseline_rows=0
    mapping_file=(private/'concept_mapping.jsonl').open('x',encoding='utf-8')
    try:
        for ordinal,(case,packet) in enumerate(zip(cases,packets),1):
            sid,source_rows=next(grouped);refs=list(source_rows)
            assert sid==packet['snapshot_id'] and case['agent_visible']['longitudinal']==packet['private_identity']
            identity=packet['private_identity'];candidate=packet['query_candidate'];q=queries[sid]
            validate_candidate(candidate,case['environment']['current_encounter_events'],identity)
            assert q['query_fingerprint']==candidate['query_fingerprint']==digest(candidate['query'])
            assert q['complete_source_coding'] is True
            originals={r['memory_id']:r for v in case['agent_visible']['past_visits'] for r in v['records']}
            assert len(originals)==len(refs) and {r['memory_id'] for r in refs}==set(originals)
            nativepath=Path(refs[0]['source_provenance']['native_file'])
            native_sha=sha(nativepath);assert native_sha==q['native_sha256']
            document=read(nativepath);assert document['attributes']['id']==identity['patient_id']
            for a in q['alignments']:assert native_check(a,document,identity['visit_id'])
            for r in refs:
                validate_reference(r,originals[r['memory_id']],identity);labels_checked+=1
                assert Path(r['source_provenance']['native_file'])==nativepath
                assert r['source_provenance']['native_sha256']==native_sha
                native_links+=native_check(r['source_alignment'],document,originals[r['memory_id']]['visit_id'])
                key,status=reference_key(r);mapping[status]+=1;totals[r['label']]+=1
                if r['label']=='UNKNOWN':reasons[r['reason']]+=1
                source_checks+=10
                mapping_file.write(json.dumps({'snapshot_id':sid,'memory_id':r['memory_id'],
                    'original_identity_label':r['label'],'concept_key':key,'mapping_status':status,
                    'reference_evidence_fingerprint':r['evidence_fingerprint'],
                    'clinical_relevance':None,'stale':None,'conflict':None},sort_keys=True)+'\n')
            returned={}
            labels={r['memory_id']:r['label'] for r in refs}
            for mode in MODES:
                row=trace[(sid,mode)]
                assert row['query_fingerprint']==candidate['query_fingerprint'] and row['cutoff']==identity['started_at']
                returned[mode]=row['memory_ids']
                assert original_metrics(labels,returned[mode])==oldcases[sid]['modes'][mode]
                baseline_rows+=1
            result=evaluate_query(refs,q['alignments'],returned,query_complete=q['complete_source_coding'])
            allresults.append(result)
            public=public_query(result);public.update(query_id='Query%03d'%ordinal,patient_ordinal=ordinal)
            patient_public.append(public)
            patient_private.append({'query_id':public['query_id'],'snapshot_id':sid,
                'identity':identity,'query_fingerprint':q['query_fingerprint'],'query_alignments':q['alignments'],
                'reference_result':result,'source_native_sha256':native_sha})
            print('Validated Query%03d/050; records=%d'%(ordinal,len(refs)),flush=True)
        try:next(grouped);raise ValueError('EXTRA_REFERENCE_SNAPSHOT')
        except StopIteration:pass
    finally:mapping_file.close()
    assert totals=={'POSITIVE':3884,'NEGATIVE':68602,'UNKNOWN':104} and labels_checked==72590
    aggregates={m:aggregate(allresults,m) for m in MODES}
    sel=aggregates['selective'];assert sel['returned_records']==250
    assert sel['record_precision_micro']==ratio(243,250) and sel['record_recall_micro']==ratio(198,2242)
    assert sel['record_recall_eligible_queries']==40
    write(pub/'record_baseline_reproduction.json',{'status':'EXACT_MATCH','memory_records':labels_checked,
        'labels':dict(totals),'trace_rows_recomputed':baseline_rows,'selective':{
        'returned':250,'positive_returned':243,'precision':ratio(243,250),'recall':ratio(198,2242),
        'complete_positive_reference_queries':40},'input_reference':'Final csv_export_aligned','retrieval_calls':0})
    write(pub/'concept_retrieval_metrics.json',{'scope':'SOURCE_IDENTITY_NOT_CLINICAL_RELEVANCE',
        'primary_selective':sel,'eligibility_distribution':dict(Counter(r['concept_eligibility_status'] for r in allresults)),
        'single_gold_concept_queries':sum(r['gold_unique_concept_count']==1 for r in allresults),
        'multi_gold_concept_queries':sum(r['gold_unique_concept_count']>1 for r in allresults),
        'empty_gold_concept_queries':sum(r['gold_unique_concept_count']==0 for r in allresults)})
    write(pub/'patient_level_metrics.json',{'registered_patients':50,'rows':patient_public})
    write(private/'patient_concept_sets.json',patient_private)
    write(pub/'mode_comparison.json',{'modes':aggregates,'same_query_cutoff_source':True,
        'equal_return_budget':False,'off_control':'RETRIEVAL_DISABLED','existing_equal_budget_baseline':'NOT_FOUND_IN_PINNED_THREE_MODE_TRACES',
        'no_new_algorithm_or_top_k_tuning':True})
    eligible=[r for r in allresults if r['record_recall_eligible']]
    capacity=sum(min(5,r['gold_record_count']) for r in eligible);positives=sum(r['gold_record_count'] for r in eligible)
    got=sum(r['modes']['selective']['verified_relevant_records'] for r in eligible)
    assert (capacity,positives,got)==(200,2242,198)
    write(pub/'record_recall_ceiling.json',{'eligible_queries':len(eligible),'actual_record_recall':ratio(got,positives),
        'arithmetic_top5_capacity_ceiling':ratio(capacity,positives),'ceiling_normalized_recall_diagnostic_only':ratio(got,capacity),
        'ceiling_gap_positive_occurrences':capacity-got,'not_clinical_or_general_retrieval_quality':True})
    write(pub/'concept_identity_inventory.json',{'concept_key':protocol['concept_key'],
        'mapping_status_counts':dict(mapping),'requested_concept_count_distribution':dict(Counter(r['requested_concept_count'] for r in allresults)),
        'gold_concept_count_distribution':dict(Counter(r['gold_unique_concept_count'] for r in allresults)),
        'positive_reference_count_distribution':dict(Counter(r['gold_record_count'] for r in allresults)),
        'selective_unique_concepts_distribution':dict(Counter(r['modes']['selective']['retrieved_verified_concept_count'] for r in allresults)),
        'selective_duplicate_ratio_distribution':dict(Counter(str(r['modes']['selective']['duplicate_concept_ratio']['value']) for r in allresults)),
        'exact_keys_and_source_links':'Private concept_mapping.jsonl and patient_concept_sets.json',
        'original_identity_labels_changed':0,'clinical_labels_created':0})
    fails=Counter(r['modes']['selective']['retrieval_failure_reason'] or 'NO_IDENTIFIED_CONCEPT_MISS' for r in allresults)
    write(pub/'unknown_and_failure_analysis.json',{'unknown_record_reasons':dict(reasons),
        'unknown_queries':sum(r['unknown_count']>0 for r in allresults),
        'zero_positive_queries':sum(r['gold_record_count']==0 for r in allresults),
        'selective_query_outcomes':dict(fails),'verified_concept_total_miss_eligible_queries':sum(
            r['concept_recall_eligible'] and r['modes']['selective']['concept_hit'] is False for r in allresults),
        'partial_concept_miss_eligible_queries':sum(r['concept_recall_eligible'] and r['modes']['selective']['missing_known_positive_concepts']>0 for r in allresults),
        'record_capacity_limited_eligible_queries':sum(r['gold_record_count']>5 for r in eligible),
        'duplicate_saturation_definition':'One verified concept fills all5 selective records; descriptive concentration, not proven harmful relevance',
        'single_concept_top5_saturation_queries':sum(r['modes']['selective']['returned_records']==5 and r['modes']['selective']['verified_concept_records']==5 and r['modes']['selective']['retrieved_verified_concept_count']==1 for r in allresults),
        'source_time_cutoff_uncertainty_queries':50,'structural_cutoff_violations':0,'stale':'NOT_REVIEWED','conflict':'NOT_REVIEWED'})
    write(pub/'source_and_leakage_audit.json',{'reference_fingerprints_checked':labels_checked,
        'original_record_fingerprints_checked':labels_checked,'source_boundary_checks_revalidated':source_checks,
        'native_source_pointers_rechecked':native_links,'native_query_pointers_rechecked':250,
        'query_reconstruction_count':50,'saved_trace_binding_count':150,
        'patient_visit_or_cutoff_violations':0,'new_gold_in_retrieval_input':False,
        'retrieval_execution':'NONE_SAVED_RESULT_METRICS_ONLY','query_source':'Existing released Ask/Test facts validated against original source events',
        'actual_clinical_release':'NOT_RECORDED_PROXY_UNREVIEWED','actual_source_hashes':'Initial/final protection plus percase native SHA',
        'original_unknown_preserved':104,'new_clinical_reference':False})
    write(pub/'execution_result.json',{'status':'C4_21_CONCEPT_EVALUATION_COMPLETE',
        'reference_coverage':'PARTIAL_40_COMPLETE_POSITIVE_9_UNKNOWN_1_POSITIVE0',
        'registered_queries':50,'records':72590,'new_generation':0,'retrieval_calls':0,'fitting':0,
        'protocol_fingerprint':protocol['fingerprint']})
    print(json.dumps({'status':'C4_21_CONCEPT_EVALUATION_COMPLETE','selective':sel}),flush=True)


def seal(ns):
    pub,private=locations(ns)
    a,b=check(read(private/'protected_hashes.json')),check(read(private/'source_hashes.json'))
    assert a['status']==b['status']=='PASS' and check(read(private/'code_hashes.json'))['status']=='PASS'
    write(pub/'verification.json',{'status':'PASS','initial':read(private/'initial_preservation.json'),
        'final':{'protected':a,'sources':b},'code_matches_pre_result_protocol':True,
        'existing_core_data_artifact_changes':0,'generation':0,'fitting':0,'clinical_labels_generated':0})
    pi={'files':{p.relative_to(ROOT).as_posix():sha(p) for p in private.iterdir() if p.is_file()}}
    write(private/'private_index.json',pi)
    files=[p for p in pub.iterdir() if p.is_file()]+[ROOT/n for n in CODE]+[ROOT/REPORT]
    m={'status':'C4_21_CONCEPT_EVALUATION_COMPLETE','files':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'private_index_path':(private/'private_index.json').relative_to(ROOT).as_posix(),
        'private_index_sha256':sha(private/'private_index.json'),'clinical_performance':'NOT_EVALUATED',
        'reference_coverage':'PARTIAL','generation':0,'fitting':0}
    m['fingerprint']=digest(m);write(pub/'output_manifest.json',m)
    print(json.dumps({'seal':'PASS','protected':a['checked'],'sources':b['checked']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run','seal'])
    p.add_argument('--namespace',default='c4_21_v1');a=p.parse_args()
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    try:globals()[a.action](a.namespace)
    except Exception as e:
        pub,private=locations(a.namespace)
        if private.exists() and not (private/'failure.json').exists():
            write(private/'failure.json',{'error_type':type(e).__name__,'stage':a.action,
                'error_message':'Private failure; no raw evidence emitted','generation':0})
        raise
