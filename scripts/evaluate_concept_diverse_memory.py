"""C4-22 isolated development experiment: freeze -> audit -> evaluate -> seal."""
import argparse
from collections import Counter
from itertools import groupby
from pathlib import Path
import json
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check,locations
from scripts.evaluate_memory_concepts import BASE,PB,jsonlines,offline
from scripts.evaluate_measurement_identity import load_packets
from app.locdi.llm_setup import digest
from app.locdi.source_enrichment import validate_candidate
from app.locdi.concept_diverse_memory import policy_spec,select
from app.locdi.concept_diverse_memory_evaluation import project_source_metadata,reconstruct_candidates,candidate_bounds,replacement_counts
from app.locdi.memory_concept_evaluation import validate_reference,evaluate_query,public_query,aggregate,ratio

CODE=['app/locdi/concept_diverse_memory.py','app/locdi/concept_diverse_memory_evaluation.py',
      'scripts/evaluate_concept_diverse_memory.py','tests/locdi/test_concept_diverse_memory.py']
REPORT=next((ROOT/'docs').glob('*/C4_21_MEMORY_CONCEPT_LEVEL_EVALUATION_REPORT.md')).parent/'C4_22_CONCEPT_DIVERSE_MEMORY_REPORT.md'


def freeze(ns):
    pub,pr=locations(ns)
    if pub.exists() or pr.exists():raise ValueError('NAMESPACE_USED_NO_RESUME')
    parent=ROOT/'outputs/private/research_evaluation/c4_21_v1'
    pins=read(parent/'protected_hashes.json');sources=read(parent/'source_hashes.json')
    mp=ROOT/'outputs/research_evaluation/c4_21_v1/output_manifest.json';m=read(mp)
    assert m['fingerprint']==digest({k:v for k,v in m.items() if k!='fingerprint'})
    ip=ROOT/m['private_index_path'];assert sha(ip)==m['private_index_sha256']
    for n,h in {**m['files'],**read(ip)['files']}.items():
        if n in pins:assert pins[n]==h
        pins[n]=h
    for p in [mp,ip]:pins[p.relative_to(ROOT).as_posix()]=sha(p)
    a,b=check(pins),check(sources);assert a['status']==b['status']=='PASS'
    write(pr/'protected_hashes.json',pins);write(pr/'source_hashes.json',sources)
    write(pr/'initial_preservation.json',{'protected':a,'sources':b})
    inputs=[ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl',BASE/'verified_identity_reference.jsonl',
        BASE/'query_source_alignment.jsonl',BASE/'retrieval_trace.json',PB/'retrieval_evaluation.json',
        ROOT/'outputs/research_evaluation/c4_21_v1/patient_level_metrics.json',
        ROOT/'outputs/research_evaluation/c4_21_v1/mode_comparison.json',mp]
    ps={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}
    assert all(pins[n]==h for n,h in ps.items());write(pr/'input_hashes.json',ps)
    codes={n:sha(ROOT/n) for n in CODE};write(pr/'code_hashes.json',codes)
    protocol={'version':'locdi.concept_diverse_memory.evaluation.c4_22.v1','policy':policy_spec(),
        'population':'All Original50 and72590 records; complete-positive40/UNKNOWN9/positive0one retained',
        'evidence_level':'DEVELOPMENT_EXPLORATORY_EVALUATION','new_patient_holdout':False,
        'score_and_boundaries':'Unchanged PatientMemoryStore full_history supplies scores/status on eligible records; nonsuperseded score>0, -score/-available_at/id ordering; independent Core selective5 call must match saved IDs',
        'source_key_provenance':'C4-10 source_alignment/source_type/original record hash ONLY; label/reason/review/gold keys never projected to policy',
        'temporal_key_assumption':'Historical native metadata joined to cutoff-eligible original record; actual clinical metadata release time unrecorded. Development source enrichment, not a live-deployment availability claim',
        'policy_features':['memory_id','score','available_at','concept_key'],
        'unknown':'No forced discard or fabricated unique key; known verified concepts get first pass, all remaining including unknown participate in ranked fill',
        'mapping':'Exact C4-21 system/code/unit/NativeSyntheaObservation; unverified stays missing; no crosswalk or unit conversion',
        'evaluation_reference':'Final C4-10 labels and C4-21 metrics unchanged; concept primary eligible40; partial9 and positive0 separate',
        'metrics':'Reuse C4-21 evaluate_query/aggregate for both selective returns; same5-record budget. Report precision and positive-concept coverage together',
        'ceilings':'Preselection candidate distinct-count bound; Gold-conditioned bound computed evaluation-only after selection; min(5,searchable known Gold)/Gold denominator',
        'tradeoff':'Removed/added label counts, gained/lost positive concepts, new negative concepts, changed slots and dropped return counts; no unobserved correctness',
        'repetition':'One frozen policy; repeat same inputs only for determinism; no tuning or search',
        'runtime_injection':False,'generation':0,'fitting':0,'clinical_metrics':'NOT_EVALUATED',
        'inputs':ps,'code_hashes':codes,'source_pins_fingerprint':digest(sources)}
    protocol['fingerprint']=digest(protocol);write(pub/'concept_diverse_protocol.json',protocol)
    print(json.dumps({'freeze':'PASS','protected':len(pins),'sources':len(sources),'protocol_fingerprint':protocol['fingerprint']}),flush=True)


def guards(pub,pr):
    p=read(pub/'concept_diverse_protocol.json')
    assert p['fingerprint']==digest({k:v for k,v in p.items() if k!='fingerprint'})
    assert check(read(pr/'input_hashes.json'))['status']=='PASS'
    assert check(read(pr/'code_hashes.json'))['status']=='PASS'
    return p


def audit(ns):
    pub,pr=locations(ns);p=guards(pub,pr)
    if (pr/'audit_started.json').exists():raise ValueError('AUDIT_ALREADY_STARTED_NO_RESUME')
    write(pr/'audit_started.json',{'protocol_fingerprint':p['fingerprint'],'policy_calls':0})
    cases=list(jsonlines(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'));packets=load_packets()
    traces={(r['snapshot_id'],r['mode']):r for r in read(BASE/'retrieval_trace.json')}
    groups=iter(groupby(jsonlines(BASE/'verified_identity_reference.jsonl'),key=lambda r:r['snapshot_id']))
    fs=[];mapping=Counter();total=0
    with (pr/'candidate_pools.jsonl').open('x',encoding='utf-8') as out:
        for i,(case,packet) in enumerate(zip(cases,packets),1):
            sid,rs=next(groups);refs=list(rs);assert sid==packet['snapshot_id']
            ident=case['agent_visible']['longitudinal'];q=packet['query_candidate']
            assert ident==packet['private_identity']
            validate_candidate(q,case['environment']['current_encounter_events'],ident)
            records=[r for v in case['agent_visible']['past_visits'] for r in v['records']];byid={r['memory_id']:r for r in records}
            assert len(refs)==len(records) and set(byid)=={r['memory_id'] for r in refs}
            for r in refs:validate_reference(r,byid[r['memory_id']],ident)
            # This projection has no labels/Gold/review content; selector cannot inspect the reference rows.
            metadata=[project_source_metadata(r) for r in refs]
            bundle=reconstruct_candidates(records,ident,q['query'],metadata)
            assert bundle['baseline_ids']==traces[(sid,'selective')]['memory_ids']
            assert bundle['full_ids']==traces[(sid,'full_history')]['memory_ids']
            assert q['query_fingerprint']==traces[(sid,'selective')]['query_fingerprint']
            assert ident['started_at']==traces[(sid,'selective')]['cutoff']
            cs=bundle['candidates'];keys={tuple(c['concept_key']) for c in cs if c['concept_key'] is not None}
            top={tuple(c['concept_key']) for c in cs[:5] if c['concept_key'] is not None}
            f={'query_id':'Query%03d'%i,'patient_ordinal':i,'eligible_memory':bundle['eligible_count'],
                'positive_score_eligible':bundle['positive_score_eligible_count'],'superseded':bundle['superseded_count'],
                'searchable_candidates':len(cs),'eligible_source_key_records':bundle['eligible_verified_concept_records'],
                'eligible_source_concepts':bundle['eligible_verified_concepts'],
                'candidate_source_key_records':sum(c['concept_key'] is not None for c in cs),
                'candidate_concepts':len(keys),'existing_top5_concepts':len(top),
                'additional_candidate_concepts_outside_top5':len(keys-top),
                'candidate_key_missing':sum(c['concept_key'] is None for c in cs),
                'candidate_diversity_count_ceiling':min(5,len(keys)),
                'reason':'NO_VERIFIED_CANDIDATE_KEY_RANKED_FILL_ONLY' if not keys else 'VERIFIED_CANDIDATES_AVAILABLE',
                'source_mapping_availability':'STRUCTURAL_PROXY_ONLY_ACTUAL_RELEASE_UNREVIEWED'}
            fs.append(f);mapping.update(m['status'] for m in bundle['mapping']);total+=len(records)
            out.write(json.dumps({'query_id':f['query_id'],'snapshot_id':sid,'query_fingerprint':q['query_fingerprint'],
                'candidate_input_fingerprint':digest(cs),'candidates':cs,'baseline_ids':bundle['baseline_ids'],
                'full_ids':bundle['full_ids'],'source_metadata_fingerprint':bundle['metadata_fingerprint'],
                'feasibility':f},sort_keys=True)+'\n')
            print('Audited candidate pool %d/50'%i,flush=True)
    assert len(fs)==50 and total==72590
    try:next(groups);raise ValueError('EXTRA_REFERENCE_GROUP')
    except StopIteration:pass
    write(pub/'candidate_feasibility_audit.json',{'status':'FEASIBILITY_PASS_WITH_PARTIAL_SOURCE_MAPPING',
        'registered_queries':50,'original_memory':total,'core_calls':100,'policy_calls':0,
        'baseline_saved_ids_equal':50,'source_mapping_status':dict(mapping),'rows':fs,
        'safe_candidate_reconstruction':True,'actual_clinical_source_key_release':'NOT_VALIDATED',
        'reason_for_concentration':'Unchanged record-overlap ranking has no distinct-concept constraint; repeated high-ranked occurrences can occupy all slots',
        'no_gold_used_in_audit_candidate_construction':True})
    write(pr/'candidate_pool_manifest.json',{'sha256':sha(pr/'candidate_pools.jsonl'),'protocol_fingerprint':p['fingerprint']})
    print(json.dumps({'audit':'PASS','queries':50,'policy_calls':0}),flush=True)


def evaluate(ns,enabled=False):
    pub,pr=locations(ns);p=guards(pub,pr)
    if not enabled:
        print(json.dumps({'status':'DRY_RUN_DEVELOPMENT_OPT_IN_REQUIRED','policy_calls':0}));return
    assert read(pub/'candidate_feasibility_audit.json')['safe_candidate_reconstruction'] is True
    assert sha(pr/'candidate_pools.jsonl')==read(pr/'candidate_pool_manifest.json')['sha256']
    if (pr/'execution_started.json').exists():raise ValueError('NO_RESUME_OR_DUPLICATE_EXECUTION')
    write(pr/'execution_started.json',{'protocol_fingerprint':p['fingerprint'],'opt_in':True,'generation':0})
    pools=list(jsonlines(pr/'candidate_pools.jsonl'))
    groups=iter(groupby(jsonlines(BASE/'verified_identity_reference.jsonl'),key=lambda r:r['snapshot_id']))
    qs={r['snapshot_id']:r for r in jsonlines(BASE/'query_source_alignment.jsonl')}
    oldrows=read(ROOT/'outputs/research_evaluation/c4_21_v1/patient_level_metrics.json')['rows']
    aresults=[];bresults=[];patients=[];bounds=[];trades=[];traces=[];labels_count=Counter()
    with (pr/'selection_trace.jsonl').open('x',encoding='utf-8') as out:
        for i,pool in enumerate(pools,1):
            sid,rs=next(groups);refs=list(rs);assert sid==pool['snapshot_id']
            cs=pool['candidates'];before=digest(cs)
            selected=select(cs,development_opt_in=True);repeated=select(cs,development_opt_in=True)
            assert selected==repeated and before==digest(cs)==pool['candidate_input_fingerprint']
            # Evaluation labels are read into scoring ONLY AFTER the two identical policy calls.
            labels={r['memory_id']:r['label'] for r in refs};labels_count.update(labels.values())
            old=pool['baseline_ids'];new=selected['memory_ids'];q=qs[sid]
            returned={'none':[],'full_history':pool['full_ids'],'selective':old}
            ar=evaluate_query(refs,q['alignments'],returned,query_complete=q['complete_source_coding'])
            expected=public_query(ar);expected.update(query_id=pool['query_id'],patient_ordinal=i)
            assert expected==oldrows[i-1]
            returned['selective']=new;br=evaluate_query(refs,q['alignments'],returned,query_complete=q['complete_source_coding'])
            aresults.append(ar);bresults.append(br)
            # JSON-friendly tuples are needed for mathematical set operations.
            ak=set(map(tuple,ar['modes']['selective']['private_sets']['intersection']))
            bk=set(map(tuple,br['modes']['selective']['private_sets']['intersection']))
            allak=set(map(tuple,ar['modes']['selective']['private_sets']['retrieved']))
            allbk=set(map(tuple,br['modes']['selective']['private_sets']['retrieved']))
            g=set(map(tuple,ar['private_sets']['gold_recoverable']))
            bd=candidate_bounds(cs,g);bd.update(query_id=pool['query_id'],primary_reference_eligible=ar['concept_recall_eligible'])
            bounds.append(bd)
            t=replacement_counts(old,new,labels);t.update({'query_id':pool['query_id'],'patient_ordinal':i,
                'gained_known_positive_concepts':len(bk-ak),'lost_known_positive_concepts':len(ak-bk),
                'new_verified_nonpositive_concepts':len((allbk-allak)-g),
                'returned_records_decreased':len(new)<len(old),
                'precision_delta':br['modes']['selective']['record_precision']['value']-ar['modes']['selective']['record_precision']['value'],
                'primary_coverage_delta':br['modes']['selective']['concept_coverage']['value']-ar['modes']['selective']['concept_coverage']['value'] if ar['concept_recall_eligible'] else None})
            trades.append(t)
            baseline_keys={tuple(c['concept_key']) for c in cs[:5] if c['concept_key'] is not None}
            byid={c['memory_id']:c for c in cs};newkeys={tuple(byid[mid]['concept_key']) for mid in new if byid[mid]['concept_key'] is not None}
            tr={'query_id':pool['query_id'],'patient_ordinal':i,'policy_input_fingerprint':selected['input_fingerprint'],
                'source_metadata_fingerprint':pool['source_metadata_fingerprint'],'policy_fingerprint':selected['policy_fingerprint'],
                'candidate_pool_scope':'ALL_ELIGIBLE_POSITIVE_SCORE_NONSUPERSEDED','candidate_count':len(cs),
                'baseline_returned':len(old),'diverse_returned':len(new),
                'new_records_outside_original_top5':len(set(new)-set(old)),
                'new_candidate_concepts_outside_original_top5':len(newkeys-baseline_keys),
                'selected_ranks':[x['rank'] for x in selected['selection_trace']],
                'selection_passes':[x['pass'] for x in selected['selection_trace']],
                'repeat_equal':True,'input_immutable':True}
            traces.append(tr)
            patients.append({'query_id':pool['query_id'],'patient_ordinal':i,
                'unknown_count':ar['unknown_count'],'gold_record_count':ar['gold_record_count'],
                'gold_unique_concept_count':ar['gold_unique_concept_count'],'eligibility_status':ar['concept_eligibility_status'],
                'baseline':public_query(ar)['modes']['selective'],'concept_diverse':public_query(br)['modes']['selective'],
                'tradeoff':t,'candidate_bounds':bd})
            out.write(json.dumps({'query_id':pool['query_id'],'selection':selected,'baseline_ids':old,
                'baseline_sets':ar['modes']['selective']['private_sets'],
                'diverse_sets':br['modes']['selective']['private_sets'],
                'gold_evaluation_only':ar['private_sets']['gold_recoverable']},sort_keys=True)+'\n')
    assert labels_count=={'POSITIVE':3884,'NEGATIVE':68602,'UNKNOWN':104}
    aa=aggregate(aresults,'selective');bb=aggregate(bresults,'selective')
    for out,rs in [(aa,aresults),(bb,bresults)]:
        out['source_mapping_coverage']=ratio(sum(r['modes']['selective']['verified_concept_records'] for r in rs),out['returned_records'])
        eligible=[r for r in rs if r['record_recall_eligible']]
        out['ceiling_normalized_record_recall']=ratio(sum(r['modes']['selective']['verified_relevant_records'] for r in eligible),sum(min(5,r['gold_record_count']) for r in eligible))
    assert aa['record_precision_micro']==ratio(243,250) and aa['record_recall_micro']==ratio(198,2242) and aa['concept_coverage_micro']==ratio(60,200)
    assert all(len(x['baseline_ids'])==len(x['candidates'][:5]) for x in pools)
    write(pub/'baseline_reproduction.json',{'status':'EXACT_MATCH','core_selective_saved_ids':50,'core_full_saved_ids':50,
        'c4_21_patient_rows_exact':50,'labels':dict(labels_count),'baseline':aa,'baseline_mismatch_count':0})
    write(pub/'same_budget_comparison.json',{'scope':'DEVELOPMENT_EXPLORATORY_EVALUATION','baseline':aa,'concept_diverse':bb,
        'registered_queries':50,'primary_complete_positive_queries':40,'top_k_each':5,
        'changed_selection_queries':sum(t['selection_order_changed'] for t in trades),
        'changed_selection_set_queries':sum(t['selection_set_changed'] for t in trades),
        'returned_count_decreased_queries':sum(t['returned_records_decreased'] for t in trades),
        'clinical_relevance':'NOT_EVALUATED'})
    eligible_bounds=[r for r in bounds if r['primary_reference_eligible']]
    write(pub/'concept_coverage_ceiling.json',{'rows':bounds,'primary_gold_conditioned_bound':ratio(
        sum(r['known_gold_concept_coverage_upper_bound']['numerator'] for r in eligible_bounds),
        sum(r['known_gold_concept_coverage_upper_bound']['denominator'] for r in eligible_bounds)),
        'gold_used_only_in_evaluator_after_policy_selection':True,'not_actual_policy_performance':True})
    write(pub/'precision_coverage_tradeoff.json',{'rows':trades,
        'gained_known_positive_concepts':sum(t['gained_known_positive_concepts'] for t in trades),
        'lost_known_positive_concepts':sum(t['lost_known_positive_concepts'] for t in trades),
        'new_verified_nonpositive_concepts':sum(t['new_verified_nonpositive_concepts'] for t in trades),
        'removed':{l:sum(t['removed'][l] for t in trades) for l in labels_count},
        'added':{l:sum(t['added'][l] for t in trades) for l in labels_count},
        'interpretation':'Counts across all50 exposed queries; primary coverage separately eligible40; nonpositive known concepts are not clinical irrelevance'})
    write(pub/'concept_diverse_selection_trace.json',{'rows':traces,'selection_calls':100,'unique_input_runs':50,
        'repeat_calls':50,'live_runtime_injection':False})
    write(pub/'patient_level_comparison.json',{'registered_patients':50,'rows':patients})
    write(pub/'unknown_and_mapping_audit.json',{'original_reference_labels':dict(labels_count),'labels_modified':0,
        'baseline_unknown_returned':aa['unknown_returned_records'],'diverse_unknown_returned':bb['unknown_returned_records'],
        'baseline_mapping':aa['source_mapping_coverage'],'diverse_mapping':bb['source_mapping_coverage'],
        'unknown_key_handling':policy_spec()['missing_key'],'query_unknown_reference_count':9,'positive0_count':1})
    write(pub/'leakage_and_split_audit.json',{'policy_input_fields':['memory_id','score','available_at','concept_key'],
        'label_free_metadata_projection':True,'gold_passed_to_selector':False,'reference_coverage_passed_to_selector':False,
        'source_key_boundary':'Metadata joined only to cutoff-eligible same-patient original memory fingerprint; prior C4-21 native source binding and unchanged source bytes',
        'actual_clinical_metadata_availability':'NOT_RECORDED_PROXY_UNREVIEWED','baseline_core_calls':100,
        'reference_dataset':'Same frozen final C4-10; original UNKNOWN retained',
        'development_exposure':True,'new_independent_holdout':False,'fit_training_or_new_llm':0,
        'semantic_historical_diagnosis_independence':'NOT_ESTABLISHED_NO_CLINICAL_GENERALIZATION_CLAIM'})
    write(pub/'execution_result.json',{'status':'C4_22_CONCEPT_DIVERSIFICATION_EVALUATED','reference_coverage':'PARTIAL_40_COMPLETE_POSITIVE',
        'generation':0,'fitting':0,'clinical_reference_created':0,'new_core_changes':0})
    print(json.dumps({'status':'C4_22_CONCEPT_DIVERSIFICATION_EVALUATED','baseline':aa,'diverse':bb}),flush=True)


def seal(ns):
    pub,pr=locations(ns);p=guards(pub,pr)
    a,b=check(read(pr/'protected_hashes.json')),check(read(pr/'source_hashes.json'))
    assert a['status']==b['status']=='PASS'
    write(pub/'verification.json',{'status':'PASS','initial':read(pr/'initial_preservation.json'),
        'final':{'protected':a,'sources':b},'new_code_matches_frozen_protocol':True,
        'generation':0,'fitting':0,'clinical_labels_created':0,'original_core_data_artifact_changes':0})
    index={'files':{f.relative_to(ROOT).as_posix():sha(f) for f in pr.iterdir() if f.is_file()}}
    write(pr/'private_index.json',index)
    files=[f for f in pub.iterdir() if f.is_file()]+[ROOT/n for n in CODE]+[REPORT]
    m={'status':'C4_22_CONCEPT_DIVERSIFICATION_EVALUATED','files':{f.relative_to(ROOT).as_posix():sha(f) for f in files},
        'private_index_path':(pr/'private_index.json').relative_to(ROOT).as_posix(),'private_index_sha256':sha(pr/'private_index.json'),
        'scope':'DEVELOPMENT_EXPLORATORY_EVALUATION','generation':0,'fitting':0,'clinical_metrics':'NOT_EVALUATED'}
    m['fingerprint']=digest(m);write(pub/'output_manifest.json',m)
    print(json.dumps({'seal':'PASS','protected':a['checked'],'sources':b['checked']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['freeze','audit','evaluate','seal'])
    ap.add_argument('--namespace',default='c4_22_v1');ap.add_argument('--enable-development-concept-diversity',action='store_true');args=ap.parse_args()
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    if args.action=='evaluate':evaluate(args.namespace,args.enable_development_concept_diversity)
    else:globals()[args.action](args.namespace)
