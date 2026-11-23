"""Candidate reconstruction from unchanged Core; reference-free policy projection."""
from copy import deepcopy
from app.core.longitudinal_memory import PatientMemoryStore
from app.locdi.llm_setup import digest
from app.locdi.memory_concept_evaluation import concept_key,ratio
from app.locdi.concept_diverse_memory import rank_candidates,reject_hidden

META_FIELDS={'memory_id','original_record_fingerprint','source_type','source_alignment'}


def project_source_metadata(reference_row):
    # Deliberately does not read label/reason, query targets, review fields or reference fingerprint.
    return {k:deepcopy(reference_row[k]) for k in META_FIELDS}


def reconstruct_candidates(records,identity,query,source_metadata):
    """Different from reranking already selected5: reads the entire eligible history.

    FullHistory provides original Core scores/status; same filters/order are applied
    without truncation. Actual selective5 call independently verifies the reconstruction.
    """
    reject_hidden(source_metadata)
    if len({m['memory_id'] for m in source_metadata})!=len(source_metadata):raise ValueError('DUPLICATE_SOURCE_METADATA')
    meta={m['memory_id']:m for m in source_metadata}
    if any(set(m)!=META_FIELDS for m in source_metadata):raise ValueError('SOURCE_METADATA_WHITELIST')
    originals={r['memory_id']:r for r in records}
    store=PatientMemoryStore(records)
    full=store.retrieve(identity,query,mode='full_history',top_k=5,max_chars=None)
    baseline=store.retrieve(identity,query,mode='selective',top_k=5,max_chars=None)
    candidates=[];mapping=[];eligible_keys=set();eligible_mapped=0
    for r in full['records']:
        m=meta.get(r['memory_id'])
        if m is None:raise ValueError('SOURCE_METADATA_MISSING')
        if digest(originals[r['memory_id']])!=m['original_record_fingerprint']:raise ValueError('SOURCE_TO_RECORD_BINDING')
        key,status=concept_key(m['source_alignment'],m['source_type'])
        if key is not None:eligible_keys.add(key);eligible_mapped+=1
        mapping.append({'memory_id':r['memory_id'],'status':status})
        if r['status_as_of_visit']=='superseded' or r['retrieval_score']<=0:continue
        candidates.append({'memory_id':r['memory_id'],'score':r['retrieval_score'],
            'available_at':r['available_at'],'concept_key':list(key) if key else None})
    ranked=rank_candidates(candidates)
    baseline_ids=[r['memory_id'] for r in baseline['records']]
    if [c['memory_id'] for c in ranked[:5]]!=baseline_ids:raise ValueError('CORE_SCORE_FILTER_ORDER_MISMATCH')
    return {'candidates':ranked,'baseline_ids':baseline_ids,'full_ids':[r['memory_id'] for r in full['records']],
        'eligible_count':full['eligible_count'],'positive_score_eligible_count':sum(r['retrieval_score']>0 for r in full['records']),
        'superseded_count':sum(r['status_as_of_visit']=='superseded' for r in full['records']),
        'eligible_verified_concept_records':eligible_mapped,'eligible_verified_concepts':len(eligible_keys),
        'mapping':mapping,'policy_input_fingerprint':digest(ranked),
        'metadata_fingerprint':digest(source_metadata),'core_calls':2}


def candidate_bounds(candidates,gold,*,top_k=5):
    """Evaluation-only bounds. This function is never called by the selector."""
    keys={tuple(c['concept_key']) for c in candidates if c['concept_key'] is not None}
    gold=set(map(tuple,gold))
    known=gold&keys
    return {'verified_candidate_concepts':len(keys),'distinct_concept_count_upper_bound':min(top_k,len(keys)),
        'known_gold_candidate_concepts':len(known),'known_gold_concept_coverage_upper_bound':ratio(min(top_k,len(known)),len(gold)),
        'known_gold_not_searchable_count':len(gold-keys),
        'scope':'STRUCTURAL_CANDIDATE_BOUND_NOT_POLICY_PERFORMANCE; incomplete-reference scope inherited'}


def replacement_counts(baseline,diverse,labels):
    removed=set(baseline)-set(diverse);added=set(diverse)-set(baseline)
    return {'removed':{l:sum(labels[i]==l for i in removed) for l in ('POSITIVE','NEGATIVE','UNKNOWN')},
            'added':{l:sum(labels[i]==l for i in added) for l in ('POSITIVE','NEGATIVE','UNKNOWN')},
            'changed_record_slots':len(added),'selection_set_changed':set(baseline)!=set(diverse),
            'selection_order_changed':baseline!=diverse}
