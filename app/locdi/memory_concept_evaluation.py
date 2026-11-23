"""Exact Source identity metrics only. No retrieval, clinical labels or model calls."""
from collections import Counter
from copy import deepcopy
import statistics
from app.locdi.llm_setup import digest
from app.core.longitudinal_memory import timestamp

VERSION = 'locdi.memory_concept_evaluation.c4_21.v1'
MODES = ('none', 'full_history', 'selective')


def validate_reference(row, original, identity):
    if digest({k:v for k,v in row.items() if k!='evidence_fingerprint'}) != row.get('evidence_fingerprint'):
        raise ValueError('REFERENCE_FINGERPRINT_MISMATCH')
    if digest(original) != row['original_record_fingerprint'] or original['memory_id']!=row['memory_id']:
        raise ValueError('ORIGINAL_MEMORY_BINDING_MISMATCH')
    if any(row.get(k) is not None for k in ('clinical_relevance','stale','conflict')):
        raise ValueError('CLINICAL_LABEL_NOT_PERMITTED')
    cutoff=timestamp(identity['started_at'])
    if (original['patient_id']!=identity['patient_id'] or original['visit_id']==identity['visit_id']
            or timestamp(original['visit_started_at'])>=cutoff or timestamp(original['available_at'])>cutoff
            or timestamp(original['observed_at'])>timestamp(original['available_at'])):
        raise ValueError('PATIENT_VISIT_OR_CUTOFF_VIOLATION')
    required={'M01_PATIENT','M02_PREVIOUS_VISIT','M03_CHRONOLOGY','M04_FIXED_MEMORY_CUTOFF',
              'M05_ORIGINAL_RECORD','M06_ORIGINAL_FINGERPRINT','M07_SOURCE_RECORD',
              'M08_SOURCE_FINGERPRINT','M09_SOURCE_TO_MEMORY','M10_SOURCE_TIME_BOUND'}
    checks={c['rule_id']:c['result'] for c in row['checks']}
    if any(checks.get(k)!='PASS' for k in required):
        raise ValueError('EXISTING_SOURCE_CHECK_NOT_PASS')
    return True


def ratio(numerator, denominator, reason='NOT_APPLICABLE_ZERO_DENOMINATOR'):
    return {'numerator':numerator, 'denominator':denominator,
            'value':numerator / denominator if denominator else None,
            'status':'COMPUTED' if denominator else reason}


def concept_key(alignment, source_type='observations'):
    """No lexical normalization or clinical crosswalk; uses C4-10 canonical URI.

    Observation type is the verified native resource location/type, not the native
    `type` value (which can itself be a measurement code, not a datatype).
    """
    if source_type != 'observations':
        return None, 'NOT_APPLICABLE_NON_OBSERVATION'
    if alignment.get('status') != 'VERIFIED_SOURCE_CODING':
        return None, alignment.get('reason', 'UNVERIFIED_SOURCE_CODING')
    native = alignment.get('native') or {}
    fields = (alignment.get('system_uri'), alignment.get('code'), alignment.get('unit'),
              native.get('resource_type'))
    if any(not isinstance(v,str) or not v.strip() for v in fields):
        return None, 'MISSING_EXPLICIT_CONCEPT_FIELD'
    if native.get('resource_type') != 'NativeSyntheaObservation':
        return None, 'UNVERIFIED_OBSERVATION_TYPE'
    if native.get('unit') != alignment['unit']:
        raise ValueError('NATIVE_UNIT_CONTRADICTION')
    if str((native.get('coding') or {}).get('code')) != alignment['code']:
        raise ValueError('NATIVE_CODE_CONTRADICTION')
    if not native.get('pointer','').startswith('/record/encounters/') or '/observations/' not in native['pointer']:
        raise ValueError('NATIVE_OBSERVATION_LOCATION_INVALID')
    return fields, 'VERIFIED_EXACT_SOURCE_CONCEPT'


def reference_key(row):
    if row['label'] == 'UNKNOWN':
        return None, 'ORIGINAL_UNKNOWN_' + row['reason']
    return concept_key(row['source_alignment'], row['source_type'])


def evaluate_query(references, query_alignments, returned_by_mode, *, top_k=5, query_complete=True):
    """Per patient/query; never deduplicates or changes the original references.

    Gold here means recoverable verified-positive Source concepts, not medical Gold.
    Requested source concepts are separately retained, including ones with no past record.
    """
    if type(top_k) is not int or top_k<1:
        raise ValueError('TOP_K_INVALID')
    if set(returned_by_mode) != set(MODES):
        raise ValueError('EXACT_THREE_MODES_REQUIRED')
    labels = {r['memory_id']:r['label'] for r in references}
    if len(labels)!=len(references) or any(v not in {'POSITIVE','NEGATIVE','UNKNOWN'} for v in labels.values()):
        raise ValueError('DUPLICATE_RECORD_OR_INVALID_LABEL')
    if any(r.get(k) is not None for r in references for k in ('clinical_relevance','stale','conflict')):
        raise ValueError('CLINICAL_LABEL_NOT_PERMITTED')
    mapped = {r['memory_id']:reference_key(r) for r in references}
    requested = set()
    for a in query_alignments:
        key, _ = concept_key(a)
        if key is None:
            query_complete = False
        else:
            requested.add(key)
    positive_ids = {mid for mid,v in labels.items() if v=='POSITIVE'}
    missing_positive_mapping = {mid for mid in positive_ids if mapped[mid][0] is None}
    gold = {mapped[mid][0] for mid in positive_ids if mapped[mid][0] is not None}
    if not gold<=requested:
        raise ValueError('POSITIVE_CONCEPT_OUTSIDE_PUBLIC_QUERY')
    unknown = sum(v=='UNKNOWN' for v in labels.values())
    record_eligible = not unknown and bool(positive_ids)
    if not query_complete:
        concept_status='NOT_EVALUATED_QUERY_CONCEPT_MAPPING_INCOMPLETE'
    elif unknown:
        concept_status='NOT_EVALUATED_INCOMPLETE_REFERENCE'
    elif missing_positive_mapping:
        concept_status='NOT_EVALUATED_POSITIVE_CONCEPT_MAPPING_INCOMPLETE'
    elif not gold:
        concept_status='NOT_APPLICABLE_NO_POSITIVE_REFERENCE'
    else:
        concept_status='ELIGIBLE_COMPLETE_POSITIVE_REFERENCE'
    eligible = concept_status=='ELIGIBLE_COMPLETE_POSITIVE_REFERENCE'
    result = {'gold_record_count':len(positive_ids),'gold_unique_concept_count':len(gold),
        'requested_concept_count':len(requested),'unknown_count':unknown,
        'missing_positive_concept_mapping_count':len(missing_positive_mapping),
        'record_recall_eligible':record_eligible,'concept_eligibility_status':concept_status,
        'concept_recall_eligible':eligible,'modes':{},
        'private_sets':{'requested':sorted(requested),'gold_recoverable':sorted(gold)}}
    for mode in MODES:
        ids=returned_by_mode[mode]
        if not isinstance(ids,list) or len(set(ids))!=len(ids) or any(mid not in labels for mid in ids):
            raise ValueError('RETURNED_POOL_BOUNDARY_OR_DUPLICATE_ID')
        if (mode=='none' and ids) or (mode=='selective' and len(ids)>top_k):
            raise ValueError('RETURN_BUDGET_VIOLATION')
        selected_keys=[mapped[mid][0] for mid in ids if mapped[mid][0] is not None]
        retrieved=set(selected_keys)
        hit=gold&retrieved
        positive=sum(mid in positive_ids for mid in ids)
        unjudged=sum(labels[mid]=='UNKNOWN' for mid in ids)
        concept_unmapped=sum(mapped[mid][0] is None and mapped[mid][1]!='NOT_APPLICABLE_NON_OBSERVATION' for mid in ids)
        non_observation=sum(mapped[mid][1]=='NOT_APPLICABLE_NON_OBSERVATION' for mid in ids)
        duplicate=len(selected_keys)-len(retrieved)
        positive_duplicate=positive-len(hit)
        known_hit=bool(hit) if gold else None
        capacity=min(top_k,len(positive_ids))
        value={'returned_records':len(ids),'judged_returned_records':len(ids)-unjudged,
            'verified_relevant_records':positive,'unknown_returned_records':unjudged,
            'verified_concept_records':len(selected_keys),'retrieved_verified_concept_count':len(retrieved),
            'retrieved_positive_concept_count':len(hit),'missing_known_positive_concepts':len(gold-hit),
            'unjudged_concept_record_count':concept_unmapped,'non_observation_records':non_observation,
            'duplicate_concept_count':duplicate,'positive_duplicate_concept_count':positive_duplicate,
            'record_precision':ratio(positive,len(ids)-unjudged),
            'record_recall':ratio(positive,len(positive_ids)) if record_eligible else {
                'numerator':positive,'denominator':len(positive_ids),'value':None,
                'status':'NOT_EVALUATED_INCOMPLETE_REFERENCE' if unknown else 'NOT_APPLICABLE_NO_POSITIVE_REFERENCE'},
            'concept_hit':known_hit if eligible else None,
            'concept_coverage':ratio(len(hit),len(gold)) if eligible else {
                'numerator':len(hit),'denominator':len(gold),'value':None,'status':concept_status},
            'known_positive_concept_hit':known_hit,
            'known_positive_concept_coverage_not_full_recall':ratio(len(hit),len(gold)),
            'duplicate_concept_ratio':ratio(duplicate,len(ids)),
            'verified_mapping_coverage':ratio(len(selected_keys),len(ids)),
            'record_recall_ceiling_at5':ratio(capacity,len(positive_ids)) if record_eligible else {
                'numerator':capacity,'denominator':len(positive_ids),'value':None,'status':concept_status},
            'ceiling_normalized_record_recall':ratio(positive,capacity) if record_eligible and mode=='selective' else {
                'numerator':None,'denominator':None,'value':None,'status':'NOT_APPLICABLE_OR_INELIGIBLE'},
            'retrieval_failure_reason': 'RETRIEVAL_DISABLED_CONTROL' if mode=='none' else
                'VERIFIED_CONCEPT_RETRIEVAL_MISS' if eligible and not hit else
                'PARTIAL_VERIFIED_CONCEPT_MISS' if eligible and gold-hit else
                'INCOMPLETE_REFERENCE_NOT_FULLY_EVALUABLE' if unknown else
                'MISSING_POSITIVE_REFERENCE' if not gold else None,
            'clinical_relevance':'NOT_REVIEWED','stale':'NOT_REVIEWED','conflict':'NOT_REVIEWED'}
        # These sets belong only in Private, including concepts from negative observations.
        value['private_sets']={'retrieved':sorted(retrieved),'intersection':sorted(hit),
            'missing':sorted(gold-hit),'mapped_key_counts':[{'key':list(k),'count':v} for k,v in sorted(Counter(selected_keys).items())]}
        result['modes'][mode]=value
    result['mapping_status_counts']=dict(Counter(v[1] for v in mapped.values()))
    return result


def public_query(result):
    r=deepcopy(result)
    r.pop('private_sets',None)
    for mode in r['modes'].values():mode.pop('private_sets',None)
    return r


def mean_metric(values):
    return {'value':statistics.mean(values) if values else None,'eligible_query_count':len(values),
            'status':'COMPUTED' if values else 'NOT_APPLICABLE_ZERO_DENOMINATOR'}


def aggregate(results, mode):
    """Distinct concepts are counted within each query, never across patients."""
    if mode not in MODES or not results:raise ValueError('REGISTERED_QUERIES_AND_MODE_REQUIRED')
    ms=[r['modes'][mode] for r in results]
    record=[r for r in results if r['record_recall_eligible']]
    concept=[r for r in results if r['concept_recall_eligible']]
    total=lambda field:sum(m[field] for m in ms)
    cnum=sum(r['modes'][mode]['retrieved_positive_concept_count'] for r in concept)
    cden=sum(r['gold_unique_concept_count'] for r in concept)
    rnum=sum(r['modes'][mode]['verified_relevant_records'] for r in record)
    rden=sum(r['gold_record_count'] for r in record)
    hnum=sum(r['modes'][mode]['concept_hit'] is True for r in concept)
    return {'mode':mode,'registered_queries':len(results),'returned_records':total('returned_records'),
        'verified_relevant_records':total('verified_relevant_records'),'unknown_returned_records':total('unknown_returned_records'),
        'unique_concepts_sum_within_queries':total('retrieved_verified_concept_count'),
        'duplicate_records':total('duplicate_concept_count'),
        'positive_duplicate_records':total('positive_duplicate_concept_count'),
        'unjudged_concept_records':total('unjudged_concept_record_count'),
        'non_observation_records':total('non_observation_records'),
        'record_precision_micro':ratio(total('verified_relevant_records'),total('judged_returned_records')),
        'record_recall_micro':ratio(rnum,rden),'record_recall_macro':mean_metric([r['modes'][mode]['record_recall']['value'] for r in record]),
        'record_recall_eligible_queries':len(record),'concept_eligible_queries':len(concept),
        'concept_hit_rate':ratio(hnum,len(concept)), 'concept_coverage_micro':ratio(cnum,cden),
        'concept_coverage_macro':mean_metric([r['modes'][mode]['concept_coverage']['value'] for r in concept]),
        'duplicate_ratio_micro':ratio(total('duplicate_concept_count'),total('returned_records')),
        'duplicate_ratio_macro':mean_metric([m['duplicate_concept_ratio']['value'] for m in ms if m['returned_records']]),
        'returned_query_coverage':ratio(sum(m['returned_records']>0 for m in ms),len(results)),
        'confirmed_known_hit_queries':sum(m['known_positive_concept_hit'] is True for m in ms),
        'confirmed_known_hit_scope':'Count among all registered queries; not full-cohort complete-reference accuracy',
        'retrieval_runtime_failure_count':0, 'runtime_failure_scope':'All registered saved traces exist; source retrieval runtime not rerun',
        'clinical_performance':'NOT_EVALUATED'}
