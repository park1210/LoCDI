"""Source-only C4-05 layer and development replay projection; no clinical labels."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
import random
import re

from app.core.clinical_observation import memory_query
from app.core.longitudinal_memory import PatientMemoryStore, timestamp
from app.locdi.llm_setup import digest
from app.locdi.memory_reference_evaluation import reject_hidden, ratio

VERSION = 'locdi.source_grounded_enrichment.c4_05.v1'
PARTIAL_VERSION = 'locdi.memory.judged_pool.c4_05.v1'
PARTIAL_PROCEDURE = 'locdi.memory.partial_source_relevance.c4_05.v1'
POOL_SEED = 20261107


def pointer(document, path):
    value = document
    for key in path.lstrip('/').split('/'):
        key = key.replace('~1','/').replace('~0','~')
        value = value[int(key)] if isinstance(value,list) else value[key]
    return value


def iso_ms(value):
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=value)).isoformat() if type(value) in (int,float) else None


def scalar_count(value):
    if isinstance(value,dict):return sum(scalar_count(x) for x in value.values())
    if isinstance(value,list):return sum(scalar_count(x) for x in value)
    return int(value is not None and value != '')


def fact_row(case, source_path, source_sha, path, value, resource_type, encounter_pointer):
    recorded = iso_ms(value.get('start')) if isinstance(value,dict) else None
    actual = value.get('available_at') if isinstance(value,dict) else None
    return {'schema_version':VERSION,'dataset':case['dataset'],'case_ref':case['case_ref'],
        'patient_reference':case['patient_id'],'visit_reference':case['visit_id'],
        'source_resource_type':resource_type,'source_record_id':value.get('uuid') if isinstance(value,dict) else None,
        'source_file':str(source_path),'source_file_sha256':source_sha,'source_field':path,
        'source_encounter_pointer':encounter_pointer,'source_value':deepcopy(value),
        'recorded_at':recorded,'available_at':actual,
        'availability_evidence':'EXPLICIT_SOURCE_AVAILABLE_AT' if actual else 'NOT_OBSERVED',
        'observation_status':value.get('status','NOT_RECORDED') if isinstance(value,dict) else 'NOT_RECORDED',
        'projection_role':'PRIVATE_SOURCE_ONLY_NOT_AGENT_INPUT',
        'gold_label_risk':'REFERENCE_RESTRICTED' if resource_type in ('Condition','EncounterReason','AttributeReasonState') else 'REQUIRES_EXPOSURE_REVIEW',
        'clinical_review_status':'UNREVIEWED','source_value_fingerprint':digest(value),
        'provenance_fingerprint':digest([str(source_path),source_sha,path,digest(value)])}


def validate_fact(row, document, actual_sha):
    if actual_sha != row['source_file_sha256']:raise ValueError('SOURCE_HASH_MISMATCH')
    if document['attributes']['id'] != row['patient_reference']:raise ValueError('SOURCE_PATIENT_MISMATCH')
    encounter = pointer(document,row['source_encounter_pointer'])
    if encounter['uuid'] != row['visit_reference']:raise ValueError('SOURCE_VISIT_MISMATCH')
    value = pointer(document,row['source_field'])
    if value != row['source_value'] or digest(value) != row['source_value_fingerprint']:raise ValueError('SOURCE_FACT_FINGERPRINT')
    if digest([row['source_file'],actual_sha,row['source_field'],digest(value)]) != row['provenance_fingerprint']:
        raise ValueError('SOURCE_PROVENANCE_FINGERPRINT')


def clean_event(event, decision_time):
    raw = json.loads(event['evidence']);reject_hidden(raw)
    if event['action']=='Ask':
        if set(raw)-{'facts','kind','replay_bin'}:raise ValueError('ASK_EVIDENCE_FIELDS')
        facts = raw['facts']
    elif event['action']=='Test':
        if set(raw)-{'records','replay_bin'}:raise ValueError('TEST_EVIDENCE_FIELDS')
        facts=[]
        for r in raw['records']:
            if r['event_type']!='observations' or r['recorded_evidence']['kind']!='recorded_observation':
                raise ValueError('CONDITION_OR_NONOBSERVATION_FORBIDDEN')
            facts.append({'code':r['code'],'description':r['description'],'units':r['recorded_evidence'].get('units'),
                          'value':r['recorded_evidence'].get('value'),'recorded_at':r['recorded_at']})
    else:raise ValueError('UNSUPPORTED_RELEASE_TYPE')
    result=[]
    for f in facts:
        if not f.get('code') or not f.get('description') or f.get('value') in (None,''):
            raise ValueError('SOURCE_FACT_VALUE_REQUIRED')
        if timestamp(f['recorded_at'])>timestamp(decision_time):raise ValueError('FUTURE_SOURCE_FACT')
        result.append({k:f.get(k) for k in ('code','description','units','value','recorded_at')})
    return result


def released_projection(observed, events, results, identity):
    """Use only exact observed Ask answers and Scheduler-received Test results."""
    minute = observed['elapsed_minutes']
    decision_time=(timestamp(identity['started_at'])+timedelta(minutes=minute)).isoformat()
    projection={};links=[];facts=[];history=[];findings=[]
    for h in observed.get('history',[]):
        if h.get('answer_status')!='provided':continue
        matches=[e for e in events if e['action']=='Ask' and e['target']==h['question']]
        if len(matches)!=1 or h['answer']!=matches[0]['evidence']:raise ValueError('ASK_SOURCE_VALUE_MISMATCH')
        e=matches[0]
        if not e['available_minute']<=h['observed_minute']<=minute:raise ValueError('ASK_NOT_RELEASED')
        values=clean_event(e,decision_time);facts.extend(values)
        history.append({'question':h['question'],'answer':json.dumps({'facts':values},sort_keys=True,ensure_ascii=False),
                        'answer_status':'provided','observed_minute':h['observed_minute']})
        links.append({'event_id':e['event_id'],'action':'Ask','source_fingerprint':digest(e),
                      'observed_minute':h['observed_minute'],'actual_clinical_available_at':None})
    for finding in observed.get('objective_findings',[]):
        matches=[e for e in events if e['action']=='Test' and finding==e['target']+': '+e['evidence']]
        if len(matches)!=1:raise ValueError('TEST_SOURCE_VALUE_MISMATCH')
        e=matches[0]
        received=[r for r in results if r['test_name']==e['target'] and r['observed_minute']<=minute]
        if len(received)!=1 or received[0]['observed_minute']<e['available_minute']:raise ValueError('TEST_NOT_RECEIVED')
        values=clean_event(e,decision_time);facts.extend(values)
        findings.append({'test':e['target'],'facts':values,'observed_minute':received[0]['observed_minute']})
        links.append({'event_id':e['event_id'],'action':'Test','source_fingerprint':digest(e),
                      'observed_minute':received[0]['observed_minute'],'actual_clinical_available_at':None})
    if history:projection['history']=history
    if findings:projection['objective_findings']=findings
    reject_hidden(projection)
    if not links:raise ValueError('NO_RELEASED_OBSERVATION')
    return {'projection_version':VERSION,'decision_minute':minute,'decision_time':decision_time,
        'projection':projection,'query':memory_query(projection),'query_fingerprint':digest(memory_query(projection)),
        'source_links':links,'source_facts':facts,'missing_information':['CHIEF_COMPLAINT','SYMPTOM_NARRATIVE','OBSERVED_CLINICAL_RELEASE_TIME'],
        'availability':'OBSERVED_OFFLINE_REPLAY_WITH_DECLARED_RECORDED_TIME_ASSUMPTION',
        'clinical_suitability':'UNREVIEWED','purpose':'DEVELOPMENT_ONLY'}


def validate_candidate(candidate, events, identity):
    history=[];findings=[];results=[];seen=set()
    for link in candidate['source_links']:
        if link['event_id'] in seen:raise ValueError('DUPLICATE_RELEASE_LINK')
        seen.add(link['event_id'])
        matches=[e for e in events if e['event_id']==link['event_id']]
        if len(matches)!=1 or digest(matches[0])!=link['source_fingerprint']:raise ValueError('RELEASE_SOURCE_FINGERPRINT')
        e=matches[0]
        if e['action']!=link['action']:raise ValueError('RELEASE_ACTION_MISMATCH')
        if e['action']=='Ask':history.append({'question':e['target'],'answer':e['evidence'],'answer_status':'provided','observed_minute':link['observed_minute']})
        else:
            findings.append(e['target']+': '+e['evidence']);results.append({'test_name':e['target'],'observed_minute':link['observed_minute']})
    reconstructed=released_projection({'elapsed_minutes':candidate['decision_minute'],'history':history,'objective_findings':findings},events,results,identity)
    if any(candidate.get(k)!=v for k,v in reconstructed.items()):raise ValueError('QUERY_CANDIDATE_REPRODUCTION')


def make_pool(records, identity, query, facts, ordinal, per_method=5):
    """Review candidates only; full underlying pool and production algorithms stay fixed."""
    store=PatientMemoryStore(records)
    full=store.retrieve(identity,query,'full_history')['records']
    lexical=store.retrieve(identity,query,'selective',5)['records']
    tokens=lambda text:set(re.findall(r'\w+',text.casefold()))
    q=tokens(query)
    def jaccard(r):
        t=tokens(r['text']);return len(q&t)/len(q|t) if q|t else 0.
    other=sorted(full,key=lambda r:(-jaccard(r),r['memory_id']))
    recency=sorted(full,key=lambda r:(-timestamp(r['available_at']).timestamp(),r['memory_id']))
    sample=sorted(full,key=lambda r:r['memory_id']);random.Random(POOL_SEED+ordinal).shuffle(sample)
    codes={str(f['code']).casefold() for f in facts}
    descriptions={f['description'].casefold() for f in facts}
    source=[r for r in full if any(re.search(r'(?<!\w)'+re.escape(c)+r'(?!\w)',r['text'].casefold()) for c in codes)
            or any(d in r['text'].casefold() for d in descriptions)]
    methods={'existing_lexical_top5':lexical,'fixed_token_jaccard':other[:per_method],
        'recency':recency[:per_method],'seeded_nonsearch_sample':sample[:per_method],
        'source_code_or_description_text_match':source[:per_method]}
    members={r['memory_id'] for values in methods.values() for r in values}
    # Chronological review order hides ranking/method membership from reviewers.
    selected=[r for r in full if r['memory_id'] in members]
    ledger={k:[r['memory_id'] for r in v] for k,v in methods.items()}
    return selected,ledger,{'eligible_count':len(full),'pooled_count':len(selected),
        'unpooled_count':len(full)-len(selected),'unpooled_fraction':(len(full)-len(selected))/len(full) if full else None,
        'unjudged_count':len(full),'seed':POOL_SEED+ordinal,'method_counts':{k:len(v) for k,v in ledger.items()}}


def judged_metrics(selected_ids, pool_ids, judgments):
    """Partial metrics only. Does not imply review authorization or full-pool recall."""
    selected=set(selected_ids);pool=set(pool_ids)
    if len(judgments)!=len(pool) or {r['memory_id'] for r in judgments}!=pool:raise ValueError('JUDGED_POOL_COVERAGE')
    if any(type(r.get('relevant')) not in (bool,type(None)) or type(r.get('stale')) not in (bool,type(None)) for r in judgments):
        raise ValueError('INVALID_PARTIAL_JUDGMENT')
    judged={r['memory_id'] for r in judgments if type(r.get('relevant')) is bool}
    relevant={r['memory_id'] for r in judgments if r.get('relevant') is True}
    stale_judged={r['memory_id'] for r in judgments if type(r.get('stale')) is bool}
    stale={r['memory_id'] for r in judgments if r.get('stale') is True}
    return {'judged_set_precision':ratio(len(selected&judged&relevant),len(selected&judged)),
        'judged_coverage':ratio(len(selected&judged),len(selected)),
        'known_relevant_retrieval_count':len(selected&relevant),
        'judged_stale_rate':ratio(len(selected&stale),len(selected&stale_judged)),
        'unjudged_returned_count':len(selected-judged),'full_pool_recall':'NOT_IDENTIFIABLE_NOT_COMPUTED'}


def validate_partial_review(packet, review):
    if (review.get('schema_version')!=PARTIAL_VERSION or review.get('packet_fingerprint')!=digest(packet)
        or review.get('query_fingerprint')!=packet['query_candidate']['query_fingerprint']):raise ValueError('PARTIAL_REFERENCE_BINDING')
    if (review.get('review_status')!='REVIEWED' or review.get('reviewer_independent') is not True
        or not isinstance(review.get('reviewer_private_ref'),str) or not review['reviewer_private_ref'].strip()
        or review.get('procedure_version')!=PARTIAL_PROCEDURE
        or review.get('query_suitable') is not True or review.get('visibility_reviewed') is not True
        or review.get('recorded_time_proxy_reviewed') is not True or review.get('methods_or_predictions_seen') is not False):
        raise ValueError('INDEPENDENT_PARTIAL_REVIEW_REQUIRED')
    when=timestamp(review['reviewed_at'])
    if when>datetime.now(timezone.utc) or when<timestamp(packet['query_candidate']['decision_time']):raise ValueError('REVIEW_TIME_INVALID')
    if not isinstance(review.get('rationale'),str) or not review['rationale'].strip():raise ValueError('REVIEW_RATIONALE_REQUIRED')
    rows=review.get('records',[])
    if [r.get('memory_id') for r in rows]!=[r['memory_id'] for r in packet['candidate_memories']]:raise ValueError('PARTIAL_RECORD_ORDER')
    for r in rows:
        if not isinstance(r.get('rationale'),str) or not r['rationale'].strip():
            raise ValueError('PARTIAL_RECORD_NOT_REVIEWED')
        if any(type(r.get(k)) not in (bool,type(None)) for k in ('relevant','stale','conflict')):
            raise ValueError('INVALID_PARTIAL_JUDGMENT')
        if any(r.get(k) is None for k in ('relevant','stale','conflict')) and not r.get('not_evaluable_reason'):
            raise ValueError('PARTIAL_RECORD_NOT_REVIEWED')
    return rows
