"""External C4-02 source/review boundary over unchanged Memory algorithms.

No clinical judgments, model calls, fitting or synthetic relevance labels.
"""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from copy import deepcopy
import json,re
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha,load
from app.core.clinical_observation import memory_query
from app.core.longitudinal_memory import PatientMemoryStore,timestamp
from app.core.memory_evaluation import evaluate_memory

VERSION='locdi.memory_independent_review.v1'
PROCEDURE='locdi.memory_source_relevance_review.v1'
FORBIDDEN={'gold','evaluation','task_annotations','provenance','source_event_store','predicted_diagnosis','llm_audit','working_assessment'}

def reject_hidden(value):
    if isinstance(value,dict):
        if set(value)&FORBIDDEN:raise ValueError('HIDDEN_FIELD_NOT_ALLOWED')
        for child in value.values():reject_hidden(child)
    elif isinstance(value,list):
        for child in value:reject_hidden(child)

def released_query(history,*,snapshot_minute,source_event):
    """Use only an actually replay-released Ask answer, never an initial label."""
    reject_hidden(history)
    if type(snapshot_minute) is not int or snapshot_minute<0:raise ValueError('INVALID_DECISION_MINUTE')
    if source_event['action']!='Ask':raise ValueError('ASK_SOURCE_REQUIRED')
    matches=[h for h in history if h.get('question')==source_event['target'] and h.get('answer_status')=='provided']
    if len(matches)!=1:raise ValueError('ONE_RELEASED_ANSWER_REQUIRED')
    h=matches[0];minute=h.get('observed_minute')
    if type(minute) is not int or not source_event['available_minute']<=minute<=snapshot_minute:raise ValueError('ANSWER_NOT_YET_RELEASED')
    if h.get('answer')!=source_event['evidence']:raise ValueError('RELEASED_SOURCE_VALUE_MISMATCH')
    answer=json.loads(h['answer']);reject_hidden(answer)
    if set(answer)-{'facts','kind','replay_bin'}:raise ValueError('SOURCE_ANSWER_FIELDS')
    facts=answer.get('facts')
    if not isinstance(facts,list) or not facts:raise ValueError('SOURCE_FACT_VALUE_REQUIRED')
    for f in facts:
        if set(f)-{'code','description','units','value','recorded_at','onset','duration','active_status'} or any(not f.get(k) for k in ('code','description','recorded_at')) or f.get('value') in (None,''):
            raise ValueError('SOURCE_FACT_FIELDS')
    # memory_query's algorithm is unchanged; only grounded history enters it.
    grounded={'facts':[{k:f.get(k) for k in ('code','description','units','value','recorded_at')} for f in facts]}
    clean={'history':[{'question':h['question'],'answer':json.dumps(grounded,ensure_ascii=False,sort_keys=True),
                      'answer_status':'provided','observed_minute':minute}]}
    return memory_query(clean),clean

def ratio(n,d):return {'numerator':n,'denominator':d,'value':n/d if d else None,'status':'available' if d else 'not_applicable'}

def validate_packet(packet):
    if not re.fullmatch(r'Snapshot[0-9]{3}',packet['snapshot_id']):raise ValueError('ANONYMOUS_SNAPSHOT_ID_REQUIRED')
    identity=packet['private_identity'];cutoff=timestamp(identity['started_at'])
    if timestamp(packet['decision_time'])!=cutoff+timedelta(minutes=packet['decision_minute']):raise ValueError('DECISION_TIME_BINDING')
    records=packet['eligible_memory'];PatientMemoryStore(records)
    if any(r['patient_id']!=identity['patient_id'] or r['visit_id']==identity['visit_id']
        or timestamp(r['visit_started_at'])>=cutoff or timestamp(r['available_at'])>cutoff for r in records):
        raise ValueError('PACKET_MEMORY_BOUNDARY')
    query,_=released_query(packet['released_history'],snapshot_minute=packet['decision_minute'],source_event=packet['released_source_event'])
    if query!=packet['query'] or digest(query)!=packet['query_fingerprint']:raise ValueError('QUERY_PROJECTION_MISMATCH')
    for fact in json.loads(packet['released_source_event']['evidence'])['facts']:
        if timestamp(fact['recorded_at'])>timestamp(packet['decision_time']):raise ValueError('FUTURE_SOURCE_FACT')
    if packet.get('review_status')!='UNREVIEWED' or packet.get('model_prediction_included') is not False or packet.get('retrieval_output_included') is not False:
        raise ValueError('BLINDED_UNREVIEWED_PACKET_REQUIRED')

def _private_file(path):
    p=Path(path).resolve()
    if not p.is_relative_to((ROOT/'outputs/private').resolve()) or p.is_symlink():raise ValueError('PRIVATE_REFERENCE_PATH_REQUIRED')
    return p

def load_bundle(root):
    root=_private_file(root);manifest=load(root/'manifest.json')
    if digest({k:v for k,v in manifest.items() if k!='fingerprint'})!=manifest['fingerprint']:raise ValueError('PACKET_MANIFEST_FINGERPRINT')
    packets=[]
    for item in manifest['packets']:
        name=item['file']
        if Path(name).name!=name or not name.endswith('.json'):raise ValueError('PACKET_PATH_ESCAPE')
        p=root/name
        if sha(p)!=item['sha256']:raise ValueError('PACKET_HASH_MISMATCH')
        packet=load(p)
        if packet['snapshot_id']!=item['snapshot_id'] or digest(packet['query'])!=packet['query_fingerprint']:raise ValueError('PACKET_QUERY_BINDING')
        validate_packet(packet)
        packets.append(packet)
    if not packets or len({p['snapshot_id'] for p in packets})!=len(packets):raise ValueError('FIXED_NONEMPTY_COHORT_REQUIRED')
    for name,h in manifest['source_hashes'].items():
        path=Path(name)
        if not path.is_absolute():path=ROOT/path
        if not path.is_file() or sha(path)!=h:raise ValueError('SOURCE_HASH_MISMATCH')
    return manifest,packets

def validate_review(manifest,packets,review):
    """Strict pre-retrieval gate. A declaration is not clinical authentication."""
    if review.get('schema_version')!=VERSION or review.get('manifest_fingerprint')!=manifest['fingerprint']:
        raise ValueError('REFERENCE_MANIFEST_BINDING')
    rows=review.get('snapshots')
    if not isinstance(rows,list) or [r.get('snapshot_id') for r in rows]!=[p['snapshot_id'] for p in packets]:raise ValueError('FIXED_COHORT_MISMATCH')
    now=datetime.now(timezone.utc)
    for p,r in zip(packets,rows):
        if (r.get('review_status')!='REVIEWED' or r.get('reviewer_independent') is not True
            or not isinstance(r.get('reviewer_private_ref'),str) or not r['reviewer_private_ref'].strip()
            or r.get('procedure_version')!=PROCEDURE or r.get('query_suitable') is not True
            or r.get('query_visibility_reviewed') is not True or r.get('whole_eligible_pool_reviewed') is not True
            or r.get('model_or_retrieval_outputs_seen') is not False
            or r.get('packet_sha256')!=manifest['packets'][packets.index(p)]['sha256']
            or r.get('query_fingerprint')!=p['query_fingerprint']
            or r.get('visibility_scope')!='DEVELOPMENT_REPLAY_ONLY'
            or r.get('recorded_time_proxy_explicitly_reviewed') is not True):raise ValueError('INDEPENDENT_COMPLETE_REVIEW_REQUIRED')
        when=datetime.fromisoformat(r['reviewed_at'])
        if when.tzinfo is None or when>now or when<timestamp(p['decision_time']):raise ValueError('REVIEW_TIME_INVALID')
        if not isinstance(r.get('rationale'),str) or not r['rationale'].strip():raise ValueError('REVIEW_RATIONALE_REQUIRED')
        judgments=r.get('records');expected=[x['memory_id'] for x in p['eligible_memory']]
        if not isinstance(judgments,list) or [x.get('memory_id') for x in judgments]!=expected:raise ValueError('ANNOTATION_COVERAGE_OR_ORDER')
        for j in judgments:
            if (type(j.get('relevant')) is not bool or type(j.get('stale')) is not bool
                or type(j.get('conflict')) is not bool or j.get('not_evaluable_reason') is not None
                or not isinstance(j.get('rationale'),str) or not j['rationale'].strip()):raise ValueError('UNREVIEWED_OR_UNCERTAIN_RECORD')
    return rows

def score_packet(packet,judgment):
    records=packet['eligible_memory'];identity=packet['private_identity'];store=PatientMemoryStore(records)
    eligible={r['memory_id'] for r in records}
    annotation={'relevant_memory_ids':[r['memory_id'] for r in judgment['records'] if r['relevant']],
                'stale_memory_ids':[r['memory_id'] for r in judgment['records'] if r['stale']]}
    results=[]
    for mode in ('full_history','selective'):
        context=store.retrieve(identity,packet['query'],mode,5);selected=context['records']
        metrics=evaluate_memory(context,[],annotation,eligible)
        patient=sum(r['patient_id']!=identity['patient_id'] for r in selected)
        future=sum(r['visit_id']==identity['visit_id'] or timestamp(r['visit_started_at'])>=timestamp(identity['started_at'])
            or timestamp(r['available_at'])>timestamp(identity['started_at']) for r in selected)
        row={'snapshot_id':packet['snapshot_id'],'mode':mode,'query_fingerprint':packet['query_fingerprint'],
            'returned_record_count':len(selected),'eligible_coverage':ratio(len({r['memory_id'] for r in selected}&eligible),len(eligible)),
            'patient_leakage_count':patient,'future_cutoff_leakage_count':future,'missing_annotation_count':0,
            'precision':metrics['precision'],'stale_retrieval_rate':metrics['stale_retrieval_rate'],
            'recall_at_5':metrics['recall'] if mode=='selective' else None,
            'full_history_recall':metrics['recall'] if mode=='full_history' else None}
        results.append(row)
    if results[0]['query_fingerprint']!=results[1]['query_fingerprint']:raise ValueError('PAIRED_QUERY_CHANGED')
    return results

def evaluate(root,reference,reference_sha256):
    """No retrieval until every snapshot/record and source has passed the gate."""
    if reference is None or reference_sha256 is None:raise ValueError('REVIEWED_REFERENCE_REQUIRED')
    p=_private_file(reference)
    if not p.is_file() or sha(p)!=reference_sha256:raise ValueError('REVIEWED_REFERENCE_HASH_MISMATCH')
    manifest,packets=load_bundle(root);review=load(p);rows=validate_review(manifest,packets,review)
    results=[row for packet,judgment in zip(packets,rows) for row in score_packet(packet,judgment)]
    return {'status':'C4_02_RETRIEVAL_QUALITY_EVALUATED','manifest_fingerprint':manifest['fingerprint'],
        'reference_sha256':reference_sha256,'registered_snapshots':len(packets),'rows':results,
        'scope':'INDEPENDENTLY_REVIEWED_DEVELOPMENT_REPLAY_RELEVANCE_NOT_CLINICAL_DIAGNOSIS',
        'review_authentication':'CALLER_PINNED_DECLARATION_NOT_CRYPTOGRAPHIC_REVIEWER_IDENTITY_AUTHENTICATION',
        'memory_off':'DEFERRED_DECISION_BASELINE_NOT_RETRIEVAL_QUALITY_CONDITION',
        'clinical_metrics':'NOT_EVALUATED','model_generations':0}
