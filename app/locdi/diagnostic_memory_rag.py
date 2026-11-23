"""Opt-in M2/K2, no Core replacements, no evaluation labels or fitted parameters."""
from collections import Counter
from copy import deepcopy
import json,math,re
from app.core.longitudinal_memory import timestamp
from app.core.memory_conflicts import detect_conflicts
from app.locdi.llm_setup import digest
from app.locdi.medical_knowledge import MedicalKnowledgeIndex,assert_public,historical_labels
from app.locdi.common_runtime_recovery import released_measurement_view

VERSION='diagnostic_bm25_full_pool.v1'
MEMORY_MODES=('legacy_top5','historical_display','diagnostic_retrieval')
KNOWLEDGE_MODES=('none','common_pack','adaptive_bm25')
INSTRUMENT='Pain severity - 0-10 verbal numeric rating [Score] - Reported'
STOP={'the','a','an','and','or','of','in','at','to','is','for','with','source','observation','recorded','records','value','units','kind','provided','known'}
FIELDS={'memory_id','patient_id','visit_id','visit_started_at','observed_at','available_at','source_id','kind','text','supersedes'}
def terms(text):
    return [t for t in re.findall(r'\b\d{3,7}-\d\b|[a-z][a-z0-9]+',str(text).casefold()) if t not in STOP]
def canonical_record(r):return {k:r[k] for k in FIELDS}

def public_projection(observed,binding,runtime=None):
    """Semantic names only for actually visible values, not the whole hidden binding."""
    assert_public(observed)
    findings=list(observed.get('objective_findings',[]));history=[h for h in observed.get('history',[]) if h.get('answer_status')!='unknown']
    evidence=findings+[h.get('answer','') for h in history]
    expanded=released_measurement_view({'observation':observed,'evidence':evidence},binding)
    if runtime is not None:
        if observed['elapsed_minutes']!=runtime.current_time_min-runtime.arrival_time_min:raise ValueError('QUERY_CLOCK_MISMATCH')
        for m in expanded['released_measurements']:
            r=next(r for r in binding['rows'] if r['code']==m['code'] and r['unit']==m['unit'] and r['kind']==m['kind'])
            if r['kind']=='Test' and (r['target'] not in runtime.completed_tests or r['target']+': '+r['evidence'] not in runtime.objective_findings):raise ValueError('QUERY_TEST_NOT_RECEIVED')
            if r['kind']=='Ask' and not any(h.get('question')==r['target'] and h.get('answer')==r['evidence'] and h.get('answer_status')!='unknown' for h in runtime.question_history):raise ValueError('QUERY_ASK_NOT_EXECUTED')
    # Exclude an instrument name alone; actual released pain answers remain usable.
    symptoms=[x for x in observed.get('current_symptoms',[]) if x!=INSTRUMENT]
    query_obs={'chief_complaint':observed.get('chief_complaint',''),'current_symptoms':symptoms,
        'objective_findings':findings,'history':deepcopy(history)}
    for m in expanded['released_measurements']:
        query_obs['objective_findings'].append('{} code {} value {} unit {}'.format(m['description'],m['code'],m['value'],m['unit']))
    text=' '.join([str(query_obs['chief_complaint']),*map(str,symptoms),*map(str,query_obs['objective_findings']),
                   *[str(h.get('answer','')) for h in history]])
    return query_obs,{'query':text,'query_fingerprint':digest(query_obs),'released_measurements':expanded['released_measurements'],
        'instrument_only_labels_skipped':len(observed.get('current_symptoms',[]))-len(symptoms),
        'visibility_status':'RUNTIME_RECEIPT_VERIFIED' if runtime is not None else 'SAVED_PAYLOAD_MATCHED_NOT_LIVE_RUNTIME'}

class DiagnosticMemoryRetriever:
    """Candidate source/time validation, independent BM25, stable ties, max Top5."""
    def __init__(self,source_map):self.sources=source_map
    def retrieve(self,legacy,full_pool,identity,query,*,top_k=5,max_chars=None):
        if type(top_k) is not int or not 1<=top_k<=5:raise ValueError('DIAGNOSTIC_TOP5_LIMIT')
        if max_chars is not None and (type(max_chars) is not int or max_chars<2):raise ValueError('MEMORY_BUDGET')
        candidates=[];seen=set();duplicate=0;cutoff=timestamp(identity['started_at'])
        for r in full_pool:
            if r['patient_id']!=identity['patient_id'] or r['visit_id']==identity['visit_id'] or not timestamp(r['visit_started_at'])<=timestamp(r['observed_at'])<=timestamp(r['available_at'])<=cutoff or timestamp(r['visit_started_at'])>=cutoff:raise ValueError('MEMORY_PATIENT_VISIT_OR_FUTURE_BOUNDARY')
            source=self.sources.get(r['memory_id'])
            if not source or source['status']!='VERIFIED_EXACT_CSV_SOURCE' or source['record_fingerprint']!=digest(canonical_record(r)):raise ValueError('UNVERIFIED_MEMORY_SOURCE')
            if r.get('status_as_of_visit')=='superseded':continue
            key=(source['source_file'],source['source_row'],r['patient_id'])
            if key in seen:duplicate+=1;continue
            seen.add(key);candidates.append(deepcopy(r))
        docs=[Counter(terms(r['text'])) for r in candidates];df=Counter(t for d in docs for t in d)
        avg=max(1,sum(sum(d.values()) for d in docs)/len(docs)) if docs else 1;words=set(terms(query));rank=[]
        for r,tf in zip(candidates,docs):
            length=sum(tf.values());score=0.
            for t in words:
                f=tf[t]
                if f:score+=math.log(1+(len(docs)-df[t]+.5)/(df[t]+.5))*f*2.2/(f+1.2*(.25+.75*length/avg))
            if score>0:r['retrieval_score']=score;rank.append(r)
        rank.sort(key=lambda r:(-r['retrieval_score'],-timestamp(r['available_at']).timestamp(),r['memory_id']))
        selected=[];size=2
        for r in rank[:top_k]:
            added=len(json.dumps(r,ensure_ascii=False))+(2 if selected else 0)
            if max_chars is None or size+added<=max_chars:selected.append(r);size+=added
        result=deepcopy(legacy);result.update(records=selected,selected_count=len(selected),eligible_count=len(full_pool),
            selection_method=VERSION,max_chars=max_chars,records_json_chars=size,budget_excluded_count=min(top_k,len(rank))-len(selected),
            conflicts=detect_conflicts(selected))
        trace={'version':VERSION,'query_fingerprint':digest(query),'query':query,'candidate_scope':'ALL_ELIGIBLE_FIXED_PAST_RECORDS',
            'candidate_count':len(candidates),'duplicate_source_records_removed':duplicate,'legacy_selected_ids':[r['memory_id'] for r in legacy['records']],
            'selected':[{'memory_id':r['memory_id'],'scope':'HISTORICAL','observed_at':r['observed_at'],'available_at':r['available_at'],
                'visit_started_at':r['visit_started_at'],'source':self.sources[r['memory_id']],
                'score':r['retrieval_score'],'clinical_relevance':'NOT_REVIEWED'} for r in selected],
            'status':'AVAILABLE' if selected else 'MEMORY_UNAVAILABLE','zero_query_no_oracle_fallback':not words}
        return result,trace

def knowledge_projection(index,observed,binding,*,runtime=None):
    query_obs,q=public_projection(observed,binding,runtime)
    result=index.search(query_obs,top_k=3)
    # Fixed summaries, no runtime truncation of clinical text, no document Evidence IDs.
    packet={'scope':'EXTERNAL_KNOWLEDGE_NOT_PATIENT_EVIDENCE','mode':'adaptive_bm25','status':result['status'],
        'corpus_fingerprint':index.fingerprint,'query_fingerprint':q['query_fingerprint'],
        'warning':'Literature is not a patient observation. Never cite it as clinical evidence or update M5 from it.',
        'results':[{'document_id':r['card']['concept_id'],'score':r['score'],'disease':r['card']['disease'],
            'note':r['card']['common_note'],'references':r['references']} for r in result['results']]}
    return packet,q

def historical_series(records,source_map):
    """Exact code+unit grouping, no conversion, medical trend or abnormal label."""
    groups={}
    for r in records:
        s=source_map[r['memory_id']];key=(s.get('code'),s.get('unit'))
        if s.get('source_type')!='observations':continue
        groups.setdefault(key,[]).append({'memory_id':r['memory_id'],'observed_at':r['observed_at'],
            'value':s.get('value'),'unit':s.get('unit')})
    return [{'code':k[0],'unit':k[1],'records':sorted(v,key=lambda r:(timestamp(r['observed_at']),r['memory_id'])),
             'interpretation':'HISTORICAL_SOURCE_SERIES_NO_UNIT_CONVERSION_OR_CLINICAL_JUDGMENT'} for k,v in sorted(groups.items(),key=str)]
