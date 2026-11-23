"""Small deterministic BM25; no model, vector DB, patient labels or network."""
from collections import Counter
from copy import deepcopy
import json,math,re
from urllib.parse import urlparse
from app.locdi.llm_setup import digest
VERSION='common_knowledge_and_bm25.v1'
FORBIDDEN={'gold','evaluation','task_annotations','source_event_store','accepted_diagnoses','accepted_departments','reference_diagnosis','correct_diagnosis'}
HOSTS={'www.nhlbi.nih.gov','www.niddk.nih.gov','www.niams.nih.gov','www.nhs.uk','www.kidney.org','medlineplus.gov','kdigo.org'}
def tokens(text):return re.findall(r'[a-z0-9]+',str(text).casefold())
def assert_public(obj):
    if isinstance(obj,dict):
        if any(k.casefold() in FORBIDDEN for k in obj):raise ValueError('PRIVATE_REFERENCE_IN_KNOWLEDGE_QUERY')
        for v in obj.values():assert_public(v)
    elif isinstance(obj,list):
        for v in obj:assert_public(v)

def public_query(observed):
    assert_public(observed)
    parts=[observed.get('chief_complaint',''),*observed.get('current_symptoms',[]),*observed.get('objective_findings',[])]
    parts += [h.get('answer','') for h in observed.get('history',[]) if h.get('answer_status')!='unknown']
    # No working_assessment, predicted diagnosis, Source appendix or unprovided Memory.
    return ' '.join(str(p) for p in parts)

class MedicalKnowledgeIndex:
    def __init__(self,corpus):
        self.corpus=deepcopy(corpus);self.fingerprint=digest(corpus)
        sources=corpus.get('sources',{});cards=corpus.get('cards',[])
        if not sources or not cards:raise ValueError('KNOWLEDGE_UNAVAILABLE')
        for s in sources.values():
            u=urlparse(s['url'])
            if u.scheme!='https' or u.hostname not in HOSTS or s.get('verification')!='PRIMARY_SOURCE_CONTENT_CHECKED':
                raise ValueError('UNTRUSTED_KNOWLEDGE_SOURCE')
        if len({c['concept_id'] for c in cards})!=len(cards):raise ValueError('DUPLICATE_KNOWLEDGE_ID')
        for c in cards:
            if (c.get('status')!='SOURCE_CHECKED_SUMMARY_CLINICAL_REVIEW_NOT_DONE'
                or not c.get('source_ids') or any(s not in sources for s in c['source_ids'])):raise ValueError('UNVERIFIED_KNOWLEDGE_CARD')
        from app.locdi.medical_knowledge_catalog import corpus as canonical
        if digest(corpus)!=digest(canonical()):raise ValueError('CORPUS_NOT_CANONICAL_SOURCE_CHECKED_SNAPSHOT')
        self.cards=cards;self.tf=[];self.df=Counter()
        for c in cards:
            counts=Counter(tokens(' '.join(str(c[k]) for k in ('disease','symptoms','tests','diagnostic_features','differential'))))
            self.tf.append(counts);self.df.update(counts.keys())
        self.average_length=sum(sum(t.values()) for t in self.tf)/len(self.tf)

    def common(self):
        # Exactly the same whole catalog pack for every condition and patient.
        return {'version':VERSION,'scope':'EXTERNAL_KNOWLEDGE_NOT_PATIENT_EVIDENCE','corpus_fingerprint':self.fingerprint,
            'warning':'Literature is not an observed patient fact; never cite it as evidence_indices or update M5 from it.',
            'cards':[{'id':c['concept_id'],'note':c['common_note'],'sources':c['source_ids']} for c in self.cards]}

    def search(self,observed,top_k=3):
        if type(top_k) is not int or not 1<=top_k<=5:raise ValueError('BOUNDED_KNOWLEDGE_TOP_K')
        query=tokens(public_query(observed));rank=[];n=len(self.cards)
        for i,tf in enumerate(self.tf):
            score=0.;length=sum(tf.values())
            for t in set(query):
                f=tf[t]
                if f:
                    idf=math.log(1+(n-self.df[t]+.5)/(self.df[t]+.5))
                    score+=idf*f*2.2/(f+1.2*(.25+.75*length/self.average_length))
            if score>0:rank.append((score,self.cards[i]))
        rank.sort(key=lambda x:(-x[0],x[1]['concept_id']))
        return {'status':'AVAILABLE' if rank else 'KNOWLEDGE_UNAVAILABLE','scope':'EXTERNAL_KNOWLEDGE_NOT_PATIENT_EVIDENCE',
            'corpus_fingerprint':self.fingerprint,'results':[{'score':s,'card':deepcopy(c),
                'references':{sid:deepcopy(self.corpus['sources'][sid]) for sid in c['source_ids']}} for s,c in rank[:top_k]]}

def historical_labels(observed):
    """Read-only labels, not relevance/stale/conflict judgments or new records."""
    records=observed.get('longitudinal_memory',{}).get('records',[])
    return {'contract':'current_historical_display.v1','current_scope':'Current evidence is in observation/evidence only.',
        'historical_records':[{'memory_index':i,'scope':'HISTORICAL_NOT_CURRENT_EVIDENCE',
            'visit_started_at':r.get('visit_started_at'),'observed_at':r.get('observed_at'),
            'available_at':r.get('available_at'),'source_status':r.get('status_as_of_visit')}
            for i,r in enumerate(records)],
        'warning':'Past measurements do not establish current findings. Clinical relevance, stale and conflict remain unreviewed.'}
