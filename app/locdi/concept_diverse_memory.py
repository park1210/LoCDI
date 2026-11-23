"""Explicit opt-in development selector; no Reference or clinical verdict inputs."""
import json
from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest

VERSION='locdi.concept_diverse_memory.development.v1'
FIELDS={'memory_id','score','available_at','concept_key'}
FORBIDDEN={'gold','gold_verified_concepts','gold_concepts','evaluation','label','positive','negative',
    'correct','correctness','relevant','clinical_relevance','reviewer','reference_coverage',
    'future_source_condition','source_condition','task_annotations','accepted_specialties'}


def reject_hidden(value):
    if isinstance(value,dict):
        if {str(k).casefold() for k in value}&FORBIDDEN:raise ValueError('REFERENCE_IN_POLICY_INPUT')
        for v in value.values():reject_hidden(v)
    elif isinstance(value,(list,tuple)):
        for v in value:reject_hidden(v)
    elif isinstance(value,str) and value.lstrip().startswith(('{','[')):
        try:v=json.loads(value)
        except ValueError:return
        reject_hidden(v)


def policy_spec():
    return {'version':VERSION,'scope':'DEVELOPMENT_ONLY_NO_RUNTIME_INJECTION',
        'ranking':'-existing lexical overlap integer score, -available_at UTC timestamp, memory_id lexical',
        'first_pass':'Visit complete ranked positive-score nonsuperseded candidates; take first occurrence of each verified concept until5',
        'missing_key':'Defer missing/unverified/non-observation keys to fill; never treat each unknown as a unique concept',
        'second_pass':'Fill remaining slots with unselected candidates in unchanged original ranking, including missing keys and duplicate concepts',
        'budget':5,'no_target_concept_preference':True,'no_score_threshold_tuning':True,
        'source_enrichment':'Historical Source metadata key joined only after patient/visit/cutoff and original-record binding',
        'actual_clinical_key_availability':'NOT_VALIDATED_RECORDED_TIME_PROXY',
        'clinical_superiority':False}


def rank_candidates(candidates):
    reject_hidden(candidates)
    if not isinstance(candidates,list):raise ValueError('CANDIDATE_LIST_REQUIRED')
    for c in candidates:
        if not isinstance(c,dict) or set(c)!=FIELDS:raise ValueError('CANDIDATE_WHITELIST')
        if not isinstance(c['memory_id'],str) or not c['memory_id'] or type(c['score']) is not int or c['score']<=0:
            raise ValueError('POSITIVE_INTEGER_SCORE_AND_ID_REQUIRED')
        timestamp(c['available_at'])
        k=c['concept_key']
        if k is not None and (not isinstance(k,(list,tuple)) or len(k)!=4 or any(not isinstance(x,str) or not x.strip() for x in k)):
            raise ValueError('EXACT_FOUR_FIELD_CONCEPT_KEY_REQUIRED')
    if len({c['memory_id'] for c in candidates})!=len(candidates):raise ValueError('DUPLICATE_CANDIDATE_ID')
    return sorted(candidates,key=lambda c:(-c['score'],-timestamp(c['available_at']).timestamp(),c['memory_id']))


def select(candidates,*,top_k=5,development_opt_in=False):
    if development_opt_in is not True:raise ValueError('EXPLICIT_DEVELOPMENT_OPT_IN_REQUIRED')
    if type(top_k) is not int or top_k!=5:raise ValueError('FIXED_TOP5_REQUIRED')
    ranked=rank_candidates(candidates);seen=set();chosen=[];trace=[]
    for rank,c in enumerate(ranked,1):
        k=tuple(c['concept_key']) if c['concept_key'] is not None else None
        if k is not None and k not in seen:
            chosen.append(c['memory_id']);seen.add(k)
            trace.append({'memory_id':c['memory_id'],'rank':rank,'pass':'VERIFIED_DISTINCT_CONCEPT'})
            if len(chosen)==top_k:break
    for rank,c in enumerate(ranked,1):
        if len(chosen)==min(top_k,len(ranked)):break
        if c['memory_id'] not in chosen:
            chosen.append(c['memory_id']);trace.append({'memory_id':c['memory_id'],'rank':rank,'pass':'ORIGINAL_RANK_FILL'})
    assert len(chosen)==min(top_k,len(ranked)) and len(set(chosen))==len(chosen)
    return {'version':VERSION,'memory_ids':chosen,'selection_trace':trace,
        'input_fingerprint':digest(candidates),'ranked_input_fingerprint':digest(ranked),
        'policy_fingerprint':digest(policy_spec()),'top_k':top_k,'clinical_metrics':'NOT_EVALUATED'}
