"""C4-26 two-field development contract; no clinical labels or legacy repair."""
from collections import Counter
from copy import deepcopy
import re
from jsonschema import Draft202012Validator
from app.locdi.reference_acquisition import strict_json
from app.locdi.llm_setup import digest
from app.locdi.agentclinic_benchmark_scoring import normalize
from app.locdi.answer_grounded_routing import model_input,validate_pair

VERSION='locdi.answer_grounded.compact_task.v2'
PROMPT_VERSION='locdi.answer_grounded.compact_prompt.v2'
PROMPT=('Answer one static diagnostic question using only the provided case information. '
    'All supplied observations are already disclosed; do not propose actions or request tests. '
    'Return one diagnosis if you can support an answer. If the information is insufficient, return unresolved. '
    'Return only the required JSON object with exactly two keys: status and diagnosis. '
    'Use {"status":"supported","diagnosis":"single diagnosis name"} or '
    '{"status":"unresolved","diagnosis":null}. '
    'Do not include reasoning, explanations, rationale, evidence fields, multiple diagnoses, markdown, or extra text. '
    'Do not invent observations or use hidden answers. Abstention is allowed; do not guess to avoid abstention.')
REPETITION_RULE={'id':'SIX_TOKEN_REPEAT.v1','tokenization':'Python re \\w+ after casefold',
    'ngram':6,'repeated_occurrences_threshold':3,
    'interpretation':'At least three six-token occurrences duplicate an earlier six-token tuple; mechanical flag, not clinical judgment'}


def schema():
    # Semantic relation is checked locally; native support of conditional keywords is not assumed.
    return {'type':'object','required':['status','diagnosis'],'additionalProperties':False,
        'properties':{'status':{'enum':['supported','unresolved']},
            'diagnosis':{'type':['string','null'],'maxLength':512}}}


def parse(content):
    value=strict_json(content) if isinstance(content,str) else deepcopy(content)
    Draft202012Validator(schema()).validate(value)
    if value['status']=='unresolved':
        if value['diagnosis'] is not None:raise ValueError('UNRESOLVED_DIAGNOSIS_NOT_NULL')
    else:
        d=value['diagnosis']
        if not isinstance(d,str) or not d.strip():raise ValueError('SUPPORTED_DIAGNOSIS_REQUIRED')
        if '\n' in d or ';' in d or re.search(r'\s(?:or|versus|vs\.)\s',d,re.I):
            raise ValueError('MULTIPLE_DIAGNOSES_NOT_ALLOWED')
    return value


def score(q,ref,content=None,*,attempted=True,error_type=None):
    """Validate compact schema directly, reuse only the frozen exact normalization.

    Never fabricate answer_id/rationale to pass the old four-field schema.
    Linguistic diagnosis equivalence and clinical rationale are NOT_REVIEWED.
    """
    validate_pair(q,ref)
    if ref['answer_type']!='exact_text' or model_input(q)['options']:raise ValueError('FREE_TEXT_ONLY')
    result={'question_id':q['question_id'],'state':None,'answer_correct':None,'answer_fingerprint':None}
    if not attempted:
        if content is not None or error_type is not None:raise ValueError('NOT_ATTEMPTED_HAS_RESULT')
        result['state']='NOT_ATTEMPTED';return result
    if error_type is not None:result['state']='GENERATION_FAILED';return result
    try:
        r=parse(content);result['answer_fingerprint']=digest(r)
        if r['status']=='unresolved':
            result.update(state='UNRESOLVED',answer_signature=digest('UNRESOLVED'))
        else:
            correct=normalize(r['diagnosis'])==normalize(ref['answer'])
            result.update(state='CORRECT_ANSWER' if correct else 'INCORRECT_ANSWER',
                answer_correct=correct,answer_signature=digest(normalize(r['diagnosis'])))
    except Exception:result.update(state='INVALID_RESPONSE',answer_correct=None)
    return result


def compact_binding(original):
    text=original['system_prompt'];start=text.index('You are the '+original['department']+' department.')
    focus=text[start:]
    removed=[s for s in focus.splitlines() if s.startswith('- If certainty is limited,')]
    focus='\n'.join(s for s in focus.splitlines() if s not in removed)
    value={**original,'system_prompt':PROMPT+'\n\n'+focus}
    value['parent_benchmark_prompt_fingerprint']=original['benchmark_prompt_fingerprint']
    value['benchmark_prompt_fingerprint']=digest(value['system_prompt'])
    value['prompt_version']=PROMPT_VERSION
    return value


def build_bundle(parent):
    qs=deepcopy(parent['questions'][:2]);refs=deepcopy(parent['references'][:2])
    if len(qs)!=2 or len({q['question_id'] for q in qs})!=2:raise ValueError('FIRST_TWO_QUESTIONS_REQUIRED')
    bindings=[compact_binding(a) for a in parent['bindings']]
    if len(bindings)!=5:raise ValueError('FIVE_BINDINGS_REQUIRED')
    tasks=[]
    for q,r in zip(qs,refs):
        validate_pair(q,r)
        if r['answer_type']!='exact_text' or model_input(q)['options']:raise ValueError('FREE_TEXT_ONLY')
        for a in bindings:tasks.append({'question_id':q['question_id'],'agent_id':a['agent_id'],
            'input_fingerprint':digest(model_input(q)),'schema_fingerprint':digest(schema()),
            'prompt_fingerprint':a['benchmark_prompt_fingerprint'],'model_identifier':a['model_identifier']})
    plan={'version':VERSION,'prompt_version':PROMPT_VERSION,'run_id':'c4_26_compact_pilot_001',
        'namespace':'c4_26_v1','parent_plan_fingerprint':parent['plan']['fingerprint'],
        'max_generations':10,'settings':deepcopy(parent['plan']['settings']),'tasks':tasks,
        'bindings_fingerprint':digest(bindings),'question_reference_fingerprint':digest(list(zip(qs,refs))),
        'selection_rule':'FIRST_TWO_FIXED_C4_24_QUESTIONS_IN_PARENT_ORDER; five original agents in parent order',
        'development_only':True,'clinical_reference_approved':False,'resume':False}
    plan['fingerprint']=digest(plan)
    return {'plan':plan,'questions':qs,'references':refs,'bindings':bindings,'source_pins':deepcopy(parent['source_pins'])}


def repetition(text):
    tokens=re.findall(r'\w+',text.casefold());seen=set();n=0;first=None
    for i in range(max(0,len(tokens)-5)):
        gram=tuple(tokens[i:i+6])
        if gram in seen:
            n+=1
            if first is None:first=i
        seen.add(gram)
    return {'rule_id':REPETITION_RULE['id'],'repeated_six_token_occurrences':n,
        'first_repeat_token_offset':first,'repetitive':n>=3}
