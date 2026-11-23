"""Separate answer-grounded development tasks; no clinical routing/fit adapter."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from jsonschema import Draft202012Validator
from app.core.agent_registry import AgentRegistry
from app.locdi.agentclinic_scenario_schema import validate_record,REVISION,SOURCE_SHA256
from app.locdi.agentclinic_benchmark_scoring import normalize
from app.locdi.external_reference_task import question_guard
from app.locdi.reference_acquisition import strict_json
from app.locdi.llm_setup import digest

VERSION='locdi.answer_grounded_routing.c4_24.v1'
TARGET='ANSWER_GROUNDED_AGENT_SELECTION'
DEPARTMENTS={'cardiology','neurology','nephrology','endocrinology','pulmonology'}
STATES=('CORRECT_ANSWER','INCORRECT_ANSWER','UNRESOLVED','INVALID_RESPONSE','GENERATION_FAILED','NOT_ATTEMPTED')
TASK_PROMPT=('Answer one static diagnostic question using only its supplied facts and public options. '
    'All facts in this static task are disclosed at once; there is no action or time simulation. '
    'Return only JSON with status (supported or unresolved), answer_id, diagnosis, rationale. '
    'For public options, use exactly one option ID and diagnosis=null. For no options, '
    'use answer_id=null and a single most specific supported diagnosis, without adding alternatives. '
    'You may abstain with status=unresolved and both answer_id and diagnosis=null. '
    'Do not invent observations. The answer key and explanations are not available.')


def task(row,*,dataset,version,ordinal,original_split,exposure,case_group=None):
    """Read explicit MedQA options, preserving option order. Not Raw admission."""
    question=row.get('question');question_guard(question,(row.get('explanation'),row.get('exp')))
    options=row.get('options');answer=row.get('answer_idx')
    if not isinstance(options,dict) or list(options)!=list('ABCDE'[:len(options)]) or len(options) not in (4,5):
        raise ValueError('EXPLICIT_ORDERED_OPTIONS_REQUIRED')
    if not isinstance(answer,str) or answer not in options:raise ValueError('EXPLICIT_ANSWER_ID_REQUIRED')
    if any(not isinstance(x,str) or not x.strip() for x in options.values()):raise ValueError('OPTION_TEXT')
    if len({normalize(x) for x in options.values()})!=len(options):raise ValueError('DUPLICATE_OPTIONS')
    for v in options.values():question_guard(v,(row.get('explanation'),row.get('exp')))
    if row.get('answer') is not None and normalize(row['answer'])!=normalize(options[answer]):raise ValueError('ANSWER_TEXT_ID_DISAGREE')
    return _pair({'question':question,'options':[{'id':k,'text':v} for k,v in options.items()]},row,
        dataset,version,ordinal,original_split,exposure,'option_id',answer,case_group)


def _pair(public,raw,dataset,version,ordinal,split,exposure,kind,answer,case_group):
    source=digest(raw);qid='%s:%s:%06d:%s'%(dataset,version,ordinal,source[:16])
    if dataset=='MedQA' and 'id' in raw:
        if not isinstance(raw['id'],str) or not raw['id'].strip():raise ValueError('SOURCE_QUESTION_ID')
        qid='%s:%s:source-id:%s'%(dataset,version,raw['id'])
    q={'question_id':qid,'dataset_id':dataset,'dataset_version':version,'original_split':split,
       'source_fingerprint':source,'input_fingerprint':digest(public),'input':deepcopy(public),
       'case_group':case_group or source,'exposure':exposure,'training_overlap':'UNKNOWN',
       'model_pretraining_exposure':'UNKNOWN','role':'development'}
    ref={'question_id':qid,'input_fingerprint':digest(public),'source_fingerprint':source,
        'answer_type':kind,'answer':answer,'reference_type':'DATASET_DECLARED_ANSWER',
        'clinical_reference_approved':False,'scorer_version':'option_id_exact.v1' if kind=='option_id' else 'agentclinic.lexical_scoring.v1'}
    validate_pair(q,ref)
    return q,ref


def read_medqa(path,*,version,split,exposure):
    """Reader only; caller must separately establish actual acquisition/use status."""
    rows=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():raise ValueError('EMPTY_JSONL_ROW')
        rows.append(task(strict_json(line),dataset='MedQA',version=version,ordinal=i,
            original_split=split,exposure=exposure))
    return rows


def read_agentclinic(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256:raise ValueError('SOURCE_HASH_MISMATCH')
    rows=[strict_json(line) for line in raw.decode('utf-8').splitlines()]
    if len(rows)!=107:raise ValueError('SOURCE_COUNT')
    pairs=[];excluded=[]
    for ordinal,row in enumerate(rows,1):
        c=validate_record(row)
        public={'question':json.dumps({k:deepcopy(c[k]) for k in ('Objective_for_Doctor','Patient_Actor',
            'Physical_Examination_Findings','Test_Results')},ensure_ascii=False,sort_keys=True),'options':[]}
        answer=c['Correct_Diagnosis']
        # Pre-fixed source-only leakage screen. Exclude whole problem, never redact facts.
        literal=re.search(r'(?<!\w)'+re.escape(normalize(answer))+r'(?!\w)',normalize(public['question']))
        if literal:
            excluded.append({'source_ordinal':ordinal,'reason':'REFERENCE_LITERAL_IN_FULL_STATIC_INPUT','source_fingerprint':digest(row)})
            continue
        pair=_pair(public,row,'AgentClinic-MedQA',REVISION,ordinal,'UNKNOWN_ORIGINAL_SPLIT',
            'DEVELOPMENT_EXPOSED_C3_11_ALL107', 'exact_text',answer,digest(row))
        pairs.append((ordinal,pair[0],pair[1]))
    return pairs,excluded


def model_input(q):
    value=q['input']
    if set(value)!={'question','options'} or digest(value)!=q['input_fingerprint']:raise ValueError('MODEL_INPUT_BINDING')
    question_guard(value['question'])
    if not isinstance(value['options'],list):raise ValueError('OPTIONS_LIST')
    for r in value['options']:
        if set(r)!={'id','text'}:raise ValueError('MODEL_INPUT_OPTION_FIELDS')
        question_guard(r['text'])
    return deepcopy(value)


def validate_pair(q,ref):
    public=model_input(q)
    if (ref['question_id']!=q['question_id'] or ref['input_fingerprint']!=q['input_fingerprint']
            or ref['source_fingerprint']!=q['source_fingerprint']):raise ValueError('REFERENCE_BINDING')
    if not isinstance(ref['answer'],str) or not ref['answer'].strip():raise ValueError('REFERENCE_REQUIRED')
    labels=[o['id'] for o in public['options']]
    if ref['answer_type']=='option_id':
        if ref['answer'] not in labels:raise ValueError('ANSWER_OPTION_ID')
    elif ref['answer_type']=='exact_text':
        if labels:raise ValueError('FREE_TEXT_HAS_OPTIONS')
    else:raise ValueError('REFERENCE_KIND')
    return True


def answer_schema(q):
    ids=[o['id'] for o in model_input(q)['options']]
    answer_id={'enum':[None]+ids};diagnosis={'type':['string','null'],'maxLength':512}
    return {'type':'object','additionalProperties':False,'required':['status','answer_id','diagnosis','rationale'],
        'properties':{'status':{'enum':['supported','unresolved']},'answer_id':answer_id,
            'diagnosis':diagnosis,'rationale':{'type':'string','minLength':1,'maxLength':4096}}}


def score(q,ref,content=None,*,attempted=True,error_type=None):
    validate_pair(q,ref)
    result={'question_id':q['question_id'],'state':None,'answer_correct':None,'answer_fingerprint':None}
    if not attempted:
        if content is not None or error_type is not None:raise ValueError('NOT_ATTEMPTED_HAS_RESULT')
        result['state']='NOT_ATTEMPTED';return result
    if error_type is not None:result['state']='GENERATION_FAILED';return result
    try:
        r=strict_json(content) if isinstance(content,str) else deepcopy(content)
        Draft202012Validator(answer_schema(q)).validate(r)
        result['answer_fingerprint']=digest(r)
        if r['status']=='unresolved':
            if r['answer_id'] is not None or r['diagnosis'] is not None:raise ValueError('UNRESOLVED_CONTRADICTION')
            result['state']='UNRESOLVED';return result
        if ref['answer_type']=='option_id':
            if r['answer_id'] is None or r['diagnosis'] is not None:raise ValueError('SUPPORTED_OPTION_REQUIRED')
            correct=r['answer_id']==ref['answer']
        else:
            if r['answer_id'] is not None or not isinstance(r['diagnosis'],str) or not r['diagnosis'].strip():raise ValueError('SUPPORTED_DIAGNOSIS_REQUIRED')
            correct=normalize(r['diagnosis'])==normalize(ref['answer'])
        result.update(state='CORRECT_ANSWER' if correct else 'INCORRECT_ANSWER',answer_correct=correct)
    except Exception:
        # Error bodies and clinical response text never enter public score rows.
        result.update(state='INVALID_RESPONSE',answer_correct=None)
    return result


def split_audit(questions,*,training_inputs=(),training_case_groups=(),training_scan_complete=False):
    seen={k:{} for k in ('id','question','input','case')};issues=[];rows=[]
    training=set(training_inputs);groups=set(training_case_groups)
    for q in questions:
        public=model_input(q);role=q['role']
        keys={'id':q['question_id'],'question':normalize(public['question']),
              'input':digest(public),'case':q['case_group']}
        for kind,key in keys.items():
            if key in seen[kind]:
                issues.append({'code':'DUPLICATE_'+kind.upper(),'cross_split':seen[kind][key]!=role})
            seen[kind][key]=role
        overlap=digest(public) in training or q['case_group'] in groups
        if overlap and role in ('validation','test'):issues.append({'code':'TRAINING_OVERLAP','cross_split':True})
        if role=='test' and q['exposure']!='INDEPENDENT_UNEXPOSED_VERIFIED':issues.append({'code':'DEVELOPMENT_EXPOSED_TEST','cross_split':True})
        rows.append({'question_id':q['question_id'],'role':role,'exposure':q['exposure'],
            'training_overlap':'MATCH' if overlap else ('NO_MATCH_WITHIN_COMPLETE_INSPECTED_CORPUS' if training_scan_complete else 'UNKNOWN_NO_MATCH_IN_SCANNED_INPUTS'),
            'model_pretraining_exposure':q['model_pretraining_exposure']})
    return {'status':'BLOCKED' if issues else 'STRUCTURAL_PASS_INDEPENDENCE_NOT_ESTABLISHED',
        'issues':issues,'rows':rows,'training_scan_complete':training_scan_complete,'independent_test_ready':False}


def bind_agents(registry,config,prompt_loader):
    reg=AgentRegistry(registry);rows=[]
    agents=[a for a in reg.snapshot()['agents'] if a['role']=='specialty']
    if len(agents)!=5 or {a['department'] for a in agents}!=DEPARTMENTS:raise ValueError('FIVE_SPECIALTY_BINDINGS_REQUIRED')
    for a in agents:
        b=next((b for b in config['agent_bindings'] if b['agent_id']==a['agent_id']),None)
        if b is None or a['model']!=config['model'] or a['version']!=config['model_version'] or b['model_identifier']!=config['model_identifier']:
            raise ValueError('MODEL_VERSION_DIGEST_BINDING')
        original=prompt_loader(a['department'])
        rows.append({**a,'model_identifier':b['model_identifier'],'original_prompt_fingerprint':digest(original),
            'system_prompt':TASK_PROMPT+'\n\n'+original,'benchmark_prompt_fingerprint':digest(TASK_PROMPT+'\n\n'+original),
            'adapter_lora_binding':'NOT_CONFIGURED_IN_CURRENT_REGISTRY'})
    return rows


def build_plan(questions,refs,bindings,settings,*,run_id='c4_24_pilot_001'):
    if len(questions)!=5 or len(bindings)!=5:raise ValueError('PILOT_REQUIRES_FIVE_BY_FIVE')
    if len({q['question_id'] for q in questions})!=5:raise ValueError('DUPLICATE_PILOT_QUESTION')
    if split_audit(questions)['issues']:raise ValueError('PILOT_SPLIT_OR_DUPLICATE')
    required={'temperature':0,'seed':42,'context':32768,'max_output_tokens':1024,'timeout_seconds':120,'retry':0,'repair':0}
    if settings!=required:raise ValueError('FROZEN_GENERATION_SETTINGS')
    tasks=[]
    for q,ref in zip(questions,refs):
        validate_pair(q,ref)
        for a in bindings:
            tasks.append({'question_id':q['question_id'],'agent_id':a['agent_id'],
                'input_fingerprint':digest(model_input(q)),'schema_fingerprint':digest(answer_schema(q)),
                'prompt_fingerprint':a['benchmark_prompt_fingerprint'],'model_identifier':a['model_identifier']})
    p={'version':VERSION,'target':TARGET,'run_id':run_id,'purpose':'DEVELOPMENT_ONLY',
        'max_generations':25,'tasks':tasks,'settings':settings,'bindings_fingerprint':digest(bindings),
        'question_reference_fingerprint':digest(list(zip(questions,refs))),
        'actual_generation':0,'execution_authorized':False,'clinical_routing_ready':False,
        'utility_fitting_authorized':False,'resume':False}
    p['fingerprint']=digest(p);return p
