"""Private external QA tasks and strict model projection, independent of Core.

No patient simulation, clinical grading, model dispatcher or routing-fit adapter.
Index contracts must be externally pinned against exact source/option ordering.
"""
from copy import deepcopy
import json
import re
from jsonschema import Draft202012Validator
from app.locdi.llm_setup import digest
from app.locdi.reference_acquisition import strict_json

REFERENCE_TYPES=('SOURCE_GROUNDED_REFERENCE','EXAM_GOLD','RUBRIC_REFERENCE','CLINICAL_CASE_REFERENCE_CANDIDATE','INDEPENDENT_CLINICAL_REFERENCE')
FIELDS=('task_id','dataset_id','dataset_version','original_split','source_fingerprint','prompt_fingerprint',
 'task_type','input_messages','answer_format','answer_key_reference','usage_scope','exposure_status','evaluation_status')
TASK_SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'locdi.external_reference_task.v1',
 'type':'object','required':['schema_version']+list(FIELDS),'additionalProperties':False,
 'properties':{'schema_version':{'const':'locdi.external_reference_task.v1'},
  **{k:{'type':'string','minLength':1} for k in FIELDS if k not in ('input_messages','answer_format')},
  'source_fingerprint':{'type':'string','pattern':'^[0-9a-f]{64}$'},'prompt_fingerprint':{'type':'string','pattern':'^[0-9a-f]{64}$'},
  'input_messages':{'type':'array','minItems':1,'maxItems':1,'items':{'type':'object','required':['role','content'],'additionalProperties':False,
    'properties':{'role':{'const':'user'},'content':{'type':'string','minLength':1}}}},
  'answer_format':{'type':'object','required':['type','labels','abstention'],'additionalProperties':False,
    'properties':{'type':{'enum':['single_option','free_text']},'labels':{'type':'array','items':{'type':'string'},'uniqueItems':True},'abstention':{'const':'ABSTAIN'}}},
  'task_type':{'enum':['multiple_choice','case_diagnosis_candidate']},'usage_scope':{'const':'SEPARATE_EXTERNAL_TASK_ONLY'},
  'evaluation_status':{'enum':['NOT_EVALUATED','REVIEW_REQUIRED']}}}
FORBIDDEN={'answer','answer_idx','cop','exp','explanation','gold','rubric','rubrics','canary','final_diagnosis','diagnostic_reasoning','evaluation','provenance','topic','topic_name','subject','subject_name'}


def text(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('NONEMPTY_TEXT_REQUIRED')
    return value


def question_guard(question,private_strings=()):
    text(question)
    if re.search(r'''(?i)\b(answer(?:_idx)?|correct\s+(?:answer|option)|gold|explanation|rubrics?|canary)["']?\s*[:=]''',question):
        raise ValueError('EXPLICIT_ANSWER_OR_METADATA_IN_QUESTION')
    if any(isinstance(v,str) and len(v)>=12 and v in question for v in private_strings):raise ValueError('PRIVATE_REFERENCE_IN_QUESTION')


def _messages(question,options):
    return [{'role':'user','content':json.dumps({'question':question,'options':options},ensure_ascii=False,separators=(',',':'))}]


def index_contract(rows,dataset_id,version,split,convention):
    """Construct a REVIEW input, not approval. Caller must pin its independently checked hash.

    No default convention and no 0/1-base guessing. Originals/ordering are bound.
    """
    return {'schema_version':'locdi.private.answer_index_contract.v1','dataset_id':dataset_id,'version':version,
      'split':split,'index_convention':convention,'rows_fingerprint':digest(rows),
      'row_fingerprints':[digest(r) for r in rows],
      'review_status':'REQUIRES_EXTERNAL_INDEX_VERIFICATION_NOT_CLINICAL_REFERENCE'}


def adapt_exam(rows,contract,*,verified_contract_fingerprint):
    if not isinstance(rows,list) or not rows:raise ValueError('NONEMPTY_EXAM_ROWS_REQUIRED')
    if digest(contract)!=verified_contract_fingerprint or contract.get('rows_fingerprint')!=digest(rows):raise ValueError('SOURCE_OR_INDEX_CONTRACT_FINGERPRINT')
    if contract.get('row_fingerprints')!=[digest(r) for r in rows]:raise ValueError('ROW_ORDER_FINGERPRINT')
    dataset=contract['dataset_id'];split=contract['split'];encoding=contract['index_convention']
    if dataset not in ('MedQA','MedMCQA'):raise ValueError('EXAM_DATASET_REQUIRED')
    if split not in (('train','dev','validation','test') if dataset=='MedQA' else ('train','validation','test')):raise ValueError('SPLIT_REQUIRED')
    if dataset=='MedMCQA' and split=='test':raise ValueError('OFFICIAL_TEST_ANSWER_WITHHELD')
    if (dataset=='MedQA' and encoding!='medqa_letter') or (dataset=='MedMCQA' and encoding not in ('medmcqa_csv_one_based','medmcqa_hf_zero_based')):raise ValueError('EXPLICIT_INDEX_CONVENTION_REQUIRED')
    seen_ids=set();seen_questions=set();tasks=[];keys={}
    for ordinal,row in enumerate(rows,1):
        question=text(row.get('question'));question_guard(question,(row.get('exp'),row.get('explanation')))
        original=row.get('id',str(ordinal))
        if not isinstance(original,str) or not original or original in seen_ids:raise ValueError('DUPLICATE_OR_INVALID_ID')
        qkey=' '.join(question.casefold().split())
        if qkey in seen_questions:raise ValueError('DUPLICATE_QUESTION')
        seen_ids.add(original);seen_questions.add(qkey)
        if dataset=='MedQA':
            opts=row.get('options')
            if not isinstance(opts,dict) or len(opts) not in (4,5) or list(opts)!=list('ABCDE'[:len(opts)]):raise ValueError('OPTION_ORDER_OR_COUNT')
            labels=list(opts);values=[text(opts[k]) for k in labels];answer=row.get('answer_idx')
            if answer not in labels:raise ValueError('ANSWER_KEY_MISSING_OR_OUT_OF_RANGE')
        else:
            if row.get('choice_type')!='single':raise ValueError('UNSUPPORTED_CHOICE_TYPE')
            labels=list('ABCD');values=[text(row.get('op'+k.lower())) for k in labels]
            key=row.get('cop');base=1 if encoding=='medmcqa_csv_one_based' else 0
            if type(key) is not int or not base<=key<base+4:raise ValueError('ANSWER_KEY_MISSING_OR_OUT_OF_RANGE')
            answer=labels[key-base]
        if len(set(' '.join(v.casefold().split()) for v in values))!=len(values):raise ValueError('DUPLICATE_OPTIONS')
        options=[{'label':k,'text':v} for k,v in zip(labels,values)]
        for v in values:question_guard(v,(row.get('exp'),row.get('explanation')))
        source=digest(row);ref='AnswerKey%06d'%ordinal;messages=_messages(question,options)
        task={'schema_version':'locdi.external_reference_task.v1','task_id':'Task%06d'%ordinal,
         'dataset_id':dataset,'dataset_version':contract['version'],'original_split':split,'source_fingerprint':source,
         'prompt_fingerprint':digest(messages),'task_type':'multiple_choice','input_messages':messages,
         'answer_format':{'type':'single_option','labels':labels,'abstention':'ABSTAIN'},'answer_key_reference':ref,
         'usage_scope':'SEPARATE_EXTERNAL_TASK_ONLY','exposure_status':'TRAINING_SPLIT_DEVELOPMENT_EXPOSURE' if split=='train' else 'CONTAMINATION_AND_EXPOSURE_NOT_CLEARED',
         'evaluation_status':'NOT_EVALUATED'}
        keys[ref]={'answer_label':answer,'original_sample_reference':original,'source_fingerprint':source,
         'original_split':split,'index_contract_fingerprint':digest(contract),'reference_type':'EXAM_GOLD',
         'option_order_fingerprint':digest(options),'contamination':'UNKNOWN','clinical_gold_approved':False}
        validate_task(task);tasks.append(task)
    return tasks,keys


def validate_task(task):
    Draft202012Validator(TASK_SCHEMA).validate(task)
    if digest(task['input_messages'])!=task['prompt_fingerprint']:raise ValueError('PROMPT_FINGERPRINT')
    content=task['input_messages'][0]['content']
    if task['task_type']=='multiple_choice':
        payload=strict_json(content)
        if set(payload)!= {'question','options'}:raise ValueError('MODEL_INPUT_FIELD_NOT_ALLOWED')
        question_guard(payload['question'])
        opts=payload['options'];labels=task['answer_format']['labels']
        if not isinstance(opts,list) or len(opts)!=len(labels) or len(labels) not in (4,5):raise ValueError('MODEL_OPTIONS_COUNT')
        if task['answer_format']['type']!='single_option':raise ValueError('ANSWER_FORMAT')
        for row,label in zip(opts,labels):
            if set(row)!= {'label','text'} or row['label']!=label:raise ValueError('MODEL_OPTION_ORDER')
            text(row['text']);question_guard(row['text'])
    else:
        if task['answer_format']['type']!='free_text' or task['answer_format']['labels']:raise ValueError('CASE_ANSWER_FORMAT')
        question_guard(content)
    return task


def model_projection(task):
    validate_task(task)
    return deepcopy(task['input_messages'])


def public_projection(tasks):
    # No task/item IDs, original text, answers or source paths; fingerprints are audit only.
    for task in tasks:validate_task(task)
    return {'schema_version':'locdi.external_reference_public.v1','task_count':len(tasks),
      'items':[{'ordinal':i,'dataset_id':t['dataset_id'],'original_split':t['original_split'],
        'task_type':t['task_type'],'source_fingerprint':t['source_fingerprint'],'prompt_fingerprint':t['prompt_fingerprint'],
        'exposure_status':t['exposure_status'],'evaluation_status':t['evaluation_status']} for i,t in enumerate(tasks,1)],
      'clinical_metrics':'NOT_EVALUATED','clinical_reference_approved':False}


def require_use(task,*,purpose,independent_test=False):
    validate_task(task)
    if purpose!='external_evaluation':raise ValueError('EXTERNAL_REFERENCE_PROMOTION_FORBIDDEN')
    if independent_test:raise ValueError('INDEPENDENT_PATIENT_TEST_NOT_AUTHENTICATED')
    return 'SEPARATE_TASK_ONLY'


def medcase_spec():
    return {'schema_version':'locdi.medcase_adapter_spec.v1','status':'DESIGN_ONLY_RAW_SCHEMA_UNVERIFIED',
      'input_allowlist':['case_prompt'],'private_fields':['final_diagnosis','diagnostic_reasoning','pmcid','publication_date','text','title','article_link'],
      'reference_type':'CLINICAL_CASE_REFERENCE_CANDIDATE','index_rule':'NOT_APPLICABLE_FREE_TEXT',
      'duplicate_rule':'Reject duplicate normalized case_prompt, sample ID, PMCID-derived cases until grouped source-article split review',
      'patient_disjoint_from_distinct_pmcid':False,'publication_date_is_result_release':False,
      'grading':'NOT_IMPLEMENTED; synonym/partial/hierarchy policy and validated adjudicator required; exact match is lexical only',
      'automatic_current_decision_reference':False,'acquired_rows':0}


def inspect_case_candidates(rows):
    """TEST_ONLY/pre-acquisition review guard, not a schema-verified real adapter."""
    seen=set();papers=set();ids=set()
    for i,row in enumerate(rows,1):
        prompt=text(row.get('case_prompt'));paper=text(row.get('pmcid'));item=row.get('id',str(i))
        key=' '.join(prompt.casefold().split())
        if paper in papers:raise ValueError('SHARED_PMCID_REQUIRES_GROUPED_REVIEW')
        if key in seen or item in ids:raise ValueError('DUPLICATE_CASE_OR_ID')
        question_guard(prompt,(row.get('diagnostic_reasoning'),row.get('text')))
        # Literal final diagnosis presence is a review trigger, not clinical equivalence.
        diagnosis=row.get('final_diagnosis')
        if isinstance(diagnosis,str) and diagnosis.strip() and diagnosis.casefold() in prompt.casefold():raise ValueError('CASE_REFERENCE_TEXT_REVIEW_REQUIRED')
        seen.add(key);papers.add(paper);ids.add(item)
    return {'status':'DESIGN_GUARDS_CHECKED_RAW_SCHEMA_UNVERIFIED','rows':len(rows),'clinical_equivalence':None}


def qa_metrics(tasks,outputs,answer_keys=None):
    """Symbolic option scoring only; no client, grader or diagnosis judgments."""
    if len(tasks)!=len(outputs):raise ValueError('OUTPUT_COUNT_MISMATCH')
    n=len(tasks);answered=abstained=invalid=correct=known=0
    for task,output in zip(tasks,outputs):
        validate_task(task)
        if task['task_type']!='multiple_choice':raise ValueError('FREE_TEXT_GRADING_NOT_IMPLEMENTED')
        key=(answer_keys or {}).get(task['answer_key_reference'])
        if key:
            if key.get('source_fingerprint')!=task['source_fingerprint'] or key.get('option_order_fingerprint')!=digest(json.loads(task['input_messages'][0]['content'])['options']):raise ValueError('ANSWER_SOURCE_OR_OPTIONS_MISMATCH')
            if key.get('answer_label') not in task['answer_format']['labels']:raise ValueError('INVALID_PRIVATE_ANSWER_KEY')
        if output=='ABSTAIN':abstained+=1;continue
        if not isinstance(output,str) or output not in task['answer_format']['labels']:invalid+=1;continue
        answered+=1
        if key:
            if key.get('source_fingerprint')!=task['source_fingerprint'] or key.get('option_order_fingerprint')!=digest(json.loads(task['input_messages'][0]['content'])['options']):raise ValueError('ANSWER_SOURCE_OR_OPTIONS_MISMATCH')
            if key.get('answer_label') not in task['answer_format']['labels']:raise ValueError('INVALID_PRIVATE_ANSWER_KEY')
            known+=1;correct+=output==key['answer_label']
    def rate(x,d):return {'numerator':x,'denominator':d,'value':x/d if d else None,'status':'TEST_ONLY_OR_SEPARATE_EXAM_METRIC' if d else 'NOT_APPLICABLE'}
    all_keys=bool(tasks) and answer_keys is not None and all(bool(answer_keys.get(t['answer_key_reference'])) for t in tasks)
    return {'coverage':rate(answered,n),'abstention_rate':rate(abstained,n),'invalid_output_rate':rate(invalid,n),
      'selective_accuracy':rate(correct,answered) if known==answered and answered and all_keys else {'status':'NOT_EVALUATED','value':None},
      'overall_answer_accuracy':rate(correct,n) if all_keys else {'status':'NOT_EVALUATED','value':None},
      'denominator_policy':'Coverage/abstention/invalid: all planned tasks; selective: valid answered only with complete independent exam keys; overall: all tasks, abstention/invalid non-success',
      'clinical_metrics':'NOT_EVALUATED','routing_or_voi_validation':False}


def execute_model(*args,**kwargs):
    raise RuntimeError('C3_06_MODEL_AND_GRADER_EXECUTION_FORBIDDEN')
