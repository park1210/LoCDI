"""D4 schema/input-boundary inspection only; not a D5 agent adapter or grader."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def validate_healthbench(row):
    if not isinstance(row,dict) or not {'prompt_id','prompt','rubrics'}.issubset(row):raise ValueError('REQUIRED_FIELDS')
    if not isinstance(row['prompt_id'],str) or not row['prompt_id'].strip():raise ValueError('SAMPLE_ID')
    if not isinstance(row['prompt'],list) or not row['prompt']:raise ValueError('PROMPT')
    for message in row['prompt']:
        if not isinstance(message,dict) or set(message)!={'role','content'}:raise ValueError('MESSAGE_SCOPE')
        if message['role'] not in {'system','user','assistant'} or not isinstance(message['content'],str) or not message['content'].strip():raise ValueError('MESSAGE_CONTENT')
    if row['prompt'][-1]['role']!='user':raise ValueError('FINAL_USER_REQUIRED')
    if not isinstance(row['rubrics'],list) or not row['rubrics']:raise ValueError('RUBRICS')
    for rubric in row['rubrics']:
        if not isinstance(rubric,dict) or not {'criterion','points','tags'}.issubset(rubric):raise ValueError('RUBRIC_FIELDS')
        if not isinstance(rubric['criterion'],str) or not rubric['criterion'].strip() or type(rubric['points']) is not int or rubric['points']==0:raise ValueError('RUBRIC_VALUES')
        if not isinstance(rubric['tags'],list) or any(not isinstance(t,str) for t in rubric['tags']):raise ValueError('RUBRIC_TAGS')
    if sum(max(0,r['points']) for r in row['rubrics'])<=0:raise ValueError('POSITIVE_RUBRIC_DENOMINATOR')
    return True


def separate_healthbench(row,*,purpose='schema_inspection',assigned_split=None):
    """Lossless prompt-only preview. Never approves training/splits/reviews."""
    validate_healthbench(row)
    if purpose not in {'schema_inspection','development_preview'} or assigned_split is not None:raise ValueError('SPLIT_OR_PURPOSE_NOT_APPROVED')
    return {'agent_input':{'messages':deepcopy(row['prompt'])},
        'private_evaluation':{k:deepcopy(v) for k,v in row.items() if k!='prompt'},
        'original_split':'test','assigned_split':None,'independent_test_approved':False,
        'clinical_diagnosis_reference_status':'NOT_REVIEWED','clinical_diagnosis_gold':None,
        'routing_correctness':None,'reviewer':None,'reviewed_at':None}


def inventory_healthbench(path):
    path=Path(path);count=0;rubrics=0;ids=set();keys=Counter();roles=Counter();turns=Counter();negative=0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():raise ValueError('BLANK_DATA_ROW')
            row=json.loads(line);validate_healthbench(row);separate_healthbench(row)
            if row['prompt_id'] in ids:raise ValueError('DUPLICATE_SAMPLE_ID')
            ids.add(row['prompt_id']);count+=1;keys.update(row.keys());rubrics+=len(row['rubrics'])
            roles.update(m['role'] for m in row['prompt']);turns[len(row['prompt'])]+=1
            negative+=sum(r['points']<0 for r in row['rubrics'])
    return {'sample_count':count,'unique_sample_ids':len(ids),'field_presence':dict(keys),'rubric_count':rubrics,
        'negative_rubric_items':negative,'message_role_counts':dict(roles),'conversation_length_counts':{str(k):v for k,v in turns.items()},
        'schema_valid_count':count,'original_split':'test','patient_identity_field':'NOT_PRESENT_AS_STRUCTURED_FIELD',
        'patient_level_linkage':'UNAVAILABLE','clinical_diagnosis_gold_created':0,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
