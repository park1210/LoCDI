"""D6 adapted bounded protocol. No real grader configured, no clinical verdicts."""
import ast
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import re
from app.locdi.llm_setup import ROOT,digest
from app.locdi.external_dialogue import MANIFEST,DATA_NAME,messages_for,budget
from app.locdi.dataset_suitability import sha
from app.locdi.source_replay import atomic_rows

REVISION='652c89d0ca9df547706735883097e9537d40dc47'
EXPECTED_EVALUATOR='b763d8c1f53ecd165f0d170a32bc948216b3dfeadafe238b2083328e1af4b2df'
GRADER_CONFIG={'model':None,'version':None,'model_identifier':None,'provider':None,'endpoint':None,
    'prompt_version':REVISION+':GRADER_TEMPLATE','temperature':None,'seed':None,'max_output_tokens':None,
    'timeout_seconds':30,'sdk_retries':0,'max_attempts_per_rubric':1,'quality_validated':False}


def official_template():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));path=ROOT/manifest['files']['healthbench_eval.py']['path']
    if sha(path)!=EXPECTED_EVALUATOR:raise ValueError('OFFICIAL_EVALUATOR_HASH_MISMATCH')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRADER_TEMPLATE' for t in n.targets))
    if not isinstance(assignment.value,ast.Call) or assignment.value.func.attr!='strip':raise ValueError('TEMPLATE_SHAPE_CHANGED')
    return ast.literal_eval(assignment.value.func.value).strip()


def grader_messages(row,completion,rubric):
    prompt=messages_for(row)
    if not isinstance(completion,str) or not completion.strip():raise ValueError('COMPLETION_REQUIRED')
    conversation=prompt+[{'role':'assistant','content':completion}]
    text='\n\n'.join(m['role']+': '+m['content'] for m in conversation)
    # Identical template/substitution shape; no reference, tags, canary or IDs.
    return [{'role':'user','content':official_template().replace('<<conversation>>',text).replace(
        '<<rubric_item>>','['+str(rubric['points'])+'] '+rubric['criterion'])}]


def parse_grade(text):
    if not isinstance(text,str):raise ValueError('GRADING_JSON_INVALID')
    cleaned=re.sub(r'^```json\s*|\s*```$','',text.strip())
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('DUPLICATE_GRADING_KEY')
            out[k]=v
        return out
    def constant(_):raise ValueError('NONFINITE_GRADING_JSON')
    value=json.loads(cleaned,object_pairs_hook=pairs,parse_constant=constant)
    if (not isinstance(value,dict) or set(value)!={'criteria_met','explanation'}
            or type(value['criteria_met']) is not bool or not isinstance(value['explanation'],str)):
        raise ValueError('GRADING_SCHEMA_INVALID')
    return value


def adapted_score(rubrics,grades):
    """Arithmetic counterpart tested against official function in Python 3.11.

    No clipping here. Incomplete sets never receive a score.
    """
    if len(rubrics)!=len(grades):raise ValueError('INCOMPLETE_GRADING')
    if any(type(g.get('criteria_met')) is not bool for g in grades):raise ValueError('INVALID_GRADE')
    denominator=sum(r['points'] for r in rubrics if r['points']>0)
    return sum(r['points'] for r,g in zip(rubrics,grades) if g['criteria_met'])/denominator if denominator else None


@dataclass
class Ledger:
    global_cap:int=80
    per_sample_cap:int=40
    reserved_generation:int=0
    grading_requests:int=0
    identity_requests:int=0

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.global_cap,self.per_sample_cap,self.reserved_generation))
                or not 1<=self.global_cap<=80 or self.per_sample_cap<1 or not 0<=self.reserved_generation<=self.global_cap):raise ValueError('INVALID_BUDGET')
        self.per_sample={}

    def reserve(self,sample):
        # One generation slot per sample is reserved, never reported as a real call.
        if self.reserved_generation+self.grading_requests>=self.global_cap:raise RuntimeError('GLOBAL_REQUEST_BUDGET_EXHAUSTED')
        if 1+self.per_sample.get(sample,0)>=self.per_sample_cap:raise RuntimeError('PER_SAMPLE_BUDGET_EXHAUSTED')
        self.grading_requests+=1;self.per_sample[sample]=self.per_sample.get(sample,0)+1


class TestOnlyGrader:
    """Offline protocol marker. No production transport may impersonate it."""
    test_only=True
    def identity(self):raise NotImplementedError()
    def grade(self,messages,*,model,timeout):raise NotImplementedError()


def deterministic_run_id(plan):return digest(plan)[:32]


def run_grading(rows,completions,*,client=None,execute=False,allow_grader=False,test_only=False,
                max_attempts=1,global_cap=80,per_sample_cap=40,timeout=30,output_root=None):
    """Dry-run default. Real execution stays blocked until a reviewed binding exists.

    Retry means another unchanged rubric request; repair variants are not added.
    An injected TestOnlyGrader must itself honor the timeout argument, as must a
    future authorized SDK transport with retries=0. No background requests run.
    """
    if (any(type(v) is not bool for v in (execute,allow_grader,test_only)) or type(max_attempts) is not int or not 1<=max_attempts<=3
            or type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<timeout<=120):raise ValueError('INVALID_EXECUTION_OPTIONS')
    if not 1<=len(rows)<=3 or len(completions)!=len(rows):raise ValueError('MAX_THREE_COMPLETE_SAMPLES_REQUIRED')
    for r in rows:messages_for(r)
    if any(not isinstance(c,str) or not c.strip() for c in completions):raise ValueError('COMPLETION_REQUIRED')
    counts=[len(r['rubrics']) for r in rows];limits=budget(len(rows),sum(counts),grader_retries=max_attempts-1,total_cap=global_cap)
    ledger=Ledger(global_cap,per_sample_cap,len(rows))
    plan={'schema_version':'locdi.D6.bounded_grader_plan.v1','evaluation_kind':'ADAPTED_PROTOCOL',
        'input_fingerprints':[digest(messages_for(r)) for r in rows],'completion_fingerprints':[digest(c) for c in completions],
        'rubric_fingerprints':[digest(r['rubrics']) for r in rows],'rubrics_per_sample':counts,
        'prompt_fingerprint':digest(official_template()),'grader_binding':deepcopy(GRADER_CONFIG),
        'budget':limits,'per_sample_cap':per_sample_cap,'timeout_seconds':timeout,'max_attempts':max_attempts,
        'generation_approval_independent':True,'sdk_retries':0,'repair_requests':0}
    run_id=deterministic_run_id(plan)
    result={'schema_version':'locdi.D6.bounded_grader_result.v1','run_id':run_id,'status':'DRY_RUN','plan':plan,
        'test_only':test_only,'actual_generation_calls':0,'actual_grader_calls':0,'fake_grading_calls':0,
        'identity_check_requests':0,'rubric_count':sum(counts),'rubrics_recorded':0,'records':[],
        'official_healthbench_score':None,'clinical_gold_generated':0,'research_evaluation_ready':False,'blocker_codes':[]}
    if not limits['within_cap'] or any(1+n*max_attempts>per_sample_cap for n in counts):
        result.update(status='BLOCKED',blocker_codes=['PLANNED_BUDGET_EXCEEDS_CAP']);return result
    if not execute:return result
    if not test_only or allow_grader or not isinstance(client,TestOnlyGrader):
        result.update(status='BLOCKED',blocker_codes=['REAL_GRADER_BINDING_QUALITY_AND_APPROVAL_REQUIRED']);return result
    destination=None
    if output_root is not None:
        destination=(ROOT/'outputs/private/D6_grader_test_only'/run_id) if output_root is True else output_root
        from pathlib import Path
        destination=Path(destination).resolve()
        if not destination.is_relative_to((ROOT/'outputs/private').resolve()):raise ValueError('PRIVATE_OUTPUT_REQUIRED')
        destination.mkdir(parents=True,exist_ok=False)
        atomic_rows(destination,'started.json',{'plan':plan,'test_only':True},jsonl=False)
    try:
        ledger.identity_requests+=1
        identity=client.identity()
        if identity!={'model':'TEST_ONLY_GRADER','version':'TEST_ONLY_v1','digest':'TEST_ONLY_NO_MODEL_ARTIFACT'}:raise RuntimeError('GRADER_IDENTITY_MISMATCH')
        for sample,(row,completion) in enumerate(zip(rows,completions),1):
            for rubric_index,rubric in enumerate(row['rubrics']):
                record={'sample_ref':'Sample{:03d}'.format(sample),'rubric_ordinal':rubric_index+1,
                    'rubric_fingerprint':digest(rubric),'points':rubric['points'],'status':'UNATTEMPTED','attempts':[]}
                result['records'].append(record)
                for attempt in range(1,max_attempts+1):
                    ledger.reserve(sample)
                    item={'attempt':attempt,'status':'STARTED'};record['attempts'].append(item)
                    if destination:atomic_rows(destination,'s{:03d}_r{:03d}_a{:02d}_before.json'.format(sample,rubric_index,attempt),item,jsonl=False)
                    try:
                        result['fake_grading_calls']+=1
                        raw=client.grade(grader_messages(row,completion,rubric),model='TEST_ONLY_GRADER',timeout=timeout)
                        grade=parse_grade(raw);item['status']='SCHEMA_VALID_TEST_ONLY'
                        record['status']='TEST_ONLY_RECORDED';result['rubrics_recorded']+=1
                        # Synthetic labels never become clinical judgments or real scores.
                        if destination:atomic_rows(destination,'s{:03d}_r{:03d}_a{:02d}_after.json'.format(sample,rubric_index,attempt),{'status':item['status'],'synthetic_protocol_grade':grade,'test_only':True},jsonl=False)
                        break
                    except Exception as exc:
                        if isinstance(exc,OSError) and not isinstance(exc,(TimeoutError,ConnectionError)):
                            raise RuntimeError('GRADER_STORAGE_FAILED') from None
                        item['status']='TIMEOUT' if isinstance(exc,TimeoutError) or 'timeout' in type(exc).__name__.lower() else 'INVALID_JSON_OR_SCHEMA' if isinstance(exc,ValueError) else 'MODEL_OR_STORAGE_ERROR'
                        if attempt==max_attempts:
                            record['status']='FAILED';raise RuntimeError('RUBRIC_MAX_ATTEMPTS_REACHED') from None
        result['status']='TEST_ONLY_COMPLETED'
    except Exception as exc:
        safe={'GLOBAL_REQUEST_BUDGET_EXHAUSTED','PER_SAMPLE_BUDGET_EXHAUSTED','GRADER_IDENTITY_MISMATCH','RUBRIC_MAX_ATTEMPTS_REACHED','GRADER_STORAGE_FAILED'}
        result['status']='PARTIAL' if result['rubrics_recorded'] else 'FAILED'
        result['blocker_codes']=[str(exc) if str(exc) in safe else 'GRADER_EXECUTION_OR_STORAGE_FAILED']
    finally:
        result['identity_check_requests']=ledger.identity_requests
        result['grading_budget_reservations']=ledger.grading_requests
    result['unrecorded_rubrics']=result['rubric_count']-result['rubrics_recorded']
    if destination:atomic_rows(destination,'result.json',result,jsonl=False)
    return result
