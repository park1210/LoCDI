"""C4-27 fixed remaining15; immutable compact scorer/transport from C4-26."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import re
import time
from app.locdi.answer_grounded_live_collection import (ROOT,MODEL,DIGEST,TOKENIZER,TREE,sha,atomic_json,
    load_bundle as load_parent,OllamaHTTP,server_state,selection_potential)
from app.locdi.answer_grounded_compact import (VERSION,PROMPT_VERSION,PROMPT,schema,parse,score,
    compact_binding,repetition,REPETITION_RULE)
from app.locdi.answer_grounded_compact_collection import request_payload
from app.locdi.answer_grounded_routing import model_input,validate_pair
from app.locdi.answer_grounded_routing_evaluation import evaluate,rate
from app.locdi.llm_setup import digest,local_endpoint
from app.locdi.reference_acquisition import strict_json
from app.locdi.external_reference_readiness import pinned_private_json

NAMESPACE='c4_27_v1'
PREVIOUS=ROOT/'outputs/private/research_evaluation/c4_26_v1'
CODE=['app/locdi/answer_grounded_compact_extension.py','scripts/collect_compact_answer_grounded_extension.py',
    'tests/locdi/test_answer_grounded_compact_extension.py']


def build_bundle(parent):
    qs=deepcopy(parent['questions'][2:5]);refs=deepcopy(parent['references'][2:5])
    if len(qs)!=3 or len({q['question_id'] for q in qs})!=3:raise ValueError('EXACT_REMAINING_THREE_REQUIRED')
    bindings=[compact_binding(a) for a in parent['bindings']]
    if len(bindings)!=5:raise ValueError('FIVE_BINDINGS_REQUIRED')
    tasks=[]
    for q,r in zip(qs,refs):
        validate_pair(q,r)
        if r['answer_type']!='exact_text' or model_input(q)['options']:raise ValueError('FREE_TEXT_ONLY')
        for a in bindings:tasks.append({'question_id':q['question_id'],'agent_id':a['agent_id'],
            'input_fingerprint':digest(model_input(q)),'schema_fingerprint':digest(schema()),
            'prompt_fingerprint':a['benchmark_prompt_fingerprint'],'model_identifier':a['model_identifier']})
    plan={'version':'locdi.compact.extension_plan.c4_27.v1','task_version':VERSION,'prompt_version':PROMPT_VERSION,
        'run_id':'c4_27_compact_extension_001','namespace':NAMESPACE,'parent_plan_fingerprint':parent['plan']['fingerprint'],
        'max_generations':15,'settings':deepcopy(parent['plan']['settings']),'tasks':tasks,
        'bindings_fingerprint':digest(bindings),'question_reference_fingerprint':digest(list(zip(qs,refs))),
        'task_order_fingerprint':digest(tasks),'selection_rule':'Original parent questions3,4,5; original five agent order',
        'development_only':True,'resume':False}
    plan['fingerprint']=digest(plan)
    return {'plan':plan,'questions':qs,'references':refs,'bindings':bindings,'source_pins':deepcopy(parent['source_pins'])}


def load_bundle():
    parent,config=load_parent();b=build_bundle(parent)
    old=strict_json((PREVIOUS/'mini_pilot_plan.json').read_text(encoding='utf-8'))
    if old['questions']!=parent['questions'][:2] or old['references']!=parent['references'][:2] or old['bindings']!=b['bindings']:
        raise ValueError('C4_26_PARENT_OR_BINDING_CHANGED')
    prior=strict_json((ROOT/'outputs/research_evaluation/c4_26_v1/mini_pilot_protocol.json').read_text(encoding='utf-8'))
    if prior['schema_fingerprint']!=digest(schema()) or prior['task_prompt_fingerprint']!=digest(PROMPT):raise ValueError('COMPACT_CONTRACT_CHANGED')
    for n,h in {**prior['implementation_hashes'],**prior['reused_code_hashes']}.items():
        if sha(ROOT/n)!=h:raise ValueError('SEALED_COMPACT_CODE_CHANGED')
    return b,config


def prior_audit():
    """Mechanical source/payload audit; never infer why an agent abstained."""
    from app.locdi.answer_grounded_routing import read_agentclinic
    parent,config=load_parent()
    source=ROOT/'outputs/private/research_evaluation/c3_10_v1/upstream_snapshot/agentclinic_medqa.jsonl'
    pairs,_=read_agentclinic(source);lookup={q['question_id']:(q,r) for _,q,r in pairs}
    run=PREVIOUS/'live/c4_26_compact_pilot_001';manifest=strict_json((run/'completion_manifest.json').read_text(encoding='utf-8'))
    if manifest['fingerprint']!=digest(manifest['files']) or any(sha(run/n)!=h for n,h in manifest['files'].items()):raise ValueError('PRIOR_RUN_INTEGRITY')
    rows=[];errors=[]
    for ordinal,q in enumerate(parent['questions'],1):
        original,ref=lookup[q['question_id']];fields=strict_json(model_input(q)['question'])
        expected={'Objective_for_Doctor','Patient_Actor','Physical_Examination_Findings','Test_Results'}
        ok=q==original and set(fields)==expected and all(fields.get(k) not in (None,'',[],{}) for k in expected)
        hidden='Correct_Diagnosis' in model_input(q)['question']
        if not ok or hidden:errors.append('SOURCE_FIELD_OR_HIDDEN_KEY_ERROR')
        rows.append({'question_ordinal':ordinal,'source_input_equality':q==original,'required_fields_present':set(fields)==expected,
            'empty_required_fields':[k for k in expected if fields.get(k) in (None,'',[],{})],
            'hidden_reference_key_in_input':hidden,'input_fingerprint':q['input_fingerprint']})
    old=strict_json((PREVIOUS/'mini_pilot_plan.json').read_text(encoding='utf-8'))
    for i,t in enumerate(old['plan']['tasks'],1):
        before=strict_json((run/('Request%03d_before.json'%i)).read_text(encoding='utf-8'))
        after=strict_json((run/('Request%03d_after.json'%i)).read_text(encoding='utf-8'))
        q=next(q for q in parent['questions'] if q['question_id']==t['question_id']);a=next(a for a in old['bindings'] if a['agent_id']==t['agent_id'])
        user=strict_json(before['payload']['messages'][1]['content'])
        if user!=model_input(q) or before['payload']['messages'][0]['content']!=a['system_prompt'] or before['payload']['format']!=schema():errors.append('PRIOR_PAYLOAD_BINDING_ERROR')
        if after['server_done_reason']!='stop' or after['server_token_counts']['prompt_eval_count']!=before['planned_input_tokens']:errors.append('PRIOR_TRUNCATION_OR_TOKEN_COUNT_ERROR')
        if after['server_after']['loaded_model']['context_length']!=32768:errors.append('PRIOR_CONTEXT_ERROR')
    return {'status':'PASS' if not errors else 'BLOCKED','blocker_codes':sorted(set(errors)),'rows':rows,
        'prior_request_count_checked':10,'saved_payload_complete':not errors,
        'internal_model_abstention_cause':'NOT_OBSERVABLE_WITHOUT_RATIONALE','clinical_review':'NOT_PERFORMED'}


def protocol(bundle,config):
    p=bundle['plan'];prior=strict_json((ROOT/'outputs/research_evaluation/c4_26_v1/mini_pilot_protocol.json').read_text(encoding='utf-8'))
    value={'version':'locdi.compact.extension_live.c4_27.v1','task_version':VERSION,'prompt_version':PROMPT_VERSION,
        'parent_plan_fingerprint':p['parent_plan_fingerprint'],'c4_26_protocol_fingerprint':prior['fingerprint'],
        'extension_plan_fingerprint':p['fingerprint'],'task_order_fingerprint':p['task_order_fingerprint'],
        'source_sha256':prior['source_sha256'],'model':MODEL,'model_digest':DIGEST,'model_version':config['model_version'],
        'schema_fingerprint':digest(schema()),'task_prompt_fingerprint':digest(PROMPT),'settings':p['settings'],
        'namespace':NAMESPACE,'run_id':p['run_id'],'max_generations':15,'endpoint':local_endpoint(config['endpoint'])[:-3],
        'context_margin':512,'input_bytes_max':65536,'tokenizer_tree_sha256':TREE,
        'retry':0,'repair':0,'warmup':0,'resume':False,'first_technical_error_stops':True,
        'first_fixed_task_load':'Explicit new approval may permit unloaded model to load32768 on question3/cardiology; verify api/ps synchronously before next task',
        'repetition_rule':REPETITION_RULE,'implementation_hashes':{n:sha(ROOT/n) for n in CODE},
        'compact_code_hashes':prior['implementation_hashes'],'reused_code_hashes':prior['reused_code_hashes']}
    value['fingerprint']=digest(value);return value


class ExtensionApproval:
    def __init__(self,path,expected_sha):self.path,self.expected_sha=path,expected_sha
    def verify(self,ep,now=None):
        a=pinned_private_json(self.path,self.expected_sha);now=now or datetime.now(timezone.utc)
        fields={'scope':'LIVE_COMPACT_EXTENSION_REMAINING_15_ONLY','authorization_source':'EXPLICIT_USER_STATEMENT',
            'operator_reference':'USER_LOCAL_SERVER_OPERATOR','execution_protocol_fingerprint':ep['fingerprint'],
            'parent_plan_fingerprint':ep['parent_plan_fingerprint'],'c4_26_protocol_fingerprint':ep['c4_26_protocol_fingerprint'],
            'extension_plan_fingerprint':ep['extension_plan_fingerprint'],'task_order_fingerprint':ep['task_order_fingerprint'],
            'source_sha256':ep['source_sha256'],'model_digest':DIGEST,'task_version':VERSION,'prompt_version':PROMPT_VERSION,
            'schema_fingerprint':ep['schema_fingerprint'],'endpoint':ep['endpoint'],'max_generations':15,
            'run_id':ep['run_id'],'namespace':NAMESPACE,'retry':0,'repair':0,'warmup':0,'allow_first_fixed_task_context_load':True}
        if any(a.get(k)!=v for k,v in fields.items()) or not a.get('user_statement'):raise ValueError('EXTENSION_APPROVAL_SCOPE')
        issued=datetime.fromisoformat(a['recorded_at']);expiry=datetime.fromisoformat(a['expires_at'])
        if issued.tzinfo is None or expiry.tzinfo is None or not issued<=now<expiry or not 0<(expiry-issued).total_seconds()<=86400:raise ValueError('EXTENSION_APPROVAL_VALIDITY')
        return {'approval_sha256':self.expected_sha,'scope':fields['scope']}


def validate(bundle,config,ep,test_only=False):
    p=bundle['plan']
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('EXTENSION_PLAN_FINGERPRINT')
    if ep['fingerprint']!=digest({k:v for k,v in ep.items() if k!='fingerprint'}):raise ValueError('EXTENSION_PROTOCOL_FINGERPRINT')
    if ep['extension_plan_fingerprint']!=p['fingerprint'] or ep['max_generations']!=15 or len(p['tasks'])!=15:raise ValueError('EXTENSION_PLAN_BINDING')
    if ep['namespace']!=NAMESPACE or ep['run_id']!='c4_27_compact_extension_001':raise ValueError('EXTENSION_NAMESPACE')
    if not test_only:
        fresh,cfg=load_bundle()
        if fresh!=bundle or cfg!=config or protocol(bundle,config)!=ep:raise ValueError('CURRENT_EXTENSION_CODE_OR_INPUT_CHANGED')
        if prior_audit()['status']!='PASS':raise ValueError('PRIOR_SOURCE_OR_CONTRACT_BLOCKED')


def run_collection(bundle,config,ep,root,*,execute=False,allow_llm=False,approval=None,
                   counter=None,transport_factory=None,test_only=False):
    validate(bundle,config,ep,test_only)
    if not execute:return {'mode':'DRY_RUN','registered':15,'generation_requests':0,'client_constructed':False}
    if not allow_llm or type(approval) is not ExtensionApproval:raise ValueError('NEW_EXTENSION_APPROVAL_REQUIRED')
    proof=approval.verify(ep)
    if not test_only:
        if counter is not None or transport_factory is not None:raise ValueError('LIVE_INJECTION_FORBIDDEN')
        from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter
        counter=LocalTokenCounter(TOKENIZER,TREE)
    if counter is None:raise ValueError('TOKEN_COUNTER_REQUIRED')
    root=Path(root).resolve()
    if not test_only and root!=(ROOT/'outputs/private/research_evaluation'/NAMESPACE/'live'/ep['run_id']).resolve():raise ValueError('EXTENSION_EXECUTION_PATH')
    root.mkdir(parents=True,exist_ok=False);atomic_json(root/'started.json',{'protocol_fingerprint':ep['fingerprint'],'approval':proof})
    qi={q['question_id']:q for q in bundle['questions']};ri={r['question_id']:r for r in bundle['references']};ai={a['agent_id']:a for a in bundle['bindings']}
    transport=None;rows=[];records=[];attempts=0;identities=0;stopped=None
    try:
        transport=transport_factory() if test_only else OllamaHTTP(ep['endpoint'])
        if test_only and (getattr(transport,'TEST_ONLY',False) is not True or type(transport).__module__.startswith('app.')):raise ValueError('TEST_ONLY_TRANSPORT_REQUIRED')
        atomic_json(root/'server_before.json',server_state(transport,ep));identities+=1
        for i,t in enumerate(bundle['plan']['tasks'],1):
            if stopped:break
            if attempts>=15:raise ValueError('GLOBAL_BUDGET_15')
            approval.verify(ep)
            if not test_only:validate(bundle,config,ep)
            q=qi[t['question_id']];ref=ri[t['question_id']];a=ai[t['agent_id']]
            payload,n=request_payload(q,a,ep,counter)
            if digest(model_input(q))!=t['input_fingerprint'] or digest(payload['format'])!=t['schema_fingerprint'] or digest(a['system_prompt'])!=t['prompt_fingerprint']:raise ValueError('TASK_HASH_MISMATCH')
            before=server_state(transport,ep);identities+=1
            if attempts and before['loaded_context_status']!='VERIFIED':raise ValueError('CONTEXT_LOST_BETWEEN_TASKS')
            rec={'registered_task_index':i,'parent_task_index':i+10,'question_id':q['question_id'],'agent_id':a['agent_id'],'specialty':a['department'],
                'model_tag':MODEL,'model_digest':DIGEST,'prompt_fingerprint':t['prompt_fingerprint'],'input_fingerprint':t['input_fingerprint'],
                'schema_fingerprint':t['schema_fingerprint'],'request_fingerprint':digest(payload),'requested_context':32768,'requested_output_limit':1024,
                'planned_input_tokens':n,'server_before':before,'generation_budget_consumed':True,'status':'REQUEST_STARTING','http_status':None,
                'json_valid':False,'schema_valid':False,'response_fingerprint':None,'raw_wire_status':'HTTP_ENTITY_BYTES_NOT_TCP_WIRE'}
            atomic_json(root/('Request%03d_before.json'%i),{**rec,'payload':payload})
            attempts+=1;start=time.monotonic();result=None
            try:
                reply=transport.generate(payload);raw=reply['body'];rec['http_status']=reply['http_status']
                with (root/('Request%03d_response_body.bin'%i)).open('xb') as f:f.write(raw)
                rec['http_entity_sha256']=hashlib.sha256(raw).hexdigest()
                if reply['http_status']!=200:raise ValueError('GENERATION_HTTP_ERROR')
                envelope=strict_json(raw.decode('utf-8'));rec['parsed_envelope']=envelope
                if envelope.get('model')!=MODEL or envelope.get('done') is not True:raise ValueError('RESPONSE_MODEL_OR_DONE')
                content=envelope.get('message',{}).get('content')
                if not isinstance(content,str):raise ValueError('RESPONSE_CONTENT')
                rec.update(raw_model_content=content,response_fingerprint=digest(content),repetition=repetition(content),
                    server_done_reason=envelope.get('done_reason'),server_token_counts={k:envelope.get(k) for k in
                        ('prompt_eval_count','eval_count','total_duration','load_duration','prompt_eval_duration','eval_duration')})
                try:rec['parsed_response']=strict_json(content);rec['json_valid']=True
                except (ValueError,TypeError):rec['parsed_response']=None
                result=score(q,ref,content);rec['schema_valid']=result['state']!='INVALID_RESPONSE'
                if type(envelope.get('eval_count')) is int and envelope['eval_count']>1024:raise ValueError('SERVER_OUTPUT_TOKEN_BUDGET_EXCEEDED')
                if type(envelope.get('prompt_eval_count')) is int and envelope['prompt_eval_count']+1024+512>32768:raise ValueError('SERVER_INPUT_TOKEN_BUDGET_EXCEEDED')
                if envelope.get('done_reason')=='length':result.update(state='INVALID_RESPONSE',answer_correct=None);rec['schema_valid']=False;stopped='OUTPUT_LIMIT_TRUNCATION'
                after=server_state(transport,ep);identities+=1;rec['server_after']=after
                if after['loaded_context_status']!='VERIFIED':raise ValueError('POST_REQUEST_CONTEXT_NOT_VERIFIED')
                if result['state']=='INVALID_RESPONSE':stopped=stopped or 'INVALID_RESPONSE'
            except Exception as exc:
                stopped=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z0-9_]+',str(exc)) else type(exc).__name__
                rec.update(error_type=type(exc).__name__,error_code=stopped)
                if result is None:result=score(q,ref,error_type=type(exc).__name__)
            rec.update(scoring=result,status='STOPPED' if stopped else 'RECORDED',wall_seconds=time.monotonic()-start,completed_at=datetime.now(timezone.utc).isoformat())
            records.append(rec);rows.append({**result,'agent_id':a['agent_id']});atomic_json(root/('Request%03d_after.json'%i),rec)
    except Exception as exc:stopped=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z0-9_]+',str(exc)) else type(exc).__name__
    finally:
        if transport is not None:
            transport.close();atomic_json(root/'metadata_log.json',{'responses':getattr(transport,'metadata_log',[]),'raw_wire_saved':False})
    seen={(r['question_id'],r['agent_id']) for r in rows}
    for t in bundle['plan']['tasks']:
        if (t['question_id'],t['agent_id']) not in seen:rows.append({**score(qi[t['question_id']],ri[t['question_id']],attempted=False),'agent_id':t['agent_id']})
    result={'status':'COMPLETE' if attempts==15 and not stopped else ('PARTIAL' if attempts else 'PREFLIGHT_BLOCKED'),
        'registered':15,'attempted':attempts,'stop_reason':stopped,'identity_checks':identities,
        'metadata_requests':getattr(transport,'metadata_requests',0),'actual_llm_generations':0 if test_only else attempts,
        'test_only':test_only,'rows':rows,'evaluation':evaluate(bundle['questions'],bundle['bindings'],rows)}
    atomic_json(root/'collection.json',result);files={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()}
    atomic_json(root/'completion_manifest.json',{'files':files,'fingerprint':digest(files),'protocol_fingerprint':ep['fingerprint']})
    return result


def combine(parent,old_bundle,old_ep,old_rows,new_bundle,new_ep,new_rows):
    """Contract-gated aggregation only; never rewrite either independent run."""
    checks=(old_bundle['questions']==parent['questions'][:2],new_bundle['questions']==parent['questions'][2:],
        old_bundle['references']==parent['references'][:2],new_bundle['references']==parent['references'][2:],
        old_bundle['bindings']==new_bundle['bindings'],
        old_ep['parent_plan_fingerprint']==new_ep['parent_plan_fingerprint']==parent['plan']['fingerprint'],
        old_ep['plan_fingerprint']==old_bundle['plan']['fingerprint'],new_ep['extension_plan_fingerprint']==new_bundle['plan']['fingerprint'],
        old_ep.get('task_version')==new_ep.get('task_version')==VERSION,
        old_ep.get('prompt_version')==new_ep.get('prompt_version')==PROMPT_VERSION,
        old_ep.get('schema_fingerprint')==new_ep.get('schema_fingerprint')==digest(schema()))
    fields=['task_version','prompt_version','schema_fingerprint','task_prompt_fingerprint','model','model_digest','model_version',
        'settings','endpoint','tokenizer_tree_sha256','retry','repair','warmup','source_sha256']
    if not all(checks) or any(old_ep.get(k)!=new_ep.get(k) for k in fields):raise ValueError('INCOMPATIBLE_COMPACT_RUNS')
    if old_ep['fingerprint']!=new_ep['c4_26_protocol_fingerprint']:raise ValueError('PRIOR_PROTOCOL_PIN')
    if len(old_rows)!=10 or len(new_rows)!=15:raise ValueError('REGISTERED_ROWS_REQUIRED')
    rows=deepcopy(old_rows+new_rows)
    if len({(r['question_id'],r['agent_id']) for r in rows})!=25:raise ValueError('DUPLICATE_COMBINED_TASK')
    # evaluate rejects unregistered IDs and inconsistent correctness semantics.
    ev=evaluate(parent['questions'],old_bundle['bindings'],rows)
    return {'rows':rows,'evaluation':ev,'potential':selection_potential({'questions':parent['questions'],
        'bindings':old_bundle['bindings']},rows),'contract_compatible':True}
