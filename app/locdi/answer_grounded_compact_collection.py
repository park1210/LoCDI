"""Compact orchestration using the sealed C4-25 transport/checkpoint primitives."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import time
from app.locdi.answer_grounded_live_collection import (ROOT,MODEL,DIGEST,TOKENIZER,TREE,
    OllamaHTTP,server_state,atomic_json,sha,load_bundle as load_parent,request_payload as legacy_payload,
    selection_potential)
from app.locdi.external_reference_readiness import pinned_private_json
from app.locdi.llm_setup import digest,local_endpoint
from app.locdi.reference_acquisition import strict_json
from app.locdi.answer_grounded_routing import model_input
from app.locdi.answer_grounded_routing_evaluation import evaluate,rate
from app.locdi.answer_grounded_compact import (VERSION,PROMPT_VERSION,PROMPT,REPETITION_RULE,
    build_bundle,schema,score,parse,repetition)

NAMESPACE='c4_26_v1'
CODE=['app/locdi/answer_grounded_compact.py','app/locdi/answer_grounded_compact_collection.py',
    'scripts/collect_compact_answer_grounded.py','tests/locdi/test_answer_grounded_compact.py']


def load_bundle():
    parent,config=load_parent()
    return build_bundle(parent),config


def protocol(bundle,config):
    p=bundle['plan']
    value={'version':'locdi.compact.live.c4_26.v1','task_version':VERSION,'prompt_version':PROMPT_VERSION,
        'plan_fingerprint':p['fingerprint'],'parent_plan_fingerprint':p['parent_plan_fingerprint'],
        'source_sha256':sha(ROOT/'outputs/private/research_evaluation/c3_10_v1/upstream_snapshot/agentclinic_medqa.jsonl'),
        'run_id':p['run_id'],'namespace':NAMESPACE,'model':MODEL,'model_digest':DIGEST,'model_version':config['model_version'],
        'endpoint':local_endpoint(config['endpoint'])[:-3],'max_generations':10,'settings':p['settings'],
        'task_prompt_fingerprint':digest(PROMPT),'schema_fingerprint':digest(schema()),
        'transport':'REUSE_C4_25_OLLAMA_NATIVE_API_CHAT_STREAM_FALSE','context_margin':512,'input_bytes_max':65536,
        'tokenizer_tree_sha256':TREE,'context_policy':'First fixed task may load32768 only if explicitly authorized; post-context must verify before next request',
        'repetition_rule':REPETITION_RULE,'retry':0,'repair':0,'warmup':0,'resume':False,
        'first_technical_error_stops':True,'group2_requires_group1_schema_valid':True,
        'metrics':'Registered10; attempted requests; complete paired questions separately; zero denominator null',
        'implementation_hashes':{n:sha(ROOT/n) for n in CODE},
        'reused_code_hashes':{n:sha(ROOT/n) for n in ('app/locdi/answer_grounded_live_collection.py',
            'app/locdi/answer_grounded_routing.py','app/locdi/answer_grounded_routing_evaluation.py')}}
    value['fingerprint']=digest(value);return value


class CompactApproval:
    def __init__(self,path,expected_sha):self.path,self.expected_sha=path,expected_sha
    def verify(self,ep,now=None):
        a=pinned_private_json(self.path,self.expected_sha);now=now or datetime.now(timezone.utc)
        fields={'scope':'LIVE_COMPACT_DIAGNOSTIC_MINI_PILOT_10_ONLY','authorization_source':'EXPLICIT_USER_STATEMENT',
            'operator_reference':'USER_LOCAL_SERVER_OPERATOR','execution_protocol_fingerprint':ep['fingerprint'],
            'pilot_plan_fingerprint':ep['plan_fingerprint'],'parent_plan_fingerprint':ep['parent_plan_fingerprint'],
            'run_id':ep['run_id'],'namespace':NAMESPACE,'source_sha256':ep['source_sha256'],
            'task_version':VERSION,'prompt_version':PROMPT_VERSION,'schema_fingerprint':ep['schema_fingerprint'],
            'model_digest':DIGEST,'endpoint':ep['endpoint'],'max_generations':10,'retry':0,'repair':0,'warmup':0,
            'allow_first_fixed_task_context_load':True}
        if any(a.get(k)!=v for k,v in fields.items()) or not a.get('user_statement'):raise ValueError('COMPACT_APPROVAL_SCOPE')
        issued=datetime.fromisoformat(a['recorded_at']);expiry=datetime.fromisoformat(a['expires_at'])
        if issued.tzinfo is None or expiry.tzinfo is None or not issued<=now<expiry or not 0<(expiry-issued).total_seconds()<=86400:
            raise ValueError('COMPACT_APPROVAL_VALIDITY')
        return {'approval_sha256':self.expected_sha,'scope':fields['scope'],'source':'EXPLICIT_USER_STATEMENT'}


def request_payload(q,a,ep,counter):
    # Reuse original input/token/context/options logic; send ONLY the new schema.
    payload,n=legacy_payload(q,a,ep,counter);payload['format']=schema()
    return payload,n


def validate(bundle,config,ep,test_only=False):
    p=bundle['plan']
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('COMPACT_PLAN_FINGERPRINT')
    if ep['fingerprint']!=digest({k:v for k,v in ep.items() if k!='fingerprint'}):raise ValueError('COMPACT_PROTOCOL_FINGERPRINT')
    if ep['plan_fingerprint']!=p['fingerprint'] or ep['max_generations']!=10 or len(p['tasks'])!=10:raise ValueError('COMPACT_PLAN_BINDING')
    if ep['namespace']!=NAMESPACE or ep['run_id']!='c4_26_compact_pilot_001':raise ValueError('COMPACT_NAMESPACE')
    if not test_only:
        fresh,cfg=load_bundle()
        if fresh!=bundle or cfg!=config or protocol(bundle,config)!=ep:raise ValueError('CURRENT_COMPACT_CODE_OR_INPUT_CHANGED')


def run_collection(bundle,config,ep,root,*,execute=False,allow_llm=False,approval=None,
                   counter=None,transport_factory=None,test_only=False):
    validate(bundle,config,ep,test_only)
    if not execute:return {'mode':'DRY_RUN','registered':10,'generation_requests':0,'client_constructed':False}
    if not allow_llm or type(approval) is not CompactApproval:raise ValueError('NEW_COMPACT_APPROVAL_REQUIRED')
    approval_result=approval.verify(ep)
    if not test_only:
        if counter is not None or transport_factory is not None:raise ValueError('LIVE_INJECTION_FORBIDDEN')
        from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter
        counter=LocalTokenCounter(TOKENIZER,TREE)
    if counter is None:raise ValueError('TOKEN_COUNTER_REQUIRED')
    root=Path(root).resolve()
    if not test_only and root!=(ROOT/'outputs/private/research_evaluation'/NAMESPACE/'live'/ep['run_id']).resolve():raise ValueError('COMPACT_EXECUTION_PATH')
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'started.json',{'protocol_fingerprint':ep['fingerprint'],'approval':approval_result})
    qi={q['question_id']:q for q in bundle['questions']};ri={r['question_id']:r for r in bundle['references']};ai={a['agent_id']:a for a in bundle['bindings']}
    transport=None;records=[];rows=[];attempts=0;identities=0;stopped=None
    try:
        transport=transport_factory() if test_only else OllamaHTTP(ep['endpoint'])
        if test_only and (getattr(transport,'TEST_ONLY',False) is not True or type(transport).__module__.startswith('app.')):raise ValueError('TEST_ONLY_TRANSPORT_REQUIRED')
        atomic_json(root/'server_before.json',server_state(transport,ep));identities+=1
        for i,t in enumerate(bundle['plan']['tasks'],1):
            if stopped:break
            if attempts>=10:raise ValueError('GLOBAL_BUDGET_10')
            approval.verify(ep)
            if not test_only:
                validate(bundle,config,ep)
            q=qi[t['question_id']];r=ri[t['question_id']];a=ai[t['agent_id']]
            payload,n=request_payload(q,a,ep,counter)
            if (digest(model_input(q))!=t['input_fingerprint'] or digest(schema())!=t['schema_fingerprint']
                    or digest(a['system_prompt'])!=t['prompt_fingerprint']):raise ValueError('COMPACT_TASK_HASH')
            before=server_state(transport,ep);identities+=1
            if attempts and before['loaded_context_status']!='VERIFIED':raise ValueError('CONTEXT_LOST_BETWEEN_TASKS')
            rec={'registered_task_index':i,'question_id':q['question_id'],'agent_id':a['agent_id'],'specialty':a['department'],
                'model_tag':MODEL,'model_digest':DIGEST,'prompt_fingerprint':t['prompt_fingerprint'],
                'input_fingerprint':t['input_fingerprint'],'schema_fingerprint':t['schema_fingerprint'],
                'request_fingerprint':digest(payload),'requested_context':32768,'requested_output_limit':1024,
                'planned_input_tokens':n,'server_before':before,'generation_budget_consumed':True,'status':'REQUEST_STARTING',
                'http_status':None,'json_valid':False,'schema_valid':False,'response_fingerprint':None,
                'raw_wire_status':'HTTP_ENTITY_BYTES_NOT_TCP_WIRE'}
            atomic_json(root/('Request%03d_before.json'%i),{**rec,'payload':payload})
            attempts+=1;start=time.monotonic();result=None
            try:
                reply=transport.generate(payload);raw=reply['body'];rec['http_status']=reply['http_status']
                with (root/('Request%03d_response_body.bin'%i)).open('xb') as f:f.write(raw)
                rec['http_entity_sha256']=hashlib.sha256(raw).hexdigest()
                if reply['http_status']!=200:raise ValueError('GENERATION_HTTP_ERROR')
                envelope=strict_json(raw.decode('utf-8'));rec['parsed_envelope']=envelope
                if envelope.get('model')!=MODEL or envelope.get('done') is not True:raise ValueError('RESPONSE_MODEL_OR_DONE')
                text=envelope.get('message',{}).get('content')
                if not isinstance(text,str):raise ValueError('RESPONSE_CONTENT')
                rec.update(raw_model_content=text,response_fingerprint=digest(text),repetition=repetition(text),
                    server_done_reason=envelope.get('done_reason'),server_token_counts={k:envelope.get(k) for k in
                        ('prompt_eval_count','eval_count','total_duration','load_duration','prompt_eval_duration','eval_duration')})
                try:rec['parsed_response']=strict_json(text);rec['json_valid']=True
                except (ValueError,TypeError):rec['parsed_response']=None
                result=score(q,r,text);rec['schema_valid']=result['state']!='INVALID_RESPONSE'
                if type(envelope.get('eval_count')) is int and envelope['eval_count']>1024:raise ValueError('SERVER_OUTPUT_TOKEN_BUDGET_EXCEEDED')
                if type(envelope.get('prompt_eval_count')) is int and envelope['prompt_eval_count']+1024+512>32768:raise ValueError('SERVER_INPUT_TOKEN_BUDGET_EXCEEDED')
                if envelope.get('done_reason')=='length':
                    result.update(state='INVALID_RESPONSE',answer_correct=None);rec['schema_valid']=False;stopped='OUTPUT_LIMIT_TRUNCATION'
                after=server_state(transport,ep);identities+=1;rec['server_after']=after
                if after['loaded_context_status']!='VERIFIED':raise ValueError('POST_REQUEST_CONTEXT_NOT_VERIFIED')
                if result['state']=='INVALID_RESPONSE':stopped=stopped or 'INVALID_RESPONSE'
            except Exception as exc:
                stopped=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z0-9_]+',str(exc)) else type(exc).__name__
                rec.update(error_type=type(exc).__name__,error_code=stopped)
                if result is None:result=score(q,r,error_type=type(exc).__name__)
            rec.update(scoring=result,status='STOPPED' if stopped else 'RECORDED',wall_seconds=time.monotonic()-start,
                completed_at=datetime.now(timezone.utc).isoformat())
            records.append(rec);rows.append({**result,'agent_id':a['agent_id']})
            atomic_json(root/('Request%03d_after.json'%i),rec)
    except Exception as exc:
        stopped=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z0-9_]+',str(exc)) else type(exc).__name__
    finally:
        if transport is not None:
            transport.close();atomic_json(root/'metadata_log.json',{'responses':getattr(transport,'metadata_log',[]),'raw_wire_saved':False})
    seen={(r['question_id'],r['agent_id']) for r in rows}
    for t in bundle['plan']['tasks']:
        if (t['question_id'],t['agent_id']) not in seen:rows.append({**score(qi[t['question_id']],ri[t['question_id']],attempted=False),'agent_id':t['agent_id']})
    ev=evaluate(bundle['questions'],bundle['bindings'],rows)
    result={'status':'COMPLETE' if attempts==10 and not stopped else ('PARTIAL' if attempts else 'PREFLIGHT_BLOCKED'),
        'registered':10,'attempted':attempts,'stop_reason':stopped,'identity_checks':identities,
        'metadata_requests':getattr(transport,'metadata_requests',0),'actual_llm_generations':0 if test_only else attempts,
        'test_only':test_only,'rows':rows,'evaluation':ev}
    atomic_json(root/'collection.json',result)
    files={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()}
    atomic_json(root/'completion_manifest.json',{'files':files,'fingerprint':digest(files),'protocol_fingerprint':ep['fingerprint']})
    return result


def stability(bundle,collection,records):
    ev=evaluate(bundle['questions'],bundle['bindings'],collection['rows']);n=collection['attempted']
    counts={'http_success':sum(r['http_status']==200 for r in records),'json_valid':sum(r['json_valid'] for r in records),
        'schema_valid':sum(r['schema_valid'] for r in records),
        'output_length_failures':sum(r.get('server_done_reason')=='length' for r in records),
        'repetitive_output':sum(r.get('repetition',{}).get('repetitive',False) for r in records)}
    return {'registered':10,'attempted':n,'counts':{**counts,**ev['all_agents']['counts']},
        'registered_denominator_rates':{k:rate(v,10) for k,v in counts.items()},
        'attempted_denominator_rates':{k:rate(v,n) for k,v in counts.items()},
        'complete_collection_groups':ev['complete_collection_groups'],'complete_scorable_groups':ev['complete_scorable_groups'],
        'paired_denominator_questions':ev['complete_collection_groups'],'repetition_rule':REPETITION_RULE,
        'reported_prompt_tokens':sum(r.get('server_token_counts',{}).get('prompt_eval_count') or 0 for r in records),
        'reported_eval_tokens':sum(r.get('server_token_counts',{}).get('eval_count') or 0 for r in records),
        'wall_seconds':sum(r['wall_seconds'] for r in records),'all_slots_metrics':ev['all_agents']}
