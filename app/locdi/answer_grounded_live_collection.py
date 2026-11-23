"""Bounded live answer collection; frozen C4-24 data/scoring, no utility fitting."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import re

from app.locdi.llm_setup import ROOT,digest,local_endpoint,normalize_model_identifier
from app.locdi.reference_acquisition import strict_json
from app.locdi.external_reference_readiness import pinned_private_json
from app.locdi.answer_grounded_routing import (VERSION,TASK_PROMPT,model_input,answer_schema,
    validate_pair,bind_agents,build_plan)
from app.locdi.answer_grounded_routing_evaluation import evaluate,rate
from app.locdi.answer_grounded_routing import score
from app.locdi.agentclinic_benchmark_scoring import normalize
from app.llm.prompt_loader import specialty_system_prompt

NAMESPACE='c4_25_v1'
PARENT=ROOT/'outputs/private/research_evaluation/c4_24_v1'
MODEL='qwen2.5:1.5b-instruct'
DIGEST='sha256:65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b'
TREE='de5d71eac66d8658dca7a53895eea69ca7180def1de601c7dc80efd30a958ad3'
TOKENIZER=ROOT/'artifacts/external_tokenizers/c3_13c_qwen2_5_1_5b_989aa798_acquired/tokenizer'
CODE=['app/locdi/answer_grounded_live_collection.py','scripts/collect_answer_grounded_routing.py',
      'tests/locdi/test_answer_grounded_live_collection.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    """Exclusive UTF-8 publication; failure never replaces a previous result."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    import os,tempfile
    fd,tmp=tempfile.mkstemp(dir=str(path.parent),prefix='.pending_')
    try:
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        # Hard-link publication is atomic and refuses destination collisions.
        os.link(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def validate_bundle(bundle,config):
    p=bundle['plan'];questions=bundle['questions'];refs=bundle['references'];bindings=bundle['bindings']
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    if len(questions)!=5 or len(refs)!=5 or len(bindings)!=5 or len(p['tasks'])!=25:raise ValueError('FIXED_25_TASKS')
    if p['run_id']!='c4_24_pilot_001' or p['version']!=VERSION:raise ValueError('FROZEN_RUN_OR_VERSION')
    if digest(bindings)!=p['bindings_fingerprint'] or digest(list(zip(questions,refs)))!=p['question_reference_fingerprint']:
        raise ValueError('PLAN_CONTENT_BINDING')
    expected=build_plan(questions,refs,bindings,p['settings'],run_id=p['run_id'])
    if p!=expected:raise ValueError('TASK_ORDER_OR_SETTINGS')
    if config['model']!=MODEL or config['model_identifier']!=DIGEST:raise ValueError('MODEL_DIGEST_PIN')
    for q,r in zip(questions,refs):validate_pair(q,r)
    for a in bindings:
        if a['model']!=MODEL or a['model_identifier']!=DIGEST or a['version']!=config['model_version']:
            raise ValueError('AGENT_MODEL_BINDING')
        if digest(a['system_prompt'])!=a['benchmark_prompt_fingerprint']:raise ValueError('PROMPT_FINGERPRINT')
    return p


def load_bundle():
    bundle=strict_json((PARENT/'pilot_execution_plan.json').read_text(encoding='utf-8'))
    private=strict_json((PARENT/'agent_binding_inventory.json').read_text(encoding='utf-8'))
    config=private['llm_config'];validate_bundle(bundle,config)
    for name,h in bundle['source_pins'].items():
        if not Path(name).is_file() or sha(name)!=h:raise ValueError('SOURCE_OR_BINDING_CHANGED')
    registry=strict_json((ROOT/'configs/agent_registry.specialty_development.json').read_text(encoding='utf-8'))
    if bind_agents(registry,config,specialty_system_prompt)!=bundle['bindings']:raise ValueError('CURRENT_REGISTRY_PROMPT_BINDING')
    saved=strict_json((PARENT/'pilot_model_inputs.json').read_text(encoding='utf-8'))
    inputs=[]
    for q in bundle['questions']:
        for a in bundle['bindings']:
            inputs.append({'question_id':q['question_id'],'agent_id':a['agent_id'],'system':a['system_prompt'],
                'user':model_input(q),'schema':answer_schema(q),'input_fingerprint':q['input_fingerprint']})
    if inputs!=saved:raise ValueError('SAVED_25_INPUTS_CHANGED')
    return bundle,config


def protocol(bundle,config):
    p=validate_bundle(bundle,config)
    value={'version':'locdi.answer_grounded.live.c4_25.v1','task_version':VERSION,
        'parent_plan_fingerprint':p['fingerprint'],'parent_plan_sha256':sha(PARENT/'pilot_execution_plan.json'),
        'run_id':p['run_id'],'namespace':NAMESPACE,'source_sha256':sha(ROOT/'outputs/private/research_evaluation/c3_10_v1/upstream_snapshot/agentclinic_medqa.jsonl'),
        'task_prompt_version':'locdi.answer_grounded.task_prompt.v1','task_prompt_fingerprint':digest(TASK_PROMPT),
        'endpoint':local_endpoint(config['endpoint'])[:-3], 'transport':'OLLAMA_NATIVE_API_CHAT_STREAM_FALSE_JSON_SCHEMA',
        'model':MODEL,'model_version':config['model_version'],'model_digest':DIGEST,'settings':p['settings'],
        'max_generations':25,'context_policy':'REQUEST_NUM_CTX_32768; if unloaded first fixed task may load, then synchronous api/ps check before any second task',
        'context_margin':512,'tokenizer_tree_sha256':TREE,'input_bytes_max':65536,'response_bytes_max':1048576,
        'retry':0,'repair':0,'warmup':0,'server_restart':False,'parallel_requests':False,'resume':False,
        'implementation_hashes':{n:sha(ROOT/n) for n in CODE}}
    value['fingerprint']=digest(value);return value


class PinnedCollectionApproval:
    def __init__(self,path,expected_sha):self.path,self.expected_sha=path,expected_sha
    def verify(self,execution,now=None):
        a=pinned_private_json(self.path,self.expected_sha);now=now or datetime.now(timezone.utc)
        fields={'scope':'LIVE_ANSWER_GROUNDED_PILOT_25_ONLY','execution_protocol_fingerprint':execution['fingerprint'],
            'pilot_plan_fingerprint':execution['parent_plan_fingerprint'],'run_id':execution['run_id'],
            'namespace':execution['namespace'],'source_sha256':execution['source_sha256'],'task_version':execution['task_version'],
            'model_digest':execution['model_digest'],'task_prompt_version':execution['task_prompt_version'],
            'endpoint':execution['endpoint'],'max_generations':25,'retry':0,'repair':0,
            'authorization_source':'EXPLICIT_USER_STATEMENT','allow_first_fixed_task_context_load':True}
        if any(a.get(k)!=v for k,v in fields.items()) or not a.get('user_statement') or a.get('operator_reference')!='USER_LOCAL_SERVER_OPERATOR':
            raise ValueError('COLLECTION_APPROVAL_SCOPE')
        issued=datetime.fromisoformat(a['recorded_at']);expiry=datetime.fromisoformat(a['expires_at'])
        if issued.tzinfo is None or expiry.tzinfo is None or not issued<=now<expiry or (expiry-issued).total_seconds()>86400:
            raise ValueError('COLLECTION_APPROVAL_VALIDITY')
        return {'approval_sha256':self.expected_sha,'basis':'EXPLICIT_USER_STATEMENT_NOT_PROFESSOR_SIGNATURE'}


class OllamaHTTP:
    """Existing loopback/strict-JSON/httpx no-proxy, no-redirect, no-retry contract.

    Native /api/chat adds observable num_ctx and prompt/eval token counts. No SDK
    fallback, repair, server change, warmup or hidden retry. Response body bytes
    are an HTTP entity, not captured TCP/TLS wire bytes.
    """
    def __init__(self,endpoint):
        import httpx
        self.base=local_endpoint(endpoint+'/v1')[:-3]
        self.http=httpx.Client(timeout=httpx.Timeout(120,connect=8),trust_env=False,follow_redirects=False)
        self.metadata_requests=0;self.metadata_log=[]
    def get(self,path):
        if path not in ('/api/version','/api/tags','/api/ps'):raise ValueError('METADATA_PATH')
        self.metadata_requests+=1;r=self.http.get(self.base+path,timeout=8)
        entry={'path':path,'http_status':r.status_code,'body_sha256':hashlib.sha256(r.content).hexdigest()}
        self.metadata_log.append(entry)
        if r.status_code!=200:raise ValueError('METADATA_HTTP_ERROR')
        value=strict_json(r.content.decode('utf-8'));entry['parsed_response']=value
        return value
    def show(self):
        self.metadata_requests+=1;r=self.http.post(self.base+'/api/show',json={'model':MODEL},timeout=8)
        entry={'path':'/api/show','http_status':r.status_code,'body_sha256':hashlib.sha256(r.content).hexdigest()}
        self.metadata_log.append(entry)
        if r.status_code!=200:raise ValueError('SHOW_HTTP_ERROR')
        value=strict_json(r.content.decode('utf-8'));entry['parsed_response']=value
        return value
    def generate(self,payload):
        with self.http.stream('POST',self.base+'/api/chat',json=payload,timeout=120) as response:
            raw=b''
            for part in response.iter_bytes():
                raw+=part
                if len(raw)>1048576:raise ValueError('RESPONSE_BYTE_LIMIT')
            return {'http_status':response.status_code,'body':raw,'headers':{'content-type':response.headers.get('content-type')}}
    def close(self):self.http.close()


def server_state(transport,execution):
    version=transport.get('/api/version');tags=transport.get('/api/tags');ps=transport.get('/api/ps')
    matches=[r for r in tags.get('models',[]) if r.get('name')==execution['model'] or r.get('model')==execution['model']]
    if len(matches)!=1 or normalize_model_identifier(matches[0].get('digest'))!=execution['model_digest']:
        raise ValueError('SERVER_MODEL_DIGEST_MISMATCH')
    loaded=[r for r in ps.get('models',[]) if r.get('name')==execution['model'] or r.get('model')==execution['model']]
    if len(loaded)>1:raise ValueError('LOADED_MODEL_AMBIGUOUS')
    if loaded:
        if normalize_model_identifier(loaded[0].get('digest'))!=execution['model_digest']:raise ValueError('LOADED_DIGEST_MISMATCH')
        if loaded[0].get('context_length')!=execution['settings']['context']:raise ValueError('LOADED_CONTEXT_MISMATCH')
    return {'server_version':version,'tag_digest':execution['model_digest'],
        'loaded_context_status':'VERIFIED' if loaded else 'PENDING_FIRST_FIXED_TASK_LOAD',
        'loaded_model':loaded[0] if loaded else None,'identity_checks':1,'metadata_gets':3}


def request_payload(q,agent,execution,counter):
    public=model_input(q);user=json.dumps(public,ensure_ascii=False,sort_keys=True)
    if len(user.encode('utf-8'))>execution['input_bytes_max']:raise ValueError('INPUT_BYTE_BUDGET')
    n=counter.count(agent['system_prompt'],user);s=execution['settings']
    if n+s['max_output_tokens']+execution['context_margin']>s['context']:raise ValueError('CONTEXT_TOKEN_BUDGET')
    return {'model':execution['model'],'stream':False,'messages':[{'role':'system','content':agent['system_prompt']},
        {'role':'user','content':user}],'format':answer_schema(q),'options':{'temperature':s['temperature'],
        'seed':s['seed'],'num_ctx':s['context'],'num_predict':s['max_output_tokens']}},n


def run_collection(bundle,config,execution,root,*,execute=False,allow_llm=False,approval=None,
                   transport_factory=None,counter=None,test_only=False):
    """Default DRY_RUN is offline. Authorization verified before constructing transport."""
    validate_bundle(bundle,config)
    if execution['fingerprint']!=digest({k:v for k,v in execution.items() if k!='fingerprint'}):raise ValueError('EXECUTION_PROTOCOL_FINGERPRINT')
    if execution['parent_plan_fingerprint']!=bundle['plan']['fingerprint']:raise ValueError('EXECUTION_PLAN_BINDING')
    if not execute:return {'mode':'DRY_RUN','planned_tasks':25,'attempted':0,'generation_requests':0}
    if not allow_llm or not isinstance(approval,PinnedCollectionApproval):raise ValueError('EXPLICIT_PINNED_APPROVAL_REQUIRED')
    approval_result=approval.verify(execution)
    if not test_only:
        if type(approval) is not PinnedCollectionApproval:raise ValueError('LIVE_PINNED_APPROVAL_TYPE_REQUIRED')
        fresh_bundle,fresh_config=load_bundle()
        if fresh_bundle!=bundle or fresh_config!=config or protocol(bundle,config)!=execution:raise ValueError('CURRENT_EXECUTION_INPUT_OR_CODE_CHANGED')
        if transport_factory is not None or counter is not None:raise ValueError('LIVE_INJECTION_FORBIDDEN')
        from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter
        counter=LocalTokenCounter(TOKENIZER,TREE)
    if counter is None:raise ValueError('TOKEN_COUNTER_REQUIRED')
    root=Path(root).resolve()
    if not test_only and not root.is_relative_to((ROOT/'outputs/private/research_evaluation'/NAMESPACE).resolve()):raise ValueError('EXECUTION_PATH')
    # Reserve an exclusive run directory before metadata or generation. No resume.
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'started.json',{'protocol_fingerprint':execution['fingerprint'],'approval':approval_result,'run_id':execution['run_id']})
    transport=None;rows=[];attempts=0;stopped=None;identities=0;metadata_requests=0;results=[]
    qi={q['question_id']:q for q in bundle['questions']};ri={r['question_id']:r for r in bundle['references']};ai={a['agent_id']:a for a in bundle['bindings']}
    try:
        transport=transport_factory() if test_only else OllamaHTTP(execution['endpoint'])
        if test_only and (getattr(transport,'TEST_ONLY',False) is not True or type(transport).__module__.startswith('app.')):
            raise ValueError('TEST_ONLY_TRANSPORT_REQUIRED')
        initial=server_state(transport,execution);identities+=1
        atomic_json(root/'server_before.json',initial)
        for index,t in enumerate(bundle['plan']['tasks'],1):
            q=qi[t['question_id']];ref=ri[t['question_id']];a=ai[t['agent_id']]
            if stopped or attempts>=execution['max_generations']:
                sc=score(q,ref,attempted=False);rows.append({**sc,'agent_id':a['agent_id']});continue
            approval.verify(execution)
            if not test_only:
                for name,h in bundle['source_pins'].items():
                    if sha(name)!=h:raise ValueError('SOURCE_CHANGED_DURING_COLLECTION')
            payload,token_count=request_payload(q,a,execution,counter)
            if digest(model_input(q))!=t['input_fingerprint'] or digest(payload['format'])!=t['schema_fingerprint'] or digest(a['system_prompt'])!=t['prompt_fingerprint']:
                raise ValueError('TASK_HASH_MISMATCH')
            state=server_state(transport,execution);identities+=1
            if attempts>0 and state['loaded_context_status']!='VERIFIED':raise ValueError('CONTEXT_LOST_BETWEEN_TASKS')
            record={'question_id':q['question_id'],'agent_id':a['agent_id'],'specialty':a['department'],
                'registered_task_index':index,'model_tag':MODEL,'model_digest':DIGEST,
                'prompt_fingerprint':t['prompt_fingerprint'],'input_fingerprint':t['input_fingerprint'],
                'request_fingerprint':digest(payload),'requested_context':execution['settings']['context'],
                'requested_output_limit':1024,'planned_input_tokens':token_count,'server_before':state,
                'schema_fingerprint':digest(payload['format']),'http_status':None,'response_fingerprint':None,
                'server_token_counts':None,'schema_valid':False,'raw_wire_status':'HTTP_ENTITY_BYTES_ONLY_NOT_TCP_WIRE',
                'generation_budget_consumed':True,'status':'REQUEST_STARTING'}
            atomic_json(root/('Request%03d_before.json'%index),{**record,'payload':payload})
            attempts+=1;started=time.monotonic();result=None;content=None
            try:
                reply=transport.generate(payload)
                record['http_status']=reply['http_status']
                raw=reply['body']
                # Preserve actual HTTP entity bytes, including non-200 error bodies privately.
                with (root/('Request%03d_response_body.bin'%index)).open('xb') as f:f.write(raw)
                record['http_entity_sha256']=hashlib.sha256(raw).hexdigest()
                if reply['http_status']!=200:raise ValueError('GENERATION_HTTP_ERROR')
                envelope=strict_json(raw.decode('utf-8'))
                if envelope.get('model')!=MODEL or envelope.get('done') is not True:raise ValueError('RESPONSE_MODEL_OR_DONE')
                content=envelope.get('message',{}).get('content')
                if not isinstance(content,str):raise ValueError('RESPONSE_CONTENT')
                record.update(response_fingerprint=digest(content),server_token_counts={k:envelope.get(k) for k in ('prompt_eval_count','eval_count','total_duration','load_duration','prompt_eval_duration','eval_duration')},
                    server_done_reason=envelope.get('done_reason'),raw_model_content=content,parsed_envelope=envelope)
                result=score(q,ref,content)
                record['schema_valid']=result['state'] not in ('INVALID_RESPONSE','GENERATION_FAILED')
                try:record['parsed_response']=strict_json(content)
                except (ValueError,TypeError):record['parsed_response']=None
                if result['state'] in ('CORRECT_ANSWER','INCORRECT_ANSWER'):
                    parsed=record['parsed_response'];result['answer_signature']=digest(normalize(parsed['diagnosis']))
                elif result['state']=='UNRESOLVED':result['answer_signature']=digest('UNRESOLVED')
                if type(envelope.get('eval_count')) is int and envelope['eval_count']>1024:
                    raise ValueError('SERVER_OUTPUT_TOKEN_BUDGET_EXCEEDED')
                if type(envelope.get('prompt_eval_count')) is int and envelope['prompt_eval_count']+1024+execution['context_margin']>execution['settings']['context']:
                    raise ValueError('SERVER_INPUT_TOKEN_BUDGET_EXCEEDED')
                if envelope.get('done_reason')=='length':
                    result.update(state='INVALID_RESPONSE',answer_correct=None);record['schema_valid']=False
                    stopped='OUTPUT_LIMIT_TRUNCATION'
                after=server_state(transport,execution);identities+=1
                record['server_after']=after
                if after['loaded_context_status']!='VERIFIED':raise ValueError('POST_REQUEST_CONTEXT_NOT_VERIFIED')
                if result['state']=='INVALID_RESPONSE':stopped=stopped or 'INVALID_RESPONSE'
            except Exception as exc:
                code=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z0-9_]+',str(exc)) else type(exc).__name__
                stopped=code;record['error_type']=type(exc).__name__;record['error_code']=code
                # A response already scored is preserved if a later context check fails.
                if result is None:result=score(q,ref,error_type=type(exc).__name__)
            record.update(wall_seconds=time.monotonic()-started,scoring=result,status='RECORDED' if not stopped else 'STOPPED',
                completed_at=datetime.now(timezone.utc).isoformat())
            results.append(record);rows.append({**result,'agent_id':a['agent_id']})
            atomic_json(root/('Request%03d_after.json'%index),record)
        metadata_requests=getattr(transport,'metadata_requests',None)
    except Exception as exc:
        stopped=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z0-9_]+',str(exc)) else type(exc).__name__
    finally:
        if transport is not None:
            metadata_requests=getattr(transport,'metadata_requests',metadata_requests)
            transport.close()
            atomic_json(root/'metadata_log.json',{'parsed_metadata_responses':getattr(transport,'metadata_log',[]),
                'raw_wire_saved':False})
    # Preserve every planned slot even if a pre-generation check raised.
    seen={(r['question_id'],r['agent_id']) for r in rows}
    for t in bundle['plan']['tasks']:
        if (t['question_id'],t['agent_id']) not in seen:
            rows.append({**score(qi[t['question_id']],ri[t['question_id']],attempted=False),'agent_id':t['agent_id']})
    summary={'status':'COMPLETE' if attempts==25 and not stopped else ('PARTIAL' if attempts else 'PREFLIGHT_BLOCKED'),
        'registered':25,'attempted':attempts,'http_success':sum(r['http_status']==200 for r in results),
        'schema_valid':sum(r['schema_valid'] for r in results),'stop_reason':stopped,'identity_checks':identities,
        'metadata_requests':metadata_requests,'test_only':test_only,'actual_llm_generations':0 if test_only else attempts,
        'usage_scope':'Requests attempted, including timeout uncertainty; server eval tokens reported separately',
        'rows':rows,'evaluation':evaluate(bundle['questions'],bundle['bindings'],rows)}
    atomic_json(root/'collection.json',summary)
    files={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()}
    atomic_json(root/'completion_manifest.json',{'status':summary['status'],'protocol_fingerprint':execution['fingerprint'],
        'files':files,'fingerprint':digest(files)})
    return summary


def selection_potential(bundle,rows):
    """Paired complete observed questions only; no unknown-agent oracle imputation."""
    qids=[q['question_id'] for q in bundle['questions']];aids=[a['agent_id'] for a in bundle['bindings']]
    indexed={(r['question_id'],r['agent_id']):r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('DUPLICATE_OUTCOME')
    complete=[q for q in qids if all((q,a) in indexed and indexed[q,a]['state'] in ('CORRECT_ANSWER','INCORRECT_ANSWER','UNRESOLVED') for a in aids)]
    n=len(complete);success={a:sum(indexed[q,a]['state']=='CORRECT_ANSWER' for q in complete) for a in aids}
    oracle=sum(any(indexed[q,a]['state']=='CORRECT_ANSWER' for a in aids) for q in complete)
    best=max(success.values()) if n else 0
    return {'complete_paired_questions':n,'incomplete_excluded':len(qids)-n,
        'fixed_cardiology':rate(success[aids[0]],n),'best_fixed_posthoc':rate(best,n),
        'best_fixed_agents':[a for a in aids if success[a]==best] if n else [],
        'uniform_random_expected':rate(sum(success.values())/len(aids),n),
        'oracle_posthoc_upper_bound':rate(oracle,n),'oracle_gap':(oracle-best)/n if n else None,
        'per_agent_success':success if n else {a:None for a in aids},
        'interpretation':'Pilot posthoc potential, not achievable router performance or held-out estimate'}
