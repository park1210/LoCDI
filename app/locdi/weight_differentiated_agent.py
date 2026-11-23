"""C4-28 pinned deployment comparison; no Core/registry/training modification."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import time
from app.locdi.answer_grounded_live_collection import (ROOT,sha,atomic_json,OllamaHTTP,server_state,
    load_bundle as load_parent,TOKENIZER,TREE,selection_potential)
from app.locdi.answer_grounded_compact import VERSION,PROMPT_VERSION,PROMPT,compact_binding,schema,parse,score,repetition
from app.locdi.answer_grounded_routing import model_input,validate_pair
from app.locdi.answer_grounded_routing_evaluation import evaluate,rate,metrics
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.reference_acquisition import strict_json
from app.locdi.external_reference_readiness import pinned_private_json

NAMESPACE='c4_28_v1'
RUN_ID='c4_28_weight_comparison_001'
TAGS=['qwen2.5:1.5b-instruct','qwen-neurology:latest','qwen-pulmonology:latest']
EXPECTED_DIGESTS={'qwen2.5:1.5b-instruct':'65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b',
    'qwen-neurology:latest':'d56baced7349a8a3a911933b4e58a2656ce6754fa0bd28fd9c7fe6c9ebfc0128',
    'qwen-pulmonology:latest':'f49ce9dbb5036498fe3ad0d9600fec881df85877253dc3c3f74cbabd0e92956c'}
EXPECTED_BLOBS={'qwen-neurology:latest':'61c76cf449f8138e00700c23738561429889829a738218649f8ba957e4e2b98b',
    'qwen-pulmonology:latest':'3e5bfc63c4d22febf645b6b6cf9092878e872bcfe1af7abfff222973a63af8d4'}
CODE=['app/locdi/weight_differentiated_agent.py','scripts/evaluate_weight_differentiated_agents.py',
    'tests/locdi/test_weight_differentiated_agent.py']


def gguf_metadata(path):
    """Read bounded GGUFv2/3 KV metadata only; never deserialize model tensors."""
    with Path(path).open('rb') as f:
        def unpack(fmt):
            size=struct.calcsize('<'+fmt);raw=f.read(size)
            if len(raw)!=size:raise ValueError('GGUF_TRUNCATED')
            if f.tell()>67108864:raise ValueError('GGUF_METADATA_BYTE_LIMIT')
            return struct.unpack('<'+fmt,raw)[0]
        def string():
            n=unpack('Q')
            if n>16777216 or f.tell()+n>67108864:raise ValueError('GGUF_STRING_LIMIT')
            raw=f.read(n)
            if len(raw)!=n:raise ValueError('GGUF_TRUNCATED')
            return raw.decode('utf-8')
        def value(t,depth=0):
            if depth>2:raise ValueError('GGUF_ARRAY_DEPTH')
            fmts={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
            if t in fmts:return unpack(fmts[t])
            if t==8:return string()
            if t==9:
                subtype=unpack('I');n=unpack('Q')
                if n>2000000:raise ValueError('GGUF_ARRAY_LIMIT')
                return [value(subtype,depth+1) for _ in range(n)]
            raise ValueError('GGUF_METADATA_TYPE')
        if f.read(4)!=b'GGUF':raise ValueError('GGUF_MAGIC')
        version=unpack('I')
        if version not in (2,3):raise ValueError('GGUF_VERSION')
        tensors=unpack('Q');count=unpack('Q')
        if count>100000:raise ValueError('GGUF_KEY_LIMIT')
        values={}
        for _ in range(count):
            key=string()
            if key in values:raise ValueError('GGUF_DUPLICATE_KEY')
            values[key]=value(unpack('I'))
        return {'version':version,'tensor_count':tensors,'metadata_bytes':f.tell(),'values':values}


class WeightHTTP(OllamaHTTP):
    def show_model(self,tag):
        if tag not in TAGS:raise ValueError('UNREGISTERED_MODEL')
        self.metadata_requests+=1;r=self.http.post(self.base+'/api/show',json={'model':tag},timeout=8)
        entry={'path':'/api/show','model':tag,'http_status':r.status_code,'body_sha256':hashlib.sha256(r.content).hexdigest()};self.metadata_log.append(entry)
        if r.status_code!=200:raise ValueError('SHOW_MODEL_HTTP')
        v=strict_json(r.content.decode('utf-8'));entry['parsed_response']=v;return v
    def unload(self,tag):
        if tag not in TAGS:raise ValueError('UNREGISTERED_UNLOAD')
        self.metadata_requests+=1
        r=self.http.post(self.base+'/api/generate',json={'model':tag,'keep_alive':0,'stream':False},timeout=8)
        entry={'path':'/api/generate','operation':'UNLOAD_NO_PROMPT','model':tag,'http_status':r.status_code,'body_sha256':hashlib.sha256(r.content).hexdigest()};self.metadata_log.append(entry)
        if r.status_code!=200:raise ValueError('UNLOAD_HTTP_ERROR')
        v=strict_json(r.content.decode('utf-8'));entry['parsed_response']=v
        if (v.get('done') is not True or v.get('done_reason')!='unload' or v.get('response') not in ('',None)
                or (v.get('eval_count') or 0)!=0 or (v.get('prompt_eval_count') or 0)!=0):raise ValueError('UNLOAD_NOT_NON_GENERATIVE')
        return v


def model_state(http,ep,tag):
    bound=ep['model_bindings'][tag]
    state=server_state(http,{'model':tag,'model_digest':'sha256:'+bound['manifest_digest'],'settings':{'context':ep['context']}})
    return state


def token_budget(q,agent,contract,counter):
    user=json.dumps(model_input(q),ensure_ascii=False,sort_keys=True)
    if len((agent['system_prompt']+user).encode('utf-8'))>contract['input_bytes_max']:raise ValueError('INPUT_BYTE_BUDGET')
    # The bound is pinned by a separate tokenizer/template audit before approval.
    n=counter.count(agent['system_prompt'],user)
    bound=n+contract['native_template_token_reserve']
    if bound+1024+512>contract['context']:raise ValueError('CONTEXT_TOKEN_BUDGET')
    return user,n,bound


def build_bundle(parent,artifacts,contract):
    if len(parent['questions'])!=5:raise ValueError('FIXED_FIVE_QUESTIONS_REQUIRED')
    roles={a['department']:compact_binding(a) for a in parent['bindings']}
    definitions=[('A',TAGS[0],'neurology'),('B',TAGS[1],'neurology'),('C',TAGS[0],'pulmonology'),('D',TAGS[2],'pulmonology')]
    bindings=[]
    for cid,tag,role in definitions:
        bindings.append({**roles[role],'agent_id':cid,'condition':cid,'model':tag,'model_identifier':'sha256:'+artifacts[tag]['manifest_digest'],
            'department':role,'deployment_status':'UNKNOWN_TRAINING_PROVENANCE; UNKNOWN_TRAINING_OVERLAP',
            'quantization':artifacts[tag]['quantization'],'original_registry_version':roles[role]['version'],
            'version':'C4_28_DEPLOYMENT_PINNED','training_version':'UNKNOWN','deployment_version':artifacts[tag]['manifest_digest']})
    tasks=[]
    # Condition blocks minimize model reloads; fixed before any new outcome.
    for a in bindings:
        for i,(q,r) in enumerate(zip(parent['questions'],parent['references']),1):
            validate_pair(q,r)
            tasks.append({'parent_question_ordinal':i,'question_id':q['question_id'],'agent_id':a['agent_id'],'model':a['model'],
                'input_fingerprint':q['input_fingerprint'],'prompt_fingerprint':digest(a['system_prompt']),'schema_fingerprint':digest(schema())})
    plan={'version':'locdi.weight_differentiated_agent.c4_28.v1','task_version':VERSION,'prompt_version':PROMPT_VERSION,
        'parent_plan_fingerprint':parent['plan']['fingerprint'],'run_id':RUN_ID,'namespace':NAMESPACE,'tasks':tasks,
        'task_order':'A questions1..5, B1..5, C1..5, D1..5; not chosen by outcomes','max_generations':20,
        'question_reference_fingerprint':digest(list(zip(parent['questions'],parent['references']))),'bindings_fingerprint':digest(bindings),
        'execution_contract_fingerprint':digest(contract),'purpose':'DEVELOPMENT_EXPLORATORY_EVALUATION',
        'causal_claim':'NO_FINE_TUNING_CAUSAL_CLAIM','resume':False}
    plan['fingerprint']=digest(plan)
    return {'plan':plan,'questions':deepcopy(parent['questions']),'references':deepcopy(parent['references']),
        'bindings':bindings,'source_pins':deepcopy(parent['source_pins'])}


def load_bundle():
    pr=ROOT/'outputs/private/research_evaluation'/NAMESPACE;pub=ROOT/'outputs/research_evaluation'/NAMESPACE
    parent,cfg=load_parent();artifacts=strict_json((pr/'model_artifacts_private.json').read_text(encoding='utf-8'))
    contract=strict_json((pub/'common_execution_contract.json').read_text(encoding='utf-8'))
    return build_bundle(parent,artifacts['models'],contract),cfg,artifacts,contract


def protocol(bundle,artifacts,contract):
    p=bundle['plan'];prior=ROOT/'outputs/research_evaluation/c4_27_v1/completion_manifest.json'
    v={'version':'locdi.weight_differentiated_agent.c4_28.v1','run_id':RUN_ID,'namespace':NAMESPACE,'plan_fingerprint':p['fingerprint'],
        'parent_plan_fingerprint':p['parent_plan_fingerprint'],'parent_completion_manifest_sha256':sha(prior),
        'source_sha256':sha(ROOT/'outputs/private/research_evaluation/c3_10_v1/upstream_snapshot/agentclinic_medqa.jsonl'),
        'task_version':VERSION,'prompt_version':PROMPT_VERSION,'schema_fingerprint':digest(schema()),'task_prompt_fingerprint':digest(PROMPT),
        'task_order_fingerprint':digest(p['tasks']),'model_bindings':{tag:{k:a[k] for k in ('manifest_digest','gguf_sha256','template_fingerprint','parameters_fingerprint','metadata_fingerprint','tokenizer_fingerprint')} for tag,a in artifacts['models'].items()},
        'contract':contract,'context':contract['context'],'endpoint':'http://127.0.0.1:11434','max_generations':20,
        'switch_policy':'At most1 loaded experiment model; block foreign/multiple models. Unload old experiment model before switches using no-prompt keep_alive0; verify ps empty. Max4 unloads, no final forced unload. All request keep_alive120s.',
        'retry':0,'repair':0,'warmup':0,'resume':False,'first_technical_error_stops':True,
        'implementation_hashes':{n:sha(ROOT/n) for n in CODE},
        'sealed_compact_code_hashes':{n:sha(ROOT/n) for n in ('app/locdi/answer_grounded_compact.py','app/locdi/answer_grounded_live_collection.py','app/locdi/answer_grounded_routing_evaluation.py')},
        'artifact_snapshot_sha256':sha(ROOT/'outputs/private/research_evaluation'/NAMESPACE/'model_artifacts_private.json'),
        'tokenizer_tree_sha256':TREE,'interpretation':'UNKNOWN_TRAINING_PROVENANCE; UNKNOWN_TRAINING_OVERLAP; DEVELOPMENT_EXPLORATORY_EVALUATION; NO_FINE_TUNING_CAUSAL_CLAIM'}
    v['fingerprint']=digest(v);return v


class WeightApproval:
    def __init__(self,path,expected_sha):self.path,self.expected_sha=path,expected_sha
    def verify(self,ep,now=None):
        a=pinned_private_json(self.path,self.expected_sha);now=now or datetime.now(timezone.utc)
        fields={'scope':'LIVE_WEIGHT_COMPARISON_20_ONLY','authorization_source':'EXPLICIT_USER_STATEMENT','operator_reference':'USER_LOCAL_SERVER_OPERATOR',
            'protocol_fingerprint':ep['fingerprint'],'plan_fingerprint':ep['plan_fingerprint'],'parent_plan_fingerprint':ep['parent_plan_fingerprint'],
            'source_sha256':ep['source_sha256'],'task_order_fingerprint':ep['task_order_fingerprint'],'model_bindings_fingerprint':digest(ep['model_bindings']),
            'contract_fingerprint':digest(ep['contract']),'task_version':VERSION,'prompt_version':PROMPT_VERSION,'schema_fingerprint':ep['schema_fingerprint'],
            'endpoint':ep['endpoint'],'namespace':NAMESPACE,'run_id':RUN_ID,'max_generations':20,'retry':0,'repair':0,'warmup':0,
            'allow_experiment_model_unloads':True,'max_unloads':4,'no_other_ollama_work_confirmed':True}
        if any(a.get(k)!=v for k,v in fields.items()) or not a.get('user_statement'):raise ValueError('WEIGHT_APPROVAL_SCOPE')
        issued=datetime.fromisoformat(a['recorded_at']);expiry=datetime.fromisoformat(a['expires_at'])
        if issued.tzinfo is None or expiry.tzinfo is None or not issued<=now<expiry or not 0<(expiry-issued).total_seconds()<=86400:raise ValueError('WEIGHT_APPROVAL_VALIDITY')
        return {'approval_sha256':self.expected_sha,'scope':fields['scope']}


def validate(bundle,artifacts,contract,ep,test_only=False):
    p=bundle['plan']
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('WEIGHT_PLAN_FINGERPRINT')
    if ep['fingerprint']!=digest({k:v for k,v in ep.items() if k!='fingerprint'}):raise ValueError('WEIGHT_PROTOCOL_FINGERPRINT')
    if ep['plan_fingerprint']!=p['fingerprint'] or ep['max_generations']!=20 or len(p['tasks'])!=20:raise ValueError('FIXED20_BINDING')
    if p['run_id']!=RUN_ID or ep['namespace']!=NAMESPACE:raise ValueError('WEIGHT_RUN_NAMESPACE')
    if not test_only:
        b,c,a,k=load_bundle()
        if b!=bundle or a!=artifacts or k!=contract or protocol(b,a,k)!=ep:raise ValueError('CURRENT_WEIGHT_CODE_OR_INPUT_CHANGED')
        for tag,m in artifacts['models'].items():verify_local_artifact(m)


def verify_local_artifact(model,full=False):
    if sha(model['manifest_path'])!=model['manifest_digest']:raise ValueError('LOCAL_MODEL_MANIFEST_CHANGED')
    if Path(model['blob_path']).stat().st_size!=model['gguf_bytes']:raise ValueError('MODEL_BLOB_SIZE_CHANGED')
    if full:
        h=hashlib.sha256()
        with Path(model['blob_path']).open('rb') as f:
            for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
        if h.hexdigest()!=model['gguf_sha256']:raise ValueError('MODEL_BLOB_CHANGED')
    return True


def gpu_usage():
    import subprocess
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        return {'status':'MEASURED','raw_query':r.stdout.strip()} if r.returncode==0 else {'status':'NOT_MEASURED'}
    except (OSError,subprocess.TimeoutExpired):return {'status':'NOT_MEASURED'}


def run_collection(bundle,artifacts,contract,ep,root,*,execute=False,allow_llm=False,approval=None,
                   counter=None,transport_factory=None,test_only=False):
    validate(bundle,artifacts,contract,ep,test_only)
    if not execute:return {'mode':'DRY_RUN','registered':20,'generation_requests':0,'client_constructed':False,'unloads':0}
    if not allow_llm or type(approval) is not WeightApproval:raise ValueError('NEW_WEIGHT_APPROVAL_REQUIRED')
    proof=approval.verify(ep)
    if not test_only:
        if counter is not None or transport_factory is not None:raise ValueError('LIVE_INJECTION_FORBIDDEN')
        from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter
        counter=LocalTokenCounter(TOKENIZER,TREE)
    if counter is None:raise ValueError('TOKEN_COUNTER_REQUIRED')
    root=Path(root).resolve()
    if not test_only and root!=(ROOT/'outputs/private/research_evaluation'/NAMESPACE/'live'/RUN_ID).resolve():raise ValueError('WEIGHT_EXECUTION_PATH')
    root.mkdir(parents=True,exist_ok=False);atomic_json(root/'started.json',{'protocol_fingerprint':ep['fingerprint'],'approval':proof})
    qi={q['question_id']:q for q in bundle['questions']};ri={r['question_id']:r for r in bundle['references']};ai={a['agent_id']:a for a in bundle['bindings']}
    attempts=0;uncertain=0;unloads=0;active=None;stopped=None;rows=[];records=[];http=None
    try:
        http=transport_factory() if test_only else WeightHTTP(ep['endpoint'])
        if test_only and (getattr(http,'TEST_ONLY',False) is not True or type(http).__module__.startswith('app.')):raise ValueError('TEST_ONLY_TRANSPORT_REQUIRED')
        initial=http.get('/api/ps');atomic_json(root/'server_before.json',initial)
        for i,t in enumerate(bundle['plan']['tasks'],1):
            if stopped:break
            if attempts+uncertain>=20:raise ValueError('GLOBAL_BUDGET20')
            approval.verify(ep)
            if not test_only:validate(bundle,artifacts,contract,ep)
            q=qi[t['question_id']];ref=ri[t['question_id']];a=ai[t['agent_id']];tag=a['model']
            ps=http.get('/api/ps');loaded=ps.get('models',[])
            if len(loaded)>1 or any(r.get('name',r.get('model')) not in TAGS for r in loaded):raise ValueError('OTHER_OR_MULTIPLE_MODELS_LOADED')
            switching=tag!=active
            if switching:
                if loaded:
                    previous=loaded[0].get('name',loaded[0].get('model'))
                    if unloads>=4:raise ValueError('UNLOAD_BUDGET4')
                    unloads+=1;atomic_json(root/('Unload%03d_before.json'%unloads),{'model':previous,'no_prompt':True,'keep_alive':0,'before':ps})
                    try:reply=http.unload(previous)
                    except Exception:
                        uncertain+=1;raise
                    empty=http.get('/api/ps');atomic_json(root/('Unload%03d_after.json'%unloads),{'reply':reply,'after':empty,'generation_tokens':0})
                    if empty.get('models'):raise ValueError('UNLOAD_DID_NOT_CLEAR_MODELS')
                show=http.show_model(tag);bound=ep['model_bindings'][tag]
                if digest(show.get('template',''))!=bound['template_fingerprint'] or digest(show.get('parameters',''))!=bound['parameters_fingerprint']:raise ValueError('MODEL_TEMPLATE_OR_PARAMETERS_CHANGED')
                if not test_only:
                    # Full blob bytes verified in audit; rehash each distinct model once before loading.
                    checked={r['model_tag'] for r in records}
                    if tag not in checked:
                        verify_local_artifact(artifacts['models'][tag],full=True)
                active=tag
            before=model_state(http,ep,tag)
            if not switching and before['loaded_context_status']!='VERIFIED':raise ValueError('MODEL_CONTEXT_LOST')
            user,n,input_bound=token_budget(q,a,contract,counter)
            payload={'model':tag,'stream':False,'keep_alive':120,'messages':[{'role':'system','content':a['system_prompt']},{'role':'user','content':user}],
                'format':schema(),'options':deepcopy(contract['options'])}
            if digest(model_input(q))!=t['input_fingerprint'] or digest(a['system_prompt'])!=t['prompt_fingerprint'] or digest(payload['format'])!=t['schema_fingerprint']:raise ValueError('WEIGHT_TASK_HASH')
            rec={'registered_task_index':i,'question_id':q['question_id'],'condition':a['agent_id'],'agent_id':a['agent_id'],'parent_question_ordinal':t['parent_question_ordinal'],
                'model_tag':tag,'model_digest':ep['model_bindings'][tag]['manifest_digest'],'gguf_sha256':ep['model_bindings'][tag]['gguf_sha256'],
                'request_fingerprint':digest(payload),'input_fingerprint':t['input_fingerprint'],'prompt_fingerprint':t['prompt_fingerprint'],
                'local_hf_input_token_estimate':n,'conservative_input_token_bound':input_bound,'server_before':before,'http_status':None,'json_valid':False,'schema_valid':False,
                'status':'REQUEST_STARTING','budget_consumed':True,'raw_wire_status':'HTTP_ENTITY_BYTES_NOT_TCP_WIRE'}
            atomic_json(root/('Request%03d_before.json'%i),{**rec,'payload':payload});attempts+=1;start=time.monotonic();result=None
            try:
                response=http.generate(payload);raw=response['body'];rec['http_status']=response['http_status']
                with (root/('Request%03d_response_body.bin'%i)).open('xb') as f:f.write(raw)
                rec['http_entity_sha256']=hashlib.sha256(raw).hexdigest()
                if response['http_status']!=200:raise ValueError('GENERATION_HTTP_ERROR')
                envelope=strict_json(raw.decode('utf-8'));rec['parsed_envelope']=envelope
                if envelope.get('model')!=tag or envelope.get('done') is not True:raise ValueError('RESPONSE_MODEL_OR_DONE')
                content=envelope.get('message',{}).get('content')
                if not isinstance(content,str):raise ValueError('RESPONSE_CONTENT')
                rec.update(raw_model_content=content,response_fingerprint=digest(content),repetition=repetition(content),server_done_reason=envelope.get('done_reason'),
                    server_token_counts={k:envelope.get(k) for k in ('prompt_eval_count','eval_count','total_duration','load_duration','prompt_eval_duration','eval_duration')})
                try:rec['parsed_response']=strict_json(content);rec['json_valid']=True
                except (ValueError,TypeError):rec['parsed_response']=None
                result=score(q,ref,content);rec['schema_valid']=result['state']!='INVALID_RESPONSE'
                if type(envelope.get('prompt_eval_count')) is not int or envelope['prompt_eval_count']>input_bound:raise ValueError('SERVER_INPUT_EXCEEDS_CONSERVATIVE_BOUND')
                if type(envelope.get('eval_count')) is not int or envelope['eval_count']>1024:raise ValueError('SERVER_OUTPUT_BUDGET')
                if envelope.get('done_reason')=='length':result.update(state='INVALID_RESPONSE',answer_correct=None);rec['schema_valid']=False;stopped='OUTPUT_LIMIT_TRUNCATION'
                after=model_state(http,ep,tag);rec['server_after']=after
                if after['loaded_context_status']!='VERIFIED':raise ValueError('POST_REQUEST_CONTEXT_NOT_VERIFIED')
                psafter=http.get('/api/ps')
                if len(psafter.get('models',[]))!=1:raise ValueError('POST_REQUEST_MULTIPLE_MODELS')
                rec['gpu']=gpu_usage() if not test_only else {'status':'TEST_ONLY'}
                if result['state']=='INVALID_RESPONSE':stopped=stopped or 'INVALID_RESPONSE'
            except Exception as exc:
                stopped=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z0-9_]+',str(exc)) else type(exc).__name__
                rec.update(error_code=stopped,error_type=type(exc).__name__)
                if result is None:result=score(q,ref,error_type=type(exc).__name__)
            rec.update(scoring=result,status='STOPPED' if stopped else 'RECORDED',wall_seconds=time.monotonic()-start,completed_at=datetime.now(timezone.utc).isoformat())
            records.append(rec);rows.append({**result,'agent_id':a['agent_id'],'execution_contract_valid':not stopped});atomic_json(root/('Request%03d_after.json'%i),rec)
    except Exception as exc:stopped=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z0-9_]+',str(exc)) else type(exc).__name__
    finally:
        if http is not None:
            http.close();atomic_json(root/'metadata_log.json',{'responses':getattr(http,'metadata_log',[]),'raw_wire_saved':False})
    seen={(r['question_id'],r['agent_id']) for r in rows}
    for t in bundle['plan']['tasks']:
        if (t['question_id'],t['agent_id']) not in seen:rows.append({**score(qi[t['question_id']],ri[t['question_id']],attempted=False),'agent_id':t['agent_id']})
    result={'status':'COMPLETE' if attempts==20 and not stopped else ('PARTIAL' if attempts else 'PREFLIGHT_BLOCKED'),'registered':20,'attempted':attempts,
        'stop_reason':stopped,'generation_budget_consumed':attempts+uncertain,'uncertain_or_unexpected_unload_budget_charges':uncertain,'unload_requests':unloads,
        'actual_task_generation_requests':0 if test_only else attempts,'test_only':test_only,'metadata_requests':getattr(http,'metadata_requests',0),
        'rows':rows,'evaluation':evaluate(bundle['questions'],bundle['bindings'],rows)}
    atomic_json(root/'collection.json',result);files={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()}
    atomic_json(root/'completion_manifest.json',{'files':files,'fingerprint':digest(files),'protocol_fingerprint':ep['fingerprint']});return result


def selection_menus(bundle,rows):
    index={(r['question_id'],r['agent_id']):r for r in rows};aids=['A','B','C','D']
    if len(index)!=len(rows):raise ValueError('DUPLICATE_WEIGHT_OUTCOME')
    complete=[q for q in bundle['questions'] if all(index.get((q['question_id'],a),{}).get('state') in ('CORRECT_ANSWER','INCORRECT_ANSWER','UNRESOLVED') and index.get((q['question_id'],a),{}).get('execution_contract_valid',True) for a in aids)]
    out={}
    for name,ids in [('base_only',['A','C']),('distinct_weight',['B','D']),('mixed',aids)]:
        b={'questions':complete,'bindings':[a for a in bundle['bindings'] if a['agent_id'] in ids]}
        rs=[index[q['question_id'],a] for q in complete for a in ids];v=selection_potential(b,rs)
        v['all_unresolved_questions']=sum(all(index[q['question_id'],a]['state']=='UNRESOLVED' for a in ids) for q in complete)
        v['no_correct_provider_questions']=sum(not any(index[q['question_id'],a]['state']=='CORRECT_ANSWER' for a in ids) for q in complete)
        v['correctness_disagreement_questions']=sum(any(index[q['question_id'],a]['state']=='CORRECT_ANSWER' for a in ids) and not all(index[q['question_id'],a]['state']=='CORRECT_ANSWER' for a in ids) for q in complete)
        v['unique_success_conditions']={a:sum(index[q['question_id'],a]['state']=='CORRECT_ANSWER' and sum(index[q['question_id'],z]['state']=='CORRECT_ANSWER' for z in ids)==1 for q in complete) for a in ids}
        # Generic helper's first-condition legacy label is renamed, never a cardiology comparison here.
        v['fixed_first_condition']=v.pop('fixed_cardiology');out[name]=v
    return {'shared_complete_question_denominator':len(complete),'menus':out,
        'interpretation':'All menus use same four-condition-complete cohort; Oracle posthoc, menu sizes2/2/4 affect random expectation and upper-bound opportunities'}


def role_transitions(bundle,rows):
    index={(r['question_id'],r['agent_id']):r for r in rows};result={}
    for role,a,b in [('neurology','A','B'),('pulmonology','C','D')]:
        transitions=[]
        for i,q in enumerate(bundle['questions'],1):
            left=index[q['question_id'],a]['state'];right=index[q['question_id'],b]['state']
            valid=all(s in ('CORRECT_ANSWER','INCORRECT_ANSWER','UNRESOLVED') for s in (left,right)) and all(index[q['question_id'],z].get('execution_contract_valid',True) for z in (a,b))
            transitions.append({'question_ordinal':i,'base_state':left,'deployment_state':right,'paired_evaluable':valid,
                'transition':('SAME_OUTCOME' if left==right else left+' -> '+right) if valid else 'NOT_EVALUATED'})
        full=[q for q in bundle['questions'] if next(t for t in transitions if t['question_ordinal']==bundle['questions'].index(q)+1)['paired_evaluable']]
        result[role]={'rows':transitions,'paired_question_denominator':len(full),
            'base':metrics([index[q['question_id'],a] for q in full]),'deployment':metrics([index[q['question_id'],b] for q in full])}
    return result


def research_decision(collection,ev,menus):
    if collection['stop_reason']:return 'STOP_CURRENT_ROUTING_TRACK_TECHNICAL_FAILURE'
    if not menus['shared_complete_question_denominator']:return 'STOP_CURRENT_ROUTING_TRACK_NO_COMPLETE_COMPARISON'
    mixed=menus['menus']['mixed'];base=menus['menus']['base_only']
    correct=lambda ids:sum(ev['per_agent'][k]['counts']['CORRECT_ANSWER'] for k in ids)
    supported=lambda ids:sum(ev['per_agent'][k]['counts']['CORRECT_ANSWER']+ev['per_agent'][k]['counts']['INCORRECT_ANSWER'] for k in ids)
    if (mixed['oracle_gap'] and mixed['oracle_gap']>0 and correct(('B','D')) and
            mixed['oracle_posthoc_upper_bound']['numerator']>base['oracle_posthoc_upper_bound']['numerator']):
        return 'GO_DISTINCT_WEIGHT_AGENT_SELECTION_EXPLORATORY'
    if correct(('B','D'))>correct(('A','C')) and mixed['oracle_gap']==0:return 'MODEL_IMPROVEMENT_WITHOUT_ROUTING_GAIN'
    if supported(('B','D'))>supported(('A','C')) and correct(('B','D'))<=correct(('A','C')):return 'COVERAGE_ONLY_IMPROVEMENT'
    return 'STOP_CURRENT_ROUTING_TRACK'
