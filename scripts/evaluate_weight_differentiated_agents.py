"""C4-28 audit -> fixed plan -> new explicit approval -> sequential comparison."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha as stream_sha,locations
from app.locdi.weight_differentiated_agent import *


def text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value)


def audit():
    pub,pr=locations(NAMESPACE);assert read(pr/'initial_preservation.json')['protected']['status']=='PASS'
    http=WeightHTTP('http://127.0.0.1:11434');metadata={};models={};safe=[];tokenizers={}
    hftok=read(TOKENIZER/'tokenizer.json');hfconfig=read(TOKENIZER/'tokenizer_config.json')
    vocab=hftok['model']['vocab'];hfmerges=hftok['model']['merges'];hfmerges=[' '.join(x) if isinstance(x,list) else x for x in hfmerges]
    try:
        metadata={p:http.get(p) for p in ('/api/version','/api/tags','/api/ps')}
        tags=metadata['/api/tags']['models']
        for tag in TAGS:
            row=next(t for t in tags if t.get('name')==tag)
            assert row['digest'].removeprefix('sha256:')==EXPECTED_DIGESTS[tag],'TAG_CHANGED'
            name,version=tag.split(':');mp=Path.home()/'.ollama/models/manifests/registry.ollama.ai/library'/name/version
            assert stream_sha(mp)==EXPECTED_DIGESTS[tag],'LOCAL_MANIFEST_SERVER_MISMATCH'
            manifest=read(mp);layers=[r for r in manifest['layers'] if r['mediaType']=='application/vnd.ollama.image.model'];assert len(layers)==1
            layer=layers[0];h=layer['digest'].removeprefix('sha256:');blob=Path.home()/'.ollama/models/blobs'/('sha256-'+h)
            assert blob.stat().st_size==layer['size'] and stream_sha(blob)==h,'BLOB_HASH'
            if tag in EXPECTED_BLOBS:assert h==EXPECTED_BLOBS[tag],'PRIOR_SPECIALTY_BLOB_CHANGED'
            gg=gguf_metadata(blob);values=gg['values'];show=http.show_model(tag)
            tokens=values['tokenizer.ggml.tokens'];merges=values['tokenizer.ggml.merges']
            tokenizer={k:v for k,v in values.items() if k.startswith('tokenizer.ggml.')}
            vocabok=all(i<len(tokens) and tokens[i]==s for s,i in vocab.items())
            mergesok=merges==hfmerges
            model={'tag':tag,'manifest_digest':EXPECTED_DIGESTS[tag],'manifest_path':str(mp),'manifest_sha256':stream_sha(mp),
                'blob_path':str(blob),'gguf_sha256':h,'gguf_bytes':blob.stat().st_size,'architecture':values.get('general.architecture'),
                'parameter_count':show.get('model_info',{}).get('general.parameter_count'),'quantization':show.get('details',{}).get('quantization_level'),
                'gguf_metadata':gg,'metadata_fingerprint':digest(values),'tokenizer_fingerprint':digest(tokenizer),
                'template_fingerprint':digest(show.get('template','')),'parameters_fingerprint':digest(show.get('parameters','')),
                'show':show,'gguf_chat_template_fingerprint':digest(values.get('tokenizer.chat_template','')),
                'hf_vocabulary_id_mapping_match':vocabok,'hf_merge_order_match':mergesok}
            models[tag]=model;tokenizers[tag]=tokenizer
            safe.append({k:model[k] for k in ('tag','manifest_digest','gguf_sha256','gguf_bytes','architecture','parameter_count','quantization',
                'metadata_fingerprint','tokenizer_fingerprint','template_fingerprint','parameters_fingerprint','gguf_chat_template_fingerprint','hf_vocabulary_id_mapping_match','hf_merge_order_match')})
        current=http.get('/api/tags')['models']
        assert all(next(r for r in current if r['name']==tag)['digest'].removeprefix('sha256:')==EXPECTED_DIGESTS[tag] for tag in TAGS)
    finally:http.close()
    atomic_json(pr/'model_artifacts_private.json',{'models':models,'server_snapshot':metadata,'metadata_log':http.metadata_log,'generation':0})
    atomic_json(pub/'model_artifact_verification.json',{'status':'ARTIFACT_VERIFIED','models':safe,'generation':0,
        'interpretation':'Verified local deployment bytes and manifests, not training provenance or fine-tuning benefit'})
    tokenizerrows=[]
    for tag,m in models.items():
        vals=m['gguf_metadata']['values'];show=m['show']
        tokenizerrows.append({'tag':tag,'token_count':len(vals['tokenizer.ggml.tokens']),'merge_count':len(vals['tokenizer.ggml.merges']),
            'hf_vocab_id_mapping_match':m['hf_vocabulary_id_mapping_match'],'hf_merge_order_match':m['hf_merge_order_match'],
            'gguf_tokenizer_fingerprint':m['tokenizer_fingerprint'],'model':vals.get('tokenizer.ggml.model'),'pre':vals.get('tokenizer.ggml.pre'),
            'bos_id':vals.get('tokenizer.ggml.bos_token_id'),'eos_id':vals.get('tokenizer.ggml.eos_token_id'),
            'gguf_template_matches_hf':vals.get('tokenizer.chat_template')==hfconfig.get('chat_template'),
            'native_template_fingerprint':m['template_fingerprint'],'native_template_utf8_bytes':len(show.get('template','').encode('utf-8')),
            'modelfile_fingerprint':digest(show.get('modelfile','')),'default_parameters':show.get('parameters')})
    atomic_json(pub/'tokenizer_and_template_audit.json',{'rows':tokenizerrows,'all_gguf_tokenizers_identical':len({digest(t) for t in tokenizers.values()})==1,
        'native_runtime_full_equivalence':'NOT_CLAIMED; artifact vocab/merges and templates compared; guarded request budget and actual server counts required',
        'tokenizer_tree_sha256':TREE,'generation':0})
    # Reuse existing observed documents, do not rescan adjacent research trees.
    prior=read(ROOT/'outputs/private/research_evaluation/c4_27_v1/specialty_adapter_inventory_private.json')
    atomic_json(pr/'training_provenance_inventory_private.json',{'previous_inventory_sha256':stream_sha(ROOT/'outputs/private/research_evaluation/c4_27_v1/specialty_adapter_inventory_private.json'),
        'observed_document_pins':[{'path':r['path'],'sha256':r['sha256'],'current_match':Path(r['path']).is_file() and stream_sha(Path(r['path']))==r['sha256']} for r in prior['evidence_documents']],
        'hpc_access_attempted':False,'downloads':0})
    atomic_json(pub/'training_provenance_inventory.json',{'status':'UNKNOWN_TRAINING_PROVENANCE','training_overlap':'UNKNOWN_TRAINING_OVERLAP',
        'mode':'DEVELOPMENT_EXPLORATORY_EVALUATION','causal_claim':'NO_FINE_TUNING_CAUSAL_CLAIM','source':'Reused C4-27 evidence documents and exact pins; no redundant directory scan',
        'source_documents':len(prior['evidence_documents']),'original_adapter':'MISSING_IN_INSPECTED_LOCAL_PROJECT_TREES','training_logs_and_checkpoint_lineage':'NOT_VERIFIED',
        'base_model_compatibility':'Qwen2 architecture/tokenizer metadata reviewed; training base revision NOT_VERIFIED',
        'remote_checkpoint_paths':'Existing export TSV lists /data/wodntkfkd/checkpoints/...; no remote access or download',
        'hpc_followup_needed':['actual host/account/access approval','checkpoint adapter_config and weight hashes','base revision/tokenizer','training data and eval lineage'],
        'prior_documentation_conflict':'Cardiology README names MediPhi while old export TSV points to Qwen checkpoints; not resolved by deployed model tag'})
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    atomic_json(pub/'gpu_preflight.json',{'observed_at':datetime.now(timezone.utc).isoformat(),'query':'nvidia-smi name,totalMiB,usedMiB,utilization,driver','result':gpu.stdout.strip() if gpu.returncode==0 else 'NOT_MEASURED','generation':0})
    print(json.dumps({'status':'ARTIFACTS_AUDITED','vocabulary_match':[r['hf_vocab_id_mapping_match'] for r in tokenizerrows],
        'merge_match':[r['hf_merge_order_match'] for r in tokenizerrows],'template_lengths':[r['native_template_utf8_bytes'] for r in tokenizerrows],'generation':0}))


def prepare():
    pub,pr=locations(NAMESPACE);artifacts=read(pr/'model_artifacts_private.json');audit=read(pub/'tokenizer_and_template_audit.json')
    if not all(r['hf_vocab_id_mapping_match'] and r['hf_merge_order_match'] and r['model']=='gpt2' and r['pre']=='qwen2' and r['bos_id']==151643 and r['eos_id']==151645 for r in audit['rows']):raise ValueError('TOKENIZER_CONTRACT_NOT_ESTABLISHED')
    options={'temperature':0,'seed':42,'num_ctx':4096,'num_predict':1024,'top_k':40,'top_p':0.9,'min_p':0.0,
        'repeat_penalty':1.1,'repeat_last_n':64,'mirostat':0,'stop':['<|im_start|>','<|im_end|>','<|endoftext|>']}
    contract={'context':4096,'output_tokens':1024,'timeout_seconds':120,'margin_tokens':512,'options':options,
        'native_template_token_reserve':1024,'input_bytes_max':8192,'retry':0,'repair':0,'warmup':0,'sequential':True,
        'input_budget_method':'Shared HF-vocab/merge-ID-compatible Qwen BPE canonical two-message estimate +1024 native-template/BOS reserve; +512 context margin; native prompt_eval_count must not exceed bound',
        'runtime_equivalence':'NOT_CLAIMED; native templates differ and base/add_bos metadata differs; no tools/history/images, explicit system and one user message only',
        'tokenizer_artifact_match_required':True,'required_stop_policy':'Common union of Qwen turn/start/end-of-text stops; model EOS handling recorded separately',
        'model_state_policy':'Only experiment tags, at most1 loaded; approved no-prompt unload before each switch; at most4; keep_alive120s on requests, no forced final unload',
        'gpu_fit_status':'NOT_VERIFIED_UNTIL_ACTUAL_FIRST_FIXED_REQUEST_FOR_MODEL','template_assumptions':'Pinned observed templates with two text messages and no tools; no native exact token count claim; fail closed on count/context mismatch'}
    atomic_json(pub/'common_execution_contract.json',contract)
    parent,c=load_parent();bundle=build_bundle(parent,artifacts['models'],contract);ep=protocol(bundle,artifacts,contract)
    atomic_json(pr/'weight_comparison_plan.json',bundle);atomic_json(pub/'weight_comparison_protocol.json',ep)
    atomic_json(pub/'weight_comparison_plan.json',{'plan_fingerprint':bundle['plan']['fingerprint'],'parent_plan_fingerprint':bundle['plan']['parent_plan_fingerprint'],
        'run_id':RUN_ID,'registered':20,'conditions':[{k:a[k] for k in ('condition','model','model_identifier','department','quantization','benchmark_prompt_fingerprint')} for a in bundle['bindings']],
        'tasks':[{'ordinal':i,**{k:v for k,v in t.items() if k!='question_id'}} for i,t in enumerate(bundle['plan']['tasks'],1)],
        'order':bundle['plan']['task_order'],'prior32k_results_in_primary_comparison':False})
    from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter
    counter=LocalTokenCounter(TOKENIZER,TREE);qs={q['question_id']:q for q in bundle['questions']};agents={a['agent_id']:a for a in bundle['bindings']};rows=[]
    for i,t in enumerate(bundle['plan']['tasks'],1):
        user,n,bound=token_budget(qs[t['question_id']],agents[t['agent_id']],contract,counter)
        rows.append({'task_ordinal':i,'condition':t['agent_id'],'question_ordinal':t['parent_question_ordinal'],'hf_input_estimate':n,'conservative_input_bound':bound,'total_bound_with_output_and_margin':bound+1024+512})
    atomic_json(pub/'planned_token_budget.json',{'rows':rows,'estimate_scope':'HF canonical chat template, not asserted native input counts for all deployments','requested_context':4096,'native_reserve':1024,'margin':512,'generation':0})
    atomic_json(pub/'execution_approval_status.json',{'status':'AWAITING_NEW20_AND_MODEL_UNLOAD_APPROVAL','protocol_fingerprint':ep['fingerprint'],'prior_approvals_reused':False,'generation':0})
    text(pub/'actual_execution_command.md','# Prepared command; NOT EXECUTED\n\n```powershell\npython -B scripts/evaluate_weight_differentiated_agents.py dry-run\npython -B scripts/evaluate_weight_differentiated_agents.py execute --execute --allow-llm --approval outputs/private/research_evaluation/c4_28_v1/approval.json --approval-sha256 <ACTUAL_NEW_APPROVAL_SHA256>\n```\n\nRequires new20-request and named experiment-model unload authorization, including no other Ollama work. No old budget reuse.\n')
    print(json.dumps({'status':'PREPARED','protocol_fingerprint':ep['fingerprint'],'context':4096,'maximum_budget_bound':max(r['total_bound_with_output_and_margin'] for r in rows),'generation':0}))


def preflight():
    pub,pr=locations(NAMESPACE);bundle,c,artifacts,contract=load_bundle();ep=read(pub/'weight_comparison_protocol.json');validate(bundle,artifacts,contract,ep)
    http=WeightHTTP(ep['endpoint']);out={'status':'BLOCKED','generation':0,'unloads':0};private={}
    try:
        version=http.get('/api/version');tags=http.get('/api/tags');ps=http.get('/api/ps');private={'version':version,'tags':tags,'ps':ps}
        if len(ps.get('models',[]))>1 or any(r.get('name',r.get('model')) not in TAGS for r in ps.get('models',[])):raise ValueError('OTHER_OR_MULTIPLE_MODELS_LOADED')
        for tag,a in artifacts['models'].items():
            r=next(r for r in tags['models'] if r.get('name')==tag)
            if r['digest'].removeprefix('sha256:')!=a['manifest_digest']:raise ValueError('PREFLIGHT_DIGEST_CHANGED')
            if stream_sha(Path(a['blob_path']))!=a['gguf_sha256']:raise ValueError('PREFLIGHT_BLOB_CHANGED')
            show=http.show_model(tag)
            if digest(show.get('template',''))!=a['template_fingerprint'] or digest(show.get('parameters',''))!=a['parameters_fingerprint']:raise ValueError('PREFLIGHT_TEMPLATE_CHANGED')
        out.update(status='READY_FOR_NEW_APPROVAL_FIRST_TASK_LOADING_AND_SWITCHES_UNTESTED',observed_at=datetime.now(timezone.utc).isoformat(),server_version=version,
            model_digests={tag:a['manifest_digest'] for tag,a in artifacts['models'].items()},requested_common_context=4096,
            loaded_models=[{'name':r.get('name'),'context_length':r.get('context_length')} for r in ps.get('models',[])],
            actual_context_for_all_models='NOT_VERIFIED_AT_COMMON4096; must observe each after first registered task',gpu_fit='NOT_VERIFIED',server_changes=0)
    except Exception as exc:out.update(blocker=str(exc) if str(exc).isupper() else type(exc).__name__)
    finally:out['metadata_requests']=http.metadata_requests;private['metadata_log']=http.metadata_log;http.close()
    atomic_json(pr/'server_preflight_private.json',private);atomic_json(pub/'server_preflight.json',out);print(json.dumps(out))


def analyze():
    pub,pr=locations(NAMESPACE);bundle,c,artifacts,contract=load_bundle();ep=read(pub/'weight_comparison_protocol.json');root=pr/'live'/RUN_ID
    manifest=read(root/'completion_manifest.json')
    if manifest['protocol_fingerprint']!=ep['fingerprint'] or manifest['fingerprint']!=digest(manifest['files']) or any(sha(root/n)!=h for n,h in manifest['files'].items()):raise ValueError('WEIGHT_RUN_INTEGRITY')
    collection=read(root/'collection.json');rows=collection['rows'];ev=evaluate(bundle['questions'],bundle['bindings'],rows);menus=selection_menus(bundle,rows);paired=role_transitions(bundle,rows)
    records=[read(p) for p in sorted(root.glob('Request*_after.json'))];out=pub/'actual';out.mkdir(exist_ok=False)
    if len(records)!=collection['attempted']:raise ValueError('MISSING_ATTEMPT_CHECKPOINT')
    atomic_json(out/'collection_manifest.json',{k:v for k,v in collection.items() if k not in ('rows','evaluation')})
    atomic_json(out/'model_level_metrics.json',{'registered_per_condition':5,'conditions':ev['per_agent'],'all_registered':ev['all_agents'],'primary_comparison':'New common4096 run only; C4-26/27 historical32768 not pooled'})
    atomic_json(out/'role_matched_comparison.json',paired);atomic_json(out/'response_transition_analysis.json',paired)
    atomic_json(out/'oracle_menu_comparison.json',menus)
    resources=[];mismatches=[];qs={q['question_id']:q for q in bundle['questions']};refs={r['question_id']:r for r in bundle['references']}
    from app.locdi.agentclinic_benchmark_scoring import normalize
    for r in records:
        loaded=(r.get('server_after',{}).get('loaded_model') or {})
        resources.append({'task_ordinal':r['registered_task_index'],'question_ordinal':r['parent_question_ordinal'],'condition':r['condition'],'model':r['model_tag'],
            'http_status':r['http_status'],'json_valid':r['json_valid'],'schema_valid':r['schema_valid'],'state':r['scoring']['state'],
            'server_token_counts':r.get('server_token_counts'),'hf_input_estimate':r['local_hf_input_token_estimate'],'input_bound':r['conservative_input_token_bound'],
            'wall_seconds':r['wall_seconds'],'done_reason':r.get('server_done_reason'),'repetition':r.get('repetition'),'gpu':r.get('gpu',{'status':'NOT_MEASURED'}),
            'actual_context':loaded.get('context_length'),'model_size':loaded.get('size'),'size_vram':loaded.get('size_vram'),
            'offload_layers':'NOT_MEASURED; size/size_vram are server reported bytes, not exact layer placement'})
        if r['scoring']['state']=='INCORRECT_ANSWER':
            pred=r['parsed_response']['diagnosis'];ref=refs[r['question_id']]['answer']
            mismatches.append({'question_id':r['question_id'],'condition':r['condition'],'prediction':pred,'reference':ref,
                'normalized_prediction':normalize(pred),'normalized_reference':normalize(ref),'formatting_only_difference':normalize(pred)==normalize(ref),
                'abbreviation_synonym_or_localization':'REVIEW_REQUIRED','clinical_correctness':None,'review_status':'NOT_REVIEWED','score_changed':False})
    atomic_json(out/'resource_usage_comparison.json',{'rows':resources,'actual_requests':collection['attempted'],'budget_consumed':collection['generation_budget_consumed'],
        'unload_requests':collection['unload_requests'],'unload_uncertainty':collection['uncertain_or_unexpected_unload_budget_charges'],
        'reported_prompt_tokens':sum((r.get('server_token_counts') or {}).get('prompt_eval_count') or 0 for r in records),
        'reported_eval_tokens':sum((r.get('server_token_counts') or {}).get('eval_count') or 0 for r in records),
        'wall_seconds_sum':sum(r['wall_seconds'] for r in records),'output_length_failure':sum(r.get('server_done_reason')=='length' for r in records),
        'repetition_flag':sum((r.get('repetition') or {}).get('repetitive',False) for r in records)})
    atomic_json(pr/'answer_mismatch_inventory.json',{'rows':mismatches,'clinical_judgments_created':0})
    atomic_json(out/'answer_mismatch_inventory.json',{'lexical_mismatches':len(mismatches),'clinical_review':'NOT_REVIEWED','scores_changed':0,'private_file':'outputs/private/research_evaluation/c4_28_v1/answer_mismatch_inventory.json'})
    # All selection decisions use the shared complete four-condition cohort.
    values=menus['menus'];denom=menus['shared_complete_question_denominator'];mixed=values['mixed'];distinct=values['distinct_weight'];base=values['base_only']
    deploy_correct=sum(ev['per_agent'][k]['counts']['CORRECT_ANSWER'] for k in ('B','D'));base_correct=sum(ev['per_agent'][k]['counts']['CORRECT_ANSWER'] for k in ('A','C'))
    deploy_supp=sum(ev['per_agent'][k]['counts']['CORRECT_ANSWER']+ev['per_agent'][k]['counts']['INCORRECT_ANSWER'] for k in ('B','D'))
    base_supp=sum(ev['per_agent'][k]['counts']['CORRECT_ANSWER']+ev['per_agent'][k]['counts']['INCORRECT_ANSWER'] for k in ('A','C'))
    decision=research_decision(collection,ev,menus)
    text(out/'routing_research_decision.md','# Decision\n\n'+decision+'\n\nDevelopment-exposed five questions; deployment comparison includes quantization/template differences and unknown training provenance/overlap. No fine-tuning causal claim, clinical verdict or Utility fitting. Oracle is posthoc and four conditions are not the official five-specialty utility training group. No automatic additional generation.\n')
    print(json.dumps({'status':collection['status'],'attempted':collection['attempted'],'counts':ev['all_agents']['counts'],'shared_complete_questions':denom,'mixed_oracle_gap':mixed['oracle_gap'],'decision':decision}))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','prepare','preflight','dry-run','execute','analyze'],nargs='?',default='dry-run')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-llm',action='store_true');p.add_argument('--approval');p.add_argument('--approval-sha256');a=p.parse_args()
    if a.phase=='audit':return audit()
    if a.phase=='prepare':return prepare()
    if a.phase=='preflight':return preflight()
    if a.phase=='analyze':return analyze()
    if a.phase=='execute' and not a.execute:raise ValueError('EXECUTE_FLAG_REQUIRED')
    if a.phase=='dry-run' and a.execute:raise ValueError('USE_EXECUTE_SUBCOMMAND')
    pub,pr=locations(NAMESPACE);bundle,c,artifacts,contract=load_bundle();ep=read(pub/'weight_comparison_protocol.json')
    authority=WeightApproval(a.approval,a.approval_sha256) if a.approval and a.approval_sha256 else None
    result=run_collection(bundle,artifacts,contract,ep,pr/'live'/RUN_ID,execute=a.execute,allow_llm=a.allow_llm,approval=authority)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','evaluation')}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        code=str(exc) if isinstance(exc,ValueError) and str(exc).isupper() else type(exc).__name__
        print(json.dumps({'status':'BLOCKED','error_code':code,'automatic_retry':False}));sys.exit(2)
