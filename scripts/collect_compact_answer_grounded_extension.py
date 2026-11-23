"""C4-27 remaining15 and contract-gated aggregate; default offline dry-run."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,locations
from app.locdi.answer_grounded_compact_extension import (NAMESPACE,CODE,PREVIOUS,VERSION,PROMPT_VERSION,
    MODEL,DIGEST,TOKENIZER,TREE,atomic_json,sha,load_bundle,load_parent,protocol,prior_audit,request_payload,
    ExtensionApproval,run_collection,combine,OllamaHTTP,server_state,selection_potential,model_input,schema)
from app.locdi.answer_grounded_compact import score,parse,repetition
from app.locdi.answer_grounded_routing_evaluation import evaluate,rate
from app.locdi.agentclinic_benchmark_scoring import normalize
from app.locdi.llm_setup import digest


def text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value)


def prepare():
    pub,pr=locations(NAMESPACE)
    assert read(pr/'initial_preservation.json')['protected']['status']=='PASS'
    b,c=load_bundle();parent,_=load_parent();audit=prior_audit()
    atomic_json(pub/'unresolved_cause_audit.json',{**audit,'unresolved_count':8,
        'prompt_conflict':'No contradictory output keys found; same compact prompt explicitly allows unresolved',
        'specialty_scope':'Original specialty Focus remains and may limit diagnostic scope; causal effect NOT_ESTABLISHED',
        'possible_explanations':['SPECIALTY_SCOPE_EFFECT_POSSIBLE','MODEL_FREE_DIAGNOSIS_LIMITATION_POSSIBLE','INFORMATION_INTERPRETATION_POSSIBLE','LEGITIMATE_ABSTENTION_POSSIBLE'],
        'observed_missing_fields':0,'observed_saved_input_truncation':False,'unresolved_correctness':None})
    inventory=read(PREVIOUS/'answer_mismatch_review_inventory.json');private=[];public=[]
    for i,row in enumerate(inventory['rows'],1):
        pred,ref=row['prediction'],row['dataset_declared_reference']
        r={**row,'normalized_prediction':normalize(pred),'normalized_reference':normalize(ref),
            'normalized_equal':normalize(pred)==normalize(ref),'formatting_only_difference':False,
            'abbreviation_equivalence':'NOT_ESTABLISHED','synonym_equivalence':'REVIEW_REQUIRED',
            'clinical_equivalence':'REVIEW_REQUIRED','source_reference_ambiguity':'NOT_REVIEWED',
            'possible_type':'LOCALIZATION_LABEL_VS_DISEASE_REVIEW_REQUIRED' if i==1 else 'DISTINCT_DIAGNOSTIC_TERMS_REVIEW_REQUIRED',
            'original_score_unchanged':True}
        private.append(r);public.append({'mismatch_ordinal':i,'task_ordinal':row['task_ordinal'],
            'normalized_equal':r['normalized_equal'],'formatting_only_difference':False,'possible_type':r['possible_type'],
            'clinical_equivalence':'REVIEW_REQUIRED','original_exact_state':'INCORRECT_ANSWER','score_changed':False})
    atomic_json(pr/'existing_answer_mismatch_review.json',{'rows':private,'clinical_judgments_created':0})
    atomic_json(pub/'existing_answer_mismatch_review.json',{'rows':public,'actual_strings':'PRIVATE_ONLY','clinical_judgments_created':0})
    if audit['status']!='PASS':raise ValueError('SOURCE_AUDIT_BLOCKED')
    ep=protocol(b,c);atomic_json(pub/'compact_extension_protocol.json',ep);atomic_json(pr/'compact_extension_plan.json',b)
    atomic_json(pub/'compact_extension_plan.json',{'run_id':ep['run_id'],'plan_fingerprint':b['plan']['fingerprint'],
        'task_order_fingerprint':b['plan']['task_order_fingerprint'],'registered':15,'parent_question_ordinals':[3,4,5],
        'tasks':[{'ordinal':i,'parent_task_ordinal':i+10,'parent_question_ordinal':(i-1)//5+3,
            'specialty':b['bindings'][(i-1)%5]['department'],**{k:v for k,v in t.items() if k not in ('question_id','agent_id')}} for i,t in enumerate(b['plan']['tasks'],1)]})
    from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter
    counter=LocalTokenCounter(TOKENIZER,TREE);qi={q['question_id']:q for q in b['questions']};ai={a['agent_id']:a for a in b['bindings']}
    tokens=[]
    for i,t in enumerate(b['plan']['tasks'],1):
        payload,n=request_payload(qi[t['question_id']],ai[t['agent_id']],ep,counter)
        tokens.append({'ordinal':i,'planned_input_tokens':n,'request_fingerprint':digest(payload)})
    atomic_json(pub/'planned_token_budget.json',{'rows':tokens,'input_token_sum':sum(r['planned_input_tokens'] for r in tokens),
        'max_generation':15,'output_limit':1024,'margin':512,'context':32768,'generation':0})
    atomic_json(pub/'execution_approval_status.json',{'status':'AWAITING_NEW_EXTENSION_APPROVAL','protocol_fingerprint':ep['fingerprint'],
        'generation_authorized':False,'prior_10_or_25_budget_reused':False})
    text(pub/'actual_execution_command.md','# Prepared CLI, not executed\n\n```powershell\npython -B scripts/collect_compact_answer_grounded_extension.py dry-run\npython -B scripts/collect_compact_answer_grounded_extension.py execute --execute --allow-llm --approval outputs/private/research_evaluation/c4_27_v1/approval.json --approval-sha256 <ACTUAL_NEW_APPROVAL_SHA256>\n```\n\nActual approval file and fresh explicit authorization required. No C4-26 rerun/resume.\n')
    print(json.dumps({'status':'PREPARED','protocol_fingerprint':ep['fingerprint'],'task_order_fingerprint':ep['task_order_fingerprint'],'generation':0}))


def preflight():
    pub,pr=locations(NAMESPACE);b,c=load_bundle();ep=read(pub/'compact_extension_protocol.json')
    if protocol(b,c)!=ep or prior_audit()['status']!='PASS':raise ValueError('CURRENT_PROTOCOL_OR_SOURCE_CHANGED')
    http=OllamaHTTP(ep['endpoint']);priv={};out={'status':'BLOCKED','generation':0}
    try:
        state=server_state(http,ep);show=http.show();priv={'state':state,'show':show}
        maximum=show.get('model_info',{}).get('qwen2.context_length')
        if type(maximum) is not int or maximum<32768:raise ValueError('MODEL_MAX_CONTEXT')
        out.update(status='READY_FOR_NEW_APPROVAL',observed_at=datetime.now(timezone.utc).isoformat(),
            server_version=state['server_version'],model_digest=state['tag_digest'],actual_context=(state['loaded_model'] or {}).get('context_length'),
            actual_context_status=state['loaded_context_status'],model_maximum_context=maximum,server_changes=0,warmup=0,
            comparison_with_c4_26='Same pinned model/settings; loaded context must match or first approved fixed task must verify it')
    except Exception as exc:out.update(blocker=str(exc) if str(exc).isupper() else type(exc).__name__)
    finally:priv['metadata_log']=http.metadata_log;out['metadata_requests']=http.metadata_requests;http.close()
    atomic_json(pr/'server_preflight_private.json',priv);atomic_json(pub/'server_preflight.json',out);print(json.dumps(out))


def inventory():
    pub,pr=locations(NAMESPACE);base=ROOT.parents[1]/'003_SoftwareCapstonDesign';research=ROOT.parent
    cmd=['rg','--files','--hidden','--no-ignore','-g','!**/.git/**','-g','!**/.venv*/**','-g','!**/node_modules/**']
    for pattern in ('adapter_config.json','adapter_model.safetensors','adapter_model.bin','*.gguf','*Modelfile*','*trainer_state*.json','*train_results*.json'):
        cmd+=['-g',pattern]
    scan=subprocess.run(cmd+[str(base),str(research)],capture_output=True,text=True,encoding='utf-8',timeout=90)
    if scan.returncode not in (0,1):raise ValueError('ADAPTER_SCAN_ERROR')
    files=[Path(n) for n in scan.stdout.splitlines() if n.strip()]
    found=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
    evidence=[]
    for project in ('008_seraph-cardiology_lora','009_seraph-neurology_lora','017_seraph-pulmonology_lora','020_seraph-nephrology_lora','023_seraph-endocrinology_lora','027_seraph-export_ollama'):
        p=base/project
        names=[n for n in p.rglob('*') if n.is_file() and n.suffix in ('.md','.yaml','.yml','.json','.tsv') and '.git' not in n.parts]
        for n in names:
            if n.stat().st_size<524288:evidence.append({'path':str(n),'sha256':sha(n),'text':n.read_text(encoding='utf-8',errors='replace')})
    b,c=load_bundle();http=OllamaHTTP(protocol(b,c)['endpoint']);metadata={};shows=[]
    try:
        metadata={p:http.get(p) for p in ('/api/version','/api/tags','/api/ps')}
        tags=metadata['/api/tags'].get('models',[])
        for tag in tags[:32]:
            name=tag.get('name')
            if not name:continue
            response=http.http.post(http.base+'/api/show',json={'model':name},timeout=8)
            value=response.json() if response.status_code==200 else {}
            shows.append({'name':name,'tag_digest':tag.get('digest'),'http_status':response.status_code,'body_sha256':__import__('hashlib').sha256(response.content).hexdigest(),'parsed':value})
    finally:http.close()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    atomic_json(pr/'specialty_adapter_inventory_private.json',{'scan_roots':[str(base),str(research)],'scan_command':cmd+[str(base),str(research)],
        'files':found,'evidence_documents':evidence,'server_metadata':metadata,'model_shows':shows,
        'gpu_query':{'returncode':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr},'generation':0})
    publicmodels=[]
    for row in shows:
        v=row['parsed'];info=v.get('model_info',{});mf=v.get('modelfile','')
        publicmodels.append({'name':row['name'],'tag_digest':row['tag_digest'],'metadata_http_status':row['http_status'],
            'model_details':v.get('details'),'capabilities':v.get('capabilities'),'declared_architecture':info.get('general.architecture'),
            'adapter_directive_present':any(line.lstrip().startswith('ADAPTER ') for line in mf.splitlines()),
            'artifact_status':'ARTIFACT_FOUND' if row['http_status']==200 else 'NOT_VERIFIED',
            'training_provenance':'NOT_VERIFIED','base_model_compatibility':'NOT_VERIFIED','inference_ready':'NOT_VERIFIED_NO_GENERATION',
            'benchmark_training_overlap':'UNKNOWN_NO_RAW_TRAINING_CORPUS'})
    atomic_json(pub/'specialty_adapter_inventory.json',{'scan_scope':'Current project and adjacent thesis/capstone project trees; ignored files included except git/venvs/node_modules; no remote HPC access',
        'adapter_config_count':sum(Path(r['path']).name=='adapter_config.json' for r in found),
        'adapter_weight_count':sum(Path(r['path']).name in ('adapter_model.safetensors','adapter_model.bin') for r in found),
        'gguf_export_count':sum(Path(r['path']).suffix=='.gguf' for r in found),'local_artifact_files':len(found),
        'training_source_documents':len(evidence),'models':publicmodels,'gpu_query':gpu.stdout.strip() if gpu.returncode==0 else 'NOT_CHECKED',
        'python_packages_available':{n:importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft','safetensors')},
        'generation':0,'downloads':0,'merge':0,'training':0,'registry_changes':0,
        'provenance_warning':'README/config/export path is a declaration, not completed training proof. Cardiology README names MediPhi while exporter lists Qwen checkpoint paths; lineage unresolved.'})
    text(pub/'alternative_model_feasibility.md','# Alternative model feasibility\n\nLocal model tags and metadata are inventoried without generation. A tag or callable show endpoint does not establish training completion, clinical quality, inference readiness, or compatibility with the Qwen-pinned registry.\n\nNo local adapter config/weights or GGUF exports were found in the inspected project trees unless explicitly counted in the inventory. Adjacent export config lists external /data/... HPC paths; these are not locally acquired artifacts. Cardiology documentation declares a MediPhi base whereas export configuration names Qwen checkpoints. Do not merge or bind without actual config, weights, base revision, tokenizer, dataset lineage and overlap review. Raw training/held-out overlap remains UNKNOWN.\n\nNext options are metadata/provenance verification of an existing specialty tag, operator-provided verified adapter artifacts, or a separately approved alternative model pilot. No download, merge, training, registry change or alternative model generation occurred.\n')
    print(json.dumps({'artifact_files':len(found),'model_tags_reviewed':len(shows),'generation':0}))


def analyze():
    pub,pr=locations(NAMESPACE);b,c=load_bundle();parent,_=load_parent();ep=read(pub/'compact_extension_protocol.json');root=pr/'live'/ep['run_id']
    oldb=read(PREVIOUS/'mini_pilot_plan.json');oldep=read(ROOT/'outputs/research_evaluation/c4_26_v1/mini_pilot_protocol.json')
    oldrun=PREVIOUS/'live/c4_26_compact_pilot_001';records=[]
    for run,expected in ((oldrun,oldep),(root,ep)):
        m=read(run/'completion_manifest.json')
        if m['fingerprint']!=digest(m['files']) or m['protocol_fingerprint']!=expected['fingerprint'] or any(sha(run/n)!=h for n,h in m['files'].items()):raise ValueError('RUN_MANIFEST_INTEGRITY')
        records.extend(read(p) for p in sorted(run.glob('Request*_after.json')))
    oldcol=read(oldrun/'collection.json');col=read(root/'collection.json')
    combined=combine(parent,oldb,oldep,oldcol['rows'],b,ep,col['rows']);rows=combined['rows'];ev=combined['evaluation'];out=pub/'actual';out.mkdir(exist_ok=False)
    atomic_json(pr/'combined_25_task_rows.json',{'rows':rows,'source_runs':[{'path':str(r),'manifest_sha256':sha(r/'completion_manifest.json')} for r in (oldrun,root)]})
    newrecords=[read(p) for p in sorted(root.glob('Request*_after.json'))]
    def technical(rs,n,attempted):
        counts={'http_success':sum(r['http_status']==200 for r in rs),'json_valid':sum(r['json_valid'] for r in rs),
            'schema_valid':sum(r['schema_valid'] for r in rs),'output_length_failure':sum(r.get('server_done_reason')=='length' for r in rs),
            'repetition_flag':sum(r.get('repetition',{}).get('repetitive',False) for r in rs)}
        return {'registered':n,'attempted':attempted,'counts':counts,'registered_rates':{k:rate(v,n) for k,v in counts.items()},
            'attempted_rates':{k:rate(v,attempted) for k,v in counts.items()},
            'prompt_tokens':sum(r.get('server_token_counts',{}).get('prompt_eval_count') or 0 for r in rs),
            'eval_tokens':sum(r.get('server_token_counts',{}).get('eval_count') or 0 for r in rs),'wall_seconds_sum':sum(r['wall_seconds'] for r in rs)}
    atomic_json(out/'extension_collection_metrics.json',{'status':col['status'],'technical':technical(newrecords,15,col['attempted']),'outcomes':col['evaluation']['all_agents']})
    atomic_json(out/'combined_25_task_metrics.json',{'contract_compatible':True,'technical':technical(records,25,oldcol['attempted']+col['attempted']),
        'outcomes':ev['all_agents'],'source_runs':2,'c4_25_v1_responses_included':False})
    dep={a['agent_id']:a['department'] for a in b['bindings']};idx={(r['question_id'],r['agent_id']):r for r in rows};aids=list(dep)
    complete=[q['question_id'] for q in parent['questions'] if next(g for g in ev['groups'] if g['question_id']==q['question_id'])['complete_collection']]
    from app.locdi.answer_grounded_routing_evaluation import metrics
    qrows=[];unique=Counter()
    for i,q in enumerate(parent['questions'],1):
        rs=[idx[q['question_id'],a] for a in aids];full=q['question_id'] in complete;supported=[r for r in rs if r['state'] in ('CORRECT_ANSWER','INCORRECT_ANSWER')];correct=[r for r in rs if r['state']=='CORRECT_ANSWER']
        if full and len(correct)==1:unique[correct[0]['agent_id']]+=1
        qrows.append({'parent_question_ordinal':i,'states':{dep[r['agent_id']]:r['state'] for r in rs},'complete_collection':full,
            'complete_scorable':len(supported)==5,'all_unresolved':all(r['state']=='UNRESOLVED' for r in rs) if full else None,
            'some_supported':bool(supported) and len(supported)<5 if full else None,
            'answer_or_abstention_disagreement':len({r.get('answer_signature') for r in rs})>1 if full else None,
            'supported_string_disagreement':len({r.get('answer_signature') for r in supported})>1 if full else None,
            'supported_correctness_disagreement':len({r['answer_correct'] for r in supported})>1 if full else None,
            'correct_supported_success_disagreement':bool(correct) and len(correct)<5 if full else None,
            'correct_provider_count':len(correct) if full else None})
    atomic_json(out/'question_level_comparison.json',{'rows':qrows})
    atomic_json(out/'combined_agent_metrics.json',{'registered_denominator_per_agent':5,'agents':{dep[a]:ev['per_agent'][a] for a in aids},
        'complete_paired_questions':len(complete),'paired_agent_metrics':{dep[a]:metrics([r for r in rows if r['agent_id']==a and r['question_id'] in complete]) for a in aids},
        'unique_correct_questions':{dep[a]:unique[a] for a in aids}})
    atomic_json(out/'agent_disagreement_analysis.json',{'complete_paired_questions':len(complete),
        **{k:rate(sum(q[k] is True for q in qrows),len(complete)) for k in ('answer_or_abstention_disagreement','supported_string_disagreement','supported_correctness_disagreement','correct_supported_success_disagreement')},
        'multiple_correct_agent_questions':sum(q['correct_provider_count'] is not None and q['correct_provider_count']>1 for q in qrows),
        'no_correct_provider_questions':sum(q['correct_provider_count']==0 for q in qrows),
        'complete_scorable_groups':ev['complete_scorable_groups'],'all_unresolved_groups':sum(q['all_unresolved'] is True for q in qrows),
        'partially_supported_groups':sum(q['some_supported'] is True for q in qrows)})
    routes={r['question_id']:r for r in read(ROOT/'outputs/private/research_evaluation/c4_24_v1/routing_baseline_trace.json')};rule=[];success=0;abstained=0
    for i,q in enumerate(parent['questions'],1):
        route=routes[q['question_id']]['rule'];department=route.get('selected_specialty');full=q['question_id'] in complete
        status='POLICY_ABSTAINED' if route['decision_status']=='abstained' else 'SELECTED'
        aid=next((a for a,d in dep.items() if d==department),None) if status=='SELECTED' else None
        if status=='SELECTED' and aid is None:raise ValueError('RULE_CANDIDATE_MISMATCH')
        state=idx[q['question_id'],aid]['state'] if aid else None
        if full:success+=state=='CORRECT_ANSWER';abstained+=status=='POLICY_ABSTAINED'
        rule.append({'parent_question_ordinal':i,'policy_status':status,'specialty':department,'agent_state':state,'paired_evaluated':full})
    atomic_json(out/'oracle_and_fixed_agent_comparison.json',{**combined['potential'],'rule_similarity':{'rows':rule,'correct_supported_answer_rate':rate(success,len(complete)),
        'policy_abstentions_paired':abstained,'incomplete_questions':'NOT_EVALUATED'},'adaptive':'MISSING_UTILITY'})
    gap=combined['potential']['oracle_gap'];counts=ev['all_agents']['counts']
    decision='REASSESS_MODEL' if gap==0 and counts['CORRECT_ANSWER']==0 else ('GO_EXPLORATORY_SELECTION' if gap and gap>0 else 'CONDITIONAL_GO')
    if col['stop_reason']:decision='STOP_TECHNICAL_ERROR'
    atomic_json(out/'utility_feasibility.json',{'fit_ready':False,'fit_calls':0,'complete_scorable_groups':ev['complete_scorable_groups'],
        'oracle_gap':gap,'decision':decision,'clinical_gate_used':False,'unresolved_correctness_kept_null':True,
        'blocker_codes':['DEVELOPMENT_EXPOSED_FIVE_QUESTION_PILOT','NO_TASK_NATIVE_TRAIN_VALIDATION_GATE','EXACT_MATCH_NOT_CLINICAL_REFERENCE']+
        (['NO_COMPLETE_SCORABLE_GROUP'] if not ev['complete_scorable_groups'] else [])})
    atomic_json(out/'collection_manifest.json',{'status':col['status'],'registered':15,'attempted':col['attempted'],'stop_reason':col['stop_reason'],
        'extension_completion_manifest_sha256':sha(root/'completion_manifest.json'),'prior_completion_manifest_sha256':sha(oldrun/'completion_manifest.json'),
        'two_original_runs_preserved':True,'combined_aggregate_separate':True})
    text(out/'next_action_decision.md','# Next decision\n\n'+decision+'\n\nFive development questions cannot establish general model capability. No clinical relabeling, fitting, automatic model swap or further generation. Inspect actual provenance and paired selection evidence before expanding.\n')
    print(json.dumps({'status':col['status'],'extension_attempted':col['attempted'],'combined_counts':counts,'oracle_gap':gap,'decision':decision}))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','preflight','inventory','dry-run','execute','analyze'],nargs='?',default='dry-run')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-llm',action='store_true');p.add_argument('--approval');p.add_argument('--approval-sha256');a=p.parse_args()
    if a.phase=='prepare':return prepare()
    if a.phase=='preflight':return preflight()
    if a.phase=='inventory':return inventory()
    if a.phase=='analyze':return analyze()
    if a.phase=='execute' and not a.execute:raise ValueError('EXECUTE_FLAG_REQUIRED')
    if a.phase=='dry-run' and a.execute:raise ValueError('USE_EXECUTE_SUBCOMMAND')
    pub,pr=locations(NAMESPACE);b,c=load_bundle();ep=read(pub/'compact_extension_protocol.json')
    approval=ExtensionApproval(a.approval,a.approval_sha256) if a.approval and a.approval_sha256 else None
    result=run_collection(b,c,ep,pr/'live'/ep['run_id'],execute=a.execute,allow_llm=a.allow_llm,approval=approval)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','evaluation')}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        code=str(exc) if isinstance(exc,ValueError) and str(exc).isupper() else type(exc).__name__
        print(json.dumps({'status':'BLOCKED','error_code':code,'automatic_retry':False}));sys.exit(2)
