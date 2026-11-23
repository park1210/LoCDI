"""C4-25: prepare -> metadata preflight -> explicitly approved live -> analyze."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check,locations
from app.locdi.llm_setup import digest
from app.locdi.answer_grounded_live_collection import (NAMESPACE,PARENT,CODE,load_bundle,protocol,
    OllamaHTTP,server_state,request_payload,run_collection,PinnedCollectionApproval,selection_potential,
    TOKENIZER,TREE)
from app.locdi.answer_grounded_routing_evaluation import evaluate,support_audit,rate
from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter


def execution_protocol(pub):
    final=pub/'execution_protocol_final.json'
    return read(final if final.exists() else pub/'execution_protocol.json')


def prepare():
    pub,pr=locations(NAMESPACE)
    if pub.exists() or pr.exists():raise ValueError('NAMESPACE_ALREADY_USED')
    pins=read(PARENT/'protected_hashes.json');sources=read(PARENT/'source_hashes.json')
    mp=ROOT/'outputs/research_evaluation/c4_24_v1/output_manifest.json';m=read(mp)
    assert m['fingerprint']==digest({k:v for k,v in m.items() if k!='fingerprint'})
    ip=ROOT/m['private_index_path'];assert sha(ip)==m['private_index_sha256']
    for n,h in {**m['files'],**read(ip)['files']}.items():
        if n in pins:assert pins[n]==h
        pins[n]=h
    for p in (mp,ip):pins[p.relative_to(ROOT).as_posix()]=sha(p)
    a,b=check(pins),check(sources);assert a['status']==b['status']=='PASS'
    write(pr/'protected_hashes.json',pins);write(pr/'source_hashes.json',sources)
    write(pr/'initial_preservation.json',{'protected':a,'sources':b})
    bundle,config=load_bundle();ep=protocol(bundle,config)
    write(pr/'code_hashes.json',ep['implementation_hashes']);write(pub/'execution_protocol.json',ep)
    write(pr/'validated_parent_inputs.json',{'parent_plan_fingerprint':bundle['plan']['fingerprint'],
        'private_bundle_sha256':sha(PARENT/'pilot_execution_plan.json'),'parent_inputs':len(bundle['plan']['tasks']),
        'source_pins':bundle['source_pins']})
    counter=LocalTokenCounter(TOKENIZER,TREE);qs={q['question_id']:q for q in bundle['questions']};agents={a['agent_id']:a for a in bundle['bindings']}
    rows=[]
    for i,t in enumerate(bundle['plan']['tasks'],1):
        payload,n=request_payload(qs[t['question_id']],agents[t['agent_id']],ep,counter)
        rows.append({'task_ordinal':i,'question_ordinal':(i-1)//5+1,'agent':agents[t['agent_id']]['department'],
            'request_fingerprint':digest(payload),'planned_input_tokens':n,'within_context':n+1024+512<=32768})
    assert sum(r['planned_input_tokens'] for r in rows)==19945
    write(pub/'request_budget.json',{'registered_tasks':25,'max_generation_attempts':25,'retry':0,'repair':0,'warmup':0,
        'timeout_seconds':120,'output_tokens_per_request':1024,'margin_tokens':512,'requested_context':32768,
        'planned_input_token_sum':sum(r['planned_input_tokens'] for r in rows),'rows':rows,'attempted':0,
        'uncertain_timeout_consumes_budget':True,'metadata_requests_separate':True})
    write(pub/'live_runner_implementation.json',{'status':'IMPLEMENTED_PENDING_TEST_RESULTS','files':ep['implementation_hashes'],
        'functions':['load_bundle','protocol','PinnedCollectionApproval.verify','server_state','request_payload','run_collection','selection_potential'],
        'http':'Native Ollama /api/chat via existing httpx loopback/no-proxy/no-redirect/no-retry contracts',
        'reason_native_transport':'Explicit request num_ctx and actual prompt_eval/eval token counts; old OpenAI adapter has no num_ctx setter',
        'legacy_scorer':'Unchanged C4-24 score and evaluate','raw_capture':'HTTP entity bytes stored privately, not TCP/TLS wire',
        'first_context_check':'No separate bootstrap; first approved fixed task can load model, post-request context must verify before second task'})
    print(json.dumps({'prepared':True,'protocol_fingerprint':ep['fingerprint'],'plan_fingerprint':ep['parent_plan_fingerprint'],
        'protected':len(pins),'sources':len(sources),'generation':0}),flush=True)


def preflight():
    pub,pr=locations(NAMESPACE);bundle,config=load_bundle();ep=execution_protocol(pub)
    if protocol(bundle,config)!=ep:raise ValueError('CURRENT_CODE_OR_BINDING_CHANGED')
    transport=OllamaHTTP(ep['endpoint']);r={'status':'BLOCKED','generation':0}
    private={}
    try:
        state=server_state(transport,ep);show=transport.show();private={'state':state,'show':show}
        model_max=show.get('model_info',{}).get('qwen2.context_length')
        if type(model_max) is not int or model_max<32768:raise ValueError('MODEL_CONTEXT_LIMIT')
        r={'status':'READY_FOR_AUTHORIZED_FIRST_FIXED_TASK' if state['loaded_context_status']=='PENDING_FIRST_FIXED_TASK_LOAD' else 'SERVER_CONTEXT_VERIFIED',
            'observed_at':datetime.now(timezone.utc).isoformat(),'server_version':state['server_version'],
            'model_digest':state['tag_digest'],'actual_context':(state['loaded_model'] or {}).get('context_length'),
            'actual_context_status':state['loaded_context_status'],'model_maximum_context':model_max,
            'schema_support':'NOT_CHECKED_WITH_GENERATION; local schema valid; native format sent on first task',
            'requested_context':32768,'minimum_context_needed_for_largest_input_output_margin':886+1024+512,
            'first_fixed_task_context_load_requires_explicit_approval':True,'capabilities':show.get('capabilities'),
            'server_parameters':show.get('parameters'),'generation':0,'metadata_requests':transport.metadata_requests,
            'earlier_exploratory_metadata_requests':4,'server_changes':0,'warmup':0}
    except Exception as exc:r.update(blocker=type(exc).__name__,error_code=str(exc) if str(exc).isupper() else None,metadata_requests=transport.metadata_requests)
    finally:
        private['parsed_metadata_log']=transport.metadata_log
        transport.close()
    write(pr/'server_preflight_private.json',private);write(pub/'server_preflight.json',r)
    print(json.dumps(r,ensure_ascii=True))


def execute(args):
    pub,pr=locations(NAMESPACE);bundle,config=load_bundle();ep=execution_protocol(pub)
    approval=PinnedCollectionApproval(args.approval,args.approval_sha256) if args.approval and args.approval_sha256 else None
    result=run_collection(bundle,config,ep,pr/'live'/ep['run_id'],execute=args.execute,
        allow_llm=args.allow_llm,approval=approval)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','evaluation')},ensure_ascii=True))
    return result


def analyze():
    pub,pr=locations(NAMESPACE);bundle,config=load_bundle();ep=execution_protocol(pub)
    root=pr/'live'/ep['run_id'];collection=read(root/'collection.json');rows=collection['rows']
    ev=evaluate(bundle['questions'],bundle['bindings'],rows);potential=selection_potential(bundle,rows)
    qi={q['question_id']:i for i,q in enumerate(bundle['questions'],1)};agents={a['agent_id']:a['department'] for a in bundle['bindings']}
    index={(r['question_id'],r['agent_id']):r for r in rows};aids=list(agents);qids=list(qi)
    complete=[q for q in qids if all(index[q,a]['state'] in ('CORRECT_ANSWER','INCORRECT_ANSWER','UNRESOLVED') for a in aids)]
    question_rows=[];unique=Counter();mismatches=[]
    for qid in qids:
        rs=[index[qid,a] for a in aids];is_complete=qid in complete
        sigs=[r.get('answer_signature') for r in rs];supp=[r for r in rs if r['state'] in ('CORRECT_ANSWER','INCORRECT_ANSWER')]
        suppsig={r.get('answer_signature') for r in supp};correct=[r for r in rs if r['state']=='CORRECT_ANSWER']
        if is_complete and len(correct)==1:unique[correct[0]['agent_id']]+=1
        question_rows.append({'question_ordinal':qi[qid],'states':{agents[r['agent_id']]:r['state'] for r in rs},
            'complete_collection':is_complete,'complete_scorable':len(supp)==5,
            'answer_or_abstention_disagreement':len(set(sigs))>1 if is_complete else None,
            'supported_string_disagreement':len(suppsig)>1 if is_complete else None,
            'supported_exact_match_disagreement':len({r['answer_correct'] for r in supp})>1 if is_complete else None,
            'correct_supported_success_disagreement':bool(correct) and len(correct)<5 if is_complete else None,
            'supported_unresolved_mix':bool(supp) and any(r['state']=='UNRESOLVED' for r in rs) if is_complete else None})
        for r in rs:
            if r['state']=='INCORRECT_ANSWER':mismatches.append({'question_id':qid,'agent_id':r['agent_id'],
                'state':'EXACT_LEXICAL_MISMATCH','clinical_review':'NOT_REVIEWED','reclassified':False})
    metrics={agents[a]:{**ev['per_agent'][a],'unique_correct_questions_on_complete_cohort':unique[a]} for a in aids}
    agreements=[]
    for j,a in enumerate(aids):
        for b in aids[j+1:]:agreements.append({'agents':[agents[a],agents[b]],'agreement':rate(sum(index[q,a].get('answer_signature')==index[q,b].get('answer_signature') for q in complete),len(complete))})
    counts=ev['all_agents']['counts'];summary={k:v for k,v in collection.items() if k not in ('rows','evaluation')}
    write(pub/'collection_summary.json',{**summary,'all_slots_metrics':ev['all_agents']})
    write(pub/'agent_level_metrics.json',{'status':'EVALUATED_DATASET_EXACT_MATCH_ONLY','denominator_per_agent':5,'agents':metrics,'pairwise_agreement_complete_cohort':agreements})
    write(pub/'question_level_comparison.json',{'rows':question_rows})
    write(pub/'disagreement_analysis.json',{'complete_questions':len(complete),
        **{k:sum(r[k] is True for r in question_rows) for k in ('answer_or_abstention_disagreement','supported_string_disagreement','supported_exact_match_disagreement','correct_supported_success_disagreement','supported_unresolved_mix')}})
    # Restrict every counterfactual baseline to the same fully observed cohort.
    write(pub/'oracle_and_fixed_agent_comparison.json',potential)
    saved=read(PARENT/'routing_baseline_trace.json');routes={r['question_id']:r for r in saved}
    rulerows=[];success=0;abstained=0
    for q in complete:
        route=routes[q]['rule'];dep=route.get('selected_specialty')
        if route['decision_status']=='abstained':
            abstained+=1;rulerows.append({'question_ordinal':qi[q],'status':'POLICY_ABSTAINED','agent_answer_state':None});continue
        aid=next((a for a,d in agents.items() if d==dep),None)
        if aid is None:raise ValueError('SAVED_RULE_CANDIDATE_MISMATCH')
        state=index[q,aid]['state'];success+=state=='CORRECT_ANSWER'
        rulerows.append({'question_ordinal':qi[q],'status':'SELECTED','specialty':dep,'agent_answer_state':state})
    write(pub/'rule_similarity_comparison.json',{'complete_paired_questions':len(complete),'rows':rulerows,
        'correct_supported_answer_rate':rate(success,len(complete)),'policy_abstentions':abstained,
        'partial_questions':'NOT_EVALUATED','adaptive':'MISSING_UTILITY','policy_abstention_is_not_incorrect_diagnosis':True})
    write(pub/'complete_group_analysis.json',{'complete_collection_groups':ev['complete_collection_groups'],
        'complete_scorable_groups':ev['complete_scorable_groups'],'question_count':5})
    write(pr/'answer_mismatch_review_inventory.json',{'rows':mismatches,'raw_responses_and_references':'Private live request files and unchanged C4-24 reference manifest','clinical_verdicts_created':0})
    write(pub/'answer_mismatch_review_inventory.json',{'lexical_mismatches':len(mismatches),'clinical_review':'NOT_REVIEWED','synonym_reclassification':0,
        'private_path':'outputs/private/research_evaluation/c4_25_v1/answer_mismatch_review_inventory.json'})
    support=support_audit(bundle['questions'])
    write(pub/'utility_feasibility.json',{'fit_ready':False,'fit_calls':0,'complete_scorable_groups':ev['complete_scorable_groups'],
        'oracle_gap':potential['oracle_gap'],'agent_success_disagreement_questions':sum(r['correct_supported_success_disagreement'] is True for r in question_rows),
        'potential_jaccard_support':support['scope'],'min_samples':support['min_samples'],
        'other_questions_with_positive_jaccard':[r['other_questions_with_positive_jaccard'] for r in support['rows']],
        'shared_json_fields_overlap':'Possible; no calibrated support or clinical sufficiency implied',
        'blockers':['NO_TASK_NATIVE_TARGET_AND_SPLIT_GATE','DEVELOPMENT_EXPOSED_FIVE_QUESTION_PILOT','EXACT_MATCH_NOT_CLINICAL_EQUIVALENCE']+
        (['UNRESOLVED_BLOCKS_COMPLETE_SCORABLE_GROUPS'] if counts['UNRESOLVED'] else []),
        'clinical_gate_promotion':False,'existing_adaptive':'MISSING_UTILITY'})
    if collection['stop_reason']:decision='STOP / REASSESS: technical execution contract failed; no automatic retry.'
    elif len(complete)==5 and not any(r['answer_or_abstention_disagreement'] for r in question_rows):decision='STOP / REASSESS: all five prompted roles returned the same normalized answer/abstention on each pilot question; no selection benefit observed in this pilot.'
    elif potential['oracle_gap'] and potential['oracle_gap']>0:decision='GO for a separately authorized larger development study; observed Oracle potential is not achieved routing.'
    elif counts['UNRESOLVED'] or len(mismatches):decision='CONDITIONAL GO: review exact-match/task and abstention constraints before more collection; no observed Oracle gain if gap=0.'
    else:decision='STOP / REASSESS current pilot selection benefit; no positive Oracle gap observed.'
    with (pub/'next_experiment_decision.md').open('x',encoding='utf-8') as f:f.write('# Next experiment decision\n\n'+decision+'\n\nNo Utility fitting or independent clinical inference. Five development questions cannot decide general routing usefulness.\n')
    files={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()}
    write(pub/'collection_manifest.json',{'status':collection['status'],'private_run_path':root.relative_to(ROOT).as_posix(),
        'private_completion_manifest_sha256':sha(root/'completion_manifest.json'),'registered_tasks':25,
        'plan_fingerprint':bundle['plan']['fingerprint'],'execution_protocol_fingerprint':ep['fingerprint'],'private_run_files':files})
    print(json.dumps({'status':collection['status'],'attempts':collection['attempted'],'counts':counts,'oracle_gap':potential['oracle_gap']}))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','preflight','dry-run','execute','analyze'],nargs='?',default='dry-run')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-llm',action='store_true');p.add_argument('--approval');p.add_argument('--approval-sha256')
    args=p.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='preflight':preflight()
    elif args.phase=='analyze':analyze()
    else:
        if args.phase=='execute' and not args.execute:raise ValueError('EXECUTE_FLAG_REQUIRED')
        if args.phase=='dry-run' and args.execute:raise ValueError('USE_EXPLICIT_EXECUTE_SUBCOMMAND')
        execute(args)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        code=str(exc) if isinstance(exc,ValueError) and str(exc).isupper() else type(exc).__name__
        print(json.dumps({'status':'BLOCKED','error_code':code,'automatic_retry':False}));sys.exit(2)
