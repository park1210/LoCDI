"""C4-26 explicit prepare/preflight/dry-run/execute/analyze; default offline."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.answer_grounded_compact_collection import (NAMESPACE,CODE,load_bundle,protocol,request_payload,
    CompactApproval,run_collection,stability,server_state,OllamaHTTP,TOKENIZER,TREE,atomic_json,sha,selection_potential)
from app.locdi.answer_grounded_compact import schema,PROMPT,PROMPT_VERSION,VERSION,REPETITION_RULE,repetition
from app.locdi.answer_grounded_routing_evaluation import evaluate
from app.locdi.answer_grounded_live_collection import load_bundle as parent_bundle
from app.locdi.llm_setup import digest
from scripts.audit_research_gaps import locations,read,check


def text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value)


def prepare():
    pub,pr=locations(NAMESPACE)
    # Protection snapshot is made before preparation. It must already verify PASS.
    assert read(pr/'initial_preservation.json')['protected']['status']=='PASS'
    if (pub/'mini_pilot_protocol.json').exists():raise ValueError('PREPARATION_ALREADY_EXISTS')
    b,c=load_bundle();ep=protocol(b,c);parent,_=parent_bundle()
    atomic_json(pub/'mini_pilot_protocol.json',ep);atomic_json(pr/'mini_pilot_plan.json',b)
    atomic_json(pub/'mini_pilot_plan.json',{'plan_fingerprint':b['plan']['fingerprint'],
        'parent_plan_fingerprint':b['plan']['parent_plan_fingerprint'],'run_id':ep['run_id'],
        'registered_tasks':10,'selection':'First two of unchanged C4-24 fixed five questions',
        'tasks':[{'ordinal':i,'question_ordinal':(i-1)//5+1,'specialty':b['bindings'][(i-1)%5]['department'],
            'input_fingerprint':t['input_fingerprint'],'prompt_fingerprint':t['prompt_fingerprint'],
            'schema_fingerprint':t['schema_fingerprint']} for i,t in enumerate(b['plan']['tasks'],1)]})
    atomic_json(pub/'compact_prompt_spec.json',{'version':PROMPT_VERSION,'common_prompt':PROMPT,
        'common_fingerprint':digest(PROMPT),'bindings':[{'department':a['department'],
            'parent_prompt_fingerprint':a['parent_benchmark_prompt_fingerprint'],
            'prompt_fingerprint':a['benchmark_prompt_fingerprint']} for a in b['bindings']],
        'original_prompts_modified':False,'role_focus_preserved':True})
    atomic_json(pub/'compact_schema_spec.json',{'version':VERSION,'schema':schema(),
        'semantic_rules':['supported requires nonempty single diagnosis','unresolved diagnosis is null',
            'no extras/duplicates/fences/text','no fabricated rationale or answer_id'],
        'single_diagnosis_limit':'Explicit alternatives/lists rejected; arbitrary natural-language diagnostic equivalence is not clinically adjudicated',
        'native_schema_keyword_support':'NOT_ESTABLISHED; payload format object verified locally, actual compact support requires live response',
        'normalization':'Existing agentclinic_benchmark_scoring.normalize unchanged','rationale_evaluation':'NOT_SUPPORTED'})
    conflicts=[]
    for old,new in zip(parent['bindings'],b['bindings']):
        s=old['system_prompt'];conflicts.append({'specialty':old['department'],
            'legacy_rationale_required':'Every decision needs a brief evidence-based rationale.' in s,
            'legacy_four_field_contract':'answer_id, diagnosis, rationale' in s,
            'legacy_evidence_field_required':False,'legacy_action_request_required':False,
            'legacy_multiple_diagnoses_required':False,'legacy_clinical_catalog_reference':False,
            'legacy_limited_certainty_syndrome_instruction':'If certainty is limited' in s,
            'resolution':'New compact prefix replaces old four-field/common rationale contract; preserve department Focus except limited-certainty syndrome fallback, use explicit unresolved instead',
            'new_no_rationale_instruction': 'Do not include reasoning' in new['system_prompt']})
    atomic_json(pub/'prompt_conflict_audit.json',{'rows':conflicts,
        'old_v1_internal_schema_conflict':'No four-field conflict found: both v1 task and common specialty prompt required rationale',
        'new_contract_conflict_if_old_concatenated':'Would conflict; new builder does not concatenate old common/task prefixes',
        'old_files_unchanged':True})
    run=ROOT/'outputs/private/research_evaluation/c4_25_v1/live/c4_24_pilot_001';audit=[];safe=[]
    from collections import Counter as Counts
    for i in (1,2):
        before=read(run/('Request%03d_before.json'%i));after=read(run/('Request%03d_after.json'%i));s=after['raw_model_content']
        tokens=re.findall(r'\w+',s.casefold());grams=Counts(tuple(tokens[j:j+6]) for j in range(max(0,len(tokens)-5)))
        try:json.loads(s);ended=True
        except ValueError:ended=False
        field_match=re.search(r'"rationale"\s*:\s*"',s);start=field_match.end() if field_match else None
        repeated=repetition(s);spans=list(re.finditer(r'\w+',s.casefold()))
        repeat_char=spans[repeated['first_repeat_token_offset']].start() if repeated['first_repeat_token_offset'] is not None else None
        row={'task_ordinal':i,'specialty':after['specialty'],'repetition':repeated,
            'repetition_field':'rationale' if repeat_char is not None and start is not None and repeat_char>=start else None,
            'first_repeat_character_offset':repeat_char,'rationale_value_character_offset':start,
            'done_reason':after['server_done_reason'],'eval_tokens':after['server_token_counts']['eval_count'],
            'output_limit_reached':after['server_done_reason']=='length','json_terminated':ended,
            'format_schema_transmitted':before['payload']['format'],
            'schema_transmission':'Native /api/chat top-level format receives schema object; stream=false; semantic constraints checked locally',
            'model_content_fingerprint':digest(s),'entity_sha256':after['http_entity_sha256'],
            'rationale_required_in_task_and_specialty':True}
        audit.append({**row,'most_common_repeated_six_tokens':[{'tokens':list(k),'count':n} for k,n in grams.most_common(4)],
            'original_content':s,'request_payload':before['payload']})
        safe.append(row)
    atomic_json(pr/'previous_failure_root_cause_private.json',{'rows':audit,'raw_original_run_unchanged':True})
    atomic_json(pub/'previous_failure_root_cause.json',{'rows':safe,
        'observed_mechanism':'Repeated passages inside rationale in request2, unclosed string/object at1024 cap',
        'causal_limit':'No proof rationale alone caused failure; prompt and schema change jointly; no timeout/context/GPU shortage demonstrated',
        'repetition_rule':REPETITION_RULE,'comparison_to_c4_25_counts':'Same casefold word six-gram duplicate-occurrence rule; new threshold>=3 pinned before results'})
    from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter
    counter=LocalTokenCounter(TOKENIZER,TREE);qi={q['question_id']:q for q in b['questions']};ai={a['agent_id']:a for a in b['bindings']}
    tokenrows=[]
    for i,t in enumerate(b['plan']['tasks'],1):
        payload,n=request_payload(qi[t['question_id']],ai[t['agent_id']],ep,counter)
        tokenrows.append({'ordinal':i,'planned_input_tokens':n,'request_fingerprint':digest(payload)})
    atomic_json(pub/'planned_token_budget.json',{'rows':tokenrows,'max_generation':10,'output_per_request':1024,'margin':512,
        'input_token_sum':sum(t['planned_input_tokens'] for t in tokenrows),'generation':0})
    atomic_json(pub/'execution_approval_status.json',{'status':'AWAITING_NEW_EXPLICIT_APPROVAL','generation_authorized':False,
        'protocol_fingerprint':ep['fingerprint'],'previous_c4_25_approval_reused':False,'generation_count':0})
    text(pub/'actual_execution_command.md','# 실행 전 승인 필요\n\n```powershell\n'
        'python -B scripts/collect_compact_answer_grounded.py dry-run\n'
        'python -B scripts/collect_compact_answer_grounded.py execute --execute --allow-llm --approval outputs/private/research_evaluation/c4_26_v1/approval.json --approval-sha256 <ACTUAL_APPROVAL_FILE_SHA256>\n'
        '```\n\n승인 파일이 아직 없으며 두 번째 명령은 미실행이다. 기존 Run 재개가 아니다.\n')
    print(json.dumps({'status':'PREPARED','protocol_fingerprint':ep['fingerprint'],'planned_tokens':sum(t['planned_input_tokens'] for t in tokenrows),'generation':0}))


def preflight():
    pub,pr=locations(NAMESPACE);b,c=load_bundle();ep=read(pub/'mini_pilot_protocol.json')
    if protocol(b,c)!=ep:raise ValueError('CODE_OR_PROTOCOL_CHANGED')
    http=OllamaHTTP(ep['endpoint']);out={'status':'BLOCKED','generation':0};priv={}
    try:
        state=server_state(http,ep);show=http.show();priv={'state':state,'show':show}
        maximum=show.get('model_info',{}).get('qwen2.context_length')
        if not isinstance(maximum,int) or maximum<32768:raise ValueError('MODEL_MAX_CONTEXT')
        out.update(status='READY_FOR_APPROVAL' if state['loaded_context_status']=='PENDING_FIRST_FIXED_TASK_LOAD' else 'CONTEXT_VERIFIED_AWAITING_APPROVAL',
            observed_at=datetime.now(timezone.utc).isoformat(),model_digest=state['tag_digest'],server_version=state['server_version'],
            actual_context=(state['loaded_model'] or {}).get('context_length'),actual_context_status=state['loaded_context_status'],
            model_maximum_context=maximum,compact_schema_live_validation='NOT_RUN',warmup=0,server_changes=0)
    except Exception as exc:out.update(error_type=type(exc).__name__,blocker=str(exc) if str(exc).isupper() else type(exc).__name__)
    finally:
        priv['metadata_log']=http.metadata_log;out['metadata_requests']=http.metadata_requests;http.close()
    atomic_json(pr/'server_preflight_private.json',priv);atomic_json(pub/'server_preflight.json',out)
    print(json.dumps(out,ensure_ascii=True))


def analyze():
    pub,pr=locations(NAMESPACE);b,c=load_bundle();ep=read(pub/'mini_pilot_protocol.json');run=pr/'live'/ep['run_id']
    m=read(run/'completion_manifest.json')
    if m['fingerprint']!=digest(m['files']) or m['protocol_fingerprint']!=ep['fingerprint'] or any(sha(run/n)!=h for n,h in m['files'].items()):raise ValueError('RUN_MANIFEST_INTEGRITY')
    col=read(run/'collection.json');records=[read(p) for p in sorted(run.glob('Request*_after.json'))]
    out=pub/'actual';out.mkdir(exist_ok=False);ev=evaluate(b['questions'],b['bindings'],col['rows']);potential=selection_potential(b,col['rows'])
    atomic_json(out/'collection_manifest.json',{'status':col['status'],'private_run_completion_sha256':sha(run/'completion_manifest.json'),
        'protocol_fingerprint':ep['fingerprint'],'registered':10,'attempted':col['attempted'],'stop_reason':col['stop_reason']})
    atomic_json(out/'output_stability_metrics.json',stability(b,col,records))
    dep={a['agent_id']:a['department'] for a in b['bindings']}
    complete={g['question_id'] for g in ev['groups'] if g['complete_collection']}
    paired={a:sum(r['state']=='CORRECT_ANSWER' for r in col['rows'] if r['agent_id']==a and r['question_id'] in complete) for a in dep}
    paired_rows=[]
    for q in b['questions']:
        if q['question_id'] not in complete:continue
        rs=[r for r in col['rows'] if r['question_id']==q['question_id']]
        supp=[r for r in rs if r['state'] in ('CORRECT_ANSWER','INCORRECT_ANSWER')]
        from app.locdi.answer_grounded_routing_evaluation import metrics
        paired_rows.append({'question_ordinal':b['questions'].index(q)+1,
            'supported_unresolved_disagreement':bool(supp) and len(supp)<5,
            'supported_answer_string_disagreement':len({r.get('answer_signature') for r in supp})>1,
            'supported_exact_match_disagreement':len({r['answer_correct'] for r in supp})>1,
            'correct_supported_success_disagreement':any(r['state']=='CORRECT_ANSWER' for r in rs) and not all(r['state']=='CORRECT_ANSWER' for r in rs)})
    from app.locdi.answer_grounded_routing_evaluation import metrics
    atomic_json(out/'agent_level_metrics.json',{'registered_per_agent':2,'all_registered':{dep[a]:v for a,v in ev['per_agent'].items()},
        'complete_paired_question_denominator':len(complete),'paired_correct_supported_counts':{dep[a]:v for a,v in paired.items()} if complete else None,
        'paired_agent_metrics':{dep[a]:metrics([r for r in col['rows'] if r['agent_id']==a and r['question_id'] in complete]) for a in dep},
        'paired_question_disagreements':paired_rows,'answer_disagreement':ev['agent_disagreement_rate'],'clinical_accuracy':'NOT_EVALUATED'})
    atomic_json(out/'oracle_gap_analysis.json',potential)
    qs={q['question_id']:i for i,q in enumerate(b['questions'],1)}
    atomic_json(out/'complete_group_analysis.json',{'complete_collection_groups':ev['complete_collection_groups'],
        'complete_scorable_groups':ev['complete_scorable_groups'],
        'groups':[{**{k:v for k,v in g.items() if k!='question_id'},'question_ordinal':qs[g['question_id']]} for g in ev['groups']]})
    comp=[]
    previous=ROOT/'outputs/private/research_evaluation/c4_25_v1/live/c4_24_pilot_001'
    for i in (1,2):
        old=read(previous/('Request%03d_after.json'%i));new=next((r for r in records if r['registered_task_index']==i),None)
        def brief(r):
            if r is None:return {'status':'NOT_ATTEMPTED'}
            return {'schema_valid':r['schema_valid'],'state':r['scoring']['state'],'eval_tokens':r.get('server_token_counts',{}).get('eval_count'),
                'prompt_tokens':r.get('server_token_counts',{}).get('prompt_eval_count'),'done_reason':r.get('server_done_reason'),
                'repetition':repetition(r['raw_model_content']) if 'raw_model_content' in r else None,'wall_seconds':r['wall_seconds']}
        comp.append({'specialty':old['specialty'],'question_ordinal':1,'c4_25':brief(old),'c4_26':brief(new)})
    comparison={'rows':comp,'interpretation':'Prompt and schema changed jointly; one observation per matched task, no causal separation or population stability claim'}
    atomic_json(out/'c4_25_vs_c4_26_comparison.json',comparison);atomic_json(out/'token_and_latency_comparison.json',comparison)
    supported=ev['all_agents']['counts']['CORRECT_ANSWER']+ev['all_agents']['counts']['INCORRECT_ANSWER']
    if col['stop_reason']:decision='STOP_REASSESS: first technical error; no automatic retry or bulk collection.'
    elif col['attempted']==10:
        decision='MINI_PILOT_TECHNICALLY_VALID; '
        decision+=('UNRESOLVED_ONLY: review task/information/model limitations before remaining15.' if supported==0 else 'consider separately approved remaining15; assess paired selection value and task-native target, no fitting.')
    else:decision='PARTIAL; no stability success claim.'
    text(out/'next_action_decision.md','# Next decision\n\n'+decision+'\n\nNo Utility fitting or clinical verdict. Remaining15 always need new approval.\n')
    print(json.dumps({'status':col['status'],'attempted':col['attempted'],'supported':supported,'oracle_gap':potential['oracle_gap']}))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','preflight','dry-run','execute','analyze'],nargs='?',default='dry-run')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-llm',action='store_true');p.add_argument('--approval');p.add_argument('--approval-sha256');args=p.parse_args()
    if args.phase=='prepare':return prepare()
    if args.phase=='preflight':return preflight()
    if args.phase=='analyze':return analyze()
    if args.phase=='execute' and not args.execute:raise ValueError('EXECUTE_FLAG_REQUIRED')
    if args.phase=='dry-run' and args.execute:raise ValueError('USE_EXECUTE_SUBCOMMAND')
    pub,pr=locations(NAMESPACE);b,c=load_bundle();ep=read(pub/'mini_pilot_protocol.json')
    approval=CompactApproval(args.approval,args.approval_sha256) if args.approval and args.approval_sha256 else None
    r=run_collection(b,c,ep,pr/'live'/ep['run_id'],execute=args.execute,allow_llm=args.allow_llm,approval=approval)
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','evaluation')}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        code=str(exc) if isinstance(exc,ValueError) and str(exc).isupper() else type(exc).__name__
        print(json.dumps({'status':'BLOCKED','error_code':code,'automatic_retry':False}));sys.exit(2)
