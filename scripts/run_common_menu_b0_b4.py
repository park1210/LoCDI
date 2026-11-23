# -*- coding: utf-8 -*-
"""User-operated common-menu development evaluation. Default check, no model calls.

New Source/Registry protocol; never resumes or relabels an old B012 experiment.
Generation must be user-operated. Non-generating prepare/check/analyze are safe.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.dataset_loader import load_cases
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes,features_for
from app.locdi.m5_common_source import bind_cases,M5_SHA,POLICY
from app.locdi.common_menu_backend import make_common_backend,routing_admission
from app.locdi.synthea_diagnosis_audit import matches
from app.locdi.development_routing_proxy import MARKERS,admit_development_proxy
from app.locdi.source_paths import resolve_source_file,MANIFEST as SOURCE_MANIFEST

NS='B34_CONNECTION_v1'
DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
M5=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
OLD=ROOT/'outputs/private/research_evaluation/FINAL_B012_MEMORY_STABILITY_RECOVERY_v1/retry_002'
OLDPUB=ROOT/'outputs/research_evaluation/FINAL_B012_MEMORY_STABILITY_RECOVERY_v1/retry_002'
CODE=['app/locdi/m5_common_source.py','app/locdi/common_menu_backend.py','scripts/run_common_menu_b0_b4.py','app/locdi/common_prediction_contract.py','app/locdi/common_runtime_recovery.py',
      'app/locdi/development_routing_proxy.py','app/locdi/run_simulation.py',
      'app/locdi/source_paths.py',
      'scripts/collect_common_routing_training.py','scripts/build_development_routing_proxy.py',
      'app/core/routing_utility.py','app/core/agent_registry.py','app/core/information_value.py',
      'app/core/voi_planning.py','app/core/hospital_llm_policy.py','app/core/hospital_parallel_scheduler.py',
      'validation/routing_correctness_gate.py','validation/routing_training_data.py','validation/research_artifact_gate.py']
CODE += ['app/locdi/medical_knowledge.py','app/locdi/medical_knowledge_catalog.py',
         'app/locdi/literature_rule_router.py','app/locdi/evidence_selection_wire.py']


def paths(name):
    if not re.fullmatch('[a-zA-Z0-9_]{1,64}',name):raise ValueError('SAFE_FRESH_PLAN_ID_REQUIRED')
    return ROOT/'outputs/research_evaluation'/NS/name,ROOT/'outputs/private/research_evaluation'/NS/name


def prepare(name,conditions,evidence_path=None,experiment_kind='CLINICAL_EVALUATION',patient_count=10,catalog_selection=False,runtime_recovery=False,
            knowledge_path=None,memory_time_labels=False,rule_based=False,evidence_set=False):
    pub,pr=paths(name)
    if pub.exists() or pr.exists():raise ValueError('NEW_PLAN_ONLY_NO_OVERWRITE')
    if len(conditions)!=len(set(conditions)):raise ValueError('DUPLICATE_CONDITION')
    if runtime_recovery and not catalog_selection:raise ValueError('RECOVERY_REQUIRES_SHARED_CATALOG_SELECTION')
    if sha(M5)!=M5_SHA:raise ValueError('FROZEN_M5_CHANGED')
    oldp=read(OLDPUB/'B012_EXECUTION_PROTOCOL.json')
    # Only this adapter has an intentional new opt-in flag; old pinned manifest
    # is never rewritten. Every other historical pinned file must still match.
    prior={k:v for g in ('input_hashes','binding_hashes','code_hashes') for k,v in oldp[g].items()}
    adapter_changes={k:v for k,v in prior.items() if Path(k).name=='run_simulation.py'}
    if check({k:v for k,v in prior.items() if k not in adapter_changes})['status']!='PASS':
        raise ValueError('PREVIOUS_PIN_CHANGED')
    model=read(M5);cases=load_cases(DATA);index=read(OLD/'case_index.json');panel=read(OLD/'panel.json')
    if len(panel)!=10 or len(index)!=10:raise ValueError('SAME_FIXED_TEN_REQUIRED')
    if type(patient_count) is not int or not 1<=patient_count<=10:raise ValueError('PATIENT_COUNT_ONE_TO_TEN')
    panel=panel[:patient_count];index={k:v for k,v in index.items() if k in {p['case_ref'] for p in panel}}
    bindings=bind_cases({ref:cases[n-1] for ref,n in index.items()},model)
    registry=read(ROOT/'outputs/research_evaluation/final_exp_v1/routing_3b_agent_registry.json')['registry']
    write(pr/'registry.json',registry)
    config=read(OLD/'llm_config.json');config['registry_path']=(pr/'registry.json').relative_to(ROOT).as_posix()
    config['agent_bindings']=[{'agent_id':a['agent_id'],'prompt':'prompts/er_department_system.txt' if a['role']=='er' else 'prompts/specialty/'+a['department']+'.txt',
        'available':True,'model_identifier':config['model_identifier'],'shared_base_model':True} for a in registry['agents']]
    write(pr/'llm_config.json',config);write(pr/'bindings.json',bindings);write(pr/'panel.json',panel);write(pr/'case_index.json',index)
    write(pr/'history_selection_plans.json',read(OLD/'history_selection_plans.json'))
    files=[DATA,M5,pr/'registry.json',pr/'llm_config.json',pr/'bindings.json',pr/'panel.json',pr/'case_index.json',pr/'history_selection_plans.json']
    files += [ROOT/b['prompt'] for b in config['agent_bindings']]+[ROOT/config['patient_prompt']]
    files += [resolve_source_file(r['source_event']['source_metadata']['source_file']) for b in bindings.values() for r in b['rows']]
    files.append(SOURCE_MANIFEST)
    if evidence_path is not None:files.append(Path(evidence_path))
    if knowledge_path is not None:
        from app.locdi.medical_knowledge import MedicalKnowledgeIndex
        knowledge=read(knowledge_path);MedicalKnowledgeIndex(knowledge)
        if {c['disease'] for c in knowledge['cards']}!=set(model['diagnoses']):raise ValueError('SAME_FIFTEEN_KNOWLEDGE_TOPICS_REQUIRED')
        files.append(Path(knowledge_path))
    plan={'version':'locdi.common_menu.development.v1','plan_id':name,'run_id':'run_'+name,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY',
        'experiment_kind':experiment_kind,'historical_adapter_pin':adapter_changes,
        'conditions':conditions,'registered_patients':patient_count,'registered_episodes':patient_count*len(conditions),
        'features':{c:features_for(c) for c in conditions},'source_policy':POLICY,'source_binding_fingerprint':digest(bindings),
        'M5_SHA256':M5_SHA,'model_identifier':config['model_identifier'],'generation_per_episode':12,
        'generation_total':12*patient_count*len(conditions),'wall_seconds':2700,'process_bytes':4*1024**3,'output_bytes':1024**3,
        'planner_calls_per_episode':12,'planner_states_per_episode':240000,'planner_states_per_call':20000,
        'no_resume':True,'retry':0,'repair':0,'warmup':0,'clinical_research_admission':False,
        'stop_policy':'first technical/source/identity/global error or failed Core Episode; preserve incomplete outcomes',
        'source_rule':'latest eligible exact raw row, reject same-timestamp conflict; no Gold selection',
        'missing_action_policy':'unavailable; never impute; same per-patient menu in every condition',
        'B1_name':'CONTEXT_BUDGETED_HISTORY_NOT_FULL_HISTORY',
        'catalog_selection':bool(catalog_selection),
        'knowledge_corpus_path':str(Path(knowledge_path).resolve()) if knowledge_path else None,
        'knowledge_corpus_fingerprint':digest(read(knowledge_path)) if knowledge_path else None,
        'memory_time_labels':bool(memory_time_labels),'rule_based':bool(rule_based),'evidence_index_set':bool(evidence_set),
        'effective_primary_router':'literature_rule_based.v1' if rule_based else 'original_adaptive_or_er_only',
        'runtime_recovery':bool(runtime_recovery),
        'terminal_contract':'released_measurements_llm_at_voi_stop.v1' if runtime_recovery else 'FROZEN_M5_SUPPORTED_GATE',
        'catalog_selection_contract':'shared_diagnosis_department_selection.v1' if catalog_selection else None,
        'routing_evidence_path':str(Path(evidence_path).resolve()) if evidence_path else None,
        'routing_evidence_fingerprint':digest(read(evidence_path)) if evidence_path else None,
        'code_hashes':{**oldp['code_hashes'],**{n:sha(ROOT/n) for n in CODE}},
        'input_hashes':{str(p.resolve()):sha(p) for p in set(files)}}
    if experiment_kind=='DEVELOPMENT_EXPERIMENT':plan.update(MARKERS)
    plan['fingerprint']=digest(plan);write(pub/'plan.json',plan)
    approval={'scope':plan['scope'],'plan_fingerprint':plan['fingerprint'],
        'common_knowledge':knowledge_path is not None,'memory_time_labels':bool(memory_time_labels),
        'rule_based':bool(rule_based),'evidence_index_set':bool(evidence_set),
        'runtime_recovery':plan['runtime_recovery'],'terminal_contract':plan['terminal_contract'],
        'conditions':conditions,'max_generation':plan['generation_total'],'accept_recorded_time_proxy':False,
        'accept_common_m5_menu':False,'clinical_release_approved':False,'operator_approved':False,
        'no_other_ollama_work':False,'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None}
    if experiment_kind=='DEVELOPMENT_EXPERIMENT':
        approval.update(MARKERS);approval['accept_source_label_proxy']=False
    write(pr/'approval_template.json',approval)
    write(pub/'source_coverage.json',{'status':'PREPARED_NOT_TESTED_NOT_EXECUTED','source_policy':POLICY,
        'rows':[{'case_ordinal':i,'available_actions':len(bindings[p['case_ref']]['rows']),
                 'missing_actions':len(bindings[p['case_ref']]['missing_actions']),
                 'conflicts':len(bindings[p['case_ref']]['rejected'])} for i,p in enumerate(panel,1)],
        'clinical_review':'NOT_REVIEWED','generation':0,'research_evaluation_ready':False,
        **(MARKERS if experiment_kind=='DEVELOPMENT_EXPERIMENT' else {})})
    print(json.dumps({'status':'PREPARED_NOT_EXECUTED','plan':plan['fingerprint'],'generation':0}))


def load_plan(name):
    pub,pr=paths(name);p=read(pub/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    if check({**p['code_hashes'],**p['input_hashes']})['status']!='PASS':raise ValueError('PIN_CHANGED')
    return pub,pr,p


def preflight(name,evidence_path=None):
    pub,pr,p=load_plan(name);config=read(pr/'llm_config.json');setup=validate_setup(config)
    blockers=[];routing=None
    development=p.get('experiment_kind')=='DEVELOPMENT_EXPERIMENT'
    if not development:blockers.append('CLINICAL_EVALUATION_REQUIRES_UNCHANGED_OFFICIAL_GATES')
    if setup['static_readiness']!='READY':blockers.append('MODEL_REGISTRY_PROMPT_STATIC_BLOCKED')
    if (pr/p['run_id']).exists():blockers.append('RUN_USED_NO_RESUME')
    bindings=read(pr/'bindings.json')
    if any(not b['rows'] for b in bindings.values()):blockers.append('EMPTY_SOURCE_MENU_PATIENT_MUST_BE_PRESERVED')
    if set(p['conditions'])&{'B3','B4'}:
        evidence_path=evidence_path or (Path(p['routing_evidence_path']) if p.get('routing_evidence_path') else None)
        if evidence_path is None or p.get('routing_evidence_fingerprint') is None:
            blockers.append('CURRENT_REVIEWED_ROUTING_EVIDENCE_MUST_BE_PINNED_IN_NEW_PLAN')
        elif setup['static_readiness']=='READY':
            try:
                evidence=read(evidence_path)
                if digest(evidence)!=p['routing_evidence_fingerprint']:raise ValueError('ROUTING_EVIDENCE_NOT_APPROVED_PLAN')
                admission=admit_development_proxy if development else routing_admission
                routing=admission(evidence,setup['prepared']['registry'],[b['patient_id'] for b in bindings.values()],config)
                if 'B4' in p['conditions']:
                    tp=read(evidence['collection_plan'])
                    validation={c['decision']['patient_id'] for c in tp['candidates'] if c['decision']['split']=='validation'}
                    if validation&set(read(M5)['provenance']['training_patients']):blockers.append('ROUTING_VALIDATION_VOI_TRAIN_OVERLAP')
            except (ValueError,KeyError,TypeError):blockers.append('EXISTING_ROUTING_CORRECTNESS_OR_BINDING_GATE_BLOCKED')
    return {'status':'BLOCKED' if blockers else 'STATIC_CONTRACT_READY_USER_TEST_AND_APPROVAL_REQUIRED',
        'blocker_codes':blockers,'generation':0,'new_test_status':'NOT_TESTED',
        'clinical_research_admission':False,'source_policy':POLICY,'plan_fingerprint':p['fingerprint'],
        **(MARKERS if development else {'experiment_kind':'CLINICAL_EVALUATION'})}


def permission(a,p):
    expected={'scope':p['scope'],'plan_fingerprint':p['fingerprint'],'conditions':p['conditions'],
        'max_generation':p['generation_total'],'accept_recorded_time_proxy':True,'accept_common_m5_menu':True,
        'clinical_release_approved':False,'operator_approved':True,'no_other_ollama_work':True}
    if any(a.get(k)!=v for k,v in expected.items()):raise ValueError('NEW_COMMON_MENU_OPERATOR_APPROVAL_REQUIRED')
    if p.get('experiment_kind')!='DEVELOPMENT_EXPERIMENT' or a.get('accept_source_label_proxy') is not True or any(a.get(k)!=v for k,v in MARKERS.items() if k in ('experiment_kind','source_label_proxy','independent_clinical_review','clinical_accuracy')):
        raise ValueError('EXPLICIT_DEVELOPMENT_PROXY_APPROVAL_ONLY')
    if not all(isinstance(a.get(k),str) and a[k].strip() for k in ('operator_ref','user_statement')):raise ValueError('ACTUAL_OPERATOR_STATEMENT_REQUIRED')
    issued=datetime.fromisoformat(a['recorded_at']);expires=datetime.fromisoformat(a['expires_at']);now=datetime.now(timezone.utc)
    if issued.tzinfo is None or expires.tzinfo is None or not issued<=now<expires or not 0<(expires-issued).total_seconds()<=86400:
        raise ValueError('APPROVAL_TIME_INVALID')


def execute(name,approval_path,evidence_path):
    report=preflight(name,evidence_path)
    if report['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p=load_plan(name);a=read(approval_path);permission(a,p)
    evidence_path=evidence_path or (Path(p['routing_evidence_path']) if p.get('routing_evidence_path') else None)
    config=read(pr/'llm_config.json');bindings=read(pr/'bindings.json');cases=load_cases(DATA);index=read(pr/'case_index.json')
    run=pr/p['run_id'];run.mkdir(exist_ok=False);write(run/'approval.json',a)
    source_stats={k:(Path(k).stat().st_size,Path(k).stat().st_mtime_ns) for k in p['input_hashes']}
    started=time.monotonic()
    def guard():
        permission(a,p)
        if time.monotonic()-started>p['wall_seconds']:raise ValueError('GLOBAL_WALL_BUDGET')
        if _process_private_bytes()>p['process_bytes']:raise ValueError('PROCESS_MEMORY_BUDGET')
        if sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>p['output_bytes']:raise ValueError('OUTPUT_BUDGET')
        if any((Path(k).stat().st_size,Path(k).stat().st_mtime_ns)!=v for k,v in source_stats.items()):raise ValueError('SOURCE_OR_CONFIG_CHANGED')
    budget=GenerationBudget(p['generation_total'],12,resource_guard=guard)
    backend=make_common_backend({k:cases[n-1] for k,n in index.items()},config=config,bindings=bindings,model=read(M5),
        authorization=a,plan_fingerprint=p['fingerprint'],checkpoint_root=run/'requests',
        routing_evidence=read(evidence_path) if evidence_path else None,
        experiment_kind=p['experiment_kind'],
        catalog_selection=p.get('catalog_selection',False),
        runtime_recovery=p.get('runtime_recovery',False),
        knowledge_corpus=read(p['knowledge_corpus_path']) if p.get('knowledge_corpus_path') else None,
        memory_time_labels=p.get('memory_time_labels',False),rule_based=p.get('rule_based',False),evidence_set=p.get('evidence_index_set',False),
        history_policy=ContextBudgetedHistory(read(pr/'history_selection_plans.json')))
    rows=[];stop=None
    with (run/'episodes.jsonl').open('x',encoding='utf-8') as f:
        for i,patient in enumerate(read(pr/'panel.json'),1):
            for condition in p['conditions']:
                row={'episode_ordinal':len(rows)+1,'patient_ordinal':i,'case_ref':patient['case_ref'],
                    'condition':condition,'status':'NOT_ATTEMPTED','request_count':0,'result':None,'error_type':None}
                if stop is None:
                    before=budget.used;budget.episode_used=0;token=_BUDGET.set(budget)
                    try:
                        guard();result=backend({'case_ref':patient['case_ref']},features_for(condition))
                        if result['status'] not in ('completed','incomplete','failed') or result['request_count']!=budget.used-before:
                            raise ValueError('ACTUAL_RESULT_OR_REQUEST_COUNT_CONTRACT')
                        row.update(result=result,status=result['status'].upper());guard()
                        if result['status']=='failed':stop='FIRST_FAILED_EPISODE'
                    except Exception as exc:row.update(status='FAILED',error_type=type(exc).__name__);stop='FIRST_TECHNICAL_FAILURE'
                    finally:_BUDGET.reset(token);row['request_count']=budget.used-before
                row['stop_reason']=stop;f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();rows.append(row)
    write(run/'collection.json',{'rows':rows,'generation':budget.used,'stop_reason':stop,'wall_seconds':time.monotonic()-started,
        'plan_fingerprint':p['fingerprint'],'clinical_metrics':'NOT_EVALUATED',**MARKERS})
    write(pub/'execution_status.json',{'states':dict(Counter(r['status'] for r in rows)),
        'generation':budget.used,'stop_reason':stop,'clinical_metrics':'NOT_EVALUATED',**MARKERS})
    if check({**p['code_hashes'],**p['input_hashes']})['status']!='PASS':raise ValueError('POST_EXECUTION_PIN_CHANGED_PARTIAL_RESULTS_PRESERVED')


def analyze(name):
    pub,pr,p=load_plan(name);run=pr/p['run_id'];c=read(run/'collection.json');panel=read(pr/'panel.json')
    lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if s]
    if lines!=c['rows']:raise ValueError('TRACE_COLLECTION_MISMATCH')
    if c['plan_fingerprint']!=p['fingerprint'] or sum(r['request_count'] for r in lines)!=c['generation']:
        raise ValueError('TRACE_GENERATION_OR_PLAN_MISMATCH')
    scored=[]
    for r in lines:
        ref=panel[r['patient_ordinal']-1]['reference'];prediction=(r['result'] or {}).get('prediction') or {}
        done=r['status']=='COMPLETED';supported=done and prediction.get('certainty')=='supported'
        m=matches(prediction.get('diagnosis') if supported else None,ref['accepted_diagnoses'])
        scored.append({**{k:r[k] for k in ('episode_ordinal','patient_ordinal','condition','status','request_count')},
            'source_exact':m['source_exact'] if done else None,'normalized':m['normalized_match'] if done else None,
            'source_department':prediction.get('department') in ref['accepted_departments'] if done else None,
            'algorithmic_source_exact':matches(prediction.get('algorithmic_diagnosis_choice'),ref['accepted_diagnoses'])['source_exact'] if done and r['condition']=='B4' else None,
            'certainty':prediction.get('certainty') if done else None})
    summary={}
    for condition in p['conditions']:
        rr=[r for r in scored if r['condition']==condition]
        summary[condition]={'registered':p['registered_patients'],'states':dict(Counter(r['status'] for r in rr)),
            'Source_exact_success':sum(r['source_exact'] is True for r in rr),
            'Source_department_success':sum(r['source_department'] is True for r in rr),
            'algorithmic_source_success_separate':sum(r['algorithmic_source_exact'] is True for r in rr),
            'generation':sum(r['request_count'] for r in rr),
            **{key:sum((r.get('result') or {}).get(key,0) for r in lines if r['condition']==condition)
               for key in ('routing_call_count','routing_selected_count','routing_abstention_count','voi_information_action_count','voi_STOP_count')}}
        chosen=[r for r in lines if r['condition']==condition]
        usage=Counter()
        for row in chosen:
            folder=run/'requests'/('Episode%04d'%row['episode_ordinal'])
            for ap in folder.glob('Request*_after.json'):
                record=read(ap);envelope=record.get('response_envelope') or {}
                usage['responses_received']+=record.get('response_received') is True
                usage['prompt_tokens_reported']+=envelope.get('prompt_eval_count',0)
                usage['output_tokens_reported']+=envelope.get('eval_count',0)
                usage['HTTP_wall_seconds']+=record.get('wall_seconds',0)
        summary[condition]['server_usage']=dict(usage)
        summary[condition]['planner_calls']=sum((r.get('result') or {}).get('computation',{}).get('planner_calls',0) for r in chosen)
        summary[condition]['evaluated_states']=sum((r.get('result') or {}).get('computation',{}).get('evaluated_states',0) for r in chosen)
    paired={}
    for first,second in (('B0','B1'),('B1','B2'),('B0','B2'),('B2','B3'),('B3','B4')):
        if first not in p['conditions'] or second not in p['conditions']:continue
        prs=[]
        for i in range(1,p['registered_patients']+1):
            x=next(r for r in scored if r['patient_ordinal']==i and r['condition']==first)
            y=next(r for r in scored if r['patient_ordinal']==i and r['condition']==second)
            if x['status']==y['status']=='COMPLETED':
                prs.append({'patient_ordinal':i,'source_diagnosis_difference':int(y['source_exact'])-int(x['source_exact'])
                    if type(x['source_exact']) is bool and type(y['source_exact']) is bool else None,
                    'source_department_difference':int(y['source_department'])-int(x['source_department'])})
        known=[r['source_diagnosis_difference'] for r in prs if r['source_diagnosis_difference'] is not None]
        paired[second+'-'+first]={'completed_pairs':len(prs),'scorable_diagnosis_pairs':len(known),
            'diagnosis_difference':sum(known)/len(known) if known else None,'rows':prs}
    write(pub/'saved_results.json',{'run_id':p['run_id'],'conditions':summary,'rows':scored,
        'paired':paired,'actual_generation':c['generation'],'wall_seconds':c['wall_seconds'],
        'clinical_accuracy':'NOT_EVALUATED','original_B012_results_merged':False,
        'source_menu':'new common M5-bound menu; not original expanded Native menu',**MARKERS})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','check','execute','analyze','monitor'],nargs='?',default='check')
    parser.add_argument('--plan-id',default='common_all_001')
    parser.add_argument('--conditions',nargs='+',choices=['B0','B1','B2','B3','B4'])
    parser.add_argument('--experiment-kind',choices=['DEVELOPMENT_EXPERIMENT','CLINICAL_EVALUATION'])
    parser.add_argument('--patient-count',type=int,choices=range(1,11))
    parser.add_argument('--catalog-selection',action='store_true',help='Opt-in shared diagnosis/department lists; new plan only')
    parser.add_argument('--runtime-recovery',action='store_true',help='New common released-measurement display and separate LLM finalization at M5 STOP')
    parser.add_argument('--common-knowledge',type=Path,help='Fixed patient-independent corpus JSON; new plan only')
    parser.add_argument('--memory-time-labels',action='store_true')
    parser.add_argument('--rule-based',action='store_true',help='Separate development registry adapter, not fitted Adaptive Utility')
    parser.add_argument('--evidence-index-set',action='store_true')
    parser.add_argument('--routing-evidence',type=Path)
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true')
    a=parser.parse_args()
    if a.mode!='prepare' and a.catalog_selection:parser.error('CATALOG_SELECTION_FIXED_AT_PREPARE')
    if a.mode!='prepare' and a.runtime_recovery:parser.error('RECOVERY_FIXED_AT_PREPARE')
    if a.mode!='prepare' and (a.common_knowledge or a.memory_time_labels or a.rule_based or a.evidence_index_set):parser.error('KNOWLEDGE_OPTIONS_FIXED_AT_PREPARE')
    if a.mode!='prepare' and a.conditions is not None:parser.error('CONDITIONS_ARE_FIXED_DURING_PREPARE_ONLY')
    if a.mode!='prepare' and a.experiment_kind is not None:parser.error('EXPERIMENT_KIND_FIXED_AT_PREPARE')
    if a.mode!='prepare' and a.patient_count is not None:parser.error('PATIENT_COUNT_FIXED_AT_PREPARE')
    if a.mode!='execute' and (a.execute or a.allow_llm or a.authorization):parser.error('EXECUTION_FLAGS_ARE_EXECUTE_ONLY')
    if a.mode=='prepare':prepare(a.plan_id,a.conditions or ['B0','B1','B2','B3','B4'],a.routing_evidence,a.experiment_kind or 'CLINICAL_EVALUATION',a.patient_count or 10,a.catalog_selection,a.runtime_recovery,a.common_knowledge,a.memory_time_labels,a.rule_based,a.evidence_index_set)
    elif a.mode=='check':
        result=preflight(a.plan_id,a.routing_evidence);print(json.dumps(result,ensure_ascii=True))
        if result['status']=='BLOCKED':raise SystemExit(2)
    elif a.mode=='execute':
        if not(a.execute and a.allow_llm and a.authorization):parser.error('USER_EXECUTE_ALLOW_LLM_AND_NEW_APPROVAL_REQUIRED')
        execute(a.plan_id,a.authorization,a.routing_evidence)
    elif a.mode=='analyze':analyze(a.plan_id)
    else:
        pub,pr=paths(a.plan_id);p=read(pub/'plan.json');run=pr/p['run_id'];trace=run/'episodes.jsonl';rows=[]
        if trace.exists():
            for line in trace.read_text(encoding='utf-8').splitlines():
                try:rows.append(json.loads(line))
                except ValueError:pass
        checkpoints=sorted(run.glob('requests/Episode*/Request*_after.json'))
        print(json.dumps({'rows':len(rows),'states':dict(Counter(r['status'] for r in rows)),
            'request_after_count':len(checkpoints),'last_checkpoint':str(checkpoints[-1].relative_to(run)) if checkpoints else None},ensure_ascii=True))


if __name__=='__main__':main()
