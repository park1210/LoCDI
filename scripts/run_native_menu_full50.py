"""Admission-bound Original50 D2 B0..B4 stages; default no inference."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from contextlib import contextmanager
from unittest.mock import patch
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import native_routing_transfer as transfer
from scripts import run_original50_final as legacy
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import digest
from app.locdi.native_menu_routing import backend as transfer_backend,VERSION
from app.locdi.development_routing_proxy import MARKERS

PUB=transfer.PUB;PR=transfer.PR
STAGES={'Smoke5':5,'Verify10':10,'Full50':50}
CONDS=['B0','B1','B2','B3','B4']
ORIGINAL_PERMISSION=legacy.permission

def name(stage):return 'native_menu_d2_'+stage.lower()+'_002'

def prepare():
    result=transfer.admission_check(write=False)
    if result['status']=='BLOCKED' or not transfer.EVIDENCE.exists():raise ValueError('ACTUAL_VALIDATION_FIT_AND_ADMISSION_REQUIRED')
    if any((PUB/name(s)).exists() or (PR/name(s)).exists() for s in STAGES):raise FileExistsError('NEW_EVALUATION_PLANS_ONLY')
    evidence=read(transfer.EVIDENCE)
    _,basepr,base=legacy.load('Full50');source_tasks=read(basepr/'tasks.json');order=list(dict.fromkeys(t['original_ordinal'] for t in source_tasks))
    old={t['original_ordinal'] for t in source_tasks if t['old_development10']}
    code=[Path(__file__),ROOT/'scripts/native_routing_transfer.py',ROOT/'app/locdi/native_menu_routing.py',ROOT/'scripts/Invoke-LoCDINativeTransfer.ps1']
    pins={**base['pins'],**evidence['origin_files'],str(transfer.EVIDENCE.resolve()):sha_file(transfer.EVIDENCE),**{str(f.resolve()):sha_file(f) for f in code}}
    contract=digest({'source_comparison':base['comparison_fingerprint'],'feature_version':VERSION,'utility_evidence_sha256':sha_file(transfer.EVIDENCE),'pins':pins})
    for stage,n in STAGES.items():
        p=deepcopy(base);p.pop('fingerprint');p.update(plan_id=name(stage),run_id='run_'+name(stage),stage=stage,
            conditions=CONDS,execution_scope='B0_B4_NATIVE_PUBLIC_MENU_TRANSFER',registered_patients=n,registered_episodes=n*5,
            generation_total=n*60,comparison_fingerprint=contract,utility_sha256=sha_file(transfer.EVIDENCE),
            utility_selection='ADMITTED_NATIVE_SOURCE_PROXY_WITH_REPORTED_VALIDATION_LIMITS',blocker_codes=[],
            wall_seconds=1800 if n==5 else 3600 if n==10 else 14400,feature_version=VERSION,
            terminal_only='FROZEN_STOP_EXTRA1_WITHIN_EPISODE12',source_parent_B012_plan_fingerprint=base['fingerprint'])
        tasks=[{'episode_ordinal':i+1,'original_ordinal':ordinal,'case_ref':'Original%03d'%ordinal,'condition':condition,'old_development10':ordinal in old}
               for i,(ordinal,condition) in enumerate((o,c) for o in order[:n] for c in CONDS)]
        pr=PR/name(stage);pub=PUB/name(stage);transfer.write_new(pr/'tasks.json',tasks)
        p['pins']={**pins,str((pr/'tasks.json').resolve()):sha_file(pr/'tasks.json')};p['fingerprint']=digest(p);transfer.write_new(pub/'plan.json',p)
        a=read(basepr/'approval_template.json');a.update(conditions=CONDS,plan_fingerprint=p['fingerprint'],execution_scope=p['execution_scope'],
            max_generation=n*60,accept_native_public_menu_feature=False,accept_prior_VoI_exposure_and_bounded_independence=False,
            admitted_utility_sha256=p['utility_sha256'])
        transfer.write_new(pr/'approval_template.json',a)
    return {'status':'NEW_B0_B4_PLANS_PREPARED_NOT_EXECUTED','full_episodes':250,'full_max_generation':3000,'new_generation':0}

def load(stage):
    p=read(PUB/name(stage)/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('CODE_SOURCE_UTILITY_PIN_CHANGED')
    return PUB/name(stage),PR/name(stage),p

def permission(a,p):
    ORIGINAL_PERMISSION(a,p)
    if (a.get('accept_native_public_menu_feature') is not True or a.get('accept_prior_VoI_exposure_and_bounded_independence') is not True
        or a.get('admitted_utility_sha256')!=p['utility_sha256']):raise ValueError('EXACT_NATIVE_TRANSFER_EXECUTION_APPROVAL_REQUIRED')

def check(stage):
    if not (PUB/name(stage)/'plan.json').exists():return {'status':'BLOCKED','blocker_codes':['EVALUATION_PLAN_NOT_PREPARED_ACTUAL_VALIDATION_AND_FIT_REQUIRED'],'new_generation':0}
    pub,pr,p=load(stage);blocks=[]
    if not transfer.EVIDENCE.exists() or sha_file(transfer.EVIDENCE)!=p['utility_sha256']:blocks.append('EXACT_TRANSFER_ADMISSION_REQUIRED')
    if (pr/p['run_id']).exists():blocks.append('RUN_USED_NO_RESUME')
    fake=PUB/'TEST_ONLY_validation_and_full250.json'
    if not fake.exists() or read(fake).get('status')!='PASS':blocks.append('FRESH_FULL250_TRANSFER_FAKE_REQUIRED')
    proof=None
    if stage!='Smoke5':
        try:
            with patch.object(legacy,'load',new=load):
                proof=legacy.technical_proof('Smoke5')
                if stage=='Full50' and not proof['Verify10_skip_allowed']:proof=legacy.technical_proof('Verify10')
            if proof['comparison_fingerprint']!=p['comparison_fingerprint']:raise ValueError('STAGE_CONTRACT_CHANGED')
        except (ValueError,FileNotFoundError):blocks.append('PRECEDING_REAL_TECHNICAL_STAGE_REQUIRED')
    return {'status':'BLOCKED' if blocks else 'STATIC_READY_USER_APPROVAL_REQUIRED','blocker_codes':blocks,'plan_id':name(stage),
        'registered_episodes':p['registered_episodes'],'generation_cap':p['generation_total'],'promotion':proof,'new_generation':0}

def execute(stage,approval):
    if transfer.admission_check()['status']=='BLOCKED':raise ValueError('ACTUAL_TRANSFER_ADMISSION_REQUIRED')
    evidence=read(transfer.EVIDENCE)
    def factory(cases,**kw):return transfer_backend(cases,admitted_evidence=evidence,admitter=transfer.admitted,**kw)
    with patch.object(legacy,'load',new=load),patch.object(legacy,'check',new=check),\
            patch.object(legacy,'permission',new=permission),patch.object(legacy,'make_paired_backend',new=factory):
        return legacy.execute(stage,approval)

def analyze(stage):
    from scripts.analyze_original50_d0_d2 import facts
    from app.locdi.full50_evaluation_metrics import aggregate_by_exposure
    pub,pr,p=load(stage);run=pr/p['run_id'];c=read(run/'collection.json');tasks=read(pr/'tasks.json')
    lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if lines!=c['rows'] or len(lines)!=len(tasks) or c['plan_fingerprint']!=p['fingerprint'] or sum(r['request_count'] for r in lines)!=c['generation']:raise ValueError('SAVED_TRACE_LINEAGE')
    sources={r['summary']['ordinal']:r for r in read(legacy.PR/'case_audit_private.json')};private=[];metrics=[]
    from app.locdi.original_source_inventory import original_cases,eligible
    originals=original_cases()
    for r,t in zip(lines,tasks):
        if any(r.get(k)!=v for k,v in t.items()):raise ValueError('REGISTERED_TASK_CHANGED')
        folder=run/'requests'/('Episode%04d'%r['episode_ordinal']);f=facts(folder,r['result'],sources[r['original_ordinal']]['reference'])
        if r['status']!='COMPLETED':
            for key in ('source_exact','normalized','department_match','algorithmic_source_exact'):f[key]=None
        blocked=sum(read(x).get('generation_attempted') is False for x in folder.glob('Request*_after.json'))
        if f['request_count_checkpoint']!=r['request_count']+blocked:raise ValueError('CHECKPOINT_COUNTER')
        payloads=[json.loads(read(x)['request']['messages'][1]['content']) for x in sorted(folder.glob('Request*_before.json'))]
        last=payloads[-1] if payloads else {};evidence=last.get('evidence',[]);voi=f['voi_trace'];rt=r.get('result') or {}
        cutoff=originals[r['original_ordinal']-1]['agent_visible']['longitudinal']
        bad=sum(not eligible(m,cutoff) for b in payloads for m in b['observation'].get('longitudinal_memory',{}).get('records',[]))
        private.append({'task':r,'facts':f})
        metrics.append({'ordinal':r['original_ordinal'],'condition':r['condition'],'status':r['status'],'plan_fingerprint':p['fingerprint'],
            'source_exact':f['source_exact'],'source_department':f['department_match'],'normalized':f['normalized'],
            'algorithmic_source_exact':f['algorithmic_source_exact'] if r['condition']=='B4' else None,'certainty':f['prediction']['certainty'],
            'generation':r['request_count'],'usage':f['usage'],'action_counts':rt.get('action_counts',{}),
            'wall_seconds':rt.get('computation',{}).get('wall_seconds',0),'role_counts':dict(Counter(b.get('role','')+':'+b.get('department','') for b in payloads)),
            'routing_calls':len(f['routing_trace']),'routing_selected':rt.get('routing_selected_count',0),'routing_abstained':rt.get('routing_abstention_count',0),
            'memory_selected_count':len(f['last_memory']),'memory_cutoff_violations':bad,'current_evidence_count':len(evidence),
            'planner_calls':rt.get('computation',{}).get('planner_calls',0),'stop_count':rt.get('voi_STOP_count',0),
            'terminal_extra_requests':sum(v.get('terminal_extra_requests',0) for v in voi)})
    ordinals=list(dict.fromkeys(t['original_ordinal'] for t in tasks));old={t['original_ordinal'] for t in tasks if t['old_development10']}
    report={'status':'SAVED_TRACE_ANALYZED','strata':aggregate_by_exposure(metrics,ordinals,old,plan_fingerprint=p['fingerprint']),
        'normalized_success':{k:sum(r['normalized'] is True for r in metrics if r['condition']==k) for k in CONDS},
        'generation':c['generation'],'dataset':'D2_LABEL_CONDITIONED_DEVELOPMENT','existing_results_merged':False,'utility_evidence_sha256':p['utility_sha256'],**MARKERS}
    transfer.write_new(run/'analysis_private.json',{'metrics':metrics,'details':private})
    transfer.write_new(pub/'saved_results.json',report);return {'status':report['status'],'new_generation':0}

def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','execute','analyze']);q.add_argument('--stage',choices=STAGES,default='Smoke5');q.add_argument('--execute',action='store_true');q.add_argument('--approval',type=Path);a=q.parse_args()
    if a.mode=='prepare':out=prepare()
    elif a.mode=='check':out=check(a.stage)
    elif a.mode=='analyze':out=analyze(a.stage)
    else:
        if not a.execute or not a.approval:q.error('EXPLICIT_NEW_USER_EXECUTION_APPROVAL_REQUIRED')
        out=execute(a.stage,a.approval)
    print(json.dumps(out))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)

if __name__=='__main__':main()
