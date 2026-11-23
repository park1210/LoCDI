"""One new terminal request per saved B4 STOP; cached prefix is NOT new generation.

Re-executes the original Core/Scheduler using hash-checked stored responses.
Every cached wire must match exactly; otherwise no terminal POST is permitted.
"""
from pathlib import Path
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import patch
import argparse, json, sys, time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.run_u1_voi_expansion import load as load_parent, AUG, M5
from app.locdi.original_source_inventory import read, sha_file, original_cases
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.current_ablation_runner import GenerationBudget, _BUDGET, _process_private_bytes, features_for
from app.locdi.authored_utility_transfer import make_transfer_backend
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.terminal_only_recovery import terminal_only_scope, VERSION
from app.locdi.development_routing_proxy import MARKERS

NS = 'B34_DIAGNOSIS_RECOVERY_20261123_v1'
NAME = 'u1_stop_terminal_replay_001'
PUB = ROOT/'outputs/research_evaluation'/NS
PR = ROOT/'outputs/private/research_evaluation'/NS


class StoredReply:
    status_code = 200
    def __init__(self, value): self.value=deepcopy(value); self.content=json.dumps(value).encode()
    def json(self): return deepcopy(self.value)
    def raise_for_status(self): pass


class PrefixReplayTransport:
    def __init__(self, source, prepared, factory):
        self.before=[read(p) for p in sorted(Path(source).glob('Request*_before.json'))]
        self.after=[read(p) for p in sorted(Path(source).glob('Request*_after.json'))]
        if not self.before or len(self.before)!=len(self.after): raise ValueError('COMPLETE_PREFIX_REQUIRED')
        for b,a in zip(self.before,self.after):
            if (a['status']!='VALID' or a['HTTP_status']!=200 or digest(b['request'])!=b['request_fingerprint']
                    or a['request_fingerprint']!=b['request_fingerprint']): raise ValueError('SAVED_PREFIX_INTEGRITY')
        self.index=0; self.pending_cached_ps=False; self.prepared=prepared; self.factory=factory
        self.actual=None; self.new_posts=0; self.new_attempts=0; self.ledger=[]

    def get(self,url):
        if url.endswith('/api/ps') and self.pending_cached_ps:
            self.pending_cached_ps=False
            return StoredReply({'models':[self.after[self.index-1]['identity_after']]})
        if url.endswith('/api/tags') and self.index<len(self.before):
            return StoredReply({'models':[self.before[self.index]['identity_before']]})
        if self.actual is None: self.actual=self.factory(self.prepared)
        return self.actual.get(url)

    def reserve(self, real_reserve):
        if self.index<len(self.before): return  # a saved response, never a network call
        real_reserve(); self.new_attempts+=1

    def post(self,url,*,json):
        if self.index<len(self.before):
            b=self.before[self.index];a=self.after[self.index]
            if digest(json)!=b['request_fingerprint']:raise ValueError('PREFIX_WIRE_DIFFERENCE_NO_TERMINAL_CALL')
            self.ledger.append({'stored_request_number':self.index+1,'request_fingerprint':digest(json),
                                'generation_attempted_now':False,'identity_status':'STORED_NOT_REOBSERVED'})
            self.index+=1;self.pending_cached_ps=True
            return StoredReply(a['response_envelope'])
        p=__import__('json').loads(json['messages'][1]['content'])
        last=__import__('json').loads(self.before[-1]['request']['messages'][1]['content'])
        if self.new_posts or p.get('terminal_request_contract')!=VERSION:raise ValueError('ONE_NEW_TERMINAL_POST_ONLY')
        for key in ('observation','evidence','supported_diagnosis_eligible_indices'):
            if digest(p.get(key))!=digest(last.get(key)):raise ValueError('SAVED_STOP_SNAPSHOT_DIFFERENCE')
        self.new_posts+=1
        if self.actual is None:self.actual=self.factory(self.prepared)
        return self.actual.post(url,json=json)

    def close(self):
        if self.actual is not None:self.actual.close()


def prepare():
    pub,pr=PUB/NAME,PR/NAME
    if pub.exists() or pr.exists():raise FileExistsError('NEW_PLAN_ONLY')
    _,parent_pr,parent,bp=load_parent();run=parent_pr/parent['run_id']
    details=read(run/'analysis_private.json');tasks=[];pins=dict(parent['pins'])
    for row in details:
        if row['condition']!='B4':continue
        if row['status']!='COMPLETED' or row['result']['voi_STOP_count']!=1:raise ValueError('SINGLE_COMPLETED_STOP_REQUIRED')
        source=run/'requests'/('Episode%04d'%row['episode_ordinal'])
        stop=row['facts']['voi_trace'][-1]
        if stop['selected_action']!='diagnose' or row['request_count']>=12:raise ValueError('STOP_OR_EXTRA_REQUEST_BUDGET')
        tasks.append({'panel_ordinal':row['panel_ordinal'],'case_ref':row['case_ref'],'source_episode':str(source.resolve()),
                      'stored_requests':row['request_count'],'stop_minute':stop['minute'],
                      'old_prediction':row['facts']['prediction'],'old_source_exact':row['facts']['source_exact']})
        pins.update({str(f.resolve()):sha_file(f) for f in source.rglob('*') if f.is_file()})
    if len(tasks)!=10:raise ValueError('TEN_REGISTERED_STOP_SNAPSHOTS_REQUIRED')
    pub.mkdir(parents=True);pr.mkdir(parents=True);atomic_json(pr/'tasks.json',tasks)
    for f in [Path(__file__),ROOT/'app/locdi/terminal_only_recovery.py',ROOT/'scripts/Invoke-LoCDITerminalOnlyReplay.ps1',pr/'tasks.json',run/'analysis_private.json']:
        pins[str(f.resolve())]=sha_file(f)
    p={'plan_id':NAME,'run_id':'run_'+NAME,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY',
       'paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY','dataset_views':['D0','D2'],'conditions':['B4'],
       'experiment_scope':'U1_SAVED_STOP_TERMINAL_ONLY_REPLAY10','parent_fingerprint':parent['fingerprint'],
       'terminal_contract':VERSION,'registered_snapshots':10,'generation_total':10,'generation_per_episode':1,
       'stored_prefix_responses':sum(t['stored_requests'] for t in tasks),'wall_seconds':1800,
       'process_bytes':4*1024**3,'output_bytes':1024**3,'retry':0,'repair':0,'warmup':0,'no_resume':True,
       'model':parent['model'],'model_identifier':parent['Model'],'U1_sha256':parent['U1_sha256'],
       'M5_sha256':sha_file(M5),'comparison':'FROZEN_STOP_TERMINAL_REPLAY_NOT_A_NEW_WHOLE_EPISODE_ABLATION',
       'pins':pins,**MARKERS}
    p['fingerprint']=digest(p);atomic_json(pub/'plan.json',p)
    a={**MARKERS,'scope':p['scope'],'paired_scope':p['paired_scope'],'dataset_views':['D0','D2'],
       'conditions':['B4'],'experiment_scope':p['experiment_scope'],'plan_fingerprint':p['fingerprint'],
       'max_generation':10,'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None,
       'clinical_release_approved':False}
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy',
              'accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy',
              'accept_authored_utility_transfer','accept_routing_abstention_er_fallback','accept_cached_prefix_terminal_replay'):a[k]=False
    atomic_json(pr/'approval_template.json',a)
    return {'status':'PREPARED_NOT_EXECUTED','plan_id':NAME,'fingerprint':p['fingerprint'],'generation_cap':10,
            'cached_responses':p['stored_prefix_responses'],'actual_LLM_generation':0}


def load():
    p=read(PUB/NAME/'plan.json')
    if digest({k:v for k,v in p.items() if k!='fingerprint'})!=p['fingerprint']:raise ValueError('PLAN_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('PIN_CHANGED')
    _,parent_pr,parent,bp=load_parent()
    if parent['fingerprint']!=p['parent_fingerprint']:raise ValueError('PARENT_CHANGED')
    return PUB/NAME,PR/NAME,p,bp,parent_pr


def permission(a,p):
    from scripts.run_original50_d0_d2 import permission as source_permission
    source_permission(a,p)
    if (any(a.get(k) is not True for k in ('accept_cached_prefix_terminal_replay','accept_authored_utility_transfer','accept_routing_abstention_er_fallback'))
        or a.get('experiment_scope')!=p['experiment_scope'] or a.get('conditions')!=['B4']):raise ValueError('NEW_TERMINAL_REPLAY_APPROVAL_REQUIRED')


def replay_one(task,*,p,bp,parent_pr,authorization,folder,factory=None):
    import app.locdi.current_ablation_llm_backend as native
    original_factory=native._native_transport;original_reserve=native.reserve_generation;controller=[]
    def transport(prepared):
        c=PrefixReplayTransport(task['source_episode'],prepared,factory or original_factory);controller.append(c);return c
    def reserve():
        if len(controller)!=1:raise ValueError('ONE_EPISODE_TRANSPORT_REQUIRED')
        return controller[0].reserve(original_reserve)
    raw=original_cases();idx=read(bp/'case_index.json');ref=task['case_ref'];case=raw[idx[ref]-1]
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    with terminal_only_scope(),patch.object(native,'_native_transport',new=transport),patch.object(native,'reserve_generation',new=reserve):
        backend=make_transfer_backend({ref:case},dataset_view='D2',augmented_rows={ref:aug[idx[ref]]},catalog=evidence_catalog(),
            config=read(bp/'llm_config.json'),bindings={ref:read(bp/'bindings.json')[ref]},model=read(M5),
            authorization=authorization,plan_fingerprint=p['fingerprint'],checkpoint_root=folder,
            routing_evidence=read(parent_pr/'U1_transfer_evidence.json'),history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),
            experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
        try: result=backend({'case_ref':ref},features_for('B4'))
        finally:
            if controller:
                c=controller[0];atomic_json(folder/'replay_ledger.json',{'cached_prefix':c.ledger,
                    'new_generation_attempts':c.new_attempts,'new_terminal_posts':c.new_posts,
                    'cached_identity_checks_are_not_current_observations':True,
                    'legacy_client_counters_include_replayed_calls':True})
    c=controller[0]
    if result['status']!='completed' or c.index!=len(c.before) or c.new_posts!=1:raise ValueError('REPLAY_NOT_COMPLETED')
    runtime=read(folder/'Episode0001/runtime_snapshot_private.json');trace=runtime['voi_state']['history'][-1]
    source_runtime=read(Path(task['source_episode'])/'runtime_snapshot_private.json');old=source_runtime['voi_state']['history']
    for left,right in zip(runtime['voi_state']['history'],old):
        for k in set(left)&set(right)-{'terminal_origin','terminal_contract','llm_terminal_certainty','hospital_terminal_certainty'}:
            if left[k]!=right[k]:raise ValueError('ORIGINAL_PLANNER_TRACE_CHANGED')
    if (len(runtime['voi_state']['history'])!=len(old) or trace['minute']!=task['stop_minute']
        or trace['terminal_origin']!='CURRENT_LLM_DIAGNOSE_CORE_REVALIDATED'):raise ValueError('TERMINAL_NOT_CORE_REVALIDATED')
    reference=read(bp/'panel.json')[task['panel_ordinal']-1]['reference']
    from scripts.analyze_original50_d0_d2 import facts
    f=facts(folder/'Episode0001',result,reference)
    # This is a saved-prefix technical replay, not N+1 fresh model generations.
    return {'panel_ordinal':task['panel_ordinal'],'status':'COMPLETED','cached_responses':c.index,
            'new_generation':c.new_posts,'old_source_exact':task['old_source_exact'],'source_exact':f['source_exact'],
            'prediction':result['prediction'],'facts':f,'terminal_core_revalidated':True}


def check():
    pub,pr,p,bp,parent_pr=load();blocks=['RUN_USED_NO_RESUME'] if (pr/p['run_id']).exists() else []
    proof=PUB/'TEST_ONLY_validation.json'
    if not proof.exists() or read(proof).get('status')!='PASS':blocks.append('OFFLINE_VALIDATION_REQUIRED')
    return {'status':'BLOCKED' if blocks else 'STATIC_READY_NEW_USER_APPROVAL_REQUIRED','blocker_codes':blocks,
            'plan_id':NAME,'generation_cap':10,'cached_responses':p['stored_prefix_responses'],'actual_LLM_generation':0}


def execute(approval):
    if check()['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p,bp,parent_pr=load();a=read(approval);permission(a,p)
    run=pr/p['run_id'];run.mkdir(exist_ok=False);atomic_json(run/'approval.json',a)
    atomic_json(run/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'plan_fingerprint':p['fingerprint'],'TEST_ONLY':False})
    start=time.monotonic();stats={f:(Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in p['pins']}
    def guard():
        permission(a,p)
        if (time.monotonic()-start>=p['wall_seconds'] or _process_private_bytes()>=p['process_bytes']
            or sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>=p['output_bytes']):raise ValueError('GLOBAL_RESOURCE_BUDGET')
        if any((Path(f).stat().st_size,Path(f).stat().st_mtime_ns)!=s for f,s in stats.items()):raise ValueError('PIN_CHANGED_DURING_RUN')
    budget=GenerationBudget(10,1,resource_guard=guard);rows=[];stop=None
    with (run/'rows.jsonl').open('x',encoding='utf-8') as stream:
        for t in read(pr/'tasks.json'):
            row={'panel_ordinal':t['panel_ordinal'],'status':'NOT_ATTEMPTED','new_generation':0}
            if stop is None:
                budget.episode_used=0;before=budget.used;token=_BUDGET.set(budget)
                try:
                    guard();row=replay_one(t,p=p,bp=bp,parent_pr=parent_pr,authorization=a,folder=run/('Panel%02d'%t['panel_ordinal']));guard()
                    if budget.used-before!=1:raise ValueError('NEW_REQUEST_COUNTER')
                except Exception as e:row.update(status='FAILED',error_type=type(e).__name__,private_error=str(e));stop='FIRST_ERROR_NO_RESUME'
                finally:row['new_generation']=budget.used-before;_BUDGET.reset(token)
            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
    c={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','rows':rows,'generation':budget.used,
       'wall_seconds':time.monotonic()-start,'plan_fingerprint':p['fingerprint'],'stop_reason':stop,**MARKERS}
    atomic_json(run/'collection.json',c)
    return {'status':c['status'],'generation':budget.used,'registered':10}


def analyze():
    pub,pr,p,bp,parent_pr=load();c=read(pr/p['run_id']/'collection.json')
    if c['plan_fingerprint']!=p['fingerprint'] or sum(r['new_generation'] for r in c['rows'])!=c['generation']:raise ValueError('COLLECTION_LINEAGE')
    out={'status':c['status'],'registered':10,'completed':sum(r['status']=='COMPLETED' for r in c['rows']),
         'Source_Exact_registered':sum(r.get('source_exact') is True for r in c['rows']),
         'unresolved':sum(r.get('prediction',{}).get('certainty')=='unresolved' for r in c['rows']),
         'new_generation':c['generation'],'comparison':p['comparison'],'plan_fingerprint':p['fingerprint'],**MARKERS}
    path=pub/'terminal_replay_results.json'
    if path.exists():
        if read(path)!=out:raise FileExistsError('ANALYSIS_ALREADY_EXISTS_DIFFERENT')
    else:atomic_json(path,out)
    return out


def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','execute','analyze']);q.add_argument('--execute',action='store_true');q.add_argument('--approval',type=Path);a=q.parse_args()
    if a.mode=='prepare':out=prepare()
    elif a.mode=='check':out=check()
    elif a.mode=='analyze':out=analyze()
    else:
        if not a.execute or not a.approval:q.error('NEW_USER_EXECUTION_APPROVAL_REQUIRED')
        out=execute(a.approval)
    print(json.dumps(out))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)
if __name__=='__main__':main()
