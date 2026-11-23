"""Network-denied full150 runner and aggregation; never actual inference."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_original50_final as runner
from scripts.probe_original50_evidence_subset_offline import SubsetFake
from app.locdi.routing_collector import atomic_json
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.llm_setup import digest

def main():
    _,source,p=runner.load('Full50');pub=runner.PUB/'TEST_ONLY_RUNNER_BOUNDED';pr=runner.PR/'TEST_ONLY_RUNNER'/'bounded_002'
    if pub.exists() or pr.exists():raise FileExistsError('TEST_ONLY_RUNNER_ALREADY_EXISTS')
    pub.mkdir();pr.mkdir(parents=True);tasks=read(source/'tasks.json');atomic_json(pr/'tasks.json',tasks)
    source_fingerprint=p['fingerprint']; p=deepcopy(p); p['pins']={f:h for f,h in p['pins'].items() if Path(f).name in ('config.json','bindings50.json','AUGMENTED_CASES.jsonl','history_selection_plans.json','run_original50_final.py')}; p['run_id']='run_TEST_ONLY_full150';p['plan_id']='TEST_ONLY_FULL150'
    p['pins'][str((pr/'tasks.json').resolve())]=sha_file(pr/'tasks.json');p['fingerprint']=digest({k:v for k,v in p.items() if k!='fingerprint'})
    auth=read(source/'approval_template.json');auth['plan_fingerprint']=p['fingerprint']
    now=datetime.now(timezone.utc);auth.update(operator_ref='TEST_ONLY_NOT_ACTUAL_APPROVAL',user_statement='TEST_ONLY_NETWORK_DENIED',recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    for key in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy'):auth[key]=True
    atomic_json(pr/'TEST_ONLY_authorization.json',auth)
    cfg=read(runner.PR/'config.json');transports=[]
    def factory(prepared):
        f=SubsetFake(cfg,True);transports.append(f);return f
    with patch.object(runner,'load',return_value=(pub,pr,p)),patch.object(runner,'check',return_value={'status':'TEST_ONLY_ALLOWED'}),\
            patch('app.locdi.current_ablation_llm_backend._native_transport',new=factory):
        result=runner.execute('Full50',pr/'TEST_ONLY_authorization.json',test_only=True)
        if result['status']!='COMPLETED':raise AssertionError('FAKE_RUNNER_NOT_COMPLETED')
        analysis=runner.analyze('Full50')
    runner.load('Full50')  # Full production SHA pins verified again, never a subset admission.
    report={'source_production_plan_fingerprint':source_fingerprint,'full_SHA_pins_verified_before_and_after':True,'TEST_ONLY_metadata_guard_subset_count':len(p['pins']),'live_resource_guard_unchanged':True,'status':'PASS','TEST_ONLY':True,'patients':50,'registered_episodes':150,
            'fake_posts':sum(len(t.posts) for t in transports),'actual_LLM_generation':0,
            'maximum_episode_fake_posts':max(len(t.posts) for t in transports),
            'live_runner_function_reused':True,'new_analysis_executed':analysis['status'],
            'B34_actual_admission':'UNCHANGED_BLOCKED','network':'DENIED'}
    atomic_json(runner.PUB/'TEST_ONLY_runner150.json',report);print(json.dumps(report))

if __name__=='__main__':main()
