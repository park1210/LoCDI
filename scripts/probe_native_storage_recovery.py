"""Actual Windows storage test + ten fresh Fake Native episodes, no sockets."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
import json,sys,os,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import native_transfer_recovery as recovery
from scripts import native_routing_transfer as transfer
from scripts.probe_original50_evidence_subset_offline import SubsetFake
from app.locdi.original_source_inventory import read
from app.locdi.routing_collector import atomic_json
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_atomic_json
from app.locdi.native_agent_collection import collect_one_backend
from scripts.run_original50_d0_d2 import M5

def main():
    root=recovery.PR/'TEST_ONLY_storage';root.mkdir(exist_ok=False)
    # Reproduce using the original writer, not the original experiment directory.
    with tempfile.TemporaryDirectory(dir=str(root)) as tmp:
        parent=Path(tmp)/('x'*(230-len(str(Path(tmp).resolve()))-1));parent.mkdir()
        failed=parent/'Request0001_core_response.json'
        try:_atomic_json(failed,{'TEST_ONLY':True});reproduced=False
        except (OSError,ValueError):reproduced=True
        old_probe={'absolute_file_chars':len(str(failed.resolve())),'old_writer_path_failure_reproduced':reproduced,'platform':os.name}
    with recovery.scope():_,pr,p=transfer.load_validation()
    cfg=read(pr/'llm_config.json');a=read(pr/'approval_template.json');now=datetime.now(timezone.utc)
    a.update(operator_ref='TEST_ONLY_NOT_ACTUAL_APPROVAL',user_statement='TEST_ONLY_NO_NETWORK',recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_native_training_intervention','accept_source_reference_date_proxy','accept_prior_VoI_exposure_and_bounded_independence','accept_storage_recovery_and_repeat_validation'):a[k]=True
    budget=GenerationBudget(120,12);rows=[]
    for patient in read(pr/'patients.json'):
        for agent in p['agents']:
            folder=root/('P%02d'%patient['ordinal'])/agent['department']
            fake=SubsetFake(cfg,False);budget.episode_used=0;token=_BUDGET.set(budget)
            try:
                with patch('socket.socket.connect',side_effect=AssertionError('TEST_ONLY_NETWORK_DENIED')),\
                        patch('socket.create_connection',side_effect=AssertionError('TEST_ONLY_NETWORK_DENIED')),\
                        patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake):
                    result=collect_one_backend(patient['case'],department=agent['department'],config=cfg,bindings={'TRAIN':patient['binding']},
                        model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],checkpoint_root=folder,
                        experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                if result['status']!='completed':raise AssertionError('NATIVE_FAKE_NOT_COMPLETED')
                paths=[folder/'Episode0001'/f for f in ('runtime_snapshot_private.json','Request0001_core_response.json','episode_and_policy.json')]
                if not all(f.exists() for f in paths):raise AssertionError('EXPORT_MISSING')
                for f in paths:read(f)
                rows.append({'case_ordinal':patient['ordinal'],'department':agent['department'],'status':'PASS','fake_posts':len(fake.posts),
                    'snapshot_path_chars':len(str(paths[0].resolve()))})
            finally:_BUDGET.reset(token)
    proof={'status':'PASS','TEST_ONLY':True,'old_path_probe':old_probe,'native_episodes_PASS':len(rows),'all_actual_source_agents_exported':True,
        'actual_LLM_generation':0,'actual_fit_calls':0,'rows':rows,'network':'DENIED'}
    atomic_json(recovery.PUB/'TEST_ONLY_storage_recovery.json',proof);print(json.dumps(proof))

if __name__=='__main__':main()
