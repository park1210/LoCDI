"""TEST_ONLY R0/R1/VoI real Core path; fake transport and sockets denied."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
from contextlib import ExitStack
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_b34_transfer_smoke import load,IDS,U0,AUG,M5
from scripts.probe_original50_evidence_subset_offline import SubsetFake
from app.locdi.authored_utility_transfer import make_transfer_backend
from app.locdi.original_source_inventory import read,original_cases
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.routing_collector import atomic_json

def main():
    _,pr,p,bp=load(IDS['VoI']);ref=p['case_ref'];idx=read(bp/'case_index.json');case=original_cases()[idx[ref]-1]
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    a=read(pr/'approval_template.json');now=datetime.now(timezone.utc)
    a.update(operator_ref='TEST_ONLY_NOT_ACTUAL_AUTHORITY',user_statement='TEST_ONLY_NO_NETWORK',recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy','accept_authored_utility_transfer'):a[k]=True
    folder=ROOT/'outputs/private/research_evaluation/B34_UTILITY_VOI_IMPROVEMENT_v1/TEST_ONLY';folder.mkdir(exist_ok=False);rows=[]
    import app.locdi.common_menu_backend as module
    original_admit=module.admit_development_proxy
    for branch,cond in (('R0','B3'),('R1','B3'),('R1','B4')):
        cfg=read(bp/'llm_config.json');fake=SubsetFake(cfg,False);token=_BUDGET.set(GenerationBudget(12,12));name=branch+'_'+cond
        try:
            with ExitStack() as stack:
                stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
                backend=make_transfer_backend({ref:case},dataset_view='D2',augmented_rows={ref:aug[idx[ref]]},catalog=evidence_catalog(),config=cfg,
                    bindings=read(bp/'bindings.json'),model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],checkpoint_root=folder/name,
                    routing_evidence=read(U0) if branch=='R0' else read(pr/'U1_transfer_evidence.json'),
                    history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                result=backend({'case_ref':ref},features_for(cond))
            assert result['status']=='completed',name
            assert module.admit_development_proxy is original_admit
            runtime=read(folder/name/'Episode0001/runtime_snapshot_private.json');routing=runtime['voi_state']['primary_routing_history'];selected=routing[0]['selected_specialty']
            assert selected==('nephrology' if branch=='R0' else 'neurology')
            rows.append({'profile':name,'status':'PASS','core_status':result['status'],'initial_selected':selected,'fake_requests':len(fake.posts)})
        finally:_BUDGET.reset(token)
    atomic_json(ROOT/'outputs/research_evaluation/B34_UTILITY_VOI_IMPROVEMENT_v1/TEST_ONLY_core_probe.json',{'TEST_ONLY':True,'rows':rows,'actual_LLM_generation':0,'actual_smoke_executed':False})
    print(json.dumps({'Fake_Core_PASS':len(rows),'actual_LLM_generation':0}))
if __name__=='__main__':main()
