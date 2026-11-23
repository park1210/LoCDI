"""TEST_ONLY real Core/Source/Scheduler/Exporter, sockets denied, no LLM."""
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_original50_d0_d2 import load,SMOKE,AUG,PR,PUB,CONDITIONS,M5,UTILITY
from scripts.run_common_menu_b0_b4 import load_plan as source_load
from scripts.probe_knowledge_design_offline import KnowledgeFake
from app.locdi.original_source_inventory import original_cases,read
from app.locdi.augmentation_experiment_backend import make_paired_backend
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.original50_augmentation import initial_preview
from app.locdi.common_menu_backend import make_common_backend
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.development_routing_proxy import MARKERS

class Fake(KnowledgeFake):
    def __init__(self,config):super().__init__(config);self.wires=[]
    def post(self,url,*,json):self.wires.append(deepcopy(json));return super().post(url,json=json)

def main():
    # Independently pinned Source-only preparation, not an actual paired run.
    _,bp,base=source_load('original50_d0d2_smoke_002_source');p={'fingerprint':base['fingerprint']}
    cases=original_cases();idx=read(bp/'case_index.json');ref=next(iter(idx));n=idx[ref];case=cases[n-1]
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())};rows={ref:aug[n]}
    config=read(bp/'llm_config.json');model=read(M5);bindings=read(bp/'bindings.json');now=datetime.now(timezone.utc)
    a={**read(bp/'approval_template.json'),'paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY','dataset_views':['D0','D2'],
        'operator_ref':'TEST_ONLY_NOT_AUTHORITY','user_statement':'TEST_ONLY fake no model execution',
        'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy'):a[k]=True
    folder=PR/'TEST_ONLY';folder.mkdir(exist_ok=False);out=[];original=digest(case);captured={}
    for view in ('LEGACY','D0','D2'):
        for cond in CONDITIONS:
            fake=Fake(config);token=_BUDGET.set(GenerationBudget(12,12));name=view+'_'+cond
            try:
                with ExitStack() as s:
                    s.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                    s.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                    s.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
                    kw=dict(config=config,bindings=bindings,model=model,authorization=a,plan_fingerprint=p['fingerprint'],
                        checkpoint_root=folder/name,routing_evidence=read(UTILITY),
                        history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                    backend=make_common_backend({ref:case},**kw) if view=='LEGACY' else make_paired_backend({ref:case},dataset_view=view,augmented_rows=rows,catalog=evidence_catalog(),**kw)
                    result=backend({'case_ref':ref},features_for(cond))
                if result['status']!='completed':raise AssertionError(name+' CORE_NOT_COMPLETED')
                expected=initial_preview(case,aug[n],evidence_catalog(),view='D2' if view=='D2' else 'D0',accept_development_proxy=view=='D2')
                for payload in fake.payloads:
                    o=payload['observation'];assert o['chief_complaint']==expected['doctor_visible_info'].get('chief_complaint','')
                    assert set(expected['patient_visible_info']['associated_symptoms'])<=set(o['current_symptoms'])
                    assert not {'gold','evaluation','source_reference','provenance'}&set(payload)
                    assert 'synthetic_test_results' not in payload
                if view in ('LEGACY','D0'):captured[view,cond]=fake.wires
                if view=='D0':assert captured['LEGACY',cond]==fake.wires,'D0_LEGACY_WIRE_CHANGED'
                if view=='D2' and cond=='B4':
                    rt=read(folder/name/'Episode0001/runtime_snapshot_private.json');h=rt['voi_state']['history']
                    assert h and h[-1]['selected_action']=='diagnose'
                out.append({'profile':name,'status':'PASS','core_status':result['status'],'fake_POSTs':len(fake.posts),
                    'max_input_bound':max(read(f)['guard']['conservative_input_token_bound'] for f in (folder/name).glob('Episode*/Request*_before.json')),
                    'actual_LLM_generation':0,'input_state_delivered':True})
                assert digest(case)==original,'ORIGINAL_CASE_MUTATED'
            finally:_BUDGET.reset(token)
    atomic_json(PUB/'TEST_ONLY_core_probe.json',{'TEST_ONLY':True,'profiles':out,'old_versus_D0_wire_exact_all_five':True,
        'clinical_performance':'NOT_EVALUATED','actual_LLM_generation':0,'fitting':0,'real_smoke_executed':False})
    print(json.dumps({'profiles':len(out),'PASS':len(out),'actual_LLM_generation':0,'old_versus_D0_wire_exact':True}))
if __name__=='__main__':main()
