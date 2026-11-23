"""Fresh network-denied C43510 + Original50x5 Native integration proof.

TEST_ONLY Utility/Admission are explicitly fabricated fixtures. Production
Admission cannot consume this output as actual Validation or fitting evidence.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
from collections import Counter
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import native_routing_transfer as transfer
from scripts.prepare_original50_final import PR as SOURCE_PR
from scripts.run_original50_d0_d2 import M5,AUG
from scripts.probe_original50_evidence_subset_offline import SubsetFake
from app.locdi.original_source_inventory import read,original_cases,sha_file
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.native_agent_collection import collect_one_backend
from app.locdi.native_menu_routing import backend as transfer_backend,public_menu,tokens
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.development_routing_proxy import MARKERS

def authority(plan,conditions):
    now=datetime.now(timezone.utc)
    a={**MARKERS,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY',
       'dataset_views':['D0','D2'],'conditions':conditions,'plan_fingerprint':plan,'clinical_release_approved':False,
       'operator_ref':'TEST_ONLY_NOT_ACTUAL_APPROVAL','user_statement':'TEST_ONLY_NETWORK_DENIED',
       'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy'):a[k]=True
    return a

def main():
    _,vpr,vp=transfer.load_validation();root=transfer.PR/'TEST_ONLY_transfer_002';root.mkdir(exist_ok=False)
    cfg=read(vpr/'llm_config.json');budget=GenerationBudget(3200,12);rows=[];validation=[]
    def network():return patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY'))
    # New actual-source adapter and the original fixed five-Agent collection.
    for patient in read(vpr/'patients.json'):
        signatures=set()
        for agent in vp['agents']:
            fake=SubsetFake(cfg,False);budget.episode_used=0;token=_BUDGET.set(budget)
            try:
                with network(),patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')),\
                        patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake):
                    result=collect_one_backend(patient['case'],department=agent['department'],config=cfg,
                        bindings={'TRAIN':patient['binding']},model=read(M5),authorization=authority(digest('TEST_ONLY_VALIDATION'),['B2']),
                        plan_fingerprint=digest('TEST_ONLY_VALIDATION'),checkpoint_root=root/'validation'/('Case%02d'%patient['ordinal'])/agent['department'],
                        experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                if result['status']!='completed':raise AssertionError('NATIVE_VALIDATION_FAKE_NOT_COMPLETED')
                first=fake.payloads[0]
                if first['evidence'] or first['observation']['objective_findings']:raise AssertionError('PRE_REQUEST_RESULT_LEAK')
                if first['role']!='specialty' or first['department']!=agent['department']:raise AssertionError('FIXED_ROLE_NOT_EXECUTED')
                signatures.add(digest({'observation':first['observation'],'menu':first['source_action_metadata']}))
                validation.append({'ordinal':patient['ordinal'],'agent':agent['department'],'status':'PASS','fake_posts':len(fake.posts)})
            finally:_BUDGET.reset(token)
        if len(signatures)!=1:raise AssertionError('FIVE_AGENT_PUBLIC_INPUT_DIFFERS')
    cases={'Original%03d'%i:c for i,c in enumerate(original_cases(),1)};bindings=read(SOURCE_PR/'bindings50.json')
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    menu=public_menu([row for b in bindings.values() for row in b['rows']])
    fixture={'schema_version':2,'method':'jaccard_beta_accuracy_v2','registry':vp['registry'],'min_samples':2,
        'samples':[{'patient_id':'TEST_ONLY_TRAIN_'+str(i),'decision_id':'D'+str(i),'agent_id':a['agent_id'],
                   'tokens':tokens({},menu),'correct':a['department']=='cardiology'} for i in range(2) for a in vp['agents']]}
    empty=root/'TEST_ONLY_empty_validation.json';atomic_json(empty,{'candidates':[],'TEST_ONLY':True})
    evidence={'collection_plan':str(empty),'routing_artifact':fixture,'TEST_ONLY':True}
    admitted=lambda evidence,registry,patients,config:{'artifact':fixture,'TEST_ONLY':True}
    fixture_fp=digest(fixture);auth=authority(digest('TEST_ONLY_FULL250'),['B0','B1','B2','B3','B4'])
    backend=transfer_backend(cases,admitted_evidence=evidence,admitter=admitted,dataset_view='D2',
        augmented_rows={ref:aug[int(ref[-3:])] for ref in cases},catalog=evidence_catalog(),config=cfg,bindings=bindings,
        model=read(M5),authorization=auth,plan_fingerprint=auth['plan_fingerprint'],checkpoint_root=root/'full250',
        history_policy=ContextBudgetedHistory(read(SOURCE_PR/'TEST_ONLY_full50/history_selection_plans.json')),
        experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
    slot=0
    for ref in cases:
        for condition in ('B0','B1','B2','B3','B4'):
            slot+=1;fake=SubsetFake(cfg,False);budget.episode_used=0;token=_BUDGET.set(budget)
            try:
                with network(),patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')),\
                        patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake):
                    result=backend({'case_ref':ref},features_for(condition))
                if result['status']!='completed' or len(fake.posts)>12:raise AssertionError('FULL250_CORE_COMPLETION_OR_BUDGET')
                for body in fake.payloads:
                    if {'evaluation','gold','source_event_store','reference','future_results'}&set(body):raise AssertionError('PRIVATE_MODEL_INPUT')
                    mem=body['observation'].get('longitudinal_memory',{}).get('records',[])
                    if condition=='B0' and mem:raise AssertionError('B0_MEMORY_LEAK')
                    if condition in ('B2','B3','B4') and len(mem)>5:raise AssertionError('TOP5_CHANGED')
                terminal=sum('terminal_request_contract' in body for body in fake.payloads)
                if condition=='B4' and terminal!=1:raise AssertionError('B4_TERMINAL_ONLY_SINGLE_REQUEST')
                if digest(fixture)!=fixture_fp:raise AssertionError('SAME_UTILITY_NOT_PRESERVED')
                rows.append({'ordinal':int(ref[-3:]),'condition':condition,'status':'PASS','fake_posts':len(fake.posts),'terminal_requests':terminal})
            finally:_BUDGET.reset(token)
        if slot%25==0:print('TEST_ONLY_FULL',slot,'PASS',flush=True)
    atomic_json(root/'rows_private.json',{'validation':validation,'full250':rows,'TEST_ONLY':True})
    out={'status':'PASS','TEST_ONLY':True,'network':'DENIED','validation_native_episodes_PASS':len(validation),'Original50_conditions_PASS':len(rows),
         'same_utility_B3_B4':True,'full250_fake_posts':sum(r['fake_posts'] for r in rows),'max_episode_fake_posts':max(r['fake_posts'] for r in rows),
         'B4_terminal_calls':sum(r['terminal_requests'] for r in rows if r['condition']=='B4'),'actual_LLM_generation':0,
         'actual_source_fitting':0,'production_admission_passed':False,'feature_version':transfer.VERSION}
    atomic_json(transfer.PUB/'TEST_ONLY_validation_and_full250.json',out);print(json.dumps(out))

if __name__=='__main__':main()
