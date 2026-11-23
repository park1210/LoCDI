"""Network-denied 50x5 Core integration. Synthetic Utility is TEST_ONLY.

Not model generation, fitting, Source performance, U2 admission or an experiment.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_original50_full_readiness import PUB,PR
from scripts.run_original50_d0_d2 import AUG,M5
from scripts.probe_original50_evidence_subset_offline import SubsetFake
from app.locdi.original_source_inventory import original_cases,read,eligible
from app.locdi.original50_augmentation import initial_preview
from app.locdi.augmentation_experiment_backend import make_paired_backend
from app.locdi.terminal_only_recovery import terminal_only_scope
from app.locdi.context_budgeted_history import ContextBudgetedHistory,build_plan,MAX_JOINT_BYTES,FUTURE_RESERVE_BYTES
from app.locdi.case_adapter import adapt_case
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.routing_collector import atomic_json
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.routing_utility import METHOD_V2,observed_tokens


def main():
    folder=PR/'TEST_ONLY_full50';folder.mkdir(exist_ok=False)
    cfg=read(PR/'config.json');registry=validate_setup(cfg)['prepared']['registry'];cases=original_cases()
    wrappers={'Original%03d'%i:c for i,c in enumerate(cases,1)}
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    rows_by_ref={ref:aug[i] for i,ref in enumerate(wrappers,1)};bindings=read(PR/'bindings50.json')
    from app.locdi.literature_augmentation_catalog import evidence_catalog
    catalog=evidence_catalog();tokens=set()
    for i,c in enumerate(cases,1):
        state=initial_preview(c,aug[i],catalog,view='D2',accept_development_proxy=True)
        tokens.update(observed_tokens({'chief_complaint':state['doctor_visible_info']['chief_complaint'],
            'current_symptoms':state['patient_visible_info']['associated_symptoms']}))
    fixture={'schema_version':2,'method':METHOD_V2,'registry':registry,'min_samples':2,'samples':[
        {'patient_id':'TEST_ONLY_TRAIN_'+str(i),'decision_id':'TEST_ONLY_'+str(i),'agent_id':a['agent_id'],
         'correct':a['department']=='cardiology','tokens':sorted(tokens)}
        for i in (1,2) for a in registry['agents'] if a['role']=='specialty']}
    empty_plan=folder/'TEST_ONLY_empty_validation.json';atomic_json(empty_plan,{'candidates':[],'TEST_ONLY':True})
    auth={**MARKERS,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY',
          'dataset_views':['D0','D2'],'conditions':['B0','B1','B2','B3','B4'],'plan_fingerprint':digest('TEST_ONLY_FULL50'),
          'clinical_release_approved':False,'operator_ref':'TEST_ONLY_NOT_ACTUAL_APPROVAL','user_statement':'TEST_ONLY_NETWORK_DENIED'}
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu',
              'accept_label_conditioned_development','accept_augmentation_initial_time_policy'):auth[k]=True
    now=datetime.now(timezone.utc);auth.update(recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    histories=[];rows=[];baselines={};budget=GenerationBudget(3000,12)
    fixture_sha=digest(fixture);original_fingerprints={ref:digest(c) for ref,c in wrappers.items()}
    for condition in ('B0','B1','B2','B3','B4'):
        if condition=='B1':
            for i,c in enumerate(cases,1):
                adapted=adapt_case(c);identity=adapted.runtime_case['longitudinal']
                records=[r for v in adapted.runtime_case['past_visits'] for r in v['records']]
                context=PatientMemoryStore(records).retrieve(identity,'','full_history',max_chars=None)
                capacity=max(0,min(MAX_JOINT_BYTES,32768-1024-512-baselines.get(i,32768)-FUTURE_RESERVE_BYTES))
                try:histories.append(build_plan(identity,context,capacity))
                except Exception as e:rows.append({'ordinal':i,'condition':'B1_HISTORY_PREPARATION','status':'FAIL','error_type':type(e).__name__})
            atomic_json(folder/'history_selection_plans.json',histories)
        policy=ContextBudgetedHistory(histories) if condition=='B1' else None
        backend=make_paired_backend(wrappers,dataset_view='D2',augmented_rows=rows_by_ref,catalog=catalog,
            config=cfg,bindings=bindings,model=read(M5),authorization=auth,plan_fingerprint=auth['plan_fingerprint'],
            checkpoint_root=folder/condition,routing_evidence={'collection_plan':str(empty_plan)},history_policy=policy,
            experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
        for i,ref in enumerate(wrappers,1):
            fake=SubsetFake(cfg,True);budget.episode_used=0;token=_BUDGET.set(budget)
            row={'ordinal':i,'condition':condition,'status':'FAIL','actual_LLM_generation':0,'TEST_ONLY':True}
            try:
                with patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')),\
                        patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')),\
                        patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake),\
                        patch('app.locdi.common_menu_backend.admit_development_proxy',return_value={'artifact':fixture}),\
                        terminal_only_scope():
                    result=backend({'case_ref':ref},features_for(condition))
                if result['status']!='completed':raise AssertionError('CORE_NOT_COMPLETED')
                if len(fake.posts)>12:raise AssertionError('EPISODE_BUDGET')
                for p in fake.payloads:
                    if {'gold','evaluation','task_annotations','source_event_store','reference','future_results'}&set(p):raise AssertionError('PRIVATE_INPUT')
                    if condition=='B0' and p['observation']['longitudinal_memory']['records']:raise AssertionError('B0_MEMORY_LEAK')
                    if condition in ('B2','B3','B4') and len(p['observation']['longitudinal_memory']['records'])>5:raise AssertionError('TOP5')
                episode=folder/condition/('Episode%04d'%i)
                before=[read(f) for f in sorted(episode.glob('Request*_before.json'))]
                maximum=max(b['guard']['conservative_input_token_bound'] for b in before)
                if condition=='B0':baselines[i]=maximum
                if condition=='B4':
                    if sum(p.get('terminal_request_contract') is not None for p in fake.payloads)!=1:raise AssertionError('ONE_TERMINAL_REQUEST')
                    rt=read(episode/'runtime_snapshot_private.json')
                    if rt['voi_state']['history'][-1].get('terminal_extra_requests')!=1:raise AssertionError('TERMINAL_TRACE')
                if digest(wrappers[ref])!=original_fingerprints[ref] or digest(fixture)!=fixture_sha:raise AssertionError('INPUT_OR_UTILITY_MUTATION')
                row.update(status='PASS',fake_requests=len(fake.posts),max_input_bound=maximum,
                    terminal_extra_requests=1 if condition=='B4' else 0,fixture_utility_sha=fixture_sha if condition in ('B3','B4') else None)
            except Exception as e:
                row.update(error_type=type(e).__name__,private_error=str(e))
                if condition=='B0':
                    files=list((folder/condition/('Episode%04d'%i)).glob('Request*_before.json'))
                    baselines[i]=max((read(f)['guard']['conservative_input_token_bound'] for f in files),default=32768)
            finally:_BUDGET.reset(token)
            rows.append(row)
            print(condition,i,row['status'],flush=True)
    atomic_json(folder/'rows_private.json',rows)
    from collections import Counter
    states={condition:dict(Counter(r['status'] for r in rows if r['condition']==condition)) for condition in ('B0','B1','B2','B3','B4')}
    report={'status':'PASS' if len(rows)==250 and all(r['status']=='PASS' for r in rows) else 'FAIL',
        'registered_patients':50,'registered_episodes':250,'by_condition':states,'TEST_ONLY':True,'network':'DENIED',
        'actual_LLM_generation':0,'actual_U2_fitting':0,'U2_real_admission_test':'NOT_RUN_U2_ABSENT',
        'same_TEST_ONLY_utility_B3_B4':True,'fixture_is_not_a_development_utility':True,
        'fake_transport_posts':sum(r.get('fake_requests',0) for r in rows),
        'maximum_episode_fake_calls':max(r.get('fake_requests',0) for r in rows),
        'terminal_calls_within_existing_12_request_cap':True,'rows':[{k:v for k,v in r.items() if k!='private_error'} for r in rows]}
    atomic_json(PUB/'TEST_ONLY_full50.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    if report['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
