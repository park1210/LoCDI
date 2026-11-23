"""TEST_ONLY Fake Transport through real Native/Core/Exporter; network denied.

Never connects to Ollama. Optional routing fixture is nonclinical test evidence,
not a Source-label Utility or an experiment. All outputs are explicitly TEST_ONLY.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write
from scripts.run_common_menu_b0_b4 import load_plan,M5,DATA
from app.locdi.llm_setup import digest
from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case
from app.locdi.common_menu_backend import make_common_backend
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.memory_citation_wire import _decision_nodes
from app.core.routing_utility import METHOD_V2,observed_tokens


class Reply:
    status_code=200
    def __init__(self,data):self.data=data;self.content=json.dumps(data).encode()
    def json(self):return deepcopy(self.data)
    def raise_for_status(self):return None


class TestOnlyTransport:
    def __init__(self,config):self.config=config;self.posts=[];self.payloads=[]
    def get(self,url):
        return Reply({'models':[{'name':self.config['model'],'digest':self.config['model_identifier'],
            'context_length':32768,'size':1,'size_vram':1}]})
    def post(self,url,*,json):
        self.posts.append(url);req=json;payload=__import__('json').loads(req['messages'][1]['content']);self.payloads.append(payload)
        options=payload['candidate_actions']
        selected=next((c for action in ('ask_patient','request_test') for c in options if c['action']==action),None)
        action,target=(selected['action'],selected['target']) if selected else ('diagnose','Unresolved')
        nodes=_decision_nodes(req['format'])
        codes=next(n['properties']['memory_subset_code']['enum'] for n in nodes if 'memory_subset_code' in n.get('properties',{}))
        answer={'action':action,'target':target,'rationale':'TEST_ONLY source menu; no real diagnosis.',
            'assessment':'TEST_ONLY Fake Transport; no clinical judgment.',
            'department':payload.get('department') or 'general_medicine','certainty':'unresolved',
            'evidence_indices':[],'memory_subset_code':next(c for c in codes if set(c)=={'0'})}
        return Reply({'model':req['model'],'done':True,'done_reason':'stop','prompt_eval_count':1,'eval_count':1,
            'message':{'content':__import__('json').dumps(answer)}})
    def close(self):pass


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan-id',required=True)
    p.add_argument('--output-id',required=True);p.add_argument('--routing-fixture',action='store_true');a=p.parse_args()
    pub,pr,plan=load_plan(a.plan_id);config=read(pr/'llm_config.json');bindings=read(pr/'bindings.json')
    cases=load_cases(DATA);index=read(pr/'case_index.json');ref=next(iter(index));case=cases[index[ref]-1]
    registry=read(pr/'registry.json');model=read(M5);fixture=None
    if a.routing_fixture:
        adapted=adapt_case(case,registry_manifest=registry)
        past=[r for visit in adapted.runtime_case['past_visits'] for r in visit['records']]
        tokens=observed_tokens({'chief_complaint':json.dumps(adapted.runtime_case['current_initial_state']['doctor_visible_info']),
            'longitudinal_memory':{'records':past}})
        fixture={'schema_version':2,'method':METHOD_V2,'registry':registry,'min_samples':2,'samples':[
            {'patient_id':'TEST_ONLY_TRAIN_'+str(i),'decision_id':'TEST_ONLY_'+str(i),'agent_id':agent['agent_id'],
             'correct':agent['department']=='cardiology','tokens':tokens}
            for i in (1,2) for agent in registry['agents'] if agent['role']=='specialty']}
    if not a.output_id.replace('_','').isalnum():raise ValueError('SAFE_TEST_OUTPUT_ID')
    folder=ROOT/'outputs/private/research_evaluation/B34_SOURCE_READY_v1/offline_TEST_ONLY'/a.output_id
    folder.mkdir(parents=True,exist_ok=False);out=[]
    now=datetime.now(timezone.utc)
    auth={**MARKERS,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','plan_fingerprint':plan['fingerprint'],
        'conditions':['B0','B1','B2','B3','B4'],'operator_approved':True,'no_other_ollama_work':True,
        'accept_common_m5_menu':True,'accept_recorded_time_proxy':True,'accept_source_label_proxy':True,
        'clinical_release_approved':False,'operator_ref':'TEST_ONLY_NOT_A_REAL_APPROVAL',
        'user_statement':'TEST_ONLY Fake Transport, never actual execution authority',
        'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    for condition in ('B0','B1','B2','B3','B4'):
        transport=TestOnlyTransport(config);budget=GenerationBudget(12,12);token=_BUDGET.set(budget)
        row={'condition':condition,'test_only':True,'actual_LLM_generation':0,'status':'FAIL'}
        try:
            with ExitStack() as stack:
                stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=transport))
                stack.enter_context(patch('app.locdi.common_menu_backend.admit_development_proxy',return_value={'artifact':fixture}))
                backend=make_common_backend({ref:case},config=config,bindings={ref:bindings[ref]},model=model,
                    authorization=auth,plan_fingerprint=plan['fingerprint'],checkpoint_root=folder/condition,
                    routing_evidence={'collection_plan':str(pr/'routing_training/plan.json')},
                    history_policy=ContextBudgetedHistory(read(pr/'history_selection_plans.json')),
                    experiment_kind='DEVELOPMENT_EXPERIMENT')
                result=backend({'case_ref':ref},features_for(condition))
            row.update(core_status=result['status'],fake_transport_requests=len(transport.posts),
                routing_selected=result['routing_selected_count'],routing_abstained=result['routing_abstention_count'],
                computation=result['computation'])
            if result['status']!='completed':raise AssertionError('TEST_ONLY_CORE_NOT_COMPLETED')
            for call in transport.payloads:
                if {'gold','evaluation','source_event_store','reference'}&set(call):raise AssertionError('PRIVATE_PAYLOAD')
                if condition=='B0' and call['observation']['longitudinal_memory']['records']:raise AssertionError('MEMORY_OFF')
                if condition in ('B2','B3','B4') and len(call['observation']['longitudinal_memory']['records'])>5:raise AssertionError('MEMORY_TOP5')
            if condition in ('B3','B4') and a.routing_fixture and result['routing_selected_count']==0:
                raise AssertionError('TEST_ONLY_SPECIALTY_PATH_NOT_EXECUTED')
            if condition in ('B3','B4') and not a.routing_fixture and result['routing_abstention_count']==0:
                raise AssertionError('TEST_ONLY_ABSTENTION_NOT_EXECUTED')
            if condition=='B4' and result['computation']['planner_calls']==0:raise AssertionError('TEST_ONLY_PLANNER_NOT_CALLED')
            row['status']='PASS'
        except Exception as exc:row.update(error_type=type(exc).__name__,error=str(exc)[:120])
        finally:_BUDGET.reset(token)
        out.append(row)
    report={'test_only':True,'not_clinical_or_experimental_results':True,'actual_LLM_generation':0,
        'routing_fixture':a.routing_fixture,'rows':out,'passed':sum(r['status']=='PASS' for r in out),'total':5}
    write(folder/'TEST_ONLY_report.json',report)
    public=ROOT/'outputs/research_evaluation/B34_SOURCE_READY_v1'/('TEST_ONLY_'+a.output_id+'.json')
    write(public,report);print(json.dumps(report,ensure_ascii=True))
    if report['passed']!=5:raise SystemExit(1)


if __name__=='__main__':main()
