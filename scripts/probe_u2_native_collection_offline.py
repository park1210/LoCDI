"""Five fixed specialists through actual Native Core; network forbidden."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_u2_native_collection import load,PUB,M5
from scripts.probe_original50_evidence_subset_offline import SubsetFake
from app.locdi.native_agent_collection import collect_one_backend
from app.locdi.original_source_inventory import read
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.core.hospital_llm_policy import HospitalLLMPolicy


def main():
    pub,pr,p=load('Smoke');patient=read(pr/'patients.json')[0];cfg=read(pr/'llm_config.json')
    output=pr/'TEST_ONLY';output.mkdir(exist_ok=False)
    a=read(pr/'approval_template.json');now=datetime.now(timezone.utc)
    for k in a:
        if k.startswith('accept_') or k in ('operator_approved','no_other_ollama_work'):a[k]=True
    a.update(operator_ref='TEST_ONLY_NOT_APPROVAL',user_statement='TEST_ONLY_NETWORK_DENIED',
             recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    states=[];rows=[];before=digest(patient);method=HospitalLLMPolicy._decision
    for agent in p['agents']:
        fake=SubsetFake(cfg,True);budget=GenerationBudget(12,12);token=_BUDGET.set(budget)
        try:
            with patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')),\
                    patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')),\
                    patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake):
                r=collect_one_backend(patient['case'],department=agent['department'],config=cfg,
                    bindings={'TRAIN':patient['binding']},model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],
                    checkpoint_root=output/agent['department'],experiment_kind='DEVELOPMENT_EXPERIMENT',
                    catalog_selection=True,runtime_recovery=True,evidence_set=True)
            if r['status']!='completed':raise AssertionError('NATIVE_CORE_NOT_COMPLETED')
            if budget.used!=len(fake.posts) or budget.used>12:raise AssertionError('BUDGET')
            for payload in fake.payloads:
                if payload['role']!='specialty' or payload['department']!=agent['department']:raise AssertionError('PRESPECIFIED_ACTUAL_AGENT_ROLE')
                if len(payload['observation']['longitudinal_memory']['records'])>5:raise AssertionError('TOP5')
                if {'evaluation','task_annotations','reference','source_event_store'}&set(payload):raise AssertionError('PRIVATE_INPUT')
            states.append(digest(fake.payloads[0]['observation']))
            if digest(patient)!=before or HospitalLLMPolicy._decision is not method:raise AssertionError('STATE_OR_LEGACY_CHANGED')
            rows.append({'department':agent['department'],'status':'PASS','Fake_requests':budget.used,
                'actual_LLM_generation':0,'Core_status':r['status'],'clinical_accuracy':'NOT_EVALUATED'})
        finally:_BUDGET.reset(token)
    if len(set(states))!=1:raise AssertionError('FIVE_AGENT_SHARED_INITIAL_STATE')
    atomic_json(PUB/'U2_NATIVE_TEST_ONLY_validation.json',{'status':'PASS','TEST_ONLY':True,'actual_LLM_generation':0,
        'Native_episodes':5,'same_initial_observation':True,'Source_unchanged':True,'legacy_policy_restored':True,
        'rows':rows,'plan_fingerprint':p['fingerprint'],'network':'DENIED'})
    print(json.dumps({'status':'PASS','TEST_ONLY_Native_episodes':5,'same_initial_observation':True,'actual_LLM_generation':0}))
if __name__=='__main__':main()
