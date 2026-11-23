"""TEST_ONLY full20 Core/ER fallback and five-Agent Source collection contracts."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
from contextlib import ExitStack
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_u1_voi_expansion import load,PUB,PR,AUG,M5,analyze
from scripts.probe_original50_evidence_subset_offline import SubsetFake
from scripts.probe_common_menu_offline import Reply,TestOnlyTransport
from app.locdi.authored_utility_transfer import make_transfer_backend
from app.locdi.original_source_inventory import read,original_cases
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.routing_collector import atomic_json

def main():
    pub,pr,p,bp=load();root=PR/'TEST_ONLY_FULL20';root.mkdir(exist_ok=False);run=root/'RUN_TEST_ONLY';run.mkdir();fakepub=root/'PUBLIC';fakepub.mkdir()
    idx=read(bp/'case_index.json');original=original_cases();cases={ref:original[n-1] for ref,n in idx.items()};aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    a=read(pr/'approval_template.json');now=datetime.now(timezone.utc);a.update(operator_ref='TEST_ONLY_NOT_AUTHORITY',user_statement='TEST_ONLY_NO_NETWORK',recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    for key,value in list(a.items()):
        if key.startswith('accept_') or key in ('operator_approved','no_other_ollama_work'):a[key]=True
    rows=[];profiles=[];budget=GenerationBudget(240,12);cfg=read(bp/'llm_config.json')
    for task in read(pr/'tasks.json'):
        # New closure per profile, moving its actual Episode0001 to fixed task ID
        # is avoided: give each profile a separate fixture root.
        name='P%02d_%s'%(task['panel_ordinal'],task['condition']);fake=SubsetFake(cfg,True);budget.episode_used=0;before=budget.used;token=_BUDGET.set(budget)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
                backend=make_transfer_backend(cases,dataset_view='D2',augmented_rows={ref:aug[n] for ref,n in idx.items()},catalog=evidence_catalog(),config=cfg,
                    bindings=read(bp/'bindings.json'),model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],checkpoint_root=root/name,
                    routing_evidence=read(pr/'U1_transfer_evidence.json'),history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),
                    experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                r=backend({'case_ref':task['case_ref']},features_for(task['condition']))
            assert r['status']=='completed',name
            folder=root/name/'Episode0001';runtime=read(folder/'runtime_snapshot_private.json');initial=runtime['voi_state']['initial_primary_routing'];expected='er' if initial['selected_specialty'] is None else 'specialty'
            assert fake.payloads[0]['role']==expected
            profiles.append({'profile':name,'status':'PASS','initial_selection':initial['selected_specialty'],'actual_initial_role':fake.payloads[0]['role'],
                'fake_requests':len(fake.posts),'max_input_bound':max(read(f)['guard']['conservative_input_token_bound'] for f in folder.glob('Request*_before.json'))})
        finally:_BUDGET.reset(token)
    # Source-only one patient x5 actual model-schema calls via Fake transport.
    import scripts.prepare_u2_source_collection as b
    from app.llm.prompt_loader import specialty_system_prompt
    from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient
    from app.locdi.synthetic_pilot_collection import response_schema
    _,bpr,bp2=b.load();candidate=read(bpr/'candidates.json')[0];fingerprints=[];source_count=0
    class SourceFake(TestOnlyTransport):
        def post(self,url,*,json):
            self.posts.append(url);body=__import__('json').loads(json['messages'][1]['content']);fingerprints.append(__import__('app.locdi.llm_setup',fromlist=['digest']).digest({k:v for k,v in body.items() if k!='department'}))
            answer={'action':'diagnose','diagnosis':body['diagnosis_catalog'][0],'certainty':'supported','rationale':'TEST_ONLY'}
            return Reply({'model':self.config['model'],'done':True,'done_reason':'stop','prompt_eval_count':1,'eval_count':1,'message':{'content':__import__('json').dumps(answer)}})
    bbudget=GenerationBudget(5,1)
    for agent in bp2['agents']:
        fake=SourceFake(bp2['config']);bbudget.episode_used=0;token=_BUDGET.set(bbudget)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
                client=BudgetedNativeERClient(bp2['config'],authorized=True,contract_fingerprint=bp2['fingerprint'],checkpoint_root=root/'SOURCE_FIVE_AGENT'/agent['department'])
                text=client.chat_json(specialty_system_prompt(agent['department']),json.dumps({'department':agent['department'],**candidate['shared']},sort_keys=True),bp2['config']['model'],schema=response_schema(bp2['catalog']))
                assert json.loads(text)['action']=='diagnose';source_count+=1
        finally:_BUDGET.reset(token)
    assert len(set(fingerprints))==1 and source_count==5 and bbudget.used==5
    atomic_json(PUB/'TEST_ONLY_full20_probe.json',{'TEST_ONLY':True,'profiles':profiles,'Source_fake_calls':5,'five_agent_shared_fingerprint_same':True,'actual_LLM_generation':0,'actual_experiment_run_created':False})
    print(json.dumps({'Core_profiles_PASS':len(profiles),'Source_5Agent_PASS':source_count,'actual_LLM_generation':0}))
if __name__=='__main__':main()
