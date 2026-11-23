"""TEST_ONLY Native Core probes with legal finite evidence enum; sockets denied."""
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_original50_d0_d2 import load,SMOKE,AUG,PR,PUB,CONDITIONS,M5,UTILITY
from scripts.probe_common_menu_offline import Reply,TestOnlyTransport
from app.locdi.original_source_inventory import original_cases,read
from app.locdi.augmentation_experiment_backend import make_paired_backend
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.memory_citation_wire import _decision_nodes
from app.locdi.evidence_subset_wire import FIELD,VERSION
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json


class SubsetFake(TestOnlyTransport):
    def __init__(self,config,all_tests):super().__init__(config);self.all_tests=all_tests;self.wires=[]
    def post(self,url,*,json):
        self.posts.append(url);self.wires.append(deepcopy(json));p=__import__('json').loads(json['messages'][1]['content']);self.payloads.append(p)
        tests=[c for c in p['candidate_actions'] if c['action']=='request_test'];eligible=p['supported_diagnosis_eligible_indices']
        if tests and (self.all_tests or not eligible):action='request_test';target=tests[0]['target'];code='NONE';certainty='unresolved'
        else:
            action='diagnose';target=p['diagnosis_catalog'][0];code=str(eligible[0]) if eligible else 'NONE';certainty='supported' if eligible else 'unresolved'
        node=next(n for n in _decision_nodes(json['format']) if action in n['properties']['action'].get('enum',[]))
        assert code in node['properties'][FIELD]['enum']
        memory=node['properties']['memory_subset_code']['enum'];memory_none=next(c for c in memory if set(c)=={'0'})
        answer={'action':action,'target':target,'certainty':certainty,'department':p['diagnosis_department_catalog'].get(target,p['department_catalog'][0]),
            FIELD:code,'memory_subset_code':memory_none,'rationale':'TEST_ONLY','assessment':'TEST_ONLY'}
        return Reply({'model':self.config['model'],'done':True,'done_reason':'stop','prompt_eval_count':1,'eval_count':1,'message':{'content':__import__('json').dumps(answer)}})


def main():
    _,pr,p,bp=load(SMOKE);idx=read(bp/'case_index.json');ref=next(iter(idx));case=original_cases()[idx[ref]-1]
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    a=read(pr/'approval_template.json');now=datetime.now(timezone.utc)
    a.update(operator_ref='TEST_ONLY_NOT_APPROVAL',user_statement='TEST_ONLY_NO_NETWORK',recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy'):a[k]=True
    folder=PR/'subset_recovery_v1/TEST_ONLY';folder.mkdir(exist_ok=False);rows=[];original=digest(case)
    import app.locdi.evidence_selection_wire as old
    prior=(old.encode,old.decode,old.VERSION)
    for mode in ('EARLY_TERMINAL','ALL_TESTS'):
        for view in ('D0','D2'):
            for condition in CONDITIONS:
                config=read(bp/'llm_config.json');fake=SubsetFake(config,mode=='ALL_TESTS');token=_BUDGET.set(GenerationBudget(12,12));name=mode+'_'+view+'_'+condition
                try:
                    with ExitStack() as stack:
                        stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                        stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                        stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
                        backend=make_paired_backend({ref:case},dataset_view=view,augmented_rows={ref:aug[idx[ref]]},catalog=evidence_catalog(),config=config,
                            bindings=read(bp/'bindings.json'),model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],checkpoint_root=folder/name,
                            routing_evidence=read(UTILITY),history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),
                            experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                        result=backend({'case_ref':ref},features_for(condition))
                    assert result['status']=='completed',name
                    assert (old.encode,old.decode,old.VERSION)==prior and digest(case)==original
                    checkpoints=list((folder/name/'Episode0001').glob('Request*_before.json'))
                    rows.append({'profile':name,'status':'PASS','core_status':result['status'],'fake_requests':len(fake.posts),
                        'maximum_supplied_evidence':max(len(x['evidence']) for x in fake.payloads),
                        'max_input_bound':max(read(f)['guard']['conservative_input_token_bound'] for f in checkpoints)})
                finally:_BUDGET.reset(token)
    atomic_json(PUB/'subset_recovery_v1/TEST_ONLY_core_probe.json',{'TEST_ONLY':True,'profiles':rows,'actual_LLM_generation':0,'real_smoke_executed':False,'legacy_wire_restored':True})
    print(json.dumps({'PASS':len(rows),'actual_LLM_generation':0,'real_smoke_executed':False}))


if __name__=='__main__':main()
