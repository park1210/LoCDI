"""TEST_ONLY: deny sockets, real Core/Source/Exporter, no LLM or fitting."""
import json,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
from contextlib import ExitStack
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_common_menu_b0_b4 import load_plan,M5,DATA
from scripts.probe_common_menu_offline import Reply,TestOnlyTransport
from scripts.audit_research_gaps import read,write
from app.locdi.dataset_loader import load_cases
from app.locdi.common_menu_backend import make_common_backend
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.memory_citation_wire import _decision_nodes

class KnowledgeFake(TestOnlyTransport):
    def post(self,url,*,json):
        self.posts.append(url);p=__import__('json').loads(json['messages'][1]['content']);self.payloads.append(p)
        if p['supported_diagnosis_eligible_indices']:
            action='diagnose';target=p['diagnosis_catalog'][0];indices={str(p['supported_diagnosis_eligible_indices'][0]):True};certainty='supported'
        else:
            c=next(c for c in p['candidate_actions'] if c['action']=='request_test');action=c['action'];target=c['target'];indices={};certainty='unresolved'
        node=_decision_nodes(json['format'])[0];codes=node['properties']['memory_subset_code']['enum']
        answer={'action':action,'target':target,'certainty':certainty,'department':p['diagnosis_department_catalog'].get(target,p['department_catalog'][0]),
            'evidence_selection':indices,'memory_subset_code':next(c for c in codes if set(c)=={'0'}),
            'rationale':'TEST_ONLY not clinical','assessment':'TEST_ONLY offline'}
        return Reply({'model':self.config['model'],'done':True,'done_reason':'stop','prompt_eval_count':1,'eval_count':1,
            'message':{'content':__import__('json').dumps(answer)}})

def main():
    _,pr,plan=load_plan('knowledge_e1_p1_001');config=read(pr/'llm_config.json');model=read(M5);binding=read(pr/'bindings.json');index=read(pr/'case_index.json');ref=next(iter(index));case=load_cases(DATA)[index[ref]-1]
    evidence=read(plan['routing_evidence_path']);corpus=read(plan['knowledge_corpus_path']);now=datetime.now(timezone.utc)
    auth={**MARKERS,'scope':plan['scope'],'plan_fingerprint':plan['fingerprint'],'conditions':['B0','B1','B2','B3','B4'],
        'operator_approved':True,'no_other_ollama_work':True,'accept_source_label_proxy':True,'accept_common_m5_menu':True,
        'accept_recorded_time_proxy':True,'clinical_release_approved':False,'operator_ref':'TEST_ONLY_NOT_AUTHORITY',
        'user_statement':'TEST_ONLY fake transport, no execution approval',
        'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    profiles=[('E0R_B4','B4',False,False,False)]+[('E1_'+c,c,True,False,False) for c in ('B0','B1','B2','B3','B4')]+[('E2_'+c,c,False,True,False) for c in ('B1','B2')]+[('E3_'+c,c,False,False,True) for c in ('B3','B4')]
    folder=ROOT/'outputs/private/research_evaluation/DIAGNOSIS_KNOWLEDGE_DESIGN_20261121_v1/TEST_ONLY_CORE';folder.mkdir(exist_ok=False);rows=[];common_fps=set()
    for name,condition,knowledge,labels,rules in profiles:
        fake=KnowledgeFake(config);token=_BUDGET.set(GenerationBudget(12,12))
        try:
            with ExitStack() as stack:
                stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN')))
                stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN')))
                stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
                backend=make_common_backend({ref:case},config=config,bindings=binding,model=model,authorization=auth,
                    plan_fingerprint=plan['fingerprint'],checkpoint_root=folder/name,routing_evidence=evidence,
                    history_policy=ContextBudgetedHistory(read(pr/'history_selection_plans.json')),experiment_kind='DEVELOPMENT_EXPERIMENT',
                    catalog_selection=True,runtime_recovery=True,knowledge_corpus=corpus if knowledge else None,
                    memory_time_labels=labels,rule_based=rules,evidence_set=True)
                result=backend({'case_ref':ref},features_for(condition))
            assert result['status']=='completed'
            for p in fake.payloads:
                if knowledge:
                    assert len(p['external_medical_knowledge']['cards'])==15
                    common_fps.add(p['external_medical_knowledge']['corpus_fingerprint'])
                assert 'external_medical_knowledge' not in p['observation']
                assert 'knowledge' not in ' '.join(p['evidence']).lower()
                if labels:assert 'memory_scope_annotations' in p
            if rules:
                rt=read(folder/name/'Episode0001/runtime_snapshot_private.json')
                assert all(r['mode']=='rule_based' for r in rt['voi_state']['primary_routing_history'])
            rows.append({'profile':name,'status':'PASS','core_status':result['status'],'fake_calls':len(fake.posts),'actual_LLM_generation':0})
        finally:_BUDGET.reset(token)
    assert len(common_fps)==1
    report={'TEST_ONLY':True,'not_performance_results':True,'actual_LLM_generation':0,'fitting':0,'rows':rows,'identical_common_corpus_across_conditions':True}
    write(ROOT/'outputs/research_evaluation/DIAGNOSIS_KNOWLEDGE_DESIGN_20261121_v1/TEST_ONLY_core_report.json',report);print(json.dumps(report))
if __name__=='__main__':main()
