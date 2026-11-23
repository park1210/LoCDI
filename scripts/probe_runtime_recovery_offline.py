"""TEST_ONLY real Core/Source/Scheduler/Exporter with a fake HTTP transport.

No socket, real LLM, fitting or clinical result. Invalid evidence must fail.
"""
import argparse,json,socket,sys
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime,timedelta,timezone
from pathlib import Path
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
from app.locdi.ollama_adapter import LLMResponseError

class RecoveryFake(TestOnlyTransport):
    def __init__(self,config,invalid=False):super().__init__(config);self.invalid=invalid
    def post(self,url,*,json):
        self.posts.append(url);p=__import__('json').loads(json['messages'][1]['content']);self.payloads.append(p)
        eligible=p['supported_diagnosis_eligible_indices'];codes=_decision_nodes(json['format'])[0]['properties']['memory_subset_code']['enum']
        if eligible:
            action='diagnose';target=p['diagnosis_catalog'][0];certainty='supported';indices=[999] if self.invalid else [eligible[0]]
        else:
            c=next(c for c in p['candidate_actions'] if c['action']=='request_test')
            action=c['action'];target=c['target'];certainty='unresolved';indices=[]
        department=p['diagnosis_department_catalog'][target] if action=='diagnose' else p['department_catalog'][0]
        answer={'action':action,'target':target,'certainty':certainty,'department':department,
            'rationale':'TEST_ONLY no clinical correctness claim','assessment':'TEST_ONLY offline transport',
            'evidence_indices':indices,'memory_subset_code':next(c for c in codes if set(c)=={'0'})}
        return Reply({'model':self.config['model'],'done':True,'done_reason':'stop','prompt_eval_count':1,'eval_count':1,
            'message':{'content':__import__('json').dumps(answer)}})

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-id',required=True);a=p.parse_args();pub,pr,plan=load_plan(a.plan_id)
    assert plan['runtime_recovery'] and plan['catalog_selection'];config=read(pr/'llm_config.json');model=read(M5)
    index=read(pr/'case_index.json');ref=next(iter(index));case=load_cases(DATA)[index[ref]-1];bindings=read(pr/'bindings.json')
    evidence=read(plan['routing_evidence_path']);now=datetime.now(timezone.utc)
    auth={**MARKERS,'scope':plan['scope'],'plan_fingerprint':plan['fingerprint'],'conditions':['B0','B1','B2','B3','B4'],
        'operator_approved':True,'no_other_ollama_work':True,'accept_recorded_time_proxy':True,'accept_common_m5_menu':True,
        'accept_source_label_proxy':True,'clinical_release_approved':False,'operator_ref':'TEST_ONLY_NOT_EXECUTION_AUTHORITY',
        'user_statement':'TEST_ONLY fake transport; not an actual operator approval',
        'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    folder=ROOT/'outputs/private/research_evaluation/B34_RUNTIME_RECOVERY_20261121_v1/TEST_ONLY_Core'
    folder.mkdir(exist_ok=False);rows=[]
    for condition,invalid in [(c,False) for c in ('B0','B1','B2','B3','B4')]+[('B4',True)]:
        fake=RecoveryFake(config,invalid);token=_BUDGET.set(GenerationBudget(12,12));outdir=folder/(condition+('_invalid' if invalid else ''))
        with ExitStack() as stack:
            stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN')))
            stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN')))
            stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
            try:
                backend=make_common_backend({ref:case},config=config,bindings={ref:bindings[ref]},model=model,
                    authorization=auth,plan_fingerprint=plan['fingerprint'],checkpoint_root=outdir,
                    routing_evidence=evidence,history_policy=ContextBudgetedHistory(read(pr/'history_selection_plans.json')),
                    experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True)
                result=backend({'case_ref':ref},features_for(condition))
                if invalid:
                    assert result['status']=='failed';status='INVALID_EVIDENCE_REJECTED'
                else:
                    assert result['status']=='completed' and result['prediction']['certainty']=='supported';status='PASS'
                    if condition=='B4':
                        rt=read(outdir/'Episode0001/runtime_snapshot_private.json');h=rt['voi_state']['history'][-1]
                        assert h['selected_action']=='diagnose' and h['terminal_origin']=='CURRENT_LLM_DIAGNOSE_CORE_REVALIDATED'
                        assert h['m5_clinical_support'] is False
                for payload in fake.payloads:
                    for row in payload['released_measurements']:
                        assert payload['evidence'][row['evidence_index']]
                    assert not ({'gold','evaluation','source_event_store'}&set(payload))
            except LLMResponseError as exc:
                if not invalid:raise
                status='INVALID_EVIDENCE_REJECTED';result={'status':'failed','error_type':type(exc).__name__}
            finally:_BUDGET.reset(token)
        if invalid:
            checkpoints=[read(p) for p in outdir.glob('Episode*/Request*_after.json')]
            assert any(x.get('status')!='VALID' and x.get('error_type')=='LLMResponseError' for x in checkpoints)
        rows.append({'condition':condition,'invalid_evidence_test':invalid,'status':status,'core_status':result['status'],
            'fake_transport_calls':len(fake.posts),'actual_LLM_generation':0})
    report={'TEST_ONLY':True,'not_clinical_or_experiment_results':True,'actual_LLM_generation':0,'rows':rows}
    write(ROOT/'outputs/research_evaluation/B34_RUNTIME_RECOVERY_20261121_v1/TEST_ONLY_core_report.json',report)
    print(json.dumps(report))
if __name__=='__main__':main()
