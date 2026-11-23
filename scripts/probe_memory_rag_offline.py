"""TEST_ONLY real Core/Source/Exporter + Fake Transport. Sockets forbidden."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_common_menu_b0_b4 import load_plan,M5,DATA
from scripts.probe_knowledge_design_offline import KnowledgeFake
from app.locdi.original_source_inventory import memory_source_map,read
from app.locdi.memory_rag_backend import make_memory_rag_backend
from app.locdi.common_menu_backend import make_common_backend
from app.locdi.dataset_loader import load_cases
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,features_for
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.medical_knowledge_catalog import corpus
NS='MEMORY_RAG_LEGACY_RECOVERY_20261121_v1'

class CapturingFake(KnowledgeFake):
    def __init__(self,config):super().__init__(config);self.requests=[]
    def post(self,url,*,json):
        self.requests.append(deepcopy(json));return super().post(url,json=json)

def main():
    _,pr,plan=load_plan('knowledge_e0r_p1_001');config=read(pr/'llm_config.json');model=read(M5)
    binding=read(pr/'bindings.json');index=read(pr/'case_index.json');ref=next(iter(index));case=load_cases(DATA)[index[ref]-1]
    source_map,sr=memory_source_map([case]);now=datetime.now(timezone.utc)
    auth={**MARKERS,'scope':plan['scope'],'plan_fingerprint':plan['fingerprint'],'conditions':['B2'],
        'operator_approved':True,'no_other_ollama_work':True,'accept_source_label_proxy':True,'accept_common_m5_menu':True,
        'accept_recorded_time_proxy':True,'clinical_release_approved':False,'operator_ref':'TEST_ONLY_NOT_AUTHORITY',
        'user_statement':'TEST_ONLY no clinical authority or model execution',
        'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    out=ROOT/'outputs/private/research_evaluation'/NS/'TEST_ONLY_CORE';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'memory_sources.json',source_map);atomic_json(out/'source_verification.json',sr)
    rows=[];requests={};profile=[('existing','legacy_top5','none'),('C0','legacy_top5','none'),
        ('C1','historical_display','none'),('C2','diagnostic_retrieval','none'),('C3','legacy_top5','common_pack'),
        ('C4','legacy_top5','adaptive_bm25'),('rollback','legacy_top5','none')]
    for name,mm,km in profile:
        fake=CapturingFake(config);token=_BUDGET.set(GenerationBudget(12,12))
        try:
            with ExitStack() as s:
                s.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN')))
                s.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN')))
                s.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
                kwargs=dict(config=config,bindings=binding,model=model,authorization=auth,plan_fingerprint=plan['fingerprint'],
                    checkpoint_root=out/name,experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                factory=make_common_backend if name=='existing' else make_memory_rag_backend
                if name!='existing':kwargs.update(memory_mode=mm,knowledge_mode=km,source_map=source_map,corpus=corpus())
                result=factory({ref:case},**kwargs)({'case_ref':ref},features_for('B2'))
            if result['status']!='completed':raise AssertionError(name+' CORE_INCOMPLETE')
            for p in fake.payloads:
                assert 'external_medical_knowledge' not in p['observation']
                assert all('external_medical_knowledge' not in str(e) for e in p['evidence'])
                if km!='none':assert p['external_medical_knowledge']['scope']=='EXTERNAL_KNOWLEDGE_NOT_PATIENT_EVIDENCE'
            requests[name]=fake.requests
            row={'profile':name,'status':'PASS','core_status':result['status'],'fake_calls':len(fake.requests),
                'actual_LLM_generation':0,'request_fingerprints':[digest(r) for r in fake.requests]}
            if name=='C2':
                a=read(out/name/'Episode0001/memory_rag_audit.json');assert a['memory_retrievals']
                assert all(t['candidate_scope']=='ALL_ELIGIBLE_FIXED_PAST_RECORDS' for t in a['memory_retrievals'])
                row['selected_counts']=[len(t['selected']) for t in a['memory_retrievals']]
            if km!='none':row['knowledge_packet_bytes']=[len(json.dumps(p['external_medical_knowledge'],ensure_ascii=False).encode()) for p in fake.payloads]
            rows.append(row)
        finally:_BUDGET.reset(token)
    assert requests['existing']==requests['C0']==requests['rollback'],'LEGACY_WIRE_NOT_IDENTICAL'
    report={'TEST_ONLY':True,'profiles':rows,'legacy_and_rollback_full_request_exact_match':True,
        'legacy_preserves_prompt_schema_payload_options_and_order':True,'actual_LLM_generation':0,'utility_fitting':0,
        'core_source_scheduler_exporter':'ACTUAL_EXECUTION_WITH_FAKE_TRANSPORT','clinical_performance':'NOT_EVALUATED'}
    atomic_json(ROOT/'outputs/research_evaluation'/NS/'TEST_ONLY_core_report.json',report)
    print(json.dumps(report))
if __name__=='__main__':main()
