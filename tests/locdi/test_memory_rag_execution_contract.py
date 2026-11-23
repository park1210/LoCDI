"""TEST_ONLY approvals. Never instantiate a live model client."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile,unittest,json
from unittest.mock import patch
from scripts import run_memory_rag_experiment as runner
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.memory_rag_backend import make_memory_rag_backend
from app.locdi.current_ablation_runner import features_for

class ExecutionContract(unittest.TestCase):
    def setUp(self):
        self.plan={'scope':'TEST_ONLY','fingerprint':'a'*64,'arms':{'C0':{},'C2':{}},'generation_total':24}
        now=datetime.now(timezone.utc)
        self.approval={**MARKERS,'scope':'TEST_ONLY','plan_fingerprint':'a'*64,'conditions':['B2'],
            'experiment_arms':self.plan['arms'],'max_generation':24,'operator_approved':True,'no_other_ollama_work':True,
            'accept_source_label_proxy':True,'accept_common_m5_menu':True,'accept_recorded_time_proxy':True,
            'clinical_release_approved':False,'operator_ref':'TEST_ONLY_NOT_REAL_AUTHORITY','user_statement':'TEST_ONLY no actual execution',
            'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    def test_explicit_test_only_bounded_contract(self):runner.authority(self.approval,self.plan)
    def test_no_operator_approval_rejected(self):
        self.approval['operator_approved']=False
        with self.assertRaises(ValueError):runner.authority(self.approval,self.plan)
    def test_different_arm_and_budget_rejected(self):
        for k,v in [('experiment_arms',{'C0':{}}),('max_generation',25),('plan_fingerprint','b'*64)]:
            a={**self.approval,k:v}
            with self.assertRaises(ValueError):runner.authority(a,self.plan)
    def test_clinical_upgrade_rejected(self):
        self.approval['clinical_release_approved']=True
        with self.assertRaises(ValueError):runner.authority(self.approval,self.plan)
    def test_expired_and_longer_than_twentyfour_hours_rejected(self):
        now=datetime.now(timezone.utc)
        for start,end in [(now-timedelta(hours=2),now-timedelta(hours=1)),(now,now+timedelta(hours=25))]:
            a={**self.approval,'recorded_at':start.isoformat(),'expires_at':end.isoformat()}
            with self.assertRaises(ValueError):runner.authority(a,self.plan)
    def test_existing_plan_never_overwritten_or_base_created(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'PUBLIC',Path(d)),patch.object(runner,'PRIVATE',Path(d)),patch.object(runner,'prepare_base') as base:
            (Path(d)/'TEST_ONLY').mkdir()
            with self.assertRaises(FileExistsError):runner.prepare('TEST_ONLY')
            base.assert_not_called()
    def test_static_paths_reject_traversal(self):
        with self.assertRaises(ValueError):runner.paths('../TEST_ONLY')
    def test_invalid_modes_rejected_without_backend(self):
        with patch('app.locdi.memory_rag_backend.make_common_backend') as base:
            with self.assertRaises(ValueError):make_memory_rag_backend({},memory_mode='TEST_ONLY_BAD')
            base.assert_not_called()
    def test_default_factory_noninterference_all_b0_b4(self):
        def backend(case,features):return features
        with patch('app.locdi.memory_rag_backend.make_common_backend',return_value=backend):
            for c in ('B0','B1','B2','B3','B4'):
                self.assertEqual(make_memory_rag_backend({})({},features_for(c)),features_for(c))
    def test_arm_matrix_keeps_separate_factors(self):
        self.assertEqual(runner.STUDIES['memory'],['C0','C2']);self.assertEqual(runner.STUDIES['rag'],['C0','C4'])
        self.assertEqual(runner.ARMS['C5'],('diagnostic_retrieval','adaptive_bm25'))
    def analysis_fixture(self,folder):
        pub=Path(folder)/'public';pr=Path(folder)/'private';br=Path(folder)/'base';run=pr/'TEST_ONLY_RUN'
        for d in (pub,pr,br,run):d.mkdir(parents=True,exist_ok=True)
        (br/'panel.json').write_text(json.dumps([{'reference':{'accepted_diagnoses':['TEST_ONLY_DIAGNOSIS']}}]),encoding='utf-8')
        (run/'collection.json').write_text(json.dumps({'actual_generation':0,'rows':[{'patient_ordinal':1,'arm':'C0','status':'COMPLETED','request_count':0,
            'result':{'prediction':{'diagnosis':'TEST_ONLY_DIAGNOSIS','certainty':'supported'}}}]}),encoding='utf-8')
        return pub,pr,br,{'run_id':'TEST_ONLY_RUN','registered_patients':1}
    def test_saved_analysis_repeat_is_read_only_not_another_generation(self):
        with tempfile.TemporaryDirectory() as d:
            pub,pr,br,p=self.analysis_fixture(d)
            with patch.object(runner,'check',return_value=(p,br)),patch.object(runner,'paths',return_value=(pub,pr)):
                first=runner.analyze('TEST_ONLY');before=(pub/'saved_results.json').read_bytes()
                self.assertEqual(runner.analyze('TEST_ONLY'),first);self.assertEqual(before,(pub/'saved_results.json').read_bytes())
    def test_different_existing_analysis_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            pub,pr,br,p=self.analysis_fixture(d);target=pub/'saved_results.json';target.write_text('{}',encoding='utf-8')
            with patch.object(runner,'check',return_value=(p,br)),patch.object(runner,'paths',return_value=(pub,pr)):
                with self.assertRaisesRegex(ValueError,'NO_OVERWRITE'):runner.analyze('TEST_ONLY')
                self.assertEqual(target.read_text(encoding='utf-8'),'{}')
if __name__=='__main__':unittest.main()
