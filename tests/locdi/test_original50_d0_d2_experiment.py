"""TEST_ONLY paired plans, no model, fitting, fake clinical reference or scoring."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from scripts import run_original50_d0_d2 as runner
from scripts.analyze_original50_d0_d2 import transitions
from app.locdi.augmentation_experiment_backend import make_paired_backend
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import digest
from app.locdi.development_routing_proxy import MARKERS

class ExecutionContract(unittest.TestCase):
    def setUp(self):
        self.plan={'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY',
            'fingerprint':'a'*64,'conditions':runner.CONDITIONS,'generation_total':120,'experiment_kind':'DEVELOPMENT_EXPERIMENT'}
        now=datetime.now(timezone.utc)
        self.a={**MARKERS,**{k:self.plan[k] for k in ('scope','paired_scope','conditions')},'plan_fingerprint':'a'*64,'dataset_views':['D0','D2'],
            'max_generation':120,'operator_approved':True,'no_other_ollama_work':True,'clinical_release_approved':False,
            'accept_source_label_proxy':True,'accept_recorded_time_proxy':True,'accept_common_m5_menu':True,
            'accept_label_conditioned_development':True,'accept_augmentation_initial_time_policy':True,
            'operator_ref':'TEST_ONLY','user_statement':'TEST_ONLY_NOT_ACTUAL_AUTHORITY','recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
    def test_exact_authority_for_new_scope(self):runner.permission(self.a,self.plan)
    def test_missing_new_proxy_permission_rejects(self):
        for key in ('accept_label_conditioned_development','accept_augmentation_initial_time_policy'):
            a={**self.a,key:False}
            with self.assertRaises(ValueError):runner.permission(a,self.plan)
    def test_clinical_promotion_rejected(self):
        self.a['clinical_release_approved']=True
        with self.assertRaises(ValueError):runner.permission(self.a,self.plan)
    def test_different_views_budget_or_fingerprint_rejected(self):
        for k,v in [('dataset_views',['D2']),('max_generation',121),('plan_fingerprint','b'*64)]:
            with self.assertRaises(ValueError):runner.permission({**self.a,k:v},self.plan)
    def test_expired_approval_rejected(self):
        self.a['expires_at']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        with self.assertRaises(ValueError):runner.permission(self.a,self.plan)
    def test_task_counts_fixed_pairing_and_counterbalance(self):
        rr=runner.tasks([{'case_ref':'TEST_ONLY_%d'%n} for n in range(10)])
        self.assertEqual(len(rr),100);self.assertEqual([r['dataset_view'] for r in rr[:2]],['D0','D2'])
        self.assertEqual([r['dataset_view'] for r in rr[10:12]],['D2','D0'])
        for view in runner.VIEWS:self.assertEqual([r['view_episode_ordinal'] for r in rr if r['dataset_view']==view],list(range(1,51)))
    def test_missing_smoke_cannot_prepare_full_or_create_directories(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'PUB',Path(d)/'pub'),patch.object(runner,'PR',Path(d)/'pr'),patch.object(runner,'smoke_proof',side_effect=ValueError('SMOKE_NOT_EXECUTED')),patch.object(runner,'base_prepare') as base:
            with self.assertRaisesRegex(ValueError,'SMOKE_NOT_EXECUTED'):runner.prepare('TEST_ONLY',stage='Full')
            base.assert_not_called();self.assertFalse((Path(d)/'pr').exists())
    def test_old_plan_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'PUB',Path(d)),patch.object(runner,'PR',Path(d)):
            (Path(d)/'TEST_ONLY').mkdir()
            with self.assertRaises(FileExistsError):runner.prepare('TEST_ONLY')
    def test_reanalysis_same_output_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';runner.write_equal(p,{'TEST_ONLY':True});b=p.read_bytes();runner.write_equal(p,{'TEST_ONLY':True});self.assertEqual(b,p.read_bytes())
            with self.assertRaises(ValueError):runner.write_equal(p,{'TEST_ONLY':False})
    def test_smoke_approval_not_usable_for_full_budget(self):
        with self.assertRaises(ValueError):runner.permission(self.a,{**self.plan,'generation_total':1200,'fingerprint':'b'*64})
    def test_test_only_collection_cannot_promote_full(self):
        with tempfile.TemporaryDirectory() as d:
            pr=Path(d);run=pr/'RUN';run.mkdir();atomic_json(run/'collection.json',{'test_only':True,'plan_fingerprint':'a'*64,'status':'COMPLETED','rows':[{}]*10})
            with patch.object(runner,'load',return_value=(pr,pr,{'stage':'Smoke','run_id':'RUN','fingerprint':'a'*64},pr)),patch.object(runner,'read',side_effect=lambda p:__import__('json').loads(Path(p).read_text()) if Path(p).exists() else []):
                with self.assertRaisesRegex(ValueError,'REAL_COMPLETE_SMOKE'):runner.smoke_proof('TEST_ONLY')
    def test_invalid_view_never_constructs_backend(self):
        with patch('app.locdi.augmentation_experiment_backend.make_common_backend') as base:
            with self.assertRaises(ValueError):make_paired_backend({},dataset_view='D1',augmented_rows={},catalog={},authorization={})
            base.assert_not_called()
    def test_bare_and_prefixed_saved_model_digests_have_same_identity(self):
        from app.locdi.llm_setup import normalize_model_identifier
        h='a'*64;self.assertEqual(normalize_model_identifier(h),normalize_model_identifier('sha256:'+h))

class MetricsContracts(unittest.TestCase):
    def f(self,exact,certainty='supported'):return {'source_exact':exact,'prediction':{'certainty':certainty}}
    def test_correct_loss_and_retention(self):
        self.assertTrue(transitions(self.f(True),self.f(False))['CORRECT_TO_INCORRECT'])
        self.assertTrue(transitions(self.f(True),self.f(True))['CORRECT_RETAINED'])
    def test_gain(self):self.assertTrue(transitions(self.f(False),self.f(True))['INCORRECT_TO_CORRECT'])
    def test_unresolved_never_made_false_correctness(self):
        t=transitions(self.f(True),self.f(None,'unresolved'))
        self.assertFalse(t['CORRECT_TO_INCORRECT']);self.assertTrue(t['SUPPORTED_TO_UNRESOLVED']);self.assertTrue(t['unscored_or_not_completed_present'])
    def test_patient_ids_not_accepted_in_plan_path(self):
        with self.assertRaises(ValueError):runner.paths('../TEST_ONLY')
if __name__=='__main__':unittest.main()
