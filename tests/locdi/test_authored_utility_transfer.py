"""TEST_ONLY transfer authority, immutable fit and no-generation contract tests."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
import tempfile,unittest
from app.locdi.authored_utility_transfer import verify,make_transfer_backend,MARKERS
from scripts.run_b34_transfer_smoke import permission,paths


class TransferContract(unittest.TestCase):
    def test_changed_hash_rejected_before_any_source_or_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.json';p.write_text('{}')
            with patch('app.locdi.authored_utility_transfer.check') as source:
                with self.assertRaisesRegex(ValueError,'SHA_MISMATCH'):verify(p,{},[],{})
                source.assert_not_called()
    def test_actual_transfer_approval_required(self):
        a={**MARKERS,'clinical_release_approved':False,'accept_authored_utility_transfer':False}
        with self.assertRaisesRegex(ValueError,'APPROVAL_REQUIRED'):make_transfer_backend({},authorization=a,experiment_kind='DEVELOPMENT_EXPERIMENT')
    def test_clinical_mode_rejected(self):
        with self.assertRaisesRegex(ValueError,'NOT_CLINICAL'):make_transfer_backend({},authorization={**MARKERS,'clinical_release_approved':False},experiment_kind='CLINICAL_EVALUATION')
    def test_false_clinical_promotion_rejected(self):
        for key,value in [('clinical_release_approved',True),('independent_clinical_review',True),('clinical_accuracy','VALIDATED')]:
            a={**MARKERS,'clinical_release_approved':False,key:value}
            with self.assertRaisesRegex(ValueError,'NOT_CLINICAL'):make_transfer_backend({},authorization=a,experiment_kind='DEVELOPMENT_EXPERIMENT')
    def plan_and_approval(self):
        p={'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY','conditions':['B3'],
           'generation_total':24,'fingerprint':'a'*64,'experiment_scope':'TEST_ONLY_AUTHORED_TRANSFER',**MARKERS}
        now=datetime.now(timezone.utc);a={**p,'plan_fingerprint':p['fingerprint'],'max_generation':24,'dataset_views':['D0','D2'],
           'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat(),'operator_ref':'TEST_ONLY','user_statement':'TEST_ONLY_NOT_ACTUAL_AUTHORITY','clinical_release_approved':False}
        for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy','accept_authored_utility_transfer'):a[k]=True
        return p,a
    def test_exact_new_operator_scope(self):
        p,a=self.plan_and_approval();permission(a,p)
    def test_old_approval_not_usable(self):
        p,a=self.plan_and_approval();a.pop('accept_authored_utility_transfer')
        with self.assertRaises(ValueError):permission(a,p)
    def test_other_stage_approval_not_usable(self):
        p,a=self.plan_and_approval()
        with self.assertRaises(ValueError):permission(a,{**p,'conditions':['B3','B4'],'fingerprint':'b'*64})
    def test_unsafe_plan_name_rejected(self):
        with self.assertRaises(ValueError):paths('../TEST_ONLY')
    def test_abstention_not_forced_to_specialty(self):
        # The adapter supplies no routing scores, fallback department or labels.
        from app.core.agent_registry import AgentRegistry
        from scripts.run_b34_transfer_smoke import load
        self.assertNotIn('fit_utility',verify.__code__.co_names)
        self.assertNotIn('select_specialty',make_transfer_backend.__code__.co_names)

if __name__=='__main__':unittest.main()
