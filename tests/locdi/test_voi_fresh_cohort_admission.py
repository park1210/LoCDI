"""TEST_ONLY two approvals, no fresh cohort evaluation."""
import unittest
from unittest.mock import patch
from app.locdi.voi_fresh_cohort_admission import *


class AdmissionContracts(unittest.TestCase):
    def setUp(self):
        self.proposal={'global_blockers':[],'proposed':[{'patient_id':'TEST_P','visit_id':'TEST_E'}]}
        self.contract={'proposal_fingerprint':digest(self.proposal)};self.contract['fingerprint']=digest(self.contract)
        self.approval={'approved':True,'scope':ADMISSION_SCOPE,'plan_fingerprint':self.contract['fingerprint'],
            'evidence_type':'USER_EXPLICIT','user_statement':'TEST_ONLY','recorded_at':'TEST_ONLY',
            'bounded_export_history_review_accepted':True}

    def test_no_common_pattern_auto_claim(self):self.assertEqual(classify_collision(),'UNRESOLVED_CONTENT_COLLISION')
    def test_full_history_match_suspected(self):self.assertEqual(classify_collision(full_history_match=True),'PATIENT_REUSE_SUSPECTED')
    def test_mapping_error(self):self.assertEqual(classify_collision(mapping_error=True),'SOURCE_MAPPING_ERROR')
    def test_provenance_error(self):self.assertEqual(classify_collision(provenance_conflict=True),'PROVENANCE_CONFLICT')
    def test_unpinned_module_not_approved(self):self.assertEqual(classify_collision(module_evidence={'literal_condition_and_observation_match':True}),'UNRESOLVED_CONTENT_COLLISION')
    def test_verified_module_support(self):
        e={k:True for k in ['pinned_generator_module_verified','literal_condition_and_observation_match','fixed_value_and_unit_match']}
        self.assertEqual(classify_collision(module_evidence=e),'COMMON_PATTERN_SUPPORTED')
    def test_missing_admission_approval(self):
        with self.assertRaises(ValueError):approve_admission(self.proposal,self.contract,None)
    def test_approval_A_not_B(self):
        a=approve_admission(self.proposal,self.contract,self.approval)
        self.assertFalse(a['policy_evaluation_approved'])
        with patch('app.locdi.voi_fresh_cohort_admission.execute_panel') as execute:
            with self.assertRaises(ValueError):run_approved(a,{}, {'fingerprint':'TEST_EVAL'},self.approval,lambda x:None)
            execute.assert_not_called()
    def test_export_scope_acceptance_required(self):
        self.approval.pop('bounded_export_history_review_accepted')
        with self.assertRaises(ValueError):approve_admission(self.proposal,self.contract,self.approval)
    def test_proposal_hash_changed(self):
        self.proposal['proposed'].append({'patient_id':'Q','visit_id':'X'})
        with self.assertRaises(ValueError):approve_admission(self.proposal,self.contract,self.approval)
    def test_global_failure_not_waived(self):
        self.proposal['global_blockers']=['RAW_CHANGED'];self.contract={'proposal_fingerprint':digest(self.proposal)};self.contract['fingerprint']=digest(self.contract)
        self.approval['plan_fingerprint']=self.contract['fingerprint']
        with self.assertRaises(ValueError):approve_admission(self.proposal,self.contract,self.approval)
    def test_admission_no_clinical_gold(self):
        a=approve_admission(self.proposal,self.contract,self.approval)
        self.assertFalse(a['clinical_reference_approved'])
