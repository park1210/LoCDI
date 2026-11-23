"""TEST_ONLY policy probes; stored clinical answers are never relabeled."""
from copy import deepcopy
import unittest
from app.locdi.reference_feasibility import (fraction, mapping_guard, exam_labels_usable,
    inventory, compatibility, experiments, schedule, hypotheses, collection_options, analyze_responses)
from app.locdi.llm_setup import digest
from test_unresolved_analysis import test_only_inputs


def fixture():
    bundle, refs, records, _, prompts = test_only_inputs()
    sources = [{'case_id':c['decision']['source_ref'], 'patient_id':c['decision']['patient_id'],
        'index_encounter':{'encounter_id':c['decision']['visit_id'], 'start_time':c['visit_started_at'],
                           'encounter_class':'TEST_ONLY_CLASS'}} for c in bundle['candidates']]
    return bundle, refs, records, sources, prompts


class ReferenceFeasibilityTests(unittest.TestCase):
    def test_zero_denominator_is_not_zero_percent(self):
        self.assertEqual(fraction(0,0)['status'],'NOT_APPLICABLE')
        self.assertIsNone(fraction(0,0)['value'])

    def test_descriptive_fraction(self):
        self.assertEqual(fraction(52,53)['value'],52/53)

    def test_code_is_only_candidate(self):
        self.assertEqual(mapping_guard('exact_code')['status'],'CANDIDATE_ONLY')

    def test_normalization_is_only_candidate(self):
        self.assertIsNone(mapping_guard('normalized_terminology')['clinical_equivalence'])

    def test_ontology_is_not_gold(self):
        self.assertFalse(mapping_guard('ontology_candidate')['clinical_gold_approved'])

    def test_clinical_equivalence_requires_review(self):
        self.assertEqual(mapping_guard('clinical_semantic_equivalence')['status'],'REVIEW_REQUIRED')

    def test_timing_requires_review(self):
        self.assertEqual(mapping_guard('current_decision_timing')['status'],'REVIEW_REQUIRED')

    def test_independent_promotion_blocked(self):
        r=mapping_guard('exact_code','INDEPENDENT_CLINICAL_REFERENCE')
        self.assertEqual(r['status'],'REVIEW_REQUIRED'); self.assertIsNone(r['correctness'])

    def test_unknown_mapping_rejected(self):
        with self.assertRaises(ValueError):mapping_guard('guess')

    def test_mcqa_test_never_approved_from_placeholder(self):
        self.assertEqual(exam_labels_usable('MedMCQA','test','DOCUMENTED_PUBLIC')['status'],'BLOCKED')

    def test_mcqa_train_documented_not_clinical(self):
        r=exam_labels_usable('MedMCQA','train','DOCUMENTED_PUBLIC')
        self.assertEqual(r['status'],'DOCUMENTED_LABEL_CANDIDATE'); self.assertFalse(r['clinical_gold_approved'])

    def test_no_new_dataset_acquisition_or_approval(self):
        rows=inventory()['sources']
        self.assertEqual(len(rows),7)
        self.assertTrue(all(not r['downloaded_this_phase'] and not r['project_use_approved'] for r in rows))

    def test_external_tasks_cannot_transfer_correctness(self):
        self.assertTrue(all(not r['existing75_correctness_transfer_allowed'] for r in compatibility(inventory())['rows']))

    def test_healthbench_not_internal_clinical_metric(self):
        r=next(r for r in compatibility(inventory())['rows'] if r['dataset']=='HealthBench')
        self.assertEqual(set(r['functions'].values()),{'NOT_EVALUABLE'})

    def test_compatibility_has_six_axes(self):
        self.assertTrue(all(len(r['functions'])==6 for r in compatibility(inventory())['rows']))

    def test_experiment_budget_not_execution(self):
        e=experiments()
        self.assertEqual(e['illustrative_E2_E5_generation_total'],405)
        self.assertTrue(all(not r['budget']['execution_approved'] for r in e['experiments']))

    def test_only_e1_completed_no_calls(self):
        rows=experiments()['experiments']
        self.assertEqual(rows[0]['status'],'COMPLETED_READ_ONLY')
        self.assertTrue(all(r['actual_generation_calls']==0 for r in rows))
        self.assertTrue(all(r['status']=='PROPOSED_NOT_APPROVED' for r in rows[1:]))

    def test_clinical_deadlines_not_ready(self):
        rows=schedule()['milestones']
        self.assertTrue(all(not r['new_ready_claim'] for r in rows))
        self.assertEqual(rows[3]['decision'],'NO_GO_CURRENTLY')

    def test_collection_does_not_approve_policy(self):
        self.assertFalse(collection_options()['policy_approved'])

    def test_analysis_is_immutable_and_private_separated(self):
        args=fixture(); before=digest(args)
        public,matrix,private=analyze_responses(*args)
        self.assertEqual(digest(args),before)
        self.assertEqual(public['response_count'],75); self.assertEqual(len(private),75)
        self.assertNotIn('patient_id',public); self.assertIsNone(matrix['clinical_correctness'])

    def test_literal_and_named_unresolved_distinct(self):
        args=list(fixture())
        row=next(r for r in args[2] if r['response']['certainty']=='unresolved')
        row['response']['diagnosis']='TEST_ONLY_A';row['response_fingerprint']=digest(row['response'])
        p,_,_=analyze_responses(*args)
        self.assertEqual(p['named_unresolved'],1)
        self.assertEqual(p['unresolved'],p['named_unresolved']+p['literal_unresolved'])

    def test_correct_false_cannot_replace_unreviewed(self):
        args=fixture();args[2][0]['correct']=False
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_BASELINE_CHANGED'):analyze_responses(*args)

    def test_correct_true_cannot_replace_unreviewed(self):
        args=fixture();args[2][0]['correct']=True
        with self.assertRaises(ValueError):analyze_responses(*args)

    def test_missing_agent_rejected(self):
        args=fixture();args[2].pop()
        with self.assertRaises(ValueError):analyze_responses(*args)

    def test_duplicate_agent_rejected(self):
        args=fixture();args[2][1]['agent_id']=args[2][0]['agent_id']
        with self.assertRaises(ValueError):analyze_responses(*args)

    def test_source_identity_mismatch_rejected(self):
        args=fixture();args[3][0]['patient_id']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'SOURCE_IDENTITY_MISMATCH'):analyze_responses(*args)

    def test_response_hash_tampering_rejected(self):
        args=fixture();args[2][0]['response_fingerprint']='0'*64
        with self.assertRaisesRegex(ValueError,'ORIGINAL_RESPONSE_HASH_MISMATCH'):analyze_responses(*args)

    def test_common_input_mismatch_rejected(self):
        args=fixture();args[2][0]['shared_input_fingerprint']='0'*64
        with self.assertRaisesRegex(ValueError,'COLLECTION_CONTRACT_MISMATCH'):analyze_responses(*args)

    def test_schema_invalid_rejected(self):
        args=fixture();r=args[2][0];r['response']['extra']='TEST_ONLY';r['response_fingerprint']=digest(r['response'])
        with self.assertRaisesRegex(ValueError,'COLLECTION_CONTRACT_MISMATCH'):analyze_responses(*args)

    def test_all_hypotheses_unproven_literal_absence_contradicted(self):
        p,m,_=analyze_responses(*fixture()); rows=hypotheses(p,m)['hypotheses']
        self.assertTrue(all(r['final_status']=='NOT_DETERMINED' for r in rows))
        self.assertEqual(rows[3]['observed_evidence']['literal_missing_target_claim'],'CONTRADICTED_BY_OBSERVATION')


if __name__=='__main__':unittest.main()
