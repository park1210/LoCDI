"""TEST_ONLY process/scope fixtures; never actual clinical review results."""
import copy,json,unittest
from app.locdi.clinical_review_protocol import (presence,classify_case,scope_matrix,policy_register,rubrics,
    reference_protocol,check_transition,protocol_check,REFERENCE_OUTCOMES,PURPOSES,STATES,build)


def facts(dataset='Original',ordinal=1):
    return {'dataset':dataset,'ordinal':ordinal,'initial_presence':{k:'ABSENT' for k in
        ('chief_complaint','initial_vitals','objective_findings','history','current_symptoms')},
        'encounter_class':'TEST_ONLY','memory_count':5,'ask_count':1,'test_count':4,'consult_count':0,
        'reference_candidate_count':1,'actual_availability_count':0,'development_exposed':True}


def fixture_outputs():
    return {'evaluation_scope_matrix':scope_matrix(),'case_suitability_matrix':{'cases':
        [classify_case(facts('Original',i)) for i in range(1,51)]+[classify_case(facts('D3',i)) for i in range(1,30)],
        'clinical_reference_approved_count':0},'reference_review_protocol':reference_protocol(),'review_axis_rubrics':rubrics(),
        'reviewer_workload_plan':{},'policy_decision_register':policy_register(),
        'temporal_review_priorities':{'rows':[{'purpose':'TEST_ONLY'} for _ in range(4)]}}


def evidence():
    return {'purpose':'TEST_ONLY','review_axis':'Reference Validity','reviewer_kind':'HUMAN_CLINICAL_REVIEWER',
        'policy_binding':True,'source_snapshot_binding':True,'qualified_reviewer_evidence':True,
        'independence_evidence':True,'response_blinded':True,'test_only_authority_fixture':True}


class ClinicalScopeContracts(unittest.TestCase):
    def test_absent_and_explicit_null_distinct(self):self.assertEqual((presence({},'x'),presence({'x':None},'x')),('ABSENT','NULL'))
    def test_empty_nested_values_and_whitespace_not_initial_information(self):
        self.assertEqual(presence({'x':{'value':None,'note':' '}},'x'),'EMPTY')
    def test_zero_numeric_value_is_not_absence(self):self.assertEqual(presence({'x':0},'x'),'VALUE_PRESENT_UNREVIEWED')
    def test_empty_state_workflow_development_not_diagnosis_ready(self):
        r=classify_case(facts());a={x['purpose']:x for x in r['assessments']}
        self.assertIn('DEVELOPMENT_EVALUABLE',a[PURPOSES[0]]['states']);self.assertIn('DATA_INSUFFICIENT',a['Diagnosis Accuracy']['states'])
    def test_candidate_not_current_gold(self):
        r=classify_case(facts());self.assertTrue(r['reference_candidate_present']);self.assertFalse(r['reviewed_reference_available'])
    def test_sufficiency_judgment_never_generated(self):
        r=classify_case(facts());self.assertIsNone(r['diagnosis_or_disease_assessability_verdict']);self.assertTrue(all(x['information_sufficiency_verdict'] is None for x in r['assessments']))
    def test_clinical_input_presence_does_not_approve(self):
        f=facts();f['initial_presence']['chief_complaint']='VALUE_PRESENT_UNREVIEWED';r=classify_case(f)
        self.assertTrue(r['initial_clinical_fields_present']);self.assertFalse(any(x['clinical_evaluation_approved'] for x in r['assessments']))
    def test_no_memory_retrieval_reference_inferred(self):
        r=classify_case(facts());a=next(x for x in r['assessments'] if x['purpose']==PURPOSES[1]);self.assertIn('REFERENCE_UNAVAILABLE',a['states'])
    def test_proxy_never_actual_timing(self):
        a=next(x for x in classify_case(facts())['assessments'] if x['purpose']==PURPOSES[6]);self.assertIn('SOURCE_TIMING_UNVERIFIED',a['states'])
    def test_public_case_has_no_source_identity(self):
        f=facts();f.update(patient_id='TEST_ONLY_PRIVATE',visit_id='TEST_ONLY_VISIT');r=classify_case(f);self.assertNotIn('TEST_ONLY_PRIVATE',json.dumps(r))
    def test_no_case_input_mutation(self):
        f=facts();b=copy.deepcopy(f);classify_case(f);self.assertEqual(f,b)
    def test_all_seven_scope_specs_and_denominators(self):
        r=scope_matrix();self.assertEqual({x['purpose'] for x in r['rows']},set(PURPOSES));self.assertTrue(all(x['denominator_and_exclusions'] for x in r['rows']))
    def test_four_policies_remain_unapproved(self):
        r=policy_register();self.assertEqual(len(r['decisions']),4);self.assertEqual(r['approved_policy_count'],0);self.assertTrue(all(x['status']=='POLICY_APPROVAL_REQUIRED' for x in r['decisions']))
    def test_five_axes_not_one_boolean(self):self.assertEqual(len(rubrics()['axes']),5)
    def test_current_retrospective_rejected_insufficient_not_reviewable_separate(self):
        self.assertEqual(set(reference_protocol()['permitted_human_outcomes']),set(REFERENCE_OUTCOMES))
    def test_no_real_reviewer_or_authentication_generated(self):
        self.assertEqual(rubrics()['actual_reviewers_assigned'],0);self.assertEqual(reference_protocol()['authority_verifier'],'NOT_IMPLEMENTED')
    def test_missing_protocol_output_blocks(self):
        r=fixture_outputs();r.pop('review_axis_rubrics');self.assertEqual(protocol_check(r)['protocol_contract_status'],'BLOCKED')
    def test_valid_draft_protocol_only(self):
        r=protocol_check(fixture_outputs());self.assertEqual(r['protocol_contract_status'],'PASS');self.assertFalse(r['CLINICAL_REFERENCE_READY'])
    def test_arbitrary_policy_approved_flag_blocked(self):
        r=fixture_outputs();r['policy_decision_register']['approved_policy_count']=1;self.assertIn('UNAPPROVED_POLICY_PROMOTION',protocol_check(r)['blocker_codes'])
    def test_arbitrary_clinical_gold_flag_blocked(self):
        r=fixture_outputs();r['case_suitability_matrix']['clinical_reference_approved_count']=1;self.assertEqual(protocol_check(r)['protocol_contract_status'],'BLOCKED')
    def test_duplicate_case_cannot_fake_coverage(self):
        r=fixture_outputs();r['case_suitability_matrix']['cases'][0]['ordinal']=2;self.assertIn('CASE_CLASSIFICATION_INVALID',protocol_check(r)['blocker_codes'])
    def test_missing_minimum_spec_blocked(self):
        r=fixture_outputs();del r['evaluation_scope_matrix']['rows'][0]['denominator_and_exclusions'];self.assertEqual(protocol_check(r)['protocol_contract_status'],'BLOCKED')
    def test_malformed_nested_input_fails_closed(self):
        r=fixture_outputs();r['case_suitability_matrix']['cases']=[{}];self.assertEqual(protocol_check(r)['protocol_contract_status'],'BLOCKED')
    def test_stage_cannot_skip_to_approved(self):self.assertFalse(check_transition('DRAFT','APPROVED',evidence(),test_only=True)['transition_contract_allowed'])
    def test_submitted_requires_evidence(self):self.assertFalse(check_transition('REVIEW_REQUESTED','SUBMITTED',{})['transition_contract_allowed'])
    def test_reference_review_requires_blindness(self):
        e=evidence();e['response_blinded']=False;self.assertFalse(check_transition('REVIEW_REQUESTED','SUBMITTED',e)['transition_contract_allowed'])
    def test_reference_response_correctness_not_merged(self):
        e=evidence();e['diagnosis_correctness_label']=True;self.assertFalse(check_transition('REVIEW_REQUESTED','SUBMITTED',e)['transition_contract_allowed'])
    def test_response_review_after_authenticated_reference(self):
        e=evidence();e['review_axis']='Response Correctness';self.assertFalse(check_transition('REVIEW_REQUESTED','SUBMITTED',e)['transition_contract_allowed'])
    def test_llm_or_self_report_cannot_be_human_approval(self):
        e=evidence();e['reviewer_kind']='LLM';self.assertFalse(check_transition('REVIEW_REQUESTED','SUBMITTED',e)['transition_contract_allowed'])
    def test_unresolved_never_false_or_true(self):
        for label in (False,True):
            e=evidence();e.update(certainty='unresolved',diagnosis_correctness_label=label)
            self.assertIn('UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN',check_transition('SUBMITTED','APPROVAL_PENDING',e,test_only=True)['blocker_codes'])
    def test_production_approval_blocked_even_with_all_declarations(self):
        r=check_transition('APPROVAL_PENDING','APPROVED',evidence());self.assertFalse(r['transition_contract_allowed']);self.assertFalse(r['clinical_review_ready'])
    def test_test_only_transition_does_not_approve_clinical_review(self):
        r=check_transition('APPROVAL_PENDING','APPROVED',evidence(),test_only=True);self.assertTrue(r['transition_contract_allowed']);self.assertFalse(r['clinical_review_ready'])
    def test_test_only_flag_without_fixture_not_authority(self):
        e=evidence();e['purpose']='development';self.assertFalse(check_transition('APPROVAL_PENDING','APPROVED',e,test_only=True)['transition_contract_allowed'])
    def test_unresolved_disagreement_blocks_approval(self):
        e=evidence();e['disagreement_unresolved']=True;self.assertFalse(check_transition('APPROVAL_PENDING','APPROVED',e,test_only=True)['transition_contract_allowed'])
    def test_proxy_strict_claim_blocked(self):
        e=evidence();e.update(source_time_basis='PROXY',strict_clinical_timing_claim=True);self.assertFalse(check_transition('SUBMITTED','APPROVAL_PENDING',e,test_only=True)['transition_contract_allowed'])
    def test_no_automatic_gate_adapter(self):self.assertFalse(reference_protocol()['automatic_conversion'])
    def test_actual_79_case_rows_and_four_priority_targets(self):
        outputs,links,ev=build();self.assertEqual(protocol_check(outputs)['protocol_contract_status'],'PASS')
        self.assertEqual(len(outputs['case_suitability_matrix']['cases']),79);self.assertEqual(len(links['case_links']),79)
        targets={r['target_ref'] for r in outputs['temporal_review_priorities']['rows']}
        self.assertEqual(targets,{'Original022','Original026','D3019','RoutingCandidate010'})
        self.assertEqual(outputs['reviewer_workload_plan']['proposed_reference_decision_units_if_all_included'],94)
        self.assertTrue(all(r['human_verdict'] is None for r in outputs['temporal_review_priorities']['rows']))


if __name__=='__main__':unittest.main()
