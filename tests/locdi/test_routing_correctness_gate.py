"""TEST_ONLY human-review declarations, not clinical judgments or training data."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from test_specialty_collection_analysis import fixture
from app.locdi.llm_setup import digest
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from validation.routing_correctness_gate import (POLICY, empty_review_results,
                                                validate_evaluation_gate, to_policy_fit_rows)
from scripts.check_routing_correctness import main


def test_only_reviewed_data():
    bundle, pairs, _ = fixture()
    data = {'schema_version':'locdi.routing_collection.v1', 'purpose':'TEST_ONLY', 'registry':bundle['registry'],
        'split_manifest':bundle['split_manifest'], 'decisions':[c['decision'] for c in bundle['candidates']],
        'outcomes':[r for _,r in pairs], 'judgments':[]}
    reviews = empty_review_results('TEST_ONLY'); sources, labels, contexts = [], [], []
    reference_review = {'status':'REVIEWED', 'reviewer_ref':'TEST_ONLY_REFERENCE_HUMAN',
        'reviewer_kind':'HUMAN_CLINICAL_REVIEWER', 'reviewer_qualification':'TEST_ONLY', 'independent':True,
        'procedure_version':'locdi.reference_review.v1', 'reviewed_at':'2026-10-29T00:00:00+00:00'}
    response_review = {**reference_review, 'reviewer_ref':'TEST_ONLY_RESPONSE_HUMAN',
        'procedure_version':'locdi.response_review.v1', 'reviewed_at':'2026-10-30T00:00:00+00:00'}
    for c in bundle['candidates']:
        d = c['decision']; case = c['case_ref']
        candidate = {'source_condition_rows':[{'source_record':{'PATIENT':d['patient_id'], 'ENCOUNTER':d['visit_id'], 'START':'2026-01-01'}}]}
        ctx = {'case_ref':case, **{k:d[k] for k in ('patient_id','visit_id','decision_id')},
            'source_fingerprint':'4'*64, 'encounter_started_at':'2026-01-01T00:00:00+00:00',
            'visibility':deepcopy(c['visibility']), 'reference_candidates':candidate, 'diagnosis_catalog':['TEST_ONLY_A'], 'original_response_fingerprints':{}}
        contexts.append(ctx)
        content = {'diagnosis':'TEST_ONLY_A', 'evidence_ref':'TEST_ONLY_INDEPENDENT_REFERENCE',
                   'available_at':'2026-01-01T00:00:30+00:00', 'time_scope':'CURRENT_DECISION'}
        label_ref = 'TEST_ONLY_LABEL_'+case; fp = digest(content)
        labels.append({'label_ref':label_ref, 'purpose':'TEST_ONLY', 'source_ref':'TEST_ONLY_LABEL_SOURCE_'+case,
            'source_fingerprint':'5'*64, 'patient_id':d['patient_id'], 'decision_id':d['decision_id'],
            'label_fingerprint':fp, 'reference_content':content, 'independent_reference':True,
            'clinical_review_status':'REVIEWED', 'evaluation_method':'blinded_clinical_review', 'review':deepcopy(reference_review)})
        sources.append({'source_ref':d['source_ref'], 'purpose':'TEST_ONLY',
            **{k:deepcopy(d[k]) for k in ('patient_id','visit_id','decision_id','minute','observation')},
            'source_fingerprint':'4'*64, 'visit_started_at':ctx['encounter_started_at'],
            'visibility':{p:r['available_minute'] for p,r in ctx['visibility'].items()}, 'review':deepcopy(reference_review)})
        reviews['reference_reviews'].append({'case_ref':case, **{k:d[k] for k in ('patient_id','visit_id','decision_id','source_ref')},
            'source_fingerprint':'4'*64, 'observation_fingerprint':digest(d['observation']),
            'memory_fingerprint':digest(d['observation']['longitudinal_memory']),
            'condition_candidates_fingerprint':digest(candidate), 'reference_ref':label_ref, 'reference_fingerprint':fp,
            'timing_status':'CURRENT_DECISION_REFERENCE_VALID', 'reference_basis':'SOURCE_CONDITION', 'response_blinded':True,
            'input_scope':{'observation_reviewed':True,'memory_reviewed':True,'visibility_reviewed':True,
                           'proxy_reviewed':None,'proxy_fitting_approved':None,'proxy_rationale':None},
            'review':deepcopy(reference_review), 'clinical_evidence_rationale':'TEST_ONLY_SOURCE_REVIEW', 'uncertainty_reason':None})
        for i,row in enumerate(r for r in data['outcomes'] if r['case_ref']==case):
            answer = {'action':'diagnose','diagnosis':'TEST_ONLY_A','certainty':'supported','rationale':'TEST_ONLY_RESPONSE'}
            row.update(response=answer, response_fingerprint=digest(answer), purpose='TEST_ONLY',
                correctness_status='NOT_REVIEWED', recorded_at='2026-10-28T00:00:00+00:00')
            ctx['original_response_fingerprints'][row['response_ref']]=row['response_fingerprint']
            axes = {k:{'policy_id':v['policy_id'],'status':'NOT_REVIEWED','verdict':None,
                       'clinical_evidence_rationale':None,'not_evaluable_reason':None} for k,v in POLICY['policies'].items()}
            # Deliberate clinical-free TEST_ONLY booleans; not derived from text.
            axes['diagnosis_correctness'].update(status='REVIEWED',verdict=bool(i % 2),clinical_evidence_rationale='TEST_ONLY_VERDICT')
            reviews['response_reviews'].append({'case_ref':case, **{k:row[k] for k in ('patient_id','decision_id','agent_id','response_ref','response_fingerprint')},
                'reference_ref':label_ref,'reference_fingerprint':fp,'review':deepcopy(response_review),'axes':axes})
            data['judgments'].append({'response_ref':row['response_ref'],'response_fingerprint':row['response_fingerprint'],
                **{k:row[k] for k in ('patient_id','decision_id','agent_id')}, 'label_ref':label_ref,'label_fingerprint':fp,
                'purpose':'TEST_ONLY','status':'REVIEWED','correct':bool(i%2),'evaluation_method':'blinded_clinical_review',
                'review':deepcopy(response_review)})
    opts = dict(review_results=reviews,decision_contexts=contexts,registry_manifest=bundle['registry'],source_catalog=sources,
        label_catalog=labels,holdout_patient_ids=bundle['split_manifest']['patients']['test'],
        dataset_fingerprint=bundle['split_manifest']['dataset_fingerprint'],allow_test_only=True)
    return data,opts


class CorrectnessGateContracts(unittest.TestCase):
    def setUp(self):
        self.data,self.opts = test_only_reviewed_data()

    def validate(self):
        return validate_evaluation_gate(self.data,**self.opts)

    def blocked(self, code):
        r = self.validate()
        self.assertIn(code,r['blocker_codes'])
        self.assertFalse(r['utility_fit_ready'])
        return r

    def test_supported_review_calls_legacy_then_support_without_fit(self):
        before = digest([self.data,self.opts])
        with patch('app.core.routing_utility.fit_utility',side_effect=AssertionError('Fitting forbidden')) as fit:
            r = self.validate()
        fit.assert_not_called()
        self.assertEqual(r['blocker_codes'],[])
        self.assertEqual(r['legacy_validator']['training_readiness'],'TEST_ONLY_CONTRACT_VALID')
        self.assertEqual(r['complete_train_groups'],10)
        self.assertEqual(r['validation_group_count'],5)
        self.assertTrue(r['development_contract_ready'])
        self.assertFalse(r['utility_fit_ready'] or r['research_evaluation_ready'])
        self.assertEqual(r['positive_similarity_support']['queries'][0]['matched_samples_per_agent'],10)
        self.assertEqual(before,digest([self.data,self.opts]))
        self.assertIsNone(to_policy_fit_rows(self.data,**self.opts)['rows'])

    def test_unresolved_true_or_false_imputation_blocked_before_legacy(self):
        for value in (True,False):
            with self.subTest(value=value):
                self.data,self.opts = test_only_reviewed_data()
                row = self.data['outcomes'][0]; row['response']['certainty']='unresolved'
                row['response_fingerprint']=digest(row['response'])
                self.opts['decision_contexts'][0]['original_response_fingerprints'][row['response_ref']]=row['response_fingerprint']
                rr=self.opts['review_results']['response_reviews'][0]; rr['response_fingerprint']=row['response_fingerprint']
                rr['axes']['diagnosis_correctness']['verdict']=value
                self.data['judgments'][0].update(correct=value,response_fingerprint=row['response_fingerprint'])
                with patch('validation.routing_correctness_gate.validate_training_data') as legacy:
                    r=self.blocked('UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN')
                legacy.assert_not_called()
                self.assertEqual(r['policy_complete_train_groups'],9)
                self.assertEqual(r['complete_train_groups'],0)
                self.assertEqual(len(self.data['outcomes']),75)

    def test_independent_abstention_does_not_become_diagnosis_correctness(self):
        row=self.data['outcomes'][0];row['response']['certainty']='unresolved';row['response_fingerprint']=digest(row['response'])
        self.opts['decision_contexts'][0]['original_response_fingerprints'][row['response_ref']]=row['response_fingerprint']
        rr=self.opts['review_results']['response_reviews'][0];rr['response_fingerprint']=row['response_fingerprint']
        rr['axes']['diagnosis_correctness'].update(status='NOT_EVALUABLE',verdict=None,clinical_evidence_rationale=None,not_evaluable_reason='TEST_ONLY_UNRESOLVED')
        rr['axes']['abstention_appropriateness'].update(status='REVIEWED',verdict=True,clinical_evidence_rationale='TEST_ONLY_ABSTENTION')
        self.data['judgments'].pop(0)
        r=self.blocked('UNRESOLVED_EXCLUDED_FROM_DIAGNOSIS_FITTING')
        self.assertTrue(r['clinical_review_ready'])
        self.assertFalse(r['response_policy_ready'])

    def test_future_source_condition_current_reference_blocked(self):
        ctx=self.opts['decision_contexts'][9]
        ctx['reference_candidates']['source_condition_rows'][0]['source_record']['START']='2026-01-05'
        self.opts['review_results']['reference_reviews'][9]['condition_candidates_fingerprint']=digest(ctx['reference_candidates'])
        r=self.blocked('FUTURE_SOURCE_CONDITION_NOT_CURRENT_REFERENCE')
        self.assertEqual(r['groups'][9]['source_condition_start_after_decision_days'],4)

    def test_all_timing_states_remain_distinct(self):
        for state in ('RETROSPECTIVE_OUTCOME_ONLY','REFERENCE_TIMING_NOT_REVIEWED','REFERENCE_REJECTED'):
            self.data,self.opts=test_only_reviewed_data()
            self.opts['review_results']['reference_reviews'][0]['timing_status']=state
            r=self.blocked(state)
            self.assertEqual(r['groups'][0]['reference_timing_status'],state)

    def test_reference_available_after_decision_blocked(self):
        label=self.opts['label_catalog'][0];label['reference_content']['available_at']='2026-01-02T00:00:00+00:00'
        label['label_fingerprint']=digest(label['reference_content'])
        self.blocked('REFERENCE_TIME_SCOPE_OR_AVAILABILITY_INVALID')

    def test_self_reviewer_llm_or_false_independence_blocked(self):
        for change in ('self','llm','independent','procedure','time'):
            self.data,self.opts=test_only_reviewed_data()
            rr=self.opts['review_results']['response_reviews'][0]['review']
            if change=='self':rr['reviewer_ref']=self.data['outcomes'][0]['agent_id']
            elif change=='llm':rr['reviewer_kind']='LLM'
            elif change=='independent':rr['independent']=False
            elif change=='procedure':rr['procedure_version']='TEST_ONLY_OTHER'
            else:rr['reviewed_at']='2026-10-27T00:00:00+00:00'
            self.blocked('RESPONSE_REVIEW_INDEPENDENCE_PROCEDURE_OR_TIME_INVALID')

    def test_original_response_fingerprint_not_trusted(self):
        self.data['outcomes'][0]['response']['rationale']='TEST_ONLY_TAMPER'
        self.blocked('ORIGINAL_RESPONSE_FINGERPRINT_MISMATCH')

    def test_rehashing_tampered_answer_cannot_replace_original_manifest(self):
        row=self.data['outcomes'][0];row['response']['rationale']='TEST_ONLY_TAMPER'
        row['response_fingerprint']=digest(row['response'])
        self.opts['review_results']['response_reviews'][0]['response_fingerprint']=row['response_fingerprint']
        self.data['judgments'][0]['response_fingerprint']=row['response_fingerprint']
        self.blocked('ORIGINAL_RESPONSE_FINGERPRINT_MISMATCH')

    def test_supported_unresolved_placeholder_is_semantic_contradiction(self):
        row=self.data['outcomes'][0];row['response']['diagnosis']='Unresolved';row['response_fingerprint']=digest(row['response'])
        self.blocked('DIAGNOSIS_CERTAINTY_CONTRADICTION')

    def test_reference_timezone_required_and_uncertainty_preserved(self):
        self.opts['label_catalog'][0]['reference_content']['available_at']='2026-01-01T00:00:30'
        self.blocked('REFERENCE_TIME_SCOPE_OR_AVAILABILITY_INVALID')
        self.data,self.opts=test_only_reviewed_data()
        self.opts['review_results']['reference_reviews'][0]['uncertainty_reason']='TEST_ONLY_PENDING_REVIEW'
        self.blocked('REFERENCE_UNCERTAINTY_UNRESOLVED')

    def test_no_validation_query_does_not_claim_support(self):
        keep={d['decision_id'] for d in self.data['decisions'] if d['split']=='train'}
        self.data['decisions']=[d for d in self.data['decisions'] if d['decision_id'] in keep]
        self.data['split_manifest']['decisions']=[d for d in self.data['split_manifest']['decisions'] if d['decision_id'] in keep]
        for field in ('outcomes','judgments'):
            self.data[field]=[r for r in self.data[field] if r['decision_id'] in keep]
        self.opts['decision_contexts']=[r for r in self.opts['decision_contexts'] if r['decision_id'] in keep]
        for field in ('source_catalog','label_catalog'):
            self.opts[field]=[r for r in self.opts[field] if r['decision_id'] in keep]
        for field in ('reference_reviews','response_reviews'):
            self.opts['review_results'][field]=[r for r in self.opts['review_results'][field] if r['decision_id'] in keep]
        r=self.blocked('POSITIVE_SIMILARITY_SUPPORT_NOT_ESTABLISHED')
        self.assertEqual(r['positive_similarity_support']['status'],'NOT_CHECKED_NO_VALIDATION_QUERIES')

    def test_independent_reference_missing_or_content_tampered(self):
        self.opts['label_catalog']=[]
        self.blocked('INDEPENDENT_REFERENCE_MISSING')
        self.data,self.opts=test_only_reviewed_data()
        self.opts['label_catalog'][0]['reference_content']['diagnosis']='TEST_ONLY_OTHER'
        self.blocked('INDEPENDENT_REFERENCE_CONTENT_OR_REVIEW_INVALID')

    def test_patient_split_overlap_is_global_block(self):
        self.data['split_manifest']['patients']['validation'].append(self.data['split_manifest']['patients']['train'][0])
        r=self.blocked('PATIENT_SPLIT_OVERLAP_OR_DUPLICATE')
        self.assertEqual(r['complete_train_groups'],0)

    def test_abstention_routing_policy_cannot_label_diagnosis(self):
        for name in ('abstention_appropriateness','routing_success'):
            self.data,self.opts=test_only_reviewed_data()
            self.opts['review_results']['response_reviews'][0]['axes']['diagnosis_correctness']['policy_id']=POLICY['policies'][name]['policy_id']
            self.blocked('EVALUATION_TARGET_OR_VERDICT_SEMANTICS_INVALID')
        self.data,self.opts=test_only_reviewed_data();self.opts['evaluation_target']='routing_success'
        self.blocked('TARGET_NOT_SUPPORTED_FOR_ACCURACY_FITTING')

    def test_missing_agent_or_duplicate_agent_not_salvaged(self):
        self.data['outcomes'].pop(0)
        self.blocked('FIVE_AGENT_EVALUABLE_GROUP_REQUIRED')
        self.data,self.opts=test_only_reviewed_data();self.data['outcomes'][0]['agent_id']=self.data['outcomes'][1]['agent_id']
        self.blocked('FIVE_AGENT_EVALUABLE_GROUP_REQUIRED')

    def test_proxy_optin_is_not_clinical_review(self):
        self.opts['decision_contexts'][0]['visibility']['/chief_complaint']['basis']='TEST_ONLY_RELEASE_PROXY'
        self.blocked('UNREVIEWED_PROXY_NOT_APPROVED_FOR_FITTING')

    def test_review_absent_is_not_promoted_and_legacy_not_called(self):
        self.opts['review_results']=empty_review_results('TEST_ONLY')
        with patch('validation.routing_correctness_gate.validate_training_data') as legacy:
            r=self.blocked('RESPONSE_CLINICAL_REVIEW_MISSING')
        legacy.assert_not_called()
        self.assertFalse(r['source_reference_ready'] or r['clinical_review_ready'])

    def test_schema_policy_hash_duplicate_or_orphan_reviews_blocked(self):
        self.opts['review_results']['policy_fingerprint']='0'*64
        self.blocked('REVIEW_RESULTS_SCHEMA_POLICY_OR_PURPOSE_INVALID')
        self.data,self.opts=test_only_reviewed_data()
        self.opts['review_results']['response_reviews'].append(deepcopy(self.opts['review_results']['response_reviews'][0]))
        self.blocked('GATE_INPUT_CONTRACT_INVALID')

    def test_positive_similarity_support_is_query_specific(self):
        self.opts['min_samples']=11
        r=self.blocked('POSITIVE_SIMILARITY_SUPPORT_NOT_ESTABLISHED')
        self.assertEqual(r['positive_similarity_support']['status'],'INSUFFICIENT_SUPPORT')
        self.assertEqual(r['complete_train_groups'],10)

    def test_existing_validator_still_rejects_downstream_binding_error(self):
        self.data['outcomes'][0]['version']='TEST_ONLY_OTHER'
        r=self.blocked('MODEL_VERSION_BINDING_MISMATCH')
        self.assertIsInstance(r['legacy_validator'],dict)

    def test_test_only_requires_explicit_opt_in(self):
        self.opts['allow_test_only']=False
        self.blocked('TEST_ONLY_NOT_AUTHORIZED')

    def test_actual_75_stay_blocked_and_candidate010_not_adjudicated(self):
        data,opts=actual_gate_inputs();before=digest([data,opts])
        with patch('validation.routing_correctness_gate.validate_training_data') as legacy:
            r=validate_evaluation_gate(data,**opts)
        legacy.assert_not_called()
        self.assertEqual(r['response_states'],{'UNRESOLVED':53,'SUPPORTED':22})
        self.assertEqual(r['complete_train_groups'],0)
        self.assertFalse(r['utility_fit_ready'])
        self.assertEqual(r['groups'][9]['reference_timing_status'],'REFERENCE_TIMING_NOT_REVIEWED')
        self.assertEqual(r['groups'][9]['source_condition_start_after_decision_days'],4)
        self.assertEqual(before,digest([data,opts]))
        self.assertTrue(all(row['correct'] is None for row in data['outcomes']))
        self.assertNotIn(data['decisions'][0]['patient_id'],json.dumps(r))

    def test_cli_no_write_default_and_sanitized_error(self):
        with patch('scripts.check_routing_correctness.audit_actual_gate',return_value={'utility_fit_ready':False}),patch('scripts.check_routing_correctness.atomic_json') as writer,patch('builtins.print'):
            self.assertEqual(main([]),2);writer.assert_not_called()
        with patch('scripts.check_routing_correctness.audit_actual_gate',side_effect=ValueError('TEST_ONLY_PRIVATE')),patch('builtins.print') as output:
            self.assertEqual(main([]),2)
            self.assertNotIn('TEST_ONLY_PRIVATE',str(output.call_args))


if __name__=='__main__':
    unittest.main()
