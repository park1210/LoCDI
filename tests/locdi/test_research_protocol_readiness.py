"""TEST_ONLY policy/contract fixtures, never clinical judgments or fitting data."""
import copy,json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from app.locdi.llm_setup import ROOT,digest
from app.locdi.research_protocol_audit import anchored_sources,blocker_plan,paper_inventory
from app.locdi.dataset_freeze import sha
from experiments.evaluation_protocol import metric_contracts,condition_contracts
from validation.official_evaluation_readiness import assess_official_evaluation
from validation.review_submission_contract import REQUIRED,validate_submission


def submission():
    result={k:None for k in REQUIRED}
    result.update(schema_version='locdi.review_submission.v1',submission_kind='REFERENCE_REVIEW',
        review_case_ref='TEST_ONLY_CASE',decision_ref='TEST_ONLY_DECISION',reference_source='TEST_ONLY_SOURCE',
        source_fingerprint='a'*64,reference_fingerprint='b'*64,
        observed_at='2026-01-01T00:00:00+00:00',available_at='2026-01-01T00:00:00+00:00',
        decision_at='2026-01-01T00:00:00+00:00',review_timestamp='2026-01-02T00:00:00+00:00',
        reviewer_ref='TEST_ONLY_REVIEWER',reviewer_role='HUMAN_CLINICAL_REVIEWER',reviewer_independent=True,
        decision_status='REVIEWED',evidence_reference='TEST_ONLY_EVIDENCE',review_policy_version='TEST_ONLY.v1',
        reference_timing_status='CURRENT_DECISION_REFERENCE_VALID',evaluation_target='diagnosis_correctness',
        clinical_evidence_rationale='TEST_ONLY structural fixture, no clinical claim')
    return result


def response():
    result=submission();result.update(submission_kind='RESPONSE_JUDGMENT',response_ref='TEST_ONLY_RESPONSE',
        response_fingerprint='c'*64,response_recorded_at='2026-01-01T01:00:00+00:00',certainty='supported',correctness_label=True)
    return result


class ReviewSubmissionTests(unittest.TestCase):
    def test_reference_shape_is_not_approval(self):
        r=validate_submission(submission());self.assertEqual(r['structural_status'],'READY');self.assertEqual(r['status'],'BLOCKED')
    def test_supported_shape_is_not_clinical_review(self):
        r=validate_submission(response());self.assertEqual(r['structural_status'],'READY');self.assertFalse(r['clinical_review_ready'])
    def test_unresolved_true_blocked(self):
        v=response();v['certainty']='unresolved';self.assertIn('UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN',validate_submission(v)['blocker_codes'])
    def test_unresolved_false_blocked(self):
        v=response();v.update(certainty='unresolved',correctness_label=False);self.assertEqual(validate_submission(v)['structural_status'],'BLOCKED')
    def test_unresolved_not_evaluable_null_preserved(self):
        v=response();v.update(certainty='unresolved',correctness_label=None,decision_status='NOT_EVALUABLE',not_evaluable_reason='TEST_ONLY')
        self.assertEqual(validate_submission(v)['structural_status'],'READY');self.assertIsNone(v['correctness_label'])
    def test_four_day_future_reference_blocked(self):
        v=submission();v.update(observed_at='2026-01-05T00:00:00+00:00',available_at='2026-01-05T00:00:00+00:00',review_timestamp='2026-01-06T00:00:00+00:00')
        self.assertIn('CURRENT_DECISION_REFERENCE_NOT_VERIFIED',validate_submission(v)['blocker_codes'])
    def test_unreviewed_timing_not_promoted(self):
        v=submission();v['reference_timing_status']='REFERENCE_TIMING_NOT_REVIEWED';self.assertEqual(validate_submission(v)['structural_status'],'BLOCKED')
    def test_timezone_required(self):
        v=submission();v['decision_at']='2026-01-01T00:00:00';self.assertIn('REVIEW_TIMESTAMP_INVALID',validate_submission(v)['blocker_codes'])
    def test_reviewer_independence(self):
        v=submission();v['reviewer_independent']=False;self.assertEqual(validate_submission(v)['structural_status'],'BLOCKED')
    def test_llm_not_independent_reviewer(self):
        v=submission();v['reviewer_role']='LLM';self.assertEqual(validate_submission(v)['structural_status'],'BLOCKED')
    def test_reference_review_response_blinded(self):
        v=submission();v['response_ref']='TEST_ONLY';self.assertIn('REFERENCE_REVIEW_MUST_BE_RESPONSE_BLINDED',validate_submission(v)['blocker_codes'])
    def test_other_axis_not_diagnosis(self):
        v=response();v['evaluation_target']='routing_success';self.assertIn('DIAGNOSIS_LABEL_TARGET_MIXED',validate_submission(v)['blocker_codes'])
    def test_unreviewed_label_forbidden(self):
        v=response();v['decision_status']='NOT_REVIEWED';self.assertIn('UNREVIEWED_LABEL_NOT_NULL',validate_submission(v)['blocker_codes'])
    def test_self_hash_proof_never_authenticates(self):
        v=submission();v['approval_proof']={'method':'DETACHED_SIGNATURE','issuer_ref':'TEST_ONLY',
            'approval_ref':'TEST_ONLY','payload_fingerprint':digest({k:x for k,x in v.items() if k!='approval_proof'}),'proof_sha256':'d'*64}
        r=validate_submission(v);self.assertEqual(r['structural_status'],'READY');self.assertFalse(r['clinical_review_ready'])
    def test_mismatched_proof_and_no_private_echo(self):
        v=submission();v['approval_proof']={'method':'DETACHED_SIGNATURE','issuer_ref':'TEST_ONLY','approval_ref':'TEST_ONLY',
            'payload_fingerprint':'0'*64,'proof_sha256':'d'*64}
        r=validate_submission(v);self.assertEqual(r['structural_status'],'BLOCKED');self.assertNotIn('TEST_ONLY_CASE',json.dumps(r))
    def test_original_submission_unchanged(self):
        v=response();before=copy.deepcopy(v);validate_submission(v);self.assertEqual(v,before)


class OfficialReadinessTests(unittest.TestCase):
    def setUp(self):
        self.configs={n:json.loads((ROOT/'experiments/configs'/(n+'.json')).read_text(encoding='utf8')) for n in ('B0','B1','B2','B3','B4')}
        self.case={'case_id':'TEST_ONLY_CASE','patient_id':'TEST_ONLY_PATIENT',
            'agent_visible':{'index_encounter_id':'TEST_ONLY_VISIT'},'environment':{'current_encounter_events':[]}}
    def gate(self,**kw):return assess_official_evaluation(self.configs,**kw)
    def test_missing_inputs_blocked(self):
        r=self.gate();self.assertFalse(r['research_evaluation_ready']);self.assertIn('OFFICIAL_PAIRED_RUNNER',r['blocker_codes'])
    def test_ready_flags_never_approve(self):
        for c in self.configs.values():c['ready']=True
        r=self.gate(protocol={'approved':True},research_approval={'approved':True});self.assertEqual(r['status'],'BLOCKED')
    def test_changed_condition_blocked(self):
        self.configs['B0']['features']['memory_mode']='selective'
        self.assertIn('OFFICIAL_FEATURES_B0',self.gate()['blocker_codes'])
    def test_pair_source_mismatch(self):
        sets={n:[copy.deepcopy(self.case)] for n in self.configs};sets['B4'][0]['environment']['current_encounter_events']=[{'TEST_ONLY':True}]
        self.assertIn('PAIRED_SOURCE_ACTIONS_AND_PAYLOADS',self.gate(cases=[self.case],case_sets=sets)['blocker_codes'])
    def test_patient_split_overlap(self):
        r=self.gate(cases=[self.case],split_manifest={'train':['TEST_ONLY_PATIENT'],'validation':[],'test':['TEST_ONLY_PATIENT']})
        self.assertIn('PATIENT_SPLIT_AND_TEST_COVERAGE',r['blocker_codes'])
    def test_disjoint_split_does_not_approve_references(self):
        r=self.gate(cases=[self.case],split_manifest={'train':['TEST_ONLY_OTHER'],'validation':[],'test':['TEST_ONLY_PATIENT']})
        self.assertNotIn('PATIENT_SPLIT_AND_TEST_COVERAGE',r['blocker_codes']);self.assertIn('INDEPENDENT_CLINICAL_REFERENCE',r['blocker_codes'])
    def test_artifact_gate_reused_without_fitting(self):
        with patch('validation.official_evaluation_readiness.assess_artifacts',return_value={'research_evaluation_ready':False}) as artifact:
            self.gate(artifact_inputs={'registry_manifest':{}});artifact.assert_called_once()
    def test_no_execution_or_mutation(self):
        before=copy.deepcopy(self.configs)
        with patch('experiments.run_locdi_ablation.run_ablation',side_effect=AssertionError('NO_EXECUTION')):
            r=self.gate()
        self.assertEqual(self.configs,before);self.assertEqual(r['generation_requests'],0)
    def test_handoff_not_inferred(self):
        rows=condition_contracts(self.configs);self.assertTrue(all(r['sequential_handoff']=='NOT_SPECIFIED_BY_EXISTING_OFFICIAL_JSON' for r in rows))
    def test_metric_denominators_and_unresolved(self):
        rows={r['metric_id']:r for r in metric_contracts()};self.assertEqual(len(rows),21)
        self.assertIn('/ K',rows['memory_precision_at_k']['definition']);self.assertIn('not false',rows['diagnosis_correctness']['required_reference'])
        self.assertTrue(all('NOT_APPLICABLE' in r['missing_rule'] for r in rows.values()))
    def test_original_sources_anchor_allows_additions_and_detects_mutation(self):
        with TemporaryDirectory() as t:
            root=Path(t);p=root/'old.py';p.write_text('# TEST_ONLY',encoding='utf8');files={'old.py':sha(p)}
            inv={'files':files,'source_fingerprint':digest(files)};(root/'new.py').write_text('# TEST_ONLY',encoding='utf8')
            self.assertEqual(anchored_sources(root,inv,digest(files))['status'],'PASS')
            p.write_text('# CHANGED_TEST_ONLY',encoding='utf8');self.assertEqual(anchored_sources(root,inv,digest(files))['status'],'FAIL')
    def test_blocker_dependency_sorting(self):
        rows=blocker_plan(['ACTUAL_PAIRED_EXECUTION_CONSISTENCY','CONFIRMED_PAPER_AND_PROTOCOL','RESEARCH_VOI_ARTIFACT'])
        self.assertEqual([r['priority'] for r in rows],[1,4,6]);self.assertTrue(all(r['blocks_official_evaluation'] for r in rows))
    def test_paper_inventory_utf8_paths(self):
        from types import SimpleNamespace
        with patch('app.locdi.research_protocol_audit.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='docs/시뮬레이션 구조/TEST_ONLY.md\n')) as run:
            r=paper_inventory();self.assertEqual(r['format_counts'],{'.md':1});self.assertEqual(run.call_args.kwargs['encoding'],'utf-8')
    def test_paired_order_mismatch_blocked(self):
        second=copy.deepcopy(self.case);second.update(case_id='TEST_ONLY_CASE2',patient_id='TEST_ONLY_PATIENT2')
        sets={n:[self.case,second] for n in self.configs};sets['B4']=[second,self.case]
        r=self.gate(cases=[self.case,second],case_sets=sets);self.assertIn('PAIRED_INPUT_ORDER',r['blocker_codes'])


if __name__=='__main__':unittest.main()
