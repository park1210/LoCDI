"""TEST_ONLY paired harness, private references, budgets and exact reaggregation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.current_ablation_runner import (
    CurrentAblationBlocked, METRICS, make_contract, validate_contract, execution_readiness,
    features_for, reserve_generation, run_paired_ablation, reaggregate_trace,
)
from app.locdi.llm_setup import digest


def fixture(conditions=('B0', 'B1', 'B2')):
    panel = [{'case_ref': 'TEST_ONLY_CASE_' + str(i), 'patient_ref': 'TEST_ONLY_PATIENT_' + str(i),
        'execution_case': {'chief_complaint': 'TEST_ONLY complaint', 'ordinal': i},
        'reference': {'accepted_diagnoses': ['TEST_ONLY_A'], 'accepted_departments': ['TEST_ONLY_DEPT'],
                      'memory_annotation': {'relevant_ids': ['TEST_ONLY_M1', 'TEST_ONLY_M2']}}} for i in range(2)]
    common = {k: 'a' * 64 for k in ['dataset_sha256', 'patient_split_sha256', 'registry_sha256',
        'prompt_sha256', 'memory_sha256', 'action_catalog_sha256']}
    common.update(model_bindings=[{'agent_id': 'TEST_ONLY_AGENT', 'model': 'TEST_ONLY_MODEL', 'version': 'TEST_ONLY_V1',
        'digest': 'sha256:' + 'b' * 64, 'prompt_sha256': 'c' * 64}], decoding={'temperature': 0, 'max_tokens': 1024},
        seed=42, memory_source='fixed', diagnosis_catalog=['TEST_ONLY_A', 'TEST_ONLY_B'],
        source_visibility={'basis': 'TEST_ONLY'}, time_contract={'mode': 'TEST_ONLY'},
        reference_policy={'kind': 'TEST_ONLY_SOURCE_ONLY'}, repeats=1, minute_limit=90,
        request_budget={'total': 6, 'per_episode': 1, 'retry': 0, 'repair': 0},
        resource_budget={'wall_seconds': 60, 'output_bytes': 1024**2, 'memory_bytes': 4*1024**3})
    contract = make_contract(common, panel, conditions=conditions, purpose='TEST_ONLY')
    approval = {'approved': True, 'scope': 'C4_38_CURRENT_ABLATION_DEVELOPMENT',
        'plan_fingerprint': contract['fingerprint'], 'evidence_type': 'USER_EXPLICIT',
        'user_statement': 'TEST_ONLY explicit fixture, not actual execution permission', 'recorded_at': 'TEST_ONLY'}
    return contract, panel, approval


def response(case, features):
    reserve_generation()
    return {'status': 'completed', 'request_count': 1,
        'prediction': {'certainty': 'supported', 'diagnosis': 'TEST_ONLY_A', 'department': 'TEST_ONLY_DEPT',
                       'algorithmic_diagnosis_choice': 'TEST_ONLY_B'},
        'action_counts': {'Ask': 1, 'Test': 1, 'Consult': 0, 'Diagnose': 1},
        'memory_selected_ids': [] if features['memory_mode'] == 'none' else ['TEST_ONLY_M1'],
        'routing_call_count': 0, 'routing_abstention_count': 0,
        'computation': {'wall_seconds': 0.01, 'planner_calls': 0, 'evaluated_states': 0}}


class CurrentAblationRunner(unittest.TestCase):
    def execute(self, contract, panel, approval, backend=response):
        with tempfile.TemporaryDirectory() as root:
            pub, private = Path(root)/'public', Path(root)/'private'
            result = run_paired_ablation(contract, panel, run_one=backend, execute=True,
                approval_record=approval, public_root=pub, private_root=private)
            recomputed = reaggregate_trace(private/'episodes.jsonl', contract, panel)
            return result, recomputed

    def test_dry_run_never_calls_backend(self):
        c,p,a = fixture()
        with patch(__name__ + '.response') as backend:
            result = run_paired_ablation(c,p,run_one=backend)
        self.assertEqual(result['status'], 'DRY_RUN'); backend.assert_not_called()

    def test_no_approval_no_output_or_calls(self):
        c,p,a = fixture()
        with tempfile.TemporaryDirectory() as d, patch(__name__ + '.response') as backend:
            with self.assertRaises(ValueError):
                run_paired_ablation(c,p,run_one=backend,execute=True,public_root=Path(d)/'pub',private_root=Path(d)/'pr')
            self.assertFalse((Path(d)/'pub').exists()); backend.assert_not_called()

    def test_fixed_condition_features(self):
        self.assertEqual(features_for('B0')['memory_mode'], 'none')
        self.assertEqual(features_for('B1')['memory_mode'], 'full_history')
        self.assertEqual(features_for('B2')['top_k'], 5)
        self.assertEqual(features_for('B4')['action_selector'], 'bayesian_voi')

    def test_same_input_and_reference_not_passed(self):
        c,p,a = fixture(); calls=[]
        def backend(case, features):
            self.assertNotIn('reference', case); calls.append(deepcopy(case)); return response(case,features)
        r, reproduced = self.execute(c,p,a,backend)
        self.assertEqual(calls[:2],calls[2:4]); self.assertEqual(calls[:2],calls[4:6])
        self.assertEqual(r['metrics'], reproduced)
        self.assertEqual(r['generation_count'],6)

    def test_reference_leak_nested_rejected(self):
        c,p,a = fixture();p[0]['execution_case']['nested']={'gold': 'TEST_ONLY'}
        c['panel_fingerprint']=digest(p); c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'REFERENCE'): validate_contract(c,p)

    def test_serialized_reference_leak_rejected(self):
        c,p,a=fixture();p[0]['execution_case']['text']='{"nested":{"gold":"TEST_ONLY"}}'
        c['panel_fingerprint']=digest(p);c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'REFERENCE'):validate_contract(c,p)

    def test_condition_drift_rejected(self):
        c,p,a = fixture();c['features']['B2']['top_k']=6
        c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'DRIFT'):validate_contract(c,p)

    def test_boolean_feature_not_integer_or_zero(self):
        c,p,a = fixture();c['features']['B0']['voi_enabled']=0
        c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'DRIFT'):validate_contract(c,p)

    def test_missing_common_not_filled(self):
        c,p,a = fixture();c['common']['model_bindings']=None
        c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'UNRESOLVED'):validate_contract(c,p)

    def test_actual_gate_not_ready_boolean(self):
        c,p,a = fixture(); c['purpose']='official_research_evaluation';c['ready']=True
        c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'})
        r=execution_readiness(c,p)
        self.assertEqual(r['status'],'BLOCKED');self.assertIn('EXISTING_RESEARCH_GATE_BLOCKED',r['blocker_codes'])

    def test_adaptive_missing_utility_blocked(self):
        c,p,a=fixture(('B3','B4'));r=execution_readiness(c,p)
        self.assertIn('ROUTING_UTILITY_NOT_ADMITTED',r['blocker_codes'])
        self.assertIn('VOI_RESEARCH_ARTIFACT_NOT_ADMITTED',r['blocker_codes'])

    def test_unresolved_preserves_null_and_denominator(self):
        c,p,a=fixture(('B0',))
        def backend(case,features):
            reserve_generation()
            return {'status':'completed','request_count':1,'prediction':{'certainty':'unresolved','diagnosis':None,'department':None},
                    'action_counts':{'Ask':0,'Test':0,'Consult':0,'Diagnose':1}}
        r,_=self.execute(c,p,a,backend);m=r['metrics']['conditions']['B0']
        self.assertEqual(m['unresolved'],2);self.assertEqual(m['supported_answer_agreement']['denominator'],0)
        self.assertIsNone(m['supported_answer_agreement']['value'])
        self.assertEqual(m['correct_supported_yield_registered']['denominator'],2)

    def test_algorithmic_choice_separate_from_supported(self):
        c,p,a=fixture(('B0',));r,_=self.execute(c,p,a)
        m=r['metrics']['conditions']['B0']
        self.assertEqual(m['supported_answer_agreement']['value'],1)
        self.assertEqual(m['algorithmic_choice_agreement']['value'],0)

    def test_request_limit_blocks_remaining_as_not_attempted(self):
        c,p,a=fixture();c['common']['request_budget']['total']=1
        c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'});a['plan_fingerprint']=c['fingerprint']
        r,_=self.execute(c,p,a)
        self.assertEqual(r['status'],'PARTIAL');self.assertEqual(r['generation_count'],1)
        self.assertEqual(sum(m['failed'] for m in r['metrics']['conditions'].values()),1)
        self.assertEqual(sum(m['not_attempted'] for m in r['metrics']['conditions'].values()),4)

    def test_episode_limit_includes_repair(self):
        c,p,a=fixture(('B0',))
        def backend(case,features):reserve_generation();reserve_generation()
        r,_=self.execute(c,p,a,backend)
        self.assertEqual(r['generation_count'],1);self.assertEqual(r['status'],'PARTIAL')

    def test_unmetered_request_report_rejected(self):
        c,p,a=fixture(('B0',))
        def backend(case,features):
            r=response(case,features);r['request_count']=0;return r
        r,_=self.execute(c,p,a,backend)
        self.assertEqual(r['metrics']['conditions']['B0']['failed'],1)
        self.assertEqual(r['metrics']['conditions']['B0']['source_diagnosis_correct_supported_count'],0)

    def test_backend_timeout_not_incorrect(self):
        c,p,a=fixture(('B0',))
        def backend(case,features):reserve_generation();raise TimeoutError('TEST_ONLY')
        r,_=self.execute(c,p,a,backend);m=r['metrics']['conditions']['B0']
        self.assertEqual(m['failed'],1);self.assertEqual(m['not_attempted'],1)
        self.assertEqual(m['supported_answer_agreement']['denominator'],0)
        self.assertEqual(m['unknown_action_count_cases'],1)
        self.assertIsNone(m['mean_executed_actions_registered'])

    def test_duplicate_case_rejected(self):
        c,p,a=fixture();p[1]['case_ref']=p[0]['case_ref'];c['panel_fingerprint']=digest(p)
        c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):validate_contract(c,p)

    def test_bad_trace_hash_rejected(self):
        c,p,a=fixture(('B0',))
        with tempfile.TemporaryDirectory() as d:
            pub,pr=Path(d)/'pub',Path(d)/'pr'
            run_paired_ablation(c,p,run_one=response,execute=True,approval_record=a,public_root=pub,private_root=pr)
            text=(pr/'episodes.jsonl').read_text(encoding='utf-8').replace('"status": "completed"','"status": "failed"',1)
            (pr/'episodes.jsonl').write_text(text,encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'FINGERPRINT'):reaggregate_trace(pr/'episodes.jsonl',c,p)

    def test_no_overwrite_and_public_no_predictions(self):
        c,p,a=fixture(('B0',))
        with tempfile.TemporaryDirectory() as d:
            pub,pr=Path(d)/'pub',Path(d)/'pr'
            run_paired_ablation(c,p,run_one=response,execute=True,approval_record=a,public_root=pub,private_root=pr)
            text=''.join(f.read_text(encoding='utf-8') for f in pub.iterdir())
            self.assertNotIn('TEST_ONLY_A',text);self.assertNotIn('TEST_ONLY_CASE_',text)
            with self.assertRaisesRegex(ValueError,'OVERWRITE'):
                run_paired_ablation(c,p,run_one=response,execute=True,approval_record=a,public_root=pub,private_root=pr)

    def test_memory_annotation_and_source_identity_separate(self):
        c,p,a=fixture(('B2',));r,_=self.execute(c,p,a);m=r['metrics']['conditions']['B2']
        self.assertEqual(m['memory_annotation_metrics']['cases'],2)
        self.assertEqual(m['memory_annotation_metrics']['precision']['value'],1)
        self.assertEqual(m['memory_annotation_metrics']['recall']['value'],.5)
        self.assertIsNone(m['source_identity_metrics']['precision']['value'])

    def test_memory_budget_checked_before_backend(self):
        c,p,a=fixture(('B0',));c['common']['resource_budget']['memory_bytes']=1
        c['fingerprint']=digest({k:v for k,v in c.items() if k!='fingerprint'});a['plan_fingerprint']=c['fingerprint']
        with patch(__name__+'.response') as backend:
            r,_=self.execute(c,p,a,backend)
            backend.assert_not_called();self.assertEqual(r['status'],'PARTIAL')

    def test_resource_cap_rechecked_before_each_generation(self):
        c,p,a=fixture(('B0',))
        probes=[]
        def memory():
            probes.append(1)
            return 0 if len(probes)<3 else c['common']['resource_budget']['memory_bytes']
        def backend(case,features):
            reserve_generation()
            reserve_generation()
            self.fail('Second generation must be blocked by resource guard')
        with patch('app.locdi.current_ablation_runner._process_private_bytes',side_effect=memory):
            r,_=self.execute(c,p,a,backend)
        self.assertEqual(r['generation_count'],1)
        self.assertEqual(r['metrics']['conditions']['B0']['failed'],1)


if __name__=='__main__':unittest.main()
