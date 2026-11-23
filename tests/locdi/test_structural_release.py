"""TEST_ONLY release checks; pure audit fixtures never replace Runner results."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from app.locdi.structural_release import (STRUCTURAL_REQUIRED,evaluate_release,inspect_case,
    audit_handoff,verify_boundary,validate_episode_links)
from app.locdi.dataset_loader import load_cases
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import ROOT,digest
from app.locdi.episode_exporter import _write
from scripts.verify_simulator_release import verify_manifest,stored_offline
from phase211_support import execute_case,features


class ReleaseDecisionTests(unittest.TestCase):
    def test_structural_and_research_are_independent(self):
        r=evaluate_release({k:'PASS' for k in STRUCTURAL_REQUIRED})
        self.assertEqual(r['STRUCTURAL_RELEASE_CANDIDATE'],'READY_DEVELOPMENT_ONLY')
        self.assertEqual(r['RESEARCH_EVALUATION_READY'],'BLOCKED')

    def test_fail_skip_blocked_unknown_never_satisfy_requirement(self):
        for status in ('FAIL','BLOCKED','NOT_CHECKED','SKIP',True,None):
            r=evaluate_release(dict({k:'PASS' for k in STRUCTURAL_REQUIRED},source_boundary_verified=status))
            self.assertEqual(r['STRUCTURAL_RELEASE_CANDIDATE'],'BLOCKED')

    def test_absent_regression_does_not_reuse_old_pass(self):
        r=evaluate_release({k:'PASS' for k in STRUCTURAL_REQUIRED if k!='full_regression_passed'})
        self.assertIn('full_regression_passed',r['structural_blocker_codes'])

    def test_research_metadata_cannot_self_authorize(self):
        r=evaluate_release(dict({k:'PASS' for k in STRUCTURAL_REQUIRED},clinical_reference_ready='PASS',
            live_adaptive_voi_ready='PASS',research_evaluation_ready='PASS'))
        self.assertEqual(r['RESEARCH_EVALUATION_READY'],'BLOCKED')


class SavedRunAuditTests(unittest.TestCase):
    def test_failed_runs_remain_failed_and_success_is_exactly_one(self):
        from app.locdi.structural_release import RUNS
        rows=[audit_handoff(r) for r in RUNS]
        self.assertEqual([r['status'] for r in rows],['PASS']*3)
        self.assertEqual([r['recorded_status'] for r in rows],['failed','failed','completed'])
        self.assertEqual(sum(r['handoff_completed'] for r in rows),1)
        self.assertEqual([r['generation_count'] for r in rows],[2,2,4])
        self.assertEqual(rows[0]['legacy_semantic_detail'],'NOT_CHECKED_NOT_RECORDED')
        self.assertEqual(rows[1]['requests'][1]['specialty_evidence_validation']['detail_code'],'SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE')

    def test_unknown_run_cannot_replace_registered_evidence(self):
        with self.assertRaises(ValueError):audit_handoff('f'*32)

    def test_original_40_exports_revalidated_read_only(self):
        self.assertEqual(stored_offline()['status'],'PASS')


class StaticAndObservedBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[1]
        cls.registry=load(ROOT/'configs/agent_registry.specialty_development.json')
        cls.result,cls.client,_=execute_case(cls.case,features(cls.case,2),'D')

    def test_real_second_case_runner_boundary_and_export_inputs(self):
        self.assertEqual(self.result['status'],'completed')
        self.assertEqual(verify_boundary(self.case,self.client,self.result)['status'],'PASS')
        r=inspect_case(self.case,2,self.registry)
        self.assertEqual(r['status'],'PASS');self.assertEqual(r['live_execution'],'NOT_CHECKED')
        self.assertEqual(r['episode_export'],'NOT_CHECKED_NOT_EXECUTED')
        self.assertEqual(r['validator_statuses']['provenance'],'NOT_REVIEWED')

    def test_static_future_memory_rejected_without_clinical_adjudication(self):
        c=deepcopy(self.case);c['agent_visible']['past_visits'][0]['records'][0]['available_at']='2099-01-01T00:00:00Z'
        r=inspect_case(c,2,self.registry);self.assertEqual(r['status'],'FAIL')
        self.assertNotIn(self.case['patient_id'],json.dumps(r))

    def test_invalid_registry_blocks_structural_case(self):
        self.assertEqual(inspect_case(self.case,2,{'schema_version':0})['status'],'FAIL')

    def test_quoted_source_evidence_cannot_evade_direct_release_check(self):
        # Pure TEST_ONLY verifier fixture, not a simulated/clinical case result.
        evidence='{"TEST_ONLY_result":"unreleased"}'
        case={'environment':{'current_encounter_events':[{'action':'Test','target':'TEST_ONLY_T','evidence':evidence,'available_minute':10}]},
            'agent_visible':{'past_visits':[],'index_encounter_start':'2026-01-01T00:00:00Z'}}
        observation={'elapsed_minutes':1,'objective_findings':['TEST_ONLY_T: '+evidence],'history':[],
            'longitudinal_memory':{'records':[]}}
        client=SimpleNamespace(calls=[{'payload':{'observation':observation}}])
        raw={'raw':{'execution':{'visits':[{'final_state':{'event_log':[]}}]}}}
        self.assertEqual(verify_boundary(case,client,raw)['status'],'FAIL')

    def test_injected_private_field_or_premature_answer_blocks(self):
        c=SimpleNamespace(calls=deepcopy(self.client.calls));c.calls[0]['payload']['gold']={}
        self.assertFalse(verify_boundary(self.case,c,self.result)['checks']['private_scope'])
        c=SimpleNamespace(calls=deepcopy(self.client.calls))
        answered=next(x['payload']['observation']['history'] for x in c.calls if x['payload']['observation']['history'])
        c.calls[0]['payload']['observation']['history']=deepcopy(answered)
        c.calls[0]['payload']['observation']['elapsed_minutes']=0
        self.assertFalse(verify_boundary(self.case,c,self.result)['checks']['ask_release'])

    def test_memory_identity_tampering_is_not_hidden_by_known_id(self):
        c=SimpleNamespace(calls=deepcopy(self.client.calls))
        c.calls[0]['payload']['observation']['longitudinal_memory']['records'][0]['patient_id']='TEST_ONLY_OTHER_PATIENT'
        self.assertFalse(verify_boundary(self.case,c,self.result)['checks']['memory_scope'])


class ReleaseManifestTests(unittest.TestCase):
    def fixture(self,root):
        _write({'schema_version':'TEST_ONLY','value':1},'reports/test.json',root,'reports')
        manifest={'files':[{'path':'reports/test.json','sha256':sha(root/'reports/test.json')}],
            'private_selection_sha256':None}
        manifest['fingerprint']=digest(manifest);return manifest

    def test_valid_manifest_and_changed_file_are_distinct(self):
        with TemporaryDirectory(dir=str(ROOT/'outputs')) as d:
            root=Path(d);m=self.fixture(root);self.assertTrue(verify_manifest(root,m))
            (root/'reports/test.json').write_text('{}',encoding='utf-8');self.assertFalse(verify_manifest(root,m))

    def test_fingerprint_mismatch_and_extra_file_are_rejected(self):
        with TemporaryDirectory(dir=str(ROOT/'outputs')) as d:
            root=Path(d);m=self.fixture(root);bad=deepcopy(m);bad['fingerprint']='0'*64
            self.assertFalse(verify_manifest(root,bad))
            (root/'reports/extra.txt').write_text('TEST_ONLY',encoding='utf-8');self.assertFalse(verify_manifest(root,m))

    def test_duplicate_or_escape_path_rejected_even_if_rehashed(self):
        with TemporaryDirectory(dir=str(ROOT/'outputs')) as d:
            root=Path(d);m=self.fixture(root)
            for mutation in ('duplicate','escape'):
                bad=deepcopy(m);bad.pop('fingerprint')
                if mutation=='duplicate':bad['files']*=2
                else:bad['files'][0]['path']='../outside.json'
                bad['fingerprint']=digest(bad);self.assertFalse(verify_manifest(root,bad))

    def test_private_selection_hash_is_enforced_without_public_content(self):
        with TemporaryDirectory(dir=str(ROOT/'outputs')) as d:
            root=Path(d);m=self.fixture(root);_write({'patient_id':'TEST_ONLY_PRIVATE'},'private/selection.json',root,'private')
            self.assertFalse(verify_manifest(root,m));m.pop('fingerprint');m['private_selection_sha256']=sha(root/'private/selection.json')
            m['fingerprint']=digest(m);self.assertTrue(verify_manifest(root,m))
            self.assertNotIn('TEST_ONLY_PRIVATE',json.dumps(m))

    def test_nonfinite_json_not_accepted_even_if_hash_updated(self):
        with TemporaryDirectory(dir=str(ROOT/'outputs')) as d:
            root=Path(d);m=self.fixture(root);(root/'reports/test.json').write_text('{"TEST_ONLY":NaN}',encoding='utf-8')
            m.pop('fingerprint');m['files'][0]['sha256']=sha(root/'reports/test.json');m['fingerprint']=digest(m)
            self.assertFalse(verify_manifest(root,m))


if __name__=='__main__':unittest.main()
