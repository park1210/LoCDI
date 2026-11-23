"""New runner contracts; no inference or actual Source fitting."""
from copy import deepcopy
import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from scripts import run_original50_final as runner
from app.locdi.llm_setup import digest

class FinalExecutionTests(unittest.TestCase):
    def test_stage_ids_unique(self):
        names={runner.identity(s,c) for s in runner.STAGES for c in ('B012','B034')}
        self.assertEqual(len(names),6)

    def test_no_unregistered_scope(self):
        with self.assertRaises(ValueError):runner.identity('Smoke5','B34_BYPASS')

    def test_all_conditions_blocked(self):
        p={'blocker_codes':runner.BLOCKERS,'run_id':'never','plan_id':'blocked','registered_patients':50,'registered_episodes':250,'generation_total':3000,'wall_seconds':14400}
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'load',return_value=(Path(d),Path(d),p)):
            r=runner.check('Full50','B034')
        self.assertEqual(r['status'],'BLOCKED');self.assertEqual(r['generation_cap'],3000)
        self.assertIn('U2_VALIDATION_OVERLAPS_M5_TRAIN',r['blocker_codes'])

    def test_missing_real_smoke_blocks_larger_stage(self):
        p={'blocker_codes':[],'run_id':'never','plan_id':'partial','registered_patients':50,'registered_episodes':150,'generation_total':1800,'wall_seconds':14400}
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'load',return_value=(Path(d),Path(d),p)),patch.object(runner,'technical_proof',side_effect=FileNotFoundError()):
            self.assertIn('PRECEDING_REAL_TECHNICAL_STAGE_REQUIRED',runner.check('Full50')['blocker_codes'])

    def test_technical_skip_requires_same_comparison(self):
        p={'blocker_codes':[],'comparison_fingerprint':'A','run_id':'never','plan_id':'partial','registered_patients':50,'registered_episodes':150,'generation_total':1800,'wall_seconds':14400}
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'load',return_value=(Path(d),Path(d),p)),patch.object(runner,'technical_proof',return_value={'Verify10_skip_allowed':True,'comparison_fingerprint':'B'}):
            self.assertEqual(runner.check('Full50')['status'],'BLOCKED')

    def test_hash_change_blocks_load(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/runner.identity('Smoke5');folder.mkdir();file=root/'source.txt';file.write_text('x')
            p={'pins':{str(file):'invalid'}};p['fingerprint']=digest(p);(folder/'plan.json').write_text(json.dumps(p))
            with patch.object(runner,'PUB',root),patch.object(runner,'PR',root),self.assertRaisesRegex(ValueError,'PIN_CHANGED'):runner.load('Smoke5')

    def test_execute_cannot_bypass_blocked_check(self):
        with patch.object(runner,'check',return_value={'status':'BLOCKED'}),self.assertRaisesRegex(ValueError,'PREFLIGHT_BLOCKED'):
            runner.execute('Smoke5','nonexistent')

    def test_run_used_blocks(self):
        p={'blocker_codes':[],'run_id':'used','plan_id':'partial','registered_patients':5,'registered_episodes':15,'generation_total':180,'wall_seconds':1800}
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'used').mkdir()
            with patch.object(runner,'load',return_value=(Path(d),Path(d),p)):
                self.assertIn('RUN_USED_NO_RESUME',runner.check('Smoke5')['blocker_codes'])

    def test_fake_collection_cannot_promote(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'used').mkdir();(root/'used/collection.json').write_text(json.dumps({'TEST_ONLY':True,'status':'COMPLETED','plan_fingerprint':'x'}))
            (root/'tasks.json').write_text('[]')
            p={'run_id':'used','fingerprint':'x'}
            with patch.object(runner,'load',return_value=(root,root,p)),self.assertRaisesRegex(ValueError,'REAL_COMPLETED'):
                runner.technical_proof('Smoke5')

    def test_first_setup_failure_preserves_registered_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pr=root/'TEST_ONLY_RUNNER';pr.mkdir();pub=root/'public';pub.mkdir()
            for name,value in [('config.json',{}),('bindings50.json',{})]:
                (pr/name).write_text(json.dumps(value))
            (pr/'TEST_ONLY_full50').mkdir();(pr/'TEST_ONLY_full50/history_selection_plans.json').write_text('[]')
            aug=root/'augmentation';aug.mkdir();(aug/'AUGMENTED_CASES.jsonl').write_text(json.dumps({'case_ordinal':1})+'\n')
            tasks=[{'episode_ordinal':i+1,'original_ordinal':1,'case_ref':'Original001','condition':c,'old_development10':True} for i,c in enumerate(('B0','B1','B2'))]
            (pr/'tasks.json').write_text(json.dumps(tasks));(pr/'approval.json').write_text('{}')
            p={'run_id':'used','fingerprint':'test','pins':{},'generation_total':36,'wall_seconds':60,'process_bytes':4*1024**3,'output_bytes':1024**3}
            with patch.object(runner,'load',return_value=(pub,pr,p)),patch.object(runner,'check',return_value={'status':'READY'}),\
                    patch.object(runner,'permission'),patch.object(runner,'PR',pr),patch.object(runner,'AUG',aug),\
                    patch.object(runner,'original_cases',return_value=[{'patient_id':'TEST_ONLY'}]),\
                    patch.object(runner,'make_paired_backend',side_effect=ValueError('TEST_ONLY_SETUP_FAILURE')):
                result=runner.execute('Smoke5',pr/'approval.json',test_only=True)
            saved=json.loads((pr/'used/collection.json').read_text())
            self.assertEqual(result['status'],'PARTIAL_OR_FAILED');self.assertEqual(saved['generation'],0)
            self.assertEqual(len(saved['rows']),3);self.assertTrue(saved['TEST_ONLY'])
            self.assertTrue(all(r['status']=='NOT_ATTEMPTED' for r in saved['rows']))
            self.assertTrue((pr/'used/setup_failure.json').exists())

    def test_completed_but_narrow_headroom_needs_verify(self):
        p={'blocker_codes':[],'comparison_fingerprint':'A','run_id':'never','plan_id':'partial','registered_patients':50,'registered_episodes':150,'generation_total':1800,'wall_seconds':14400}
        def proof(stage):
            if stage=='Smoke5':return {'Verify10_skip_allowed':False,'comparison_fingerprint':'A'}
            raise FileNotFoundError()
        with tempfile.TemporaryDirectory() as d,patch.object(runner,'load',return_value=(Path(d),Path(d),p)),patch.object(runner,'technical_proof',side_effect=proof) as mocked:
            self.assertEqual(runner.check('Full50')['status'],'BLOCKED')
            self.assertEqual(mocked.call_args_list[-1].args[0],'Verify10')

if __name__=='__main__':unittest.main()
