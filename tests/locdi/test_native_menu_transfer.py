"""Menu-only features and unchanged routing decisions; TEST_ONLY fixtures."""
from copy import deepcopy
from pathlib import Path
import tempfile,json,unittest
from unittest.mock import patch
from app.locdi.native_menu_routing import tokens,public_menu,fit_observation,fitting_feature_scope,registry_type
from app.core.routing_utility import observed_tokens,fit_utility
from app.core.agent_registry import AgentRegistry
from scripts import native_routing_transfer as transfer
from scripts import run_native_menu_full50 as full
from app.locdi.llm_setup import validate_setup
from scripts.prepare_original50_final import PR

class NativeMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry=validate_setup(json.loads((PR/'config.json').read_text(encoding='utf-8')))['prepared']['registry']
        cls.menu=[{'kind':'Test','code':'4548-4','unit':'%'}]

    def fixture(self,positive='cardiology'):
        r=AgentRegistry(self.registry);rows=[]
        for i in range(2):
            for a in r.snapshot()['agents']:
                if a['role']!='specialty':continue
                rows.append({'patient_id':'TEST_ONLY_'+str(i),'decision_id':'D'+str(i),'agent_id':a['agent_id'],
                    'split':'train','observation':fit_observation({},self.menu),'correct':a['department']==positive})
        with fitting_feature_scope():artifact=fit_utility(rows,r,min_samples=2)
        return artifact

    def test_empty_native_core_features_not_changed(self):
        self.assertEqual(observed_tokens({}),[]);self.assertGreater(len(tokens({},self.menu)),0)

    def test_no_hidden_value_or_reference_allowed(self):
        for key in ('value','diagnosis','reference','available_minute','source_record'):
            with self.assertRaises(ValueError):tokens({},[{**self.menu[0],key:'PRIVATE'}])

    def test_metadata_projection_never_value(self):
        m=public_menu([{**self.menu[0],'value':99,'description':'Public name','source_record':{'private':True}}])
        self.assertEqual(m,self.menu)

    def test_no_invented_symptom_or_measurement(self):
        values=tokens({},self.menu)
        self.assertEqual(values,['menu_code_4548-4','menu_kind_test','menu_unit_%'])
        self.assertNotIn('diabetes',values)

    def test_original_function_restored(self):
        import app.core.routing_utility as m
        fn=m.observed_tokens
        with fitting_feature_scope():self.assertNotEqual(m.observed_tokens,fn)
        self.assertIs(m.observed_tokens,fn)

    def test_beta_and_min_samples_unchanged(self):
        artifact=self.fixture();registry=registry_type(self.menu)(self.registry,artifact)
        result=registry.route({},[a['department'] for a in self.registry['agents'] if a['role']=='specialty'],mode='adaptive')
        self.assertEqual(result['selected_specialty'],'cardiology')
        self.assertEqual(result['utility_estimates']['cardiology']['matched_samples'],2)
        self.assertEqual(result['utility_estimates']['cardiology']['score'],0.75)

    def test_tie_keeps_abstention(self):
        artifact=self.fixture(positive='NONE')
        result=registry_type(self.menu)(self.registry,artifact).route({},[a['department'] for a in self.registry['agents'] if a['role']=='specialty'],mode='adaptive')
        self.assertEqual(result['decision_status'],'abstained');self.assertIsNone(result['selected_specialty'])

    def test_missing_menu_support_abstains(self):
        artifact=self.fixture();result=registry_type([])(self.registry,artifact).route({},[a['department'] for a in self.registry['agents'] if a['role']=='specialty'],mode='adaptive')
        self.assertEqual(result['decision_status'],'abstained')

    def test_shared_train_eval_feature_builder(self):
        artifact=self.fixture();expected=tokens({},self.menu)
        self.assertTrue(all(r['tokens']==expected for r in artifact['samples']))

    def test_private_reference_does_not_change_tokens(self):
        self.assertEqual(tokens({'evaluation':{'diagnosis':'HIDDEN'}},self.menu),tokens({},self.menu))

    def test_unverified_evidence_rejected(self):
        with patch.object(transfer,'_VALIDATED_EVIDENCE_FP',None),self.assertRaises(ValueError):transfer.admitted({},self.registry,[],{})

    def test_no_executable_full_plan_without_admission(self):
        with tempfile.TemporaryDirectory() as d,patch.object(full,'PUB',Path(d)):
            self.assertEqual(full.check('Full50')['status'],'BLOCKED')

    def test_no_train_label_for_unresolved(self):
        # Original fitter still rejects unknown labels under the new feature scope.
        a=next(x for x in self.registry['agents'] if x['role']=='specialty')
        row={'patient_id':'TEST_ONLY','decision_id':'D','agent_id':a['agent_id'],'split':'train','observation':fit_observation({},self.menu),'correct':None}
        with fitting_feature_scope(),self.assertRaises(ValueError):fit_utility([row],AgentRegistry(self.registry),min_samples=2)

    def test_fitting_still_requires_actual_approval(self):
        with patch.object(transfer,'fit_check',return_value={'status':'FIT_READY_SEPARATE_USER_APPROVAL_REQUIRED'}),self.assertRaises(ValueError):
            transfer.fit('','')

    def test_missing_actual_fit_and_validation_block_admission(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(transfer,'PR',root),patch.object(transfer,'load_validation',return_value=(root,root,{'run_id':'absent'})):
                result=transfer.admission_check()
            self.assertEqual(result['status'],'BLOCKED')
            self.assertEqual(len(result['blocker_codes']),2)

    def test_source_pin_change_rejected(self):
        from app.locdi.llm_setup import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/transfer.VALIDATION;folder.mkdir();raw=root/'source.csv';raw.write_text('changed')
            p={'pins':{str(raw):'bad_hash'}};p['fingerprint']=digest(p);(folder/'plan.json').write_text(json.dumps(p))
            with patch.object(transfer,'PUB',root),patch.object(transfer,'PR',root),self.assertRaisesRegex(ValueError,'SOURCE_CODE_PIN_CHANGED'):
                transfer.load_validation()

    def test_plain_file_is_not_derived_admission(self):
        from app.locdi.llm_setup import digest
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'evidence.json';value={'schema_version':'locdi.routing.native_menu_transfer.v1'};f.write_text(json.dumps(value))
            with patch.object(transfer,'EVIDENCE',f),patch.object(transfer,'_VALIDATED_EVIDENCE_FP',None),self.assertRaisesRegex(ValueError,'EXACT_DERIVED'):
                transfer.admitted(value,self.registry,[],{})

    def test_validation_summary_cannot_override_core_prediction(self):
        actual={'diagnosis':'A','certainty':'supported','department':'cardiology'}
        row={'result':{'status':'completed','prediction':{**actual,'diagnosis':'B'}},'correctness_proxy':True}
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'actual_core_decisions.json').write_text('[]')
            with patch('app.locdi.common_prediction_contract.extract_core_prediction',return_value=actual),self.assertRaisesRegex(ValueError,'CORE_PREDICTION_MISMATCH'):
                transfer.verify_validation_prediction(row,{'reference':{'label':'A'}},d)

    def test_validation_correctness_must_match_actual_prediction(self):
        actual={'diagnosis':'A','certainty':'supported','department':'cardiology'}
        row={'result':{'status':'completed','prediction':actual},'correctness_proxy':False}
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'actual_core_decisions.json').write_text('[]')
            with patch('app.locdi.common_prediction_contract.extract_core_prediction',return_value=actual),self.assertRaisesRegex(ValueError,'SOURCE_SCORE_MISMATCH'):
                transfer.verify_validation_prediction(row,{'reference':{'label':'A'}},d)

    def test_validation_unresolved_remains_null(self):
        actual={'diagnosis':None,'certainty':'unresolved','department':None}
        row={'result':{'status':'completed','prediction':actual},'correctness_proxy':None}
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'actual_core_decisions.json').write_text('[]')
            with patch('app.locdi.common_prediction_contract.extract_core_prediction',return_value=actual):
                self.assertIsNone(transfer.verify_validation_prediction(row,{'reference':{'label':'A'}},d))

if __name__=='__main__':unittest.main()
