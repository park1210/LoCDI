"""C4-34 TEST_ONLY gates/limits/admission/full frozen batch. No Docker or fitting."""
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_research_gaps import read,ROOT,sha
from app.locdi.new_cohort_execution import *
from scripts.run_new_voi_cohort import generator_shell,generation_argv,generate,owned_container
from app.locdi.voi_source_action_inventory import SYSTEM


class ExecutionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=read(ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json')
        cls.parent=read(ROOT/'outputs/research_evaluation/c4_33_v1/new_cohort_generation_protocol.json')

    def setUp(self):
        self.m=deepcopy(self.model)
        for a in self.m['actions']:
            for o in a['outcomes']:o['likelihood']={d:1/3 for d in self.m['diagnoses']}
        self.visits=[]
        for i in range(2):
            values={}
            for k in self.m['provenance']['action_key_mapping']:
                raw={'PATIENT':'TEST_ONLY_P'+str(i),'ENCOUNTER':'TEST_ONLY_E'+str(i),'VALUE':'1'}
                values[k]={'value':1,'data_type':'numeric','event_id':'TEST_ONLY_'+digest(k)[:8],
                    'available_minute':2,'availability_basis':'RECORDED_TIME_PROXY',
                    'actual_available_at':None,'source_record':raw,'source_record_fingerprint':digest(raw)}
            self.visits.append({'patient_id':'TEST_ONLY_P'+str(i),'visit_id':'TEST_ONLY_E'+str(i),
                'diagnosis':next(iter(self.m['diagnoses'])),'values':values})
        self.plan={'fingerprint':'TEST_PLAN','image_id':'TEST_IMAGE','jar_sha256':'TEST_JAR'}
        self.receipt={'status':'GENERATION_COMPLETED','exit_code':0,'plan_fingerprint':'TEST_PLAN',
            'image_id':'TEST_IMAGE','jar_sha256':'TEST_JAR','seed':2026111601,'attempts':1,
            'approval_sha256':'TEST_APPROVAL','stop_reason':None,
            'raw_files':{'C:/TEST_ONLY/csv/'+n+'.csv':'TEST_HASH' for n in ['patients','encounters','conditions','observations']}}
        self.approval={'approved':True,'scope':'TEST_SCOPE','plan_fingerprint':'TEST_PLAN',
            'evidence_type':'USER_EXPLICIT','user_statement':'TEST_ONLY approval fixture','recorded_at':'TEST_ONLY_TIME'}

    def test_approval_required(self):
        with self.assertRaises(ValueError):approval(None,'TEST_SCOPE','TEST_PLAN')

    def test_explicit_approval_binding(self):approval(self.approval,'TEST_SCOPE','TEST_PLAN')

    def test_generator_without_approval_never_starts(self):
        with patch('scripts.run_new_voi_cohort.verify_plan',return_value={'fingerprint':'TEST_PLAN'}), patch('subprocess.Popen') as start:
            with self.assertRaises(ValueError):generate(None,True)
            start.assert_not_called()

    def test_generation_default_dryrun(self):
        with patch('scripts.run_new_voi_cohort.verify_plan',return_value={'fingerprint':'TEST_PLAN','argv':['TEST_ONLY']}), patch('subprocess.Popen') as start:
            generate(None,False);start.assert_not_called()

    def test_foreign_container_not_owned(self):
        result={'returncode':0,'stdout':'[{"Config":{"Labels":{"locdi.namespace":"FOREIGN"}},"Image":"OTHER"}]'}
        with patch('scripts.run_new_voi_cohort.call',return_value=result):
            with self.assertRaisesRegex(ValueError,'OWNERSHIP'):owned_container()

    def test_unlabeled_container_not_owned(self):
        result={'returncode':0,'stdout':'[{"Config":{},"Image":"OTHER"}]'}
        with patch('scripts.run_new_voi_cohort.call',return_value=result):
            with self.assertRaisesRegex(ValueError,'OWNERSHIP'):owned_container()

    def test_approval_scope(self):
        with self.assertRaises(ValueError):approval(self.approval,'OTHER_SCOPE','TEST_PLAN')

    def test_approval_fingerprint(self):
        with self.assertRaises(ValueError):approval(self.approval,'TEST_SCOPE','OTHER_PLAN')

    def test_no_auto_approval(self):
        self.approval['evidence_type']='GENERATED_BY_CODE'
        with self.assertRaises(ValueError):approval(self.approval,'TEST_SCOPE','TEST_PLAN')

    def test_false_not_truthy_string(self):
        self.approval['approved']='true'
        with self.assertRaises(ValueError):approval(self.approval,'TEST_SCOPE','TEST_PLAN')

    def test_ten_gib_limit(self):self.assertEqual(resource_reason(10*GIB,1),'OUTPUT_LIMIT_REACHED')
    def test_below_disk_limit(self):self.assertIsNone(resource_reason(10*GIB-1,1))
    def test_wall_limit(self):self.assertEqual(resource_reason(0,3600),'WALL_TIME_LIMIT_REACHED')
    def test_tiny_limit_fixture(self):self.assertEqual(resource_reason(9,1,limit_bytes=8),'OUTPUT_LIMIT_REACHED')

    def test_recursive_byte_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'x').write_bytes(b'123');Path(d,'inner').mkdir();Path(d,'inner/y').write_bytes(b'4567')
            self.assertEqual(tree_size(d),7)

    def test_receipt_all_fields(self):verify_receipt(self.receipt,self.plan,lambda pins:{'changed_or_missing':0})

    def test_receipt_partial_blocked(self):
        self.receipt['status']='GENERATION_FAILED_OR_PARTIAL'
        with self.assertRaises(ValueError):verify_receipt(self.receipt,self.plan,lambda p:{'changed_or_missing':0})

    def test_receipt_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError,'HASH'):verify_receipt(self.receipt,self.plan,lambda p:{'changed_or_missing':1})

    def test_receipt_wrong_jar(self):
        self.receipt['jar_sha256']='WRONG'
        with self.assertRaises(ValueError):verify_receipt(self.receipt,self.plan,lambda p:{'changed_or_missing':0})

    def test_receipt_wrong_seed(self):
        self.receipt['seed']=1
        with self.assertRaises(ValueError):verify_receipt(self.receipt,self.plan,lambda p:{'changed_or_missing':0})

    def test_receipt_no_approval(self):
        self.receipt.pop('approval_sha256')
        with self.assertRaises(ValueError):verify_receipt(self.receipt,self.plan,lambda p:{'changed_or_missing':0})

    def test_receipt_missing_export(self):
        self.receipt['raw_files'].pop('C:/TEST_ONLY/csv/conditions.csv')
        with self.assertRaises(ValueError):verify_receipt(self.receipt,self.plan,lambda p:{'changed_or_missing':0})

    def test_pin_no_latest_no_pull(self):
        argv=generation_argv(self.parent)
        self.assertEqual(argv[argv.index('--pull')+1],'never')
        self.assertIn(self.parent['generator_image_id'],argv)
        self.assertNotIn('latest',argv)

    def test_generation_resources_exact(self):
        argv=generation_argv(self.parent)
        for flag,value in [('--cpus','4'),('--memory','6g'),('--memory-swap','6g'),('--network','none')]:
            self.assertEqual(argv[argv.index(flag)+1],value)

    def test_jar_commit_and_single_seed_script(self):
        shell=generator_shell(self.parent)
        self.assertIn(self.parent['generator_jar_sha256'],shell);self.assertIn(self.parent['generator_commit'],shell)
        self.assertIn('-s 2026111601 -cs 2026111601',shell);self.assertIn('timeout 3600',shell)
        self.assertIn('--generate.thread_pool_size=2',shell)

    def test_source_missing_refuses(self):
        self.visits[0]['values'].pop(next(iter(self.visits[0]['values'])))
        with self.assertRaisesRegex(ValueError,'MISSING'):execute_panel(self.m,self.visits,lambda r:None,random_repeats=1,test_only=True)

    def test_overlap_keeps_eligibility_denominators_without_admission(self):
        ledger=[{'complete':True,'patient_id':v['patient_id'],'visit_id':v['visit_id']} for v in self.visits]
        with patch('app.locdi.new_cohort_execution.load_csv',return_value={}), \
             patch('app.locdi.new_cohort_execution.overlap',return_value={'status':'BLOCKED','patient_overlap':1}), \
             patch('app.locdi.new_cohort_execution.adapt',return_value=(self.visits,ledger,{'generated_patients':2})):
            selected,kept,summary=admit('TEST_ONLY',self.m,{'patients':[],'encounters':[],'encounter_fingerprints':{}})
        self.assertEqual(selected,[]);self.assertEqual(len(kept),2)
        self.assertEqual(summary['source_complete_patients_before_overlap_gate'],2)
        self.assertEqual(summary['complete_patients'],0);self.assertEqual(summary['status'],'SOURCE_OVERLAP_BLOCKED')

    def test_one_patient_insufficient(self):
        with self.assertRaisesRegex(ValueError,'INSUFFICIENT'):execute_panel(self.m,self.visits[:1],lambda r:None)

    def test_random32_frozen(self):
        with self.assertRaises(ValueError):execute_panel(self.m,self.visits,lambda r:None,random_repeats=1)

    def test_bootstrap2000_frozen(self):
        with self.assertRaises(ValueError):summarize([],{},bootstrap_repetitions=2)

    def test_no_gold_policy_interface(self):
        self.assertNotIn('reference',inspect.signature(policy_episode).parameters)

    def test_full_testonly_batch_and_analysis(self):
        traces=[]
        records,membership,resources=execute_panel(self.m,self.visits,traces.append,random_repeats=2,test_only=True)
        self.assertEqual(len(records),2*9*6)
        self.assertEqual(set(r['policy'] for r in records),set(POLICIES))
        self.assertTrue(all(len(r['releases'])<=r['budget'] for r in traces))
        result=summarize(records,membership,bootstrap_repetitions=10,test_only=True)
        self.assertTrue(result['test_only']);self.assertIn('voi_h3__global_eig',result['comparison'])
        self.assertEqual(result['aubc']['voi_h3']['patients'],2)
        self.assertIn('5',result['random_monte_carlo'])
        self.assertEqual(result['comparison']['voi_h3__global_eig']['bootstrap']['unit'],'patient cluster, patient macro')
        incomplete=records[:-1]
        with self.assertRaisesRegex(ValueError,'DENOMINATOR'):summarize(incomplete,membership,bootstrap_repetitions=2,test_only=True)

    def test_missing_source_not_simulated(self):
        with self.assertRaises(FileNotFoundError):admit('TEST_ONLY_MISSING',self.m,{'patients':[],'encounters':[],'encounter_fingerprints':{}})

    def test_original_model_readonly(self):
        a=read(ROOT/'outputs/research_evaluation/c4_33_v1/final_m5_artifact_manifest.json')
        self.assertEqual(sha(ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'),a['artifact_sha256'])

    def test_frozen_original_code_hashes(self):
        for name,h in self.parent['code_hashes'].items():self.assertEqual(sha(ROOT/name),h)


if __name__=='__main__':unittest.main()
