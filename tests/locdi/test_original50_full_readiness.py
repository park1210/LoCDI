"""Read-only Original50 protocol tests. No LLM or fitting."""
import unittest
from app.locdi.original_source_inventory import read
from app.locdi.llm_setup import digest
from scripts.prepare_original50_full_readiness import PUB,PR,STAGES,check


class Full50Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=read(PUB/'data_readiness.json');cls.split=read(PUB/'validation_independence.json')
    def test_all_fifty_preserved_and_d2_present(self):
        self.assertEqual(self.data['Original50'],50)
        self.assertEqual(self.data['D2_present'],50)
        self.assertEqual({r['ordinal'] for r in self.data['rows']},set(range(1,51)))
    def test_existing_ten_and_other_forty_are_not_independent_holdout(self):
        self.assertEqual(sum(r['old_development10'] for r in self.data['rows']),10)
        self.assertFalse(self.data['other40_independent_holdout'])
    def test_source_and_memory_boundaries_preserved(self):
        self.assertEqual(self.data['Memory_total'],72590)
        self.assertEqual(self.data['Memory_exact_source_verified'],72590)
        self.assertEqual(self.data['Memory_cutoff_violations'],0)
        self.assertTrue(all(r['D0_unchanged'] and r['D2_provenance_valid'] for r in self.data['rows']))
    def test_no_synthetic_numeric_test_generation(self):
        self.assertTrue(all(r['D2_generated_numeric_tests']==0 for r in self.data['rows']))
        self.assertTrue(all(r['D2_label_conditioned'] and r['D2_release_minute']==0 for r in self.data['rows']))
    def test_bank_has_no_independent_validation_candidate(self):
        self.assertEqual(self.split['Bank_patients'],self.split['Bank_M5_patient_intersection'])
        self.assertEqual(self.split['new_validation_patients'],0)
        self.assertEqual(self.split['U2_validation_M5_overlap'],2)
        self.assertTrue(self.split['patient_level_exclusion_covers_other_encounters'])
    def test_u2_training_and_original50_are_disjoint(self):
        self.assertEqual(self.split['U2_train_validation_overlap'],0)
        self.assertEqual(self.split['U2_Original50_overlap'],0)
        self.assertEqual(self.split['Original50_M5_overlap'],0)
    def test_stage_patient_selection_nested_and_complete(self):
        panels=[]
        for stage,n in STAGES.items():
            name='original50_d2_'+stage.lower()+'_001';tasks=read(PR/name/'tasks.json')
            ids={r['original_ordinal'] for r in tasks};self.assertEqual(len(tasks),n*5);self.assertEqual(len(ids),n)
            self.assertEqual({r['condition'] for r in tasks},{'B0','B1','B2','B3','B4'});panels.append(ids)
        self.assertLessEqual(panels[0],panels[1]);self.assertLessEqual(panels[1],panels[2])
    def test_execution_is_blocked_even_when_source_ready(self):
        for stage,n in STAGES.items():
            r=check(stage);self.assertEqual(r['status'],'BLOCKED');self.assertFalse(r['execute_entry_point_created'])
            self.assertIn('NO_INDEPENDENT_VALIDATION_OUTSIDE_M5',r['blocker_codes'])
            self.assertEqual(r['generation_cap_proposed'],n*5*12)
    def test_terminal_generation_is_in_existing_total_budget(self):
        for stage,n in STAGES.items():
            p=read(PUB/('original50_d2_'+stage.lower()+'_001')/'plan.json')
            self.assertEqual(p['generation_per_episode_including_terminal'],12)
            self.assertEqual(p['B4_terminal_requests_max'],n)
            self.assertTrue(p['terminal_calls_are_included_not_unbudgeted'])
            self.assertEqual(digest({k:v for k,v in p.items() if k!='fingerprint'}),p['fingerprint'])


if __name__=='__main__':unittest.main()
