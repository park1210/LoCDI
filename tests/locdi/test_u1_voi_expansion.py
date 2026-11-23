"""TEST_ONLY matched20, abstention authority, Source-time and future U2 gates."""
import unittest
import tempfile,json
from pathlib import Path
from unittest.mock import patch
from datetime import datetime,timedelta,timezone
from scripts.run_u1_voi_expansion import task_list,permission
from scripts.prepare_u2_source_collection import reference_check,permission as collect_permission,check_future_u2
from app.locdi.development_routing_proxy import MARKERS

class ExpansionContracts(unittest.TestCase):
    def test_exact20_paired_and_counterbalanced(self):
        rows=task_list([{'case_ref':'TEST_ONLY_%d'%i} for i in range(10)])
        self.assertEqual(len(rows),20);self.assertEqual([r['episode_ordinal'] for r in rows],list(range(1,21)))
        for i in range(1,11):self.assertEqual({r['condition'] for r in rows if r['panel_ordinal']==i},{'B3','B4'})
        self.assertEqual([r['condition'] for r in rows[:4]],['B3','B4','B4','B3'])
    def authority(self):
        p={'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY','experiment_scope':'U1_D2_MATCHED_B3_B4_FULL10_ONLY','conditions':['B3','B4'],'generation_total':240,'fingerprint':'a'*64,**MARKERS}
        now=datetime.now(timezone.utc);a={**p,'plan_fingerprint':'a'*64,'max_generation':240,'dataset_views':['D0','D2'],'operator_ref':'TEST_ONLY','user_statement':'TEST_ONLY_NOT_ACTUAL_APPROVAL','recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat(),'clinical_release_approved':False}
        for key in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy','accept_authored_utility_transfer','accept_routing_abstention_er_fallback'):a[key]=True
        return p,a
    def test_new_full10_approval(self):
        p,a=self.authority();permission(a,p)
    def test_old24_call_approval_blocked(self):
        p,a=self.authority();a['max_generation']=24
        with self.assertRaises(ValueError):permission(a,p)
    def test_er_fallback_explicit_acceptance_required(self):
        p,a=self.authority();a['accept_routing_abstention_er_fallback']=False
        with self.assertRaises(ValueError):permission(a,p)
    def test_clinical_promotion_blocked(self):
        p,a=self.authority();a['clinical_accuracy']='VALIDATED'
        with self.assertRaises(ValueError):permission(a,p)
    def source(self,date='2020-01-01',labels=('TEST_ONLY',)):
        b={'training_patient_id':'TEST_ONLY_PATIENT','source_provenance':{'condition_rows':[{'source_record':{'PATIENT':'TEST_ONLY_PATIENT','ENCOUNTER':'TEST_ONLY_VISIT','START':date,'DESCRIPTION':label}} for label in labels]}}
        s={'index_encounter':{'encounter_id':'TEST_ONLY_VISIT','start_time':'2020-01-01T12:00:00Z'}}
        return b,s
    def test_future_reference_rejected(self):
        b,s=self.source('2020-01-02');self.assertEqual(reference_check(b,s,['TEST_ONLY']),(None,'FUTURE_REFERENCE'))
    def test_same_day_is_proxy_not_reviewed(self):
        b,s=self.source();label,state=reference_check(b,s,['TEST_ONLY']);self.assertEqual(label,'TEST_ONLY');self.assertIn('UNREVIEWED',state)
    def test_multiple_reference_names_rejected(self):
        b,s=self.source(labels=('TEST_ONLY','OTHER'))
        self.assertIsNone(reference_check(b,s,['TEST_ONLY','OTHER'])[0])
    def test_wrong_patient_or_visit_not_reference(self):
        b,s=self.source();b['source_provenance']['condition_rows'][0]['source_record']['ENCOUNTER']='OTHER'
        self.assertIsNone(reference_check(b,s,['TEST_ONLY'])[0])
    def test_out_of_catalog_rejected(self):
        b,s=self.source();self.assertEqual(reference_check(b,s,[]),(None,'OUTSIDE_CATALOG'))
    def test_u2_future_evaluation_is_not_executable(self):
        r=check_future_u2();self.assertEqual(r['status'],'BLOCKED');self.assertFalse(r['executable_plan_created']);self.assertEqual(r['generation_cap'],0)
    def test_static_snapshot_is_distinct_authority(self):
        p,a=self.authority()
        with self.assertRaises(ValueError):collect_permission(a,{'scope':'U2_SOURCE_STATIC_FIVE_AGENT_COLLECTION_ONLY','fingerprint':'a'*64})
    def test_no_fit_in_current_plans(self):
        from scripts.prepare_u2_source_collection import collect
        from scripts.run_u1_voi_expansion import execute
        self.assertNotIn('fit_utility',collect.__code__.co_names);self.assertNotIn('fit_utility',execute.__code__.co_names)
    def test_unattempted_pairs_remain_unscored(self):
        import scripts.run_u1_voi_expansion as module
        from app.locdi.routing_collector import atomic_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pub=root/'pub';pr=root/'pr';bp=root/'source';run=pr/'TEST_ONLY_RUN'
            for directory in (pub,pr,bp,run):directory.mkdir(parents=True,exist_ok=True)
            panel=[{'case_ref':'TEST_ONLY_%d'%i,'reference':{'accepted_diagnoses':['TEST_ONLY'],'accepted_departments':['TEST_ONLY']}} for i in range(10)]
            tasks=task_list(panel);rows=[{**t,'status':'NOT_ATTEMPTED','request_count':0,'result':None} for t in tasks]
            atomic_json(bp/'panel.json',panel);atomic_json(pr/'tasks.json',tasks)
            atomic_json(run/'collection.json',{'TEST_ONLY':True,'rows':rows,'generation':0,'wall_seconds':0,'plan_fingerprint':'TEST_ONLY'})
            (run/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            with patch.object(module,'load',return_value=(pub,pr,{'run_id':'TEST_ONLY_RUN','fingerprint':'TEST_ONLY'},bp)):
                result=module.analyze()
            self.assertTrue(all(r['unscored_present'] for r in result['pairs']))
            self.assertTrue(all(r['B3_Source_Exact'] is None and r['B4_Source_Exact'] is None for r in result['pairs']))
            self.assertFalse(any(r['CORRECT_TO_INCORRECT'] or r['INCORRECT_TO_CORRECT'] for r in result['pairs']))

if __name__=='__main__':unittest.main()
