"""TEST_ONLY CSV source/holdout contracts; fixtures are not clinical evaluation."""
from copy import deepcopy
import inspect
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.llm_setup import digest
from app.locdi.voi_source_action_inventory import action_key
from app.locdi.voi_information_improvement import bucket
from app.locdi.voi_new_cohort_source import *
from app.locdi.voi_new_cohort_evaluation import *
from app.locdi.dynamic_horizon_source_evaluation import PlannerMeter, normalized_trace
from app.locdi.voi_source_replay_environment import SourceReplayEnvironment
from app.locdi.voi_sequential_policy_evaluation import global_eig_order
from scripts.audit_research_gaps import read, ROOT


class NewCohortContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.evaluate_voi_sequential_policies import load_panel
        _, _, _, models, vals, _ = load_panel()
        cls.model = models[1, 'M5']
        cls.values = vals[1]

    def setUp(self):
        self.m = deepcopy(self.model)
        self.tables = {'patients': [(2, {'Id':'TEST_ONLY_P'})],
            'encounters': [(2, {'Id':'TEST_ONLY_E', 'PATIENT':'TEST_ONLY_P',
                'START':'2026-11-16T00:00:00Z', 'ENCOUNTERCLASS':'TEST_ONLY'})],
            'conditions': [(2, {'PATIENT':'TEST_ONLY_P','ENCOUNTER':'TEST_ONLY_E',
                'START':'2026-11-16', 'SYSTEM':'TEST_ONLY', 'CODE':'TEST_CODE',
                'DESCRIPTION':next(iter(self.m['diagnoses']))})], 'observations': []}
        import json
        for i, k in enumerate(self.m['provenance']['action_key_mapping'], 2):
            kind, system, code, unit, resource = json.loads(k)
            self.tables['observations'].append((i, {'PATIENT':'TEST_ONLY_P','ENCOUNTER':'TEST_ONLY_E',
                'DATE':'2026-11-16T00:00:00Z', 'CODE':code,'UNITS':unit,'TYPE':'numeric',
                'VALUE':str(self.m['provenance']['binning_thresholds'][k][0])}))

    def adapted(self):
        return adapt(self.tables, Path('TEST_ONLY_RAW'), self.m)

    def test_complete8(self):
        visits, ledger, s = self.adapted()
        self.assertEqual((s['generated_patients'], s['all_visits'],s['complete_panel_visits']),(1,1,1))
        self.assertEqual(len(visits[0]['values']),8)

    def test_pre1970_portable_epoch(self):
        self.assertLess(epoch('1900-01-01T00:00:00Z'),0)
        self.assertEqual(epoch('1970-01-01T00:00:00Z'),0)

    def test_pre1970_complete_projection(self):
        self.tables['encounters'][0][1]['START']='1900-01-01T00:00:00Z'
        self.tables['conditions'][0][1]['START']='1900-01-01'
        for _,r in self.tables['observations']:r['DATE']='1900-01-01T00:00:00Z'
        self.assertEqual(self.adapted()[2]['complete_panel_visits'],1)

    def test_no_source_mutation(self):
        before = deepcopy(self.tables)
        self.adapted()
        self.assertEqual(before,self.tables)

    def test_patient_overlap(self):
        self.assertEqual(overlap(self.tables,['TEST_ONLY_P'],[],[])['status'],'BLOCKED')

    def test_encounter_overlap(self):
        self.assertEqual(overlap(self.tables,[],['TEST_ONLY_E'],[])['status'],'BLOCKED')

    def test_content_duplicate_after_id_change(self):
        fp = list(encounter_fingerprints(self.tables).values())
        self.assertEqual(overlap(self.tables,[],[],fp)['content_duplicate_encounters'],1)

    def test_clean_overlap(self):
        self.assertEqual(overlap(self.tables,[],[],[])['status'],'PASS')

    def test_duplicate_patient(self):
        self.tables['patients'] *= 2
        with self.assertRaisesRegex(ValueError,'DUPLICATE'): self.adapted()

    def test_duplicate_encounter(self):
        self.tables['encounters'] *= 2
        with self.assertRaisesRegex(ValueError,'DUPLICATE'): self.adapted()

    def test_wrong_patient_link(self):
        self.tables['observations'][0][1]['PATIENT']='FOREIGN'
        with self.assertRaisesRegex(ValueError,'PATIENT_LINK'): self.adapted()

    def test_wrong_encounter_link(self):
        self.tables['observations'][0][1]['ENCOUNTER']='FOREIGN'
        with self.assertRaisesRegex(ValueError,'ENCOUNTER_LINK'): self.adapted()

    def test_diagnosis_outside_universe(self):
        self.tables['conditions'][0][1]['DESCRIPTION']='TEST_ONLY_UNKNOWN'
        v,l,s=self.adapted()
        self.assertEqual((len(v),s['diagnosis_eligible_visits'],s['complete_panel_visits']),(0,0,0))

    def test_ambiguous_universe(self):
        r=deepcopy(self.tables['conditions'][0][1]);r['DESCRIPTION']=list(self.m['diagnoses'])[1]
        self.tables['conditions'].append((3,r))
        self.assertEqual(self.adapted()[2]['diagnosis_eligible_visits'],0)

    def test_missing_action_blocks_complete(self):
        self.tables['observations'].pop()
        self.assertEqual(self.adapted()[2]['complete_panel_visits'],0)

    def test_wrong_unit_not_alias(self):
        self.tables['observations'][0][1]['UNITS']='TEST_WRONG_UNIT'
        self.assertEqual(self.adapted()[2]['complete_panel_visits'],0)

    def test_explicit_system_not_silently_aliased(self):
        self.tables['observations'][0][1]['SYSTEM']='TEST_OTHER_SYSTEM'
        self.assertEqual(self.adapted()[2]['complete_panel_visits'],0)

    def test_nonfinite_not_outcome(self):
        self.tables['observations'][0][1]['VALUE']='NaN'
        self.assertEqual(self.adapted()[2]['complete_panel_visits'],0)

    def test_categorical_not_encoded(self):
        self.tables['observations'][0][1]['TYPE']='text'
        self.assertEqual(self.adapted()[2]['complete_panel_visits'],0)

    def test_future_46minute_not_admitted(self):
        for _,r in self.tables['observations']:r['DATE']='2026-11-16T00:46:00Z'
        self.assertEqual(self.adapted()[2]['at_least1_action_visits'],0)

    def test_test45minute_proxy(self):
        for _,r in self.tables['observations']:
            if r['CODE']!='72514-3':r['DATE']='2026-11-16T00:45:00Z'
        v,l,s=self.adapted()
        self.assertEqual(s['complete_panel_visits'],1)
        self.assertTrue(all(e['actual_available_at'] is None for e in v[0]['values'].values()))

    def test_ask_after_arrival_not_invented(self):
        for _,r in self.tables['observations']:
            if r['CODE']=='72514-3':r['DATE']='2026-11-16T00:01:00Z'
        self.assertEqual(self.adapted()[2]['complete_panel_visits'],0)

    def test_source_row_hash(self):
        v,_,_=self.adapted()
        for e in v[0]['values'].values():self.assertEqual(digest(e['source_record']),e['source_record_fingerprint'])

    def test_conflicting_same_time_rejected(self):
        line,r=self.tables['observations'][0];r=deepcopy(r);r['VALUE']=str(float(r['VALUE'])+10)
        self.tables['observations'].append((99,r))
        self.assertEqual(self.adapted()[2]['complete_panel_visits'],0)

    def test_exact_stored_encoding(self):
        for bs in self.m['provenance']['binning_thresholds'].values():
            self.assertEqual(bucket(bs[0],bs),'0')
            self.assertGreater(int(bucket(math.nextafter(bs[0],math.inf),bs)),0)

    def test_fit_reject_evaluation_patient_before_selection(self):
        with self.assertRaisesRegex(ValueError,'OVERLAP'):
            fit_final(self.m,[{'patient_id':'TEST_ONLY_P'}],{},['TEST_ONLY_P'])

    def test_training_prior_one_per_visit(self):
        d=next(iter(self.m['diagnoses']));v={'patient_id':'P','visit_id':'E','diagnosis':d,'values':{'BASE':{}}}
        p=dev_selection_prior(self.m,[v],['BASE'])
        self.assertEqual(p[d],2/16)

    def test_no_selected_observation_not_prior_visit(self):
        v={'patient_id':'P','visit_id':'E','diagnosis':next(iter(self.m['diagnoses'])),'values':{}}
        with self.assertRaises(ValueError):dev_selection_prior(self.m,[v],['BASE'])

    def test_prior_duplicate_visit_rejected(self):
        v={'patient_id':'P','visit_id':'E','diagnosis':next(iter(self.m['diagnoses'])),'values':{'BASE':{}}}
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):dev_selection_prior(self.m,[v,v],['BASE'])

    def test_all9_real_policy_paths(self):
        for p in POLICIES:
            meter=PlannerMeter(cached=True)
            order=global_eig_order(self.m,meter)
            r=policy_episode(self.m,self.values,p,1,meter,order,ordinal=1)
            self.assertEqual(r['status'],'completed')
            self.assertLessEqual(r['actual_actions'],1)
            self.assertNotIn('score',r)

    def test_no_reference_in_policy_signature(self):
        self.assertNotIn('reference',inspect.signature(policy_episode).parameters)

    def test_score_only_after_replay(self):
        r=policy_episode(self.m,self.values,'fixed',1,PlannerMeter(),[],ordinal=1)
        scored=score_after_execution(r,next(iter(self.m['diagnoses'])))
        self.assertIn('score',scored);self.assertNotIn('score',r)

    def test_deterministic_random(self):
        a=policy_episode(self.m,self.values,'random',2,PlannerMeter(),[],ordinal=1)
        b=policy_episode(self.m,self.values,'random',2,PlannerMeter(),[],ordinal=1)
        self.assertEqual(a,b)

    def test_bad_random_repeat(self):
        with self.assertRaises(ValueError):policy_episode(self.m,self.values,'random',1,PlannerMeter(),[],ordinal=1,repeat=32)

    def test_budget0_all_policies(self):
        for p in POLICIES:
            r=policy_episode(self.m,self.values,p,0,PlannerMeter(),[],ordinal=1)
            self.assertEqual(r['actual_actions'],0)

    def test_native_stop_unchanged(self):
        for a in self.m['actions']:
            for o in a['outcomes']:o['likelihood']={d:1/3 for d in self.m['prior']}
        for p in ('voi_h1','voi_h2','voi_h3','dynamic'):
            r=policy_episode(self.m,self.values,p,2,PlannerMeter(),[],ordinal=1)
            self.assertEqual(r['stop_reason'],'NATIVE_STOP')

    def test_selected_release_no_gold(self):
        env=SourceReplayEnvironment(self.m,self.values,1)
        r=env.acquire(self.m['actions'][0]['id'])
        self.assertEqual(set(r),{'action_id','outcome_id','released_at_minute'})

    def test_repeated_action_rejected(self):
        env=SourceReplayEnvironment(self.m,self.values,2);aid=self.m['actions'][0]['id'];env.acquire(aid)
        with self.assertRaisesRegex(ValueError,'REPEATED'):env.acquire(aid)

    def test_no_missing_outcome_synthesis(self):
        with self.assertRaisesRegex(ValueError,'MISSING'):policy_episode(self.m,{},'fixed',1,PlannerMeter(),[],ordinal=1)

    def test_fixed_protocol(self):
        p=evaluation_protocol('MODEL','SELECTION')
        self.assertEqual(p['primary']['comparison'],['voi_h3','global_eig'])
        self.assertEqual(p['bootstrap']['resamples'],2000)
        self.assertEqual(p['random']['repeats'],32)
        self.assertEqual(p['planner']['epsilon'],1e-12)

    def test_training_protocol_one_estimate(self):
        p=training_protocol('A','B','C')
        self.assertEqual(p['maximum_estimations'],1)
        self.assertFalse(p['holdout_access_before_freeze'])

    def test_source_loader_missing_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):load_csv(d)

    def test_old_dynamic_spec_unchanged(self):
        self.assertEqual(policy_spec(),read(ROOT/'outputs/research_evaluation/c4_19_v1/dynamic_horizon_policy_spec.json'))


if __name__=='__main__':unittest.main()
