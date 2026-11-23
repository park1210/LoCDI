"""TEST_ONLY C4-35 independence facts, no real scoring or new patients."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest
from app.locdi.voi_patient_independence_review import *
from scripts.audit_research_gaps import ROOT,read


class IndependenceFacts(unittest.TestCase):
    def setUp(self):
        self.model=read(ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json')
        self.t={'patients':[(2,{'Id':'TEST_P','BIRTHDATE':'1900-01-01','GENDER':'TEST'})],
            'encounters':[(2,{'Id':'TEST_E','PATIENT':'TEST_P','START':'1900-01-02T00:00:00Z','ENCOUNTERCLASS':'TEST'})],
            'conditions':[(2,{'PATIENT':'TEST_P','ENCOUNTER':'TEST_E','START':'1900-01-02','CODE':'TEST','DESCRIPTION':next(iter(self.model['diagnoses']))})],
            'observations':[]}
        for i,k in enumerate(self.model['provenance']['action_key_mapping'],2):
            _,_,code,unit,_=json.loads(k)
            self.t['observations'].append((i,{'PATIENT':'TEST_P','ENCOUNTER':'TEST_E','DATE':'1900-01-02T00:00:00Z','CODE':code,'UNITS':unit,'TYPE':'numeric','VALUE':'1'}))

    def test_patient_intersection_other_encounter(self):
        ledger=[{'patient_id':'P','visit_id':'E_eval','diagnosis':'D','complete':True}]
        s,i,_=patient_sets([{'new_visit_ids':['E_old']}],[],ledger,{'E_old':'P','E_eval':'P'})
        self.assertEqual(i['A_and_E'],['P'])

    def test_cross_patient_within_new(self):
        s,i,_=patient_sets([], [{'new_visit_ids':['E1','E2']}],[],{'E1':'P1','E2':'P2'})
        self.assertEqual(s['B'],{'P1','P2'})

    def test_unknown_encounter_fails(self):
        with self.assertRaises(ValueError):patient_sets([{'new_visit_ids':['UNKNOWN']}],[],[],{})

    def test_history_identity_removed(self):
        a=full_history(self.t);b=deepcopy(self.t)
        b['patients'][0][1]['Id']='OTHER'
        b['encounters'][0][1].update(Id='OTHER_E',PATIENT='OTHER')
        for name in ['conditions','observations']:
            for _,r in b[name]:r.update(PATIENT='OTHER',ENCOUNTER='OTHER_E')
        self.assertEqual(a['fingerprint'],full_history(b)['fingerprint'])

    def test_history_difference_preserved(self):
        a=full_history(self.t);self.t['observations'][0][1]['VALUE']='2'
        self.assertNotEqual(a['fingerprint'],full_history(self.t)['fingerprint'])

    def test_invalid_birthdate_never_converted(self):
        self.t['patients'][0][1]['BIRTHDATE']='TEST_NOT_A_DATE'
        h=full_history(self.t)
        self.assertEqual(h['birthdate_status'],'INVALID_SOURCE_DATE_NO_CONVERSION')
        self.assertEqual(h['anchor'],'FIRST_EXPORTED_ENCOUNTER')

    def test_source_row_numbers_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'csv').mkdir()
            for name,rs in self.t.items():
                with (root/'csv'/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rs[0][1]));w.writeheader();w.writerows(r for _,r in rs)
            cache=root/'test.sqlite';build_index(root,cache);idx=SourceIndex(cache)
            self.assertEqual(idx.tables('TEST_P'),self.t);idx.close()

    def test_no_cache_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('TEST')
            with self.assertRaises(ValueError):build_index(d,p)

    def test_project_matches_existing_adapter(self):
        from app.locdi.voi_new_cohort_source import adapt
        a=project_patient(self.t,'TEST_ONLY',self.model)[0]
        b=adapt(self.t,'TEST_ONLY',self.model)[0][0]
        for k in ['values','diagnosis','patient_id','visit_id','case_id','rejections']:self.assertEqual(a[k],b[k])

    def test_raw_and_ternary_not_same_identity(self):
        a=project_patient(self.t,'TEST_ONLY',self.model)[0];b=deepcopy(a)
        key=next(iter(b['values']));bounds=self.model['provenance']['binning_thresholds'][key]
        a['values'][key].update(value=bounds[0]-2,source_record={**a['values'][key]['source_record'],'VALUE':str(bounds[0]-2)})
        b['values'][key].update(value=bounds[0]-1,source_record={**b['values'][key]['source_record'],'VALUE':str(bounds[0]-1)})
        x=input_signatures(a,self.model);y=input_signatures(b,self.model)
        self.assertNotEqual(x['raw8'],y['raw8']);self.assertEqual(x['ternary'],y['ternary'])

    def test_missing_action_signature(self):
        v=project_patient(self.t,'TEST_ONLY',self.model)[0];v['values'].pop(next(iter(v['values'])))
        self.assertIsNone(input_signatures(v,self.model))

    def test_whole_patient_quarantine(self):
        c=[{'patient_id':'P','visit_id':'E1'},{'patient_id':'P','visit_id':'E2'},{'patient_id':'Q','visit_id':'E3'}]
        p=conservative_proposal(c,{'P'},set(),set())
        self.assertEqual(p['proposed'],[c[2]]);self.assertEqual(p['excluded_visits'],2);self.assertEqual(p['admitted_visits'],0)

    def test_global_blocker(self):
        p=conservative_proposal([{'patient_id':'P','visit_id':'E'}],set(),set(),set(),['TEST_BLOCKER'])
        self.assertEqual(p['proposed_visits'],0)
