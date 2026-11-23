"""TEST_ONLY structural source fixtures, never clinical data or verdicts."""
import ast
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.dataset_suitability import symptom_asof, reference_timing, split_relationships, audit_raw, millis
from app.locdi.llm_setup import ROOT, digest
from scripts.audit_dataset_suitability import save_audit, main


class DatasetSuitabilityContracts(unittest.TestCase):
    def test_future_symptoms_and_final_resolved_state_never_replace_asof_history(self):
        symptoms={'TEST_ONLY_SYMPTOM':{'sources':{'TEST_ONLY_MODULE':{
            'resolved':True,'lastUpdateTime':300,'timeInfos':{
                'a':{'time':100,'value':1},'b':{'time':300,'value':0}}}}}}
        before=deepcopy(symptoms)
        rows,count=symptom_asof(symptoms,[('TEST_ONLY_A',99),('TEST_ONLY_B',100),('TEST_ONLY_C',200),('TEST_ONLY_D',300)])
        self.assertEqual([rows[x]['active_symptom_name_count'] for x in rows],[0,1,1,0])
        self.assertEqual(rows['TEST_ONLY_B']['active_updated_at_encounter_count'],1)
        self.assertEqual(rows['TEST_ONLY_C']['source_updates_after_cutoff'],1)
        self.assertEqual(count,2);self.assertEqual(symptoms,before)

    def test_multiple_modules_do_not_make_duplicate_symptom_names(self):
        value={'TEST_ONLY_S':{'sources':{module:{'timeInfos':{'x':{'time':100,'value':1}}}
                                      for module in ('TEST_ONLY_M1','TEST_ONLY_M2')}}}
        rows,_=symptom_asof(value,[('TEST_ONLY_E',101)])
        self.assertEqual(rows['TEST_ONLY_E']['active_symptom_name_count'],1)

    def test_reference_dates_are_not_clinical_gold_or_invented_midnight_release(self):
        self.assertEqual(reference_timing('2020-01-05','2020-01-01T12:00:00Z'),'AFTER_ENCOUNTER_DATE')
        self.assertEqual(reference_timing('2020-01-01','2020-01-01T12:00:00Z'),'SAME_DATE_TIME_UNRESOLVED')
        self.assertEqual(reference_timing('2019-12-31','2020-01-01T12:00:00Z'),'BEFORE_ENCOUNTER_DATE')
        self.assertEqual(reference_timing(None,'2020-01-01T12:00:00Z'),'UNKNOWN')
        with self.assertRaises(ValueError):millis('2020-01-01')

    def test_split_excludes_all_encounters_of_patient_and_reports_cross_module_overlap(self):
        sets={'holdout':{'TEST_ONLY_H'},'bank':{'TEST_ONLY_B'},'voi_train':{'TEST_ONLY_B'},
            'routing_validation':{'TEST_ONLY_B'},'known_development':{'TEST_ONLY_H','TEST_ONLY_B'},
            'prepared_context_pool':{'TEST_ONLY_C'},'prepared_memory_pool':{'TEST_ONLY_C','TEST_ONLY_N'}}
        before=deepcopy(sets)
        result,candidates=split_relationships({'TEST_ONLY_H','TEST_ONLY_B','TEST_ONLY_C','TEST_ONLY_N'},sets)
        self.assertEqual(candidates,{'TEST_ONLY_C','TEST_ONLY_N'})
        self.assertEqual(result['new_outside_prepared_context_pool'],1)
        self.assertEqual(result['new_outside_prepared_context_and_memory_pools'],0)
        self.assertEqual(result['approved_independent_validation_patients'],0)
        self.assertIn({'left':'routing_validation','right':'voi_train','intersection_patients':1},result['overlaps'])
        self.assertEqual(sets,before)

    def write_csv(self,root,name,fields,rows):
        path=root/'csv'/name;path.parent.mkdir(exist_ok=True)
        with path.open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    def fixture(self,root):
        p='TEST_ONLY_PATIENT';a='TEST_ONLY_PAST';b='TEST_ONLY_CURRENT'
        self.write_csv(root,'patients.csv',['Id'],[{'Id':p}])
        self.write_csv(root,'encounters.csv',['Id','START','STOP','PATIENT','ENCOUNTERCLASS','DESCRIPTION','REASONDESCRIPTION'],[
            {'Id':a,'START':'2020-01-01T00:00:00Z','PATIENT':p,'ENCOUNTERCLASS':'wellness','DESCRIPTION':'TEST_ONLY'},
            {'Id':b,'START':'2020-01-02T00:00:00Z','PATIENT':p,'ENCOUNTERCLASS':'emergency','DESCRIPTION':'TEST_ONLY'}])
        self.write_csv(root,'observations.csv',['DATE','PATIENT','ENCOUNTER','CATEGORY','CODE','DESCRIPTION','VALUE','UNITS','TYPE'],[
            {'DATE':'2020-01-01T00:00:00Z','PATIENT':p,'ENCOUNTER':a,'CATEGORY':'vital-signs','CODE':'TEST_ONLY_V','DESCRIPTION':'TEST_ONLY','VALUE':'0','TYPE':'numeric'},
            {'DATE':'2020-01-02T01:00:00Z','PATIENT':p,'ENCOUNTER':b,'CATEGORY':'laboratory','CODE':'TEST_ONLY_L','DESCRIPTION':'TEST_ONLY','VALUE':'TEST_ONLY','TYPE':'text'},
            {'DATE':'2020-01-02T00:00:00Z','PATIENT':p,'ENCOUNTER':b,'CATEGORY':'survey','CODE':'TEST_ONLY_S','DESCRIPTION':'TEST_ONLY Reported','VALUE':'','TYPE':'text'}])
        self.write_csv(root,'conditions.csv',['START','STOP','PATIENT','ENCOUNTER','SYSTEM','CODE','DESCRIPTION'],[
            {'START':'2020-01-06','PATIENT':p,'ENCOUNTER':b,'SYSTEM':'TEST_ONLY','CODE':'TEST_ONLY_D','DESCRIPTION':'TEST_ONLY'}])
        for file,fields in [('procedures.csv',['PATIENT','ENCOUNTER','DESCRIPTION']),('careplans.csv',['PATIENT','ENCOUNTER','DESCRIPTION']),
            ('imaging_studies.csv',['PATIENT','ENCOUNTER','MODALITY_DESCRIPTION']),('providers.csv',['SPECIALITY']),('claims.csv',['REFERRINGPROVIDERID'])]:
            self.write_csv(root,file,fields,[])
        (root/'json').mkdir()
        (root/'json/TEST_ONLY.json').write_text(json.dumps({'attributes':{'id':p},'symptoms':{
            'TEST_ONLY_S':{'sources':{'TEST_ONLY_M':{'resolved':True,'timeInfos':{'1':{
                'time':millis('2020-01-02T00:00:00Z'),'value':1},'2':{
                'time':millis('2020-01-03T00:00:00Z'),'value':0}}}}}},
            'record':{'encounters':[{'uuid':a,'start':millis('2020-01-01T00:00:00Z')},
                                    {'uuid':b,'start':millis('2020-01-02T00:00:00Z')}]}}),encoding='utf-8')
        return p

    def test_raw_fixture_keeps_zero_empty_and_future_results_distinct(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root=Path(directory);pid=self.fixture(root)
            before={p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            report,private=audit_raw(root,{})
            emergency=next(g for g in report['groups'] if g['scope']=='all' and g['encounter_class']=='emergency')
            self.assertEqual(emergency['encounter_coverage']['value_laboratory'],1)
            self.assertEqual(emergency['encounter_coverage']['value_laboratory_at_or_before_start'],0)
            self.assertEqual(emergency['encounter_coverage']['reported_label_candidates'],0)
            self.assertEqual(emergency['encounter_coverage']['active_symptom_name_count'],1)
            self.assertEqual(emergency['encounter_coverage']['past_recorded_observations'],1)
            self.assertEqual(emergency['totals']['condition_AFTER_ENCOUNTER_DATE'],1)
            vital=next(t for t in report['test_targets'] if t['code']=='TEST_ONLY_V')
            self.assertEqual(vital['empty_values'],0)
            self.assertEqual(private['patients'][0]['patient_id'],pid)
            self.assertEqual(len(private['patients'][0]['encounters']),2)
            self.assertIsNone(private['patients'][0]['assigned_new_split'])
            self.assertNotIn(pid,json.dumps(report))
            self.assertFalse(report['research_readiness'])
            self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_missing_source_not_found_does_not_become_complete(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            report,_=audit_raw(Path(directory),{},inspect_json=True)
        self.assertIn('csv/observations.csv',report['missing_sources'])
        self.assertIn('json/*.json',report['missing_sources'])
        self.assertEqual(report['status'],'PARTIAL_SOURCE_MISSING')

    def test_duplicate_encounter_rejected_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root=Path(directory);self.fixture(root)
            p=root/'csv/encounters.csv'
            with p.open(encoding='utf-8') as s:lines=s.readlines()
            with p.open('a',encoding='utf-8') as s:s.write(lines[-1])
            with self.assertRaisesRegex(ValueError,'DUPLICATE_ENCOUNTER_ID'):audit_raw(root,{})

    def test_save_pairs_separates_private_manifest_and_never_overwrites(self):
        public={'schema_version':'TEST_ONLY','clinical_verdicts_generated':0}
        private={'patient_id':'TEST_ONLY_PRIVATE','public_summary_fingerprint':digest(public)}
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root=Path(directory)
            with patch('scripts.audit_dataset_suitability.ROOT',root):
                a,b=save_audit(public,private,'TEST_ONLY')
                self.assertEqual(json.loads(a.read_text(encoding='utf-8')),public)
                self.assertEqual(json.loads(b.read_text(encoding='utf-8')),private)
                self.assertNotIn('TEST_ONLY_PRIVATE',a.read_text(encoding='utf-8'))
                original=a.read_bytes(),b.read_bytes()
                with self.assertRaises(FileExistsError):save_audit(public,private,'TEST_ONLY')
                with self.assertRaises(ValueError):save_audit(public,private,'../escape')
                self.assertEqual((a.read_bytes(),b.read_bytes()),original)

    def test_pair_tampering_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root=Path(directory)
            with patch('scripts.audit_dataset_suitability.ROOT',root):
                with self.assertRaises(ValueError):save_audit({'TEST_ONLY':1},{'public_summary_fingerprint':'wrong'},'TEST_ONLY')
            self.assertFalse((root/'outputs').exists())

    def test_default_cli_never_saves_and_errors_do_not_expose_evidence(self):
        stub=({'status':'AUDIT_COMPLETE_NOT_CLINICALLY_REVIEWED','raw_patients':0,'raw_encounters':0,'split':{},'missing_sources':[]},{})
        with patch('scripts.audit_dataset_suitability.audit_suitability',return_value=stub),patch('scripts.audit_dataset_suitability.save_audit') as save,patch('builtins.print'):
            self.assertEqual(main([]),0);save.assert_not_called()
        with patch('scripts.audit_dataset_suitability.audit_suitability',side_effect=ValueError('TEST_ONLY_PRIVATE')),patch('builtins.print') as output:
            self.assertEqual(main([]),2);self.assertNotIn('TEST_ONLY_PRIVATE',str(output.call_args))

    def test_no_model_fitting_probability_or_core_execution_api(self):
        tree=ast.parse((ROOT/'app/locdi/dataset_suitability.py').read_text(encoding='utf-8'))
        calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {'chat_json','fit_utility','posterior','plan_voi','run_case','run_cases','run_integrated'})


if __name__=='__main__':unittest.main()
