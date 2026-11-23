"""TEST_ONLY nonclinical coded-domain/task separation and paired measurement."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import tempfile
import unittest

from app.locdi.llm_setup import digest
from app.locdi.source_enrichment import iso_ms
from app.locdi.retrospective_routing import (FIXED,protocol,label,coverage,measurements,public_input,validate_mapping)
from app.locdi.abstention_routing import compare
from app.locdi.routing_two_decision_collection import verify_pins
from app.locdi.dataset_freeze import sha
from scripts.enrich_full_cohort_sources import write
from test_abstention_routing import fixture


def sample():
    case={'case_ref':'TEST_ONLY_CASE','dataset':'TEST_ONLY','patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_V',
          'decision_time':iso_ms(120000)}
    encounter={'patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_V','start':iso_ms(60000),
               'stop':iso_ms(240000),'completed':True}
    mapping={'mappings':[{'system':'SNOMED-CT','code':c,'specialties':[d],
        'code_identity_verified':True,'specialty_scope_verified':True,'mapping_id':'TEST_ONLY_'+c} for c,d in FIXED.items()]}
    value={'start':180000,'codes':[{'system':'SNOMED-CT','code':'44054006'}]}
    fact={'source_verified':True,'source_resource_type':'Condition','patient_reference':'TEST_ONLY_P',
        'visit_reference':'TEST_ONLY_V','source_value':value,'source_value_fingerprint':digest(value),
        'recorded_at':iso_ms(180000),'fact_ref':'TEST_ONLY_F','source_file_sha256':'a'*64,'source_field':'/TEST_ONLY'}
    return case,encounter,mapping,fact


class RetrospectiveTask(unittest.TestCase):
    def setUp(self):self.c,self.e,self.mapping,self.f=sample()
    def result(self,facts=None):return label(self.c,[self.f] if facts is None else facts,self.e,self.mapping,iso_ms(300000))
    def change_value(self,**values):
        self.f['source_value'].update(values);self.f['source_value_fingerprint']=digest(self.f['source_value'])
    def test_exact_mapping_and_four_times(self):
        r=self.result();self.assertEqual(r['confirmed_positive_domains'],['endocrinology'])
        self.assertEqual(r['router_input_time'],iso_ms(120000));self.assertEqual(r['extracted_at'],iso_ms(300000))
        self.assertEqual(r['source_facts'][0]['recorded_onset'],iso_ms(180000));self.assertEqual(r['encounter_stop'],iso_ms(240000))
        self.assertEqual(r['future_outcome_conditions'],1)
    def test_system_not_inferred_from_numeric_code(self):
        self.change_value(codes=[{'system':'OTHER','code':'44054006'}]);self.assertEqual(self.result()['coverage'],'UNKNOWN')
    def test_no_descendant_or_synonym_mapping(self):
        self.change_value(codes=[{'system':'SNOMED-CT','code':'TEST_ONLY_OTHER','display':'diabetes'}])
        self.assertEqual(self.result()['confirmed_positive_domains'],[])
    def test_wrong_patient(self):
        self.f['patient_reference']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):self.result()
    def test_wrong_visit(self):
        self.f['visit_reference']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):self.result()
    def test_wrong_encounter_binding(self):
        self.e['visit_id']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):self.result()
    def test_unverified_source(self):
        self.f['source_verified']=False
        with self.assertRaisesRegex(ValueError,'VERIFICATION'):self.result()
    def test_source_fingerprint_mismatch(self):
        self.f['source_value_fingerprint']='b'*64
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):self.result()
    def test_native_start_not_recorded_time(self):
        self.f['recorded_at']=iso_ms(190000)
        with self.assertRaisesRegex(ValueError,'ONSET'):self.result()
    def test_non_condition_rejected(self):
        self.f['source_resource_type']='Observation'
        with self.assertRaises(ValueError):self.result()
    def test_after_encounter_not_outcome(self):
        self.change_value(start=300000);self.f['recorded_at']=iso_ms(300000)
        r=self.result();self.assertEqual(r['outside_window_conditions'],1);self.assertEqual(r['confirmed_positive_domains'],[])
    def test_before_encounter_not_prevalence_inference(self):
        self.change_value(start=1000);self.f['recorded_at']=iso_ms(1000)
        self.assertEqual(self.result()['outside_window_conditions'],1)
    def test_missing_stop_no_fabricated_horizon(self):
        self.e['stop']=None;self.assertFalse(self.result()['window_verified'])
    def test_inverted_window(self):
        self.e['stop']=iso_ms(1000);self.assertEqual(self.result()['coverage'],'UNKNOWN')
    def test_unfinished_encounter(self):
        self.e['completed']=False;self.assertFalse(self.result()['window_verified'])
    def test_extraction_before_end(self):
        with self.assertRaisesRegex(ValueError,'EXTRACTION'):label(self.c,[self.f],self.e,self.mapping,iso_ms(200000))
    def test_missing_onset_unknown(self):
        self.f['recorded_at']=None;r=self.result();self.assertEqual(r['missing_time_conditions'],1)
    def test_multidomain_union(self):
        second=deepcopy(self.f);second['fact_ref']='TEST_ONLY_SECOND'
        second['source_value']['codes'][0]['code']='414545008';second['source_value_fingerprint']=digest(second['source_value'])
        r=self.result([self.f,second]);self.assertEqual(r['confirmed_positive_domains'],['cardiology','endocrinology'])
        self.assertEqual(coverage([r])['multiple_domain_cases'],1)
    def test_partial_does_not_create_negative(self):
        second=deepcopy(self.f);second['source_value']['codes']=[];second['source_value_fingerprint']=digest(second['source_value'])
        r=self.result([self.f,second]);self.assertEqual(r['coverage'],'PARTIAL_POSITIVE')
        self.assertEqual(r['negative_domains'],[]);self.assertEqual(len(r['unknown_domains']),4)
    def test_empty_inventory_not_negative(self):self.assertEqual(self.result([])['coverage'],'UNKNOWN')
    def test_mapping_cannot_be_expanded(self):
        self.mapping['mappings'][0]['code']='TEST_ONLY_NEW'
        with self.assertRaisesRegex(ValueError,'FIXED'):self.result()
    def test_mapping_evidence_required(self):
        self.mapping['mappings'][0]['code_identity_verified']=False
        with self.assertRaisesRegex(ValueError,'EVIDENCE'):self.result()
    def test_function_does_not_mutate_sources(self):
        before=deepcopy([self.c,self.e,self.mapping,self.f]);self.result()
        self.assertEqual(before,[self.c,self.e,self.mapping,self.f])
    def test_bounds_inclusive(self):
        for t in [60000,240000]:
            self.change_value(start=t);self.f['recorded_at']=iso_ms(t);self.assertEqual(self.result()['in_window_conditions'],1)
    def test_protocol_no_clinical_reference(self):
        self.assertFalse(protocol()['clinical_reference']);self.assertEqual(protocol()['llm_generation'],0)


class PairedMeasurement(unittest.TestCase):
    def setUp(self):self.reg,self.cases=fixture()
    def labels(self):return [{'case_ref':c['case_ref'],'confirmed_positive_domains':['cardiology'] if i==0 else [],'coverage':'PARTIAL_POSITIVE' if i==0 else 'UNKNOWN'} for i,c in enumerate(self.cases)]
    def traces(self,condition='D_abstention_aware'):return [t for t in compare(self.cases,self.reg) if t['condition']==condition]
    def test_partial_metrics_blocked_and_unknown_not_wrong(self):
        r=measurements(self.labels(),self.traces());self.assertEqual(r['confirmed_positive_tag_match_count'],1)
        self.assertEqual(r['unconfirmed_selection_count'],1);self.assertEqual(r['domain_disagreement_count']['status'],'NOT_EVALUATED')
        self.assertIsNone(r['source_domain_agreement']['value']);self.assertEqual(r['abstained'],2)
    def test_missing_utility_not_normal_abstention(self):
        r=measurements(self.labels(),self.traces('C_adaptive_accuracy'))
        self.assertEqual(r['missing_utility'],4);self.assertEqual(r['abstained'],0);self.assertEqual(r['selection_coverage']['status'],'NOT_APPLICABLE')
    def test_unknown_reference_count(self):self.assertEqual(measurements(self.labels(),self.traces())['unknown_no_positive_reference_count'],3)
    def test_missing_cohort_rejected(self):
        with self.assertRaisesRegex(ValueError,'FULL_PAIRED'):measurements(self.labels(),self.traces()[:-1])
    def test_reproducible_and_same_input(self):
        a=compare(self.cases,self.reg);self.assertEqual(a,compare(self.cases,self.reg))
        for c in self.cases:
            ts=[t for t in a if t['case_ref']==c['case_ref']]
            for key in ['input_fingerprint','memory_fingerprint','candidate_fingerprint','registry_fingerprint']:
                self.assertEqual(len({t[key] for t in ts}),1)
    def test_source_pin_tamper_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY_SOURCE';p.write_text('TEST_ONLY',encoding='utf-8');pins={str(p):sha(p)}
            verify_pins(pins);p.write_text('TEST_ONLY_CHANGED',encoding='utf-8')
            with self.assertRaises(ValueError):verify_pins(pins)
    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.json';write(p,{'TEST_ONLY':1});before=sha(p)
            with self.assertRaises(FileExistsError):write(p,{'TEST_ONLY':2})
            self.assertEqual(sha(p),before)
    def test_same_patient_cross_split_blocked(self):
        self.cases[0]['split']='train';self.cases[1]['patient_id']=self.cases[0]['patient_id']
        with self.assertRaisesRegex(ValueError,'SPLIT_OVERLAP'):compare(self.cases,self.reg)
    def test_nested_gold_and_future_label_blocked(self):
        for key in ['gold_department','confirmed_positive_domains','retrospective_label','source_value','correctness']:
            obs=deepcopy(self.cases[0]['observation']);obs['history']=[{key:'TEST_ONLY'}]
            with self.subTest(key=key),self.assertRaises(ValueError):public_input(obs)
    def test_json_encoded_future_label_blocked(self):
        obs=deepcopy(self.cases[0]['observation']);obs['history']=['{"retrospective_label":"TEST_ONLY"}']
        with self.assertRaises(ValueError):public_input(obs)
    def test_historical_text_not_automatic_gold(self):
        obs=deepcopy(self.cases[0]['observation']);obs['history']=['TEST_ONLY historical diabetes']
        self.assertEqual(public_input(obs),obs)


if __name__=='__main__':unittest.main()
