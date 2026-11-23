"""TEST_ONLY synthetic Source fixtures and recorded dataset structural checks."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json,socket,tempfile,unittest
from unittest.mock import patch
from app.locdi.original50_augmentation import (recover_symptom_events,scenario,validate_augmentation,initial_preview,
    directly_discloses_label,VERSION,SCOPE,pointer_part,ms_iso)
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.llm_setup import digest
from app.locdi.original_source_inventory import sha_file
from scripts.build_original50_literature_augmentation import candidate_test,POLICY

NOW=946771200000  # 2000-01-02T00:00:00Z, fixture clock only.
def fixture():
    case={'case_id':'TEST_ONLY_CASE','patient_id':'TEST_ONLY_PATIENT','agent_visible':{
        'longitudinal':{'patient_id':'TEST_ONLY_PATIENT','visit_id':'TEST_ONLY_CURRENT','started_at':'2000-01-02T00:00:00Z'},
        'current_initial_state':{'doctor_visible_info':{'initial_vitals':{}},'patient_visible_info':{'associated_symptoms':[]}}},
        'environment':{'source_event_store':[]}}
    event=lambda t,value:{'time':t,'value':value,'cause':'TEST_ONLY_PRIVATE_MODULE'}
    raw={'attributes':{'id':'TEST_ONLY_PATIENT'},'record':{'encounters':[
        {'uuid':'TEST_ONLY_PAST','start':NOW-86400000,'stop':NOW-86400000+3600000},
        {'uuid':'TEST_ONLY_CURRENT','start':NOW,'stop':NOW+3600000}]},
        'symptoms':{'Headache':{'sources':{'TEST_ONLY_MODULE':{'resolved':True,'lastUpdateTime':NOW+86400000,
            'timeInfos':{str(NOW):event(NOW,18),str(NOW-86400000+1000):event(NOW-86400000+1000,20)}}}}}}
    return case,raw

def row_for(case,raw,catalog):
    facts,_=recover_symptom_events(case,raw,'TEST_ONLY_NATIVE.json','f'*64)
    synthetic=scenario(1,catalog['cards'][0],catalog,POLICY)
    row={'version':VERSION,'case_ordinal':1,'original_case_fingerprint':digest(case),'clinical_reviewed':False,
        'clinical_diagnosability':'UNREVIEWED','evaluation_scope':SCOPE,'literature_fingerprint':digest(catalog),
        'original_findings':[],'source_recovered_findings':facts,'literature_informed_synthetic':synthetic,
        'source_reference':{'diagnosis':'PRIVATE_TEST_ONLY','department':'PRIVATE_TEST_ONLY'},
        'native_source':{'file':'TEST_ONLY_NATIVE.json','file_sha256':'f'*64}}
    row['row_fingerprint']=digest(row);return row
def seal(row):row['row_fingerprint']=digest({k:v for k,v in row.items() if k!='row_fingerprint'});return row

class RecoveryContracts(unittest.TestCase):
    def setUp(self):self.case,self.raw=fixture()
    def recover(self):return recover_symptom_events(self.case,self.raw,'TEST_ONLY_NATIVE.json','f'*64)
    def source(self):return self.raw['symptoms']['Headache']['sources']['TEST_ONLY_MODULE']
    def add(self,t,value):self.source()['timeInfos'][str(t)]={'time':t,'value':value}
    def test_current_and_historical_are_separate(self):
        facts,bad=self.recover();self.assertEqual(len(facts),2);self.assertEqual(bad,[])
        self.assertEqual({f['scope'] for f in facts},{'CURRENT','HISTORICAL'})
    def test_final_resolved_state_never_erases_past_positive_events(self):
        facts,_=self.recover();self.source()['resolved']=False;self.assertEqual(facts,self.recover()[0])
    def test_future_event_rejected_not_carried_back(self):
        self.add(NOW+46*60000,30);facts,bad=self.recover();self.assertEqual(len(facts),2)
        self.assertEqual(bad[0]['reason'],'FUTURE_SYMPTOM_EVENT')
    def test_zero_is_not_a_patient_denial(self):
        self.add(NOW+1000,0);self.assertEqual(self.recover()[1][0]['reason'],'NONPOSITIVE_ENGINE_VALUE_NOT_A_PATIENT_DENIAL')
    def test_no_encounter_interval_rejects(self):
        self.add(NOW-5*86400000,20);self.assertEqual(self.recover()[1][0]['reason'],'NO_UNIQUE_SOURCE_ENCOUNTER_INTERVAL')
    def test_overlapping_intervals_reject(self):
        self.raw['record']['encounters'].append({'uuid':'TEST_ONLY_OVERLAP','start':NOW,'stop':NOW+100})
        facts,bad=self.recover();self.assertEqual(len(facts),1);self.assertEqual(bad[0]['reason'],'NO_UNIQUE_SOURCE_ENCOUNTER_INTERVAL')
    def test_wrong_patient_rejected(self):
        self.raw['attributes']['id']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'PATIENT'):self.recover()
    def test_time_key_mismatch_rejected(self):
        self.source()['timeInfos'][str(NOW)]['time']=NOW+1
        self.assertEqual(self.recover()[1][0]['reason'],'EVENT_KEY_TIME_MISMATCH')
    def test_diagnostic_term_and_cause_are_not_public_symptoms(self):
        self.raw['symptoms']['Alzheimer diagnosis']=self.raw['symptoms'].pop('Headache')
        facts,bad=self.recover();self.assertEqual(facts,[]);self.assertTrue(all(x['reason'].startswith('DIAGNOSIS') for x in bad))
    def test_engine_value_not_validated_clinical_intensity(self):
        facts,_=self.recover();self.assertTrue(all(f['engine_value_interpretation']=='NOT_A_VALIDATED_CLINICAL_SEVERITY_SCALE' and not f['execution_release_approved'] for f in facts))
    def test_json_pointer_escaping(self):self.assertEqual(pointer_part('A/B~C'),'A~1B~0C')
    def test_pre1970_source_time_preserved_on_windows(self):self.assertEqual(ms_iso(-1000),'1969-12-31T23:59:59.000+00:00')
    def test_input_objects_unchanged(self):
        before=deepcopy((self.case,self.raw));self.recover();self.assertEqual(before,(self.case,self.raw))

class AuthoredContracts(unittest.TestCase):
    def setUp(self):
        self.case,self.raw=fixture();self.catalog=evidence_catalog();self.row=row_for(self.case,self.raw,self.catalog)
        p=patch.object(socket.socket,'connect',side_effect=AssertionError('NO_NETWORK_OR_MODEL'));p.start();self.addCleanup(p.stop)
    def test_exact_fifteen_and_twenty_six_sources(self):
        self.assertEqual(len(self.catalog['cards']),15);self.assertEqual(len(self.catalog['sources']),26)
        self.assertTrue(all(s['url'].startswith('https://') and s['publication_year']=='UNKNOWN' for s in self.catalog['sources'].values()))
    def test_failed_urls_not_accepted_as_facts(self):self.assertEqual(len(self.catalog['failed_access_not_used']),3)
    def test_deterministic_without_results_or_gold_outcomes(self):
        self.assertEqual(scenario(9,self.catalog['cards'][0],self.catalog,POLICY),scenario(9,self.catalog['cards'][0],self.catalog,POLICY))
        self.assertFalse(POLICY['outcome_or_error_conditioning'])
    def test_fifty_unique_chiefs_and_no_numeric_tests(self):
        used=set()
        for n in range(1,51):
            s=scenario(n,self.catalog['cards'][(n-1)%15],self.catalog,POLICY,used_chiefs=used)
            self.assertEqual(s['synthetic_test_results'],[]);self.assertFalse(directly_discloses_label(s['chief_complaint']['text']))
        self.assertEqual(len(used),50)
    def test_silent_profiles_do_not_force_typical_symptoms(self):
        for c in self.catalog['cards']:
            if not c['symptom_feature_ids']:self.assertEqual(scenario(1,c,self.catalog,POLICY)['symptoms'],[])
    def test_original_source_view_exact_and_returns_copy(self):
        before=deepcopy(self.case);v=initial_preview(self.case,self.row,self.catalog)
        self.assertEqual(v,self.case['agent_visible']['current_initial_state']);self.assertEqual(self.case,before)
        v['doctor_visible_info']['TEST_ONLY']=True;self.assertNotIn('TEST_ONLY',self.case['agent_visible']['current_initial_state']['doctor_visible_info'])
    def test_nonoriginal_view_requires_explicit_preview_scope(self):
        for view in ('D1','D2'):
            with self.assertRaisesRegex(ValueError,'EXPLICIT'):initial_preview(self.case,self.row,self.catalog,view=view)
    def test_d2_has_no_reference_provenance_tests_or_future_source(self):
        v=initial_preview(self.case,self.row,self.catalog,view='D2',accept_development_proxy=True);text=json.dumps(v)
        for secret in ('PRIVATE_TEST_ONLY','source_reference','provenance','candidate_tests','TEST_ONLY_MODULE'):self.assertNotIn(secret,text)
        self.assertEqual(v['doctor_visible_info']['initial_vitals'],{})
    def test_d2_appends_without_dropping_original_source_labels(self):
        self.case['agent_visible']['current_initial_state']['patient_visible_info']['associated_symptoms']=['TEST_ONLY_ORIGINAL_LABEL']
        self.row=row_for(self.case,self.raw,self.catalog)
        v=initial_preview(self.case,self.row,self.catalog,view='D2',accept_development_proxy=True)
        self.assertIn('TEST_ONLY_ORIGINAL_LABEL',v['patient_visible_info']['associated_symptoms'])
    def test_existing_source_chief_complaint_cannot_be_replaced(self):
        self.case['agent_visible']['current_initial_state']['doctor_visible_info']['chief_complaint']='TEST_ONLY_SOURCE_CHIEF'
        self.row=row_for(self.case,self.raw,self.catalog)
        with self.assertRaisesRegex(ValueError,'MUST_NOT_BE_REPLACED'):initial_preview(self.case,self.row,self.catalog,view='D2',accept_development_proxy=True)
    def test_reference_label_not_a_clinical_approval(self):
        self.assertEqual(validate_augmentation(self.case,self.row,self.catalog)['clinical_diagnosability'],'UNREVIEWED')
        self.row['clinical_reviewed']=True;seal(self.row)
        with self.assertRaisesRegex(ValueError,'CLINICAL'):validate_augmentation(self.case,self.row,self.catalog)
    def test_row_tamper_rejected(self):
        self.row['literature_informed_synthetic']['chief_complaint']['text']='TEST_ONLY_TAMPER'
        with self.assertRaisesRegex(ValueError,'ROW_FINGERPRINT'):validate_augmentation(self.case,self.row,self.catalog)
    def test_direct_label_disclosure_rejected_even_if_resealed(self):
        self.row['literature_informed_synthetic']['chief_complaint']['text']='Diabetes diagnosis';seal(self.row)
        with self.assertRaisesRegex(ValueError,'DISCLOSURE'):validate_augmentation(self.case,self.row,self.catalog)
    def test_synthetic_results_require_unimplemented_new_contract(self):
        self.row['literature_informed_synthetic']['synthetic_test_results']=[{'test':'TEST_ONLY','value':99}];seal(self.row)
        with self.assertRaisesRegex(ValueError,'RESULTS_NOT_AUTHORED'):validate_augmentation(self.case,self.row,self.catalog)
    def test_clinical_scope_promotion_rejected(self):
        self.row['evaluation_scope']='CLINICAL_EVALUATION';seal(self.row)
        with self.assertRaisesRegex(ValueError,'SCOPE'):validate_augmentation(self.case,self.row,self.catalog)
    def test_original_measurement_changes_rejected(self):
        self.row['original_findings']=[{'value':'TEST_ONLY'}];seal(self.row)
        with self.assertRaisesRegex(ValueError,'OBSERVATION_MODIFIED'):validate_augmentation(self.case,self.row,self.catalog)
    def test_source_future_timestamp_tamper_rejected(self):
        self.row['source_recovered_findings'][0]['recorded_at']='2001-01-01T00:00:00Z';seal(self.row)
        with self.assertRaisesRegex(ValueError,'FUTURE'):validate_augmentation(self.case,self.row,self.catalog)
    def test_unverified_literature_rejected(self):
        self.row['literature_informed_synthetic']['chief_complaint']['sources']=['NOT_OBSERVED'];seal(self.row)
        with self.assertRaisesRegex(ValueError,'LITERATURE'):validate_augmentation(self.case,self.row,self.catalog)
    def test_source_file_hash_and_derived_encounter_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.json';p.write_text(json.dumps(self.raw),encoding='utf-8');h=sha_file(p)
            self.row['native_source']={'file':str(p),'file_sha256':h};self.row['source_recovered_findings']=recover_symptom_events(self.case,self.raw,p,h)[0];seal(self.row)
            with patch('app.locdi.original_source_inventory.native_index',return_value={1:{'raw_path':str(p),'declared_raw_sha256':h}}):
                validate_augmentation(self.case,self.row,self.catalog,verify_source_files=True)
                v=initial_preview(self.case,self.row,self.catalog,view='D1',accept_development_proxy=True)
                self.assertEqual(v['patient_visible_info']['associated_symptoms'],['Headache'])
                p.write_text('{}',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'PIN_MISMATCH'):validate_augmentation(self.case,self.row,self.catalog,verify_source_files=True)

class ActionCompatibility(unittest.TestCase):
    def fixture(self):
        key=json.dumps(['Test','LOINC','4548-4','%','TEST_ONLY']);m={'provenance':{'action_key_mapping':{key:'TEST_ONLY'}}}
        t={'test_id':'TEST_ONLY','name':'TEST_ONLY','sources':['R11'],'matching_source_codes':['4548-4']}
        event={'event_type':'observations','event_id':'TEST_ONLY_SOURCE','timestamp_lower_epoch_ms':NOW,
            'timestamp_upper_epoch_ms':NOW,'target':{'code':'4548-4'},'evidence':{'units':'%','value':'6.1'}}
        return t,event,m
    def test_exact_source_and_existing_m5_key_only(self):
        t,e,m=self.fixture();r=candidate_test(t,[e],m,NOW);self.assertEqual(r['action_compatibility'],'CURRENT_M5_COMPATIBLE')
        self.assertFalse(r['new_action_or_likelihood_created']);self.assertFalse(r['result_generated'])
    def test_source_without_action_is_not_imputed(self):
        t,e,m=self.fixture();m['provenance']['action_key_mapping']={};self.assertEqual(candidate_test(t,[e],m,NOW)['action_compatibility'],'SOURCE_EXISTS_BUT_ACTION_UNAVAILABLE')
    def test_literature_recommendation_not_an_observed_result(self):
        t,e,m=self.fixture();self.assertEqual(candidate_test(t,[],m,NOW)['action_compatibility'],'LITERATURE_RECOMMENDED_NOT_OBSERVED')
    def test_future_source_never_m5_compatible_in_this_view(self):
        t,e,m=self.fixture();e['timestamp_lower_epoch_ms']=NOW+46*60000;e['timestamp_upper_epoch_ms']=NOW+46*60000
        self.assertEqual(candidate_test(t,[e],m,NOW)['action_compatibility'],'SOURCE_EXISTS_BUT_ACTION_UNAVAILABLE')

if __name__=='__main__':unittest.main()
