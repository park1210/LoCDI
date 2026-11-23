"""TEST_ONLY nonclinical source/response fixtures; no model requests."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator

from app.locdi.synthea_diagnostic_snapshot import (source_check, source_view, measurement_view,
    safe_observation, prepare_snapshot, shared_catalog)
from app.locdi.llm_setup import digest
from app.core.clinical_observation import encounter_evidence

REGISTRY={'schema_version':1,'agents':[{'agent_id':'TEST_ONLY','role':'er','department':'',
    'model':'TEST_ONLY_MODEL','version':'TEST_ONLY','keywords':[]}]}
CATALOG=['TEST_ONLY_DIAGNOSIS_%02d'%i for i in range(15)]


def event():
    return {'event_id':'TEST_ONLY_EVENT','patient_id':'P','encounter_id':'V','event_type':'observations',
        'timestamp':'2020-01-01T00:00:01Z','timestamp_upper_epoch_ms':1999,
        'target':{'code':'TEST_ONLY_CODE','description':'TEST_ONLY_MEASUREMENT'},
        'evidence':{'kind':'recorded_observation','category':'laboratory','value':'0','units':'TEST_UNIT'},
        'source_metadata':{'source_record':{'PATIENT':'P','ENCOUNTER':'V','TYPE':'numeric',
            'CODE':'TEST_ONLY_CODE','VALUE':'0','UNITS':'TEST_UNIT','DATE':'2020-01-01T00:00:01Z'}}}


def grant(e,mode='snapshot'):
    return {'source_fingerprint':digest(e),'mode':mode,'patient_id':'P','visit_id':'V',
        'release_reviewed':True,'diagnosis_leakage_reviewed':True,'contains_diagnosis':False,
        'recorded_time_proxy_accepted':True,'reviewer_private_ref':'TEST_ONLY_REVIEWER',
        'evidence_private_ref':'TEST_ONLY_NONCLINICAL_EVIDENCE','reviewed_at':'2020-01-02T00:00:00Z',
        'procedure_version':'TEST_ONLY'}


def observation(objective=True):
    return {'chief_complaint':'','demographics':{'age':40},'elapsed_minutes':3,
        'objective_findings':['TEST_ONLY_MEASUREMENT: 0 TEST_UNIT'] if objective else [],
        'history':[],'current_symptoms':[],'vital_signs':{},'working_assessment':'MODEL_PREDICTION',
        'longitudinal_memory':{'records':[{'text':'PRIVATE_MEMORY'}]},'evaluation':{'gold':'HIDDEN'}}


class SyntheaDiagnosticSnapshotTests(unittest.TestCase):
    def state(self,e,**kwargs):
        return source_check(e,patient='P',visit='V',index_ms=0,cutoff=2,**kwargs)
    def test_unreviewed_proxy_never_approved(self):
        self.assertEqual(self.state(event()),'RELEASE_NOT_REVIEWED')
    def test_pain_report_never_promoted_to_objective_test(self):
        e=event();e['target']['code']='72514-3';e['source_metadata']['source_record']['CODE']='72514-3'
        self.assertEqual(self.state(e,grant=grant(e)),'PATIENT_REPORTED_ASK_ONLY_NOT_OBJECTIVE_TEST')
    def test_boolean_not_promoted_to_numeric_measurement(self):
        e=event();e['evidence']['value']=True;e['source_metadata']['source_record']['VALUE']=True
        self.assertEqual(self.state(e,grant=grant(e)),'UNUSABLE_BOOLEAN_NOT_NUMERIC_MEASUREMENT')
    def test_nonclinical_complete_grant(self):
        e=event();self.assertEqual(self.state(e,grant=grant(e)),'ADDITIONAL_RELEASE_ELIGIBLE_DEVELOPMENT_ONLY')
    def test_patient_mismatch(self):
        e=event();e['patient_id']='OTHER';self.assertIn('IDENTITY',self.state(e))
    def test_visit_mismatch(self):
        e=event();e['source_metadata']['source_record']['ENCOUNTER']='OTHER';self.assertIn('IDENTITY',self.state(e))
    def test_unit_mismatch(self):
        e=event();e['evidence']['units']='OTHER';self.assertIn('VALUE_MISMATCH',self.state(e))
    def test_value_mismatch(self):
        e=event();e['evidence']['value']='1';self.assertIn('VALUE_MISMATCH',self.state(e))
    def test_future_never_released_even_with_grant(self):
        e=event();e['timestamp_upper_epoch_ms']=180000;self.assertEqual(self.state(e,grant=grant(e)),'FUTURE_NOT_ELIGIBLE')
    def test_timestamp_unknown(self):
        e=event();del e['timestamp_upper_epoch_ms'];self.assertEqual(self.state(e),'RELEASE_TIME_UNKNOWN')
    def test_conditions_not_allowed(self):
        e=event();e['event_type']='conditions';self.assertIn('NOT_NUMERIC',self.state(e))
    def test_no_nonfinite(self):
        e=event();e['evidence']['value']='nan';self.assertIn('NOT_NUMERIC',self.state(e))
    def test_source_hash_mismatch(self):
        e=event();g=grant(e);g['source_fingerprint']='x';self.assertEqual(self.state(e,grant=g),'RELEASE_GRANT_INVALID')
    def test_snapshot_grant_not_sequential(self):
        e=event();self.assertEqual(self.state(e,grant=grant(e),mode='sequential'),'RELEASE_GRANT_INVALID')
    def test_diagnosis_leakage_review_required(self):
        e=event();g=grant(e);g['diagnosis_leakage_reviewed']=None;self.assertEqual(self.state(e,grant=g),'RELEASE_GRANT_INVALID')
    def test_literal_diagnosis_not_approved(self):
        e=event();g=grant(e);g['contains_diagnosis']=True;self.assertEqual(self.state(e,grant=g),'RELEASE_GRANT_INVALID')
    def test_source_view_disabled_unchanged(self):
        x={'available_tests':[]};self.assertEqual(source_view(x,[],{}),x)
    def test_source_view_rejects_voi(self):
        with self.assertRaisesRegex(ValueError,'ONLY_B0_B2'):
            source_view({},[],{},enabled=True,voi_enabled=True)
    def test_source_view_rejects_routing(self):
        with self.assertRaises(ValueError):source_view({},[],{},enabled=True,primary_router='adaptive')
    def test_source_view_no_grant(self):
        x={'longitudinal':{'patient_id':'P','visit_id':'V'},'available_tests':[],'current_encounter_events':[]}
        with self.assertRaisesRegex(ValueError,'NOT_REVIEWED'):
            source_view(x,[event()],{},enabled=True,cutoff=2,index_ms=0)
    def test_sequential_copy_no_pre_release(self):
        e=event();x={'longitudinal':{'patient_id':'P','visit_id':'V'},'available_tests':[],
            'current_encounter_events':[],'doctor_visible_info':{}}
        y=source_view(x,[e],{e['event_id']:grant(e,'sequential')},enabled=True,cutoff=2,index_ms=0)
        self.assertEqual(x['available_tests'],[]);self.assertEqual(y['doctor_visible_info'],{})
        r=SimpleNamespace(case_data=y,current_time_min=0)
        self.assertIsNone(encounter_evidence(r,'Test',y['available_tests'][0]))
        r.current_time_min=1;z=encounter_evidence(r,'Test',y['available_tests'][0])
        value=json.loads(z['evidence']);self.assertEqual(value['value'],'0');self.assertEqual(value['units'],'TEST_UNIT')
    def test_measurement_names_not_symptoms_or_values(self):
        o=observation(False);o['current_symptoms']=['MEASUREMENT'];v=measurement_view(o,['MEASUREMENT'])
        self.assertEqual(v['current_symptoms'],[]);self.assertIsNone(v['chief_complaint'])
        self.assertIsNone(v['reported_measurement_labels'][0]['value']);self.assertEqual(o['current_symptoms'],['MEASUREMENT'])
    def test_measurement_zero_only_after_ask(self):
        o=observation(False);o['history']=[{'answer':json.dumps({'facts':[{'code':'X','value':0,'units':'S','recorded_at':'T'}]})}]
        self.assertEqual(measurement_view(o,[])['observed_reported_measurements'][0]['value'],0)
    def test_safe_projection_excludes_gold_memory_assessment(self):
        o=safe_observation(observation());self.assertNotIn('evaluation',o)
        self.assertEqual(o['working_assessment'],'');self.assertEqual(o['longitudinal_memory']['records'],[])
    def contract(self,objective=True,catalog=None):
        return prepare_snapshot(safe_observation(observation(objective)),registry=REGISTRY,catalog=catalog)
    def response(self,certainty='supported',target='TEST_ONLY_DIAGNOSIS',citations=None):
        return {'action':'diagnose','target':target,'rationale':'TEST_ONLY','assessment':'TEST_ONLY',
            'department':'general_medicine','certainty':certainty,'evidence_indices':[0] if citations is None else citations}
    def test_free_text_default(self):
        c=self.contract();self.assertNotIn('diagnosis_catalog',c['payload'])
        self.assertNotIn('enum',c['schema']['properties']['target'])
    def test_catalog_payload_schema_same15(self):
        for value in (True,False):
            c=self.contract(value,CATALOG)
            self.assertEqual(c['payload']['diagnosis_catalog'],CATALOG)
            self.assertEqual(c['schema']['properties']['target']['enum'][:15],CATALOG)
    def test_enum_actually_rejects_unlisted(self):
        c=self.contract(catalog=CATALOG)
        self.assertFalse(Draft202012Validator(c['schema']).is_valid(self.response()))
        self.assertTrue(Draft202012Validator(c['schema']).is_valid(self.response(target=CATALOG[0])))
    def test_same_observation_catalog_only(self):
        a=self.contract();b=self.contract(catalog=CATALOG)
        for key in ('observation','evidence','evidence_records','supported_diagnosis_eligible_indices'):
            self.assertEqual(a['payload'][key],b['payload'][key])
    def test_d0_unresolved_only(self):
        c=self.contract(False);self.assertEqual(c['schema']['properties']['certainty']['enum'],['unresolved'])
        r=self.response('unresolved','Unresolved',[]);self.assertTrue(Draft202012Validator(c['schema']).is_valid(r))
        self.assertEqual(c['validate'](json.dumps(r))['certainty'],'unresolved')
    def test_supported_needs_objective_citation(self):
        with self.assertRaisesRegex(ValueError,'eligible objective'):
            self.contract()['validate'](json.dumps(self.response(citations=[])))
    def test_history_citation_not_objective(self):
        o=safe_observation(observation(False));o['history']=[{'answer':'TEST_ONLY_ANSWER'}]
        c=prepare_snapshot(o,registry=REGISTRY)
        with self.assertRaisesRegex(ValueError,'eligible objective'):
            c['validate'](json.dumps(self.response()))
    def test_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError,'Invalid evidence'):
            self.contract()['validate'](json.dumps(self.response(citations=[4])))
    def test_supported_department_target_rejected(self):
        with self.assertRaises(ValueError):self.contract()['validate'](json.dumps(self.response(target='cardiology')))
    def test_no_network_during_contract(self):
        with patch('socket.socket',side_effect=AssertionError('NO_NETWORK')):
            self.contract()
    def test_catalog_hash_and_order(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'catalog.json';p.write_text(json.dumps(CATALOG),encoding='utf-8');h=sha256(p.read_bytes()).hexdigest()
            self.assertEqual(shared_catalog(CATALOG,expected_sha=h,path=p),CATALOG)
            with self.assertRaises(ValueError):shared_catalog(list(reversed(CATALOG)),expected_sha=h,path=p)
            with self.assertRaises(ValueError):shared_catalog(CATALOG,expected_sha='f'*64,path=p)

    def test_native_backend_passes_explicit_catalog_and_preserves_none(self):
        from app.locdi.current_ablation_llm_backend import make_er_backend
        from app.locdi.current_ablation_runner import features_for
        class CapturedOptions(Exception):pass
        seen=[]
        def run_case(case,*,options):
            seen.append(options.diagnosis_catalog)
            raise CapturedOptions()
        fake=SimpleNamespace(prepared={'registry':REGISTRY},close=lambda:None)
        for catalog in (None,deepcopy(CATALOG)):
            with patch('app.locdi.current_ablation_llm_backend.static_preflight',return_value={'status':'STATIC_CONTRACT_READY'}), \
                 patch('app.locdi.current_ablation_llm_backend.BudgetedNativeERClient',return_value=fake), \
                 patch('app.locdi.run_simulation.run_case',side_effect=run_case):
                backend=make_er_backend({'TEST_ONLY':{}},config={'minute_limit':120},
                    contract_fingerprint='a'*64,checkpoint_root='TEST_ONLY_NO_WRITE',authorized=True,
                    diagnosis_catalog=catalog)
                if catalog is not None:catalog.reverse()
                with self.assertRaises(CapturedOptions):backend({'case_ref':'TEST_ONLY'},features_for('B0'))
        self.assertIsNone(seen[0]);self.assertEqual(list(seen[1]),CATALOG)


if __name__=='__main__':unittest.main()
