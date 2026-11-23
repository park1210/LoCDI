"""TEST_ONLY synthetic nonclinical inputs; not actual release approvals."""
from copy import deepcopy
from datetime import datetime,timezone
import unittest
from unittest.mock import patch

from app.locdi.synthea_development_proxy import (POLICY,row_checks,resolve_duplicates,
    draft_observation,verify_authorization)
from app.locdi.llm_setup import digest


def event(value='0',code='TEST_ONLY_CODE',row=2):
    return {'event_id':'TEST_ONLY_%s'%row,'patient_id':'TEST_ONLY_PATIENT','encounter_id':'TEST_ONLY_VISIT',
        'event_type':'observations','timestamp':'1970-01-01T00:00:01Z','timing_precision':'second',
        'timestamp_lower_epoch_ms':1000,'timestamp_upper_epoch_ms':1999,
        'target':{'code':code,'description':'TEST_ONLY_MEASUREMENT'},
        'evidence':{'kind':'recorded_observation','category':'laboratory','value':value,'units':'TEST_UNIT'},
        'source_metadata':{'source_file':'TEST_ONLY.csv','source_row':row,'source_record':{
            'PATIENT':'TEST_ONLY_PATIENT','ENCOUNTER':'TEST_ONLY_VISIT','TYPE':'numeric','CODE':code,
            'DESCRIPTION':'TEST_ONLY_MEASUREMENT','CATEGORY':'laboratory','VALUE':value,
            'UNITS':'TEST_UNIT','DATE':'1970-01-01T00:00:01Z'}}}


def check(e,**kwargs):
    defaults={'patient':'TEST_ONLY_PATIENT','visit':'TEST_ONLY_VISIT','index_ms':0,'cutoff':45,
        'source_row_matches':True,'catalog':['TEST_ONLY_DIAGNOSIS (disorder)']}
    defaults.update(kwargs)
    return row_checks(e,**defaults)


def row(e):return {'patient_ordinal':1,'source_event_id':e['event_id'],'source_event':e}


class SyntheticDevelopmentProxyTests(unittest.TestCase):
    def test_structural_pass_is_not_v1_approval(self):
        r=check(event());self.assertTrue(r['structural_candidate'])
        self.assertEqual(r['strict_v1_status'],'RELEASE_NOT_REVIEWED')
        self.assertEqual(r['clinical_release_status'],'NOT_REVIEWED')
    def test_zero_preserved(self):
        e=event();r=draft_observation({'objective_findings':[]},[row(e)])
        self.assertIn(': 0 TEST_UNIT',r['objective_findings'][0])
    def test_bool_not_numeric(self):self.assertFalse(check(event(True))['structural_candidate'])
    def test_null_not_numeric(self):self.assertFalse(check(event(None))['structural_candidate'])
    def test_nan_not_numeric(self):self.assertFalse(check(event('nan'))['structural_candidate'])
    def test_pain_Ask_only(self):self.assertFalse(check(event(code='72514-3'))['structural_candidate'])
    def test_raw_row_required(self):self.assertFalse(check(event(),source_row_matches=False)['structural_candidate'])
    def test_patient_mismatch(self):self.assertFalse(check(event(),patient='OTHER')['structural_candidate'])
    def test_visit_mismatch(self):self.assertFalse(check(event(),visit='OTHER')['structural_candidate'])
    def test_value_mismatch(self):
        e=event();e['evidence']['value']='1';self.assertFalse(check(e)['structural_candidate'])
    def test_unit_mismatch(self):
        e=event();e['evidence']['units']='OTHER';self.assertFalse(check(e)['structural_candidate'])
    def test_description_mismatch(self):
        e=event();e['target']['description']='OTHER';self.assertFalse(check(e)['structural_candidate'])
    def test_date_mismatch(self):
        e=event();e['source_metadata']['source_record']['DATE']='OTHER';self.assertFalse(check(e)['structural_candidate'])
    def test_forged_epoch_blocked(self):
        e=event();e['timestamp_upper_epoch_ms']=0;self.assertFalse(check(e)['structural_candidate'])
    def test_future_cutoff_blocked(self):self.assertFalse(check(event(),cutoff=0)['structural_candidate'])
    def test_unknown_precision_blocked(self):
        e=event();e['timing_precision']='day';self.assertFalse(check(e)['structural_candidate'])
    def test_condition_not_observation(self):
        e=event();e['event_type']='conditions';self.assertFalse(check(e)['structural_candidate'])
    def test_diagnosis_literal_excluded_common_catalog(self):
        e=event();e['target']['description']='TEST_ONLY_DIAGNOSIS';e['source_metadata']['source_record']['DESCRIPTION']='TEST_ONLY_DIAGNOSIS'
        self.assertFalse(check(e)['structural_candidate'])
    def test_duplicate_exact_values_not_repeated(self):
        a,b=row(event(row=3)),row(event(row=2));before=deepcopy([a,b]);keep,audit=resolve_duplicates([a,b])
        self.assertEqual(len(keep),1);self.assertEqual(keep[0]['source_event']['source_metadata']['source_row'],2)
        self.assertEqual([a,b],before);self.assertEqual(len(audit),2)
    def test_conflicting_samekey_excludes_all(self):
        keep,audit=resolve_duplicates([row(event('1',row=2)),row(event('2',row=3))])
        self.assertEqual(keep,[]);self.assertTrue(all(r['status']=='CONFLICTING_SAME_KEY_EXCLUDED' for r in audit))
    def test_units_not_converted_or_deduped(self):
        a,b=event(row=2),event(row=3);b['evidence']['units']='OTHER';b['source_metadata']['source_record']['UNITS']='OTHER'
        keep,_=resolve_duplicates([row(a),row(b)]);self.assertEqual(len(keep),2)
    def test_code_not_diagnosis_dependent(self):
        a,b=event(code='A'),event(code='B',row=3);keep,_=resolve_duplicates([row(a),row(b)])
        self.assertEqual(len(keep),2)
    def test_observation_base_not_mutated(self):
        x={'objective_findings':['BASE'],'history':[]};b=deepcopy(x);d=draft_observation(x,[row(event())])
        self.assertEqual(x,b);self.assertEqual(d['objective_findings'][0],'BASE')
    def authorization(self):
        return {'evidence_type':'USER_EXPLICIT','user_statement':'TEST_ONLY_AUTHORIZATION_NOT_REAL',
            'plan_fingerprint':'TEST_ONLY_PLAN','policy':POLICY,'scope':'SYNTHETIC_STATIC_D2_D4_ONLY',
            'accept_recorded_time_proxy':True,'accept_intraday_diagnosis_order_unknown':True,
            'accept_reference_ordinal12_16_next_day_limit':True,'authorize_new_snapshot_policy':True,
            'authorize_fixed_D2_D4_generation':True,'clinical_release_approved':False,
            'clinical_reviewer_created':False,'strict_v1_grants_created':0,
            'recorded_at':datetime.now(timezone.utc).isoformat()}
    def test_missing_approval_fails(self):
        with self.assertRaises(ValueError):verify_authorization(None,{'fingerprint':'TEST_ONLY_PLAN'})
    def test_distinct_explicit_TEST_ONLY_authorization(self):
        self.assertTrue(verify_authorization(self.authorization(),{'fingerprint':'TEST_ONLY_PLAN'}))
    def test_old_approval_not_reused(self):
        with self.assertRaises(ValueError):verify_authorization(self.authorization(),{'fingerprint':'OTHER'})
    def test_unknown_intraday_not_silently_accepted(self):
        a=self.authorization();a['accept_intraday_diagnosis_order_unknown']=False
        with self.assertRaises(ValueError):verify_authorization(a,{'fingerprint':'TEST_ONLY_PLAN'})
    def test_no_fabricated_clinical_review(self):
        a=self.authorization();a['clinical_release_approved']=True
        with self.assertRaises(ValueError):verify_authorization(a,{'fingerprint':'TEST_ONLY_PLAN'})
    def test_no_v1_fake_grants(self):
        a=self.authorization();a['strict_v1_grants_created']=1
        with self.assertRaises(ValueError):verify_authorization(a,{'fingerprint':'TEST_ONLY_PLAN'})
    def test_no_network_for_offline_checks(self):
        with patch('socket.socket',side_effect=AssertionError('NO_NETWORK')):
            check(event());resolve_duplicates([row(event())])


if __name__=='__main__':unittest.main()
