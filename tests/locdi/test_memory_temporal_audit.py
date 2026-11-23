"""TEST_ONLY nonclinical numeric fixtures; no clinical reference judgments."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import unittest
from app.locdi.llm_setup import digest
from app.locdi.memory_temporal_audit import measurement,facts,compare,runtime_fragment

KEY=['TEST_SYSTEM','TEST_CODE','TEST_UNIT','NativeSyntheaObservation']
IDENTITY={'patient_id':'TEST_P','visit_id':'TEST_CURRENT','started_at':'2021-01-01T00:00:00Z'}


def raw(mid='TEST_M'):
    return {'memory_id':mid,'patient_id':'TEST_P','visit_id':'TEST_OLD','visit_started_at':'2020-01-01T00:00:00Z',
        'observed_at':'2020-01-02T00:00:00Z','available_at':'2020-01-03T00:00:00Z','source_id':'TEST_SOURCE',
        'kind':'objective_finding','text':'TEST_ONLY numeric measurement','supersedes':[]}


def ref(r):
    return {'memory_id':r['memory_id'],'original_record_fingerprint':digest(r),'source_type':'observations',
        'source_alignment':{'status':'VERIFIED_SOURCE_CODING','system_uri':KEY[0],'code':KEY[1],'unit':KEY[2],
            'native':{'resource_type':KEY[3],'coding':{'code':KEY[1]},'unit':KEY[2],
                'pointer':'/record/encounters/0/observations/0','start':1577923200000,'native_value':5}}}


def m(mid='TEST_M',time=1000,value=5,key=None):
    return {'memory_id':mid,'concept_key':KEY if key is None else key,'measurement_epoch_ms':time,
        'value':value,'value_key':json.dumps(value) if value is not None else None}


class TemporalTests(unittest.TestCase):
    def test_actual_native_time(self):
        r=raw();x=measurement(r,ref(r),IDENTITY)
        self.assertEqual(x['measurement_time'],'2020-01-02T00:00:00+00:00')
        self.assertIsNone(x['actual_available_at'])

    def test_no_proxy_for_missing_time(self):
        r=raw();a=ref(r);a['source_alignment']['native'].pop('start')
        self.assertIsNone(measurement(r,a,IDENTITY)['measurement_time'])

    def test_source_hash_mismatch(self):
        r=raw();a=ref(r);r['text']='CHANGED'
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):measurement(r,a,IDENTITY)

    def test_patient_boundary(self):
        r=raw();r['patient_id']='TEST_OTHER'
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):measurement(r,ref(r),IDENTITY)

    def test_current_visit_blocked(self):
        r=raw();r['visit_id']='TEST_CURRENT'
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):measurement(r,ref(r),IDENTITY)

    def test_future_proxy_blocked(self):
        r=raw();r['available_at']='2022-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):measurement(r,ref(r),IDENTITY)

    def test_future_native_blocked(self):
        r=raw();a=ref(r);a['source_alignment']['native']['start']=1700000000000
        with self.assertRaisesRegex(ValueError,'FUTURE'):measurement(r,a,IDENTITY)

    def test_invalid_time_bool(self):
        r=raw();a=ref(r);a['source_alignment']['native']['start']=True
        with self.assertRaisesRegex(ValueError,'TIMESTAMP'):measurement(r,a,IDENTITY)

    def test_no_unit_conversion(self):
        rows=[m(),m('TEST_B',key=['TEST_SYSTEM','TEST_CODE','OTHER_UNIT',KEY[3]])]
        s,_=compare(rows,rows,rows);self.assertEqual(s['modes']['baseline']['distinct_concepts'],2)

    def test_source_key_unit_mismatch_rejected(self):
        r=raw();a=ref(r);a['source_alignment']['native']['unit']='OTHER'
        with self.assertRaisesRegex(ValueError,'UNIT_CONTRADICTION'):measurement(r,a,IDENTITY)

    def test_same_time_duplicate(self):
        f=facts([m(),m('TEST_B')]);self.assertEqual(f['duplicate_timestamp_records'],1)
        self.assertFalse(f['multi_timepoint']);self.assertEqual(f['repeated_value_records'],1)

    def test_multiple_times_same_value(self):
        f=facts([m(),m('TEST_B',time=2000)]);self.assertTrue(f['multi_timepoint'])
        self.assertEqual(f['distinct_values'],1);self.assertEqual(f['span_seconds'],1)

    def test_multiple_times_changed_value(self):
        f=facts([m(),m('TEST_B',time=2000,value=8)])
        self.assertEqual(f['time_ordered_numeric_changes'],{'increases':1,'decreases':0,'equal':0})

    def test_tied_time_conflict_no_trend(self):
        f=facts([m(),m('TEST_B',value=8),m('TEST_C',time=2000,value=9)])
        self.assertIsNone(f['time_ordered_numeric_changes'])

    def test_missing_time_excluded(self):
        f=facts([m(time=None)]);self.assertEqual(f['missing_timestamp'],1)
        self.assertIsNone(f['span_seconds'])

    def test_nonnumeric_no_direction(self):
        self.assertIsNone(facts([m(value='low'),m(time=2000,value='high')])['time_ordered_numeric_changes'])

    def test_lost_time_not_lost_concept(self):
        old=[m(),m('TEST_B',time=2000)];s,d=compare(old,old[:1],old)
        self.assertEqual(s['lost_times'],1);self.assertEqual(s['dropped_concepts'],0)
        self.assertEqual(s['lost_values'],0);self.assertFalse(d[0]['latest_baseline_retained'])

    def test_lost_value(self):
        old=[m(),m('TEST_B',time=2000,value=8)];s,_=compare(old,old[:1],old)
        self.assertEqual(s['lost_values'],1)

    def test_tie_any_endpoint_retained(self):
        old=[m(),m('TEST_B')];s,_=compare(old,old[:1],old)
        self.assertEqual(s['earliest_baseline_retained']['value'],1)
        self.assertEqual(s['latest_baseline_retained']['value'],1)

    def test_eligible_endpoint_not_selected_endpoint(self):
        x=m();older=m('TEST_OLDER',time=0);s,_=compare([x],[x],[x,older])
        self.assertEqual(s['earliest_baseline_retained']['value'],1)
        self.assertEqual(s['eligible_earliest_baseline']['value'],0)

    def test_empty_denominator_null(self):
        s,_=compare([],[],[]);self.assertIsNone(s['earliest_baseline_retained']['value'])

    def test_new_concept_addition(self):
        a=m();b=m('TEST_NEW',key=['TEST_SYSTEM','OTHER','TEST_UNIT',KEY[3]])
        s,_=compare([a],[a,b],[a,b]);self.assertEqual(s['new_concepts'],1)
        self.assertEqual(s['added_times'],1)

    def test_serialization_matches_role_call_options(self):
        r=raw();cs=[{'memory_id':r['memory_id'],'score':3}];rows,s=runtime_fragment([r],cs,[r['memory_id']])
        self.assertEqual(s,json.dumps(rows,ensure_ascii=False,sort_keys=True));self.assertNotIn('retrieval_score',r)
        self.assertEqual(rows[0]['status_as_of_visit'],'historical')

    def test_serialization_trace_order(self):
        a,b=raw('TEST_A'),raw('TEST_B');cs=[{'memory_id':r['memory_id'],'score':3} for r in [a,b]]
        rs,_=runtime_fragment([a,b],cs,['TEST_B','TEST_A']);self.assertEqual(rs[0]['memory_id'],'TEST_B')

    def test_duplicate_trace_refused(self):
        r=raw()
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):runtime_fragment([r],[],['TEST_M','TEST_M'])

    def test_missing_trace_refused(self):
        with self.assertRaises(KeyError):runtime_fragment([],[],['TEST_MISSING'])

    def test_comparison_deterministic_immutable(self):
        rows=[m(),m('TEST_B',time=2000,value=8)];before=deepcopy(rows)
        self.assertEqual(compare(rows,rows[:1],rows),compare(rows,rows[:1],rows));self.assertEqual(rows,before)

    def test_public_summary_has_no_private_values(self):
        s,_=compare([m()],[m()],[m()]);text=json.dumps(s)
        for token in ['TEST_M','TEST_CODE','memory_id','measurement_epoch_ms','clinical_relevance']:
            self.assertNotIn(token,text)

    def test_no_selector_or_llm_calls_in_auditor(self):
        root=Path(__file__).resolve().parents[2]
        source=(root/'scripts/audit_memory_temporal_runtime.py').read_text(encoding='utf-8')
        calls=[n.func.attr for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ['retrieve','select','chat','chat_json','fit_utility']:self.assertNotIn(forbidden,calls)


if __name__=='__main__':unittest.main()
