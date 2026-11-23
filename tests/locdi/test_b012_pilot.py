"""TEST_ONLY split/context/budget accounting, no clinical Gold constructed."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from app.locdi.b012_pilot import schedule,collect,aggregate
from app.locdi.current_ablation_runner import reserve_generation
from app.locdi.llm_setup import digest
from app.locdi.synthea_sequential_proxy import policy_contract,SequentialSyntheticSourceView
from app.locdi.case_adapter import adapt_case
from app.locdi.episode_exporter import _configuration,_fingerprint
import test_synthea_sequential_proxy as source_tests
from test_sequential_export_contract import metadata


def panel():
    return [{'case_ref':'TEST_C%d'%i,'patient_ref':'TEST_P%d'%i,
        'reference':{'accepted_diagnoses':['TEST_ONLY'],'accepted_departments':['TEST_ONLY']}}
        for i in range(10)]


class B012Tests(unittest.TestCase):
    def test_schedule_preserves_all30(self):
        s=schedule(panel());self.assertEqual(len(s),30);self.assertEqual([x['condition'] for x in s[:3]],['B0','B1','B2'])
    def test_preflight_block_never_calls_and_first_error_stops(self):
        tasks=schedule(panel());pre=[{**t,'status':'CONTEXT_BLOCKED' if t['condition']=='B1' else 'READY'} for t in tasks]
        calls=[]
        def backend(*args):calls.append(1);reserve_generation();raise ValueError('TEST_ONLY')
        with tempfile.TemporaryDirectory() as d:
            r=collect(panel(),backend,trace_path=Path(d)/'trace.jsonl',preflight=pre,guard=lambda:None)
        self.assertEqual(len(calls),1);self.assertEqual(r['generation_count'],1)
        self.assertEqual(sum(x['status']=='PREFLIGHT_CONTEXT_BLOCKED' for x in r['rows']),10)
        self.assertEqual(sum(x['status']=='NOT_ATTEMPTED' for x in r['rows']),19)
        a=aggregate(panel(),r['rows']);self.assertFalse(a['freeze_threshold']['any_condition_reached'])
    def test_registered_threshold_not_completed_subset_percentage(self):
        rows=[]
        for t in schedule(panel()):
            result={'prediction':{'certainty':'supported','diagnosis':'TEST_ONLY','department':'TEST_ONLY'}}
            r={**t,'status':'COMPLETED' if t['patient_ordinal']==1 else 'NOT_ATTEMPTED',
                'request_count':0,'result':result if t['patient_ordinal']==1 else None}
            r['row_fingerprint']=digest(r);rows.append(r)
        a=aggregate(panel(),rows)
        self.assertFalse(a['freeze_threshold']['any_condition_reached'])
        for r in rows:
            if r['patient_ordinal']==2:
                r['status']='COMPLETED';r['result']={'prediction':{'certainty':'supported','diagnosis':'TEST_ONLY','department':'TEST_ONLY'}}
                r['row_fingerprint']=digest({k:v for k,v in r.items() if k!='row_fingerprint'})
        self.assertTrue(aggregate(panel(),rows)['freeze_threshold']['any_condition_reached'])
    def test_source_full_history_explicit_new_scope_only(self):
        source_tests.SequentialProxyTests.setUpClass();fixture=source_tests.SequentialProxyTests
        rows=deepcopy(fixture.rows);spec=policy_contract(rows,test_only=True,allow_full_history=True)
        view=SequentialSyntheticSourceView(rows,spec=spec,authorization=source_tests.fixture_permission(spec),
            plan_fingerprint='TEST_ONLY_PLAN',catalog=fixture.catalog)
        from app.locdi.current_ablation_runner import features_for
        adapted=adapt_case(fixture.case);before=deepcopy(adapted.runtime_case)
        runtime,_=view.apply(adapted,features=features_for('B1'),purpose='development_smoke')
        self.assertEqual(runtime['past_visits'],before['past_visits']);self.assertEqual(adapted.runtime_case,before)
        m=metadata();m['configuration']['memory_mode']='full_history';m['configuration']['sequential_source_view']=view.metadata()
        m['configuration_fingerprint']=_fingerprint(m['configuration'])
        self.assertEqual(_configuration(m),m['configuration'])


if __name__=='__main__':unittest.main()
