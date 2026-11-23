"""TEST_ONLY complete record/atomic block budgets and fixed retrieval boundary."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import unittest
from app.core.longitudinal_memory import PatientMemoryStore
from app.locdi.context_budgeted_history import build_plan,ContextBudgetedHistory,joint_cost,projected,source_records


def fixture():
    from test_synthea_sequential_proxy import SequentialProxyTests
    SequentialProxyTests.setUpClass();case=deepcopy(SequentialProxyTests.case)
    identity=case['agent_visible']['longitudinal']
    records=[r for v in case['agent_visible']['past_visits'] for r in v['records']]
    store=PatientMemoryStore(records)
    context=store.retrieve(identity,'TEST_ONLY','full_history',1,max_chars=None)
    return identity,store,context


class BudgetedHistory(unittest.TestCase):
    def test_atomic_records_unchanged_stable_across_queries(self):
        identity,store,ctx=fixture();before=deepcopy(ctx);plan=build_plan(identity,ctx,4096)
        self.assertEqual(ctx,before);self.assertGreater(plan['selected_count'],0);self.assertGreater(plan['excluded_count'],0)
        policy=ContextBudgetedHistory([plan])
        with policy.installed():
            a=store.retrieve(identity,'pain','full_history',1,max_chars=None)
            b=store.retrieve(identity,'different query blood pressure','full_history',1,max_chars=None)
            full=store.retrieve(identity,'pain','full_history')
        self.assertEqual([r['memory_id'] for r in a['records']],[r['memory_id'] for r in b['records']])
        self.assertEqual(source_records(a['records']),source_records(b['records']))
        self.assertEqual(len(full['records']),len(ctx['records']))
        self.assertEqual(plan['selected_count']+plan['excluded_count'],len(ctx['records']))
        self.assertEqual(a['records'],sorted(a['records'],key=lambda r:(r['available_at'],r['memory_id'])))
    def test_foreign_patient_or_future_source_rejected(self):
        identity,store,ctx=fixture()
        for key,value in (('patient_id','TEST_OTHER'),('available_at','2100-01-01T00:00:00Z')):
            c=deepcopy(ctx);c['records'][0][key]=value
            with self.assertRaises(ValueError):build_plan(identity,c,4096)
    def test_large_record_skipped_smaller_record_still_considered(self):
        identity,store,ctx=fixture();c=deepcopy(ctx);latest=max(c['records'],key=lambda r:r['available_at'])
        latest['text']='TEST_ONLY_OVERSIZE'+('x'*10000)
        plan=build_plan(identity,c,4096)
        self.assertNotIn(latest['memory_id'],plan['selected_ids']);self.assertGreater(plan['selected_count'],0)
    def test_declared_visit_block_never_partially_admitted(self):
        identity,store,ctx=fixture();chosen=ctx['records'][-2:];groups={r['memory_id']:'TEST_ATOMIC_VISIT' for r in chosen}
        plan=build_plan(identity,ctx,4096,atomic_groups=groups)
        present=[r['memory_id'] in plan['selected_ids'] for r in chosen]
        self.assertEqual(present[0],present[1])
    def test_tamper_and_unregistered_identity_rejected(self):
        identity,store,ctx=fixture();plan=build_plan(identity,ctx,4096);p=ContextBudgetedHistory([plan])
        changed=deepcopy(ctx);changed['records'][0]['text']='TEST_CHANGED'
        with self.assertRaises(ValueError):p.project(identity,changed)
        bad=dict(identity,patient_id='TEST_OTHER')
        with self.assertRaises(ValueError):p.project(bad,ctx)


if __name__=='__main__':unittest.main()
