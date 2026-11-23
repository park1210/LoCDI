"""TEST_ONLY collection-policy fixtures; no clinical verdict or real request."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.current_ablation_runner import reserve_generation
from app.locdi.memory_paired_recovery import collect, aggregate, schedule, citation_audit, SharedRecoveryStop, classify_failure,verify_delivered_memory
from scripts.run_memory_paired_recovery import verify_approval, SCOPE


def panel():
    return [{'case_ref':'TEST_ONLY_C%d'%i,'patient_ref':'TEST_ONLY_P%d'%i,
             'reference':{'accepted_diagnoses':['TEST_ONLY_LABEL'],'accepted_departments':['TEST_ONLY_DEPT']}}
            for i in range(2)]


def result(correct=False,certainty='supported'):
    return {'status':'completed','request_count':1,
        'prediction':{'certainty':certainty,'diagnosis':('TEST_ONLY_LABEL' if correct else 'TEST_ONLY_OTHER') if certainty=='supported' else None,
                      'department':'TEST_ONLY_DEPT','algorithmic_diagnosis_choice':None},
        'action_counts':{'Ask':0,'Test':0,'Consult':0,'Diagnose':1}}


def audit(task,error):
    return {'error_types':[error] if error else [],'shared_integrity_failure':error=='SharedRecoveryStop'}


class PolicyTests(unittest.TestCase):
    def run_fixture(self,backend,*,guard=lambda:None,auditor=audit):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.jsonl'
            out=collect(panel(),backend,trace_path=path,total_cap=8,per_episode_cap=2,guard=guard,audit_failure=auditor)
            saved=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(out['rows'],saved)
            return out,aggregate(panel(),saved)

    def test_pair_order_fixed(self):
        self.assertEqual([(x['patient_ordinal'],x['condition']) for x in schedule(panel())],[(1,'B0'),(1,'B2'),(2,'B0'),(2,'B2')])

    def test_duplicate_patient_rejected(self):
        p=panel();p[1]['patient_ref']=p[0]['patient_ref']
        with self.assertRaises(ValueError):schedule(p)

    def test_duplicate_case_rejected(self):
        p=panel();p[1]['case_ref']=p[0]['case_ref']
        with self.assertRaises(ValueError):schedule(p)

    def test_b0_failure_does_not_remove_same_b2(self):
        calls=[]
        def backend(case,features):
            reserve_generation();calls.append((case,features))
            if len(calls)==1:raise ValueError('TEST_ONLY_CONTRACT')
            return result()
        out,s=self.run_fixture(backend)
        self.assertEqual(len(calls),4)
        self.assertEqual([r['status'] for r in out['rows']],['CONTRACT_FAILED','COMPLETED','COMPLETED','COMPLETED'])
        self.assertEqual(s['complete_pairs'],1);self.assertEqual(s['registered_patients'],2)
        self.assertEqual(s['transitions']['one_or_both_unobserved'],1)
        self.assertEqual(set(calls[0][0]),{'case_ref'})

    def test_b2_failure_does_not_remove_next_patient(self):
        n=[0]
        def backend(case,features):
            reserve_generation();n[0]+=1
            if n[0]==2:raise ValueError('TEST_ONLY_CONTRACT')
            return result()
        out,s=self.run_fixture(backend)
        self.assertEqual(n[0],4);self.assertEqual(s['complete_pairs'],1)

    def test_timeout_isolated_no_retry(self):
        n=[0]
        def backend(case,features):
            reserve_generation();n[0]+=1
            if n[0]==1:raise TimeoutError()
            return result()
        out,s=self.run_fixture(backend)
        self.assertEqual(out['rows'][0]['status'],'TIMEOUT');self.assertEqual(out['generation_count'],4)

    def test_context_blocked_without_generation(self):
        n=[0]
        class InputContextBudgetError(ValueError):pass
        def backend(case,features):
            n[0]+=1
            if n[0]==1:raise InputContextBudgetError()
            reserve_generation();return result()
        out,s=self.run_fixture(backend)
        self.assertEqual(out['rows'][0]['status'],'CONTEXT_BLOCKED');self.assertEqual(out['generation_count'],3)

    def test_source_violation_stops_all(self):
        def backend(case,features):raise SharedRecoveryStop('TEST_ONLY_SOURCE')
        out,s=self.run_fixture(backend)
        self.assertEqual(out['generation_count'],0);self.assertTrue(all(r['status']=='NOT_ATTEMPTED' for r in out['rows']))

    def test_identity_violation_stops_all(self):
        class ModelIdentityError(ValueError):pass
        def backend(case,features):reserve_generation();raise ModelIdentityError()
        def detail(task,error):return {'error_types':[error],'shared_integrity_failure':True}
        out,s=self.run_fixture(backend,auditor=detail)
        self.assertEqual(out['generation_count'],1);self.assertEqual(out['rows'][1]['status'],'NOT_ATTEMPTED')

    def test_episode_budget_not_silently_retried(self):
        def backend(case,features):
            reserve_generation();reserve_generation();reserve_generation();return result()
        out,s=self.run_fixture(backend)
        self.assertEqual(out['generation_count'],8);self.assertEqual(s['complete_pairs'],0)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.jsonl';path.write_text('TEST_ONLY',encoding='utf-8')
            with self.assertRaises(FileExistsError):collect(panel(),lambda *a:None,trace_path=path,total_cap=8,per_episode_cap=2,guard=lambda:None,audit_failure=audit)
            self.assertEqual(path.read_text(),'TEST_ONLY')

    def test_reference_not_in_backend_input(self):
        def backend(case,features):
            self.assertNotIn('reference',case);self.assertNotIn('TEST_ONLY_LABEL',str(case))
            reserve_generation();return result()
        self.run_fixture(backend)

    def test_complete_pair_transitions(self):
        def backend(case,features):reserve_generation();return result(correct=features['memory_mode']=='selective')
        out,s=self.run_fixture(backend)
        self.assertEqual(s['transitions'],{'B0_incorrect_to_B2_correct':2});self.assertEqual(s['paired_source_exact_B2_minus_B0'],1)

    def test_unresolved_not_incorrect(self):
        def backend(case,features):reserve_generation();return result(certainty='unresolved')
        out,s=self.run_fixture(backend)
        self.assertEqual(s['complete_pairs'],2);self.assertEqual(s['source_scorable_pairs'],0)
        self.assertIsNone(s['paired_source_exact_B2_minus_B0']);self.assertEqual(s['conditions']['B2']['source_exact_supported']['status'],'NOT_APPLICABLE')

    def test_concept_not_automatically_verified_by_exact(self):
        def backend(case,features):reserve_generation();return result(correct=True)
        out,s=self.run_fixture(backend)
        self.assertTrue(s['episodes'][0]['source_diagnosis_exact']);self.assertIsNone(s['episodes'][0]['verified_concept_agreement'])

    def test_saved_row_tampering_rejected(self):
        def backend(case,features):reserve_generation();return result()
        out,s=self.run_fixture(backend);out['rows'][0]['status']='COMPLETED_TAMPERED'
        with self.assertRaises(ValueError):aggregate(panel(),out['rows'])

    def test_missing_patient_not_dropped(self):
        def backend(case,features):reserve_generation();return result()
        out,s=self.run_fixture(backend)
        with self.assertRaises(ValueError):aggregate(panel(),out['rows'][:-1])


class CitationAndApprovalTests(unittest.TestCase):
    def request(self):
        return {'messages':[{'role':'system','content':'TEST_ONLY'}, {'role':'user','content':json.dumps({
            'evidence_records':[{'kind':'objective_finding','supports_finalization':True}]*4+[{'kind':'patient_history','supports_finalization':False}],
            'supported_diagnosis_eligible_indices':[0,1,2,3]})}]}

    def response(self,indices,certainty='supported'):
        return {'message':{'content':json.dumps({'action':'diagnose','certainty':certainty,'evidence_indices':indices})}}

    def test_actual_error_type_not_out_of_range(self):
        r=citation_audit(self.request(),self.response([4]))
        self.assertTrue(r['citation_in_range']);self.assertFalse(r['eligible_objective_cited'])
        self.assertEqual(r['error_code'],'SUPPORTED_DIAGNOSIS_WITHOUT_ELIGIBLE_OBJECTIVE_CITATION')

    def test_out_of_range_separate(self):
        self.assertEqual(citation_audit(self.request(),self.response([5]))['error_code'],'EVIDENCE_INDEX_OUT_OF_RANGE')

    def test_empty_supported_not_auto_fixed(self):
        r=citation_audit(self.request(),self.response([]));self.assertIsNotNone(r['error_code']);self.assertFalse(r['response_rewritten'])

    def test_unresolved_not_upgraded(self):
        r=citation_audit(self.request(),self.response([],'unresolved'));self.assertIsNone(r['error_code']);self.assertEqual(r['certainty'],'unresolved')

    def test_new_approval_required(self):
        with self.assertRaises(SharedRecoveryStop):verify_approval({}, {'fingerprint':'a'*64})

    def test_old_approval_scope_not_reused(self):
        t=datetime.now(timezone.utc);r={'approved':True,'scope':'OLD_FINAL_EXP','plan_fingerprint':'a'*64,
            'evidence_type':'USER_EXPLICIT','user_statement':'TEST_ONLY','recorded_at':t.isoformat(),'expires_at':(t+timedelta(hours=1)).isoformat()}
        with self.assertRaises(SharedRecoveryStop):verify_approval(r,{'fingerprint':'a'*64})
        r['scope']=SCOPE;verify_approval(r,{'fingerprint':'a'*64})


class BoundaryTests(unittest.TestCase):
    def setup_source(self):
        record={'memory_id':'TEST_ONLY_M','patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_OLD',
            'available_at':'2020-01-01T00:00:00Z','observed_at':'2020-01-01T00:00:00Z',
            'visit_started_at':'2020-01-01T00:00:00Z','text':'TEST_ONLY_FACT'}
        case={'patient_id':'TEST_ONLY_P','agent_visible':{'index_encounter_id':'TEST_ONLY_CURRENT',
            'past_visits':[{'records':[deepcopy(record)]}]}}
        mem={'as_of':'2021-01-01T00:00:00Z','records':[record]}
        return case,mem

    def test_source_patient_visit_cutoff(self):
        c,m=self.setup_source();self.assertEqual(verify_delivered_memory(c,m)['status'],'PASS')

    def test_other_patient_rejected(self):
        c,m=self.setup_source();m['records'][0]['patient_id']='OTHER_TEST_PATIENT'
        self.assertIn('PATIENT_MISMATCH',verify_delivered_memory(c,m)['failure_codes'])

    def test_future_memory_rejected(self):
        c,m=self.setup_source();m['records'][0]['available_at']='2022-01-01T00:00:00Z'
        self.assertIn('FUTURE_OR_VISIT_CUTOFF',verify_delivered_memory(c,m)['failure_codes'])

    def test_superseded_rejected(self):
        c,m=self.setup_source();m['records'][0]['status_as_of_visit']='superseded'
        self.assertIn('SUPERSEDED_RECORD',verify_delivered_memory(c,m)['failure_codes'])


if __name__=='__main__':unittest.main()
