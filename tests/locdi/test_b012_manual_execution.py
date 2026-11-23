"""TEST_ONLY nonclinical fixtures; no HTTP/model calls."""
import contextlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import b012_manual_execution as manual
from scripts.audit_b012_failure_manual import schema_diagnosis


class TestManualB012(unittest.TestCase):
    def permission(self):
        now=datetime.now(timezone.utc)
        p={'fingerprint':'TEST_ONLY_PLAN','history_selection_fingerprint':'TEST_ONLY_HISTORY',
           'response_contract':'memory_citation_subset_code.v1'}
        spec={'fingerprint':'TEST_ONLY_POLICY'}
        a={'scope':'SYNTHETIC_SEQUENTIAL_ER_PILOT_ONLY','plan_fingerprint':p['fingerprint'],
           'policy_fingerprint':spec['fingerprint'],'evidence_type':'USER_EXPLICIT',
           'user_statement':'TEST_ONLY operator fixture; not actual approval',
           'accept_source_record_proxy_cutoff':True,'accept_fixed_five_minute_synthetic_turnaround':True,
           'authorize_sequential_source_and_pilot':True,'accept_reference_timing_limits':True,
           'clinical_release_approved':False,'v1_grants_created':0,'run_id':manual.RUN,
           'max_generation':360,'history_selection_fingerprint':p['history_selection_fingerprint'],
           'response_contract':p['response_contract'],'operator_approved':True,
           'no_other_ollama_work':True,'operator_ref':'TEST_ONLY_OPERATOR',
           'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=24)).isoformat()}
        return a,p,spec,now

    def test_valid_test_only_permission(self):
        a,p,s,n=self.permission();manual.validate_authorization(a,p,s,n)

    def test_missing_or_wrong_permissions_blocked(self):
        a,p,s,n=self.permission()
        for k in ('plan_fingerprint','run_id','max_generation','history_selection_fingerprint',
                  'response_contract','operator_approved','no_other_ollama_work','operator_ref',
                  'accept_source_record_proxy_cutoff','evidence_type'):
            b=deepcopy(a);b[k]=None
            with self.subTest(k=k),self.assertRaises(ValueError):manual.validate_authorization(b,p,s,n)

    def test_expired_and_future_and_excess_lifetime(self):
        for offset,end in ((-48,-24),(1,2),(0,25)):
            a,p,s,n=self.permission();a['recorded_at']=(n+timedelta(hours=offset)).isoformat()
            a['expires_at']=(n+timedelta(hours=end)).isoformat()
            with self.subTest(offset=offset,end=end),self.assertRaises(ValueError):manual.validate_authorization(a,p,s,n)

    def test_default_check_never_executes(self):
        with patch.object(sys,'argv',['manual','check']),patch.object(manual,'preflight',return_value={'generation':0}) as check,patch.object(manual,'execute') as run,contextlib.redirect_stdout(io.StringIO()):
            manual.main();check.assert_called_once_with(None,False);run.assert_not_called()

    def test_execute_requires_all_flags(self):
        for argv in (['execute'],['execute','--execute'],['execute','--execute','--allow-llm']):
            with patch.object(sys,'argv',['manual']+argv),patch.object(manual,'execute') as run,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                manual.main()
            run.assert_not_called()

    def test_monitor_and_analyze_dispatch_no_execute(self):
        for mode in ('monitor','analyze','prepare'):
            with patch.object(sys,'argv',['manual',mode]),patch.object(manual,mode) as fn,patch.object(manual,'execute') as run:
                manual.main();fn.assert_called_once_with();run.assert_not_called()

    def test_fresh_copy_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b';a.write_bytes(b'TEST_ONLY');b.write_bytes(b'EXISTING')
            with self.assertRaises(FileExistsError):manual.copy_exclusive(a,b)
            self.assertEqual(b.read_bytes(),b'EXISTING')

    def test_used_run_blocked(self):
        with tempfile.TemporaryDirectory() as d,patch.object(manual,'PR',Path(d)),patch.object(manual,'read',return_value={}),patch.object(manual,'pins',return_value={'status':'PASS'}):
            (Path(d)/manual.RUN).mkdir()
            with self.assertRaisesRegex(ValueError,'RUN_USED'):manual.preflight()


class TestStoredSchemaDiagnosis(unittest.TestCase):
    def fixture(self,indices,code='01000'):
        branch={'type':'object','properties':{'action':{'enum':['request_test']},
            'evidence_indices':{'type':'array','items':{'type':'integer'},'uniqueItems':True},
            'memory_subset_code':{'enum':['00000','01000']}},
            'required':['action','evidence_indices','memory_subset_code'],'additionalProperties':False}
        b={'request':{'format':{'anyOf':[branch]}}}
        a={'response_envelope':{'message':{'content':json.dumps({'action':'request_test',
            'evidence_indices':indices,'memory_subset_code':code})}}}
        return b,a

    def test_duplicate_evidence_reproduces_failure_without_repair(self):
        b,a=self.fixture([0,1,2,1]);original=deepcopy(a);r=schema_diagnosis(b,a)
        self.assertEqual(r['schema'],'INVALID');self.assertEqual(r['errors'][0]['validator'],'uniqueItems')
        self.assertEqual(r['errors'][0]['path'],['evidence_indices']);self.assertEqual(a,original)

    def test_valid_evidence_unchanged(self):
        b,a=self.fixture([0,1,2]);self.assertEqual(schema_diagnosis(b,a)['schema'],'VALID')

    def test_memory_enum_error_separate(self):
        b,a=self.fixture([0,1,2],'11111');r=schema_diagnosis(b,a)
        self.assertEqual(r['errors'][0]['path'],['memory_subset_code'])

    def test_missing_response_not_schema_failure(self):
        b,_=self.fixture([]);self.assertEqual(schema_diagnosis(b,{})['schema'],'NOT_CHECKED')


if __name__=='__main__':unittest.main()
