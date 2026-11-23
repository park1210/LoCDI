"""TEST_ONLY recovery lifecycle, fake client only, all real sockets denied."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import routing_collection_recovery as recovery


class RecoveryLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.private=self.root/'private';self.public=self.root/'public';self.private.mkdir()
        d={'decision_id':'TEST_ONLY_DECISION','patient_id':'TEST_ONLY_PATIENT','source_ref':'TEST_ONLY_SOURCE',
           'observation':{},'generation':{},'tokens':[]}
        self.plan={'fingerprint':'a'*64,'llm_config':{},'catalog':['TEST_ONLY_DIAGNOSIS'],'source_files':{},
            'candidates':[{'case_ref':'TEST_ONLY_CASE','decision':d,'shared_payload':{}}],
            'registry':{'agents':[{'agent_id':'TEST_ONLY_AGENT','role':'specialty','department':'cardiology','model':'TEST_ONLY','version':'TEST_ONLY'}]}}
        now=datetime.now(timezone.utc)
        self.approval={'scope':recovery.SCOPE,'plan_fingerprint':self.plan['fingerprint'],'plan_id':recovery.PLAN_ID,
            'collection_id':recovery.COLLECTION_ID,'max_generation':15,'operator_approved':True,
            'accept_source_proxy_for_collection_only':True,'no_other_ollama_work':True,**recovery.MARKERS,
            'operator_ref':'TEST_ONLY_NOT_A_REAL_OPERATOR','user_statement':'TEST_ONLY_NOT_REAL_EXECUTION_APPROVAL',
            'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
        self.approval_path=self.root/'TEST_ONLY_approval.json';self.save_approval()
        self.created=[];owner=self
        class FakeClient:
            def __init__(self,*a,**kw):self.request_count=0;owner.created.append(self)
            def chat_json(self,*a,**kw):
                from app.locdi.current_ablation_runner import reserve_generation
                reserve_generation();self.request_count+=1;raise owner.failure
            def close(self):pass
        self.failure=KeyboardInterrupt()
        for p in (patch.object(recovery,'PRIVATE',self.private),patch.object(recovery,'PUBLIC',self.public),
                  patch.object(recovery,'load_checked',return_value=deepcopy(self.plan)),
                  patch.object(recovery,'BudgetedNativeERClient',FakeClient),
                  patch('socket.create_connection',side_effect=AssertionError('NO_REAL_NETWORK')),
                  patch('socket.socket.connect',side_effect=AssertionError('NO_REAL_NETWORK'))):
            p.start();self.addCleanup(p.stop)

    def save_approval(self):self.approval_path.write_text(json.dumps(self.approval),encoding='utf-8')

    def test_keyboard_interrupt_records_partial_and_public_status(self):
        with self.assertRaises(KeyboardInterrupt):recovery.execute(self.approval_path)
        raw=json.loads((self.private/recovery.COLLECTION_ID/'collection.json').read_text(encoding='utf-8'))
        public=json.loads((self.public/'new_collection_status.json').read_text(encoding='utf-8'))
        self.assertEqual(raw['generation'],1);self.assertEqual(raw['stop_reason'],'INTERRUPTED')
        self.assertEqual(len(raw['records']),1);self.assertIsNone(raw['records'][0]['correct'])
        self.assertEqual(raw['records'][0]['error_type'],'KeyboardInterrupt');self.assertEqual(public['stop_reason'],'INTERRUPTED')

    def test_first_ordinary_failure_stops_without_retry(self):
        self.failure=RuntimeError('TEST_ONLY')
        with self.assertRaises(SystemExit) as exc:recovery.execute(self.approval_path)
        self.assertEqual(exc.exception.code,2);self.assertEqual(self.created[0].request_count,1)
        public=json.loads((self.public/'new_collection_status.json').read_text(encoding='utf-8'))
        self.assertEqual(public['stop_reason'],'FIRST_TECHNICAL_FAILURE');self.assertEqual(public['responses_received'],0)

    def test_existing_collection_never_replaced(self):
        folder=self.private/recovery.COLLECTION_ID;folder.mkdir();marker=folder/'preserve.txt';marker.write_text('TEST_ONLY_KEEP')
        with self.assertRaises(FileExistsError):recovery.execute(self.approval_path)
        self.assertEqual(marker.read_text(),'TEST_ONLY_KEEP');self.assertEqual(self.created,[])

    def test_old_approval_cannot_authorize_new_collection(self):
        self.approval['scope']='CURRENT_3B_ROUTING_COLLECTION_15_ONLY';self.save_approval()
        with self.assertRaisesRegex(ValueError,'NEW_OPERATOR_APPROVAL'):recovery.execute(self.approval_path)
        self.assertEqual(self.created,[]);self.assertFalse((self.private/recovery.COLLECTION_ID).exists())


if __name__=='__main__':unittest.main()
