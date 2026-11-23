"""TEST_ONLY C4-14A GET boundary and frozen-policy checks; no generation."""
from copy import deepcopy
import unittest
from unittest.mock import Mock
from scripts.audit_routing_collection_environment import inventory, assess_environment, verify_stage
from app.locdi.routing_two_decision_collection import signed
from scripts.audit_routing_reference_feasibility import inventory_get


class EnvironmentAuditContracts(unittest.TestCase):
    def observed(self):
        return {'/api/version': {'version':'TEST_ONLY'}, '/api/tags': {'models':[{'name':'TEST_ONLY','digest':'a'*64}]}, '/api/ps': {'models':[]}}

    def test_inventory_get_only_no_post(self):
        client=Mock();client.get.return_value.json.return_value={}
        inventory(client)
        self.assertEqual(client.get.call_count,3);client.post.assert_not_called()
        self.assertEqual([x.args[0] for x in client.get.call_args_list],['http://127.0.0.1:11434/api/version','http://127.0.0.1:11434/api/tags','http://127.0.0.1:11434/api/ps'])

    def test_generation_endpoint_rejected_before_transport(self):
        client=Mock()
        with self.assertRaises(ValueError):inventory_get(client,'/api/generate')
        client.get.assert_not_called();client.post.assert_not_called()

    def test_digest_mismatch_blocks(self):
        r=assess_environment(self.observed(),'TEST_ONLY','sha256:'+'b'*64)
        self.assertIn('PINNED_MODEL_NOT_INSTALLED_OR_DIGEST_MISMATCH',r['blocker_codes'])

    def test_unloaded_not_claimed_actual_context(self):
        r=assess_environment(self.observed(),'TEST_ONLY','sha256:'+'a'*64)
        self.assertEqual(r['blocker_codes'],[]);self.assertIsNone(r['actual_loaded_context']);self.assertFalse(r['target_loaded'])

    def test_loaded_context_actual_only_when_present(self):
        obs=self.observed();obs['/api/ps']['models']=[{'name':'TEST_ONLY','context_length':4096}]
        self.assertEqual(assess_environment(obs,'TEST_ONLY','a'*64)['actual_loaded_context'],4096)

    def test_server_error_recorded_not_success(self):
        client=Mock();client.get.side_effect=TimeoutError('TEST_ONLY_PRIVATE')
        r=inventory(client)
        self.assertTrue(all(x['error_type']=='TimeoutError' for x in r.values()))
        self.assertIn('SERVER_INVENTORY_UNAVAILABLE',assess_environment(r,'TEST_ONLY','a'*64)['blocker_codes'])

    def plan(self):return signed({'generation':{'temperature':0,'seed':42,'max_output_tokens':1024,'timeout_seconds':120}})

    def test_one_and_ten_only(self):
        p=self.plan()
        for cap in (1,10):verify_stage(p,cap,p['fingerprint'])
        for cap in (True,0,2,11):
            with self.assertRaises(ValueError):verify_stage(p,cap,p['fingerprint'])

    def test_timeout_change_detected_even_with_new_digest(self):
        p=self.plan();p['generation']['timeout_seconds']=240;p=signed({k:v for k,v in p.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'POLICY_CHANGED'):verify_stage(p,1,p['fingerprint'])

    def test_plan_fingerprint_mismatch(self):
        p=self.plan()
        with self.assertRaisesRegex(ValueError,'FINGERPRINT_MISMATCH'):verify_stage(p,1,'0'*64)


if __name__=='__main__':unittest.main()
