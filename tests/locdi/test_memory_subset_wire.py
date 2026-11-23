"""TEST_ONLY finite codes and real Core anyOf schema, no model calls."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from jsonschema import Draft202012Validator
from app.locdi.memory_subset_wire import allowed_codes,encode,decode,FIELD,VERSION
from app.locdi.current_ablation_llm_backend import _strict_object
import test_current_ablation_llm_backend as native


def fixture(n=5):
    p={'instruction':'TEST_ONLY','observation':{'longitudinal_memory':{'records':[
        {'memory_id':'TEST_%d'%i,'status_as_of_visit':'historical'} for i in range(n)]}}}
    props={'action':{'enum':['TEST_ONLY']},'memory_ids':{'type':'array','uniqueItems':True,'maxItems':min(4,n),
        'items':{'enum':['TEST_%d'%i for i in range(n)]}}}
    s={'anyOf':[{'type':'object','properties':deepcopy(props),'required':['action','memory_ids'],'additionalProperties':False}]}
    return p,s


class SubsetWire(unittest.TestCase):
    def test_all0_to4_subsets_and_five_rejected(self):
        p,s=fixture();q,w,m=encode(p,s);self.assertEqual(len(m['codes']),31);self.assertNotIn('11111',m['codes'])
        for code in m['codes']:
            v={'action':'TEST_ONLY',FIELD:code};Draft202012Validator(w).validate(v)
            r=decode(v,m,s);self.assertEqual(len(r['memory_ids']),code.count('1'));self.assertEqual(len(set(r['memory_ids'])),len(r['memory_ids']))
        for code in ('11111','1','000000','abcde',5):
            with self.assertRaises(Exception):decode({'action':'TEST_ONLY',FIELD:code},m,s)
    def test_zero_records_and_superseded(self):
        self.assertEqual(allowed_codes([]),['0'])
        p,s=fixture(3);p['observation']['longitudinal_memory']['records'][1]['status_as_of_visit']='superseded'
        _,w,m=encode(p,s);self.assertTrue(all(x[1]=='0' for x in m['codes']))
        with self.assertRaises(Exception):decode({'action':'TEST_ONLY',FIELD:'010'},m,s)
    def test_old_bad_output_never_repaired(self):
        p,s=fixture();_,w,m=encode(p,s)
        for bad in ({'action':'TEST_ONLY','memory_ids':['TEST_0']*5},{'action':'TEST_ONLY',FIELD:'00000','memory_ids':[]}):
            with self.assertRaises(Exception):decode(bad,m,s)
    def test_source_duplicate_rejected_and_payload_unchanged(self):
        p,s=fixture();before=deepcopy((p,s));encode(p,s);self.assertEqual(before,(p,s))
        p['observation']['longitudinal_memory']['records'][1]['memory_id']='TEST_0'
        with self.assertRaises(ValueError):encode(p,s)
    def test_real_anyof_saved_failure_is_not_new_valid_response(self):
        root=Path(__file__).resolve().parents[2]/'outputs/private/research_evaluation/FINAL_LOCDI_B0_B4_EVALUATION_AND_ACCURACY_RECOVERY_v1/revision_02/synthea_b012_citation_set_001/requests/Episode0002'
        b=json.loads((root/'Request0001_before.json').read_text(encoding='utf-8'))['request']
        # The saved request used the previous object contract, so capture the
        # original Core schema from the preserved TEST_ONLY pre-wire packet.
        packet=json.loads((root.parents[2]/'TEST_ONLY_Episode0003.json').read_text(encoding='utf-8'))
        original=packet['first_request']['schema'];p=json.loads(b['messages'][1]['content'])
        q,w,m=encode(p,original)
        for branch in w['anyOf']:
            self.assertIn(FIELD,branch['properties']);self.assertNotIn('memory_ids',branch['properties'])
        saved=json.loads((root/'Request0001_after.json').read_text(encoding='utf-8'))['response_envelope']['message']['content']
        with self.assertRaises(Exception):Draft202012Validator(w).validate(_strict_object(saved))


class NativeSubset(unittest.TestCase):
    setUp=native.CurrentAblationLLMBackend.setUp
    client=native.CurrentAblationLLMBackend.client
    def test_raw_enum_and_internal_array_remain_distinct(self):
        p,s=fixture();self.transport.envelope['message']['content']=json.dumps({'action':'TEST_ONLY',FIELD:'11110'})
        c=self.client(response_contract=VERSION)
        result=json.loads(c.chat_json('TEST_ONLY',json.dumps(p),model=native.MODEL,schema=s))
        self.assertEqual(result['memory_ids'],['TEST_0','TEST_1','TEST_2','TEST_3'])
        a=json.loads((c.checkpoint_root/'Request0001_after.json').read_text(encoding='utf-8'))
        self.assertIn(FIELD,json.loads(a['response_envelope']['message']['content']))
        self.assertEqual(a['internal_response'],result)
    def test_enum_rejection_has_no_retry_or_truncation(self):
        p,s=fixture();self.transport.envelope['message']['content']=json.dumps({'action':'TEST_ONLY',FIELD:'11111'})
        c=self.client(response_contract=VERSION)
        with self.assertRaises(Exception):c.chat_json('TEST_ONLY',json.dumps(p),model=native.MODEL,schema=s)
        self.assertEqual(c.request_count,1);self.assertEqual(len(self.transport.posts),1)


if __name__=='__main__':unittest.main()
