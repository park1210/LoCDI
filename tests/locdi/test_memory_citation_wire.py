"""TEST_ONLY wire shape; source clinical facts/legacy Core rules unchanged."""
from copy import deepcopy
import json
import unittest
from jsonschema import Draft202012Validator
from app.locdi.memory_citation_wire import encode,decode,VERSION
from app.locdi.current_ablation_llm_backend import _strict_object
import test_current_ablation_llm_backend as native_tests
MODEL=native_tests.MODEL


def fixture():
    payload={'instruction':'TEST_ONLY', 'observation':{'longitudinal_memory':{'records':[
        {'memory_id':'TEST_A','status_as_of_visit':'historical','text':'TEST_ONLY'},
        {'memory_id':'TEST_B','status_as_of_visit':'superseded','text':'TEST_ONLY'},
        {'memory_id':'TEST_C','status_as_of_visit':'historical','text':'TEST_ONLY'}]}}}
    schema={'type':'object','properties':{'memory_ids':{'type':'array','uniqueItems':True,
        'items':{'enum':['TEST_A','TEST_B','TEST_C']}}},'required':['memory_ids'],'additionalProperties':False}
    return payload,schema


class MemoryCitationWireTests(unittest.TestCase):
    def test_distinct_keys_map_exact_ids_without_dedup(self):
        p,s=fixture();before=deepcopy((p,s));q,w,m=encode(p,s)
        self.assertEqual((p,s),before);self.assertEqual(q['observation'],p['observation'])
        value={'memory_ids':{'0':True,'2':True}}
        Draft202012Validator(w).validate(value)
        self.assertEqual(decode(value,m,s),{'memory_ids':['TEST_A','TEST_C']})
    def test_superseded_and_out_of_range_rejected(self):
        p,s=fixture();_,w,m=encode(p,s)
        for k in ('1','3','-1','01'):
            with self.assertRaises(Exception):Draft202012Validator(w).validate({'memory_ids':{k:True}})
            with self.assertRaises(ValueError):decode({'memory_ids':{k:True}},m,s)
    def test_legacy_duplicate_array_never_repaired(self):
        p,s=fixture();_,w,m=encode(p,s)
        with self.assertRaises(Exception):Draft202012Validator(w).validate({'memory_ids':['TEST_A','TEST_A']})
        with self.assertRaises(ValueError):decode({'memory_ids':['TEST_A','TEST_A']},m,s)
    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(ValueError):_strict_object('{"memory_ids":{"0":true,"0":true}}')
    def test_empty_no_memory_maps_to_original_absence(self):
        s={'type':'object','properties':{},'additionalProperties':False}
        _,w,m=encode({'observation':{}},s)
        self.assertEqual(decode({'memory_ids':{}},m,s),{})
        with self.assertRaises(Exception):Draft202012Validator(w).validate({'memory_ids':{'0':True}})
    def test_duplicate_source_ids_not_hidden(self):
        p,s=fixture();p['observation']['longitudinal_memory']['records'][2]['memory_id']='TEST_A'
        with self.assertRaisesRegex(ValueError,'NOT_UNIQUE'):encode(p,s)
    def test_false_boolean_and_invented_field_rejected(self):
        p,s=fixture();_,w,m=encode(p,s)
        for obj in ({'memory_ids':{'0':False}},{'memory_ids':{},'gold':'TEST_ONLY'}):
            with self.assertRaises(Exception):Draft202012Validator(w).validate(obj)
    def test_original_max_citations_preserved(self):
        p,s=fixture();s['properties']['memory_ids']['maxItems']=1
        _,w,m=encode(p,s)
        with self.assertRaises(Exception):Draft202012Validator(w).validate({'memory_ids':{'0':True,'2':True}})
        with self.assertRaises(Exception):decode({'memory_ids':{'0':True,'2':True}},m,s)
    def test_closed_anyof_branches_keep_original_internal_contract(self):
        p,s=fixture();s={'anyOf':[s,{'type':'object','properties':{'action':{'const':'TEST_ONLY'}},
            'required':['action'],'additionalProperties':False}]}
        q,w,m=encode(p,s);value={'memory_ids':{'0':True}}
        Draft202012Validator(w).validate(value)
        self.assertEqual(decode(value,m,s),{'memory_ids':['TEST_A']})


class NativeWireTransportTests(unittest.TestCase):
    setUp=native_tests.CurrentAblationLLMBackend.setUp
    client=native_tests.CurrentAblationLLMBackend.client
    def test_opt_in_round_trip_raw_checkpoint_preserved(self):
        p,s=fixture();self.transport.envelope['message']['content']='{"memory_ids":{"0":true,"2":true}}'
        c=self.client(response_contract=VERSION)
        value=json.loads(c.chat_json('TEST_ONLY',json.dumps(p),model=MODEL,schema=s))
        self.assertEqual(value['memory_ids'],['TEST_A','TEST_C'])
        req=self.transport.posts[0][1];self.assertEqual(req['format']['properties']['memory_ids']['type'],'object')
        after=json.loads((c.checkpoint_root/'Request0001_after.json').read_text(encoding='utf-8'))
        self.assertEqual(after['response_envelope']['message']['content'],'{"memory_ids":{"0":true,"2":true}}')
        self.assertEqual(after['internal_response'],value)
        self.assertEqual(after['response_wire_contract'],VERSION)
    def test_actual_core_anyof_native_round_trip_with_saved_valid_response(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[2]
        folder=root/'outputs/private/research_evaluation/FINAL_LOCDI_PILOT_BASED_ACCURACY_IMPROVEMENT_v2/synthea_sequential_002/requests/Episode0002'
        before=json.loads((folder/'Request0002_before.json').read_text(encoding='utf-8'))['request']
        after=json.loads((folder/'Request0002_after.json').read_text(encoding='utf-8'))
        payload=json.loads(before['messages'][1]['content']);schema=before['format']
        original=json.loads(after['response_envelope']['message']['content'])
        self.assertEqual(len(set(original['memory_ids'])),len(original['memory_ids']))
        _,wire,mapping=encode(payload,schema)
        inverse={v:k for k,v in mapping.items()};answer=deepcopy(original)
        answer['memory_ids']={inverse[x]:True for x in original['memory_ids']}
        self.transport.envelope['message']['content']=json.dumps(answer)
        c=self.client(response_contract=VERSION)
        result=json.loads(c.chat_json(before['messages'][0]['content'],json.dumps(payload),model=MODEL,schema=schema))
        self.assertEqual(result,original)


if __name__=='__main__':unittest.main()
