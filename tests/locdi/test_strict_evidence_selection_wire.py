"""TEST_ONLY empty-object grammar regression; no actual model calls."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator, ValidationError
from app.locdi.evidence_selection_wire import encode as encode_v1
from app.locdi.strict_evidence_selection_wire import encode, decode, VERSION
from app.locdi.current_ablation_llm_backend import _strict_object


class StrictEmptyGrammar(unittest.TestCase):
    def fixture(self, action='request_test', n=2):
        props={'action':{'enum':[action]},'evidence_indices':{'type':'array','maxItems':4,'uniqueItems':True,
            'items':{'type':'integer','minimum':0,'maximum':max(0,n-1)}}}
        s={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
        return {'instruction':'TEST_ONLY','evidence':['TEST_ONLY']*n},s

    def test_empty_request_has_no_selectable_property(self):
        p,s=self.fixture();_,w=encode(p,s);e=w['properties']['evidence_selection']
        self.assertEqual(e['properties'],{});self.assertEqual(e['enum'],[{}]);self.assertFalse(e['additionalProperties'])

    def test_observed_failure_still_rejected_not_silently_repaired(self):
        p,s=self.fixture();_,w=encode(p,s);bad={'action':'request_test','evidence_selection':{'0':True}}
        with self.assertRaises(ValidationError):Draft202012Validator(w).validate(bad)
        with self.assertRaises(ValueError):decode(bad,s)
        self.assertEqual(bad['evidence_selection'],{'0':True})

    def test_empty_request_decodes_to_existing_empty_array(self):
        p,s=self.fixture();_,w=encode(p,s);v={'action':'request_test','evidence_selection':{}}
        Draft202012Validator(w).validate(v);self.assertEqual(decode(v,s)['evidence_indices'],[])

    def test_empty_ask_same_rule(self):
        p,s=self.fixture('ask_question');_,w=encode(p,s)
        self.assertEqual(w['properties']['evidence_selection']['properties'],{})
        self.assertEqual(decode({'action':'ask_question','evidence_selection':{}},s)['evidence_indices'],[])

    def test_diagnosis_selection_and_original_schema_unchanged(self):
        p,s=self.fixture('diagnose');_,a=encode_v1(p,s);_,b=encode(p,s)
        self.assertEqual(a,b)
        self.assertEqual(decode({'action':'diagnose','evidence_selection':{'1':True}},s)['evidence_indices'],[1])

    def test_zero_evidence_even_diagnosis_has_empty_grammar(self):
        p,s=self.fixture('diagnose',0);_,w=encode(p,s)
        self.assertEqual(w['properties']['evidence_selection']['enum'],[{}])
        with self.assertRaises(ValidationError):Draft202012Validator(w).validate({'action':'diagnose','evidence_selection':{'0':True}})

    def test_unknown_false_or_five_citations_rejected(self):
        p,s=self.fixture('diagnose',5);_,w=encode(p,s)
        for selection in ({'9':True},{'0':False},{str(i):True for i in range(5)}):
            with self.assertRaises(ValidationError):Draft202012Validator(w).validate({'action':'diagnose','evidence_selection':selection})

    def test_v1_accepted_language_preserved(self):
        for action in ('request_test','ask_question','diagnose'):
            p,s=self.fixture(action);_,old=encode_v1(p,s);_,new=encode(p,s)
            for sel in ({},{'0':True},{'0':False},{'2':True},{'0':True,'1':True}):
                value={'action':action,'evidence_selection':sel}
                self.assertEqual(Draft202012Validator(old).is_valid(value),Draft202012Validator(new).is_valid(value))

    def test_input_and_old_schema_not_mutated(self):
        p,s=self.fixture();before=deepcopy((p,s));q,w=encode(p,s)
        self.assertEqual((p,s),before);self.assertIn(VERSION,q['instruction'])

    def test_duplicate_raw_keys_still_rejected(self):
        with self.assertRaises(ValueError):_strict_object('{"evidence_selection":{"0":true,"0":true}}')

    def test_exception_restores_existing_global_wire(self):
        import app.locdi.evidence_selection_wire as original
        old=original.encode
        with self.assertRaises(RuntimeError):
            with patch('app.locdi.evidence_selection_wire.encode',new=encode):raise RuntimeError('TEST_ONLY')
        self.assertIs(original.encode,old)


if __name__=='__main__':unittest.main()
