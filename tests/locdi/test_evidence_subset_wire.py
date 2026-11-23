"""TEST_ONLY finite evidence citations, original Core validator preserved."""
from copy import deepcopy
import unittest
from jsonschema import Draft202012Validator,ValidationError
from app.locdi.evidence_subset_wire import encode,decode,allowed_codes,FIELD,MAX_ENUM_CODES
from app.locdi.current_ablation_llm_backend import _strict_object


class EvidenceSubsetContracts(unittest.TestCase):
    def fixture(self,action='diagnose',n=6):
        p={'evidence':['TEST_ONLY']*n,'instruction':'TEST_ONLY'}
        props={'action':{'enum':[action]},'evidence_indices':{'type':'array','maxItems':4,'uniqueItems':True,
            'items':{'type':'integer','minimum':0,'maximum':max(0,n-1)}}}
        return p,{'type':'object','properties':props,'required':list(props),'additionalProperties':False}

    def test_six_evidence_only_57_legal_combinations(self):
        codes=allowed_codes(6);self.assertEqual(len(codes),57);self.assertNotIn('0,1,2,3,4,5',codes)
        self.assertTrue(all(c=='NONE' or len(c.split(','))<=4 for c in codes))

    def test_round_trip_every_legal_code(self):
        p,s=self.fixture();_,w=encode(p,s)
        for code in w['properties'][FIELD]['enum']:
            v={'action':'diagnose',FIELD:code};Draft202012Validator(w).validate(v)
            self.assertEqual(decode(v,s)['evidence_indices'],[] if code=='NONE' else list(map(int,code.split(','))))

    def test_actual_failure_pattern_not_accepted_or_trimmed(self):
        p,s=self.fixture();_,w=encode(p,s);v={'action':'diagnose',FIELD:'0,1,2,3,4,5'}
        with self.assertRaises(ValidationError):Draft202012Validator(w).validate(v)
        with self.assertRaises(ValueError):decode(v,s)
        self.assertEqual(v[FIELD],'0,1,2,3,4,5')

    def test_non_diagnose_has_only_none(self):
        for action in ('request_test','ask_question'):
            p,s=self.fixture(action);_,w=encode(p,s);self.assertEqual(w['properties'][FIELD]['enum'],['NONE'])
            with self.assertRaises(ValueError):decode({'action':action,FIELD:'0'},s)

    def test_zero_evidence_only_none(self):
        p,s=self.fixture(n=0);_,w=encode(p,s);self.assertEqual(w['properties'][FIELD]['enum'],['NONE'])

    def test_duplicate_and_noncanonical_codes_rejected(self):
        _,s=self.fixture()
        for code in ('0,0','1,0','01','-1','1, 2','NONE,0','',None,True):
            with self.assertRaises(ValueError):decode({'action':'diagnose',FIELD:code},s)

    def test_unprovided_index_rejected_by_original_schema(self):
        _,s=self.fixture(n=2)
        with self.assertRaises(ValidationError):decode({'action':'diagnose',FIELD:'2'},s)

    def test_conflicting_legacy_fields_rejected(self):
        _,s=self.fixture()
        for key in ('evidence_indices','evidence_selection'):
            with self.assertRaises(ValueError):decode({'action':'diagnose',FIELD:'NONE',key:[]},s)

    def test_original_core_schema_runs_after_decode(self):
        _,s=self.fixture();s['properties']['action']['enum']=['request_test']
        with self.assertRaises(ValidationError):decode({'action':'diagnose',FIELD:'NONE'},s)

    def test_original_cap_lower_than_four_kept(self):
        p,s=self.fixture();s['properties']['evidence_indices']['maxItems']=2;_,w=encode(p,s)
        self.assertTrue(all(c=='NONE' or len(c.split(','))<=2 for c in w['properties'][FIELD]['enum']))

    def test_source_and_schema_not_modified(self):
        p,s=self.fixture();before=deepcopy((p,s));encode(p,s);self.assertEqual((p,s),before)

    def test_large_enumeration_blocks_before_generation(self):
        with self.assertRaisesRegex(ValueError,'ENUM_BUDGET'):allowed_codes(50)

    def test_invalid_counts_and_caps(self):
        for n,c in ((-1,4),(True,4),(6,5),(6,-1)):
            with self.assertRaises(ValueError):allowed_codes(n,c)

    def test_duplicate_json_key_still_fails(self):
        with self.assertRaises(ValueError):_strict_object('{"evidence_subset_code":"0","evidence_subset_code":"1"}')


if __name__=='__main__':unittest.main()
