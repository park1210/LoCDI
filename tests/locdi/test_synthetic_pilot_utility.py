"""TEST_ONLY contracts and real Source projection; never real LLM or real fitting."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from app.locdi.synthetic_pilot_source import load_zip,native_case,snapshot,score_response,MISSING_CONFIRMATION
from app.locdi.synthetic_pilot_collection import ZIP,M5,CONFIG,read,collect,verify_approval,MARKERS
from app.core.clinical_observation import encounter_evidence
from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.diagnostic_evidence import diagnostic_evidence
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import observed_tokens,fit_utility,RoutingUtility,jaccard_similarity

class SyntheticSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=read(M5);cls.bundle=load_zip(ZIP,cls.model['diagnoses']);cls.case=cls.bundle['cases'][0]
    def test_all15_alignment_and_exact_catalog(self):
        self.assertEqual(len(self.bundle['cases']),15)
        self.assertTrue(all(r['reference_diagnosis'] in self.model['diagnoses'] for r in self.bundle['references'].values()))
    def test_exact_member_scope_and_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../unsafe','TEST_ONLY')
            with self.assertRaisesRegex(ValueError,'ZIP_MEMBER'):load_zip(p,self.model['diagnoses'])
    def test_bad_unit_and_future_timing_rejected(self):
        for field,value in [('unit','WRONG'),('recorded_minute',100),('event_id','OTHER_CASE-OBS-01')]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'TEST_ONLY.zip'
                with zipfile.ZipFile(ZIP) as original,zipfile.ZipFile(p,'w') as z:
                    for n in original.namelist():
                        b=original.read(n)
                        if n=='training_cases_public.jsonl':
                            rows=[json.loads(s) for s in b.decode().splitlines()];rows[0]['test_timeline'][0][field]=value
                            b='\n'.join(json.dumps(r) for r in rows).encode()
                        z.writestr(n,b)
                with self.assertRaises(ValueError):load_zip(p,self.model['diagnoses'])
    def test_initial_has_no_test_values_or_reference(self):
        p,_=snapshot(self.case,self.model);text=json.dumps(p)
        self.assertEqual(p['observation']['objective_findings'],[]);self.assertEqual(p['released_measurements'],[])
        for key in ('reference_diagnosis','reference_specialty','test_timeline','provenance','CARD-01','reference_urls'):
            self.assertNotIn(key,text)
    def test_core_encounter_evidence_future_gate(self):
        engine,_=native_case(self.case,self.model);e=next(x for x in engine['current_encounter_events'] if x['action']=='Test')
        rt=HospitalCaseRuntime(case_id='TEST_ONLY',case_data=engine,arrival_time_min=0,arrival_index=0)
        rt.current_time_min=e['available_minute']-1;self.assertIsNone(encounter_evidence(rt,'Test',e['target']))
        rt.current_time_min=e['available_minute'];self.assertEqual(encounter_evidence(rt,'Test',e['target'])['evidence'],e['evidence'])
        self.assertEqual(rt.objective_findings,[]) # clock eligibility alone is not receipt
    def test_real_scheduler_orders_and_receives_all7(self):
        p,t=snapshot(self.case,self.model,stage='all_tests');self.assertEqual(len(p['released_measurements']),7)
        self.assertEqual(len(p['observation']['objective_findings']),7)
        releases=[e for e in t['events'] if e.get('event_type')=='result_received']
        self.assertEqual(len(releases),7)
        for m in p['released_measurements']:
            raw=next(r for r in self.case['test_timeline'] if r['code']==m['code'])
            self.assertEqual((m['value'],m['unit']),(raw['value'],raw['unit']))
    def test_same_snapshot_all_agents_no_case_prefix(self):
        from app.locdi.llm_setup import digest
        p,_=snapshot(self.case,self.model,stage='all_tests');before=deepcopy(p)
        self.assertEqual(len({digest({k:v for k,v in {'department':d,**p}.items() if k!='department'}) for d in ('cardiology','endocrinology','nephrology','neurology','pulmonology')}),1)
        self.assertEqual(before,p)
    def test_source_compatible_keeps_values_only_changes_view(self):
        a,_=snapshot(self.case,self.model,stage='all_tests');b,_=snapshot(self.case,self.model,stage='all_tests',view='source_compatible')
        self.assertEqual(a['released_measurements'],b['released_measurements'])
        self.assertEqual(a['observation']['objective_findings'],b['observation']['objective_findings'])
        self.assertEqual(b['observation']['current_symptoms'],[]);self.assertEqual(b['observation']['history'],[])
    def test_source_compatible_initial_has_no_routing_tokens(self):
        p,_=snapshot(self.case,self.model,view='source_compatible');self.assertEqual(observed_tokens(p['observation']),[])
    def test_narrative_is_not_objective_core_evidence(self):
        engine,_=native_case(self.case,self.model);rt=HospitalCaseRuntime(case_id='TEST_ONLY',case_data=engine,arrival_time_min=0,arrival_index=0)
        records,_=diagnostic_evidence(rt);self.assertEqual(records,[])
    def answer(self,diagnosis=None,certainty='supported'):
        return {'action':'diagnose','diagnosis':diagnosis or list(self.model['diagnoses'])[0],'certainty':certainty,'rationale':'TEST_ONLY'}
    def test_exact_diagnosis_not_reference_specialty(self):
        r={'reference_diagnosis':list(self.model['diagnoses'])[0],'reference_specialty':'TEST_ONLY_WRONG','reference_status':'NOT_INDEPENDENTLY_CLINICALLY_VERIFIED'}
        self.assertIs(score_response(self.answer(),r,self.model['diagnoses'])[0],True)
        self.assertIs(score_response(self.answer(list(self.model['diagnoses'])[1]),r,self.model['diagnoses'])[0],False)
    def test_unresolved_remains_null(self):
        self.assertIsNone(score_response(self.answer('Unresolved','unresolved'),{},self.model['diagnoses'])[0])
    def test_missing_confirmation_even_exact_remains_null(self):
        r={'reference_diagnosis':list(self.model['diagnoses'])[0],'reference_status':MISSING_CONFIRMATION}
        self.assertIsNone(score_response(self.answer(),r,self.model['diagnoses'])[0])
    def test_unknown_reference_remains_null(self):
        self.assertIsNone(score_response(self.answer(),{},self.model['diagnoses'])[0])
    def test_contradictory_or_unknown_output_rejected(self):
        for ans in (self.answer('Unresolved'),self.answer('NOT_IN_CATALOG')):
            with self.assertRaises(Exception):score_response(ans,{},self.model['diagnoses'])
    def test_no_authority_never_constructs_client(self):
        with patch('app.locdi.synthetic_pilot_collection.BudgetedNativeERClient') as c:
            with self.assertRaises(ValueError):collect('TEST_ONLY',execute=True)
            c.assert_not_called()
    def test_new_approval_scope_and_time(self):
        now=datetime.now(timezone.utc);p={'fingerprint':'TEST_ONLY','generation_cap':5}
        a={**MARKERS,'scope':'AUTHORED_SYNTHETIC_5AGENT_COLLECTION_ONLY','plan_fingerprint':'TEST_ONLY','generation_cap':5,
           'operator_approved':True,'accept_author_reference_proxy':True,'accept_synthetic_acquisition_timing':True,'no_other_ollama_work':True,
           'operator_ref':'TEST_ONLY_NOT_AUTHORITY','statement':'TEST_ONLY','recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
        verify_approval(a,p)
        a['generation_cap']=75
        with self.assertRaises(ValueError):verify_approval(a,p)
    def test_original_fit_formula_and_complete_group_contract(self):
        from app.locdi.llm_setup import validate_setup
        reg=AgentRegistry(validate_setup(read(CONFIG))['prepared']['registry']);agents=[a for a in reg.snapshot()['agents'] if a['role']=='specialty']
        rows=[{'patient_id':'TEST_ONLY_'+str(i),'decision_id':'TEST_ONLY_'+str(i),'agent_id':a['agent_id'],
               'split':'train','correct':i==0,'observation':{'chief_complaint':'TEST_ONLY stable'}} for i in range(2) for a in agents]
        artifact=fit_utility(rows,reg,min_samples=2);u=RoutingUtility(artifact,reg)
        self.assertAlmostEqual(u.score({'chief_complaint':'TEST_ONLY stable'},agents[0])['score'],.5)
        fitted_registry=AgentRegistry(reg.snapshot(),artifact)
        routed=fitted_registry.select_specialty({'chief_complaint':'TEST_ONLY stable'},{'records':[]},
                  [a['department'] for a in agents],router='adaptive')
        self.assertEqual(routed['decision_status'],'abstained')
        self.assertIsNone(routed['selected_specialty']) # existing tie abstention at 1e-12
        different=deepcopy(rows)
        for row in different:row['correct']=row['agent_id']==agents[0]['agent_id']
        differentiated=fit_utility(different,reg,min_samples=2)
        selected=AgentRegistry(reg.snapshot(),differentiated).select_specialty({'chief_complaint':'TEST_ONLY stable'},
             {'records':[]},[a['department'] for a in agents],router='adaptive')
        self.assertEqual(selected['selected_specialty'],agents[0]['department'])
        self.assertIsNone(u.score({'chief_complaint':'zzzzunique'},agents[0]));self.assertEqual(jaccard_similarity(['a'],['a','b']),.5)
        with self.assertRaises(ValueError):fit_utility(rows[:-1],reg,min_samples=2)
        rows[0]['correct']=None
        with self.assertRaises(ValueError):fit_utility(rows,reg,min_samples=2)

if __name__=='__main__':unittest.main()
