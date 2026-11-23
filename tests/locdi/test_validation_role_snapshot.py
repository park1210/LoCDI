"""TEST_ONLY role/feature contracts; no fitting or model calls."""
import unittest
from copy import deepcopy
from app.locdi.validation_role_snapshot import narrow_role,VERSION,ValidationPrefixTransport
from app.core.routing_utility import observed_tokens,jaccard_similarity,RoutingUtility,METHOD_V2
from app.core.agent_registry import AgentRegistry
from app.locdi.llm_setup import digest
from app.llm.prompt_loader import er_system_prompt,specialty_system_prompt
from scripts.run_validation_role_snapshot import load,DEFAULT
from scripts.check_routing_feature_readiness import check


class RoleSnapshotContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pub,cls.pr,cls.plan=load(DEFAULT)
    def template(self):
        return ({'role':'specialty','department':'cardiology','allowed_actions':['diagnose'],
            'observation':{'elapsed_minutes':13},'evidence':['TEST_ONLY'],'supported_diagnosis_eligible_indices':[0]},
            {'properties':{'action':{'enum':['diagnose']},'certainty':{'enum':['supported','unresolved']}}})
    def test_same_patient_fields_across_registered_roles(self):
        p,s=self.template();original=deepcopy(p);fps=set()
        for a in self.plan['registry']['agents']:
            role=a['role'];dept=a['department'] if role=='specialty' else ''
            q,ss,system,model=narrow_role(p,s,role=role,department=dept,registry=self.plan['registry'])
            fps.add(digest({k:q[k] for k in ('observation','evidence','supported_diagnosis_eligible_indices')}))
            self.assertEqual(system,er_system_prompt() if role=='er' else specialty_system_prompt(dept));self.assertEqual(model,a['model'])
        self.assertEqual(len(fps),1);self.assertEqual(p,original)
    def test_unregistered_specialty_is_rejected(self):
        p,s=self.template()
        with self.assertRaises(ValueError):narrow_role(p,s,role='specialty',department='TEST_ONLY_UNBOUND',registry=self.plan['registry'])
    def test_empty_native_features_have_zero_similarity(self):
        self.assertEqual(observed_tokens({'chief_complaint':'','current_symptoms':[],'objective_findings':[],
            'history':[],'demographics':{'age':80},'longitudinal_memory':{'records':[]}}),[])
        self.assertEqual(jaccard_similarity([],[]),0)
        self.assertEqual(jaccard_similarity([],['symptom','record']),0)
    def test_structural_fixture_can_validate_but_always_abstains(self):
        manifest=self.plan['registry'];registry=AgentRegistry(manifest)
        fixture={'schema_version':2,'method':METHOD_V2,'registry':manifest,'min_samples':2,'samples':[
            {'patient_id':'TEST_ONLY_'+str(i),'decision_id':'TEST_ONLY_'+str(i),'agent_id':a['agent_id'],
             'correct':a['department']=='cardiology','tokens':[]}
            for i in (1,2) for a in manifest['agents'] if a['role']=='specialty']}
        utility=RoutingUtility(fixture,registry) # validating a TEST_ONLY fixture, not fitting actual data
        for a in manifest['agents']:
            if a['role']=='specialty':self.assertIsNone(utility.score({'chief_complaint':'new symptoms'},a))
    def test_current_source_feature_readiness_is_blocked(self):
        result=check();self.assertEqual(result['status'],'BLOCKED');self.assertEqual(result['empty_feature_groups'],8)
        self.assertEqual(result['fit_calls'],0)
    def test_patient_value_mutation_blocks_before_new_post(self):
        req={'messages':[{'role':'system','content':'TEST_ONLY'},{'role':'user','content':'{"observation":{},"evidence":[],"supported_diagnosis_eligible_indices":[]}'}]}
        b={'request':req,'request_fingerprint':digest(req)};a={'status':'VALID','HTTP_status':200,'request_fingerprint':digest(req)}
        c=ValidationPrefixTransport([b],[a],prepared={},factory=lambda _:self.fail('NETWORK'),role='er',department='')
        p={'role':'er','department':'','terminal_request_contract':VERSION,'observation':{'invented':1},'evidence':[],'supported_diagnosis_eligible_indices':[]}
        import json
        wire={'messages':[{'role':'system','content':'TEST_ONLY'},{'role':'user','content':json.dumps(p)}]}
        with self.assertRaisesRegex(ValueError,'SNAPSHOT_CHANGED'):c.post('unused',json=wire)
        self.assertEqual(c.posts,0)


if __name__=='__main__':unittest.main()
