"""TEST_ONLY saved Source/response provenance. No network or actual model."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.development_routing_proxy import derive_rows,file_sha,admit_development_proxy,MARKERS
from app.locdi.llm_setup import digest
from app.locdi.routing_collection_preparation import project_input,response_schema
from app.llm.prompt_loader import specialty_system_prompt

ROOT=Path(__file__).resolve().parents[2]


class DevelopmentRoutingProxy(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)
        self.registry=json.loads((ROOT/'outputs/research_evaluation/final_exp_v1/routing_3b_agent_registry.json').read_text())['registry']
        self.agents=[a for a in self.registry['agents'] if a['role']=='specialty']
        self.config={'model':'qwen2.5:3b','model_identifier':'sha256:'+'a'*64,'max_requests':15}
        fps={a['agent_id']:digest(specialty_system_prompt(a['department'])) for a in self.agents}
        ready={'static_readiness':'READY','prepared':{'registry':self.registry,'prompt_fingerprints':fps}}
        self.patch=patch('app.locdi.development_routing_proxy.validate_setup',return_value=ready);self.patch.start();self.addCleanup(self.patch.stop)
        generation={'model_identifier':self.config['model_identifier'],'temperature':0,'seed':42,'max_output_tokens':1024,'prompt_fingerprints':fps}
        raw=[];candidates=[];records=[];catalog=['TEST_ONLY_DIAGNOSIS','TEST_ONLY_OTHER']
        for i in range(3):
            pid='TEST_ONLY_PATIENT_'+str(i);visit='TEST_ONLY_VISIT_'+str(i)
            row={'PATIENT':pid,'ENCOUNTER':visit,'CODE':'TEST_ONLY_CODE','SYSTEM':'TEST_ONLY_SYSTEM',
                 'DESCRIPTION':catalog[0],'START':'2000-01-01'};raw.append(row)
            obs={'chief_complaint':'TEST_ONLY complaint '+str(i),'demographics':{},'elapsed_minutes':1,
                 'current_symptoms':[],'history':[],'objective_findings':[],'longitudinal_memory':{'mode':'selective','records':[]}}
            decision={'patient_id':pid,'visit_id':visit,'decision_id':visit,'split':'train' if i<2 else 'validation',
                      'observation':obs,'generation':deepcopy(generation)}
            shared=project_input(obs,catalog)
            candidates.append({'case_ref':'TEST_ONLY_'+str(i),'decision':decision,'shared_payload':shared,
                'shared_input_fingerprint':digest(shared),'visit_started_at':'2000-01-01T00:00:00Z',
                'reference_candidates':{'source_condition_rows':[{'source_file':str(self.path/'conditions.csv'),
                    'source_row':i+2,'source_record':row}]}})
            for agent in self.agents:
                response={'action':'diagnose','diagnosis':catalog[0],'certainty':'supported','rationale':'TEST_ONLY response'}
                record={'decision_id':visit,'patient_id':pid,'agent_id':agent['agent_id'],'model':agent['model'],'version':agent['version'],
                    'observation_fingerprint':digest(obs),'configuration_fingerprint':digest(decision),'generation':deepcopy(generation),
                    'execution_status':'SUCCEEDED','response_contract_status':'VALID','response':response,
                    'response_fingerprint':digest(response),'observed_model_identifier':self.config['model_identifier']}
                records.append(record)
                req={'model':agent['model'],'format':response_schema(catalog),'messages':[
                    {'role':'system','content':specialty_system_prompt(agent['department'])},
                    {'role':'user','content':json.dumps({'department':agent['department'],**shared})}]}
                identity={'digest':self.config['model_identifier'],'context_length':32768}
                self.save('Request%04d_before.json'%len(records),{'request':req,'request_fingerprint':digest(req),'identity_before':identity})
                self.save('Request%04d_after.json'%len(records),{'status':'VALID','HTTP_status':200,'response_received':True,
                    'request_fingerprint':digest(req),'identity_after':identity,'response_envelope':{'message':{'content':json.dumps(response)}}})
        with (self.path/'conditions.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
        self.plan={'llm_config':self.config,'registry':self.registry,'candidates':candidates,'catalog':catalog,
                   'source_files':{str(self.path/'conditions.csv'):file_sha(self.path/'conditions.csv')},'code_files':{}}
        self.repin();self.collection={'records':records}

    def save(self,name,data):(self.path/name).write_text(json.dumps(data),encoding='utf-8')
    def repin(self):self.plan['fingerprint']=digest({k:v for k,v in self.plan.items() if k!='fingerprint'})
    def derive(self):return derive_rows(self.plan,self.collection,self.path,self.registry,self.config,{'TEST_ONLY_EVAL'})
    def replace_response(self,i,response):
        r=self.collection['records'][i-1];r.update(response=response,response_fingerprint=digest(response))
        path=self.path/('Request%04d_after.json'%i);a=json.loads(path.read_text());a['response_envelope']['message']['content']=json.dumps(response);self.save(path.name,a)

    def test_complete_train_groups_and_validation_excluded(self):
        r=self.derive();self.assertEqual(r['complete_train_groups'],2);self.assertEqual(len(r['rows']),10)
        self.assertTrue(all(x['split']=='train' and x['correct'] is True for x in r['rows']))

    def test_unresolved_not_false_and_partial_group_not_fitted(self):
        self.replace_response(1,{'action':'diagnose','diagnosis':'Unresolved','certainty':'unresolved','rationale':'TEST_ONLY'})
        r=self.derive();self.assertEqual(r['complete_train_groups'],1)
        self.assertIsNone(r['groups'][0]['outcomes'][0]['correctness_proxy']);self.assertEqual(len(r['rows']),5)

    def test_missing_reference_stays_null(self):
        self.plan['candidates'][0]['reference_candidates']['source_condition_rows']=[];self.repin()
        r=self.derive();self.assertEqual(r['complete_train_groups'],1)
        self.assertTrue(all(x['correctness_proxy'] is None for x in r['groups'][0]['outcomes']))

    def test_future_reference_stays_null(self):
        self.plan['candidates'][0]['visit_started_at']='1999-12-31T00:00:00Z';self.repin()
        r=self.derive();self.assertEqual(r['complete_train_groups'],1)

    def test_response_raw_mismatch_rejected(self):
        self.collection['records'][0]['response']['diagnosis']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'RESPONSE_CHANGED'):self.derive()

    def test_evaluation_patient_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'SPLIT_OVERLAP'):
            derive_rows(self.plan,self.collection,self.path,self.registry,self.config,{'TEST_ONLY_PATIENT_0'})

    def test_no_raw_without_unfitted_contract(self):
        with self.assertRaisesRegex(ValueError,'SCOPE_REQUIRED'):admit_development_proxy({},self.registry,[],self.config)

    def test_explicit_unfitted_artifact_never_claims_fit(self):
        self.save('plan.json',self.plan)
        e={'schema_version':'locdi.routing.unfitted_source_proxy.v1',**MARKERS,
            'origin_files':{str(self.path/'plan.json'):file_sha(self.path/'plan.json')},'collection_plan':str(self.path/'plan.json'),
            'collection_file':None,'collection_status':'NOT_COLLECTED','request_root':str(self.path),
            'fit_rows_fingerprint':digest([]),'routing_artifact':None}
        r=admit_development_proxy(e,self.registry,[],self.config)
        self.assertIsNone(r['artifact']);self.assertEqual(r['complete_train_groups'],0)
        self.assertEqual(r['utility_status'],'NO_ELIGIBLE_GROUPS_CORE_ABSTENTION')


if __name__=='__main__':unittest.main()
