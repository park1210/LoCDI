"""TEST_ONLY structural fixtures. No research labels, clinical facts or models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.source_replay import (assemble,iso,PROXY,STRICT,POLICY,load_replay_cases,
                                    atomic_rows,save_bundle)
from app.locdi.llm_setup import ROOT,digest
from app.locdi.case_adapter import adapt_case
from app.locdi.run_simulation import run_case
from app.locdi.episode_exporter import build_episode
from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.clinical_observation import doctor_observation,encounter_evidence
from validation.source_replay_validator import validate_source_bundle
from test_simulation import SourceActionClient,smoke_options
from smoke_support import check_boundary


def test_only_inputs():
    pid='TEST_ONLY_PATIENT';old='TEST_ONLY_PAST';current='TEST_ONLY_INDEX'
    def obs(visit,when,code='TEST_ONLY_CODE',category='laboratory'):
        return {'PATIENT':pid,'ENCOUNTER':visit,'DATE':iso(when),'CATEGORY':category,'CODE':code,
                'DESCRIPTION':'TEST_ONLY_ITEM','VALUE':'0','UNITS':'TEST_ONLY_UNIT','TYPE':'numeric'}
    def enc(eid,start):
        return {'row':{'Id':eid,'PATIENT':pid,'ENCOUNTERCLASS':'emergency','START':iso(start),'STOP':iso(start+120000)},
            'number':2 if eid==old else 3,'start_ms':start,'end_ms':start+120000,
            'json_pointer':'/TEST_ONLY','json_source':{'source_file':'TEST_ONLY.json','file_sha256':'0'*64},
            'symptoms':[],'conditions':[],'procedures':[],'imaging':[],'observations':[]}
    es={old:enc(old,1600000000000),current:enc(current,1600086400000)}
    es[old]['observations']=[(2,obs(old,1600000001000))]
    es[current]['observations']=[(3,obs(current,1600086400000,'72514-3','vital-signs')),
                                (4,obs(current,1600086401000))]
    es[current]['conditions']=[(2,{'PATIENT':pid,'ENCOUNTER':current,'START':'2020-09-18','STOP':'',
                                  'CODE':'TEST_ONLY_REFERENCE','DESCRIPTION':'TEST_ONLY_REFERENCE','SYSTEM':'TEST_ONLY'})]
    return {'patients':{pid:(2,{'Id':pid,'BIRTHDATE':'1980-01-01','GENDER':'TEST_ONLY'})},
        'encounters':es,'hashes':{'csv/'+n:'0'*64 for n in ('patients.csv','encounters.csv','observations.csv','conditions.csv')},
        'json_refs':{pid:{'source_file':'TEST_ONLY.json','file_sha256':'0'*64}},'memberships':{}}


class SourceReplayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=test_only_inputs();cls.bundle=assemble(cls.inputs)

    def test_policy_fixed_without_quotas_or_gold_and_source_immutable(self):
        before=digest(self.inputs);b=assemble(self.inputs)
        self.assertEqual(before,digest(self.inputs));self.assertFalse(POLICY['chief_complaint_required'])
        self.assertEqual(b['summary']['selected_patients'],1);self.assertEqual(b['summary']['selected_encounters'],1)
        self.assertEqual(b['summary']['clinical_gold_count'],0)
        self.assertEqual(validate_source_bundle(b)['violations'],0)

    def test_canonical_null_unknown_and_proxy_projection_are_separate(self):
        p=self.bundle['private'];initial=p['initial_observations.jsonl'][0];m=p['memory.json']['records'][0]
        self.assertIsNone(initial['chief_complaint']);self.assertIsNone(m['available_at']);self.assertTrue(m['availability_unknown'])
        w=p['generated_proxy_wrappers.jsonl'][0]
        self.assertNotIn('chief_complaint',w['agent_visible']['current_initial_state']['doctor_visible_info'])
        self.assertTrue(w['agent_visible']['past_visits'][0]['records'][0]['available_at'])
        strict=p['generated_strict_wrappers.jsonl'][0]
        self.assertEqual(strict['agent_visible']['past_visits'],[]);self.assertEqual(strict['environment']['current_encounter_events'],[])
        with self.assertRaisesRegex(ValueError,'STRICT_SOURCE'):load_replay_cases(policy=STRICT)
        with self.assertRaisesRegex(ValueError,'EXPLICIT_PROXY'):load_replay_cases(policy=PROXY)

    def test_memory_future_patient_and_source_value_tampering_rejected(self):
        for field,value,reason in [('patient_id','TEST_ONLY_OTHER','MEMORY_ISOLATION'),
            ('proxy_available_at','2099-01-01T00:00:00Z','MEMORY_CUTOFF'),('text','{}','MEMORY_CONTENT_CHANGED')]:
            b=deepcopy(self.bundle);b['private']['memory.json']['records'][0][field]=value
            with self.assertRaisesRegex(ValueError,reason):validate_source_bundle(b)

    def test_hidden_result_lookup_boundaries_and_initial_reference_isolation(self):
        w=self.bundle['private']['generated_proxy_wrappers.jsonl'][0];context=adapt_case(w)
        rt=HospitalCaseRuntime(context.case_id,context.runtime_case,0,0)
        for e in rt.case_data['current_encounter_events']:
            rt.current_time_min=e['available_minute']-1;self.assertIsNone(encounter_evidence(rt,e['action'],e['target']))
            rt.current_time_min=e['available_minute'];self.assertEqual(encounter_evidence(rt,e['action'],e['target']),e)
        initial=json.dumps(doctor_observation(rt));self.assertNotIn('TEST_ONLY_REFERENCE',initial)
        self.assertEqual(doctor_observation(rt)['objective_findings'],[])
        self.assertEqual(w['evaluation']['diagnosis_labels'],[])

    def test_future_release_and_arbitrary_value_and_consult_rejected(self):
        for field,value,reason in [('proxy_available_minute',0,'FUTURE_RESULT_RELEASE'),('value','TEST_ONLY_CHANGED','ACTION_SOURCE_VALUE_CHANGED'),
                                  ('action','Consult','CONSULT_UNSUPPORTED')]:
            b=deepcopy(self.bundle);b['private']['action_events.jsonl'][0][field]=value
            with self.assertRaisesRegex(ValueError,reason):validate_source_bundle(b)

    def test_unreviewed_references_and_split_promotion_blocked(self):
        b=deepcopy(self.bundle);b['private']['reference_candidates.jsonl'][0]['correct']=True
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_REFERENCE'):validate_source_bundle(b)
        b=deepcopy(self.bundle);b['private']['patient_split_manifest.json']['patients'][0]['assigned_split']='train'
        with self.assertRaisesRegex(ValueError,'UNAPPROVED_SPLIT'):validate_source_bundle(b)
        b=deepcopy(self.bundle);b['private']['patient_split_manifest.json']['legacy_patient_sets']={'holdout':['TEST_ONLY_PATIENT']}
        with self.assertRaisesRegex(ValueError,'LEGACY_PATIENT_OVERLAP'):validate_source_bundle(b)

    def test_units_types_are_separate_and_reported_survey_not_auto_ask(self):
        inputs=deepcopy(self.inputs);obs=inputs['encounters']['TEST_ONLY_INDEX']['observations'];r=deepcopy(obs[-1][1]);r['UNITS']='TEST_ONLY_OTHER_UNIT';obs.append((5,r))
        s=deepcopy(r);s.update(CODE='55758-7',CATEGORY='survey');obs.append((6,s))
        b=assemble(inputs);self.assertEqual(b['summary']['action_counts'],{'Ask':1,'Test':2})
        self.assertTrue(any(a.get('code')=='55758-7' and not a['execution_allowlisted'] for a in b['private']['action_candidates.jsonl']))

    def test_real_core_offline_replay_deterministic_and_no_leakage(self):
        w=self.bundle['private']['generated_proxy_wrappers.jsonl'][0];docs=[]
        for _ in range(2):
            client=SourceActionClient();result=run_case(w,options=smoke_options(client));context=adapt_case(w)
            self.assertEqual(result['status'],'completed');self.assertTrue(check_boundary(w,context,result,client)['release'])
            d=build_episode(result,context=context);docs.append(d)
            self.assertTrue(d['result']['diagnosis_unresolved']);self.assertEqual(d['clinical_metrics'],'NOT_EVALUATED')
            self.assertNotIn('TEST_ONLY_REFERENCE',json.dumps(client.calls))
        self.assertEqual(docs[0]['normalized_trace_fingerprint'],docs[1]['normalized_trace_fingerprint'])

    def test_atomic_save_no_clobber_and_private_manifest_paths(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root=Path(directory);private=root/'private';public=root/'version'
            with patch('app.locdi.source_replay.PRIVATE',private),patch('app.locdi.source_replay.PUBLIC',public):
                result=save_bundle(self.bundle)
                saved=json.loads((public/'dataset_manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(result,saved);self.assertTrue(all(x['private'] for x in saved['files'].values()))
                self.assertNotIn('TEST_ONLY_PATIENT',json.dumps(saved))
                before=(private/'generated_proxy_wrappers.jsonl').read_bytes()
                with self.assertRaises(FileExistsError):save_bundle(self.bundle)
                self.assertEqual(before,(private/'generated_proxy_wrappers.jsonl').read_bytes())
                with self.assertRaises(ValueError):atomic_rows(private,'../escape',[])


if __name__=='__main__':unittest.main()
