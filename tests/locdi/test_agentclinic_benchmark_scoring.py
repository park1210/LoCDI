from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha
from app.locdi.agentclinic_scenario_adapter import adapt_test_record
from app.locdi.agentclinic_doctor_client import FakeDoctorClient
from app.locdi.agentclinic_doctor_runtime import run_doctor_scenario
from app.locdi.agentclinic_benchmark_scoring import evaluate_test_only,normalize,ApprovedAliases
from app.locdi.agentclinic_turn_runtime import TurnOptions
from agentclinic_offline_support import fixture,bundle
from agentclinic_json_support import JsonScript


def finish(s,prediction):
    return run_doctor_scenario(s,FakeDoctorClient(JsonScript([{'action':'DIAGNOSE','predicted_diagnosis':prediction}])))


class AgentClinicScoringTests(unittest.TestCase):
    def test_exact_normalization_is_lexical_only(self):
        s,key=adapt_test_record(fixture());e=finish(s,'  test_only_private_key.  ')
        r=evaluate_test_only([s],[e],[key])
        self.assertEqual(r['metrics']['exact_match_accuracy']['value'],1)
        self.assertEqual(r['clinical_equivalence'],'CLINICAL_EQUIVALENCE_NOT_VERIFIED')
        self.assertNotEqual(normalize('TEST-A'),normalize('TEST A'))

    def test_mismatch_partial_or_compound_requires_review_not_clinical_false(self):
        s,key=adapt_test_record(fixture())
        for prediction in ('TEST_ONLY_OTHER','TEST_ONLY_PRIVATE','TEST_ONLY_PRIVATE_KEY and TEST_ONLY_OTHER'):
            r=evaluate_test_only([s],[finish(s,prediction)],[key])
            self.assertEqual(r['metrics']['exact_match_accuracy']['value'],0)
            self.assertEqual(r['metrics']['review_required_count'],1)

    def test_unreviewed_alias_is_rejected(self):
        s,key=adapt_test_record(fixture())
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_ALIAS_FORBIDDEN'):
            evaluate_test_only([s],[finish(s,'TEST_ONLY_ALIAS')],[key],aliases={'x':'y'})

    def test_only_pinned_reviewed_cohort_scoped_alias_can_adjust(self):
        s,key=adapt_test_record(fixture());scope=digest([s.scenario_id])
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory:
            path=Path(directory)/'TEST_ONLY_aliases.json'
            record={'schema_version':'agentclinic.alias_policy.v1','status':'APPROVED','scope':scope,
                    'reviewed_by':'TEST_ONLY_REVIEWER','reviewed_at':'TEST_ONLY_TIME','procedure_version':'TEST_ONLY_V1',
                    'mapping':[{'reference':'TEST_ONLY_PRIVATE_KEY','aliases':['TEST_ONLY_ALIAS']}]}
            path.write_text(json.dumps(record),encoding='utf8');aliases=ApprovedAliases(path,sha(path),scope=scope)
            r=evaluate_test_only([s],[finish(s,'TEST_ONLY_ALIAS')],[key],aliases=aliases)
            self.assertEqual(r['metrics']['alias_adjusted_accuracy']['value'],1)
            self.assertEqual(r['metrics']['exact_match_accuracy']['value'],0)
            with self.assertRaises(ValueError):ApprovedAliases(path,'0'*64,scope=scope)
            record['status']='NOT_REVIEWED';path.write_text(json.dumps(record),encoding='utf8')
            with self.assertRaises(ValueError):ApprovedAliases(path,sha(path),scope=scope)

    def test_all_registered_cases_denominator_and_missing_failed(self):
        pairs=[adapt_test_record(fixture(),i) for i in range(1,5)];sc=[s for s,k in pairs];keys=[k for s,k in pairs]
        episodes=[finish(sc[0],'TEST_ONLY_PRIVATE_KEY'),finish(sc[1],'')]
        episodes.append(run_doctor_scenario(sc[2],FakeDoctorClient(JsonScript([{'action':'ASK','target':'history'}])),options=TurnOptions(1,1)))
        r=evaluate_test_only(sc,episodes,keys)
        self.assertEqual(r['metrics']['exact_match_accuracy']['denominator'],4)
        self.assertEqual(r['state_counts'],{'completed':1,'diagnosis_unresolved':1,'incomplete':1,'failed':1})
        self.assertEqual(r['metrics']['diagnosis_completion_rate']['value'],0.25)
        self.assertEqual(r['metrics']['turns']['denominator'],3)
        self.assertEqual(r['metrics']['valid_prediction_exact_accuracy']['denominator'],1)

    def test_invalid_prediction_and_failed_are_separate(self):
        s,key=adapt_test_record(fixture());e=finish(s,'TEST_ONLY_OTHER')
        e['result']['predicted_diagnosis']=None
        r=evaluate_test_only([s],[e],[key]);self.assertEqual(r['state_counts'],{'invalid_prediction':1})
        e=run_doctor_scenario(s,FakeDoctorClient(JsonScript([TimeoutError('TEST_ONLY')])) )
        self.assertEqual(evaluate_test_only([s],[e],[key])['state_counts'],{'failed':1})

    def test_duplicate_extra_cases_missing_keys_and_wrong_source_fail_closed(self):
        s,key=adapt_test_record(fixture());e=finish(s,'TEST_ONLY_OTHER')
        with self.assertRaises(ValueError):evaluate_test_only([s],[e,e],[key])
        with self.assertRaises(ValueError):evaluate_test_only([s],[e],[])
        e['source']['record_fingerprint']='bad'
        with self.assertRaises(ValueError):evaluate_test_only([s],[e],[key])

    def test_real_107_fake_predictions_never_become_benchmark_scores(self):
        b=bundle();r=evaluate_test_only(b.scenarios,[],b.scoring_keys)
        self.assertEqual(r['status'],'NOT_EVALUATED')
        self.assertNotIn('metrics',r)

    def test_zero_action_denominators_are_not_applicable(self):
        s,key=adapt_test_record(fixture());r=evaluate_test_only([s],[],[key])
        self.assertEqual(r['metrics']['repeated_action_rate']['status'],'NOT_APPLICABLE')
        self.assertIsNone(r['metrics']['turns']['median'])
