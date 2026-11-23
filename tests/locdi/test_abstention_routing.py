"""TEST_ONLY nonclinical contracts. Does not fit utility or construct LLM clients."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.agent_registry import AgentRegistry
from app.locdi.llm_setup import digest
from app.locdi.abstention_routing import (select,compare,metrics,reference_blank,policy_spec,
    validated_reviews,validate_cases,REFERENCE,SCOPE)
from app.locdi.routing_two_decision_collection import verify_pins


def fixture():
    agents=[{'agent_id':'TEST_ONLY_ER','role':'er','department':'','model':'TEST_ONLY','version':'TEST_ONLY','keywords':[]}]
    agents += [{'agent_id':'TEST_ONLY_'+d,'role':'specialty','department':d,'model':'TEST_ONLY','version':'TEST_ONLY','keywords':k}
        for d,k in [('cardiology',['chest pain']),('neurology',['weakness']),('nephrology',['creatinine']),('endocrinology',['diabetes']),('pulmonology',[])]]
    registry=AgentRegistry({'schema_version':1,'agents':agents})
    cases=[]
    for i,text in enumerate(['chest pain','weakness','nothing matched','chest pain weakness'],1):
        obs={'chief_complaint':text,'current_symptoms':[],'objective_findings':[],
             'history':[],'longitudinal_memory':{'mode':'selective','records':[]}}
        cases.append({'case_ref':'TEST_ONLY_%d'%i,'decision_id':'TEST_ONLY_D%d'%i,'patient_id':'TEST_ONLY_P%d'%i,
            'split':'validation','decision_time':'2026-01-01T00:00:00+00:00','observation':obs,
            'observable_input_fingerprint':digest(obs),'candidate_departments':[a['department'] for a in agents[1:]],
            'registry_fingerprint':digest(registry.snapshot()),'source_fingerprint':digest(['TEST_ONLY_SOURCE',i]),
            'evidence_fingerprint':digest(['TEST_ONLY_EVIDENCE',i])})
    return registry,cases


def reviewed(cases):
    ref=reference_blank(cases)
    for i,r in enumerate(ref['reviews']):
        r.update(independent_review_status='REVIEWED',source_timing_status='CURRENT_DECISION_REFERENCE_VALID',
            observation_sufficiency_review=True,recorded_time_proxy_review='ACCEPTED_FOR_DEVELOPMENT_ROUTING_ONLY',
            accepted_specialties=['cardiology','neurology'] if i==0 else ['neurology'],acceptable_abstention=i==2,
            abstention_review_status='REVIEWED',reviewer_rationale='TEST_ONLY contract assertion, not clinical evidence',
            reviewer_kind='TEST_ONLY',reviewer_qualification='TEST_ONLY',reviewer_private_ref='TEST_ONLY_REVIEWER',
            reviewer_independent=True,reviewed_at=datetime.now(timezone.utc).isoformat())
    return ref


class AbstentionRoutingContracts(unittest.TestCase):
    def setUp(self): self.reg,self.cases=fixture()
    def trace(self,condition='D_abstention_aware'):
        return [t for t in compare(self.cases,self.reg) if t['condition']==condition]
    def metric(self,ref=None,traces=None):
        return metrics(self.cases,self.trace() if traces is None else traces,reviewed(self.cases) if ref is None else ref,test_only=True)
    def choose(self,obs):
        return select(obs,self.reg,self.cases[0]['candidate_departments'],expected_registry_fingerprint=digest(self.reg.snapshot()))

    def test_unique_current_support(self):
        self.assertEqual(self.choose(self.cases[0]['observation'])['selected_specialty'],'cardiology')
    def test_ambiguous_abstention_no_tiebreak(self):
        r=self.choose(self.cases[3]['observation']);self.assertEqual(r['decision_status'],'abstained');self.assertIsNone(r['selected_specialty'])
    def test_no_support_abstention(self): self.assertEqual(self.choose(self.cases[2]['observation'])['rationale'],'NO_CURRENT_SUPPORT')
    def test_context_alone_no_selection(self):
        obs=deepcopy(self.cases[2]['observation']);obs['longitudinal_memory']['records']=[{'text':'chest pain','status_as_of_visit':'active'}]
        r=self.choose(obs);self.assertIsNone(r['selected_specialty']);self.assertEqual(r['support'][0]['support_status'],'CONTEXT_ONLY_SUPPORT')
    def test_superseded_memory_not_support(self):
        obs=deepcopy(self.cases[2]['observation']);obs['longitudinal_memory']['records']=[{'text':'chest pain','status_as_of_visit':'superseded'}]
        self.assertEqual(self.choose(obs)['support'][0]['context_keyword_matches'],[])
    def test_keyword_empty_not_clinical_negative(self):
        self.assertEqual(self.choose(self.cases[0]['observation'])['support'][-1]['support_status'],'NO_CONFIGURED_KEYWORD_SUPPORT')
    def test_repeat_order_and_input_unchanged(self):
        old=deepcopy(self.cases);self.assertEqual(compare(self.cases,self.reg),compare(self.cases,self.reg));self.assertEqual(old,self.cases)
    def test_four_conditions_same_snapshots(self):
        ts=compare(self.cases,self.reg)
        for c in self.cases:
            rows=[t for t in ts if t['decision_id']==c['decision_id']]
            self.assertEqual(len(rows),4)
            for k in ['input_fingerprint','memory_fingerprint','candidate_fingerprint','registry_fingerprint','source_fingerprint']:
                self.assertEqual(len({t[k] for t in rows}),1)
    def test_missing_utility_not_normal_abstention(self):
        ts=self.trace('C_adaptive_accuracy');m=self.metric(traces=ts)
        self.assertTrue(all(t['execution_status']=='MISSING_UTILITY' for t in ts));self.assertEqual(m['counts']['invalid'],4);self.assertEqual(m['counts']['abstained'],0)
        self.assertEqual(m['routing_coverage']['status'],'NOT_EVALUATED');self.assertEqual(m['execution_blockers'],['MISSING_UTILITY'])
    def test_unreviewed_reference_blocks_accuracy(self):
        m=self.metric(reference_blank(self.cases));self.assertEqual(m['reference_coverage']['value'],0)
        for k in ['routing_coverage','selective_routing_accuracy','incorrect_routing_rate','routing_abstention_rate','appropriate_abstention_rate']:
            self.assertEqual(m[k]['status'],'NOT_EVALUATED');self.assertIsNone(m[k]['value'])
    def test_multiple_accepted_specialties_and_rates(self):
        m=self.metric();self.assertEqual(m['counts']['acceptable_selection'],2)
        self.assertEqual(m['routing_coverage']['value'],.5);self.assertEqual(m['selective_routing_accuracy']['value'],1)
        self.assertEqual(m['routing_abstention_rate']['value'],.5);self.assertEqual(m['reference_coverage']['value'],1)
    def test_appropriate_and_inappropriate_separate(self):
        m=self.metric();self.assertEqual(m['appropriate_abstention_rate']['value'],.5);self.assertEqual(m['inappropriate_abstention_rate']['value'],.5)
        self.assertEqual(m['counts']['outside_accepted_set'],0)
    def test_abstention_reference_missing_not_evaluated(self):
        ref=reviewed(self.cases)
        for r in ref['reviews']:r.update(abstention_review_status='NOT_REVIEWED',acceptable_abstention=None)
        m=self.metric(ref);self.assertEqual(m['appropriate_abstention_rate']['status'],'NOT_EVALUATED');self.assertEqual(m['counts']['abstention_appropriateness_unreviewed'],2)
    def test_zero_selected_not_zero_percent(self):
        ts=self.trace()
        for t in ts:t['decision'].update(decision_status='abstained',selected_specialty=None)
        m=self.metric(traces=ts);self.assertIsNone(m['selective_routing_accuracy']['value']);self.assertEqual(m['selective_routing_accuracy']['status'],'NOT_APPLICABLE')
    def test_wrong_selection_denominator(self):
        ts=self.trace();ts[0]['decision']['selected_specialty']='pulmonology'
        m=self.metric(traces=ts);self.assertEqual(m['incorrect_routing_rate']['value'],.25);self.assertEqual(m['selective_routing_accuracy']['value'],.5)
    def test_invalid_decision_separate(self):
        for status,choice in [('selected','not_bound'),('abstained','cardiology'),('unavailable',None)]:
            ts=self.trace();ts[0]['decision'].update(decision_status=status,selected_specialty=choice)
            m=self.metric(traces=ts);self.assertEqual(m['counts']['invalid'],1)
    def test_reference_scope_not_correctness(self):
        r=reviewed(self.cases);r['target']='diagnosis_correctness'
        with self.assertRaisesRegex(ValueError,'TARGET'):self.metric(r)
    def test_missing_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'REFERENCE_REQUIRED'):metrics(self.cases,self.trace(),None,test_only=True)
    def test_future_reviewer_time_rejected(self):
        r=reviewed(self.cases);r['reviews'][0]['reviewed_at']='2100-01-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'REVIEW_TIME'):self.metric(r)
    def test_automatic_unreviewed_judgment_rejected(self):
        for k,v in [('accepted_specialties',['cardiology']),('acceptable_abstention',False),('reviewer_independent',True),('permitted_diagnoses',['TEST_ONLY'])]:
            ref=reference_blank(self.cases);ref['reviews'][0][k]=v
            with self.assertRaises(ValueError):self.metric(ref)
    def test_unreviewed_timing_not_approved(self):
        ref=reference_blank(self.cases);ref['reviews'][0]['source_timing_status']='CURRENT_DECISION_REFERENCE_VALID'
        with self.assertRaises(ValueError):self.metric(ref)
    def test_reviewer_not_independent(self):
        r=reviewed(self.cases);r['reviews'][0]['reviewer_independent']=False
        with self.assertRaises(ValueError):self.metric(r)
    def test_future_or_missing_timing_review_blocked(self):
        r=reviewed(self.cases);r['reviews'][0]['source_timing_status']='RETROSPECTIVE_OUTCOME_ONLY'
        with self.assertRaises(ValueError):self.metric(r)
    def test_not_evaluable_kept_in_registered_denominator(self):
        r=reviewed(self.cases);r['reviews'][0].update(independent_review_status='NOT_EVALUABLE',not_evaluable_reason='TEST_ONLY',accepted_specialties=None,acceptable_abstention=None)
        m=self.metric(r);self.assertEqual(m['reference_coverage']['value'],.75);self.assertEqual(m['registered_decisions'],4)
    def test_all_rows_required_no_favorable_drop(self):
        ref=reviewed(self.cases);ref['reviews'].pop()
        with self.assertRaises(ValueError):self.metric(ref)
        with self.assertRaises(ValueError):self.metric(traces=self.trace()[:-1])
    def test_split_overlap_rejected(self):
        self.cases[1].update(patient_id=self.cases[0]['patient_id'],split='train')
        with self.assertRaises(ValueError):compare(self.cases,self.reg)
    def test_utility_train_overlap_blocked(self):
        with self.assertRaises(ValueError):compare(self.cases,self.reg,training_patients=[self.cases[0]['patient_id']])
    def test_repeated_patient_cluster_not_duplicate_decision(self):
        self.cases[1]['patient_id']=self.cases[0]['patient_id'];m=self.metric()
        self.assertEqual(m['unique_registered_patients'],3);self.assertEqual(m['registered_decisions'],4)
    def test_train_scoring_only_explicit_development(self):
        for c in self.cases:c['split']='train'
        ref=reviewed(self.cases);ts=self.trace()
        with self.assertRaises(ValueError):metrics(self.cases,ts,ref,test_only=True)
        m=metrics(self.cases,ts,ref,test_only=True,purpose='development_reviewed');self.assertFalse(m['research_evaluation_ready'])
    def test_gold_nested_and_serialized_blocked(self):
        for value in [{'gold_department':'cardiology'},{'acceptable_abstention':True},'{"accepted_specialties":["cardiology"]}']:
            o=deepcopy(self.cases[0]['observation']);o['history']=[value]
            with self.assertRaises(ValueError):self.choose(o)
    def test_source_and_registry_fingerprint_mismatch(self):
        for key in ['source_fingerprint','registry_fingerprint','observable_input_fingerprint']:
            cs=deepcopy(self.cases);cs[0][key]='bad'
            with self.assertRaises(ValueError):compare(cs,self.reg)
    def test_reference_snapshot_mismatch(self):
        for k in ['source_fingerprint','registry_fingerprint','observable_input_fingerprint','evidence_fingerprint']:
            ref=reviewed(self.cases);ref['reviews'][0][k]='b'*64
            with self.assertRaises(ValueError):self.metric(ref)
    def test_source_file_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY';p.write_text('TEST_ONLY',encoding='utf-8')
            verify_pins({str(p):hashlib.sha256(p.read_bytes()).hexdigest()})
            with self.assertRaises(ValueError):verify_pins({str(p):'0'*64})
    def test_invalid_candidate_and_registry(self):
        for candidates in [[],['cardiology','cardiology'],['not_bound']]:
            with self.assertRaises(ValueError):select(self.cases[0]['observation'],self.reg,candidates,expected_registry_fingerprint=digest(self.reg.snapshot()))
        with self.assertRaises(ValueError):select(self.cases[0]['observation'],self.reg,['cardiology'],expected_registry_fingerprint='0'*64)
    def test_missing_utility_no_fabrication(self):self.assertIsNone(self.reg.utility);self.assertIsNone(policy_spec()['learned_success_probabilities'])
    def test_testonly_cannot_be_clinical_review(self):
        with self.assertRaises(ValueError):metrics(self.cases,self.trace(),reviewed(self.cases),test_only=False)
    def test_mixed_conditions_rejected(self):
        ts=self.trace();ts[0]['condition']='OTHER'
        with self.assertRaises(ValueError):self.metric(traces=ts)
    def test_trace_hash_mismatch(self):
        ts=self.trace();ts[0]['input_fingerprint']='b'*64
        with self.assertRaises(ValueError):self.metric(traces=ts)
    def test_core_and_gate_not_called_for_new_target(self):
        with patch('app.core.routing_utility.fit_utility',side_effect=AssertionError('FORBIDDEN')),patch('validation.routing_correctness_gate.to_policy_fit_rows',side_effect=AssertionError('FORBIDDEN')):
            self.metric();compare(self.cases,self.reg)


if __name__=='__main__':unittest.main()
