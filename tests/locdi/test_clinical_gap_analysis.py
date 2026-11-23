"""TEST_ONLY cross-workstream analysis; no verdict generation or execution."""
import copy,json,unittest
from app.locdi.clinical_gap_analysis import (field_counts,initial_matrix,routing_groups,split_matrix,
    review_comparison,temporal_matrix,baseline_dependencies,policy_options,gap_ledger,research_purposes)
from app.locdi.llm_setup import ROOT
from validation.review_submission_contract import REQUIRED


def collection():
    return {'registry':{'agents':[{'agent_id':'TEST_ONLY_%d'%i,'role':'specialty'} for i in range(5)]},
        'decisions':[{'decision_id':'TEST_ONLY_DECISION','split':'train'}],
        'outcomes':[{'decision_id':'TEST_ONLY_DECISION','agent_id':'TEST_ONLY_%d'%i,'correct':None,
            'response':{'certainty':'unresolved' if i==0 else 'supported','diagnosis':'TEST_ONLY_NAMED'}} for i in range(5)]}


def splits():
    return {'holdout':['TEST_ONLY_H'],'D3_replay':['TEST_ONLY_D'],'bank':['TEST_ONLY_T','TEST_ONLY_V'],
        'voi_train':['TEST_ONLY_T','TEST_ONLY_V'],'routing_train':['TEST_ONLY_T'],'routing_validation':['TEST_ONLY_V']}


class ClinicalGapContracts(unittest.TestCase):
    def test_empty_null_zero_distinct(self):
        r=field_counts([None,'',[],{},0,'TEST_ONLY']);self.assertEqual((r['null'],r['empty'],r['value']),(1,3,2))
    def test_unresolved_named_not_supported(self):
        r=routing_groups(collection());self.assertEqual((r['supported'],r['unresolved'],r['named_diagnosis_unresolved']),(4,1,1))
    def test_literal_unresolved_even_with_supported_certainty(self):
        d=collection();d['outcomes'][0]['response']={'certainty':'supported','diagnosis':'Unresolved'}
        self.assertEqual(routing_groups(d)['unresolved'],1)
    def test_five_stored_not_five_evaluable(self):
        r=routing_groups(collection());self.assertEqual(r['collection_complete_groups'],1);self.assertEqual(r['supported_only_complete_groups'],0)
    def test_missing_agent_not_complete(self):
        d=collection();d['outcomes'].pop();self.assertEqual(routing_groups(d)['collection_complete_groups'],0)
    def test_duplicate_agent_not_complete(self):
        d=collection();d['outcomes'][-1]['agent_id']=d['outcomes'][0]['agent_id'];self.assertEqual(routing_groups(d)['collection_complete_groups'],0)
    def test_correctness_never_generated(self):
        d=collection();before=copy.deepcopy(d);r=routing_groups(d);self.assertEqual(d,before);self.assertEqual(r['correctness_nonnull'],0);self.assertEqual(r['clinical_verdicts_generated'],0)
    def test_supported_only_is_upper_bound_not_clinical_readiness(self):
        d=collection();d['outcomes'][0]['response']['certainty']='supported';r=routing_groups(d)
        self.assertEqual(r['supported_only_complete_groups'],1);self.assertEqual(r['correctness_nonnull'],0)
    def split(self,**changes):
        m=splits();args=dict(observed_original=m['holdout'],observed_d3=m['D3_replay'],observed_bank=m['bank'],observed_voi=m['voi_train'],
            observed_routing={'routing_train':m['routing_train'],'routing_validation':m['routing_validation']},pilot_ids=['TEST_ONLY_P'],frozen_pilot_ids=['TEST_ONLY_P']);args.update(changes)
        return split_matrix(m,**args)
    def test_cross_module_validation_overlap(self):
        r=self.split();self.assertEqual(r['routing_validation_in_voi_train'],1);self.assertFalse(r['blind_test_approved'])
    def test_changed_membership_fails(self):self.assertEqual(self.split(observed_bank=['TEST_ONLY_OTHER'])['membership_recheck'],'FAIL')
    def test_pilot_mismatch_fails(self):self.assertEqual(self.split(pilot_ids=['TEST_ONLY_OTHER'])['membership_recheck'],'FAIL')
    def test_actual_collector_split_key_names(self):
        r=self.split(observed_routing={'train':['TEST_ONLY_T'],'validation':['TEST_ONLY_V']})
        self.assertEqual(r['membership_recheck'],'PASS')
    def test_private_memberships_not_echoed(self):self.assertNotIn('TEST_ONLY_',json.dumps(self.split()))
    def test_review_all_new_fields_covered(self):
        r=review_comparison();self.assertEqual({x['source_field'] for x in r['submission_to_legacy_fields']},set(REQUIRED))
    def test_review_all_legacy_required_fields_covered(self):
        r=review_comparison();s=json.loads((ROOT/'schemas/routing_review_results_v1.schema.json').read_text(encoding='utf8'))
        expected={scope+'.'+k for scope in ('reference','response','review','scope','axis') for k in s['$defs'][scope]['required']}
        self.assertEqual({x['source_field'] for x in r['legacy_required_fields']},expected)
    def test_no_automatic_conversion_or_authentication(self):
        r=review_comparison();self.assertTrue(all(x['lossless_conversion']=='NOT_PROVEN' for x in r['submission_to_legacy_fields']));self.assertEqual(r['evidence_authentication'],'NOT_IMPLEMENTED')
    def test_five_review_dimensions_remain_separate(self):self.assertEqual(len(review_comparison()['review_dimensions']),5)
    def test_non_diagnosis_verdict_gap_recorded(self):self.assertIn('NEW_REVIEWED_NON_DIAGNOSIS_TARGET_HAS_NO_GENERIC_AXIS_VERDICT_FIELD',review_comparison()['semantic_gaps'])
    def test_temporal_proxy_not_actual(self):
        r=temporal_matrix([], [{'action':'Test','proxy_available_minute':1,'actual_available_at':None}],[])
        self.assertEqual(r['D3_actual_available_count'],0);self.assertTrue(all(not x['clinical_time_proof'] for x in r['rows']))
    def test_source_consult_zero_not_logical_handoff(self):self.assertEqual(temporal_matrix([],[],[])['source_consult_count'],0)
    def test_baselines_preserved_and_artifact_scope(self):
        configs={n:json.loads((ROOT/'experiments/configs'/(n+'.json')).read_text(encoding='utf8')) for n in ('B0','B1','B2','B3','B4')};before=copy.deepcopy(configs)
        r=baseline_dependencies(configs);self.assertEqual(configs,before);self.assertEqual(r['rows'][0]['routing_utility'],'NOT_APPLICABLE');self.assertEqual(r['rows'][4]['voi_artifact'],'BLOCKED')
    def test_memory_labels_not_required_for_B0(self):
        c={'B0':{'features':{'memory_mode':'none'},'common':{},'ready':False}}
        self.assertEqual(baseline_dependencies(c)['rows'][0]['memory_labels'],'NOT_APPLICABLE')
    def test_policy_not_automatically_selected(self):
        r=policy_options();self.assertEqual(len(r),4);self.assertTrue(all(x['state']=='PROPOSED_NOT_SELECTED' for x in r))
    def test_partial_policy_requires_existing_validator_changes(self):self.assertIn('CORE_SAMPLE_VALIDATOR',policy_options()[2]['utility_formula_change'])
    def test_ledger_complete_and_no_approved_decisions(self):
        fields={'gap_id','category','affected_module','observed_evidence','current_state','research_impact','possible_resolutions','requires_human_review','requires_code_change','required_source','upstream_dependencies','downstream_blocked_tasks','verification_method','priority','recommended_stage'}
        r=gap_ledger();self.assertTrue(all(set(x)==fields and x['possible_resolutions']['status']=='PROPOSED' for x in r));self.assertEqual(len({x['gap_id'] for x in r}),len(r))
    def test_seven_research_purposes_no_clinical_promotion(self):
        r=research_purposes();self.assertEqual(len(r),7);self.assertTrue(all(x['clinical_evaluation']!='READY' for x in r))
    def test_analysis_is_json_finite(self):json.dumps({'review':review_comparison(),'ledger':gap_ledger(),'policies':policy_options()},allow_nan=False)


if __name__=='__main__':unittest.main()
