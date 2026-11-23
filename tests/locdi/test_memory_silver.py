"""TEST_ONLY weak-label contracts; invented tokens are not clinical judgments."""
from copy import deepcopy
import unittest

from scripts.evaluate_memory_silver import (registry, label_functions, combine, sampling,
    classify_all, coverage, evaluate_labels, validate_query_evidence, A, AX, B, STALE, CONFLICT, run)
from app.locdi.llm_setup import digest


def fixture():
    s = {'type':'observations','system':'TEST_ONLY_SYSTEM','code':'TEST_ONLY_C',
         'description':'TEST_ONLY_FULL_MEASUREMENT','units':'TEST_ONLY_U','value':0}
    return {'snapshot_id':'TEST_ONLY_S','memory_id':'TEST_ONLY_M','source':s,
            'released_facts':[dict(s,source_type='observations',value=1)],'structural_pass':True,
            'assertion_scope':'HISTORICAL_MEASUREMENT','cutoff_eligible_explicit_superseder':False,
            'evidence_fingerprint':'TEST_ONLY_E','query_fingerprint':'TEST_ONLY_Q'}


def votes(c):
    return {r['rule_id']:r['output'] for r in label_functions(c)}


class MemorySilverTests(unittest.TestCase):
    def test_registry_has_required_fields_and_verified_source_urls(self):
        r = registry()
        self.assertEqual(len(r['sources']),6)
        for rule in r['rules']:
            for field in ('rule_id','rule_version','target_task','target_axis','required_fields','permitted_output',
                          'source_reference','exact_source_claim','researcher_operational_assumption','limitations',
                          'code_location','priority','conflict_policy'):
                self.assertIn(field,rule)

    def test_explicit_coding_match_positive_only_for_task_a(self):
        v=votes(fixture())
        self.assertEqual(v['A02_EXPLICIT_CODING_MATCH'],'POSITIVE')
        self.assertEqual(v['B01_CLINICAL_EVIDENCE_GUARD'],'ABSTAIN')

    def test_missing_system_cannot_be_standardized_positive(self):
        c=fixture();c['source']['system']=None
        self.assertEqual(votes(c)['A02_EXPLICIT_CODING_MATCH'],'ABSTAIN')
        self.assertEqual(votes(c)['X01_LITERAL_MATCH'],'POSITIVE')

    def test_matching_description_not_concept_identity(self):
        c=fixture();c['source']['code']='TEST_ONLY_OTHER'
        self.assertEqual(votes(c)['A02_EXPLICIT_CODING_MATCH'],'ABSTAIN')
        self.assertEqual(votes(c)['X01_LITERAL_MATCH'],'POSITIVE')

    def test_unit_difference_never_converted(self):
        c=fixture();c['source']['units']='TEST_ONLY_OTHER_UNIT'
        v=votes(c)
        self.assertEqual(v['A02_EXPLICIT_CODING_MATCH'],'ABSTAIN')
        self.assertEqual(v['X01_LITERAL_MATCH'],'ABSTAIN')
        self.assertEqual(v['X02_LITERAL_NONMATCH'],'ABSTAIN')

    def test_missing_unit_or_value_abstains(self):
        for field in ('units','value'):
            c=fixture();c['source'][field]=None
            self.assertEqual(votes(c)['X01_LITERAL_MATCH'],'ABSTAIN')
            self.assertEqual(votes(c)['A02_EXPLICIT_CODING_MATCH'],'ABSTAIN')

    def test_numeric_zero_is_not_missing(self):
        self.assertEqual(votes(fixture())['X01_LITERAL_MATCH'],'POSITIVE')

    def test_nonmeasurement_negative_is_not_clinical_negative(self):
        c=fixture();c['source']['type']='medications'
        v=votes(c)
        self.assertEqual(v['A01_NONMEASUREMENT_SCOPE'],'NEGATIVE')
        self.assertEqual(v['X02_LITERAL_NONMATCH'],'NEGATIVE')
        self.assertEqual(v['B01_CLINICAL_EVIDENCE_GUARD'],'ABSTAIN')

    def test_different_literal_description_only_auxiliary_negative(self):
        c=fixture();c['source']['description']='TEST_ONLY_DIFFERENT_MEASUREMENT'
        v=votes(c)
        self.assertEqual(v['X02_LITERAL_NONMATCH'],'NEGATIVE')
        self.assertEqual(v['A02_EXPLICIT_CODING_MATCH'],'ABSTAIN')

    def test_no_partial_token_match_label(self):
        c=fixture();c['source']['description']='TEST_ONLY_FULL_MEASUREMENT_EXTRA'
        self.assertEqual(votes(c)['X01_LITERAL_MATCH'],'ABSTAIN')

    def test_incomplete_target_set_cannot_make_literal_negative(self):
        c=fixture();c['released_facts'][0]['units']=None
        self.assertEqual(votes(c)['X02_LITERAL_NONMATCH'],'ABSTAIN')

    def test_no_source_does_not_mean_negative(self):
        c=fixture();c['source']=None
        self.assertTrue(all(v=='ABSTAIN' for v in votes(c).values()))

    def test_structural_failure_forces_all_abstain(self):
        c=fixture();c['structural_pass']=False
        self.assertTrue(all(v=='ABSTAIN' for v in votes(c).values()))

    def test_historical_measurement_not_stale_from_change_or_age(self):
        c=fixture();c['cutoff_eligible_explicit_superseder']=True
        c['source']['value']='TEST_ONLY_OLD_VALUE'
        self.assertEqual(votes(c)['B02_EXPLICIT_CURRENT_ASSERTION_REPLACEMENT'],'ABSTAIN')

    def test_explicit_current_assertion_replacement_weak_stale(self):
        c=fixture();c['assertion_scope']='CURRENT_STATE_ASSERTION';c['cutoff_eligible_explicit_superseder']=True
        self.assertEqual(votes(c)['B02_EXPLICIT_CURRENT_ASSERTION_REPLACEMENT'],'POSITIVE')

    def test_absent_stop_does_not_mean_not_stale(self):
        self.assertEqual(votes(fixture())['B02_EXPLICIT_CURRENT_ASSERTION_REPLACEMENT'],'ABSTAIN')

    def test_numeric_change_is_not_conflict(self):
        c=fixture();c['released_facts'][0]['value']=9999
        self.assertEqual(votes(c)['B03_SAME_EVENT_CONFLICT_GUARD'],'ABSTAIN')

    def test_opposite_rules_abstain_no_majority(self):
        label,reason,conflict=combine([{'output':'POSITIVE'},{'output':'POSITIVE'},{'output':'NEGATIVE'}])
        self.assertEqual((label,reason,conflict),('ABSTAIN','RULE_CONFLICT',True))

    def test_multiple_agreeing_rules_not_double_counted(self):
        self.assertEqual(combine([{'output':'POSITIVE'},{'output':'POSITIVE'}])[:2],('ABSTAIN','MULTIPLE_APPLICABLE_RULES'))

    def test_no_rule_not_negative(self):
        self.assertEqual(combine([])[0],'ABSTAIN')

    def test_invalid_output_rejected(self):
        with self.assertRaisesRegex(ValueError,'INVALID_LF_OUTPUT'):combine([{'output':False}])

    def test_determinism_and_no_input_mutation(self):
        c=fixture();saved=deepcopy(c)
        self.assertEqual(classify_all([c],registry()),classify_all([c],registry()))
        self.assertEqual(c,saved)

    def test_task_and_axis_labels_remain_separate(self):
        _,silver=classify_all([fixture()],registry())
        self.assertEqual(len(silver),5)
        self.assertEqual({(r['task_id'],r['label_axis']) for r in silver},{A,AX,B,STALE,CONFLICT})
        self.assertTrue(all(r['silver_label']=='ABSTAIN' for r in silver if r['task_id']==B[0]))

    def test_leave_one_rule_out_sensitivity(self):
        c=fixture();_,base=classify_all([c],registry());_,variant=classify_all([c],registry(),disabled='X01_LITERAL_MATCH')
        old=next(r for r in base if (r['task_id'],r['label_axis'])==AX)
        new=next(r for r in variant if (r['task_id'],r['label_axis'])==AX)
        self.assertEqual((old['silver_label'],new['silver_label']),('POSITIVE','ABSTAIN'))

    def test_partial_precision_unjudged_and_zero_denominator(self):
        c=fixture();_,silver=classify_all([c],registry())
        selections={(c['snapshot_id'],m):([] if m=='none' else [c['memory_id'],'TEST_ONLY_UNJUDGED']) for m in ('none','full_history','selective')}
        result=evaluate_labels(silver,selections,{c['snapshot_id']:[c['memory_id']]})
        ax=next(r for r in result if (r['task_id'],r['axis'])==AX and r['mode']=='selective')
        self.assertEqual(ax['silver_judged_precision']['value'],1)
        self.assertEqual(ax['silver_judged_coverage']['value'],.5)
        self.assertEqual(ax['unjudged_count'],1)
        clinical=next(r for r in result if (r['task_id'],r['axis'])==B and r['mode']=='selective')
        self.assertIsNone(clinical['silver_judged_precision']['value'])
        self.assertEqual(clinical['unjudged_count'],2)
        self.assertEqual(clinical['full_pool_recall'],'NOT_IDENTIFIABLE_NOT_COMPUTED')
        off=next(r for r in result if r['mode']=='none')
        self.assertIsNone(off['silver_judged_coverage']['value'])

    def test_human_audit_selection_not_label_or_rank_driven(self):
        packets=[{'snapshot_id':'TEST_ONLY_%d'%i,'candidate_memories':[
            {'memory_id':'TEST_ONLY_O_%d'%i,'kind':'objective_finding'},
            {'memory_id':'TEST_ONLY_H_%d'%i,'kind':'history'}]} for i in range(50)]
        selected=sampling(packets)
        self.assertEqual(len(selected['selected']),20)
        modified=deepcopy(packets)
        for p in modified:
            for r in p['candidate_memories']:r.update(silver_label='TEST_ONLY',retrieval_score=999)
        self.assertEqual(selected,sampling(modified))

    def test_namespace_escape_and_completed_namespace(self):
        with self.assertRaisesRegex(ValueError,'INVALID_NAMESPACE'):run('../TEST_ONLY')
        with self.assertRaisesRegex(ValueError,'OUTPUT_EXISTS'):run('c4_08_v1')

    def test_query_evidence_fingerprint_and_binding(self):
        r={'snapshot_id':'TEST_ONLY','query_fingerprint':'TEST_ONLY_Q'};r['evidence_fingerprint']=digest(r)
        p={'snapshot_id':'TEST_ONLY','query_candidate':{'query_fingerprint':'TEST_ONLY_Q'}}
        validate_query_evidence([r],[p])
        r['query_fingerprint']='CHANGED'
        with self.assertRaisesRegex(ValueError,'QUERY_EVIDENCE_FINGERPRINT'):validate_query_evidence([r],[p])

    def test_query_evidence_order_preserves_cohort(self):
        with self.assertRaisesRegex(ValueError,'QUERY_EVIDENCE_ORDER'):
            validate_query_evidence([],[{'snapshot_id':'TEST_ONLY'}])
