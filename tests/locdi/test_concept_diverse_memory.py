"""TEST_ONLY nonclinical Source and rank fixtures, separate from actual cohort."""
from copy import deepcopy
import unittest
import ast
from pathlib import Path
from app.locdi.llm_setup import digest
from app.locdi.concept_diverse_memory import select,rank_candidates
from app.locdi.concept_diverse_memory_evaluation import (
    project_source_metadata,reconstruct_candidates,candidate_bounds,replacement_counts)
from app.locdi.memory_concept_evaluation import evaluate_query,aggregate


def key(c):return ['TEST_SYSTEM',c,'TEST_UNIT','NativeSyntheaObservation']


def c(mid,code,score=10,when='2020-01-02T00:00:00Z'):
    return {'memory_id':mid,'score':score,'available_at':when,'concept_key':key(code) if code else None}


def record(mid,text='alpha',patient='TEST_P',visit='TEST_OLD',when='2020-01-02T00:00:00Z',supersedes=None):
    return {'memory_id':mid,'patient_id':patient,'visit_id':visit,
        'visit_started_at':'2020-01-01T00:00:00Z','observed_at':when,'available_at':when,
        'source_id':'TEST_SOURCE_'+mid,'kind':'objective_finding','text':text,'supersedes':supersedes or []}


IDENTITY={'patient_id':'TEST_P','visit_id':'TEST_NOW','started_at':'2020-02-01T00:00:00Z'}


def reference(r,code='TEST_A',label='POSITIVE'):
    return {'memory_id':r['memory_id'],'original_record_fingerprint':digest(r),'source_type':'observations',
        'source_alignment':{'status':'VERIFIED_SOURCE_CODING','system_uri':'TEST_SYSTEM','code':code,'unit':'TEST_UNIT',
            'native':{'resource_type':'NativeSyntheaObservation','unit':'TEST_UNIT','coding':{'code':code},
                      'pointer':'/record/encounters/0/observations/0'}},
        'label':label,'reason':'TEST_ONLY','clinical_relevance':None,'stale':None,'conflict':None}


class SelectionTests(unittest.TestCase):
    def test_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError,'OPT_IN'):select([])

    def test_new_concept_outside_top5(self):
        cs=[c('TEST_%d'%i,'A',20-i) for i in range(5)]+[c('TEST_NEW','B',1)]
        r=select(cs,development_opt_in=True)
        self.assertIn('TEST_NEW',r['memory_ids']);self.assertEqual(len(r['memory_ids']),5)

    def test_reranking_top5_alone_cannot_invent_concept(self):
        cs=[c('TEST_%d'%i,'A',20-i) for i in range(5)]
        r=select(cs,development_opt_in=True);self.assertEqual(set(r['memory_ids']),{x['memory_id'] for x in cs})

    def test_missing_key_not_discarded_or_made_unique(self):
        cs=[c('TEST_U%d'%i,None,30-i) for i in range(5)]+[c('TEST_K','A',1)]
        r=select(cs,development_opt_in=True)
        self.assertEqual(r['memory_ids'],['TEST_K','TEST_U0','TEST_U1','TEST_U2','TEST_U3'])

    def test_all_unknown_preserves_baseline(self):
        cs=[c('TEST_%d'%i,None,20-i) for i in range(7)]
        self.assertEqual(select(cs,development_opt_in=True)['memory_ids'],['TEST_%d'%i for i in range(5)])

    def test_five_distinct_before_duplicates(self):
        cs=[c('TEST_FIRST','A',100),c('TEST_REPEAT','A',99)]+[c('TEST_'+s,s,5) for s in 'BCDE']
        self.assertNotIn('TEST_REPEAT',select(cs,development_opt_in=True)['memory_ids'])

    def test_under_budget_preserves_all(self):
        cs=[c('TEST_A','A'),c('TEST_B','A'),c('TEST_U',None)]
        self.assertEqual(len(select(cs,development_opt_in=True)['memory_ids']),3)

    def test_empty_candidates(self):self.assertEqual(select([],development_opt_in=True)['memory_ids'],[])

    def test_no_k_tuning(self):
        with self.assertRaisesRegex(ValueError,'TOP5'):select([],top_k=4,development_opt_in=True)

    def test_stable_score_time_id_tie(self):
        cs=[c('TEST_Z','Z'),c('TEST_A','A'),c('TEST_B','B',when='2020-01-03T00:00:00Z')]
        self.assertEqual([x['memory_id'] for x in rank_candidates(cs)],['TEST_B','TEST_A','TEST_Z'])

    def test_negative_or_zero_score_rejected(self):
        for n in [0,-1,True,1.0]:
            with self.assertRaises(ValueError):select([c('TEST_A','A',n)],development_opt_in=True)

    def test_unknown_extra_fields_rejected(self):
        a=c('TEST_A','A');a['other']=1
        with self.assertRaises(ValueError):select([a],development_opt_in=True)

    def test_gold_labels_rejected(self):
        for field in ['gold','label','positive','evaluation','correctness','clinical_relevance']:
            a=c('TEST_A','A');a[field]=[]
            with self.assertRaisesRegex(ValueError,'REFERENCE'):select([a],development_opt_in=True)

    def test_encoded_gold_rejected(self):
        a=c('{"gold":[]}','A')
        with self.assertRaisesRegex(ValueError,'REFERENCE'):select([a],development_opt_in=True)

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):select([c('TEST_A','A'),c('TEST_A','B')],development_opt_in=True)

    def test_exact_key_and_no_unit_merge(self):
        cs=[c('TEST_A','A'),c('TEST_B','A')];cs[1]['concept_key'][2]='OTHER_UNIT'
        self.assertTrue(all(t['pass']=='VERIFIED_DISTINCT_CONCEPT' for t in select(cs,development_opt_in=True)['selection_trace']))

    def test_reproducibility_and_immutability(self):
        cs=[c('TEST_A','A'),c('TEST_B','B')];before=deepcopy(cs)
        self.assertEqual(select(cs,development_opt_in=True),select(cs,development_opt_in=True));self.assertEqual(cs,before)


class ReconstructionTests(unittest.TestCase):
    def build(self,rs,query='alpha'):
        return reconstruct_candidates(rs,IDENTITY,query,[project_source_metadata(reference(r)) for r in rs])

    def test_actual_core_score_order(self):
        rs=[record('TEST_B','alpha beta'),record('TEST_A','alpha')]
        r=self.build(rs,'alpha beta')
        self.assertEqual(r['baseline_ids'],['TEST_B','TEST_A'])
        self.assertEqual([c['score'] for c in r['candidates']],[2,1])

    def test_zero_score_not_candidate(self):
        r=self.build([record('TEST_A'),record('TEST_ZERO','unrelated')])
        self.assertEqual(r['eligible_count'],2);self.assertEqual(len(r['candidates']),1)

    def test_patient_and_encounter_boundary(self):
        rs=[record('TEST_A'),record('TEST_OTHER',patient='TEST_OTHER'),record('TEST_CURRENT',visit='TEST_NOW')]
        self.assertEqual(self.build(rs)['baseline_ids'],['TEST_A'])

    def test_future_cutoff(self):
        rs=[record('TEST_A'),record('TEST_FUTURE',when='2020-03-01T00:00:00Z')]
        self.assertEqual(self.build(rs)['baseline_ids'],['TEST_A'])

    def test_superseded_filter(self):
        rs=[record('TEST_OLD'),record('TEST_REPLACED',when='2020-01-03T00:00:00Z',supersedes=['TEST_OLD'])]
        self.assertEqual(self.build(rs)['baseline_ids'],['TEST_REPLACED'])

    def test_source_binding_mismatch(self):
        r=record('TEST_A');m=project_source_metadata(reference(r));r['text']='changed'
        with self.assertRaisesRegex(ValueError,'BINDING'):reconstruct_candidates([r],IDENTITY,'alpha',[m])

    def test_source_metadata_missing_blocked(self):
        with self.assertRaisesRegex(ValueError,'METADATA_MISSING'):reconstruct_candidates([record('TEST_A')],IDENTITY,'alpha',[])

    def test_no_label_dependent_projection(self):
        r=reference(record('TEST_A'));a=project_source_metadata(r)
        for label in ['NEGATIVE','UNKNOWN']:
            r['label']=label;r['reason']='DO_NOT_CONSUME';r['clinical_relevance']=False
            self.assertEqual(a,project_source_metadata(r))

    def test_metadata_gold_rejected(self):
        r=record('TEST_A');m=project_source_metadata(reference(r));m['gold']=[]
        with self.assertRaisesRegex(ValueError,'REFERENCE'):reconstruct_candidates([r],IDENTITY,'alpha',[m])

    def test_reconstruction_does_not_change_records(self):
        rs=[record('TEST_A')];before=deepcopy(rs);self.build(rs);self.assertEqual(rs,before)


class EvaluationTests(unittest.TestCase):
    def test_cli_python39_parse_and_dry_run_flag(self):
        from scripts import evaluate_concept_diverse_memory as cli
        text=Path(cli.__file__).read_text(encoding='utf-8')
        ast.parse(text,feature_version=(3,9))
        self.assertIn('--enable-development-concept-diversity',text)

    def test_candidate_ceiling_not_a_policy(self):
        cs=[c('TEST_A','A'),c('TEST_B','B')]
        b=candidate_bounds(cs,[key('A'),key('B'),key('C')])
        self.assertEqual(b['known_gold_concept_coverage_upper_bound']['value'],2/3)
        self.assertEqual(b['known_gold_not_searchable_count'],1)

    def test_no_positive_bound_null(self):
        self.assertIsNone(candidate_bounds([],[])['known_gold_concept_coverage_upper_bound']['value'])

    def test_replacement_unknown_not_false(self):
        l={'A':'POSITIVE','B':'UNKNOWN','C':'NEGATIVE'}
        self.assertEqual(replacement_counts(['A','B'],['A','C'],l)['removed']['UNKNOWN'],1)
        self.assertEqual(l['B'],'UNKNOWN')

    def test_precision_coverage_tradeoff_separate(self):
        rs=[record('A'),record('B'),record('C')]
        refs=[reference(rs[0],'A'),reference(rs[1],'B'),reference(rs[2],'N','NEGATIVE')]
        q=[reference(rs[0],'A')['source_alignment'],reference(rs[1],'B')['source_alignment']]
        x=evaluate_query(refs,q,{'none':[],'full_history':['A','B','C'],'selective':['A','C']})
        self.assertEqual(x['modes']['selective']['record_precision']['value'],.5)
        self.assertEqual(x['modes']['selective']['concept_coverage']['value'],.5)
        self.assertEqual(aggregate([x],'selective')['concept_eligible_queries'],1)


if __name__=='__main__':unittest.main()
