"""TEST_ONLY aggregation fixtures. No clinical labels/model/fitting."""
import unittest
from app.locdi.full50_evaluation_metrics import aggregate,aggregate_by_exposure


class MetricsContracts(unittest.TestCase):
    def row(self,condition,score=True,**kwargs):
        return dict(ordinal=1,condition=condition,plan_fingerprint='TEST_ONLY',status='COMPLETED',source_exact=score,**kwargs)
    def test_registered_denominator_and_not_attempted(self):
        r=aggregate([self.row('B0')],[1,2],plan_fingerprint='TEST_ONLY')['conditions']['B0']
        self.assertEqual(r['Source_Exact_registered_rate'],.5);self.assertEqual(r['states']['NOT_ATTEMPTED'],1)
    def test_unresolved_null_is_not_false(self):
        r=aggregate([self.row('B2'),self.row('B3',None,certainty='unresolved')],[1],plan_fingerprint='TEST_ONLY')
        self.assertEqual(r['pairs'][0]['transition'],'CORRECT_TO_UNRESOLVED_OR_UNSCORED')
        self.assertEqual(r['conditions']['B3']['unresolved'],1)
    def test_other_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'CROSS_RUN'):aggregate([self.row('B0')],[1],plan_fingerprint='OTHER')
    def test_duplicate_slots_are_rejected(self):
        with self.assertRaises(ValueError):aggregate([self.row('B0'),self.row('B0')],[1],plan_fingerprint='TEST_ONLY')
    def test_multiple_factual_tags_and_no_causal_claim(self):
        r=aggregate([self.row('B4',False,chief_complaint_missing=True,specialty_called=True,
            terminal_origin='M5_UNRESOLVED_LLM_NOT_TERMINAL')],[1],plan_fingerprint='TEST_ONLY')
        self.assertEqual(len(r['conditions']['B4']['nonexclusive_observation_tags']),4)
        self.assertEqual(r['causality'],'NOT_ESTABLISHED')
    def test_absent_routing_denominator_is_not_zero_percent(self):
        r=aggregate([self.row('B0')],[1],plan_fingerprint='TEST_ONLY')
        self.assertIsNone(r['conditions']['B0']['routing_coverage'])
    def test_exposure_strata_keep_full_denominator(self):
        rows=[self.row('B0')]
        r=aggregate_by_exposure(rows,[1,2],[1],plan_fingerprint='TEST_ONLY')
        self.assertEqual(r['all_registered']['conditions']['B0']['registered'],2)
        self.assertEqual(r['other40_also_development_exposed']['conditions']['B0']['registered'],1)


if __name__=='__main__':unittest.main()
