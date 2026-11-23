"""TEST_ONLY algebraic AUBC joint-count and matched-denominator regression."""
import unittest
from scripts.analyze_voi_sequential_policies import aubc_rows, paired_comparison
from app.locdi.voi_sequential_policy_evaluation import SCORE_KEYS


class AnalysisContracts(unittest.TestCase):
    def records(self,policy,correct):
        return [{'scenario_ordinal':1,'policy':policy,'budget':k,'status':'completed','actual_actions':k,
                 'score':{x:(correct[k-1] if x=='top1' else 0.0) for x in SCORE_KEYS}} for k in range(1,6)]

    def test_discordant_budget_pairs_not_product_of_averages(self):
        rs=self.records('a',[1,1,0,0,0])+self.records('b',[0,0,1,1,0])
        r=paired_comparison(aubc_rows(rs,'a'),aubc_rows(rs,'b'),{1:{'patient_id':'TEST_ONLY'}},repetitions=5)
        self.assertEqual(r['correctness_joint_counts']['both'],0)
        self.assertAlmostEqual(r['correctness_joint_counts']['left_only'],.4)
        self.assertAlmostEqual(r['correctness_joint_counts']['right_only'],.4)
        self.assertAlmostEqual(r['correctness_joint_counts']['neither'],.2)

    def test_matched_random_fractional_joint(self):
        rs=self.records('a',[1,0,1,0,1])+self.records('b',[.25]*5)
        r=paired_comparison(aubc_rows(rs,'a'),aubc_rows(rs,'b'),{1:{'patient_id':'TEST_ONLY'}},repetitions=5)
        self.assertAlmostEqual(r['correctness_joint_counts']['both'],.15)
        self.assertAlmostEqual(sum(r['correctness_joint_counts'].values()),1)

    def test_missing_budget_refused(self):
        with self.assertRaises(ValueError):aubc_rows(self.records('a',[1]*5)[:-1],'a')


if __name__=='__main__':unittest.main()
