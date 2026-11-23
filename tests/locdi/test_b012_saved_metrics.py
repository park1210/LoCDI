import unittest
from scripts.analyze_b012_full50_saved import structural_scores

class SavedMemoryMetricsTests(unittest.TestCase):
    def test_known_code_unit_overlap(self):
        m=structural_scores(['a','b'],{'a','c','d'},{('4548-4','%')})
        self.assertEqual(m['precision'],0.5);self.assertAlmostEqual(m['recall'],1/3)

    def test_no_public_query_is_not_zero_precision(self):
        m=structural_scores(['a'],set(),set())
        self.assertIsNone(m['precision']);self.assertIsNone(m['recall'])

    def test_zero_gold_recall_is_undefined(self):
        m=structural_scores(['a'],set(),{('4548-4','%')})
        self.assertEqual(m['precision'],0);self.assertIsNone(m['recall'])

    def test_memory_off_precision_undefined_recall_disabled_zero(self):
        m=structural_scores([],{'a'},{('4548-4','%')})
        self.assertIsNone(m['precision']);self.assertEqual(m['recall'],0)

    def test_duplicate_retrieved_ids_rejected(self):
        with self.assertRaises(ValueError):structural_scores(['a','a'],{'a'},{('4548-4','%')})

if __name__=='__main__':unittest.main()
