"""TEST_ONLY reporting arithmetic; no model client or clinical labels."""
import unittest
from scripts.finalize_final_3b_results import paired_bootstrap, table
from scripts.finalize_c4_39_results import no_gold


class ReportingTests(unittest.TestCase):
    def test_zero_difference(self):
        r = paired_bootstrap([0, 1], [0, 1])
        self.assertEqual(r['B2_minus_B0'], 0)
        self.assertEqual(r['percentile_95_CI'], [0, 0])

    def test_all_positive(self):
        r = paired_bootstrap([0, 0], [1, 1])
        self.assertEqual(r['B2_minus_B0'], 1)
        self.assertEqual(r['percentile_95_CI'], [1, 1])

    def test_reproducible_saved_array_bootstrap(self):
        self.assertEqual(paired_bootstrap([0, 1, 0], [1, 0, 1]), paired_bootstrap([0, 1, 0], [1, 0, 1]))

    def test_no_denominator_not_zero_percent(self):
        with self.assertRaises(ValueError): paired_bootstrap([], [])
        self.assertIn('NOT_EVALUATED', table([{'TEST_ONLY': None}], ['TEST_ONLY']))

    def test_private_gold_detected(self):
        self.assertFalse(no_gold({'observation': {'evaluation': 'TEST_ONLY'}}))
        self.assertFalse(no_gold({'observation': {'source_event_store': []}}))
        self.assertTrue(no_gold({'question': 'TEST_ONLY source facts'}))


if __name__ == '__main__': unittest.main()
