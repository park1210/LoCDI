"""Additional saved Source-reference audit checks; not clinical review."""
import unittest
from test_original50_patient_audit import PR,PUB,read

class ReferenceCompleteness(unittest.TestCase):
    def test_private_condition_reference_rows_cover_all_fifty(self):
        data=read(PR/'reference_timing_complete.json')
        self.assertEqual([r['ordinal'] for r in data['rows']],list(range(1,51)))
        self.assertTrue(all(r['timing'] for r in data['rows']))
        self.assertTrue(all(not t['clinical_gold_approved'] and t['source_row_fingerprint'] and t['source_row']>=2
            and all(d[:10]==t['condition_started_at'][:10] for d in t['native_json_starts']) for r in data['rows'] for t in r['timing']))
    def test_reference_lookup_does_not_approve_current_decision_or_publicize_source_ids(self):
        s=read(PUB/'reference_timing_summary.json')
        self.assertEqual(s['patients_with_source_reference_timing'],50)
        self.assertFalse(s['current_public_event_store_contains_reference_conditions'])
        self.assertFalse(s['independent_clinical_review']);self.assertEqual(s['LLM_generation'],0)
        self.assertEqual(s['reference_rows'],sum(s['timing_record_counts'].values()))
        self.assertEqual(read(PUB/'final_coverage_with_reference_timing.json')['reference_timing_record_counts'],s['timing_record_counts'])
if __name__=='__main__':unittest.main()
