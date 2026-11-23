"""TEST_ONLY inventory tests; invented symbols are not clinical judgments."""
from copy import deepcopy
import unittest

from app.locdi.llm_setup import digest
from scripts.audit_memory_review_progress import blank_progress


def fixture():
    packets = []
    reviews = []
    for i in range(1, 51):
        records = [{'memory_id': 'TEST_ONLY_%s_%s' % (i, j)} for j in range(20 if i <= 39 else 19)]
        packet = {'snapshot_id': 'OriginalCase%03d' % i,
                  'query_candidate': {'query_fingerprint': 'TEST_ONLY_QUERY'}, 'candidate_memories': records}
        packets.append(packet)
        reviews.append({'snapshot_id': packet['snapshot_id'], 'packet_fingerprint': digest(packet),
                        'query_fingerprint': 'TEST_ONLY_QUERY', 'review_status': 'UNREVIEWED',
                        'query_suitable': None, 'reviewer_private_ref': None, 'reviewed_at': None,
                        'records': [dict(r, relevant=None, stale=None, conflict=None) for r in records]})
    manifest = {'packets': [{'snapshot_id': p['snapshot_id']} for p in packets], 'fingerprint': 'TEST_ONLY_MANIFEST'}
    return manifest, packets, {'manifest_fingerprint': 'TEST_ONLY_MANIFEST', 'snapshots': reviews}


class ReviewProgressTests(unittest.TestCase):
    def test_full_cohort_and989_remain_unjudged(self):
        result = blank_progress(*fixture())
        self.assertEqual(len(result), 50)
        self.assertEqual(sum(r['pool_records'] for r in result), 989)
        self.assertTrue(all(r['query_status'] == 'NOT_REVIEWED' and not r['evaluable'] for r in result))

    def test_patient_removal_rejected(self):
        m, p, r = fixture()
        p.pop()
        with self.assertRaisesRegex(ValueError, 'ALL50_ORDER_REQUIRED'): blank_progress(m, p, r)

    def test_order_changes_rejected(self):
        m, p, r = fixture()
        r['snapshots'].reverse()
        with self.assertRaisesRegex(ValueError, 'ALL50_ORDER_REQUIRED'): blank_progress(m, p, r)

    def test_candidate_removal_rejected(self):
        m, p, r = fixture()
        r['snapshots'][0]['records'].pop()
        with self.assertRaisesRegex(ValueError, 'POOL_RECORD_ORDER'): blank_progress(m, p, r)

    def test_dataset_binding_rejected(self):
        m, p, r = fixture()
        r['manifest_fingerprint'] = 'OTHER'
        with self.assertRaisesRegex(ValueError, 'REFERENCE_MANIFEST_BINDING'): blank_progress(m, p, r)

    def test_query_binding_rejected(self):
        m, p, r = fixture()
        r['snapshots'][0]['query_fingerprint'] = 'OTHER'
        with self.assertRaisesRegex(ValueError, 'REFERENCE_PACKET_BINDING'): blank_progress(m, p, r)

    def test_automatic_false_not_counted_as_review(self):
        m, p, r = fixture()
        r['snapshots'][0]['records'][0]['relevant'] = False
        with self.assertRaisesRegex(ValueError, 'NOT_BLANK'): blank_progress(m, p, r)

    def test_real_submission_requires_separate_gate(self):
        m, p, r = fixture()
        r['snapshots'][0]['reviewer_private_ref'] = 'TEST_ONLY_REVIEWER'
        with self.assertRaisesRegex(ValueError, 'NOT_BLANK'): blank_progress(m, p, r)

    def test_input_unchanged(self):
        values = fixture()
        saved = deepcopy(values)
        blank_progress(*values)
        self.assertEqual(values, saved)
