"""TEST_ONLY cohort/protocol checks, no clinical reference or model calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.llm_setup import ROOT
from app.locdi.memory_reference_evaluation import evaluate, validate_review
from app.locdi.dataset_freeze import sha
from scripts.prepare_memory_task_protocol import query_inventory, select_cohort, write, metric_protocol, prepare
from test_memory_reference_evaluation import fixture


class MemoryTaskProtocolTests(unittest.TestCase):
    def packet(self):
        p, m, r = fixture()
        from app.locdi.memory_reference_evaluation import released_query
        _, projection = released_query(p['released_history'], snapshot_minute=p['decision_minute'], source_event=p['released_source_event'])
        p.update(case_ordinal=1, query_input_projection=projection, chief_complaint=None,
                 source_visibility_evidence={'physical_clinical_release_observed':False},
                 memory_provenance=[{'TEST_ONLY':True} for _ in p['eligible_memory']])
        return p, m, r

    def rows(self):
        p, _, _ = self.packet(); row = query_inventory([p])[0]
        return [dict(row, ordinal=i, snapshot_id='Snapshot%03d'%i,
                     eligible_records=10-i, source_link_count=10-i) for i in range(1,9)]

    def test_source_valid_does_not_become_clinically_reviewed(self):
        p, _, _ = self.packet(); row = query_inventory([p])[0]
        self.assertEqual(row['source_status'],'SOURCE_STRUCTURALLY_VALID')
        self.assertEqual(row['clinical_status'],'QUERY_NOT_REVIEWED')
        self.assertFalse(row['chief_complaint_present'])
        self.assertEqual(row['task_validity'],'UNDETERMINED_PENDING_INDEPENDENT_REVIEW')

    def test_query_reproduction_rejects_modified_projection(self):
        p, _, _ = self.packet(); p['query_input_projection']['TEST_ONLY_OTHER']='invalid'
        with self.assertRaisesRegex(ValueError,'QUERY_REPRODUCTION_FAILED'):query_inventory([p])

    def test_hidden_gold_rejected_before_inventory(self):
        p, _, _ = self.packet(); p['released_history'][0]['gold']='TEST_ONLY'
        with self.assertRaisesRegex(ValueError,'HIDDEN_FIELD_NOT_ALLOWED'):query_inventory([p])

    def test_fixed_smallest_pool_and_canonical_order(self):
        rows = self.rows(); before = deepcopy(rows)
        selected = select_cohort(rows)
        self.assertEqual([r['ordinal'] for r in selected],[4,5,6,7,8])
        self.assertEqual(select_cohort(list(reversed(rows))),selected)
        self.assertEqual(rows,before)

    def test_pool_size_ties_use_ordinal(self):
        rows = [dict(r,eligible_records=2,source_link_count=2) for r in self.rows()]
        self.assertEqual([r['ordinal'] for r in select_cohort(list(reversed(rows)))],[1,2,3,4,5])

    def test_predictions_or_rank_fields_never_select_cohort(self):
        rows=self.rows(); expected=select_cohort(rows)
        altered=[dict(r,prediction_correct=bool(r['ordinal']%2),retrieval_quality=100-r['ordinal']) for r in rows]
        self.assertEqual([r['snapshot_id'] for r in select_cohort(altered)], [r['snapshot_id'] for r in expected])

    def test_nonlinked_or_empty_pool_is_not_a_candidate(self):
        rows=self.rows(); rows[0]['source_link_count']=0;rows[1]['eligible_records']=0
        self.assertNotIn(1,[r['ordinal'] for r in select_cohort(rows)])
        self.assertNotIn(2,[r['ordinal'] for r in select_cohort(rows)])

    def test_small_or_invalid_cohort_rejected(self):
        for count in (1,4,11,True):
            with self.assertRaises(ValueError):select_cohort(self.rows(),count)
        with self.assertRaises(ValueError):select_cohort(self.rows()[:4])

    def test_missing_reference_never_calls_retrieval(self):
        with patch('app.locdi.memory_reference_evaluation.PatientMemoryStore.retrieve') as retrieve:
            with self.assertRaisesRegex(ValueError,'REVIEWED_REFERENCE_REQUIRED'):evaluate('UNUSED',None,None)
            retrieve.assert_not_called()

    def test_partial_annotations_cannot_use_full_pool_gate(self):
        p,m,r=self.packet(); r['snapshots'][0]['records'].pop()
        with self.assertRaisesRegex(ValueError,'ANNOTATION_COVERAGE_OR_ORDER'):validate_review(m,[p],r)

    def test_unjudged_is_not_a_negative_annotation(self):
        p,m,r=self.packet(); r['snapshots'][0]['records'][0]['relevant']=None
        with self.assertRaisesRegex(ValueError,'UNREVIEWED_OR_UNCERTAIN_RECORD'):validate_review(m,[p],r)

    def test_reviewer_metadata_and_query_review_are_required(self):
        p,m,original=self.packet()
        for fields in ({'reviewer_independent':False},{'reviewed_at':None},{'query_suitable':None}):
            r=deepcopy(original);r['snapshots'][0].update(fields)
            with self.assertRaises((ValueError,TypeError)):validate_review(m,[p],r)

    def test_no_overwrite_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            p=Path(folder)/'TEST_ONLY.json';write(p,{'a':1});before=sha(p)
            with self.assertRaises(FileExistsError):write(p,{'a':2})
            self.assertEqual(sha(p),before)

    def test_existing_namespace_rejected_before_source_reads(self):
        with patch('scripts.prepare_memory_task_protocol.load_bundle') as reader:
            with self.assertRaisesRegex(ValueError,'OUTPUT_EXISTS_NO_OVERWRITE'):prepare('c4_02_v1')
            reader.assert_not_called()

    def test_protocol_marks_partial_as_design_only(self):
        policy=metric_protocol()
        self.assertEqual(policy['recommended'],'FULL_POOL')
        self.assertEqual(policy['partial_pool']['status'],'DESIGN_ONLY_NOT_EXECUTABLE')
        self.assertEqual(policy['partial_pool']['full_pool_recall'],'FORBIDDEN')
        self.assertEqual(policy['unjudged'],'UNKNOWN_NEVER_FALSE')

    def test_partial_denominator_example_cannot_identify_full_recall(self):
        # Mathematical protocol fixture only, NOT a production partial scorer.
        selected={1,2,4,5}; judged={1,2,3}; known_relevant={1,3}
        self.assertEqual(len(selected & judged & known_relevant)/len(selected & judged), .5)
        self.assertEqual(len(selected & judged)/len(selected), .5)
        self.assertEqual(selected-judged,{4,5})
        # Same observed judgments admit different full truth sets and recall.
        possible_full_truth_a=known_relevant
        possible_full_truth_b=known_relevant | {4,5}
        recall_a=len(selected & possible_full_truth_a)/len(possible_full_truth_a)
        recall_b=len(selected & possible_full_truth_b)/len(possible_full_truth_b)
        self.assertEqual(recall_a,.5);self.assertEqual(recall_b,.75)
        self.assertNotEqual(recall_a,recall_b)


if __name__=='__main__':unittest.main()
