"""TEST_ONLY CLI source-integrity boundary; no approval or evaluation created."""
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.evaluate_admitted_voi_cohort import admit

class AdmissionSourceBoundary(unittest.TestCase):
    def test_corrupt_raw_blocks_before_admission(self):
        def fake_read(path):
            name=Path(path).name
            if name.startswith('proposed_admission_contract'):return {'fingerprint':'TEST_CONTRACT'}
            if name=='proposed_panel_review.json':return {'TEST_ONLY':True}
            if name=='actual_generation_receipt.json':return {'raw_files':{'TEST_RAW':'TEST_HASH'}}
            raise AssertionError(name)
        with patch('scripts.evaluate_admitted_voi_cohort.verify',return_value={}),patch('scripts.evaluate_admitted_voi_cohort.read',side_effect=fake_read), \
             patch('scripts.evaluate_admitted_voi_cohort.check',return_value={'changed_or_missing':1}), \
             patch('scripts.evaluate_admitted_voi_cohort.approve_admission') as apply,patch('scripts.evaluate_admitted_voi_cohort.write') as write:
            with self.assertRaisesRegex(ValueError,'RAW_INTEGRITY'):admit(None,True)
            apply.assert_not_called();write.assert_not_called()

    def test_missing_legacy_scope_acceptance_blocks(self):
        def fake_read(path):
            name=Path(path).name
            if name.startswith('proposed_admission_contract'):return {'fingerprint':'TEST_CONTRACT'}
            if name=='proposed_panel_review.json':return {'TEST_ONLY':True}
            if name=='TEST_APPROVAL':return {'approved':True,'incomplete_legacy_source_review_accepted':False}
            if name=='actual_generation_receipt.json':return {'raw_files':{'TEST_RAW':'TEST_HASH'}}
            if name=='comparison_source_hashes.json':return {'TEST_SOURCE':'TEST_HASH'}
            raise AssertionError(name)
        with patch('scripts.evaluate_admitted_voi_cohort.verify',return_value={}),patch('scripts.evaluate_admitted_voi_cohort.read',side_effect=fake_read), \
             patch('scripts.evaluate_admitted_voi_cohort.check',return_value={'changed_or_missing':0}), \
             patch('scripts.evaluate_admitted_voi_cohort.approve_admission') as apply,patch('scripts.evaluate_admitted_voi_cohort.write') as write:
            with self.assertRaisesRegex(ValueError,'LEGACY_MISSING'):admit('TEST_APPROVAL',True)
            apply.assert_not_called();write.assert_not_called()
