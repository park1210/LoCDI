"""No-inference Windows path/retention regression tests."""
from pathlib import Path
from copy import deepcopy
import tempfile,unittest,json
from unittest.mock import patch
from app.locdi.storage_path_budget import check_directory,validation_layout,evaluation_layout
from app.locdi.current_ablation_runner import _atomic_json
from scripts import native_transfer_recovery as recovery
from scripts import native_routing_transfer as transfer

class StorageRecoveryTests(unittest.TestCase):
    def test_old_endocrinology_layout_rejected_before_generation(self):
        old=recovery.OLDPR/recovery.OLDVAL/('run_'+recovery.OLDVAL)/'Patient01/endocrinology/Episode0001'
        self.assertEqual(len(str((old/'Request0001_core_response.json').resolve())),261)
        with self.assertRaisesRegex(ValueError,'STORAGE_PATH_BUDGET'):check_directory(old)

    def test_all_new_validation_role_paths_safe(self):
        agents=[{'department':d} for d in ('cardiology','endocrinology','nephrology','neurology','pulmonology')]
        result=validation_layout(recovery.PR/'v003/r003',agents)
        self.assertEqual(len(result),10);self.assertLess(max(x['reserved_max_file_chars'] for x in result),240)

    def test_future_full50_export_paths_safe(self):
        self.assertLess(evaluation_layout(recovery.PR/'f3/run_f3',250)['reserved_max_file_chars'],240)

    def test_unexpected_deep_workspace_fails_closed(self):
        with self.assertRaises(ValueError):check_directory(recovery.PR/('x'*100)/'endocrinology/Episode0001')

    def test_original_exclusive_writer_readback(self):
        with tempfile.TemporaryDirectory(dir=str(recovery.PR)) as d:
            f=Path(d)/'Request0001_core_response.json';_atomic_json(f,{'TEST_ONLY':True})
            self.assertEqual(json.loads(f.read_text(encoding='utf-8')),{'TEST_ONLY':True})

    def test_writer_still_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=str(recovery.PR)) as d:
            f=Path(d)/'runtime_snapshot_private.json';_atomic_json(f,{'original':True})
            with self.assertRaises(FileExistsError):_atomic_json(f,{'changed':True})
            self.assertEqual(json.loads(f.read_text(encoding='utf-8')),{'original':True})

    def test_scope_restores_legacy_paths_and_fit_reference(self):
        before=(transfer.PR,transfer.PUB,transfer.VALIDATION,transfer.FIT)
        with recovery.scope():
            self.assertEqual(transfer.PR,recovery.PR);self.assertEqual(transfer.VALIDATION,'v003')
            self.assertEqual(transfer.PR/transfer.FIT,recovery.OLDFIT)
        self.assertEqual((transfer.PR,transfer.PUB,transfer.VALIDATION,transfer.FIT),before)

    def test_no_recovery_fit_entry_point(self):
        self.assertFalse(hasattr(recovery,'fit'))

    def test_missing_new_proof_blocks(self):
        with tempfile.TemporaryDirectory() as d,patch.object(recovery,'PUB',Path(d)),patch.object(recovery,'storage_check',return_value={'status':'PATH_LAYOUT_PASS'}):
            self.assertEqual(recovery.check()['status'],'BLOCKED')

    def test_failed_preflight_does_not_load_approval_or_execute(self):
        with patch.object(recovery,'check',return_value={'status':'BLOCKED'}),self.assertRaisesRegex(ValueError,'PREFLIGHT_BLOCKED'):
            recovery.execute('Validation',Path('does_not_exist'))

if __name__=='__main__':unittest.main()
