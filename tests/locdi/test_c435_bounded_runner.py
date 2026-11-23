"""TEST_ONLY execution guards, no actual cohort evaluation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.evaluate_admitted_voi_cohort_bounded import Limits,private_bytes,execute

class BoundedRunner(unittest.TestCase):
    def test_memory_observable(self):self.assertGreater(private_bytes(),0)
    def test_memory_cap(self):
        with tempfile.TemporaryDirectory() as d:
            g=Limits(d,memory=lambda:4*1024**3)
            with self.assertRaisesRegex(ValueError,'MEMORY'):g.check()
    def test_wall_cap(self):
        with tempfile.TemporaryDirectory() as d:
            times=iter([0,3600]);g=Limits(d,clock=lambda:next(times),memory=lambda:1)
            with self.assertRaisesRegex(ValueError,'WALL'):g.check()
    def test_no_false_cap(self):
        with tempfile.TemporaryDirectory() as d:
            g=Limits(d,memory=lambda:1);g.check();self.assertEqual(g.peak_private,1)
    def test_dryrun_never_executes(self):
        with patch('scripts.evaluate_admitted_voi_cohort_bounded.verify',return_value={}),patch('scripts.evaluate_admitted_voi_cohort_bounded.run_approved') as run:
            execute(None,False);run.assert_not_called()
    def test_storage_cap(self):
        class Fake:
            st_size=10*1024**3
        with tempfile.TemporaryDirectory() as d:
            g=Limits(d,memory=lambda:1)
            p=Path(d)/'x';p.write_text('TEST_ONLY')
            with patch.object(Path,'rglob',return_value=iter([p])),patch.object(Path,'stat',return_value=Fake()),patch.object(Path,'is_file',return_value=True):
                with self.assertRaisesRegex(ValueError,'STORAGE'):g.check()
