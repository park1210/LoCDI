import json,sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import app.locdi.live_handoff as live
from app.locdi.handoff_audit import audit_saved_success
from app.locdi.episode_exporter import _write
_write(audit_saved_success(),'reports/phase210_saved_success_v2.json',ROOT/'outputs','reports')
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'),pattern='test_*.py')
with TemporaryDirectory(dir=str(ROOT/'outputs')) as temp, (ROOT/'outputs/reports/phase210_regression_v1.txt').open('x',encoding='utf-8') as stream:
    with patch.object(live,'HANDOFF_ROOT',Path(temp)/'HANDOFF_TEST_ONLY'):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report={'tests':result.testsRun,'pass':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),'fail':len(result.failures),'errors':len(result.errors),'skip':len(result.skipped),'success':result.wasSuccessful(),'isolation':'Existing fixed 02903 plan test uses temporary HANDOFF_ROOT; real saved successful run preserved. No test suppressed or changed.','failures':[str(t) for t,_ in result.failures+result.errors],'skips':[(str(t),reason) for t,reason in result.skipped],'new_model_generations':0}
_write(report,'reports/phase210_regression_v1.json',ROOT/'outputs','reports')
print(json.dumps(report,ensure_ascii=True))
sys.exit(0 if result.wasSuccessful() else 1)
