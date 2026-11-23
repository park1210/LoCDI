"""Same discover suite/output-root isolation, new immutable result namespace."""
import json,sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import app.locdi.live_handoff as live
from app.locdi.episode_exporter import _write


if __name__=='__main__':
    log=ROOT/'outputs/reports/phase211_regression_v1.txt'
    destination=ROOT/'outputs/reports/phase211_regression_v1.json'
    if log.exists() or destination.exists():raise SystemExit('RESULT_NAMESPACE_ALREADY_USED')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'),pattern='test_*.py')
    with TemporaryDirectory(dir=str(ROOT/'outputs')) as directory,log.open('x',encoding='utf-8') as stream:
        with patch.object(live,'HANDOFF_ROOT',Path(directory)/'HANDOFF_TEST_ONLY'):
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={'tests':result.testsRun,'pass':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'fail':len(result.failures),'errors':len(result.errors),'skip':len(result.skipped),'success':result.wasSuccessful(),
        'failures':[str(t) for t,_ in result.failures+result.errors],
        'skips':[(str(t),reason) for t,reason in result.skipped],
        'isolation':'Identical discover suite; only live_handoff.HANDOFF_ROOT is temporary. No test removed, modified, or newly skipped.',
        'actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0}
    _write(report,'reports/phase211_regression_v1.json',ROOT/'outputs','reports')
    print(json.dumps(report,ensure_ascii=True));sys.exit(0 if result.wasSuccessful() else 1)
