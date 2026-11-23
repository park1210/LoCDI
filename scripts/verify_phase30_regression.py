"""Identical discover/isolation wrapper, exclusive Phase 3.0 output only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_simulator_release import run_regression
from app.locdi.episode_exporter import _write

if __name__=='__main__':
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('REGRESSION_NO_NETWORK')
    sys.addaudithook(guard)
    root=ROOT/'outputs/research_protocol/phase30_regression_v1'
    root.mkdir(parents=True,exist_ok=False)
    result=run_regression(root)
    _write(result,'reports/regression.json',root,'reports')
    print(json.dumps(result,ensure_ascii=True))
    sys.exit(0 if result['status']=='PASS' else 1)
