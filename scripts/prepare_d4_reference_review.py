"""Read-only by default; --write creates a new blinded reference-review package."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.reference_readiness import prepare_internal
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');args=p.parse_args()
    print(json.dumps(prepare_internal(write=args.write),ensure_ascii=True))
