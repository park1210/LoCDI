"""D3 no-model dataset builder. Dry-run by default, --write new version only."""
import argparse
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.source_replay import read_sources,assemble,save_bundle
from validation.source_replay_validator import validate_source_bundle


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args(argv)
    try:
        bundle=assemble(read_sources())
        validation=validate_source_bundle(bundle)
        bundle['summary']['canonical_validation']=validation
        if args.write:
            manifest=save_bundle(bundle)
            print(json.dumps({'status':'CREATED','manifest_fingerprint':manifest['manifest_fingerprint'],**bundle['summary']},ensure_ascii=True))
        else:print(json.dumps({'status':'DRY_RUN',**bundle['summary']},ensure_ascii=True))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'BUILD_BLOCKED','error_type':type(exc).__name__,
            'reason':'SOURCE_OR_ADAPTER_CONTRACT_ERROR','location':[(Path(f.filename).name,f.lineno) for f in traceback.extract_tb(exc.__traceback__)],
            'new_llm_generations':0}));return 2


if __name__=='__main__':
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:raise RuntimeError('External calls disabled')
    sys.addaudithook(guard)
    raise SystemExit(main())
