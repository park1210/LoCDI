"""D2 read-only audit; --save NEW_ID reserves separate Public/Private outputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.dataset_suitability import audit_suitability
from app.locdi.episode_exporter import _write
from app.locdi.llm_setup import ROOT, digest
from app.locdi.routing_collector import _id


def save_audit(public, private, run_id):
    name=_id(run_id)
    root=ROOT/'outputs'
    if root.resolve().parent != ROOT.resolve():
        raise ValueError('OUTPUT_ROOT_REDIRECTED')
    target=root/'reports'/(name+'.json')
    private_root=root/'private/dataset_audit'/name
    if not private_root.resolve().is_relative_to(root.resolve()) or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('OUTPUT_REDIRECTED')
    if target.exists() or private_root.exists():
        raise FileExistsError('AUDIT_ALREADY_EXISTS')
    if private.get('public_summary_fingerprint') != digest(public):
        raise ValueError('AUDIT_PAIR_MISMATCH')
    private_root.mkdir(parents=True,exist_ok=False)
    _write(private,'manifests/split_relations.json',private_root,'manifests')
    _write(public,'reports/'+name+'.json',root,'reports')
    return target,private_root/'manifests/split_relations.json'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',metavar='NEW_ID')
    args=parser.parse_args(argv)
    try:
        public,private=audit_suitability()
        if args.save:save_audit(public,private,args.save)
        print(json.dumps({'status':public['status'],'raw_patients':public['raw_patients'],
            'raw_encounters':public['raw_encounters'],'split':public['split'],
            'new_llm_generations':0,'utility_fit_count':0,'clinical_verdicts_generated':0},ensure_ascii=True))
        return 0 if not public['missing_sources'] else 2
    except Exception:
        print(json.dumps({'status':'AUDIT_BLOCKED','reason':'SOURCE_OR_STORAGE_CONTRACT_ERROR',
                         'new_llm_generations':0,'utility_fit_count':0}))
        return 2


if __name__=='__main__':
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:
            raise RuntimeError('External execution disabled')
    sys.addaudithook(guard)
    raise SystemExit(main())
