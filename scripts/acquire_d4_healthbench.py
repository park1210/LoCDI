"""D4 official immutable raw acquisition. No models; no downloaded code executed."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.llm_setup import ROOT,digest
from app.locdi.source_replay import atomic_rows

REVISION='40ee1968852fc57f625934251ac22be47077a8fb'
CODE_REVISION='652c89d0ca9df547706735883097e9537d40dc47'
PRIVATE=ROOT/'outputs/private/external_benchmarks/healthbench'/REVISION/'raw'
PUBLIC=ROOT/'datasets/external/healthbench'/REVISION
FILES={
    'README.md':'https://huggingface.co/datasets/openai/healthbench/resolve/'+REVISION+'/README.md',
    '2025-05-07-06-14-12_oss_eval.jsonl':'https://huggingface.co/datasets/openai/healthbench/resolve/'+REVISION+'/2025-05-07-06-14-12_oss_eval.jsonl',
    'healthbench_eval.py':'https://raw.githubusercontent.com/openai/simple-evals/'+CODE_REVISION+'/healthbench_eval.py',
    'EVALUATOR_LICENSE':'https://raw.githubusercontent.com/openai/simple-evals/'+CODE_REVISION+'/LICENSE',
}


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'LoCDI-D4-ReadOnly-Acquisition'})
    with urllib.request.urlopen(request,timeout=60) as response:
        data=response.read(100*1024*1024+1)
        if len(data)>100*1024*1024:raise ValueError('SOURCE_SIZE_LIMIT')
        return data,dict(response.headers)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true');args=p.parse_args()
    if not args.download:print(json.dumps({'status':'DRY_RUN','revision':REVISION,'source_urls':FILES,'model_calls':0}));return
    if PRIVATE.exists() or PUBLIC.exists():raise FileExistsError('IMMUTABLE_ACQUISITION_ALREADY_EXISTS')
    readme,headers=fetch(FILES['README.md'])
    if 'license: mit' not in readme.decode('utf-8').lower():raise ValueError('LICENSE_NOT_CONFIRMED')
    PRIVATE.mkdir(parents=True,exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    entries={}
    for name,url in FILES.items():
        data,h=(readme,headers) if name=='README.md' else fetch(url)
        path=PRIVATE/name
        # Atomic byte-preserving no-clobber publish.
        import tempfile,os
        with tempfile.NamedTemporaryFile(dir=str(PRIVATE),delete=False) as stream:
            temporary=Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
        try:os.link(str(temporary),str(path))
        finally:temporary.unlink()
        entries[name]={'source_url':url,'downloaded_at':datetime.now(timezone.utc).isoformat(),
            'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'etag':h.get('ETag'),'private':True,'revision':REVISION if name in {'README.md','2025-05-07-06-14-12_oss_eval.jsonl'} else CODE_REVISION}
    from app.locdi.external_benchmark_contract import inventory_healthbench
    inventory=inventory_healthbench(PRIVATE/'2025-05-07-06-14-12_oss_eval.jsonl')
    manifest={'schema_version':'locdi.D4.external_acquisition.v1','dataset':'HealthBench','revision':REVISION,
        'original_split':'test','files':entries,'inventory':inventory,'license':'MIT_DATASET_CARD',
        'access_status':'PUBLIC_UNAUTHENTICATED_DOWNLOAD','examples_publish_online':False,
        'training_use_allowed_by_project':False,'independent_test_approved':False,
        'model_training_contamination':'UNKNOWN','clinical_reference_status':'RUBRIC_NOT_PATIENT_DIAGNOSIS_GOLD',
        'model_calls':0,'utility_fit_count':0,'clinical_verdict_count':0}
    manifest=json.loads(json.dumps(manifest))
    manifest['manifest_fingerprint']=digest(manifest)
    atomic_rows(PUBLIC,'dataset_manifest.json',manifest,jsonl=False)
    print(json.dumps({'status':'ACQUIRED','revision':REVISION,'file_count':len(entries),'inventory':inventory,'manifest_fingerprint':manifest['manifest_fingerprint']},ensure_ascii=True))


if __name__=='__main__':main()
