"""Read-only D4 byte, packet, schema and review-state verification. No network."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_suitability import sha
from app.locdi.external_benchmark_contract import inventory_healthbench
from app.locdi.reference_readiness import validate_blinded_packet


def main():
    baseline=json.loads((ROOT/'outputs/private/D4_baseline_hashes.json').read_text(encoding='utf-8'))
    changed=[k for k,v in baseline.items() if not (ROOT/k).is_file() or sha(ROOT/k)!=v]
    if changed:raise ValueError('EXISTING_DATA_CHANGED')
    public=ROOT/'datasets/external/healthbench/40ee1968852fc57f625934251ac22be47077a8fb'
    manifest=json.loads((public/'dataset_manifest_canonical_v2.json').read_text(encoding='utf-8'))
    if digest({k:v for k,v in manifest.items() if k!='manifest_fingerprint'})!=manifest['manifest_fingerprint']:raise ValueError('MANIFEST_HASH')
    for entry in manifest['files'].values():
        p=(ROOT/entry['path']).resolve()
        if not p.is_relative_to((ROOT/'outputs/private/external_benchmarks').resolve()) or sha(p)!=entry['sha256']:raise ValueError('EXTERNAL_RAW_HASH')
    original=public/'dataset_manifest.json'
    if sha(original)!=manifest['original_receipt_sha256']:raise ValueError('ORIGINAL_RECEIPT_CHANGED')
    inventory=inventory_healthbench(ROOT/manifest['files']['2025-05-07-06-14-12_oss_eval.jsonl']['path'])
    if inventory!=manifest['inventory']:raise ValueError('SCHEMA_INVENTORY_CHANGED')
    private=ROOT/'outputs/private/reference_review/D4_reference_v1'
    receipt=json.loads((private/'manifest.json').read_text(encoding='utf-8'))
    for entry in receipt['files'].values():
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('REVIEW_FILE_CHANGED')
    packets=list(private.glob('*case*.json'))+list(private.glob('routing_Candidate010_blinded*.json'))
    for p in packets:validate_blinded_packet(json.loads(p.read_text(encoding='utf-8')))
    d3=json.loads((ROOT/'datasets/versions/simulation_source_replay_v1/dataset_manifest.json').read_text(encoding='utf-8'))
    for entry in d3['files'].values():
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('D3_CHANGED')
    return {'schema_version':'locdi.D4.final_verification.v1','protected_files_unchanged':len(baseline),
        'core_files_unchanged':sum(k.startswith('app/core/') for k in baseline),'d3_private_hash_checks':len(d3['files']),
        'external_raw_hash_checks':len(manifest['files']),'external_samples_schema_valid':inventory['sample_count'],
        'blinded_packets_verified':len(packets),'review_fields_remain_blank':True,
        'canonical_manifest_fingerprint':manifest['manifest_fingerprint'],'new_llm_generations':0,
        'utility_fit_count':0,'voi_probability_estimates':0,'clinical_verdict_count':0}


if __name__=='__main__':print(json.dumps(main(),ensure_ascii=True))
