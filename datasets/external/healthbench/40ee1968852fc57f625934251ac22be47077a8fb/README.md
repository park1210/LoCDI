# HealthBench acquisition — D4

Use **dataset_manifest_canonical_v2.json** as the acquisition receipt.
The original dataset_manifest.json is preserved for provenance; its self
fingerprint was computed before normalizing integer JSON object keys and is
invalid after JSON reloading. Its raw file SHA-256 entries remain correct.
The v2 receipt binds the original receipt SHA and verifies all four raw files.

The official 5,000-row JSONL, Dataset Card, evaluator source and software license
are stored under outputs/private/external_benchmarks/healthbench/ at this
revision. Do not publish example text, rubric text, references or canary values.
No models, graders or downloaded evaluator code were run during D4.

The original evaluation split is test. No internal split, clinical diagnosis
Gold, patient linkage or independent evaluation approval is inferred.
See datasets/external/D4_healthbench_compatibility_and_D5_spec.json for the
proposed dialogue adapter, excluded workflow tasks and remaining blockers.
