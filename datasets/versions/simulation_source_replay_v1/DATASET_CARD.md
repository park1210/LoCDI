# simulation_source_replay_v1

Development-only source replay: 29 patients / 29 index encounters selected from
368 patients / 12,971 encounters. This is not an approved clinical benchmark.

`dataset_manifest.json` is the public entry point. All canonical cases, source
records, patient/visit identities, memory, hidden results and references remain
under `outputs/private/simulation_datasets/simulation_source_replay_v1/`.
Its generated wrappers are compatibility projections for the existing loader.
Do not redistribute the private directory. Licensing remains UNKNOWN.

Read `replay_policy.json` before using the wrappers. All Ask questions are
synthetic templates querying actual recorded values; there was no observed
conversation. Test orders are simulated, with unmodified source values.
Recorded-time offsets are development release proxies, not clinical turnaround.
Actual availability remains unknown in the canonical data.

STRICT_SOURCE execution is blocked. Proxy loading requires
`load_replay_cases(policy=PROXY, accept_development_proxy=True)`.
Initial clinical values are absent; age, sex and encounter metadata are exposed
only under the explicit development assumption. Encoded symptom candidates
and unreviewed source Conditions are private and are not agent inputs.
Consult is disabled. All new splits are null; no blind clinical test is approved.

`offline_validation.json` records 29 cases executed twice with a TEST_ONLY
client through the actual runner/scheduler: 29 completed in each round, all
diagnosis-unresolved, 29 matching normalized traces. Only 210/301 available
Test candidates were exercised in each round; decision limits were preserved.
Clinical metrics are NOT_EVALUATED. See `regression_result.json` and
`preservation_check.json` for separate verification evidence.

Full report: `docs/데이터/D3_028_SOURCE_GROUNDED_SIMULATION_DATASET_REPORT.md`.
