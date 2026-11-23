param([switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$private='outputs/private/research_evaluation/B34_DIAGNOSIS_RECOVERY_20261123_v1/u1_stop_terminal_replay_001'
python -B scripts/run_terminal_only_replay.py check
if ($LASTEXITCODE -ne 0) {throw 'Static contract blocked. No model request.'}
Write-Host 'Ten frozen STOP snapshots: cached 68 old responses, at most ten NEW terminal-only generations. Not a new complete B4 ablation. Clinical accuracy NOT_EVALUATED.'
if (-not $Execute) {return}
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Approval exists; inspect saved state. No overwrite/resume.'}
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
if ((Read-Host 'Type APPROVE_TERMINAL_REPLAY10 to accept the exact plan, cached-prefix replay, frozen snapshots and at most 10 new requests') -cne 'APPROVE_TERMINAL_REPLAY10') {throw 'Not approved'}
$a.operator_ref=Read-Host 'Your actual private operator reference'
$a.user_statement=Read-Host 'Your actual statement accepting this exact development plan and Source/augmentation limitations'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual operator statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other work not confirmed'}
foreach ($k in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy','accept_authored_utility_transfer','accept_routing_abstention_er_fallback','accept_cached_prefix_terminal_replay')) {$a.$k=$true}
$now=[DateTimeOffset]::UtcNow;$format="yyyy-MM-dd'T'HH:mm:ss.ffffffzzz";$a.recorded_at=$now.ToString($format);$a.expires_at=$now.AddHours(24).ToString($format)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
python -B scripts/run_terminal_only_replay.py execute --execute --approval $target
if ($LASTEXITCODE -ne 0) {throw 'Stopped. Preserve partial records. Analyze only; no retry/resume.'}
python -B scripts/run_terminal_only_replay.py analyze
if ($LASTEXITCODE -ne 0) {throw 'Inspect saved analysis; do not regenerate.'}
