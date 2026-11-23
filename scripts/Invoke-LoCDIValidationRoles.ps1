param([string]$PlanId='role_validation_snapshot_001',[switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if ($PlanId -notmatch '^[A-Za-z0-9_]+$') {throw 'Safe prepared Plan ID required'}
$private="outputs/private/research_evaluation/ROUTING_ROOT_CAUSE_20261123_v1/$PlanId"
python -B scripts/run_validation_role_snapshot.py check --plan-id $PlanId
if ($LASTEXITCODE -ne 0) {throw 'Static check blocked. No generation.'}
Write-Host 'Two existing Validation snapshots x ER and five Specialty roles: at most 12 NEW terminal generations. Cached acquisition is not new generation. VoI OFF; not Native skill or B4 admission.'
if (-not $Execute) {return}
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Approval exists; inspect saved state. No overwrite/resume.'}
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
if ((Read-Host 'Type APPROVE_VALIDATION_ROLES12 for this exact diagnostic-only Plan') -cne 'APPROVE_VALIDATION_ROLES12') {throw 'Not approved'}
$a.operator_ref=Read-Host 'Actual private operator reference'
$a.user_statement=Read-Host 'Actual statement accepting frozen snapshots, diagnostic-only budget, development exposure and M5 overlap for VoI-OFF comparison only'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual operator statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other work not confirmed'}
foreach ($k in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_frozen_snapshot_role_comparison','accept_validation_m5_overlap_for_voi_off_only','accept_no_further_acquisition_task_budget')) {$a.$k=$true}
$now=[DateTimeOffset]::UtcNow;$format="yyyy-MM-dd'T'HH:mm:ss.ffffffzzz";$a.recorded_at=$now.ToString($format);$a.expires_at=$now.AddHours(24).ToString($format)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
python -B scripts/run_validation_role_snapshot.py execute --plan-id $PlanId --execute --approval $target
if ($LASTEXITCODE -ne 0) {throw 'Stopped. Preserve partial data. No retry/resume.'}
python -B scripts/run_validation_role_snapshot.py analyze --plan-id $PlanId
if ($LASTEXITCODE -ne 0) {throw 'Inspect saved analysis; do not regenerate.'}
