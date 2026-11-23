param([ValidateSet('Routing','VoI')][string]$Stage='Routing',[switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$plan=if ($Stage -eq 'Routing') {'u1_routing_smoke_002'} else {'u1_voi_smoke_002'}
$public="outputs/research_evaluation/B34_UTILITY_VOI_SMOKE_v1/$plan"
$private="outputs/private/research_evaluation/B34_UTILITY_VOI_SMOKE_v1/$plan"
if (-not (Test-Path -LiteralPath "$public/plan.json")) {
    python -B scripts/run_b34_transfer_smoke.py prepare --stage $Stage
    if ($LASTEXITCODE -ne 0) {throw 'Preparation blocked; no generation'}
}
python -B scripts/run_b34_transfer_smoke.py check --stage $Stage
if ($LASTEXITCODE -ne 0) {throw 'Static check blocked; VoI requires actual Routing Smoke completion'}
Write-Host 'Development-only authored initial/compact to Native-action domain transfer. Low similarity support; improvement not established. 1 patient, 2 Episodes, maximum24 Generation, 15min wall, no retry/resume.'
if (-not $Execute) {return}
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Existing approval: inspect prior run, no overwrite/resume'}
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
if ((Read-Host 'Type APPROVE_AUTHORED_TRANSFER to accept this exact plan and development domain shift') -cne 'APPROVE_AUTHORED_TRANSFER') {throw 'Not approved'}
$a.operator_ref=Read-Host 'Actual private operator reference'
$a.user_statement=Read-Host 'Actual statement accepting exact plan,24-call cap, authored Reference proxy, initial-time augmentation and domain-transfer limitations'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual operator statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other work not confirmed'}
foreach ($key in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy','accept_authored_utility_transfer')) {$a.$key=$true}
$now=[DateTimeOffset]::UtcNow;$a.recorded_at=$now.ToString('o');$a.expires_at=$now.AddHours(24).ToString('o')
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
python -B scripts/run_b34_transfer_smoke.py execute --stage $Stage --execute --approval $target
if ($LASTEXITCODE -ne 0) {throw 'Stopped: preserve partial results; no retry/resume'}
python -B scripts/run_b34_transfer_smoke.py analyze --stage $Stage
if ($LASTEXITCODE -ne 0) {throw 'Read saved checkpoints; do not regenerate'}
