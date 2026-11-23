param([switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$public='outputs/research_evaluation/U1_SUPPORT_VOI_EXPANSION_v1/u1_d2_b34_10p_001'
$private='outputs/private/research_evaluation/U1_SUPPORT_VOI_EXPANSION_v1/u1_d2_b34_10p_001'
if (-not (Test-Path -LiteralPath "$public/plan.json")) {
    python -B scripts/run_u1_voi_expansion.py prepare
    if ($LASTEXITCODE -ne 0) {throw 'Preparation blocked; no generation'}
}
python -B scripts/run_u1_voi_expansion.py check
if ($LASTEXITCODE -ne 0) {throw 'Static check blocked; no generation'}
Write-Host 'Fixed10 D2, B3/B4 paired20 Episodes, maximum240 Generation; initial8 ties may fall back to ER. Authored Utility domain transfer only; clinical accuracy NOT_EVALUATED.'
if (-not $Execute) {return}
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Approval exists; inspect prior state. No overwrite/resume.'}
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
if ((Read-Host 'Type APPROVE_U1_FULL10 to accept exact plan,240-call cap, Source proxies, D2 initial augmentation and abstention/ER fallback') -cne 'APPROVE_U1_FULL10') {throw 'Not approved'}
$a.operator_ref=Read-Host 'Actual private operator reference'
$a.user_statement=Read-Host 'Actual statement accepting exact Plan and authored domain-transfer limitations'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other work not confirmed'}
foreach ($k in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy','accept_authored_utility_transfer','accept_routing_abstention_er_fallback')) {$a.$k=$true}
$now=[DateTimeOffset]::UtcNow;$format="yyyy-MM-dd'T'HH:mm:ss.ffffffzzz";$a.recorded_at=$now.ToString($format);$a.expires_at=$now.AddHours(24).ToString($format)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
python -B scripts/run_u1_voi_expansion.py execute --execute --approval $target
if ($LASTEXITCODE -ne 0) {throw 'Stopped; preserve partial data. Analyze only, no retry/resume.'}
python -B scripts/run_u1_voi_expansion.py analyze
if ($LASTEXITCODE -ne 0) {throw 'Analyze saved checkpoints; do not regenerate'}
