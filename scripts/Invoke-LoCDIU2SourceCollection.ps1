param([switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$public='outputs/research_evaluation/U1_SUPPORT_VOI_EXPANSION_v1/u2_source_snapshot_collect_001'
$private='outputs/private/research_evaluation/U1_SUPPORT_VOI_EXPANSION_v1/u2_source_snapshot_collect_001'
if (-not (Test-Path -LiteralPath "$public/plan.json")) {
    python -B scripts/prepare_u2_source_collection.py prepare
    if ($LASTEXITCODE -ne 0) {throw 'Source preparation blocked; no generation'}
}
python -B scripts/prepare_u2_source_collection.py check
if ($LASTEXITCODE -ne 0) {throw 'Static check blocked; no generation'}
Write-Host '10 disjoint Train patients x5 Agents,50 single diagnostic-proposal calls. STATIC Source predisclosure proxy, NOT Native Episodes; no fitting. No synthetic symptoms authored.'
if (-not $Execute) {return}
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Existing approval; inspect partial state. No overwrite/resume.'}
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
if ((Read-Host 'Type APPROVE_U2_STATIC_SOURCE to accept50 calls, static numeric Source predisclosure and same-day Reference proxy; NOT Native Episode equivalence') -cne 'APPROVE_U2_STATIC_SOURCE') {throw 'Not approved'}
$a.operator_ref=Read-Host 'Actual private operator reference'
$a.user_statement=Read-Host 'Actual statement accepting exact Source snapshot Plan, domain gap,50 calls and30min wall'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other work not confirmed'}
foreach ($k in @('operator_approved','no_other_ollama_work','accept_static_predisclosure_proxy','accept_source_reference_date_proxy')) {$a.$k=$true}
$now=[DateTimeOffset]::UtcNow;$format="yyyy-MM-dd'T'HH:mm:ss.ffffffzzz";$a.recorded_at=$now.ToString($format);$a.expires_at=$now.AddHours(24).ToString($format)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
python -B scripts/prepare_u2_source_collection.py collect --execute --approval $target
if ($LASTEXITCODE -ne 0) {throw 'Stopped. Preserve partial response/checkpoints. No retry/resume.'}
python -B scripts/prepare_u2_source_collection.py analyze
if ($LASTEXITCODE -ne 0) {throw 'Analyze stored data; do not regenerate'}
