param(
    [ValidateSet('B4','B012','All')][string]$Condition = 'B4',
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$py = (Get-Command python -ErrorAction Stop).Source
$plan = switch ($Condition) {
    'B4' { 'runtime_recovery_b4_001' }
    'B012' { 'runtime_recovery_b012_001' }
    'All' { 'runtime_recovery_all_001' }
}
$pr = "outputs/private/research_evaluation/B34_CONNECTION_v1/$plan"
$pub = "outputs/research_evaluation/B34_CONNECTION_v1/$plan"
& $py -B scripts/run_common_menu_b0_b4.py check --plan-id $plan
if ($LASTEXITCODE -ne 0) { throw 'Static checks blocked. No generation.' }
$p = Get-Content -Raw -Encoding UTF8 -LiteralPath "$pub/plan.json" | ConvertFrom-Json
if ($p.runtime_recovery -ne $true -or $p.catalog_selection -ne $true) { throw 'Wrong experiment contract' }
Write-Host "Fresh plan: $plan / episodes: $($p.registered_episodes) / maximum Generation: $($p.generation_total)"
Write-Host 'New hybrid development contract: released measurements view; frozen M5 information/STOP plus independently Core-validated current LLM terminal diagnosis.'
Write-Host 'Source-label proxy only; no independent clinical review; clinical_accuracy=NOT_EVALUATED.'
Write-Warning 'New Utility has only one positive agent (nephrology); personalized routing or accuracy improvement is NOT established.'
if (-not $Execute) {
    Write-Host 'Inspection only. Use -Execute yourself after reviewing the new scope. No model call was made.'
    return
}
& "$PSScriptRoot/new_common_operator_approval.ps1" -Template "$pr/approval_template.json" -Output "$pr/operator_runtime_approval.json"
& $py -B scripts/run_common_menu_b0_b4.py execute --plan-id $plan --execute --allow-llm --authorization "$pr/operator_runtime_approval.json"
if ($LASTEXITCODE -ne 0) { throw 'Stopped. Preserve checkpoints and partial results; no retry or resume.' }
& $py -B scripts/run_common_menu_b0_b4.py analyze --plan-id $plan
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed. Preserve results; do not rerun generation.' }
Get-Content -Raw -Encoding UTF8 -LiteralPath "$pub/saved_results.json"
Write-Host 'Keep algorithmic diagnosis separate from the Hospital supported/unresolved result; completed does not mean correct.'
