param(
    [ValidateSet('B3','B4','All')][string]$Condition = 'B3',
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$py = (Get-Command python -ErrorAction Stop).Source
$plan = switch ($Condition) {
    'B3' {'recovery_llm_b3_001'}
    'B4' {'recovery_llm_b4_001'}
    'All' {'recovery_llm_all_001'}
}
$private = "outputs/private/research_evaluation/B34_CONNECTION_v1/$plan"
$public = "outputs/research_evaluation/B34_CONNECTION_v1/$plan"
& $py -B scripts/run_common_menu_b0_b4.py check --plan-id $plan
if ($LASTEXITCODE -ne 0) { throw 'Source/Utility/Model/config checks blocked. No new generation.' }
$metadata = Get-Content -Encoding UTF8 -Raw -LiteralPath "$public/plan.json" | ConvertFrom-Json
Write-Host "Plan: $plan / fixed patients: $($metadata.registered_patients) / episodes: $($metadata.registered_episodes) / maximum generations: $($metadata.generation_total)"
Write-Host 'DEVELOPMENT_EXPERIMENT: Source-label proxy, no independent clinical review, clinical_accuracy=NOT_EVALUATED.'
if (-not $Execute) {
    Write-Host 'Inspect only. Use -Execute yourself for actual LLM inference with a fresh operator approval.'
    return
}
& "$PSScriptRoot/new_common_operator_approval.ps1" -Template "$private/approval_template.json" -Output "$private/operator_llm_test_approval.json"
& $py -B scripts/run_common_menu_b0_b4.py execute --plan-id $plan --execute --allow-llm --authorization "$private/operator_llm_test_approval.json"
if ($LASTEXITCODE -ne 0) { throw 'Execution stopped. Preserve all files and checkpoints; no automatic retry/resume.' }
& $py -B scripts/run_common_menu_b0_b4.py analyze --plan-id $plan
if ($LASTEXITCODE -ne 0) { throw 'Saved-result analysis failed. Do not rerun generation; give the saved files to Codex.' }
Get-Content -Encoding UTF8 -Raw -LiteralPath "$public/saved_results.json"
Write-Host 'Read COMPLETED/FAILED/INCOMPLETE/NOT_ATTEMPTED counts; process completion alone is not clinical success.'
