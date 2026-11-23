param(
    [Parameter(Mandatory=$true)][string]$PlanId,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
if ($PlanId -notmatch '^[a-zA-Z0-9_]{1,64}$') { throw 'Invalid Plan ID' }
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$private = "outputs/private/research_evaluation/B34_CONNECTION_v1/$PlanId"
$public = "outputs/research_evaluation/B34_CONNECTION_v1/$PlanId"
python -B scripts/run_common_menu_b0_b4.py check --plan-id $PlanId
if ($LASTEXITCODE -ne 0) { throw 'Plan blocked; no Generation' }
$p = Get-Content -Raw -Encoding UTF8 -LiteralPath "$public/plan.json" | ConvertFrom-Json
if ($p.experiment_kind -ne 'DEVELOPMENT_EXPERIMENT' -or $p.evidence_index_set -ne $true) { throw 'Wrong development protocol' }
Write-Host "Plan $PlanId / $($p.registered_episodes) episodes / max $($p.generation_total) generations"
Write-Host "COMMON_KNOWLEDGE=$($null -ne $p.knowledge_corpus_path) / HISTORICAL_LABELS=$($p.memory_time_labels) / RULE_BASED=$($p.rule_based)"
Write-Host 'Literature is not patient evidence. Source Exact Agreement is primary; clinical_accuracy=NOT_EVALUATED.'
if (-not $Execute) { Write-Host 'Inspect only. Use -Execute yourself for a new explicit operator approval.'; return }
& "$PSScriptRoot/new_common_operator_approval.ps1" -Template "$private/approval_template.json" -Output "$private/operator_knowledge_approval.json"
python -B scripts/run_common_menu_b0_b4.py execute --plan-id $PlanId --execute --allow-llm --authorization "$private/operator_knowledge_approval.json"
if ($LASTEXITCODE -ne 0) { throw 'Stopped; preserve files, no retry/resume' }
python -B scripts/run_common_menu_b0_b4.py analyze --plan-id $PlanId
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed; do not regenerate' }
Get-Content -Raw -Encoding UTF8 -LiteralPath "$public/saved_results.json"
Write-Host 'Check FAILED/INCOMPLETE/NOT_ATTEMPTED explicitly; command completion is not full experiment success.'
