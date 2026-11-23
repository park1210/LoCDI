param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Status','Collect','CompleteCollection','B3','B4','All','Analyze')]
    [string]$Action,
    [switch]$UseUnfittedAbstention,
    [string]$PlanId = 'source_all_001'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$py = (Get-Command python -ErrorAction Stop).Source
$trainingPlan = 'source_ready_002'
$privateRoot = 'outputs/private/research_evaluation/B34_CONNECTION_v1'
$base = "$privateRoot/$trainingPlan"
$train = "$base/routing_training"
function Run-Python {
    param([Parameter(Mandatory=$true)][string[]]$Arguments)
    & $py -B @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped: Python exit code $LASTEXITCODE. Preserve outputs; no automatic retry/resume." }
}
function New-OperatorApproval {
    param([string]$Template,[string]$Output)
    & "$PSScriptRoot/new_common_operator_approval.ps1" -Template $Template -Output $Output
}
if ($Action -eq 'Status') {
    Get-Content -Encoding UTF8 -Raw -LiteralPath 'outputs/research_evaluation/B34_SOURCE_READY_v1/source_verification.json'
    Get-Content -Encoding UTF8 -Raw -LiteralPath "outputs/research_evaluation/B34_CONNECTION_v1/$trainingPlan/routing_training/readiness.json"
    Write-Host 'No generation or fitting. Source labels are development proxies, not clinical correctness.'
    return
}
if ($Action -eq 'Collect') {
    Write-Host 'New collection: 2 Train + 1 Validation x 5 Agents, at most 15 generations, retry/repair 0.'
    if (Test-Path -LiteralPath "$train/collection_001") { throw 'Collection directory exists. No overwrite/resume or automatic second run.' }
    New-OperatorApproval -Template "$train/approval_template.json" -Output "$train/operator_collection_approval.json"
    Run-Python -Arguments @('scripts/collect_common_routing_training.py','execute','--plan-id',$trainingPlan,'--execute','--allow-llm','--authorization',"$train/operator_collection_approval.json")
    Get-Content -Encoding UTF8 -Raw -LiteralPath "outputs/research_evaluation/B34_CONNECTION_v1/$trainingPlan/routing_training/collection_status.json"
    Write-Host 'Collection ended. This does not mean correctness or fitting succeeded. Return the saved results to Codex, or use CompleteCollection for offline validation/build.'
    return
}
if ($Action -eq 'CompleteCollection') {
    # No model client or generation command occurs in this branch.
    Run-Python -Arguments @('scripts/build_development_routing_proxy.py','--plan-id',$trainingPlan)
    Run-Python -Arguments @('scripts/build_development_routing_proxy.py','--plan-id',$trainingPlan,'--prepare-approval')
    New-OperatorApproval -Template "$train/proxy_fit_approval_template.json" -Output "$train/operator_proxy_fit_approval.json"
    Run-Python -Arguments @('scripts/build_development_routing_proxy.py','--plan-id',$trainingPlan,'--build','--allow-fit','--authorization',"$train/operator_proxy_fit_approval.json")
    $evidence = "$base/development_proxy_utility_001/development_routing_evidence.json"
    foreach ($spec in @(@('B3','source_collected_b3_min_001','1'),@('B4','source_collected_b4_min_001','1'),@('ALL','source_collected_all_001','10'))) {
        $conditions = if ($spec[0] -eq 'ALL') { @('B0','B1','B2','B3','B4') } else { @($spec[0]) }
        Run-Python -Arguments (@('scripts/run_common_menu_b0_b4.py','prepare','--plan-id',$spec[1],'--experiment-kind','DEVELOPMENT_EXPERIMENT','--patient-count',$spec[2],'--routing-evidence',$evidence,'--conditions') + $conditions)
        Run-Python -Arguments @('scripts/run_common_menu_b0_b4.py','check','--plan-id',$spec[1])
    }
    Write-Host 'Offline processing ended. Inspect development_proxy_utility_status.json: zero eligible groups means no fitted Utility, and Core abstention remains active.'
    return
}
if ($Action -eq 'Analyze') {
    if ($PlanId -notmatch '^[a-zA-Z0-9_]{1,64}$') { throw 'Invalid plan ID' }
    Run-Python -Arguments @('scripts/run_common_menu_b0_b4.py','analyze','--plan-id',$PlanId)
    Get-Content -Encoding UTF8 -Raw -LiteralPath "outputs/research_evaluation/B34_CONNECTION_v1/$PlanId/saved_results.json"
    return
}
$runPlan = switch ($Action) {
    'B3' { if ($UseUnfittedAbstention) {'source_b3_min_001'} else {'source_collected_b3_min_001'} }
    'B4' { if ($UseUnfittedAbstention) {'source_b4_min_001'} else {'source_collected_b4_min_001'} }
    'All' { if ($UseUnfittedAbstention) {'source_all_001'} else {'source_collected_all_001'} }
}
if ($UseUnfittedAbstention) {
    Write-Host 'EXPLICIT UNFITTED RUN: no learned Utility. Adaptive routing abstains and continues ER; this is not an adaptive-performance result.'
}
$pr = "$privateRoot/$runPlan"
if (-not (Test-Path -LiteralPath "$pr/approval_template.json")) { throw 'Prepared plan missing. Collect responses and complete offline processing first, or explicitly choose -UseUnfittedAbstention.' }
Run-Python -Arguments @('scripts/run_common_menu_b0_b4.py','check','--plan-id',$runPlan)
New-OperatorApproval -Template "$pr/approval_template.json" -Output "$pr/operator_execution_approval.json"
Run-Python -Arguments @('scripts/run_common_menu_b0_b4.py','execute','--plan-id',$runPlan,'--execute','--allow-llm','--authorization',"$pr/operator_execution_approval.json")
Run-Python -Arguments @('scripts/run_common_menu_b0_b4.py','analyze','--plan-id',$runPlan)
Get-Content -Encoding UTF8 -Raw -LiteralPath "outputs/research_evaluation/B34_CONNECTION_v1/$runPlan/saved_results.json"
