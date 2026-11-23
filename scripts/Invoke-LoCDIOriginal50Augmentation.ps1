param(
    [ValidateSet('Smoke','Full')][string]$Stage='Smoke',
    [string]$PlanId='',
    [string]$SmokePlanId='original50_d0d2_smoke_005',
    [switch]$Execute
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if ([string]::IsNullOrWhiteSpace($PlanId)) { $PlanId=if ($Stage -eq 'Smoke') { 'original50_d0d2_smoke_005' } else { 'original50_d0d2_full_003' } }
if ($PlanId -notmatch '^[A-Za-z0-9_]{1,50}$' -or $SmokePlanId -notmatch '^[A-Za-z0-9_]{1,50}$') { throw 'Invalid new Plan ID' }
$private="outputs/private/research_evaluation/ORIGINAL50_D0_D2_EXPERIMENT_v1/$PlanId"
$public="outputs/research_evaluation/ORIGINAL50_D0_D2_EXPERIMENT_v1/$PlanId"
if (-not (Test-Path -LiteralPath "$public/plan.json")) {
    if ($Stage -eq 'Full') {
        python -B scripts/run_original50_d0_d2.py prepare-full --plan-id $PlanId --smoke-id $SmokePlanId
    } else {
        python -B scripts/run_original50_d0_d2.py prepare-smoke --plan-id $PlanId
    }
    if ($LASTEXITCODE -ne 0) { throw 'Preparation blocked. Full requires a successful actual Smoke; no generation.' }
}
python -B scripts/run_original50_d0_d2.py check --plan-id $PlanId
if ($LASTEXITCODE -ne 0) { throw 'Static validation blocked. No generation.' }
$p=Get-Content -Raw -Encoding UTF8 -LiteralPath "$public/plan.json" | ConvertFrom-Json
if ($p.stage -ne $Stage) { throw 'Wrong stage for Plan ID' }
Write-Host "Stage=$Stage Plan=$PlanId Episodes=$($p.registered_episodes) Maximum Generation=$($p.generation_total)"
Write-Host 'D0 unchanged Source; D2 label-conditioned subjective initial information at minute0 only. No new test results/actions/likelihood.'
Write-Host 'Source-label development agreement, not clinical accuracy or independent generalization.'
if (-not $Execute) { Write-Host 'Static only. No model request.'; return }
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) { throw 'Existing approval. Inspect prior partial state; no overwrite/resume.' }
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
if ((Read-Host 'Accept exact D0/D2 plan, Source-time proxy and label-conditioned limits. Type APPROVE_D0_D2') -cne 'APPROVE_D0_D2') { throw 'Not approved' }
$a.operator_ref=Read-Host 'Your actual private operator reference'
$a.user_statement=Read-Host 'Your actual statement accepting this exact plan, maximum calls, initial synthetic time policy and development-only scope'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) { throw 'Actual statement required' }
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') { throw 'Other work not confirmed' }
foreach ($k in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy')) { $a.$k=$true }
$now=[DateTimeOffset]::UtcNow;$a.recorded_at=$now.ToString('o');$a.expires_at=$now.AddHours(24).ToString('o')
$bytes=[System.Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100))
$stream=[System.IO.File]::Open($target,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
python -B scripts/run_original50_d0_d2.py execute --plan-id $PlanId --execute --allow-llm --approval $target
if ($LASTEXITCODE -ne 0) { throw 'Stopped. Preserve partial results; analyze saved trace only. No retry/resume.' }
python -B scripts/run_original50_d0_d2.py analyze --plan-id $PlanId
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed. Do not regenerate; inspect saved checkpoints.' }
