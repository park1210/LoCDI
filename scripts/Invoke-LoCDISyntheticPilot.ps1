param(
 [Parameter(Mandatory=$true)][string]$PlanId,
 [switch]$Execute
)
$ErrorActionPreference='Stop'
if ($PlanId -notmatch '^[A-Za-z0-9_]{1,64}$') { throw 'Invalid Plan ID' }
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
python -B scripts/synthetic_pilot_utility.py check --plan-id $PlanId
if ($LASTEXITCODE -ne 0) { throw 'Static check blocked' }
$private="outputs/private/research_evaluation/SYNTHETIC_PILOT_UTILITY_20261121_v1/$PlanId"
$plan=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/plan.json" | ConvertFrom-Json
Write-Host "Plan=$PlanId / Stage=$($plan.stage) / View=$($plan.view) / Max Generation=$($plan.generation_cap)"
if (-not $Execute) { Write-Host 'Check only. No Generation or approval created.'; return }
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) { throw 'Existing approval: inspect it; no overwrite or automatic second run' }
$approval=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/operator_approval_template.json" | ConvertFrom-Json
$approval | ConvertTo-Json -Depth 100 | Write-Host
Write-Host 'Authored synthetic label proxy only; four illustrative References stay null; no clinical approval or Utility fitting.'
Write-Host 'All-tests stage uses fixed native Source acquisition before each shared snapshot. No acquisition LLM calls.'
if ((Read-Host 'Read scope and type APPROVE_SYNTHETIC_COLLECTION') -cne 'APPROVE_SYNTHETIC_COLLECTION') { throw 'Not approved' }
$approval.operator_ref=Read-Host 'Your actual private operator reference'
$approval.statement=Read-Host 'Your explicit statement accepting this exact Plan, authored Reference proxy and synthetic time policy'
if ([string]::IsNullOrWhiteSpace($approval.operator_ref) -or [string]::IsNullOrWhiteSpace($approval.statement)) { throw 'Actual statement required' }
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK') -cne 'NO_OTHER_OLLAMA_WORK') { throw 'Other job status not confirmed' }
$approval.operator_approved=$true
$approval.accept_author_reference_proxy=$true
$approval.accept_synthetic_acquisition_timing=$true
$approval.no_other_ollama_work=$true
$now=[DateTimeOffset]::UtcNow
$approval.recorded_at=$now.ToString('o')
$approval.expires_at=$now.AddHours(24).ToString('o')
$bytes=[System.Text.UTF8Encoding]::new($false).GetBytes(($approval | ConvertTo-Json -Depth 100))
$stream=[System.IO.File]::Open($target,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
python -B scripts/synthetic_pilot_utility.py collect --plan-id $PlanId --execute --allow-llm --approval $target
if ($LASTEXITCODE -ne 0) { throw 'Inspect partial checkpoints. No retry/resume.' }
python -B scripts/synthetic_pilot_utility.py analyze --plan-id $PlanId
if ($LASTEXITCODE -ne 0) { throw 'Offline analysis failed; do not repeat Generation' }
