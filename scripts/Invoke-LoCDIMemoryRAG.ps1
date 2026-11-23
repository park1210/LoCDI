param([Parameter(Mandatory=$true)][string]$PlanId,[switch]$Execute)
$ErrorActionPreference='Stop'
if ($PlanId -notmatch '^[A-Za-z0-9_]{1,64}$') { throw 'Invalid Plan ID' }
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
python -B scripts/run_memory_rag_experiment.py check --plan-id $PlanId
if ($LASTEXITCODE -ne 0) { throw 'Static check blocked; no Generation' }
$root='outputs/private/research_evaluation/MEMORY_RAG_LEGACY_RECOVERY_20261121_v1'
$private=Join-Path $root $PlanId
if (-not $Execute) { Write-Host 'Static check only. No model request.'; return }
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) { throw 'Existing approval: inspect state; no overwrite or automatic resume' }
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $private 'approval_template.json') | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
if ((Read-Host 'Read exact arms/budget/proxy limits. Type APPROVE_MEMORY_RAG') -cne 'APPROVE_MEMORY_RAG') { throw 'Not approved' }
$a.operator_ref=Read-Host 'Actual private operator reference'
$a.user_statement=Read-Host 'Your actual acceptance of this plan, recorded-time development proxy and Source-label limitations'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) { throw 'Actual statement required' }
if ((Read-Host 'Confirm NO_OTHER_OLLAMA_WORK') -cne 'NO_OTHER_OLLAMA_WORK') { throw 'Other work not confirmed' }
foreach ($key in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu')) { $a.$key=$true }
$now=[DateTimeOffset]::UtcNow;$a.recorded_at=$now.ToString('o');$a.expires_at=$now.AddHours(24).ToString('o')
$bytes=[System.Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100))
$stream=[System.IO.File]::Open($target,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
python -B scripts/run_memory_rag_experiment.py execute --plan-id $PlanId --execute --allow-llm --approval $target
if ($LASTEXITCODE -ne 0) { throw 'Stopped; preserve partial checkpoints and NOT_ATTEMPTED. No retry/resume.' }
python -B scripts/run_memory_rag_experiment.py analyze --plan-id $PlanId
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed; do not regenerate' }
