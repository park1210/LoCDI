param(
    [ValidateSet('Inspect','Collect')][string]$Action = 'Inspect'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$py = (Get-Command python -ErrorAction Stop).Source
$private = 'outputs/private/research_evaluation/B34_COLLECTION_RECOVERY_v1/source_collection_recovery_001'
$template = "$private/approval_template.json"
$approvalPath = "$private/operator_collection_recovery_approval.json"
$collectionPath = "$private/collection_recovery_001"
& $py -B scripts/routing_collection_recovery.py check
if ($LASTEXITCODE -ne 0) { throw 'New plan/source/code verification failed. No generation was started.' }
$approval = Get-Content -Encoding UTF8 -Raw -LiteralPath $template | ConvertFrom-Json
$approval | ConvertTo-Json -Depth 30 | Write-Host
Write-Host 'New patients/decisions only. Old collection_001 is preserved; no resume or automatic response merging.'
Write-Host 'Budget: at most 15 generations, retry/repair/warmup 0, timeout120s, output1024, context32768, wall1800s.'
if ($Action -eq 'Inspect') {
    Write-Host 'Inspect only: no approval created and no model/fitting invoked.'
    return
}
if (Test-Path -LiteralPath $collectionPath) { throw 'New collection already exists. Do not delete, overwrite or resume it.' }
if (Test-Path -LiteralPath $approvalPath) { throw 'New approval already exists. Preserve it and ask Codex to inspect; no automatic reuse.' }
if ($approval.scope -ne 'CURRENT_3B_ROUTING_RECOVERY_COLLECTION_15_ONLY' -or $approval.operator_approved -ne $false) {
    throw 'Only the new unsigned operator template is accepted.'
}
$confirm = Read-Host 'Approve ONLY this displayed new plan: type APPROVE_NEW_SCOPE'
if ($confirm -cne 'APPROVE_NEW_SCOPE') { throw 'Not approved. No generation.' }
$approval.operator_ref = Read-Host 'operator_ref: your own private operator ID (not a professor/reviewer identity)'
if ([string]::IsNullOrWhiteSpace($approval.operator_ref)) { throw 'Actual operator reference required' }
$approval.user_statement = Read-Host 'user_statement: enter your explicit approval of this exact new 15-request Source-proxy development scope'
if ([string]::IsNullOrWhiteSpace($approval.user_statement)) { throw 'Actual user statement required' }
$other = Read-Host 'Only if no other Ollama inference job is running, type NO_OTHER_OLLAMA_WORK'
if ($other -cne 'NO_OTHER_OLLAMA_WORK') { throw 'Other-work status not confirmed; no generation' }
$approval.operator_approved = $true
$approval.accept_source_proxy_for_collection_only = $true
$approval.no_other_ollama_work = $true
$now = [DateTimeOffset]::UtcNow
$approval.recorded_at = $now.ToString('yyyy-MM-ddTHH:mm:ss.ffffffzzz')
$approval.expires_at = $now.AddHours(24).ToString('yyyy-MM-ddTHH:mm:ss.ffffffzzz')
$json = $approval | ConvertTo-Json -Depth 30
$stream = [System.IO.File]::Open((Join-Path (Get-Location) $approvalPath),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
try { $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($json); $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
& $py -B scripts/routing_collection_recovery.py execute --execute --allow-llm --authorization $approvalPath
if ($LASTEXITCODE -ne 0) {
    throw 'New collection stopped/failed. Preserve the new approval, collection and checkpoints; do not rerun. Give the paths to Codex.'
}
Get-Content -Encoding UTF8 -Raw -LiteralPath 'outputs/research_evaluation/B34_COLLECTION_RECOVERY_v1/new_collection_status.json'
Write-Host 'Collection finished. Send the saved results to Codex for offline Source-proxy analysis. No fitting or clinical verdict was generated here.'
