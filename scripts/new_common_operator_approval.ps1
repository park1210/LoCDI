param(
    [Parameter(Mandatory=$true)][string]$Template,
    [Parameter(Mandatory=$true)][string]$Output
)
$ErrorActionPreference = 'Stop'
$taskWorkspace = Split-Path -Parent $PSScriptRoot
$approvedPrivateRoot = [System.IO.Path]::GetFullPath((Join-Path $taskWorkspace 'outputs/private/research_evaluation/B34_CONNECTION_v1')) + [System.IO.Path]::DirectorySeparatorChar
$nominalTarget = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path (Get-Location).Path $Output }
$target = [System.IO.Path]::GetFullPath($nominalTarget)
if (-not $target.StartsWith($approvedPrivateRoot,[System.StringComparison]::OrdinalIgnoreCase)) { throw 'Approval must stay in the new private namespace' }
if (Test-Path -LiteralPath $target) { throw 'Existing approval must not be overwritten/reused' }
$approval = Get-Content -Raw -Encoding UTF8 -LiteralPath $Template | ConvertFrom-Json
$scopes = @('B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','CURRENT_3B_ROUTING_COLLECTION_15_ONLY','REVIEWED_CURRENT_3B_ROUTING_FIT_ONCE','DEVELOPMENT_SOURCE_LABEL_PROXY_BUILD_ONCE')
if ($approval.scope -notin $scopes -or $approval.operator_approved -ne $false) { throw 'Unrecognized or already approved template' }
$approval | ConvertTo-Json -Depth 100 | Write-Host
$answer = Read-Host 'After reading the exact scope, budget and limitations, type APPROVE_NEW_SCOPE to approve as operator'
if ($answer -cne 'APPROVE_NEW_SCOPE') { throw 'Not approved' }
$approval.operator_ref = Read-Host 'Actual private operator reference'
$approval.user_statement = Read-Host 'Your own explicit approval statement for this exact scope and plan'
if ([string]::IsNullOrWhiteSpace($approval.operator_ref) -or [string]::IsNullOrWhiteSpace($approval.user_statement)) { throw 'Actual operator statement required' }
$approval.operator_approved = $true
if ($null -ne $approval.PSObject.Properties['no_other_ollama_work']) {
    $otherWork = Read-Host 'Confirm that no other Ollama job is running: type NO_OTHER_OLLAMA_WORK'
    if ($otherWork -cne 'NO_OTHER_OLLAMA_WORK') { throw 'Other-work status not confirmed' }
}
foreach ($name in @('no_other_ollama_work','accept_source_proxy_for_collection_only','accept_recorded_time_proxy','accept_common_m5_menu','accept_source_label_proxy')) {
    if ($null -ne $approval.PSObject.Properties[$name]) { $approval.$name = $true }
}
$now = [DateTimeOffset]::UtcNow
$approval.recorded_at = $now.ToString('yyyy-MM-ddTHH:mm:ss.ffffffzzz')
$approval.expires_at = $now.AddHours(24).ToString('yyyy-MM-ddTHH:mm:ss.ffffffzzz')
$json = $approval | ConvertTo-Json -Depth 100
$stream = [System.IO.File]::Open($target,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
try { $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($json); $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
Write-Host 'Created operator self-declaration only. No clinical approval, model call or fitting performed.'
