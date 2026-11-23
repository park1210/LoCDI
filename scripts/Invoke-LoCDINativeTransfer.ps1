param([ValidateSet('Validation','Smoke5','Verify10','Full50')][string]$Stage='Validation',[switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$root='outputs/private/research_evaluation/NATIVE_ROUTING_TRANSFER_20261123_v1'
if ($Stage -eq 'Validation') {
    python -B scripts/native_routing_transfer.py validation-check
    $private=Join-Path $root 'c435_native_validation2_002'
} else {
    python -B scripts/run_native_menu_full50.py check --stage $Stage
    $private=Join-Path $root ('native_menu_d2_'+$Stage.ToLower()+'_002')
}
if ($LASTEXITCODE -ne 0) {throw 'Static/admission/stage check blocked. No generation.'}
if (-not $Execute) {return}
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Approval exists. Preserve; no overwrite/resume.'}
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
$phrase='APPROVE_NATIVE_'+$Stage.ToUpper()
Write-Host 'Source-label proxy development only. Validation has prior VoI exposure and bounded export independence; no clinical review/untouched Holdout. Menu features describe availability, never performed results.'
if ((Read-Host "Accept exact displayed plan/call cap. Type $phrase") -cne $phrase) {throw 'Not approved'}
$a.operator_ref=Read-Host 'Actual private operator reference'
$a.user_statement=Read-Host 'Actual statement accepting this exact plan, Source timing proxy, prior VoI exposure, feature version, maximum calls/resources and no-resume'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual operator statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other work not excluded'}
foreach ($k in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_prior_VoI_exposure_and_bounded_independence')) {$a.$k=$true}
if ($Stage -eq 'Validation') {
    $a.accept_native_training_intervention=$true;$a.accept_source_reference_date_proxy=$true
} else {
    $a.accept_label_conditioned_development=$true;$a.accept_augmentation_initial_time_policy=$true;$a.accept_native_public_menu_feature=$true
}
$now=[DateTimeOffset]::UtcNow;$format="yyyy-MM-dd'T'HH:mm:ss.ffffffzzz";$a.recorded_at=$now.ToString($format);$a.expires_at=$now.AddHours(24).ToString($format)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
if ($Stage -eq 'Validation') {
    python -B scripts/native_routing_transfer.py validation-execute --execute --approval $target
    if ($LASTEXITCODE -ne 0) {throw 'Stopped. Preserve partial requests. No retry/resume.'}
    python -B scripts/native_routing_transfer.py validation-analyze
} else {
    python -B scripts/run_native_menu_full50.py execute --stage $Stage --execute --approval $target
    if ($LASTEXITCODE -ne 0) {throw 'Stopped. Preserve partial requests. No retry/resume.'}
    python -B scripts/run_native_menu_full50.py analyze --stage $Stage
}
if ($LASTEXITCODE -ne 0) {throw 'Saved analysis failed. Do not regenerate.'}
