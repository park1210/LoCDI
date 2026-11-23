param([ValidateSet('Validation','Smoke5','Verify10','Full50')][string]$Stage='Validation',[switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
python -B scripts/native_transfer_recovery.py check --stage $Stage
if ($LASTEXITCODE -ne 0) {throw 'Recovery/Path/Admission check blocked. No generation.'}
if (-not $Execute) {return}
$name=switch($Stage){'Validation'{'v003'} 'Smoke5'{'s3'} 'Verify10'{'q3'} 'Full50'{'f3'}}
$private="outputs/private/nt3/$name"
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Approval already exists. No overwrite/resume.'}
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
Write-Host 'New short-path plan; old7 generations/results remain separate. Reuse fitted candidate, no refit. Same Source/model/roles/budget. No clinical review claim.'
$phrase='APPROVE_PATH_RECOVERY_'+$Stage.ToUpper()
if ((Read-Host "Type $phrase for this exact fresh plan") -cne $phrase) {throw 'Not approved'}
$a.operator_ref=Read-Host 'Actual private operator reference'
$a.user_statement=Read-Host 'Actual statement accepting new plan, repeated10 validation episodes after storage failure, prior exposure/Source proxy and call/resource limits'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual operator statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other work not excluded'}
foreach($k in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_prior_VoI_exposure_and_bounded_independence')) {$a.$k=$true}
$a | Add-Member -NotePropertyName accept_storage_recovery_and_repeat_validation -NotePropertyValue $true -Force
if ($Stage -eq 'Validation') {$a.accept_native_training_intervention=$true;$a.accept_source_reference_date_proxy=$true}
else {$a.accept_native_public_menu_feature=$true;$a.accept_label_conditioned_development=$true;$a.accept_augmentation_initial_time_policy=$true}
$now=[DateTimeOffset]::UtcNow;$format="yyyy-MM-dd'T'HH:mm:ss.ffffffzzz";$a.recorded_at=$now.ToString($format);$a.expires_at=$now.AddHours(24).ToString($format)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
python -B scripts/native_transfer_recovery.py execute --stage $Stage --execute --approval $target
if ($LASTEXITCODE -ne 0) {throw 'Stopped. Keep partial checkpoints. Do not retry/resume.'}
python -B scripts/native_transfer_recovery.py analyze --stage $Stage
if ($LASTEXITCODE -ne 0) {throw 'Stored analysis failed; do not regenerate.'}
