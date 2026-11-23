param([ValidateSet('Smoke','Full')][string]$Stage='Smoke',[switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$name=if ($Stage -eq 'Smoke') {'u2_native_5agent_smoke_001'} else {'u2_native_5agent_full_001'}
$private="outputs/private/research_evaluation/B34_DIAGNOSIS_RECOVERY_20261123_v1/$name"
python -B scripts/run_u2_native_collection.py check --stage $Stage
if ($LASTEXITCODE -ne 0) {throw 'Native collection preflight blocked. No model request.'}
$cap=if ($Stage -eq 'Smoke') {60} else {600}
Write-Host "Five FIXED Specialty Agents, real Native Ask/Test/diagnosis, Top5, VoI OFF. Maximum $cap Generation. No fitting, no B3/B4 evaluation. Validation/M5 overlap still blocks B4 admission."
if (-not $Execute) {return}
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Approval already exists. Inspect saved state; no overwrite/resume.'}
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
if ((Read-Host "Type APPROVE_NATIVE_COLLECTION to accept this exact $Stage plan and maximum $cap requests") -cne 'APPROVE_NATIVE_COLLECTION') {throw 'Not approved'}
$a.operator_ref=Read-Host 'Your actual private operator reference'
$a.user_statement=Read-Host 'Your actual statement accepting Native five-Agent intervention, Source-label and recorded-time proxies'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual operator statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other work not confirmed'}
foreach ($k in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_native_training_intervention','accept_source_reference_date_proxy')) {$a.$k=$true}
$now=[DateTimeOffset]::UtcNow;$format="yyyy-MM-dd'T'HH:mm:ss.ffffffzzz";$a.recorded_at=$now.ToString($format);$a.expires_at=$now.AddHours(24).ToString($format)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
python -B scripts/run_u2_native_collection.py execute --stage $Stage --execute --approval $target
if ($LASTEXITCODE -ne 0) {throw 'Stopped. Preserve partial results; no retry/resume.'}
python -B scripts/run_u2_native_collection.py analyze --stage $Stage
if ($LASTEXITCODE -ne 0) {throw 'Analyze saved trace; do not regenerate.'}
