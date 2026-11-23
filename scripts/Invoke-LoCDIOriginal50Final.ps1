param([ValidateSet('Smoke5','Verify10','Full50')][string]$Stage='Smoke5',[switch]$AllConditions,[switch]$Execute)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if ($AllConditions -and $Execute) {throw 'B3/B4 admission is blocked. No all-condition inference is permitted.'}
$checkArgs=@('-B','scripts/run_original50_final.py','check','--stage',$Stage)
if ($AllConditions) {$checkArgs+='--all-conditions'}
& python @checkArgs
if ($LASTEXITCODE -ne 0) {throw 'Static/promotion/admission check blocked. No generation.'}
if (-not $Execute) {return}
$planId='o50_d2_'+$Stage.ToLower()+'_b012_002'
$private="outputs/private/research_evaluation/ORIGINAL50_FINAL_EXECUTION_20261123_v1/$planId"
$a=Get-Content -Raw -Encoding UTF8 -LiteralPath "$private/approval_template.json" | ConvertFrom-Json
$a | ConvertTo-Json -Depth 100 | Write-Host
Write-Host 'D2 only; B0/B1/B2 only. Label-conditioned synthetic Source agreement, not clinical accuracy. No B3/B4, fitting, server change, download or warmup.'
$target=Join-Path $private 'operator_approval.json'
if (Test-Path -LiteralPath $target) {throw 'Approval already exists: inspect results. No overwrite/resume.'}
$phrase='APPROVE_O50_'+$Stage.ToUpper()+'_B012'
if ((Read-Host "Accept exact displayed Plan/budget. Type $phrase") -cne $phrase) {throw 'Not approved'}
$a.operator_ref=Read-Host 'Actual private operator reference'
$a.user_statement=Read-Host 'Actual statement accepting exact D2 B012 plan, label-conditioned development, recorded-time proxy, reference timing limits, call/resource caps and no-resume'
if ([string]::IsNullOrWhiteSpace($a.operator_ref) -or [string]::IsNullOrWhiteSpace($a.user_statement)) {throw 'Actual operator reference and statement required'}
if ((Read-Host 'Type NO_OTHER_OLLAMA_WORK only if true') -cne 'NO_OTHER_OLLAMA_WORK') {throw 'Other Ollama work not excluded'}
foreach ($k in @('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy')) {$a.$k=$true}
$now=[DateTimeOffset]::UtcNow;$format="yyyy-MM-dd'T'HH:mm:ss.ffffffzzz";$a.recorded_at=$now.ToString($format);$a.expires_at=$now.AddHours(24).ToString($format)
$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($a | ConvertTo-Json -Depth 100));$stream=[IO.File]::Open($target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {$stream.Write($bytes,0,$bytes.Length)} finally {$stream.Dispose()}
python -B scripts/run_original50_final.py execute --stage $Stage --approval $target --execute --allow-llm
if ($LASTEXITCODE -ne 0) {throw 'Stopped. Preserve failed and not-attempted rows. No retry/resume/new approval reuse.'}
python -B scripts/run_original50_final.py analyze --stage $Stage
if ($LASTEXITCODE -ne 0) {throw 'Saved analysis failed. Inspect files; do not rerun generation.'}
