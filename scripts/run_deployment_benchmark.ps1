$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\deployment_benchmark'
$LogRoot = Join-Path $ReportRoot 'logs'
$StatusPath = Join-Path $ReportRoot 'status.json'

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
function Update-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        process_id = $PID
        warmup = 50
        repeats = 200
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
    Update-Status -State 'running' -Stage 'benchmark' -Message 'Running six checkpoints in FP32 and FP16.'
    Set-Location -LiteralPath $ProjectRoot
    & $Python scripts\benchmark_end_to_end.py --warmup 50 --repeats 200 2>&1 | Tee-Object -FilePath (Join-Path $LogRoot 'deployment_benchmark.log')
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed with exit code $LASTEXITCODE" }
    Update-Status -State 'completed' -Stage 'done' -Message 'Three-seed end-to-end deployment benchmark completed.'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'completed.flag') -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'pipeline_error.log') -Encoding UTF8
    throw
}
