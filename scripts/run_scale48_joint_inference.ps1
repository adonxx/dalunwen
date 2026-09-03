param([string]$Device = '0')

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportDir = Join-Path $ProjectRoot 'reports\scale48_joint_inference'
$MetricsDir = Join-Path $ReportDir 'metrics'
$LogsDir = Join-Path $ReportDir 'logs'
$FiguresDir = Join-Path $ReportDir 'figures'
$StatusPath = Join-Path $ReportDir 'status.json'

New-Item -ItemType Directory -Force -Path $MetricsDir, $LogsDir, $FiguresDir | Out-Null

function Write-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    [ordered]@{
        updated_at = [DateTimeOffset]::Now.ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        process_id = $PID
        images = 548
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

try {
    Write-Status -State 'running' -Stage 'joint_inference' -Message 'Running true Scale-48 joint inference on 548 images'
    & $Python (Join-Path $ProjectRoot 'scripts\evaluate_scale48_joint.py') `
        --weights (Join-Path $ProjectRoot 'runs\full_paper\weights\best.pt') `
        --ground-truth (Join-Path $ProjectRoot 'reports\sahi_inference\metrics\visdrone_val_coco_ground_truth.json') `
        --offline-record (Join-Path $ProjectRoot 'reports\sahi_scale_fusion\metrics\scale_48.json') `
        --output-json (Join-Path $MetricsDir 'joint_scale48.json') `
        --predictions-json (Join-Path $MetricsDir 'joint_scale48_predictions.json') `
        --device $Device `
        --batch 4 `
        --conf 0.001 `
        --iou 0.7 `
        --merge-iou 0.5 `
        --max-det 300 `
        --slice-size 960 `
        --overlap 0.2 `
        --scale-threshold 48 *> (Join-Path $LogsDir 'joint_inference.log')
    if ($LASTEXITCODE -ne 0) { throw "joint inference failed with exit code $LASTEXITCODE" }

    Write-Status -State 'running' -Stage 'report' -Message 'Generating joint accuracy and latency report'
    & $Python (Join-Path $ProjectRoot 'scripts\generate_scale48_joint_report.py') *> (Join-Path $LogsDir 'report_generation.log')
    if ($LASTEXITCODE -ne 0) { throw "report generation failed with exit code $LASTEXITCODE" }

    New-Item -ItemType File -Force -Path (Join-Path $ReportDir 'completed.flag') | Out-Null
    Write-Status -State 'completed' -Stage 'done' -Message 'True Scale-48 joint inference and report completed'
}
catch {
    Write-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    throw
}
