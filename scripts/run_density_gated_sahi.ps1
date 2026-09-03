param([string]$Device = '0')

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportDir = Join-Path $ProjectRoot 'reports\density_gated_sahi'
$MetricsDir = Join-Path $ReportDir 'metrics'
$LogsDir = Join-Path $ReportDir 'logs'
$FiguresDir = Join-Path $ReportDir 'figures'
$StatusPath = Join-Path $ReportDir 'status.json'
$Weights = Join-Path $ProjectRoot 'runs\full_paper\weights\best.pt'
$GroundTruth = Join-Path $ProjectRoot 'reports\sahi_inference\metrics\visdrone_val_coco_ground_truth.json'
$Calibration = Join-Path $MetricsDir 'calibration.json'

New-Item -ItemType Directory -Force -Path $MetricsDir, $LogsDir, $FiguresDir | Out-Null

function Write-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    [ordered]@{
        updated_at = [DateTimeOffset]::Now.ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

function Invoke-Gate {
    param([string]$Strategy)
    Write-Status -State 'running' -Stage $Strategy -Message "Running $Strategy on 548 validation images"
    & $Python (Join-Path $ProjectRoot 'scripts\evaluate_density_gate.py') `
        --mode evaluate `
        --weights $Weights `
        --calibration $Calibration `
        --strategy $Strategy `
        --ground-truth $GroundTruth `
        --output-json (Join-Path $MetricsDir "$Strategy.json") `
        --predictions-json (Join-Path $MetricsDir "${Strategy}_predictions.json") `
        --device $Device `
        --batch 4 `
        --conf 0.001 `
        --iou 0.7 `
        --merge-iou 0.5 `
        --max-det 300 *> (Join-Path $LogsDir "$Strategy.log")
    if ($LASTEXITCODE -ne 0) { throw "$Strategy failed with exit code $LASTEXITCODE" }
}

try {
    Write-Status -State 'running' -Stage 'calibration' -Message 'Calibrating density thresholds on 512 unlabeled training images'
    & $Python (Join-Path $ProjectRoot 'scripts\evaluate_density_gate.py') `
        --mode calibrate `
        --weights $Weights `
        --output-json $Calibration `
        --calibration-images 512 `
        --device $Device `
        --batch 4 `
        --conf 0.001 `
        --iou 0.7 `
        --merge-iou 0.5 `
        --max-det 300 *> (Join-Path $LogsDir 'calibration.log')
    if ($LASTEXITCODE -ne 0) { throw "calibration failed with exit code $LASTEXITCODE" }

    Invoke-Gate -Strategy 'top1'
    Invoke-Gate -Strategy 'gate_q20'
    Invoke-Gate -Strategy 'gate_q30'

    Write-Status -State 'running' -Stage 'report' -Message 'Generating density gate tables, figures, and Markdown report'
    & $Python (Join-Path $ProjectRoot 'scripts\generate_density_gate_report.py') *> (Join-Path $LogsDir 'report_generation.log')
    if ($LASTEXITCODE -ne 0) { throw "report generation failed with exit code $LASTEXITCODE" }

    New-Item -ItemType File -Force -Path (Join-Path $ReportDir 'completed.flag') | Out-Null
    Write-Status -State 'completed' -Stage 'done' -Message 'Density-gated SAHI calibration, three evaluations, and report completed'
}
catch {
    Write-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    throw
}
