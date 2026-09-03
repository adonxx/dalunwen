param(
    [string]$Device = '0'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Weights = Join-Path $ProjectRoot 'runs\full_paper\weights\best.pt'
$ReportDir = Join-Path $ProjectRoot 'reports\sahi_inference'
$MetricsDir = Join-Path $ReportDir 'metrics'
$LogsDir = Join-Path $ReportDir 'logs'
$FiguresDir = Join-Path $ReportDir 'figures'
$StatusPath = Join-Path $ReportDir 'status.json'
$GroundTruth = Join-Path $MetricsDir 'visdrone_val_coco_ground_truth.json'

New-Item -ItemType Directory -Force -Path $MetricsDir, $LogsDir, $FiguresDir | Out-Null

function Write-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    $status = [ordered]@{
        updated_at = [DateTimeOffset]::Now.ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        process_id = $PID
        total_images_per_run = 548
        total_runs = 3
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

function Invoke-Evaluation {
    param(
        [string]$Key,
        [string]$Mode,
        [int]$Size
    )
    Write-Status -State 'running' -Stage $Key -Message "Running $Key on 548 validation images"
    $arguments = @(
        (Join-Path $ProjectRoot 'scripts\evaluate_sahi.py'),
        '--mode', $Mode,
        '--weights', $Weights,
        '--input-size', $Size,
        '--batch', '4',
        '--device', $Device,
        '--conf', '0.001',
        '--iou', '0.7',
        '--merge-iou', '0.5',
        '--max-det', '300',
        '--output-json', (Join-Path $MetricsDir "$Key.json"),
        '--predictions-json', (Join-Path $MetricsDir "${Key}_predictions.json"),
        '--ground-truth-json', $GroundTruth
    )
    if ($Mode -eq 'sliced') {
        $arguments += @('--slice-size', $Size, '--overlap', '0.2')
    }
    & $Python @arguments *> (Join-Path $LogsDir "$Key.log")
    if ($LASTEXITCODE -ne 0) {
        throw "$Key failed with exit code $LASTEXITCODE"
    }
}

try {
    Invoke-Evaluation -Key 'whole' -Mode 'whole' -Size 640
    Invoke-Evaluation -Key 'sahi640' -Mode 'sliced' -Size 640
    Invoke-Evaluation -Key 'sahi960' -Mode 'sliced' -Size 960
    Write-Status -State 'running' -Stage 'report' -Message 'Generating SAHI tables, figures, and Markdown report'
    & $Python (Join-Path $ProjectRoot 'scripts\generate_sahi_report.py') *> (Join-Path $LogsDir 'report_generation.log')
    if ($LASTEXITCODE -ne 0) {
        throw "report generation failed with exit code $LASTEXITCODE"
    }
    New-Item -ItemType File -Force -Path (Join-Path $ReportDir 'completed.flag') | Out-Null
    Write-Status -State 'completed' -Stage 'done' -Message 'Three SAHI inference runs and report completed'
}
catch {
    Write-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    throw
}
