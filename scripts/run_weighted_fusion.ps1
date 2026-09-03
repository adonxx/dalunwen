param(
    [int]$Epochs = 50
)

$ErrorActionPreference = 'Stop'
if ($Epochs -ne 50) {
    throw 'This screening pipeline is pre-registered for exactly 50 epochs.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\weighted_fusion'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunRoot = Join-Path $ProjectRoot 'runs'
$RunName = 'mffpn_weighted_50e'
$RunDir = Join-Path $RunRoot $RunName
$StatusPath = Join-Path $ReportRoot 'status.json'

New-Item -ItemType Directory -Force -Path $LogRoot, $MetricRoot | Out-Null
Set-Location -LiteralPath $ProjectRoot

function Update-Status {
    param(
        [string]$State,
        [string]$Stage,
        [string]$Message
    )
    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        epochs = $Epochs
        process_id = $PID
        run_name = $RunName
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Logged {
    param(
        [string]$Label,
        [string[]]$Arguments,
        [string]$LogPath
    )
    Update-Status -State 'running' -Stage $Label -Message "Running $Label"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Missing project Python environment: $Python"
    }
    if (Test-Path -LiteralPath $RunDir) {
        throw "Refusing to overwrite existing run directory: $RunDir"
    }

    Update-Status -State 'running' -Stage 'initializing' -Message 'Adaptive weighted-fusion 50-epoch screening started'
    Invoke-Logged -Label 'train' -LogPath (Join-Path $LogRoot 'mffpn_weighted_train.log') -Arguments @(
        'train.py',
        '--stage', 'mffpn_weighted',
        '--weights', 'yolo11s.pt',
        '--epochs', [string]$Epochs,
        '--name', $RunName
    )

    $BestWeights = Join-Path $RunDir 'weights\best.pt'
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing best checkpoint: $BestWeights"
    }

    Invoke-Logged -Label 'validation' -LogPath (Join-Path $LogRoot 'mffpn_weighted_val.log') -Arguments @(
        'evaluate.py',
        '--stage', 'mffpn_weighted',
        '--weights', $BestWeights,
        '--split', 'val',
        '--imgsz', '640',
        '--batch', '4',
        '--device', '0',
        '--workers', '0',
        '--conf', '0.001',
        '--iou', '0.7',
        '--max-det', '300',
        '--no-half',
        '--plots',
        '--project', $RunRoot,
        '--name', 'weighted_eval_mffpn_50e',
        '--output-json', (Join-Path $MetricRoot 'mffpn_weighted_val.json')
    )

    Invoke-Logged -Label 'size-validation' -LogPath (Join-Path $LogRoot 'mffpn_weighted_size_val.log') -Arguments @(
        'scripts\evaluate_object_sizes.py',
        '--stage', 'mffpn_weighted',
        '--weights', $BestWeights,
        '--split', 'val',
        '--imgsz', '640',
        '--batch', '4',
        '--device', '0',
        '--workers', '0',
        '--conf', '0.001',
        '--iou', '0.7',
        '--max-det', '300',
        '--no-half',
        '--project', $RunRoot,
        '--name', 'weighted_size_eval_mffpn_50e',
        '--ground-truth-json', (Join-Path $ProjectRoot 'reports\scale_ablation\metrics\visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot 'mffpn_weighted_size_val.json')
    )

    Invoke-Logged -Label 'report' -LogPath (Join-Path $LogRoot 'report_generation.log') -Arguments @(
        'scripts\generate_weighted_fusion_report.py',
        '--report-dir', $ReportRoot
    )

    Update-Status -State 'completed' -Stage 'done' -Message 'Training, evaluation, fusion-weight extraction, and report generation completed'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'completed.flag') -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'pipeline_error.log') -Encoding UTF8
    throw
}
