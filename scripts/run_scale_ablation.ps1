param(
    [int]$Epochs = 50
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\scale_ablation'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunRoot = Join-Path $ProjectRoot 'runs'
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

function Run-Stage {
    param(
        [string]$Stage,
        [string]$RunName
    )
    $RunDir = Join-Path $RunRoot $RunName
    if (Test-Path -LiteralPath $RunDir) {
        throw "Refusing to overwrite existing run directory: $RunDir"
    }

    Invoke-Logged -Label "${Stage}:train" -LogPath (Join-Path $LogRoot "${Stage}_train.log") -Arguments @(
        'train.py',
        '--stage', $Stage,
        '--weights', 'yolo11s.pt',
        '--epochs', [string]$Epochs,
        '--name', $RunName
    )

    $BestWeights = Join-Path $RunDir 'weights\best.pt'
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing best checkpoint: $BestWeights"
    }

    Invoke-Logged -Label "${Stage}:validation" -LogPath (Join-Path $LogRoot "${Stage}_val.log") -Arguments @(
        'evaluate.py',
        '--stage', $Stage,
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
        '--name', "scale_eval_${Stage}_${Epochs}e",
        '--output-json', (Join-Path $MetricRoot "${Stage}_val.json")
    )

    Invoke-Logged -Label "${Stage}:size-validation" -LogPath (Join-Path $LogRoot "${Stage}_size_val.log") -Arguments @(
        'scripts\evaluate_object_sizes.py',
        '--stage', $Stage,
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
        '--name', "scale_size_eval_${Stage}_${Epochs}e",
        '--ground-truth-json', (Join-Path $MetricRoot 'visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot "${Stage}_size_val.json")
    )
}

try {
    Update-Status -State 'running' -Stage 'initializing' -Message 'Scale-ablation pipeline started'
    Run-Stage -Stage 'mffpn_p2p5' -RunName "scale_mffpn_p2p5_${Epochs}e"
    Run-Stage -Stage 'mffpn_p3p5' -RunName "scale_mffpn_p3p5_${Epochs}e"
    Invoke-Logged -Label 'report' -LogPath (Join-Path $LogRoot 'report_generation.log') -Arguments @(
        'scripts\generate_scale_ablation_report.py',
        '--report-dir', $ReportRoot
    )
    Update-Status -State 'completed' -Stage 'done' -Message 'Training, evaluation, and report generation completed'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'completed.flag') -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'pipeline_error.log') -Encoding UTF8
    throw
}
