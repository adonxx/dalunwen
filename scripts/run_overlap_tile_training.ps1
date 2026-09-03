param(
    [int]$Epochs = 50
)

$ErrorActionPreference = 'Stop'
if ($Epochs -ne 50) {
    throw 'This overlap-tile screening pipeline is pre-registered for exactly 50 epochs.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\overlap_tile_training'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunRoot = Join-Path $ProjectRoot 'runs'
$StatusPath = Join-Path $ReportRoot 'status.json'
$RunName = "overlap_tiles_${Epochs}e"

New-Item -ItemType Directory -Force -Path $LogRoot, $MetricRoot | Out-Null
Set-Location -LiteralPath $ProjectRoot

function Update-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        current_variant = 'full_with_overlap_tile_training'
        epochs = $Epochs
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Logged {
    param([string]$Label, [string[]]$Arguments, [string]$LogPath)
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
    if (-not (Test-Path -LiteralPath (Join-Path $RunRoot 'loss_inner_wiou_50e\weights\best.pt'))) {
        throw 'Missing frozen 50-epoch control checkpoint.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ReportRoot 'EXPERIMENT_PLAN.md'))) {
        throw 'Missing experiment preregistration.'
    }

    foreach ($Name in @($RunName, "overlap_tiles_eval_${Epochs}e", "overlap_tiles_size_eval_${Epochs}e")) {
        $Path = Join-Path $RunRoot $Name
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing run directory: $Path"
        }
    }

    Update-Status -State 'running' -Stage 'initializing' -Message 'Overlap-tile training pipeline started'
    Invoke-Logged -Label 'train' -LogPath (Join-Path $LogRoot 'overlap_tiles_train.log') -Arguments @(
        'train.py',
        '--stage', 'full',
        '--weights', 'yolo11s.pt',
        '--overlap-tile-training',
        '--epochs', [string]$Epochs,
        '--seed', '0',
        '--name', $RunName
    )

    $BestWeights = Join-Path $RunRoot "$RunName\weights\best.pt"
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing best checkpoint: $BestWeights"
    }

    Invoke-Logged -Label 'validation' -LogPath (Join-Path $LogRoot 'overlap_tiles_val.log') -Arguments @(
        'evaluate.py',
        '--stage', 'full',
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
        '--name', "overlap_tiles_eval_${Epochs}e",
        '--output-json', (Join-Path $MetricRoot 'overlap_tiles_val.json')
    )

    Invoke-Logged -Label 'size-validation' -LogPath (Join-Path $LogRoot 'overlap_tiles_size_val.log') -Arguments @(
        'scripts\evaluate_object_sizes.py',
        '--stage', 'full',
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
        '--name', "overlap_tiles_size_eval_${Epochs}e",
        '--ground-truth-json', (Join-Path $ProjectRoot 'reports\scale_ablation\metrics\visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot 'overlap_tiles_size_val.json')
    )

    Invoke-Logged -Label 'report' -LogPath (Join-Path $LogRoot 'report_generation.log') -Arguments @(
        'scripts\generate_overlap_tile_report.py',
        '--report-dir', $ReportRoot
    )

    Update-Status -State 'completed' -Stage 'done' -Message 'Training, evaluations, figures, and report completed'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'completed.flag') -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'pipeline_error.log') -Encoding UTF8
    throw
}
