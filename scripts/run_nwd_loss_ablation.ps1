param(
    [int]$Epochs = 50
)

$ErrorActionPreference = 'Stop'
if ($Epochs -ne 50) {
    throw 'This NWD screening pipeline is pre-registered for exactly 50 epochs.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\nwd_loss_ablation'
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
        [string]$Message,
        [string]$Variant = ''
    )
    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        current_variant = $Variant
        epochs_per_run = $Epochs
        total_runs = 3
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Logged {
    param(
        [string]$Label,
        [string[]]$Arguments,
        [string]$LogPath,
        [string]$Variant
    )
    Update-Status -State 'running' -Stage $Label -Message "Running $Label" -Variant $Variant
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Run-Variant {
    param(
        [string]$Variant,
        [string]$Stage,
        [string]$RunName
    )
    $RunDir = Join-Path $RunRoot $RunName
    Invoke-Logged -Label "${Variant}:train" -LogPath (Join-Path $LogRoot "${Variant}_train.log") -Variant $Variant -Arguments @(
        'train.py',
        '--stage', $Stage,
        '--weights', 'yolo11s.pt',
        '--epochs', [string]$Epochs,
        '--seed', '0',
        '--name', $RunName
    )

    $BestWeights = Join-Path $RunDir 'weights\best.pt'
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing best checkpoint: $BestWeights"
    }

    Invoke-Logged -Label "${Variant}:validation" -LogPath (Join-Path $LogRoot "${Variant}_val.log") -Variant $Variant -Arguments @(
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
        '--name', "loss_eval_${Variant}_${Epochs}e",
        '--output-json', (Join-Path $MetricRoot "${Variant}_val.json")
    )

    Invoke-Logged -Label "${Variant}:size-validation" -LogPath (Join-Path $LogRoot "${Variant}_size_val.log") -Variant $Variant -Arguments @(
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
        '--name', "loss_size_eval_${Variant}_${Epochs}e",
        '--ground-truth-json', (Join-Path $ProjectRoot 'reports\scale_ablation\metrics\visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot "${Variant}_size_val.json")
    )
}

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Missing project Python environment: $Python"
    }

    $ExpectedRunNames = @(
        "loss_inner_wiou_${Epochs}e",
        "loss_nwd_${Epochs}e",
        "loss_hybrid_${Epochs}e",
        "loss_eval_inner_wiou_${Epochs}e",
        "loss_eval_nwd_${Epochs}e",
        "loss_eval_hybrid_${Epochs}e",
        "loss_size_eval_inner_wiou_${Epochs}e",
        "loss_size_eval_nwd_${Epochs}e",
        "loss_size_eval_hybrid_${Epochs}e"
    )
    foreach ($Name in $ExpectedRunNames) {
        $Path = Join-Path $RunRoot $Name
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing run directory: $Path"
        }
    }

    Update-Status -State 'running' -Stage 'initializing' -Message 'NWD loss-ablation pipeline started'
    Run-Variant -Variant 'inner_wiou' -Stage 'full' -RunName "loss_inner_wiou_${Epochs}e"
    Run-Variant -Variant 'nwd' -Stage 'lscd_nwd' -RunName "loss_nwd_${Epochs}e"
    Run-Variant -Variant 'hybrid' -Stage 'full_nwd_hybrid' -RunName "loss_hybrid_${Epochs}e"

    Invoke-Logged -Label 'report' -LogPath (Join-Path $LogRoot 'report_generation.log') -Variant 'aggregate' -Arguments @(
        'scripts\generate_nwd_loss_report.py',
        '--report-dir', $ReportRoot
    )

    Update-Status -State 'completed' -Stage 'done' -Message 'Three runs, all evaluations, and NWD report completed'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'completed.flag') -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'pipeline_error.log') -Encoding UTF8
    throw
}
