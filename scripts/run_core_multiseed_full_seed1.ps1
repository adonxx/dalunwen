param(
    [int]$Epochs = 150
)

$ErrorActionPreference = 'Stop'
if ($Epochs -ne 150) {
    throw 'This pre-registered full-model replication is fixed at 150 epochs.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\core_multiseed'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunRoot = Join-Path $ProjectRoot 'runs'
$StatusPath = Join-Path $ReportRoot 'status.json'
$RunName = 'full_paper_seed1'
$RunDir = Join-Path $RunRoot $RunName

New-Item -ItemType Directory -Force -Path $LogRoot, $MetricRoot | Out-Null
Set-Location -LiteralPath $ProjectRoot

function Update-Status {
    param(
        [Parameter(Mandatory)][string]$State,
        [Parameter(Mandatory)][string]$Stage,
        [Parameter(Mandatory)][string]$Message
    )

    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        current_seed = 1
        current_variant = 'full'
        epochs = $Epochs
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Logged {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$LogPath
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
    foreach ($Name in @($RunName, 'core_multiseed_full_seed1_eval', 'core_multiseed_full_seed1_size_eval')) {
        $Path = Join-Path $RunRoot $Name
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing run directory: $Path"
        }
    }

    Invoke-Logged -Label 'full-seed1-train' -LogPath (Join-Path $LogRoot 'full_seed1_train.log') -Arguments @(
        'train.py',
        '--stage', 'full',
        '--weights', 'yolo11s.pt',
        '--epochs', [string]$Epochs,
        '--seed', '1',
        '--name', $RunName
    )

    $BestWeights = Join-Path $RunDir 'weights\best.pt'
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing best checkpoint: $BestWeights"
    }

    Invoke-Logged -Label 'full-seed1-validation' -LogPath (Join-Path $LogRoot 'full_seed1_val.log') -Arguments @(
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
        '--name', 'core_multiseed_full_seed1_eval',
        '--output-json', (Join-Path $MetricRoot 'full_seed1_val.json')
    )

    Invoke-Logged -Label 'full-seed1-size-validation' -LogPath (Join-Path $LogRoot 'full_seed1_size_val.log') -Arguments @(
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
        '--name', 'core_multiseed_full_seed1_size_eval',
        '--ground-truth-json', (Join-Path $ProjectRoot 'reports\scale_ablation\metrics\visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot 'full_seed1_size_val.json')
    )

    Update-Status -State 'completed' -Stage 'done' -Message 'Full-model seed=1 training and both fixed evaluations completed'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'full_seed1_completed.flag') -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'full_seed1_pipeline_error.log') -Encoding UTF8
    throw
}
