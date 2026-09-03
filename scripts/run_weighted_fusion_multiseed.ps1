param(
    [int]$Epochs = 50
)

$ErrorActionPreference = 'Stop'
if ($Epochs -ne 50) {
    throw 'This paired multi-seed pipeline is pre-registered for exactly 50 epochs.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\weighted_fusion_multiseed'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunRoot = Join-Path $ProjectRoot 'runs'
$StatusPath = Join-Path $ReportRoot 'status.json'
$Seeds = @(1, 2)

New-Item -ItemType Directory -Force -Path $LogRoot, $MetricRoot | Out-Null
Set-Location -LiteralPath $ProjectRoot

function Update-Status {
    param(
        [string]$State,
        [string]$Stage,
        [string]$Message,
        [int]$Seed = -1,
        [string]$Variant = ''
    )
    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        current_seed = $Seed
        current_variant = $Variant
        epochs_per_run = $Epochs
        total_new_runs = 4
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Logged {
    param(
        [string]$Label,
        [string[]]$Arguments,
        [string]$LogPath,
        [int]$Seed,
        [string]$Variant
    )
    Update-Status -State 'running' -Stage $Label -Message "Running $Label" -Seed $Seed -Variant $Variant
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Run-Experiment {
    param(
        [int]$Seed,
        [string]$Variant,
        [string]$Stage,
        [string]$Stem
    )
    $RunName = "${Stem}_seed${Seed}_${Epochs}e"
    $RunDir = Join-Path $RunRoot $RunName

    Invoke-Logged -Label "seed${Seed}:${Variant}:train" -LogPath (Join-Path $LogRoot "${Stem}_seed${Seed}_train.log") -Seed $Seed -Variant $Variant -Arguments @(
        'train.py',
        '--stage', $Stage,
        '--weights', 'yolo11s.pt',
        '--epochs', [string]$Epochs,
        '--seed', [string]$Seed,
        '--name', $RunName
    )

    $BestWeights = Join-Path $RunDir 'weights\best.pt'
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing best checkpoint: $BestWeights"
    }

    Invoke-Logged -Label "seed${Seed}:${Variant}:validation" -LogPath (Join-Path $LogRoot "${Stem}_seed${Seed}_val.log") -Seed $Seed -Variant $Variant -Arguments @(
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
        '--name', "multiseed_eval_${Stem}_seed${Seed}_${Epochs}e",
        '--output-json', (Join-Path $MetricRoot "${Stem}_seed${Seed}_val.json")
    )

    Invoke-Logged -Label "seed${Seed}:${Variant}:size-validation" -LogPath (Join-Path $LogRoot "${Stem}_seed${Seed}_size_val.log") -Seed $Seed -Variant $Variant -Arguments @(
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
        '--name', "multiseed_size_eval_${Stem}_seed${Seed}_${Epochs}e",
        '--ground-truth-json', (Join-Path $ProjectRoot 'reports\scale_ablation\metrics\visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot "${Stem}_seed${Seed}_size_val.json")
    )
}

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Missing project Python environment: $Python"
    }

    $ExpectedRunNames = foreach ($Seed in $Seeds) {
        "mffpn_seed${Seed}_${Epochs}e"
        "mffpn_weighted_seed${Seed}_${Epochs}e"
        "multiseed_eval_mffpn_seed${Seed}_${Epochs}e"
        "multiseed_eval_mffpn_weighted_seed${Seed}_${Epochs}e"
        "multiseed_size_eval_mffpn_seed${Seed}_${Epochs}e"
        "multiseed_size_eval_mffpn_weighted_seed${Seed}_${Epochs}e"
    }
    foreach ($Name in $ExpectedRunNames) {
        $Path = Join-Path $RunRoot $Name
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing run directory: $Path"
        }
    }

    Update-Status -State 'running' -Stage 'initializing' -Message 'Three-seed paired replication started'
    foreach ($Seed in $Seeds) {
        Run-Experiment -Seed $Seed -Variant 'control' -Stage 'mffpn' -Stem 'mffpn'
        Run-Experiment -Seed $Seed -Variant 'weighted' -Stage 'mffpn_weighted' -Stem 'mffpn_weighted'
    }

    Invoke-Logged -Label 'report' -LogPath (Join-Path $LogRoot 'report_generation.log') -Seed -1 -Variant 'aggregate' -Arguments @(
        'scripts\generate_weighted_fusion_multiseed_report.py',
        '--report-dir', $ReportRoot
    )

    Update-Status -State 'completed' -Stage 'done' -Message 'Four new runs, all evaluations, and the three-seed report completed'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'completed.flag') -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'pipeline_error.log') -Encoding UTF8
    throw
}
