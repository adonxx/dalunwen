param(
    [Parameter(Mandatory)]
    [ValidateSet('baseline', 'full')]
    [string]$Variant,

    [Parameter(Mandatory)]
    [ValidateSet(1, 2)]
    [int]$Seed,

    [int]$Epochs = 150
)

$ErrorActionPreference = 'Stop'
if ($Epochs -ne 150) {
    throw 'Core-model replications are pre-registered for exactly 150 epochs.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\core_multiseed'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunRoot = Join-Path $ProjectRoot 'runs'
$StatusPath = Join-Path $ReportRoot 'status.json'

if ($Variant -eq 'baseline') {
    $Stage = 'baseline'
    $RunName = "baseline_paper_seed${Seed}"
}
else {
    $Stage = 'full'
    $RunName = "full_paper_seed${Seed}"
}
$RunDir = Join-Path $RunRoot $RunName
$Stem = "${Variant}_seed${Seed}"
$EvalName = "core_multiseed_${Stem}_eval"
$SizeEvalName = "core_multiseed_${Stem}_size_eval"

New-Item -ItemType Directory -Force -Path $LogRoot, $MetricRoot | Out-Null
Set-Location -LiteralPath $ProjectRoot

function Update-Status {
    param(
        [Parameter(Mandatory)][string]$State,
        [Parameter(Mandatory)][string]$StageName,
        [Parameter(Mandatory)][string]$Message
    )

    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        state = $State
        stage = $StageName
        message = $Message
        current_seed = $Seed
        current_variant = $Variant
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

    Update-Status -State 'running' -StageName $Label -Message "Running $Label"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Missing project Python environment: $Python"
    }
    foreach ($Name in @($RunName, $EvalName, $SizeEvalName)) {
        $Path = Join-Path $RunRoot $Name
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing run directory: $Path"
        }
    }

    Invoke-Logged -Label "${Stem}-train" -LogPath (Join-Path $LogRoot "${Stem}_train.log") -Arguments @(
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

    Invoke-Logged -Label "${Stem}-validation" -LogPath (Join-Path $LogRoot "${Stem}_val.log") -Arguments @(
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
        '--name', $EvalName,
        '--output-json', (Join-Path $MetricRoot "${Stem}_val.json")
    )

    Invoke-Logged -Label "${Stem}-size-validation" -LogPath (Join-Path $LogRoot "${Stem}_size_val.log") -Arguments @(
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
        '--name', $SizeEvalName,
        '--ground-truth-json', (Join-Path $ProjectRoot 'reports\scale_ablation\metrics\visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot "${Stem}_size_val.json")
    )

    Update-Status -State 'completed' -StageName 'done' -Message "$Variant seed=$Seed training and both fixed evaluations completed"
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot "${Stem}_completed.flag") -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -StageName 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot "${Stem}_pipeline_error.log") -Encoding UTF8
    throw
}
