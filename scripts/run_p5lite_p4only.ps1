param(
    [ValidateSet(50, 150)]
    [int]$Epochs = 50
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\p5lite_p4only'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunRoot = Join-Path $ProjectRoot 'runs'
$Tag = "${Epochs}e"
$StatusPath = Join-Path $ReportRoot "status_${Tag}.json"
$RunName = "p5lite_p4only_seed0_${Tag}"
$EvalName = "p5lite_p4only_seed0_${Tag}_eval"
$SizeEvalName = "p5lite_p4only_seed0_${Tag}_size_eval"
$GroundTruth = Join-Path $ProjectRoot 'reports\scale_ablation\metrics\visdrone_val_coco_ground_truth.json'

if ($Epochs -eq 50) {
    $ControlStandard = Join-Path $ProjectRoot 'reports\nwd_loss_ablation\metrics\inner_wiou_val.json'
    $ControlSize = Join-Path $ProjectRoot 'reports\nwd_loss_ablation\metrics\inner_wiou_size_val.json'
}
else {
    $ControlStandard = Join-Path $ProjectRoot 'reports\unified_evaluation\metrics\full_val.json'
    $ControlSize = Join-Path $ProjectRoot 'reports\unified_evaluation\metrics\full_size_val.json'
}

New-Item -ItemType Directory -Force -Path $LogRoot, $MetricRoot | Out-Null

function Update-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    [ordered]@{
        updated_at = (Get-Date).ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        seed = 0
        epochs = $Epochs
        candidate = 'full_p5lite_p4only'
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
    foreach ($Path in @(
        $Python,
        (Join-Path $ProjectRoot 'yolo11s.pt'),
        $ControlStandard,
        $ControlSize,
        $GroundTruth
    )) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Required file is missing: $Path"
        }
    }
    foreach ($Name in @($RunName, $EvalName, $SizeEvalName)) {
        $Path = Join-Path $RunRoot $Name
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing run directory: $Path"
        }
    }

    Set-Location -LiteralPath $ProjectRoot
    Invoke-Logged -Label "train-${Tag}" -LogPath (Join-Path $LogRoot "p5lite_p4only_${Tag}_train.log") -Arguments @(
        'train.py', '--stage', 'full_p5lite_p4only', '--weights', 'yolo11s.pt',
        '--epochs', [string]$Epochs, '--seed', '0', '--name', $RunName
    )

    $BestWeights = Join-Path $RunRoot "$RunName\weights\best.pt"
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing candidate checkpoint: $BestWeights"
    }
    Invoke-Logged -Label "validation-${Tag}" -LogPath (Join-Path $LogRoot "p5lite_p4only_${Tag}_val.log") -Arguments @(
        'evaluate.py', '--stage', 'full_p5lite_p4only', '--weights', $BestWeights,
        '--split', 'val', '--imgsz', '640', '--batch', '4', '--device', '0', '--workers', '0',
        '--conf', '0.001', '--iou', '0.7', '--max-det', '300', '--no-half', '--plots',
        '--project', $RunRoot, '--name', $EvalName,
        '--output-json', (Join-Path $MetricRoot "p5lite_p4only_${Tag}_val.json")
    )
    Invoke-Logged -Label "size-validation-${Tag}" -LogPath (Join-Path $LogRoot "p5lite_p4only_${Tag}_size_val.log") -Arguments @(
        'scripts\evaluate_object_sizes.py', '--stage', 'full_p5lite_p4only', '--weights', $BestWeights,
        '--split', 'val', '--imgsz', '640', '--batch', '4', '--device', '0', '--workers', '0',
        '--conf', '0.001', '--iou', '0.7', '--max-det', '300', '--no-half',
        '--project', $RunRoot, '--name', $SizeEvalName, '--ground-truth-json', $GroundTruth,
        '--output-json', (Join-Path $MetricRoot "p5lite_p4only_${Tag}_size_val.json")
    )
    Invoke-Logged -Label "report-${Tag}" -LogPath (Join-Path $LogRoot "p5lite_p4only_${Tag}_report.log") -Arguments @(
        'scripts\generate_p5lite_p4only_report.py', '--epochs', [string]$Epochs
    )
    Update-Status -State 'completed' -Stage 'done' -Message "P5-lite P4-only ${Tag} training and evaluations completed."
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot "completed_${Tag}.flag") -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot "p5lite_p4only_${Tag}_error.log") -Encoding UTF8
    throw
}
