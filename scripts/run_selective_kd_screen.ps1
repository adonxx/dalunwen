param(
    [int]$Epochs = 50,
    [double]$Weight = 0.5
)

$ErrorActionPreference = 'Stop'
if ($Epochs -ne 50) {
    throw 'Selective-KD screening is pre-registered for exactly 50 epochs.'
}
if ($Weight -ne 0.5) {
    throw 'Selective-KD screening is pre-registered for weight=0.5.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Teacher = Join-Path $ProjectRoot 'runs\baseline_paper\weights\best.pt'
$RunRoot = Join-Path $ProjectRoot 'runs'
$ReportRoot = Join-Path $ProjectRoot 'reports\selective_kd'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunName = 'selective_kd_p34_seed0_50e'
$EvalName = 'selective_kd_p34_seed0_50e_eval'
$SizeEvalName = 'selective_kd_p34_seed0_50e_size_eval'
$StatusPath = Join-Path $ReportRoot 'status.json'

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
        kd_weight = $Weight
        teacher = $Teacher
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
    foreach ($Path in @($Python, $Teacher)) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Required file is missing: $Path"
        }
    }
    foreach ($Name in @($RunName, $EvalName, $SizeEvalName)) {
        if (Test-Path -LiteralPath (Join-Path $RunRoot $Name)) {
            throw "Refusing to overwrite existing run directory: $(Join-Path $RunRoot $Name)"
        }
    }

    Set-Location -LiteralPath $ProjectRoot
    Invoke-Logged -Label 'train' -LogPath (Join-Path $LogRoot 'selective_kd_train.log') -Arguments @(
        'train.py',
        '--stage', 'full',
        '--weights', 'yolo11s.pt',
        '--epochs', [string]$Epochs,
        '--seed', '0',
        '--name', $RunName,
        '--selective-kd-teacher', $Teacher,
        '--selective-kd-weight', [string]$Weight
    )

    $BestWeights = Join-Path $RunRoot "$RunName\weights\best.pt"
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing student checkpoint: $BestWeights"
    }
    Invoke-Logged -Label 'validation' -LogPath (Join-Path $LogRoot 'selective_kd_val.log') -Arguments @(
        'evaluate.py', '--stage', 'full', '--weights', $BestWeights,
        '--split', 'val', '--imgsz', '640', '--batch', '4', '--device', '0', '--workers', '0',
        '--conf', '0.001', '--iou', '0.7', '--max-det', '300', '--no-half', '--plots',
        '--project', $RunRoot, '--name', $EvalName,
        '--output-json', (Join-Path $MetricRoot 'selective_kd_val.json')
    )
    Invoke-Logged -Label 'size-validation' -LogPath (Join-Path $LogRoot 'selective_kd_size_val.log') -Arguments @(
        'scripts\evaluate_object_sizes.py', '--stage', 'full', '--weights', $BestWeights,
        '--split', 'val', '--imgsz', '640', '--batch', '4', '--device', '0', '--workers', '0',
        '--conf', '0.001', '--iou', '0.7', '--max-det', '300', '--no-half',
        '--project', $RunRoot, '--name', $SizeEvalName,
        '--ground-truth-json', (Join-Path $ProjectRoot 'reports\scale_ablation\metrics\visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot 'selective_kd_size_val.json')
    )
    Invoke-Logged -Label 'report' -LogPath (Join-Path $LogRoot 'selective_kd_report.log') -Arguments @(
        'scripts\generate_selective_kd_report.py'
    )
    Update-Status -State 'completed' -Stage 'done' -Message 'Selective-KD 50-epoch screen and fixed evaluations completed.'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'completed.flag') -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'pipeline_error.log') -Encoding UTF8
    throw
}
