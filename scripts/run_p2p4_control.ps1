param(
    [int]$Epochs = 50
)

$ErrorActionPreference = 'Stop'
if ($Epochs -ne 50) {
    throw 'This control pipeline is fixed to 50 epochs so that it remains comparable with the scale-ablation runs.'
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReportRoot = Join-Path $ProjectRoot 'reports\scale_ablation'
$LogRoot = Join-Path $ReportRoot 'logs'
$MetricRoot = Join-Path $ReportRoot 'metrics'
$RunRoot = Join-Path $ProjectRoot 'runs'
$RunName = 'scale_mffpn_p2p4_control_50e'
$RunDir = Join-Path $RunRoot $RunName
$StatusPath = Join-Path $ReportRoot 'control_status.json'
$ControlNotePath = Join-Path $ReportRoot 'CONTROL_RUN.md'

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

    Update-Status -State 'running' -Stage 'initializing' -Message 'Independent P2-P4 50-epoch control started'

    Invoke-Logged -Label 'mffpn_p2p4_control:train' -LogPath (Join-Path $LogRoot 'mffpn_p2p4_control_train.log') -Arguments @(
        'train.py',
        '--stage', 'mffpn',
        '--weights', 'yolo11s.pt',
        '--epochs', [string]$Epochs,
        '--name', $RunName
    )

    $BestWeights = Join-Path $RunDir 'weights\best.pt'
    if (-not (Test-Path -LiteralPath $BestWeights -PathType Leaf)) {
        throw "Missing best checkpoint: $BestWeights"
    }

    Invoke-Logged -Label 'mffpn_p2p4_control:validation' -LogPath (Join-Path $LogRoot 'mffpn_p2p4_control_val.log') -Arguments @(
        'evaluate.py',
        '--stage', 'mffpn',
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
        '--name', 'scale_eval_mffpn_p2p4_control_50e',
        '--output-json', (Join-Path $MetricRoot 'mffpn_p2p4_control_val.json')
    )

    Invoke-Logged -Label 'mffpn_p2p4_control:size-validation' -LogPath (Join-Path $LogRoot 'mffpn_p2p4_control_size_val.log') -Arguments @(
        'scripts\evaluate_object_sizes.py',
        '--stage', 'mffpn',
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
        '--name', 'scale_size_eval_mffpn_p2p4_control_50e',
        '--ground-truth-json', (Join-Path $MetricRoot 'visdrone_val_coco_ground_truth.json'),
        '--output-json', (Join-Path $MetricRoot 'mffpn_p2p4_control_size_val.json')
    )

    Invoke-Logged -Label 'report' -LogPath (Join-Path $LogRoot 'control_report_generation.log') -Arguments @(
        'scripts\generate_scale_ablation_report.py',
        '--report-dir', $ReportRoot
    )

    Update-Status -State 'completed' -Stage 'done' -Message 'Control training, evaluation, and fair comparison report completed'
    (Get-Date).ToString('o') | Set-Content -LiteralPath (Join-Path $ReportRoot 'control_completed.flag') -Encoding UTF8
    @(
        '# P2–P4 独立50轮对照实验',
        '',
        '- 状态：已完成',
        "- 完成时间：$((Get-Date).ToString('o'))",
        '- 训练：MF-FPN P2–P4，50 epochs，seed=0，VisDrone2019，imgsz=640',
        '- 训练结果：`../../runs/scale_mffpn_p2p4_control_50e/`',
        '- 统一结论：见 `README.md`',
        '- 结构化状态：见 `control_status.json`'
    ) | Set-Content -LiteralPath $ControlNotePath -Encoding UTF8
}
catch {
    Update-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $LogRoot 'control_pipeline_error.log') -Encoding UTF8
    throw
}
