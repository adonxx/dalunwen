$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$SourceMetrics = Join-Path $ProjectRoot 'reports\sahi_inference\metrics'
$ReportDir = Join-Path $ProjectRoot 'reports\sahi_scale_fusion'
$MetricsDir = Join-Path $ReportDir 'metrics'
$LogsDir = Join-Path $ReportDir 'logs'
$FiguresDir = Join-Path $ReportDir 'figures'
$StatusPath = Join-Path $ReportDir 'status.json'

New-Item -ItemType Directory -Force -Path $MetricsDir, $LogsDir, $FiguresDir | Out-Null

function Write-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    [ordered]@{
        updated_at = [DateTimeOffset]::Now.ToString('o')
        state = $State
        stage = $Stage
        message = $Message
        process_id = $PID
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding utf8
}

try {
    Write-Status -State 'running' -Stage 'fusion' -Message 'Running three pre-registered fusion strategies'
    & $Python (Join-Path $ProjectRoot 'scripts\fuse_sahi_predictions.py') `
        --whole-predictions (Join-Path $SourceMetrics 'whole_predictions.json') `
        --sahi-predictions (Join-Path $SourceMetrics 'sahi960_predictions.json') `
        --whole-record (Join-Path $SourceMetrics 'whole.json') `
        --sahi-record (Join-Path $SourceMetrics 'sahi960.json') `
        --ground-truth (Join-Path $SourceMetrics 'visdrone_val_coco_ground_truth.json') `
        --output-dir $MetricsDir `
        --merge-iou 0.5 `
        --max-det 300 *> (Join-Path $LogsDir 'fusion.log')
    if ($LASTEXITCODE -ne 0) { throw "fusion failed with exit code $LASTEXITCODE" }

    Write-Status -State 'running' -Stage 'report' -Message 'Generating fusion tables, figures, and Markdown report'
    & $Python (Join-Path $ProjectRoot 'scripts\generate_sahi_scale_fusion_report.py') *> (Join-Path $LogsDir 'report_generation.log')
    if ($LASTEXITCODE -ne 0) { throw "report generation failed with exit code $LASTEXITCODE" }

    New-Item -ItemType File -Force -Path (Join-Path $ReportDir 'completed.flag') | Out-Null
    Write-Status -State 'completed' -Stage 'done' -Message 'Scale-aware fusion ablation and report completed'
}
catch {
    Write-Status -State 'failed' -Stage 'error' -Message $_.Exception.Message
    throw
}
