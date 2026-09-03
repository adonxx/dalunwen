<#
.SYNOPSIS
Waits for the overlap-tile experiment to finish and then schedules a safe shutdown.

.DESCRIPTION
Shutdown is scheduled only when the pipeline reports a completed state and both the
final Markdown report and its completion marker are present. A failed, incomplete,
or timed-out experiment intentionally leaves the computer on.
#>
[CmdletBinding()]
param(
    [int]$PollSeconds = 30,
    [int]$TimeoutMinutes = 120,
    [int]$ShutdownDelaySeconds = 120
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$reportDirectory = Join-Path $projectRoot 'reports\overlap_tile_training'
$statusPath = Join-Path $reportDirectory 'status.json'
$reportPath = Join-Path $reportDirectory 'REPORT.md'
$completionPath = Join-Path $reportDirectory 'completed.flag'
$watchdogStatePath = Join-Path $reportDirectory 'shutdown_watchdog_status.json'
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)

function Write-WatchdogState {
    param(
        [Parameter(Mandatory)][string]$State,
        [Parameter(Mandatory)][string]$Message
    )

    [pscustomobject]@{
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $watchdogStatePath -Encoding UTF8
}

Write-WatchdogState -State 'watching' -Message 'Waiting for overlap-tile experiment completion and final report.'

while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $statusPath) {
        try {
            $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        }
        catch {
            Write-WatchdogState -State 'watching' -Message 'Status file is being updated; retrying.'
            Start-Sleep -Seconds $PollSeconds
            continue
        }

        if ($status.state -eq 'completed') {
            if ((Test-Path -LiteralPath $reportPath) -and (Test-Path -LiteralPath $completionPath)) {
                $message = "Experiment completed and final report recorded. Shutdown begins in $ShutdownDelaySeconds seconds."
                Write-WatchdogState -State 'shutdown_scheduled' -Message $message
                & shutdown.exe /s /t $ShutdownDelaySeconds /c 'Overlap-tile experiment completed; results and report were saved.'
                exit 0
            }

            Write-WatchdogState -State 'incomplete' -Message 'Pipeline marked complete, but the final report or completion marker is missing. Shutdown was not scheduled.'
            exit 1
        }

        if ($status.state -eq 'failed') {
            Write-WatchdogState -State 'failed' -Message 'Experiment pipeline failed. Shutdown was not scheduled.'
            exit 1
        }
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-WatchdogState -State 'timeout' -Message 'Watchdog timeout reached before a verified completed result. Shutdown was not scheduled.'
exit 1
