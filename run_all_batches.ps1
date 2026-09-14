<#
.SYNOPSIS
    Run all three KG pipelines (our_approach, rakg, kggen) sequentially against
    one vLLM endpoint. Designed for overnight runs — start it and walk away.

.DESCRIPTION
    Each pipeline's run_mine_batch.py is independently resumable: rerunning
    skips docs whose final_output.json already exists. So if any single batch
    crashes mid-run, the other two still execute, and rerunning this script
    later picks up where every batch left off.

    Order is fastest-first so failures in the slow ones don't waste good runs:
        1. our_approach  (CHUNK_PARALLELISM=6)
        2. rakg          (MAX_PARALLEL=16)
        3. kggen         (sequential, smallest per-doc work on short docs)

.PARAMETER Endpoint
    vLLM /v1/chat/completions URL. Defaults to the local qwen3-14b endpoint.

.EXAMPLE
    .\run_all_batches.ps1
    .\run_all_batches.ps1 -Endpoint http://<LOCAL_LLM_HOST>:8000/v1/chat/completions
#>

param(
    [string]$Endpoint = "http://<LOCAL_LLM_HOST>:8000/v1/chat/completions"
)

$ErrorActionPreference = "Continue"
$RepoRoot = $PSScriptRoot
$LogFile  = Join-Path $RepoRoot "run_all_batches.log"

$Batches = @(
    @{ Name = "our_approach"; Dir = "our_approach" },
    @{ Name = "rakg";         Dir = "baselines\rakg" },
    @{ Name = "kggen";        Dir = "baselines\kggen" }
)

function Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts | $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "============================================================"
Log "Starting orchestrated batch run"
Log "Endpoint: $Endpoint"
Log "Repo:     $RepoRoot"
Log "Logfile:  $LogFile"
Log "============================================================"

$overallStart = Get-Date
$summary = @()

foreach ($b in $Batches) {
    $name = $b.Name
    $dir  = Join-Path $RepoRoot $b.Dir

    Log ""
    Log "------------------------------------------------------------"
    Log "[$name] starting in $dir"
    Log "------------------------------------------------------------"

    $batchStart = Get-Date
    $exitCode   = $null

    try {
        Push-Location $dir
        # Stream child output to console AND tee into our log file.
        & python run_mine_batch.py --endpoint $Endpoint 2>&1 | Tee-Object -FilePath $LogFile -Append
        $exitCode = $LASTEXITCODE
    } catch {
        Log "[$name] EXCEPTION: $_"
        $exitCode = -1
    } finally {
        Pop-Location
    }

    $batchElapsed = (Get-Date) - $batchStart
    $status = if ($exitCode -eq 0) { "OK" } else { "EXIT=$exitCode" }
    Log "[$name] $status  duration=$($batchElapsed.ToString('hh\:mm\:ss'))"
    $summary += [PSCustomObject]@{
        Batch    = $name
        Status   = $status
        Duration = $batchElapsed.ToString("hh\:mm\:ss")
    }
}

$overallElapsed = (Get-Date) - $overallStart
Log ""
Log "============================================================"
Log "ALL BATCHES COMPLETE  total=$($overallElapsed.ToString('hh\:mm\:ss'))"
Log "============================================================"
$summary | Format-Table -AutoSize | Out-String | ForEach-Object { Log $_ }
