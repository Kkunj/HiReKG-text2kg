<#
.SYNOPSIS
  Run faithfulness evaluation for OURS, RAKG, KGGEN on MINE using
  qwen3-8b experiment results, judged by qwen3-14b.

.DESCRIPTION
  Evaluates KGs built by the qwen3-8b model against the MINE ground-truth
  source texts. The judge LLM is qwen3-14b hosted at the endpoint below.

  Experiment folders:
    KGGEN : baselines/kggen/experiments/kggen_mine_qwen3_8b
    RAKG  : baselines/rakg/experiments/rakg_mine_qwen3_8b
    OURS  : our_approach/experiments/our_mine_qwen3_8b

  Artifacts land at:
    results/<approach>_mine_<experiment_folder>__local/
    batch_workdir/<approach>_mine_<experiment_folder>__local/

.PARAMETER Workers
  Concurrent local-LLM worker threads per run. Default: 32.

.PARAMETER Limit
  Optional --limit value (smoke test).

.PARAMETER ContinueOnError
  Keep running remaining approaches even if one fails.

.EXAMPLE
  ./run_all_mine_qwen3_8b.ps1
  ./run_all_mine_qwen3_8b.ps1 -Workers 16
  ./run_all_mine_qwen3_8b.ps1 -Limit 5
#>

[CmdletBinding()]
param(
    [int]$Workers = 32,
    [Nullable[int]]$Limit = $null,
    [switch]$ContinueOnError
)

$ErrorActionPreference = if ($ContinueOnError) { 'Continue' } else { 'Stop' }

$scriptDir = $PSScriptRoot
$python    = 'python'
$runner    = Join-Path $scriptDir 'run_faithfulness.py'
$endpoint  = 'http://<LOCAL_LLM_HOST>:<PORT>/v1/chat/completions'

$runs = @(
    @{ Approach = 'OURS';  Folder = 'our_mine_qwen3_8b'   },
    @{ Approach = 'RAKG';  Folder = 'rakg_mine_qwen3_8b'  },
    @{ Approach = 'KGGEN'; Folder = 'kggen_mine_qwen3_8b' }
)

$summary = @()

foreach ($r in $runs) {
    $approach = $r.Approach
    $folder   = $r.Folder

    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "  $approach  /  MINE  /  $folder" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan

    $runArgs = @(
        $runner,
        '--approach',          $approach,
        '--dataset',           'MINE',
        '--experiment-folder', $folder,
        '--workers',           $Workers,
        '--endpoint',          $endpoint
    )
    if ($null -ne $Limit) { $runArgs += @('--limit', $Limit) }

    $start = Get-Date
    & $python @runArgs
    $exitCode = $LASTEXITCODE
    $elapsed  = (Get-Date) - $start

    $summary += [pscustomobject]@{
        Approach = $approach
        Folder   = $folder
        ExitCode = $exitCode
        Minutes  = [math]::Round($elapsed.TotalMinutes, 1)
    }

    if ($exitCode -ne 0) {
        Write-Host "  $approach FAILED (exit=$exitCode)" -ForegroundColor Red
        if (-not $ContinueOnError) { break }
    } else {
        Write-Host "  $approach done in $([math]::Round($elapsed.TotalMinutes, 1)) min" -ForegroundColor Green
    }
}

Write-Host ''
Write-Host ('=' * 70) -ForegroundColor Cyan
Write-Host '  Summary' -ForegroundColor Cyan
Write-Host ('=' * 70) -ForegroundColor Cyan
$summary | Format-Table -AutoSize
