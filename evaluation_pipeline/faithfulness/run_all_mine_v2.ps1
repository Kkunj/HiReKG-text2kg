<#
.SYNOPSIS
  Re-run faithfulness evaluation for OURS, RAKG, KGGEN on MINE under a
  v2 tag, after fixing the verbalize-stage token-budget bug.

.DESCRIPTION
  Existing __local results are preserved. New artifacts land at:
    results/<approach>_mine_<experiment_folder>__local__v2/
    batch_workdir/<approach>_mine_<experiment_folder>__local__v2/

  This is achieved by passing --run-tag-suffix v2 to the python runner,
  so the run_tag is "<approach>_mine_<folder>__local__v2" and every cache
  path is fresh (no risk of seeding from the broken __local run).

  The SCIERC embedding cache is shared across approaches by dataset, so
  the second and third runs reuse Stage-2 work for free. Same for MINE.

.PARAMETER Workers
  Concurrent local-LLM worker threads per run. Default: 8.

.PARAMETER Limit
  Optional --limit value (smoke test).

.PARAMETER ContinueOnError
  Keep running remaining approaches even if one fails.

.EXAMPLE
  ./run_all_mine_v2.ps1
  ./run_all_mine_v2.ps1 -Workers 16
  ./run_all_mine_v2.ps1 -Limit 5
#>

[CmdletBinding()]
param(
    [int]$Workers = 8,
    [Nullable[int]]$Limit = $null,
    [switch]$ContinueOnError
)

$ErrorActionPreference = if ($ContinueOnError) { 'Continue' } else { 'Stop' }

$scriptDir = $PSScriptRoot
$python    = 'python'
$runner    = Join-Path $scriptDir 'run_faithfulness.py'

$runs = @(
    @{ Approach = 'OURS';  Folder = 'our_mine_qwen3_14b_batch' },
    @{ Approach = 'RAKG';  Folder = 'rakg_mine_qwen3_14b'      },
    @{ Approach = 'KGGEN'; Folder = 'kggen_mine_qwen3_14b'     }
)

$summary = @()

foreach ($r in $runs) {
    $approach = $r.Approach
    $folder   = $r.Folder

    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "  $approach  /  MINE  /  $folder  (v2)" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan

    $args = @(
        $runner,
        '--approach',          $approach,
        '--dataset',           'MINE',
        '--experiment-folder', $folder,
        '--workers',           $Workers,
        '--run-tag-suffix',    'v2'
    )
    if ($null -ne $Limit) { $args += @('--limit', $Limit) }

    $start = Get-Date
    & $python @args
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
