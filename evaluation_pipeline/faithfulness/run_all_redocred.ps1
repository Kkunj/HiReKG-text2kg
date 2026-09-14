<#
.SYNOPSIS
  Run faithfulness evaluation for OURS, RAKG, KGGEN on the REDOCRED dataset
  back-to-back against the locally hosted Qwen3-14B judge.

.DESCRIPTION
  Each approach produces:
    results/<approach>_redocred_<experiment_folder>__local/dataset_summary.json

  Before each run, this script wipes that approach's verbalize_batch and
  judge_batch JSONL caches so a previous (broken) smoke run can't seed the
  new run with empty-content entries. The embedding cache for REDOCRED is
  kept — it depends only on source text + chunker config, so it's safe to
  reuse across approaches.

  Stops on first failure unless -ContinueOnError is specified.

.PARAMETER Workers
  Concurrent local-LLM worker threads per run. Default: 8.

.PARAMETER Limit
  Optional --limit value (smoke test).

.PARAMETER KeepCache
  Skip the per-approach verbalize/judge cache wipe. Use only when you know
  the caches are clean and you just want to resume.

.PARAMETER ContinueOnError
  Keep running remaining approaches even if one fails.

.EXAMPLE
  ./run_all_redocred.ps1
  ./run_all_redocred.ps1 -Workers 16
  ./run_all_redocred.ps1 -Limit 5
  ./run_all_redocred.ps1 -KeepCache
#>

[CmdletBinding()]
param(
    [int]$Workers = 8,
    [Nullable[int]]$Limit = $null,
    [switch]$KeepCache,
    [switch]$ContinueOnError
)

$ErrorActionPreference = if ($ContinueOnError) { 'Continue' } else { 'Stop' }

$scriptDir = $PSScriptRoot
$python    = 'python'
$runner    = Join-Path $scriptDir 'run_faithfulness.py'
$endpoint  = 'http://<LOCAL_LLM_HOST>:8003/v1/chat/completions'

$runs = @(
    @{ Approach = 'KGGEN'; Folder = 'kggen_redocred_qwen3_14b' },
    @{ Approach = 'RAKG';  Folder = 'rakg_redocred_qwen3_14b'  },
    @{ Approach = 'OURS';  Folder = 'our_redocred_qwen3_14b'   }
)

$summary = @()

foreach ($r in $runs) {
    $approach = $r.Approach
    $folder   = $r.Folder
    $runTag   = "$($approach.ToLower())_redocred_${folder}__local"
    $workdir  = Join-Path $scriptDir "batch_workdir\$runTag"

    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor Cyan
    Write-Host "  $approach  /  REDOCRED  /  $folder" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor Cyan

    if (-not $KeepCache) {
        foreach ($sub in 'verbalize_batch','judge_batch') {
            $stalePath = Join-Path $workdir $sub
            if (Test-Path $stalePath) {
                Write-Host "  wiping stale cache: $stalePath" -ForegroundColor DarkYellow
                Remove-Item -Recurse -Force $stalePath
            }
        }
    }

    $args = @(
        $runner,
        '--approach',          $approach,
        '--dataset',           'REDOCRED',
        '--experiment-folder', $folder,
        '--endpoint',          $endpoint,
        '--workers',           $Workers
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
