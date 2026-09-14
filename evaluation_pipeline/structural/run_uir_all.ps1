<#
.SYNOPSIS
    Runs UIR (Unique Information Ratio / Entity Redundancy) evaluation for
    kggen, rakg, and our_approach experiments.

.DESCRIPTION
    Executes run_uir_evaluation.py for each of the three approaches, storing
    results and logs in the structural evaluation directory.

    Outputs per approach:
        <approach>_uir_results.json   - per-doc + aggregate metrics
        <approach>_uir.log            - full execution log

    After all runs, prints a side-by-side comparison table.

.NOTES
    Author  : Kunj
    Created : 2026-05-15
#>

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RunnerPy    = Join-Path $ScriptDir "run_uir_evaluation.py"
$OutputDir   = $ScriptDir   # results stored in structural/

$Experiments = @(
    @{
        Name = "kggen"
        Dir  = "C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_mine_qwen3_14b"
    },
    @{
        Name = "rakg"
        Dir  = "C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_mine_qwen3_14b"
    },
    @{
        Name = "our_approach"
        Dir  = "C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_mine_qwen3_14b_batch"
    }
)

$Model     = "all-MiniLM-L6-v2"
$Threshold = 0.75

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  UIR (Entity Redundancy) Evaluation Pipeline" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "  Model    : $Model" -ForegroundColor Cyan
Write-Host "  Threshold: $Threshold" -ForegroundColor Cyan
Write-Host "  Output   : $OutputDir" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Run each approach
# ---------------------------------------------------------------------------
$OverallStart = Get-Date

foreach ($exp in $Experiments) {
    $name = $exp.Name
    $dir  = $exp.Dir

    Write-Host ""
    Write-Host ("-" * 70) -ForegroundColor Yellow
    Write-Host "  Running: $name" -ForegroundColor Yellow
    Write-Host "  Source : $dir" -ForegroundColor Yellow
    Write-Host ("-" * 70) -ForegroundColor Yellow
    Write-Host ""

    if (-not (Test-Path $dir)) {
        Write-Host "  [ERROR] Directory not found: $dir" -ForegroundColor Red
        continue
    }

    $startTime = Get-Date

    python $RunnerPy $dir $name $OutputDir --model $Model --threshold $Threshold

    $exitCode = $LASTEXITCODE
    $elapsed  = (Get-Date) - $startTime

    if ($exitCode -ne 0) {
        Write-Host "  [FAILED] $name exited with code $exitCode" -ForegroundColor Red
    } else {
        Write-Host ""
        Write-Host "  [DONE] $name completed in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    }
}

$totalElapsed = (Get-Date) - $OverallStart

# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  COMPARISON SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

$header = "{0,-18} {1,>8} {2,>12} {3,>12} {4,>12} {5,>10} {6,>10}" -f `
    "Approach", "UIR", "Mean Redun.", "Median", "Std Dev", "Red.Pairs", "Entities"
Write-Host $header -ForegroundColor White
Write-Host ("-" * 84) -ForegroundColor Gray

foreach ($exp in $Experiments) {
    $resultFile = Join-Path $OutputDir "$($exp.Name)_uir_results.json"
    if (-not (Test-Path $resultFile)) {
        Write-Host ("{0,-18} {1}" -f $exp.Name, "-- results file not found --") -ForegroundColor Red
        continue
    }

    $data = Get-Content $resultFile -Raw | ConvertFrom-Json
    $agg  = $data.aggregate

    $row = "{0,-18} {1,8:N4} {2,12:N4} {3,12:N4} {4,12:N4} {5,10} {6,10}" -f `
        $exp.Name,
        $agg.UIR_score,
        $agg.mean_redundancy,
        $agg.median_redundancy,
        $agg.std_redundancy,
        $agg.total_redundant_pairs,
        $agg.total_entities_evaluated

    Write-Host $row
}

Write-Host ""
Write-Host "Total wall time: $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "Results directory: $OutputDir" -ForegroundColor Cyan
Write-Host ""
