# Run Gemini batch KG pipeline across all three datasets.
#
# Usage:
#   .\run_all_datasets.ps1              # submit all three
#   .\run_all_datasets.ps1 -DryRun      # dry-run only (no API calls)

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner    = Join-Path $ScriptDir "run_mine_batch.py"

# ── Dataset definitions ──────────────────────────────────────────────────
$datasets = @(
    @{
        Name     = "MINE_gemini_batch"
        TextsDir = "C:\<PROJECT_ROOT>\graph_rag\datasets\MINE\texts"
    },
    @{
        Name     = "scierc_gemini_batch"
        TextsDir = "C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\texts"
    },
    @{
        Name     = "redocred_gemini_batch"
        TextsDir = "C:\<PROJECT_ROOT>\graph_rag\datasets\windows_redocred\texts"
    }
)

# ── Run each dataset ─────────────────────────────────────────────────────
foreach ($ds in $datasets) {
    $textsDir = $ds.TextsDir
    $name     = $ds.Name

    Write-Host ""
    Write-Host ("=" * 60)
    Write-Host "  Dataset:    $name"
    Write-Host "  Texts dir:  $textsDir"
    Write-Host ("=" * 60)

    if (-not (Test-Path $textsDir)) {
        Write-Host "  SKIPPING — texts dir not found: $textsDir" -ForegroundColor Yellow
        continue
    }

    $args = @("$Runner", "--all", "--texts-dir", "$textsDir", "--name", "$name")
    if ($DryRun) {
        $args += "--dry-run"
    }

    python @args

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host "  Completed: $name" -ForegroundColor Green
}

Write-Host ""
Write-Host "All datasets processed." -ForegroundColor Green
