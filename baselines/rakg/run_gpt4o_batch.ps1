# Run RAKG pipeline with GPT-4o on SCIERC and ReDocRED datasets
# Uses the OpenAI Batch API (baselines/rakg/batch/) for lower cost and separate rate limits
# Usage: .\run_gpt4o_batch.ps1

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$BATCH_RUNNER = Join-Path $SCRIPT_DIR "batch\run_mine_batch.py"

# ── SCIERC ──────────────────────────────────────────────────────────────
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Running RAKG GPT-4o on SCIERC (Batch API)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

python $BATCH_RUNNER `
    --all `
    --texts-dir "C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\texts" `
    --experiment-name "rakg_scierc_gpt4o" `
    --model "gpt-4o"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nSCIERC batch failed with exit code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "`nSCIERC batch completed successfully." -ForegroundColor Green
}

# ── ReDocRED ────────────────────────────────────────────────────────────
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Running RAKG GPT-4o on ReDocRED (Batch API)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

python $BATCH_RUNNER `
    --all `
    --texts-dir "C:\<PROJECT_ROOT>\graph_rag\datasets\windows_redocred\texts" `
    --experiment-name "rakg_redocred_gpt4o" `
    --model "gpt-4o"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nReDocRED batch failed with exit code $LASTEXITCODE" -ForegroundColor Red
} else {
    Write-Host "`nReDocRED batch completed successfully." -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  All batches finished." -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green
