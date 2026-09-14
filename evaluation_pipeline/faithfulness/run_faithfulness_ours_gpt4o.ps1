# Faithfulness evaluation for OURS (gpt-4o) over SCIERC and ReDocRED
# Judge uses OpenAI Batch API (gpt-5)
# Both datasets run in PARALLEL as background jobs.

$ErrorActionPreference = "Stop"
$SCRIPT = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\faithfulness\run_faithfulness.py"

Write-Host "Starting OURS faithfulness evaluations in parallel..." -ForegroundColor Cyan

# --- OURS + SCIERC ---
$scierc = Start-Job -Name "faith_ours_scierc" -ScriptBlock {
    python $using:SCRIPT `
        --approach OURS `
        --dataset SCIERC `
        --experiment-folder ours_scierc_gpt4o `
        --backend openai 2>&1
}

# --- OURS + ReDocRED ---
$redocred = Start-Job -Name "faith_ours_redocred" -ScriptBlock {
    python $using:SCRIPT `
        --approach OURS `
        --dataset REDOCRED `
        --experiment-folder ours_redocred_gpt4o `
        --backend openai 2>&1
}

Write-Host "Jobs launched: $($scierc.Id) (SCIERC), $($redocred.Id) (ReDocRED)" -ForegroundColor Yellow
Write-Host "Waiting for both to finish..." -ForegroundColor Yellow

$scierc, $redocred | Wait-Job | ForEach-Object {
    Write-Host "`n========== $($_.Name) ==========" -ForegroundColor Cyan
    Receive-Job $_
    if ($_.State -eq 'Failed') {
        Write-Host "JOB $($_.Name) FAILED" -ForegroundColor Red
    } else {
        Write-Host "JOB $($_.Name) COMPLETED" -ForegroundColor Green
    }
    Remove-Job $_
}

Write-Host "`nAll faithfulness evaluations complete." -ForegroundColor Green
