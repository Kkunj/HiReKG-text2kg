# Run "our approach" batched KG pipeline for SCIERC and ReDocRED using gpt-4o.
# Both datasets run in PARALLEL as background jobs.
# Uses OpenAI Batch API via batch_pipeline.py; OPENAI_API_KEY_2 is swapped
# internally by run_mine_batch.py.

$ErrorActionPreference = "Stop"

$GRAPH_RAG = "C:\<PROJECT_ROOT>\graph_rag"
$RUNNER    = "$GRAPH_RAG\our_approach\batch\run_mine_batch.py"

Write-Host "Starting SCIERC and ReDocRED in parallel..." -ForegroundColor Cyan

# --- SCIERC (100 docs) ---
$scierc = Start-Job -Name "ours_scierc" -ScriptBlock {
    python $using:RUNNER `
        --all `
        --texts-dir "$using:GRAPH_RAG\datasets\scierc\texts" `
        --experiment-name "ours_scierc_gpt4o" `
        --model "gpt-4o" 2>&1
}

# --- ReDocRED (15 docs) ---
$redocred = Start-Job -Name "ours_redocred" -ScriptBlock {
    python $using:RUNNER `
        --all `
        --texts-dir "$using:GRAPH_RAG\datasets\windows_redocred\texts" `
        --experiment-name "ours_redocred_gpt4o" `
        --model "gpt-4o" 2>&1
}

Write-Host "Jobs launched: $($scierc.Id) (SCIERC), $($redocred.Id) (ReDocRED)" -ForegroundColor Yellow
Write-Host "Waiting for both to finish..." -ForegroundColor Yellow

# Wait and stream output
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

Write-Host "`nAll runs complete." -ForegroundColor Green
