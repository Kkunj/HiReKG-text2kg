# Faithfulness evaluation for KGGen and RAKG (gpt-4o) over SCIERC and ReDocRED
# Judge uses OpenAI Batch API (gpt-5)
# Usage: .\run_faithfulness_gpt4o.ps1

$SCRIPT = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\faithfulness\run_faithfulness.py"

# --- KGGen + SCIERC ---
Write-Host "`n========== KGGen faithfulness (scierc, gpt-4o) ==========" -ForegroundColor Cyan
python $SCRIPT `
    --approach KGGEN `
    --dataset SCIERC `
    --experiment-folder kggen_scierc_gpt4o `
    --backend openai

# --- KGGen + ReDocRED ---
Write-Host "`n========== KGGen faithfulness (redocred, gpt-4o) ==========" -ForegroundColor Cyan
python $SCRIPT `
    --approach KGGEN `
    --dataset REDOCRED `
    --experiment-folder kggen_redocred_gpt4o `
    --backend openai

# --- RAKG + SCIERC ---
Write-Host "`n========== RAKG faithfulness (scierc, gpt-4o) ==========" -ForegroundColor Cyan
python $SCRIPT `
    --approach RAKG `
    --dataset SCIERC `
    --experiment-folder rakg_scierc_gpt4o `
    --backend openai

# --- RAKG + ReDocRED ---
Write-Host "`n========== RAKG faithfulness (redocred, gpt-4o) ==========" -ForegroundColor Cyan
python $SCRIPT `
    --approach RAKG `
    --dataset REDOCRED `
    --experiment-folder rakg_redocred_gpt4o `
    --backend openai

Write-Host "`nAll 4 faithfulness evaluations complete." -ForegroundColor Green
