# MINE evaluation — scierc dataset, gpt-4o KG generation, gpt-5 judge
# Evaluates RAKG predicted KGs against GT atomic facts.

$GT_DIR    = "C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\atomic_facts\stage3_final"
$SCRIPT    = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\mine_score\run_mine_evaluation.py"

# --- RAKG ---
Write-Host "`n========== RAKG (scierc, gpt-4o) ==========" -ForegroundColor Cyan
python $SCRIPT all `
    --gt-dir   $GT_DIR `
    --pred-dir "C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_scierc_gpt4o" `
    --run-tag  rakg_scierc_gpt4o `
    --backend  openai

Write-Host "`nEvaluation complete." -ForegroundColor Green
