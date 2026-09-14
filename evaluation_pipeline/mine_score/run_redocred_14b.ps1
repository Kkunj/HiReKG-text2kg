# MINE evaluation — redocred dataset, qwen3-14b KG generation
# Evaluates KGGen, RAKG, and OURS predicted KGs against GT atomic facts.

$GT_DIR    = "C:\<PROJECT_ROOT>\graph_rag\datasets\windows_redocred\atomic_facts\stage3_final"
$SCRIPT    = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\mine_score\run_mine_evaluation.py"
$LLM_URL   = "http://<LOCAL_LLM_HOST>:8084/v1/chat/completions"

# --- KGGen ---
Write-Host "`n========== KGGen (redocred, qwen3-14b) ==========" -ForegroundColor Cyan
python $SCRIPT all `
    --gt-dir   $GT_DIR `
    --pred-dir "C:\<PROJECT_ROOT>\graph_rag\baselines\kggen\experiments\kggen_redocred_qwen3_14b" `
    --run-tag  kggen_redocred_qwen3_14b `
    --backend  local `
    --local-model  qwen3-14b `
    --local-llm-url $LLM_URL `
    --local-workers 32

# --- RAKG ---
Write-Host "`n========== RAKG (redocred, qwen3-14b) ==========" -ForegroundColor Cyan
python $SCRIPT all `
    --gt-dir   $GT_DIR `
    --pred-dir "C:\<PROJECT_ROOT>\graph_rag\baselines\rakg\experiments\rakg_redocred_qwen3_14b" `
    --run-tag  rakg_redocred_qwen3_14b `
    --backend  local `
    --local-model  qwen3-14b `
    --local-llm-url $LLM_URL `
    --local-workers 32

# --- OURS ---
Write-Host "`n========== OURS (redocred, qwen3-14b) ==========" -ForegroundColor Cyan
python $SCRIPT all `
    --gt-dir   $GT_DIR `
    --pred-dir "C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\our_redocred_qwen3_14b" `
    --run-tag  ours_redocred_qwen3_14b `
    --backend  local `
    --local-model  qwen3-14b `
    --local-llm-url $LLM_URL `
    --local-workers 32

Write-Host "`nAll 3 evaluations complete." -ForegroundColor Green
