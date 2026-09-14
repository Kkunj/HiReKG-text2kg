# Run faithfulness evaluation for SCIERC dataset across 3 approaches (qwen3_8b experiments)
# Judge: local qwen3-14b at http://<LOCAL_LLM_HOST>:8004/v1/chat/completions

$ErrorActionPreference = "Stop"
$SCRIPT = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\faithfulness\run_faithfulness.py"
$ENDPOINT = "http://<LOCAL_LLM_HOST>:8004/v1/chat/completions"

Write-Host "=== [1/3] KGGEN - SCIERC ===" -ForegroundColor Cyan
python $SCRIPT `
    --approach KGGEN `
    --dataset SCIERC `
    --experiment-folder kggen_scierc_qwen3_8b `
    --endpoint $ENDPOINT

Write-Host "=== [2/3] RAKG - SCIERC ===" -ForegroundColor Cyan
python $SCRIPT `
    --approach RAKG `
    --dataset SCIERC `
    --experiment-folder rakg_scierc_qwen3_8b `
    --endpoint $ENDPOINT

Write-Host "=== [3/3] OURS - SCIERC ===" -ForegroundColor Cyan
python $SCRIPT `
    --approach OURS `
    --dataset SCIERC `
    --experiment-folder our_scierc_qwen3_8b `
    --endpoint $ENDPOINT

Write-Host "=== ALL DONE ===" -ForegroundColor Green
