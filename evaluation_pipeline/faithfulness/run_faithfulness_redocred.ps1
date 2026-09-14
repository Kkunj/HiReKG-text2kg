# Run faithfulness evaluation for all three approaches on RedocRED dataset
# Endpoint: qwen3-14b on port 8003

$scriptDir = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\faithfulness"
$endpoint = "http://<LOCAL_LLM_HOST>:8003/v1/chat/completions"

Write-Host "=== Running Faithfulness Evaluation for RedocRED ===" -ForegroundColor Cyan
Write-Host ""

# 1. KGGEN
Write-Host "[1/3] KGGEN - kggen_redocred_qwen3_8b" -ForegroundColor Yellow
python "$scriptDir\run_faithfulness.py" `
    --approach KGGEN `
    --dataset REDOCRED `
    --experiment-folder kggen_redocred_qwen3_8b `
    --endpoint $endpoint
Write-Host ""

# 2. RAKG
Write-Host "[2/3] RAKG - rakg_redocred_qwen3_8b" -ForegroundColor Yellow
python "$scriptDir\run_faithfulness.py" `
    --approach RAKG `
    --dataset REDOCRED `
    --experiment-folder rakg_redocred_qwen3_8b `
    --endpoint $endpoint
Write-Host ""

# 3. OURS
Write-Host "[3/3] OURS - our_redocred_qwen3_8b" -ForegroundColor Yellow
python "$scriptDir\run_faithfulness.py" `
    --approach OURS `
    --dataset REDOCRED `
    --experiment-folder our_redocred_qwen3_8b `
    --endpoint $endpoint
Write-Host ""

Write-Host "=== All three faithfulness runs complete ===" -ForegroundColor Green
