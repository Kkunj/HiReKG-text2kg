# run_scierc_faithfulness.ps1
# Runs faithfulness evaluation for three SCIERC experiment results
# against ground-truth atomic facts using the qwen3-14b endpoint.

$ErrorActionPreference = "Stop"

$ENDPOINT = "http://<LOCAL_LLM_HOST>:8082/v1/chat/completions"
$ATOMIC_FACTS_DIR = "C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\atomic_facts\stage3_final"
$SCRIPT = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\faithfulness\run_faithfulness.py"

$runs = @(
    @("KGGEN",  "SCIERC", "kggen_scierc_qwen3_14b"),
    @("RAKG",   "SCIERC", "rakg_scierc_qwen3_14b"),
    @("OURS",   "SCIERC", "our_scierc_qwen3_14b")
)

foreach ($run in $runs) {
    $approach = $run[0]
    $dataset  = $run[1]
    $folder   = $run[2]

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  Running: $approach / $dataset / $folder"
    Write-Host "============================================================"
    Write-Host ""

    python $SCRIPT `
        --approach $approach `
        --dataset $dataset `
        --experiment-folder $folder `
        --endpoint $ENDPOINT `
        --atomic-facts-dir $ATOMIC_FACTS_DIR

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $approach / $folder failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "  Completed: $approach / $folder" -ForegroundColor Green
    Write-Host ""
}

Write-Host ""
Write-Host "All three runs completed." -ForegroundColor Green