# run_scierc_faithfulness_32b.ps1
# Re-runs the SCIERC faithfulness evaluation (previously done with qwen3_8b
# and qwen3_14b) using the locally hosted Qwen3-32B-FP8 endpoints.
#
# Speedup vs the original sequential script:
#   - 3 vLLM instances on 8083/8084/8085, one per approach in parallel
#   - workers=8 per run (each endpoint sees 8 concurrent requests)
#
# Output: each run's results dir is moved into
#   evaluation_pipeline/faithfulness/qwen_32b_results/
# once the run completes.

$ErrorActionPreference = "Stop"

$ROOT             = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\faithfulness"
$SCRIPT           = Join-Path $ROOT "run_faithfulness.py"
$RESULTS_DIR      = Join-Path $ROOT "qwen_32b_results"
$SOURCE_RESULTS   = Join-Path $ROOT "results"

$MODEL            = "qwen3-32b"
$RUN_SUFFIX       = "qwen32b"
$WORKERS          = 8

# (approach, dataset, experiment_folder, endpoint)
# The experiment folders are the same KGs that were evaluated with qwen3_8b;
# we're just re-evaluating those KGs with the larger judge model.
$runs = @(
    @{ approach="KGGEN"; dataset="SCIERC"; folder="kggen_scierc_qwen3_8b"; endpoint="http://<LOCAL_LLM_HOST>:<PORT>/v1/chat/completions" },
    @{ approach="RAKG";  dataset="SCIERC"; folder="rakg_scierc_qwen3_8b";  endpoint="http://<LOCAL_LLM_HOST>:8084/v1/chat/completions" },
    @{ approach="OURS";  dataset="SCIERC"; folder="our_scierc_qwen3_8b";   endpoint="http://<LOCAL_LLM_HOST>:8085/v1/chat/completions" }
)

Write-Host "============================================================"
Write-Host "  SCIERC faithfulness re-eval with $MODEL"
Write-Host "  Launching $($runs.Count) parallel jobs..."
Write-Host "============================================================"

$jobs = @()
foreach ($r in $runs) {
    $jobName = "$($r.approach)_$($r.dataset)"
    Write-Host ("  [{0,-5}] -> endpoint={1}  folder={2}" -f $r.approach, $r.endpoint, $r.folder)

    $jobs += Start-Job -Name $jobName -ScriptBlock {
        param($Script, $Approach, $Dataset, $Folder, $Endpoint,
              $Model, $Workers, $RunSuffix)

        & python $Script `
            --approach $Approach `
            --dataset $Dataset `
            --experiment-folder $Folder `
            --endpoint $Endpoint `
            --verbalize-model $Model `
            --judge-model $Model `
            --workers $Workers `
            --run-tag-suffix $RunSuffix

        if ($LASTEXITCODE -ne 0) {
            throw "python exited with code $LASTEXITCODE"
        }
    } -ArgumentList $SCRIPT, $r.approach, $r.dataset, $r.folder, $r.endpoint,
                    $MODEL, $WORKERS, $RUN_SUFFIX
}

Write-Host ""
Write-Host "  Jobs launched. Streaming logs as they arrive..."
Write-Host "  (each line is prefixed with [JOB_NAME])"
Write-Host ""

# Stream output from each job as it produces lines
while ($jobs | Where-Object { $_.State -eq 'Running' }) {
    foreach ($j in $jobs) {
        $chunk = Receive-Job -Job $j -Keep:$false 2>&1
        if ($chunk) {
            foreach ($line in $chunk) {
                Write-Host "[$($j.Name)] $line"
            }
        }
    }
    Start-Sleep -Milliseconds 1500
}

# Drain any final output
foreach ($j in $jobs) {
    $chunk = Receive-Job -Job $j -Keep:$false 2>&1
    if ($chunk) {
        foreach ($line in $chunk) {
            Write-Host "[$($j.Name)] $line"
        }
    }
}

Write-Host ""
Write-Host "============================================================"
$anyFailed = $false
foreach ($j in $jobs) {
    if ($j.State -eq 'Completed') {
        Write-Host ("  {0,-15}  OK" -f $j.Name) -ForegroundColor Green
    } else {
        Write-Host ("  {0,-15}  FAILED ({1})" -f $j.Name, $j.State) -ForegroundColor Red
        $anyFailed = $true
    }
    Remove-Job -Job $j
}

if ($anyFailed) {
    Write-Host "One or more jobs failed; not moving partial results." -ForegroundColor Red
    exit 1
}

# Move each run's result folder into qwen_32b_results/
Write-Host ""
Write-Host "Moving result folders into $RESULTS_DIR ..." -ForegroundColor Cyan
foreach ($r in $runs) {
    $runTag = "$($r.approach.ToLower())_$($r.dataset.ToLower())_$($r.folder)__local__$RUN_SUFFIX"
    $src = Join-Path $SOURCE_RESULTS $runTag
    $dst = Join-Path $RESULTS_DIR $runTag
    if (-not (Test-Path $src)) {
        Write-Host "  WARNING: expected results dir not found: $src" -ForegroundColor Yellow
        continue
    }
    if (Test-Path $dst) {
        Remove-Item -Recurse -Force $dst
    }
    Move-Item -Path $src -Destination $dst
    Write-Host "  $runTag  -> $dst" -ForegroundColor Green
}

Write-Host ""
Write-Host "All three runs completed and moved." -ForegroundColor Green
