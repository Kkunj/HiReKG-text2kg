# run_mine_redocred_faithfulness_32b.ps1
# Re-runs the MINE + REDOCRED faithfulness evaluations using the locally hosted
# Qwen3-32B-FP8 endpoints. Six runs total (KGGEN/RAKG/OURS x MINE/REDOCRED).
#
# Strategy: launch all 6 in parallel, round-robin assigned to the 3 endpoints
# (2 jobs per endpoint). workers=4 per job so each endpoint still sees ~8
# concurrent requests, matching the SCIERC throughput pattern.

$ErrorActionPreference = "Stop"

$ROOT             = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\faithfulness"
$SCRIPT           = Join-Path $ROOT "run_faithfulness.py"
$RESULTS_DIR      = Join-Path $ROOT "qwen_32b_results"
$SOURCE_RESULTS   = Join-Path $ROOT "results"

$MODEL            = "qwen3-32b"
$RUN_SUFFIX       = "qwen32b"
$WORKERS          = 4

$EP_8083 = "http://<LOCAL_LLM_HOST>:<PORT>/v1/chat/completions"
$EP_8084 = "http://<LOCAL_LLM_HOST>:8084/v1/chat/completions"
$EP_8085 = "http://<LOCAL_LLM_HOST>:8085/v1/chat/completions"

# Endpoint pairing chosen to spread the (likely heavier) RAKG runs across all
# three endpoints rather than piling both onto one. Each endpoint gets one
# MINE job and one REDOCRED job from different approaches.
$runs = @(
    @{ approach="KGGEN"; dataset="MINE";     folder="kggen_mine_qwen3_8b";     endpoint=$EP_8083 },
    @{ approach="RAKG";  dataset="MINE";     folder="rakg_mine_qwen3_8b";      endpoint=$EP_8084 },
    @{ approach="OURS";  dataset="MINE";     folder="our_mine_qwen3_8b";       endpoint=$EP_8085 },
    @{ approach="OURS";  dataset="REDOCRED"; folder="our_redocred_qwen3_8b";   endpoint=$EP_8083 },
    @{ approach="KGGEN"; dataset="REDOCRED"; folder="kggen_redocred_qwen3_8b"; endpoint=$EP_8084 },
    @{ approach="RAKG";  dataset="REDOCRED"; folder="rakg_redocred_qwen3_8b";  endpoint=$EP_8085 }
)

Write-Host "============================================================"
Write-Host "  MINE + REDOCRED faithfulness re-eval with $MODEL"
Write-Host "  Launching $($runs.Count) parallel jobs across 3 endpoints..."
Write-Host "============================================================"

$jobs = @()
foreach ($r in $runs) {
    $jobName = "$($r.approach)_$($r.dataset)"
    Write-Host ("  [{0,-16}] -> endpoint={1}  folder={2}" -f $jobName, $r.endpoint, $r.folder)

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
        Write-Host ("  {0,-16}  OK" -f $j.Name) -ForegroundColor Green
    } else {
        Write-Host ("  {0,-16}  FAILED ({1})" -f $j.Name, $j.State) -ForegroundColor Red
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
Write-Host "All six runs completed and moved." -ForegroundColor Green
