# run_all_14b_faithfulness_32b.ps1
# Re-runs the faithfulness eval against all qwen3_14b extraction results
# (MINE / REDOCRED / SCIERC x KGGEN / RAKG / OURS) using the locally hosted
# Qwen3-32B-FP8 endpoints. Nine runs total.
#
# Strategy: launch all 9 in parallel, round-robin assigned to the 3 endpoints
# (3 jobs per endpoint). workers=3 per job so each endpoint sees ~9 concurrent
# requests — same per-endpoint load that's been working for the 8b runs.
#
# Endpoint assignment: each endpoint gets one of each approach (KGGEN/RAKG/OURS),
# spread across datasets, so no single endpoint is stuck with all the heavy runs.

$ErrorActionPreference = "Stop"

$ROOT             = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\faithfulness"
$SCRIPT           = Join-Path $ROOT "run_faithfulness.py"
$RESULTS_DIR      = Join-Path $ROOT "qwen_32b_results"
$SOURCE_RESULTS   = Join-Path $ROOT "results"

$MODEL            = "qwen3-32b"
$RUN_SUFFIX       = "qwen32b"
$WORKERS          = 3

$EP_8083 = "http://<LOCAL_LLM_HOST>:<PORT>/v1/chat/completions"
$EP_8084 = "http://<LOCAL_LLM_HOST>:8084/v1/chat/completions"
$EP_8085 = "http://<LOCAL_LLM_HOST>:8085/v1/chat/completions"

# NOTE: the OURS+MINE+14B folder is named `our_mine_qwen3_14b_batch`
# (everything else follows `<prefix>_<dataset>_qwen3_14b`).
$runs = @(
    # MINE
    @{ approach="KGGEN"; dataset="MINE";     folder="kggen_mine_qwen3_14b";       endpoint=$EP_8083 },
    @{ approach="RAKG";  dataset="MINE";     folder="rakg_mine_qwen3_14b";        endpoint=$EP_8084 },
    @{ approach="OURS";  dataset="MINE";     folder="our_mine_qwen3_14b_batch";   endpoint=$EP_8085 },
    # REDOCRED
    @{ approach="OURS";  dataset="REDOCRED"; folder="our_redocred_qwen3_14b";     endpoint=$EP_8083 },
    @{ approach="KGGEN"; dataset="REDOCRED"; folder="kggen_redocred_qwen3_14b";   endpoint=$EP_8084 },
    @{ approach="RAKG";  dataset="REDOCRED"; folder="rakg_redocred_qwen3_14b";    endpoint=$EP_8085 },
    # SCIERC
    @{ approach="RAKG";  dataset="SCIERC";   folder="rakg_scierc_qwen3_14b";      endpoint=$EP_8083 },
    @{ approach="OURS";  dataset="SCIERC";   folder="our_scierc_qwen3_14b";       endpoint=$EP_8084 },
    @{ approach="KGGEN"; dataset="SCIERC";   folder="kggen_scierc_qwen3_14b";     endpoint=$EP_8085 }
)

Write-Host "============================================================"
Write-Host "  qwen3_14b faithfulness re-eval with $MODEL"
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
Write-Host "All nine runs completed and moved." -ForegroundColor Green
