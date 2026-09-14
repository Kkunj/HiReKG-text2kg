# MINE evaluation — scierc dataset, gpt-4o KG generation, gpt-5 judge
# Evaluates our approach predicted KGs against GT atomic facts.

$GT_DIR    = "C:\<PROJECT_ROOT>\graph_rag\datasets\scierc\atomic_facts\stage3_final"
$SCRIPT    = "C:\<PROJECT_ROOT>\graph_rag\evaluation_pipeline\mine_score\run_mine_evaluation.py"

Write-Host "`n========== OURS (scierc, gpt-4o) ==========" -ForegroundColor Cyan
python $SCRIPT all `
    --gt-dir   $GT_DIR `
    --pred-dir "C:\<PROJECT_ROOT>\graph_rag\our_approach\experiments\ours_scierc_gpt4o" `
    --run-tag  ours_scierc_gpt4o `
    --backend  openai

Write-Host "`nEvaluation complete." -ForegroundColor Green
