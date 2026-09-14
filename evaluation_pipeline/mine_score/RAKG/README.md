# RAKG — MINE benchmark evaluation (batched)

End-to-end evaluation of the KGs produced by the **batched RAKG** pipeline
(`baselines/rakg/batch/`) against the public **MINE** benchmark
ground-truth answers, using **OpenAI Batch API** with **gpt-5** as the LLM
judge.

This folder is a near-clone of [`OURS/`](../OURS/README.md). The only
differences are paths (RAKG's experiment root and a separate results dir)
and labels — same retrieval stack, same judge prompt, same batch
infrastructure, so RAKG's score is directly comparable to OURS.

The original evaluation scripts next to it (`_1_evaluation.py`,
`_2_compare_results.py`, `_3_visualize.py`, `_4_analysis.py`) are **not
modified** — this runner writes results in the exact format those scripts
consume.

---

## Directory layout

```
graph_rag/
├── datasets/MINE/texts/                                      100 MINE essay txts
├── baselines/rakg/
│   └── experiments/RAKG_MINE_batch_results/                  RAKG KGs (per-doc folders)
│       ├── doc_0/final_output.json                            triples + entities for doc_0
│       ├── doc_1/...
│       └── doc_99/...
└── evaluation_pipeline/mine_score/
    ├── _1_evaluation.py           [original kg-gen evaluation — untouched]
    ├── _2_compare_results.py      [original aggregation/plotting — untouched]
    ├── _3_visualize.py            [original Streamlit dashboard — untouched]
    ├── _4_analysis.py             [original deeper stats — untouched]
    ├── data/
    │   ├── essays_filtered.json             100 essays that match our KGs
    │   ├── answers_filtered.json            100 answer-groups, 1:1 with essays_filtered
    │   └── filtered_index_to_mine_doc.json  positional map: row i → doc_N
    ├── OURS/                                 our_approach evaluator
    ├── RAKG/                                ← THIS FOLDER
    │   ├── README.md                        (you are here)
    │   ├── kg_adapter.py                    triples → kg-gen {entities,edges,relations}
    │   ├── run_rakg_evaluation.py           orchestrator (retrieve + batch + assemble)
    │   └── batch_workdir/                   batch state + retrieval cache (created at runtime)
    │       ├── retrieved.json
    │       ├── run_<timestamp>.log
    │       └── judge_batch/
    │           ├── input.jsonl
    │           ├── meta.json
    │           ├── output.jsonl
    │           └── errors.jsonl
    └── results/RAKG_MINE_batch_results/   (created at runtime — outputs go here)
        ├── results_0.json   ...  results_99.json
        └── summary_rakg.json
```

Important absolute paths (Windows, this repo):

| Thing | Path |
|---|---|
| RAKG's KGs (one folder per doc) | `<PROJECT_ROOT>/graph_rag/baselines/rakg/experiments/RAKG_MINE_batch_results/` |
| Filtered GT essays | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/data/essays_filtered.json` |
| Filtered GT answers | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/data/answers_filtered.json` |
| Row-index ↔ doc map | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/data/filtered_index_to_mine_doc.json` |
| RAKG evaluator | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/RAKG/run_rakg_evaluation.py` |
| Per-run state / batch caches | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/RAKG/batch_workdir/` |
| Final results | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/results/RAKG_MINE_batch_results/` |

---

## What MINE measures (in one paragraph)

For each essay, the benchmark stores a list of **factual statements** that
the essay supports. Evaluation asks: given a KG built from the essay and a
retrieval stack, can the KG surface enough context for an LLM judge to
verify each statement?

- Retrieve = kg-gen's `retrieve(query, node_embeddings, graph)` with
  MiniLM node embeddings — pulls top-k similar nodes and stringifies the
  surrounding subgraph into "context" text.
- Judge = an LLM reads `(context, correct_answer)` and returns `1` if the
  context supports the answer, else `0`.
- Per-essay **accuracy = correct / total queries**. The benchmark then
  averages across essays.

This is KG-format-agnostic. Any triple store that can answer
`retrieve(query)` can be scored. RAKG's KGs plug straight in via
`kg_adapter.py` because the batched RAKG pipeline writes the same
`triples_final` schema as `our_approach`.

---

## Why this folder exists separately from OURS/

OURS scores the KGs from `our_approach/experiments/MINE_batch_results/`.
RAKG scores the KGs from `baselines/rakg/experiments/RAKG_MINE_batch_results/`.
Same MINE benchmark, same judge prompt, same retrieval stack — the only
thing that differs is *which set of KGs* gets scored. Keeping them in
separate sibling folders means:

1. The two evaluators don't fight over `batch_workdir/` state.
2. `_2_compare_results.py` automatically picks up both `OURS_*` and
   `RAKG_*` folders under `results/` and produces side-by-side numbers.
3. Re-running the RAKG eval doesn't touch any OURS artefacts.

---

## Runner architecture

`run_rakg_evaluation.py` has three stages. Each is resumable: re-running
skips work whose output is already on disk. Implementation is identical
to OURS — see `../OURS/README.md` for the deep-dive on stage internals.

| Stage | What it does | Time |
|---|---|---|
| 1. `retrieve` | Per doc: load RAKG KG → kg-gen graph → MiniLM embeddings → `retrieve(query)` for each GT answer. Outputs `batch_workdir/retrieved.json`. | ~1–2 min for 100 docs |
| 2. `submit` | Build one batch request per (essay, query) pair (~1500 total). Submit to `/v1/responses` with `model=gpt-5`, `reasoning_effort=high`, `text.format=json_object`. Poll, download. | minutes to hours (OpenAI batch queue) |
| 3. `assemble` | Parse output, sync-retry any failures, write `results_<i>.json` per essay + `summary_rakg.json`. | seconds |

---

## Running it

The runner lives inside the `RAKG/` sub-package. Run from the
`mine_score/` directory so Python can resolve the package:

```bash
cd <PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score

# activate kgvenv first (the one that has kg-gen installed)

# smoke test — 2 docs end-to-end
python -m RAKG.run_rakg_evaluation all --limit 2

# full run, one stage at a time (easier to diagnose if something goes wrong)
python -m RAKG.run_rakg_evaluation retrieve
python -m RAKG.run_rakg_evaluation submit
python -m RAKG.run_rakg_evaluation assemble

# or all in one go
python -m RAKG.run_rakg_evaluation all
```

Useful flags:

| Flag | Effect |
|---|---|
| `--limit N` | Only the first N docs (fast smoke test). |
| `--force-retrieve` | Delete `retrieved.json` before Stage 1 (reruns retrieval). |
| `--retrieve-workers N` | Thread pool for Stage 1 (default 4). |

To force a fresh judge batch, delete `batch_workdir/judge_batch/` — next
`submit` will re-upload and re-submit.

### API key

The runner reads `OPENAI_API_KEY_2` first and promotes it to
`OPENAI_API_KEY`, matching the KG-generation run.

---

## Output — what you get

```
results/RAKG_MINE_batch_results/
├── results_0.json          ... results_99.json
│   └── each: per-query records + trailing {"accuracy": "..%"}
└── summary_rakg.json
    {
      "total_essays": 100,
      "total_queries": ~1500,
      "correct": ...,
      "overall_accuracy_pct": ...,
      "sync_retries": N,
      "sync_retry_failures": M
    }
```

To compare against OURS and the other baselines:

```bash
python _2_compare_results.py        # writes results/results.png + summary.txt
streamlit run _3_visualize.py       # interactive per-query browser
```

`_2_compare_results.py` discovers every folder under `results/`
automatically, so once both `OURS_MINE_batch_results/` and
`RAKG_MINE_batch_results/` exist, you'll get them on the same chart.

---

## Cost and time expectations

- **Stage 1 (retrieve)**: CPU-only, ~1–2 min for 100 docs.
- **Stage 2 (judge batch)**: ~1500 gpt-5 requests at `reasoning_effort=high`
  through the Batch API (50% discount). Runtime depends on OpenAI's batch
  queue; MINE batch SLA is 24 h, but typical completion has been
  5–30 min per batch.
- **Stage 3 (assemble)**: seconds, plus sync retries for any failed
  requests (rare).

---

## Troubleshooting

- **`FileNotFoundError: ... final_output.json`** — a doc's KG never got
  generated. Re-run the RAKG batch pipeline for that doc or remove it from
  `filtered_index_to_mine_doc.json`.
- **Batch status stuck at `validating`** — the batch input failed parsing.
  Inspect `batch_workdir/judge_batch/meta.json` and OpenAI's batch dashboard.
- **Many sync retries in Stage 3** — either the gpt-5 response came back
  wrapped in unexpected markdown, or quotas were hit. Inspect
  `errors.jsonl` in the judge job folder.
- **`ModuleNotFoundError: kg_gen`** — activate `kgvenv` (or whichever env
  has `kg-gen==0.4.0` installed).
