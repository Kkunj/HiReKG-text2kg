# OURS — MINE benchmark evaluation (batched)

End-to-end evaluation of the KGs produced by our pipeline
(`our_approach/`) against the public **MINE** benchmark ground-truth answers,
using **OpenAI Batch API** with **gpt-5** as the LLM judge.

This folder is self-contained. The original evaluation scripts next to it
(`_1_evaluation.py`, `_2_compare_results.py`, `_3_visualize.py`,
`_4_analysis.py`) are **not modified** — our runner writes results in the
exact format those scripts consume, so you can run them on our outputs
without changes.

---

## Directory layout

```
graph_rag/
├── datasets/MINE/texts/                                       100 MINE essay txts
├── our_approach/
│   └── experiments/MINE_batch_results/                        our KGs (per-doc folders)
│       ├── doc_0/final_output.json                             triples + entities for doc_0
│       ├── doc_1/...
│       └── doc_99/...
└── evaluation_pipeline/mine_score/
    ├── _1_evaluation.py           [original kg-gen evaluation — untouched]
    ├── _2_compare_results.py      [original aggregation/plotting — untouched]
    ├── _3_visualize.py            [original Streamlit dashboard — untouched]
    ├── _4_analysis.py             [original deeper stats — untouched]
    ├── data/
    │   ├── kg_gen_eval_essays.json          105 original essays (as downloaded)
    │   ├── kg_gen_eval.json                 105 original answer-groups
    │   ├── essays_filtered.json             100 essays that match our KGs
    │   ├── answers_filtered.json            100 answer-groups, 1:1 with essays_filtered
    │   ├── filtered_index_to_mine_doc.json  positional map: row i → doc_N
    │   └── data_download.py                 (your HF download helper)
    ├── OURS/                              ← THIS FOLDER
    │   ├── README.md                        (you are here)
    │   ├── kg_adapter.py                    triples → kg-gen {entities,edges,relations}
    │   ├── run_ours_evaluation.py           orchestrator (retrieve + batch + assemble)
    │   └── batch_workdir/                   batch state + retrieval cache (created at runtime)
    │       ├── retrieved.json
    │       ├── run_<timestamp>.log
    │       └── judge_batch/
    │           ├── input.jsonl
    │           ├── meta.json
    │           ├── output.jsonl
    │           └── errors.jsonl
    └── results/OURS_MINE_batch_results/   (created at runtime — outputs go here)
        ├── results_0.json   ...  results_99.json
        └── summary_ours.json
```

Important absolute paths (Windows, this repo):

| Thing | Path |
|---|---|
| Our KGs (one folder per doc) | `<PROJECT_ROOT>/graph_rag/our_approach/experiments/MINE_batch_results/` |
| Filtered GT essays | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/data/essays_filtered.json` |
| Filtered GT answers | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/data/answers_filtered.json` |
| Row-index ↔ doc map | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/data/filtered_index_to_mine_doc.json` |
| Our evaluator | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/OURS/run_ours_evaluation.py` |
| Per-run state / batch caches | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/OURS/batch_workdir/` |
| Final results | `<PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score/results/OURS_MINE_batch_results/` |

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
`retrieve(query)` can be scored. That's why we can plug our KGs straight in.

---

## Data preparation — what already ran

Before building this runner:

1. Downloaded the `kyssen/kg-gen-evaluation-answers` dataset (**105 rows** —
   essays + answer-groups).
2. Matched each of the 105 essays against each of our 100 MINE texts using
   8-word shingle Jaccard overlap (written inline during setup; see the
   `filtered_index_to_mine_doc.json` for the final mapping).
3. All 100 of our docs matched (lowest Jaccard = 0.81). Five eval rows
   didn't correspond to any MINE essay and were dropped. Topics of dropped
   rows (and original 105-dataset indices):
   - row 100 — The Effect of Music on Cognitive Development
   - row 101 — The Role of Technology in Crime Prevention
   - row 102 — The Importance of Volunteerism in Society
   - row 103 — The Influence of Pop Culture on Global Trends
   - row 104 — The Role of Water in the Development of Civilization
4. Wrote `essays_filtered.json`, `answers_filtered.json`,
   `filtered_index_to_mine_doc.json` to `data/`. These are what our runner
   reads; the original 105-row files are preserved untouched as
   `kg_gen_eval*.json`.

Effective evaluation scope: **100 essays** (every KG we generated).

---

## Runner architecture

`run_ours_evaluation.py` has three stages. Each is resumable: re-running
skips work whose output is already on disk.

### Stage 1 — `retrieve`  (local, fast, ~1–2 min for 100 docs)

For each doc:

1. `kg_adapter.load_kg_for_doc(experiments_root, doc_id)` — reads our
   `final_output.json` and returns
   `{"entities":[...], "edges":[...], "relations":[[s,r,o],...]}`.
2. `KGGen.from_dict(...)` → Graph → `to_nx` → `generate_embeddings`
   (sentence-transformers `all-MiniLM-L6-v2`, local, free).
3. For each query in the doc's answer-group: `kggen.retrieve(q, ...)` →
   `(top_nodes, neighbor_set, context_text)`. We keep `context_text`.

All outputs are pooled into one JSON file:

```
batch_workdir/retrieved.json
[
  { "filtered_idx": 0, "doc_id": "doc_0",
    "items": [
      { "query_idx": 0, "correct_answer": "...", "retrieved_context": "..." },
      ...
    ],
    "retrieval_error": null
  },
  ...
]
```

Concurrency: `--retrieve-workers N` (default 4). MiniLM embedding model is
small enough that 4 parallel docs on CPU is fine; bump higher if you have
GPU or more cores.

### Stage 2 — `submit`  (OpenAI Batch API)

One batch request per (essay, query) pair. For 100 essays with ~15 queries
each ⇒ ~1500 requests. A single .jsonl under ~5 MB.

- Endpoint: `/v1/responses`
- Model: `gpt-5`
- `reasoning: {"effort": "high"}`    (mirrors DSPy's `reasoning_effort="high"`)
- `text: {"format": {"type": "json_object"}}`
- `max_output_tokens: 16000`
- `custom_id`: `judge|<filtered_idx>|<query_idx>`

System prompt asks the model for strict JSON:
```json
{"reasoning": "<brief>", "evaluation": 0 or 1}
```

Uses the same `BatchClient` from `our_approach/batch/batch_client.py`:
uploads input file, creates batch, polls every **60 s**, downloads output +
error files. State (batch id, file ids, status, counts) is persisted to
`batch_workdir/judge_batch/meta.json` so a killed process can resume.

### Stage 3 — `assemble`  (local, fast)

Reads `batch_workdir/judge_batch/output.jsonl` (and `errors.jsonl`), parses
the JSON body for each `custom_id`, and writes one file per essay:

```
results/OURS_MINE_batch_results/results_<filtered_idx>.json
[
  { "correct_answer": "...", "retrieved_context": "...", "evaluation": 1 },
  ...,
  { "accuracy": "73.33%" }
]
```

This is the **exact schema** `_1_evaluation.py` writes, so
`_2_compare_results.py` and `_3_visualize.py` work on our directory
unchanged (they discover every folder under `results/` automatically).

If a request came back with an API error, empty content, or unparseable
JSON, the assembler falls back to a **sync retry** via
`openai.responses.create(...)` at the same model/effort. Counts of retries
and remaining failures are logged and written to
`results/OURS_MINE_batch_results/summary_ours.json`.

---

## Running it

The runner lives inside the `OURS/` sub-package. Run from the `mine_score/`
directory so Python can resolve the package:

```bash
cd <PROJECT_ROOT>/graph_rag/evaluation_pipeline/mine_score

# activate kgvenv first (the one that has kg-gen installed)

# smoke test — 2 docs end-to-end
python -m OURS.run_ours_evaluation all --limit 2

# full run, one stage at a time (easier to diagnose if something goes wrong)
python -m OURS.run_ours_evaluation retrieve
python -m OURS.run_ours_evaluation submit
python -m OURS.run_ours_evaluation assemble

# or all in one go
python -m OURS.run_ours_evaluation all
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
`OPENAI_API_KEY`, matching the KG-generation run. Drop the `_2` variable
in `.env` if you want the primary key.

---

## Output — what you get

```
results/OURS_MINE_batch_results/
├── results_0.json          ... results_99.json
│   └── each: per-query records + trailing {"accuracy": "..%"}
└── summary_ours.json
    {
      "total_essays": 100,
      "total_queries": ~1500,
      "correct": ...,
      "overall_accuracy_pct": ...,
      "sync_retries": N,
      "sync_retry_failures": M
    }
```

To compare against baselines (if you download their result dirs into
`results/`):

```bash
python _2_compare_results.py        # writes results/results.png + summary.txt
streamlit run _3_visualize.py       # interactive per-query browser
```

---

## Cost and time expectations

- **Stage 1 (retrieve)**: CPU-only, ~1–2 min for 100 docs.
- **Stage 2 (judge batch)**: ~1500 gpt-5 requests at `reasoning_effort=high`
  through the Batch API (50% discount). Runtime depends on OpenAI's batch
  queue; MINE batch SLA is 24 h, but typical completion has been
  5–30 min per batch in our experience. No tokens count against your
  per-model rate limits.
- **Stage 3 (assemble)**: seconds, plus sync retries for any failed
  requests (rare).

---

## Design decisions worth knowing

1. **Why `/v1/responses` instead of chat completions?** Reasoning models
   (o-series, gpt-5) expose `reasoning.effort` only via the Responses API.
   Chat completions ignores it.
2. **Why a plain JSON judge prompt instead of DSPy?** DSPy's prompt is
   sequential: it wraps the judgment in a ChainOfThought loop that doesn't
   serialize to a single batched request. Our system prompt asks the model
   to "think step-by-step internally" and emit a single JSON object, which
   is both batchable and semantically equivalent (gpt-5 at
   `reasoning_effort=high` does its own internal CoT).
3. **Why MiniLM for retrieval?** It's what kg-gen uses by default and what
   the other published methods (kggen, graphrag, openie) were scored with.
   Keeping it identical makes comparisons apples-to-apples.
4. **Why a separate batch vs. per-doc batches?** One submission ⇒ one
   polling session. The batch file size (~5 MB) and request count (~1500)
   are far below OpenAI's per-batch limits (200 MB, 50 000 requests), and
   grouping everything amortizes the batch turnaround.

---

## Troubleshooting

- **`FileNotFoundError: ... final_output.json`** — a doc's KG never got
  generated. Re-run the KG pipeline for that doc or remove it from
  `filtered_index_to_mine_doc.json`.
- **Batch status stuck at `validating`** — the batch input failed parsing.
  Inspect `batch_workdir/judge_batch/meta.json` and OpenAI's batch dashboard.
- **Many sync retries in Stage 3** — either the gpt-5 response came back
  wrapped in unexpected markdown, or quotas were hit. Inspect
  `errors.jsonl` in the judge job folder.
- **`ModuleNotFoundError: kg_gen`** — activate `kgvenv` (or whichever env
  has `kg-gen==0.4.0` installed).
