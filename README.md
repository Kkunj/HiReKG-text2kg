# HiReKG

This zip contains the source code, evaluation pipeline, datasets, and raw
experiment outputs that back the paper *HiReKG* (anonymised submission to
EMNLP / ACL Rolling Review).

---

## 1. Environment

Python 3.10, Windows 11 / Ubuntu 22.04. Dependencies in
[`requirements.txt`](requirements.txt).

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
pip install kg-gen                 # upstream MINE retrieval routine, not vendored
python -m spacy download en_core_web_sm
python our_approach/setup_nltk_models.py
```

Copy [`.env.example`](.env.example) to `.env` and fill in your own API keys
and self-hosted vLLM URL.

---

## 2. Data

| Dataset | Location |
|---|---|
| MINE | [`datasets/MINE/`](datasets/MINE/) |
| SciERC | [`datasets/scierc/`](datasets/scierc/) |
| RedoCRED (windowed subset) | [`datasets/windows_redocred/`](datasets/windows_redocred/) |

All three are bundled in-tree. Each has `texts/`, `triplets/` (gold), and
where applicable `atomic_facts/`.

---

## 3. Running the pipelines

Each runner is independently resumable: rerunning skips docs whose
`final_output.json` already exists. All hyperparameters and CLI flags are
defined inside the scripts themselves.

| Pipeline | Entry point |
|---|---|
| HiReKG (ours) | [`our_approach/run_mine_batch.py`](our_approach/run_mine_batch.py) |
| RAKG baseline | [`baselines/rakg/run_mine_batch.py`](baselines/rakg/run_mine_batch.py) |
| kg-gen baseline | [`baselines/kggen/run_mine_batch.py`](baselines/kggen/run_mine_batch.py) |

To run all three sequentially against one vLLM endpoint, use
[`run_all_batches.ps1`](run_all_batches.ps1).

---

## 4. Evaluation and results

Evaluation entry points live under `evaluation_pipeline/`:

- [`evaluation_pipeline/structural/`](evaluation_pipeline/structural/) — URI ratio + LLM-token accounting
- [`evaluation_pipeline/mine_score/`](evaluation_pipeline/mine_score/) — MINE benchmark scoring
- [`evaluation_pipeline/faithfulness/`](evaluation_pipeline/faithfulness/) — LLM-judge faithfulness
- [`evaluation_pipeline/object_resolution_rate/`](evaluation_pipeline/object_resolution_rate/) — entity-resolution metric

The numbers reported in the paper come from the aggregated JSON files in:

- [`evaluation_pipeline/structural/`](evaluation_pipeline/structural/) (top-level `uir_*` and `*_uir_results.json`)
- [`evaluation_pipeline/mine_score/results/`](evaluation_pipeline/mine_score/results/)
- [`evaluation_pipeline/faithfulness/results/`](evaluation_pipeline/faithfulness/results/) and [`evaluation_pipeline/faithfulness/qwen_32b_results/`](evaluation_pipeline/faithfulness/qwen_32b_results/)
- [`results/`](results/) (top-level cross-dataset aggregates and plot scripts)

Two supporting analyses sit outside `evaluation_pipeline/`:

- [`scripts/verify_entity_classification_spacy.py`](scripts/verify_entity_classification_spacy.py)
  — validates the GPT-5 single-/multi-entity labels on MINE-1 against spaCy
  `en_core_web_sm` NER, writing
  [`results/mine_spacy_agreement.json`](results/mine_spacy_agreement.json)
  (the agreement rate, Cohen's kappa and per-class precision/recall quoted in
  the appendix).
- [`scripts/append_scierc_entities.py`](scripts/append_scierc_entities.py)
  — attaches SciERC gold NER spans (excluding `Generic`) to the stage-3 atomic
  facts, which is what makes the deterministic classification in
  `fact_generation/classify_entities_deterministic.py` possible.

The KGGen GPT-4o MINE-1 run came from the upstream kg-gen release and uses a
flat `N.json` layout rather than this repo's `doc_<N>/final_output.json`, so it
has two dedicated helpers:
[`baselines/kggen/convert_kggen_for_faithfulness.py`](baselines/kggen/convert_kggen_for_faithfulness.py)
(re-lays it out for the faithfulness pipeline) and
[`evaluation_pipeline/mine_score/score_kggen_gpt4o_graphs.py`](evaluation_pipeline/mine_score/score_kggen_gpt4o_graphs.py)
(its single-/multi-entity MINE decomposition). The raw graphs are in
`baselines/kggen/experiments/kggen_mine_gpt4o_graphs/`.

### Experiment outputs

The aggregated results -- every number reported in the paper -- are committed to
this repository and need no download. The **per-document** raw outputs (the
extracted KGs and the per-triple judge verdicts behind those aggregates) come to
381 MB across ~20,000 files, so they are published as **release assets** instead
of being committed:

| Asset | Size | Contents |
|---|---|---|
| `hirekg-ours.zip` | 12 MB | `our_approach/experiments/` -- HiReKG KGs, all 11 runs |
| `hirekg-baselines.zip` | 11 MB | `baselines/{kggen,rakg}/experiments/` -- baseline KGs |
| `hirekg-evaluation.zip` | 15 MB | per-document MINE results and faithfulness judge audits |
| `hirekg-experiments-all.zip` | 39 MB | all three of the above |

Each archive stores paths relative to the repository root, so unzip at the root
and the original layout is restored in place:

```bash
unzip hirekg-experiments-all.zip -d .
```

You need these only to re-run the evaluators or inspect individual documents.
Two scripts additionally require `hirekg-baselines.zip` to be extracted, because
they read the raw KGGen GPT-4o graphs:
`baselines/kggen/convert_kggen_for_faithfulness.py` and
`evaluation_pipeline/mine_score/score_kggen_gpt4o_graphs.py`.

Once extracted, the per-document KGs the evaluators consume live under each
pipeline's `experiments/<experiment_name>/doc_<N>/final_output.json`.

A representative pipeline run log (head + tail, ~300 lines, IPs scrubbed)
is in [`sample_pipeline.log`](sample_pipeline.log).

---

## 5. Layout

```
HiReKG-submission/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── run_all_batches.ps1
├── sample_pipeline.log
├── datasets/                       MINE, SciERC, RedoCRED
├── our_approach/                   the HiReKG pipeline
├── baselines/
│   ├── rakg/                       RAKG reimplemented on our LLMClient
│   └── kggen/                      kg-gen reimplemented on our LLMClient
├── evaluation_pipeline/
│   ├── structural/
│   ├── faithfulness/
│   ├── mine_score/
│   └── object_resolution_rate/
├── fact_generation/                atomic-fact pipeline
├── experiments/                    walkthrough run
├── scripts/                        spaCy label check, SciERC entity attach
└── results/plot_scripts/           MINE-distribution figure
```

The upstream RAKG and kg-gen repositories are referenced via URL in
`baselines/rakg/howtouse.md` and `baselines/kggen/howtouse.md` respectively
(not bundled).

---

## 6. Licenses

HiReKG itself is released under the MIT License ([`LICENSE`](LICENSE)).
Third-party artifacts:

- MINE benchmark: [`kyssen/kg-gen-evaluation-essays`](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-essays) and
  [`kyssen/kg-gen-evaluation-answers`](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-answers) on Hugging Face, redistributed under their original terms.
- RAKG — [github.com/RAKG/RAKG](https://github.com/KnowledgeXLab/RAKG)
- kg-gen — [github.com/stair-lab/kg-gen](https://github.com/stair-lab/kg-gen)
