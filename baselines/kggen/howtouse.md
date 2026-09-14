# KGGen Baseline Pipeline

Reimplements the [kg-gen](https://github.com/stair-lab/kg-gen) knowledge graph extraction pipeline using the shared `LLMClient` from `our_approach/`. All LLM calls go through a single, consistent client.

## Quick Start

```python
from pipeline import run_pipeline

text = "Your input text here..."

config = {
    "model": "gpt-4o-mini",
    "model_type": "openai",
    "experiment_name": "my_experiment",
}

results = run_pipeline(text, config)
```

## Pipeline Steps

| Phase | Description |
|-------|-------------|
| **1. Chunking** | Splits text into character-limited chunks respecting sentence boundaries (NLTK) |
| **2. Entity Extraction** | Extracts entities per chunk via LLM (DSPy-style prompt) |
| **2b. Entity Filtering** | Removes entities containing double-quote characters |
| **3. Relation Extraction** | Extracts (subject, predicate, object) triples per chunk via LLM |
| **3b. Relation Fixing** | If subject/object don't match entity list, makes a second LLM call to remap them |
| **4. Deduplication** | Merges entities/relations across chunks, then runs semhash deduplication |

## Changing LLM Models

Set `model` and `model_type` in the config dict:

```python
# OpenAI
config = {"model": "gpt-4o-mini", "model_type": "openai"}

# NVIDIA NIM (requires NVIDIA_ENDPOINT_URL and NVIDIA_ENDPOINT_API in .env)
config = {"model": "nvidia/nemotron-3-nano-30b-a3b", "model_type": "nim", "max_output_tokens": 16384}

# Self-hosted / Local (requires LOCAL_LLM_URL in .env)
config = {"model": "your-model-name", "model_type": "local", "base_url": "http://localhost:8080/v1"}
```

> **Note:** Reasoning models (e.g., Nemotron) need higher `max_output_tokens` (16384+) since thinking tokens count toward the limit.

## All Config Keys

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `gpt-4o-mini` | Model name |
| `model_type` | `openai` | `"openai"`, `"nim"`, or `"local"` |
| `base_url` | `None` | Custom API base URL (required for `local`) |
| `temperature` | `0.0` | Sampling temperature |
| `max_output_tokens` | `10000` | Max output tokens per LLM call |
| `max_retries` | `3` | Retry count on API errors |
| `chunk_size` | `5000` | Max characters per chunk |
| `context` | `""` | Domain description (metadata only) |
| `deduplication` | `semhash` | `"semhash"` or `"none"` |
| `semhash_threshold` | `0.95` | Similarity threshold for semhash dedup |
| `save_experiments` | `True` | Save outputs to experiments directory |
| `experiment_name` | `kggen_exp_1` | Experiment folder name |
| `log_to_console` | `True` | Print INFO logs to terminal |
| `log_to_file` | `True` | Write DEBUG logs to file |

## Output Structure

```
experiments/{experiment_name}/
    pipeline_{timestamp}.log   # Full debug log
    processed_text.json        # Chunked text with word counts
    local_entities.json        # Raw entities before dedup + frequency
    refined_entities.json      # Entities after dedup
    triplets.json              # Final triples + statistics
    final_output.json          # Complete pipeline output
```
