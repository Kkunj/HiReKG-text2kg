# RAKG Baseline Pipeline

Reimplements the [RAKG](https://github.com/RAKG/RAKG) (Retrieval Augmented Knowledge Graph) pipeline using the shared `LLMClient` from `our_approach/`. All LLM calls go through a single, consistent client.

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

| Phase | Description | Uses LLM? |
|-------|-------------|-----------|
| **1. Sentence Segmentation** | Regex-based splitting into sentences | No |
| **2. Sentence Vectorization** | Embed all sentences for later retrieval | Embeddings |
| **3. Entity Extraction (NER)** | Extract entities per sentence via LLM | Main LLM |
| **4. Similarity Candidates** | Find entity pairs with high embedding cosine similarity | Embeddings |
| **5. Entity Disambiguation** | LLM judges if candidate pairs are the same entity | Similarity LLM |
| **6. Entity Merging** | Union-find to merge confirmed duplicates | No |
| **7. KG Construction** | Per entity: retrieve context sentences + LLM extracts entity-centric subgraph | Embeddings + Main LLM |
| **8. KG Conversion** | Merge all subgraphs into unified entities + relations | No |

## Changing LLM Models

```python
# OpenAI
config = {"model": "gpt-4o-mini", "model_type": "openai"}

# NVIDIA NIM
config = {"model": "nvidia/nemotron-3-nano-30b-a3b", "model_type": "nim", "max_output_tokens": 16384}

# Separate model for entity disambiguation (optional)
config = {
    "model": "gpt-4o-mini",
    "model_type": "openai",
    "similarity_model": "gpt-4o-mini",
    "similarity_model_type": "openai",
}
```

## All Config Keys

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `gpt-4o-mini` | Main LLM for NER + KG extraction |
| `model_type` | `openai` | `"openai"`, `"nim"`, or `"local"` |
| `base_url` | `None` | Custom API base URL |
| `temperature` | `0.0` | Sampling temperature |
| `max_output_tokens` | `10000` | Max tokens per LLM call |
| `max_retries` | `3` | Retry count on API errors |
| `similarity_model` | `None` | Separate LLM for disambiguation (defaults to main) |
| `similarity_model_type` | `None` | Model type for similarity LLM |
| `embedding_model` | `BAAI/bge-m3` | Embedding model name |
| `embedding_backend` | `local` | `"local"` (sentence-transformers) or `"openai"` |
| `similarity_threshold` | `0.60` | Cosine similarity threshold for candidate pairs |
| `retrieval_top_k` | `5` | Number of retrieved context sentences per entity |
| `save_experiments` | `True` | Save outputs to experiments directory |
| `experiment_name` | `rakg_exp_1` | Experiment folder name |
| `log_to_console` | `True` | Print INFO logs to terminal |
| `log_to_file` | `True` | Write DEBUG logs to file |

## Output Structure

```
experiments/{experiment_name}/
    pipeline_{timestamp}.log   # Full debug log
    processed_text.json        # Segmented sentences
    local_entities.json        # Raw extracted entities (name, type, description)
    refined_entities.json      # Final entity names after disambiguation
    triplets.json              # Final triples + statistics
    final_output.json          # Complete pipeline output
```
