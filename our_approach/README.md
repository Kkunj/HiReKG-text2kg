# Text to Knowledge Graph Pipeline

A multi-phase pipeline that extracts structured knowledge graphs from unstructured text using LLMs.

---

## Overview

This pipeline transforms raw text into a **knowledge graph** consisting of:
- **Entities** (nodes): Key concepts, people, organizations, technologies extracted from text
- **Relationships** (edges): Semantic connections between entities as `(Subject) --[Relation]--> (Object)` triples

**What it does:**
1. Splits text into manageable chunks
2. Extracts named entities using LLM
3. Generates semantic triples with evidence from source text
4. Auto-repairs malformed triples using LLM verification
5. Stores results to Neo4j graph database (optional)

---

<!-- ## Overall Flow

```
                          INPUT TEXT
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│  PHASE 1: Text Preprocessing                             │
│  Split text into sentence-based chunks (spaCy)           │
└──────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│  PHASE 2: Entity Extraction                              │
│  LLM extracts entities per chunk (max 3 words each)      │
└──────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│  PHASE 3: Summarization & Entity Refinement              │
│  3A: Generate document summary (optional)                │
│  3B: Deduplicate entities (deterministic or LLM)         │
└──────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│  PHASE 4: Triple Extraction & Validation                 │
│  4A: Extract (subject, relation, object, evidence)       │
│  4B: Verify & fix malformed relations                    │
│  4C: Resolve embedded entities in objects                │
└──────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│  PHASE 5-6: Semantic Linking (optional)                  │
│  Generate embeddings → Cluster entities → Discover       │
│  implicit relations via LLM                              │
└──────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│  PHASE 7: Neo4j Storage (optional)                       │
│  Persist triples to graph database                       │
└──────────────────────────────────────────────────────────┘
                               │
                               ▼
                       OUTPUT JSON FILES
```
 -->
**File Description:**

| File | Purpose |
|------|---------|
| `main.py` | Entry point - configure and run pipeline |
| `pipeline.py` | Core orchestration - implements all phases |
| `llm_client.py` | Unified LLM client (OpenAI / self-hosted) |
| `prompts.py` | System & user prompts for LLM calls |
| `text_processing.py` | Text chunking using spaCy |
| `object_resolution.py` | Triple repair for embedded entities |
| `semantic_linking.py` | Adds implicit relations between similar entities |
| `neo4j_writer.py` | Neo4j database integration |
| `setup_nltk_models.py` | One-time NLP model downloader |

---

## Installation & Setup

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download NLP Models

```bash
python setup_nltk_models.py
```

This downloads required NLTK data (punkt, wordnet) and spaCy model (en_core_web_sm).

---

## Configuration

### Environment Variables (.env)

Create a `.env` file in the project directory:

```env
# Model type: "openai" or "local"
MODEL_TYPE=openai

# For OpenAI models
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini

# For self-hosted model (if MODEL_TYPE=local)
LOCAL_LLM_URL=http://localhost:8000/v1/chat/completions

# ═══════════════════════════════════════════════════════════
# NEO4J DATABASE (Optional - only if storing to database)
# ═══════════════════════════════════════════════════════════

NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

### Pipeline Parameters

Edit these in `main.py` to customize behavior:

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `sentences_per_chunk` | `4` | Any number | Sentences per text chunk. Lower = more API calls but finer granularity |
| `summarize_text` | `True` | True/False | Generate document summary for context. True for large documents|
| `entity_refinement_mode` | `"deterministic"` | `"deterministic"` / `"llm"` | How to deduplicate entities. Deterministic is faster, LLM is more accurate |
| `enable_semantic_linking` | `False` | True/False | Enable embedding-based implicit relation discovery. Adds noise to the graph |
| `store_to_neo4j` | `False` | True/False | Store results to Neo4j database |
| `embedding_model` | `"text-embedding-3-large"` | Any embedding model | Model for semantic linking |
| `min_cluster_size` | `2` | 2+ | Minimum entities per cluster (semantic linking) |
| `experiment_name` | `"exp_1"` | Any string | Output folder name |

---

## Usage

### Running the Pipeline

```bash
python main.py
```

### Customizing Input Text

Edit the `text` variable in `main.py`:

```python
text = """
Paste your input text here. The pipeline will extract
entities and relationships from this content.
"""
```

### Full Configuration Example

```python
# main.py

from llm_client import LLMClient
from pipeline import KGCreationPipeline

# Initialize LLM client
llm_client = LLMClient(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    model_type=os.getenv("MODEL_TYPE", "openai")
)

# Configure pipeline
pipeline = KGCreationPipeline(
    llm_client=llm_client,
    sentences_per_chunk=4,
    summarize_text=True,
    entity_refinement_mode="deterministic",
    enable_semantic_linking=False,
    save_experiments=True,
    experiment_name="my_experiment",
    logger=logger
)

# Run pipeline
results = pipeline.run(text, store_to_neo4j=False)
```

---

## Output Structure

All outputs are saved to: `experiments/<experiment_name>/`

### Generated Files

| File | Description |
|------|-------------|
| `processed_text.json` | Text split into chunks |
| `local_entities.json` | Raw extracted entities with frequency counts |
| `refined_entities.json` | Deduplicated canonical entity list |
| `global_summary.json` | Document summary (if enabled) |
| `triplets.json` | Final extracted triples with statistics |
| `object_resolution.json` | Before/after of repaired triples |
| `final_output.json` | Complete pipeline result |
| `pipeline_<timestamp>.log` | Execution log |

**Additional files (if semantic linking enabled):**

| File | Description |
|------|-------------|
| `entity_embeddings.json` | Entity embedding vectors |
| `cluster_information.json` | Entity cluster assignments |
| `semantic_triplets.json` | Discovered implicit relations |


---

