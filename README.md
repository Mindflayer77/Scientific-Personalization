# Personalization via DPO

A research pipeline for training personalized language models that generate scientific hypotheses tailored to individual user personas. Starting from a Neo4j knowledge graph of research papers, it constructs SFT and DPO datasets by retrieving relevant context, generating hypothesis pairs with LLM backends (Gemini, OpenAI, DeepSeek), and simulating per-persona preferences. Models are fine-tuned with LoRA via TRL and evaluated on groundedness, relevancy, and persona adherence using an LLM judge.

## Repository Structure

```
.
├── configs/                  # Training and accelerate configurations
├── config/                   # Dataset generation config
├── chat_templates/           # Custom Jinja2 chat templates
├── prompt_templates/         # Jinja2 templates for LLM prompts
├── notebooks/                # Analysis and evaluation notebooks
├── scripts/                  # Core pipeline scripts
│   ├── dataset_generation/   # DPO dataset generation
│   ├── taxonomy/             # AI/ML taxonomy generation
│   └── weaviate_db/          # Vector database management
├── scripts_2/                # Neighbourhood-based DPO pipeline
├── server_scripts/           # SLURM job scripts for HPC cluster
└── src/personalization/      # Core Python package
    ├── api_client/           # Gemini API client
    ├── dataset/              # Dataset loading and processing
    ├── evaluation/           # Evaluation logic and metrics
    ├── models/               # Model factory (LoRA / base)
    └── train/                # SFT and DPO training strategies
```

## Setup

```bash
pip install -e .
cp config/config.yaml.example config/config.yaml  # edit paths and API keys
```

Requires a `.env` file with API keys (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `HF_TOKEN`, `GOOGLE_API_KEY` as needed).

---

## Scripts

### Training & Evaluation (`scripts/`)

#### `train_baseline.py`
Trains an SFT or DPO personalization model using LoRA. Reads config from a YAML file, loads the dataset, applies the selected training strategy (`sft`, `dpo`, `sft+dpo`, or `cl` for continual learning), and logs to Weights & Biases.

```bash
python scripts/train_baseline.py --config configs/default.yaml
# or via accelerate for multi-GPU + DeepSpeed:
accelerate launch --config_file configs/accelerate_config.yaml scripts/train_baseline.py --config configs/default.yaml
```

#### `eval_baseline.py`
Evaluates a trained checkpoint by generating personalized responses for test prompts and optionally scoring them with an LLM judge (OpenAI, DeepSeek, or Gemini). Supports few-shot generation and `generation_only` mode.

```bash
python scripts/eval_baseline.py --config configs/default.yaml
```

---

### Dataset Generation (`scripts/dataset_generation/`)

#### `generate_prompts.py`
Generates synthetic research question prompts for each user persona by querying Neo4j for article nodes and rendering Jinja2 templates with the OpenAI API.

#### `classify_articles.py`
Classifies Neo4j articles into persona categories using the OpenAI API and structured output (Pydantic schemas).

#### `extract_concepts_from_prompts.py`
Extracts key scientific concepts from generated prompts using the OpenAI API to enrich DPO data with structured metadata.

#### `generate_hypotheses_from_context.py`
Generates DPO hypothesis pairs (chosen / rejected) from pre-built context CSVs. Supports three backends: local vLLM server, Gemini API, or DeepSeek API.

#### `generate_hypotheses_from_subqueries_weaviate_hybrid.py`
Retrieves relevant context from Weaviate using hybrid search (dense + sparse) over paper chunks and generates DPO hypothesis pairs via the OpenAI API.

#### `retrieval.py`
Performs hybrid retrieval from Weaviate for a set of prompts and saves context CSVs used as input for hypothesis generation.

#### `sample_neo4j_nodes.py`
Samples nodes and relationship edges from Neo4j into lightweight CSV files for downstream dataset construction.

---

### Taxonomy Generation (`scripts/taxonomy/`)

A three-phase pipeline that builds a structured AI/ML taxonomy used to generate diverse, frontier-level research prompts. See [`scripts/taxonomy/README.md`](scripts/taxonomy/README.md) for full details.

#### `generate_taxonomy_phase1.py`
Generates a high-level hierarchical taxonomy of AI/ML fields, sub-fields, and disciplines using the OpenAI API.

#### `generate_taxonomy_phase2.py`
Expands each discipline into PhD-level academic subjects with metadata (level, description, key subtopics) using OpenAI or Gemini.

#### `filter_taxonomy_phase2.py`
Filters the raw Phase 2 output to retain only high-value frontier and advanced subjects, with per-discipline balancing.

#### `generate_taxonomy_phase3.py`
Generates session-by-session syllabi with atomic concepts for each subject using the OpenAI API.

---

### Weaviate Vector Database (`scripts/weaviate_db/`)

#### `create_db.py`
Creates the `ResearchPapers` Weaviate collection schema (text properties, no external vectorizer).

#### `import_data_to_db.py`
Bulk-imports paper chunks from JSON files into the Weaviate collection.

#### `import_backup.py`
Restores a named Weaviate backup (e.g., a pre-built 1M-vector index) with extended timeout settings.

---

### Neighbourhood DPO Pipeline (`scripts_2/`)

A second pipeline that constructs DPO pairs using a graph neighbourhood structure from Neo4j, enabling richer contextual diversity.

#### `sample_neo4j_nodes.py` → `build_neighbourhood_dataset.py`
For each sampled seed node, retrieves up to N graph neighbours from Neo4j (ensuring no overlapping neighbourhoods between seeds) and assembles DPO pairs with chosen/rejected persona assignments.

#### `stratify_dpo_dataset.py`
Balances the neighbourhood dataset so every `(chosen_persona, rejected_persona)` pair is equally represented.

#### `generate_hypotheses_neighbourhood.py`
Generates chosen hypotheses for neighbourhood DPO rows by fetching neighbour node summaries from Neo4j and calling the Gemini API.

#### `generate_prompts_neighbourhood.py`
Generates research question prompts for the chosen hypotheses using the Gemini API and Jinja2 templates.

#### `generate_full.py`
Generates both chosen and rejected hypotheses in a single pass for neighbourhood DPO pairs using the Gemini API.

#### `simulate_user_rationale.py`
Simulates a user persona explaining why the chosen hypothesis was preferred, appending rationale text to the DPO dataset.

#### `evaluate_hypotheses.py`
Scores chosen and rejected hypotheses on four LLM-judge metrics (groundedness, relevancy, persona adherence, and an overall score) using Gemini, OpenAI, or DeepSeek.

#### `judge_from_csv.py`
Runs swap-augmented win-rate evaluation (AlpacaEval / MT-Bench style) on a generation CSV, comparing trained vs. baseline model outputs with an LLM judge.

#### `metrics_from_csv.py`
Computes per-model quality scores (groundedness, relevancy, persona adherence) from a generation CSV for the trained and trained few-shot models.

#### `compute_cl_metrics_exp4.py`
Computes continual-learning metrics (ACC, BWT, FWT, FM) across multiple permutations of the task sequence for Experiment 4.

---

### Notebooks (`notebooks/`)

| Notebook | Purpose |
|---|---|
| `clean.ipynb` | Data cleaning and preprocessing of raw hypothesis CSVs |
| `eval.ipynb` | Evaluation with RAGAS metrics (faithfulness, context relevance, response groundedness) |
| `evaluation.ipynb` | Cosine similarity, Self-BLEU, UMAP visualization, and inter-persona analysis |
| `neo4j.ipynb` | Neo4j graph exploration and dataset inspection |
| `plot.ipynb` | Plotting evaluation results and metric comparisons |

---

## Configuration

### `configs/default.yaml`
Main configuration for training and evaluation. Key sections:

- `data` — paths to SFT/DPO train/val/test CSVs and persona JSON
- `model` — base model name (`Qwen/Qwen3-4B` by default), LoRA rank/alpha/dropout
- `training` — learning rate, batch size, strategy (`sft` / `dpo` / `sft+dpo` / `cl`)
- `logging` — W&B project, entity, run name, and tags
- `evaluation` — checkpoint path, generation parameters, judge provider and model

### `configs/accelerate_config.yaml`
Accelerate / DeepSpeed config for multi-GPU training: ZeRO Stage 2, bf16, 4 processes.

### `config/config.yaml`
Dataset generation paths (Neo4j connection, article directories, taxonomy file names, LLM model names). Copy from `config/config.yaml.example` before use.
