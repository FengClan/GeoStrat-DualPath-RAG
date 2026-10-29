# GeoStrat-DualPath-RAG

An Adaptive Dual-Path Retrieval-Augmented Generation Method for Lithostratigraphic Question Answering via Large Language Model and Knowledge Graph

## Overview

We propose an adaptive dual-path RAG method for lithostratigraphic QA that follows a "classify → retrieve → fuse" design:

1. **Query Classifier** — categorizes questions into FACTUAL, COMPARATIVE, REASONING, or SPATIAL
2. **Dual-Path Retrieval** — vector search over geological literature (FAISS + text2vec) + multi-hop graph traversal over a domain knowledge graph (NetworkX + BFS)
3. **Adaptive Fusion** — type-aware prompt templates that blend the two knowledge sources with differentiated weights, augmented with fine-grained citation markers [Dn]/[Gn]

## Quick Start

### Prerequisites

- Python 3.9+
- LLM API key ([Aliyun Bailian / DashScope](https://dashscope.aliyuncs.com) recommended; register for free)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and set BAILIAN_API_KEY
```

### Download Embedding Model

The `shibing624/text2vec-base-chinese` embedding model (~400 MB) is downloaded automatically on first use:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('shibing624/text2vec-base-chinese')"
```

> Model page: [https://huggingface.co/shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)
>
> If you are behind a firewall in China, set the HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`

### Start the Service

```bash
python app.py
# Service running at http://localhost:5000
```

### Quick Test

```bash
curl -X POST http://localhost:5000/api/geoqa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the geological age of the Changlin Formation?", "model": "qwen3.5-plus"}'
```

## Reproducing Experiments

### Data Preparation

| Step | Script | Input | Output |
|------|--------|-------|--------|
| Build knowledge graph | `scripts/convert_excel_to_kg.py` | Stratigraphic units Excel | `kb_storage/graphs/experiment_kb.graphml` |
| Build vector index | `scripts/load_docx_to_vectordb.py` | Geological literature .docx | `chroma_db/` |

Pre-built data is provided, so these steps can be skipped:

| Artifact | Path | Size |
|----------|------|------|
| Knowledge graph | `kb_storage/graphs/experiment_kb.graphml` | 1,623 nodes, 2,894 edges |
| FAISS vector index | `chroma_db/faiss.index` + `chroma_db/faiss_meta.pkl` | — |
| Evaluation datasets | `test_sets/` | see below |

### Datasets

| File | Items | Use |
|------|-------|-----|
| `test_set_honest.json` | 1,307 | Full stratified dataset |
| `test_set_honest_dev.json` | 150 | Development split used for tuning |
| `test_set_honest_test.json` | 1,157 | Held-out test split |
| `test_set_expanded.json` | 300 | Earlier set used for the baseline comparison |
| `test_30q.json` | 30 | Medium quick check |
| `test_10q.json` | 10 | Quick check |
| `smoke_test.json` | 4 | Smoke test |

The dev/test split is 3:1 and preserves the query-type proportions of the full set.

### Run Evaluations

```bash
# Quick smoke test (~5 min)
python evaluation/run_eval.py --baselines --test-set test_sets/test_10q.json

# Main four-baseline comparison, 300-item stratified sample
python evaluation/run_full_eval.py --total 300 --seed 42

# Multi-round repeats with mean ± SD (defaults to the 150-item dev split)
python evaluation/run_multirun.py --mode baseline --rounds 3

# Component ablation
python evaluation/run_multirun.py --mode ablation --rounds 3

# Claim-level hallucination verification
python evaluation/run_hallucination_eval.py --total 100 --round 1

# Retrieval-depth sensitivity
python evaluation/run_sensitivity.py --hops 1,2,3
```

Every runner checkpoints after each question, so an interrupted run resumes by
re-running the same command.

### Analyze Results

```bash
python scripts/analyze_results.py evaluation/results/multirun_baseline_20260917.json
```

Evaluation outputs are kept under `evaluation/results*/`, including per-question
checkpoints and the aggregated summaries reported in the paper.

## Core API

| Endpoint | Description |
|----------|-------------|
| `POST /api/geoqa` | Dual-path QA with adaptive fusion |
| `POST /api/geoqa/reasoning-path` | KG reasoning path for visualization |
| `POST /api/query/classify` | Query type classification |
| `POST /api/eval/baselines` | Run four-baseline comparison |

## Project Structure

```
├── app.py                    # Flask entry point
├── rag_core.py               # FAISS vector store & stratigraphic chunker
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
│
├── services/                 # API service blueprints (13 modules)
│   ├── rag_service.py        # Core dual-path RAG (adaptive fusion + fine-grained citations)
│   ├── kg_service.py         # Knowledge graph service (BFS multi-hop retrieval)
│   ├── query_classifier.py   # Query type classifier
│   ├── model_router.py       # LLM model routing
│   ├── ontology.py           # Geological domain ontology
│   ├── kg_extraction.py      # Ontology-constrained KG triple extraction
│   ├── entity_alignment.py   # Multi-source entity alignment
│   ├── strat_service.py      # Stratigraphic comparison
│   ├── cross_region_compare.py  # Cross-region comparison & synonym detection
│   ├── reranker.py           # Hybrid retrieval + cross-encoder reranking
│   ├── extraction_service.py # NER / RE / STE extraction
│   ├── litho_service.py      # Lithostratigraphic extraction
│   ├── eval_service.py       # Evaluation API
│   ├── pipeline_orchestrator.py  # Full 7-stage pipeline
│   ├── config.py             # Central configuration
│   └── logger.py             # Logging utilities
│
├── evaluation/               # Evaluation framework
│   ├── evaluator.py          # Multi-dimension evaluation engine (LLM-as-Judge)
│   ├── baselines.py          # Four-baseline comparison runner
│   ├── ablation.py           # Ablation study runner
│   ├── run_eval.py           # Simple CLI entry point
│   ├── run_full_eval.py      # Stratified 300-item main experiment
│   ├── run_multirun.py       # Multi-round repeats with mean ± SD
│   ├── run_hallucination_eval.py # Claim-level hallucination verification
│   ├── run_sensitivity.py    # Retrieval-depth sensitivity analysis
│   ├── run_pilot_independent.py  # Small pilot runs
│   ├── rerun_dualpath.py     # Re-run the proposed method only
│   ├── test_set_builder.py   # Test set builder
│   ├── test_set_schema.json  # Test set JSON schema
│   └── results*/             # Evaluation outputs and checkpoints
│
├── scripts/                  # Data construction & analysis
│   ├── convert_excel_to_kg.py    # Excel → knowledge graph
│   ├── load_docx_to_vectordb.py  # Document → FAISS vector index
│   ├── generate_test_questions.py # Template-based test set generation
│   ├── expand_test_set.py    # LLM-assisted test set expansion
│   └── analyze_results.py    # Result → LaTeX table generator
│
├── kb_storage/               # Knowledge graph data
│   ├── graphs/
│   │   ├── experiment_kb.graphml        # Core KG (1,623 nodes, 2,894 edges)
│   │   └── experiment_expanded.graphml  # Earlier expanded KG (1,187 nodes, 1,666 edges)
│   ├── experiment_triples*.json   # Triple extractions
│   └── d3_human_review*.csv       # Expert review sample
│
├── test_sets/                # Evaluation datasets
│   ├── test_set_honest.json       # 1,307-item stratified dataset
│   ├── test_set_honest_dev.json   # 150-item development split
│   ├── test_set_honest_test.json  # 1,157-item test split
│   ├── test_set_expanded.json     # 300-item set for baseline comparison
│   └── test_30q.json / test_10q.json / smoke_test.json
│
└── chroma_db/                # FAISS vector index
    ├── faiss.index            # Vector embeddings
    └── faiss_meta.pkl         # Document metadata
```

## Citation

```bibtex
@article{li2026adaptive,
  title={An Adaptive Dual-Path Retrieval-Augmented Generation Method for Lithostratigraphic Question Answering via Large Language Model and Knowledge Graph},
  author={Li, Yun and Lv, Yunfeng and Zhu, Hui and Xiao, Bin and Li, Dajun and Qiu, Qinjun and Zhang, Yuang and Tian, Miao},
  year={2026}
}
```

## License

MIT
