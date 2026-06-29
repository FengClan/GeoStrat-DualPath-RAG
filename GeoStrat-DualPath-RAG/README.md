# GeoStrat-DualPath-RAG

Adaptive Dual-Path Retrieval-Augmented Generation for Lithostratigraphic Question Answering.

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

Pre-built data is provided: the knowledge graph under `kb_storage/`, the FAISS index under `chroma_db/`, and test sets under `test_sets/`.

### Run Evaluations

```bash
# Four-baseline comparison (main experiment)
python evaluation/run_eval.py --baselines --test-set test_sets/test_set_expanded.json

# Ablation study
python evaluation/ablation.py --test-set test_sets/test_set_expanded.json

# Quick smoke test (~5 min, 10 questions)
python evaluation/run_eval.py --baselines --test-set test_sets/test_10q.json
```

### Analyze Results

```bash
python scripts/analyze_results.py evaluation/results/baseline_comparison_*.json
```

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
│   ├── run_eval.py           # CLI entry point
│   ├── test_set_builder.py   # Test set builder
│   ├── test_set_schema.json  # Test set JSON schema
│   └── results/              # Historical evaluation outputs
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
│   │   ├── experiment_kb.graphml      # Core KG (448 nodes, 681 edges)
│   │   └── experiment_expanded.graphml # Expanded KG (1187 nodes)
│   └── experiment_triples.json
│
├── test_sets/                # Evaluation test sets
│   ├── test_set_expanded.json # 300-question full set
│   ├── test_30q.json          # 30-question medium set
│   ├── test_10q.json          # 10-question quick test
│   └── smoke_test.json        # Smoke test
│
└── chroma_db/                # FAISS vector index
    ├── faiss.index            # Vector embeddings
    └── faiss_meta.pkl         # Document metadata
```

## Citation

```bibtex
@article{qiu2026adaptive,
  title={Adaptive Dual-Path Retrieval-Augmented Generation for Lithostratigraphic Question Answering},
  author={Qiu, Qinjun and Lv, Yunfeng and Zhang, Yuang and Tian, Miao},
  year={2026}
}
```

## License

MIT
