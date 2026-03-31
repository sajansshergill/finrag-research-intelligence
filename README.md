# FinRAG — Financial Research Intelligence (Hybrid RAG demo)

FinRAG is an end-to-end **hybrid RAG** demo over a synthetic financial research archive:

- **Ingestion**: generate synthetic research reports → sentence-window chunking
- **Metadata**: DuckDB for structured filters (sector/region/rating/coverage/date)
- **Retrieval**: BM25 + (optional) dense search + **RRF fusion**
- **Re-ranking**: cross-encoder (mock by default)
- **Answering**: grounded LLM answers with `[SOURCE: chunk_id]` citations (mock by default)
- **Eval set**: programmatic ground truth built from metadata filters

This repo is designed to run **without external services** (mock LLM, BM25-only retrieval) and optionally with Docker services (Qdrant, MLflow, Airflow).

## Quickstart (local, no Docker)

```bash
# Core app + tests (BM25, mock LLM; no torch). For embeddings/Qdrant/MLflow add [full].
python -m pip install -e ".[dev,full]"
make bootstrap
make app
```

Streamlit Community Cloud uses **`requirements.txt`** (pip). Do not rely on Poetry + `pyproject.toml` alone there, or heavy packages (e.g. `sentence-transformers` → torch/triton) can fail to install on the cloud image.

Then open `http://localhost:8501`.

## Quickstart (Docker Compose)

```bash
docker compose up -d --build
```

- **Streamlit**: `http://localhost:8501`
- **Qdrant**: `http://localhost:6333`
- **MLflow**: `http://localhost:5000`
- **Airflow**: `http://localhost:8080` (admin/admin)

## Common commands

```bash
make test
python scripts/bootstrap.py --count 500
streamlit run app/streamlit_app.py
```

## Notes

- **No `rank-bm25` required**: `hybrid.py` includes a small BM25 fallback if the package isn’t installed.
- **LLM keys optional**: the UI supports mock mode; set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` to use real models.
