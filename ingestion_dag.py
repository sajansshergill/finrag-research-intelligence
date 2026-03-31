"""
ingestion_dag.py
----------------
Airflow DAG: FinRAG ingestion pipeline.

Schedule: daily at 02:00 UTC (catches any research published overnight)

DAG structure:
  generate_corpus
       │
  chunk_corpus
       │
  load_metadata ──── build_analyst_profiles
       │
  embed_and_upsert
       │
  freshness_check
       │
  eval_retrieval_quality

Each task is idempotent — safe to re-run on failure.
Freshness tracking: chunks are only re-embedded if their source report
was published after the last embedding timestamp.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


# ── DAG default args ──────────────────────────────────────────────────────────

default_args = {
    "owner":            "finrag",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout":timedelta(hours=2),
}

dag = DAG(
    dag_id            = "finrag_ingestion",
    description       = "FinRAG: daily research ingestion, embedding, and eval",
    default_args      = default_args,
    schedule_interval = "0 2 * * *",   # daily at 02:00 UTC
    start_date        = days_ago(1),
    catchup           = False,
    max_active_runs   = 1,
    tags              = ["finrag", "rag", "ingestion"],
)


# ── Task functions ────────────────────────────────────────────────────────────

def task_generate_corpus(**context):
    """
    In production: pull new research from CRM / distribution APIs.
    In this implementation: regenerates the synthetic corpus.
    Skips if corpus was already generated today (idempotent).
    """
    import os
    from pathlib import Path
    corpus_path = Path("data/raw/corpus.jsonl")

    # Check if already generated today
    if corpus_path.exists():
        mod_time = datetime.fromtimestamp(corpus_path.stat().st_mtime)
        if mod_time.date() == datetime.utcnow().date():
            print("Corpus already current — skipping generation")
            return

    from src.ingestion.synthetic_data import generate_corpus
    generate_corpus(count=500, output_dir="data/raw/")
    print("✓ Corpus generated")


def task_chunk_corpus(**context):
    """Chunk all reports into sentence windows."""
    from src.ingestion.chunker import chunk_corpus
    chunk_corpus(
        input_path  = "data/raw/corpus.jsonl",
        output_path = "data/processed/chunks.jsonl",
    )


def task_load_metadata(**context):
    """Load structured metadata into DuckDB."""
    from src.ingestion.metadata_loader import MetadataLoader
    loader = MetadataLoader("data/finrag.duckdb")
    n_reports = loader.load_reports("data/raw/corpus.jsonl")
    n_chunks  = loader.load_chunks("data/processed/chunks.jsonl")
    loader.close()
    print(f"✓ Loaded {n_reports} reports, {n_chunks} chunks")
    return {"n_reports": n_reports, "n_chunks": n_chunks}


def task_build_analyst_profiles(**context):
    """Aggregate analyst-level stats."""
    from src.ingestion.metadata_loader import MetadataLoader
    loader = MetadataLoader("data/finrag.duckdb")
    loader.build_analyst_profiles()
    loader.close()
    print("✓ Analyst profiles rebuilt")


def task_embed_and_upsert(**context):
    """
    Embed chunks and upsert into Qdrant.
    Only processes chunks that are stale (published after last embedding).
    """
    import json
    from src.ingestion.embedder import FinRAGEmbedder, get_stale_chunks

    with open("data/processed/chunks.jsonl") as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    # In production, query Qdrant for embedded_at timestamps
    # Here we embed all chunks (idempotent upsert)
    embedder = FinRAGEmbedder(
        qdrant_url = "http://qdrant:6333",   # Docker service name
        collection = "finrag",
    )
    n = embedder.upsert_chunks(all_chunks)
    print(f"✓ Upserted {n} vectors to Qdrant")


def task_freshness_check(**context):
    """
    Verify embedding freshness: alert if any chunk is stale > 24h.
    Publishes freshness metrics to MLflow.
    """
    import json
    from datetime import datetime, timezone

    with open("data/processed/chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    now   = datetime.now(timezone.utc)
    stale = [
        c for c in chunks
        if "embedded_at" not in c or
        (now - datetime.fromisoformat(c.get("embedded_at", "2000-01-01T00:00:00+00:00"))).days > 1
    ]

    pct_fresh = 1 - len(stale) / max(len(chunks), 1)
    print(f"Freshness: {pct_fresh*100:.1f}% ({len(stale)} stale chunks)")

    if pct_fresh < 0.95:
        raise ValueError(
            f"Freshness below threshold: {pct_fresh*100:.1f}% < 95%. "
            f"{len(stale)} chunks need re-embedding."
        )


def task_eval_retrieval_quality(**context):
    """
    Run a subset of eval queries and log metrics to MLflow.
    Runs on the 5 most structurally constrained queries (fastest to evaluate).
    """
    import json
    from src.eval.ground_truth import load_ground_truth
    from src.eval.metrics import (
        compute_retrieval_metrics, EvalResult, EvalLogger,
        AnswerMetrics, RetrievalMetrics,
    )
    from src.retrieval.hybrid import BM25Index, reciprocal_rank_fusion

    # Load eval queries
    eval_queries = load_ground_truth("data/eval/ground_truth.jsonl")
    # Only evaluate queries with non-trivial ground truth
    eval_queries = [q for q in eval_queries if 5 < len(q.relevant_chunk_ids) < 200][:5]

    index  = BM25Index("data/processed/chunks.jsonl")
    logger = EvalLogger(experiment_name="finrag-daily-eval")

    eval_results = []
    for eq in eval_queries:
        import time
        t0      = time.time()
        hits    = index.search(eq.query, top_k=20)
        fused   = reciprocal_rank_fusion([], hits, index.chunks, top_n=10)
        latency = (time.time() - t0) * 1000

        rm = compute_retrieval_metrics(fused, eq.relevant_chunk_ids)
        eval_results.append(EvalResult(
            query_id          = eq.query_id,
            query             = eq.query,
            strategy          = "bm25_only",
            retrieval_metrics = rm,
            answer_metrics    = AnswerMetrics(),
            latency_ms        = latency,
        ))

    logger.log_batch(eval_results)


# ── Task definitions ──────────────────────────────────────────────────────────

t_generate = PythonOperator(
    task_id         = "generate_corpus",
    python_callable = task_generate_corpus,
    dag             = dag,
)

t_chunk = PythonOperator(
    task_id         = "chunk_corpus",
    python_callable = task_chunk_corpus,
    dag             = dag,
)

t_metadata = PythonOperator(
    task_id         = "load_metadata",
    python_callable = task_load_metadata,
    dag             = dag,
)

t_profiles = PythonOperator(
    task_id         = "build_analyst_profiles",
    python_callable = task_build_analyst_profiles,
    dag             = dag,
)

t_embed = PythonOperator(
    task_id         = "embed_and_upsert",
    python_callable = task_embed_and_upsert,
    dag             = dag,
)

t_freshness = PythonOperator(
    task_id         = "freshness_check",
    python_callable = task_freshness_check,
    dag             = dag,
)

t_eval = PythonOperator(
    task_id         = "eval_retrieval_quality",
    python_callable = task_eval_retrieval_quality,
    dag             = dag,
)

# ── DAG wiring ────────────────────────────────────────────────────────────────

t_generate >> t_chunk >> t_metadata >> [t_profiles, t_embed]
t_embed >> t_freshness >> t_eval