"""
Airflow DAG: FinRAG ingestion pipeline.

This is the runnable DAG entrypoint for the Docker Compose `airflow` service.
It mirrors the logic in the repo-level `ingestion_dag.py`, but lives in `dags/`
so Airflow auto-discovers it.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Ensure `/opt/airflow` (parent of `src/` and `data/`) is importable.
# In Docker Compose we mount `./src` to `/opt/airflow/src`.
if "/opt/airflow" not in sys.path:
    sys.path.insert(0, "/opt/airflow")


default_args = {
    "owner": "finrag",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

dag = DAG(
    dag_id="finrag_ingestion",
    description="FinRAG: daily research ingestion, embedding, and eval",
    default_args=default_args,
    schedule_interval="0 2 * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["finrag", "rag", "ingestion"],
)


def task_generate_corpus(**_context):
    from pathlib import Path

    corpus_path = Path("data/raw/corpus.jsonl")
    if corpus_path.exists():
        mod_time = datetime.fromtimestamp(corpus_path.stat().st_mtime)
        if mod_time.date() == datetime.utcnow().date():
            print("Corpus already current — skipping generation")
            return

    from src.ingestion.synthetic_data import generate_corpus

    generate_corpus(count=500, output_dir="data/raw/")
    print("✓ Corpus generated")


def task_chunk_corpus(**_context):
    from src.ingestion.chunker import chunk_corpus

    chunk_corpus(
        input_path="data/raw/corpus.jsonl",
        output_path="data/processed/chunks.jsonl",
    )


def task_load_metadata(**_context):
    from src.ingestion.metadata_loader import MetadataLoader

    loader = MetadataLoader("data/finrag.duckdb")
    n_reports = loader.load_reports("data/raw/corpus.jsonl")
    n_chunks = loader.load_chunks("data/processed/chunks.jsonl")
    loader.build_analyst_profiles()
    loader.close()
    print(f"✓ Loaded {n_reports} reports, {n_chunks} chunks")


def task_embed_and_upsert(**_context):
    import json

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    collection = os.environ.get("QDRANT_COLLECTION", "finrag")

    with open("data/processed/chunks.jsonl") as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    # If qdrant/sentence-transformers aren't available in the container,
    # this will raise; that's OK (the DAG will show the missing dependency).
    from src.ingestion.embedder import FinRAGEmbedder

    embedder = FinRAGEmbedder(qdrant_url=qdrant_url, collection=collection)
    n = embedder.upsert_chunks(all_chunks)
    print(f"✓ Upserted {n} vectors to Qdrant")


def task_build_ground_truth(**_context):
    # Ground truth requires DuckDB (metadata) + processed chunks.
    from src.eval.ground_truth import build_eval_queries, build_ground_truth, save_ground_truth

    queries = build_eval_queries()
    queries = build_ground_truth(
        queries,
        db_path="data/finrag.duckdb",
        chunks_path="data/processed/chunks.jsonl",
    )
    save_ground_truth(queries, "data/eval/ground_truth.jsonl")


t_generate = PythonOperator(
    task_id="generate_corpus",
    python_callable=task_generate_corpus,
    dag=dag,
)

t_chunk = PythonOperator(
    task_id="chunk_corpus",
    python_callable=task_chunk_corpus,
    dag=dag,
)

t_metadata = PythonOperator(
    task_id="load_metadata",
    python_callable=task_load_metadata,
    dag=dag,
)

t_embed = PythonOperator(
    task_id="embed_and_upsert",
    python_callable=task_embed_and_upsert,
    dag=dag,
)

t_gt = PythonOperator(
    task_id="build_ground_truth",
    python_callable=task_build_ground_truth,
    dag=dag,
)

t_generate >> t_chunk >> t_metadata >> [t_embed, t_gt]

