"""
Bootstrap the FinRAG demo dataset end-to-end.

This script creates:
  - data/raw/corpus.jsonl
  - data/processed/chunks.jsonl
  - data/finrag.duckdb
  - data/eval/ground_truth.jsonl

It is designed to work in "no external services" mode.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="Number of synthetic reports")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    from src.ingestion.synthetic_data import generate_corpus
    from src.ingestion.chunker import chunk_corpus
    from src.ingestion.metadata_loader import MetadataLoader
    from src.eval.ground_truth import build_eval_queries, build_ground_truth, save_ground_truth

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/eval").mkdir(parents=True, exist_ok=True)

    generate_corpus(count=args.count, output_dir="data/raw/")

    chunk_corpus(
        input_path="data/raw/corpus.jsonl",
        output_path="data/processed/chunks.jsonl",
    )

    loader = MetadataLoader("data/finrag.duckdb")
    loader.load_reports("data/raw/corpus.jsonl")
    loader.load_chunks("data/processed/chunks.jsonl")
    loader.build_analyst_profiles()
    loader.close()

    queries = build_eval_queries()
    queries = build_ground_truth(
        queries,
        db_path="data/finrag.duckdb",
        chunks_path="data/processed/chunks.jsonl",
    )
    save_ground_truth(queries, "data/eval/ground_truth.jsonl")

    print("\n✓ Bootstrap complete")
    print("Next:")
    print("  streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

