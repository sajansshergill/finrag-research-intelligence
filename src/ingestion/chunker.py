"""
Compatibility wrapper.

The project originally implemented chunking in the top-level `chunker.py`.
Tests and downstream code import it from `src.ingestion.chunker`.

We load via importlib so imports work even when the process cwd is not the
repo root (e.g. `python scripts/bootstrap.py`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_chunker_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "chunker.py"
    spec = importlib.util.spec_from_file_location("finrag_chunker_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load chunker from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_chunker_module()
Chunk = _mod.Chunk
chunk_report = _mod.chunk_report
chunk_corpus = _mod.chunk_corpus
split_sentences = _mod.split_sentences

__all__ = ["Chunk", "chunk_report", "chunk_corpus", "split_sentences"]
