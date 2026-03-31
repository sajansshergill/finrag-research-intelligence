"""
Compatibility wrapper.

The project originally implemented chunking in the top-level `chunker.py`.
Tests and downstream code import it from `src.ingestion.chunker`.
"""

from __future__ import annotations

from chunker import Chunk, chunk_report, chunk_corpus, split_sentences

__all__ = ["Chunk", "chunk_report", "chunk_corpus", "split_sentences"]

