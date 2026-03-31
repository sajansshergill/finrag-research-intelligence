"""
Compatibility wrapper.

Core reranking code currently lives in the top-level `reranker.py`.
Tests and other modules import it from `src.retrieval.reranker`.
"""

from __future__ import annotations

from reranker import CrossEncoderReranker, TOP_K_RERANK, rerank

__all__ = ["CrossEncoderReranker", "TOP_K_RERANK", "rerank"]

