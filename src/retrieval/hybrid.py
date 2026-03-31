"""
Compatibility wrapper.

Core hybrid retrieval code currently lives in the top-level `hybrid.py`.
Tests and other modules import it from `src.retrieval.hybrid`.
"""

from __future__ import annotations

from hybrid import (  # noqa: F401
    BM25Index,
    DenseRetriever,
    HybridRetriever,
    RetrievalResult,
    reciprocal_rank_fusion,
)

__all__ = [
    "BM25Index",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "reciprocal_rank_fusion",
]

