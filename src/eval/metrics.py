"""
Compatibility wrapper.

Core metric implementations currently live in the top-level `metrics.py`.
Tests and other modules import them from `src.eval.metrics`.
"""

from __future__ import annotations

from metrics import (  # noqa: F401
    AnswerMetrics,
    EvalLogger,
    EvalResult,
    RetrievalMetrics,
    compute_answer_metrics,
    compute_retrieval_metrics,
    evaluate_faithfulness,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "AnswerMetrics",
    "EvalLogger",
    "EvalResult",
    "RetrievalMetrics",
    "compute_answer_metrics",
    "compute_retrieval_metrics",
    "evaluate_faithfulness",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]

