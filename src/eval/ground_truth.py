"""
Compatibility wrapper.

Core ground-truth builder currently lives in the top-level `ground_truth.py`.
Airflow DAG imports it from `src.eval.ground_truth`.
"""

from __future__ import annotations

from ground_truth import EvalQuery, build_eval_queries, build_ground_truth, load_ground_truth, save_ground_truth

__all__ = [
    "EvalQuery",
    "build_eval_queries",
    "build_ground_truth",
    "load_ground_truth",
    "save_ground_truth",
]

