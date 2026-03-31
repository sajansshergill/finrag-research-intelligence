"""
Compatibility wrapper.

Core GraphRAG implementation currently lives in the top-level `graph_rag.py`.
Streamlit and downstream modules import it from `src.retrieval.graph_rag`.
"""

from __future__ import annotations

from graph_rag import AnalystGraph

__all__ = ["AnalystGraph"]

