"""
Compatibility wrapper.

Core query expansion implementation currently lives in the top-level
`query_expansion.py`. Streamlit imports it from `src.retrieval.query_expansion`.
"""

from __future__ import annotations

from query_expansion import QueryExpander

__all__ = ["QueryExpander"]

