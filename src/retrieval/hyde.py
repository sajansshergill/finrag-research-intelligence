"""
Compatibility wrapper.

Core HyDE implementation currently lives in the top-level `hyde.py`.
Streamlit imports it from `src.retrieval.hyde`.
"""

from __future__ import annotations

from hyde import HyDERetriever

__all__ = ["HyDERetriever"]

