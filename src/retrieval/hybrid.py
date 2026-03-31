"""
Compatibility wrapper.

Core hybrid retrieval code lives in the repo-root `hybrid.py`. We load it by file
path so `import hybrid` never accidentally resolves to this package (e.g. when
running Streamlit or editable installs with ambiguous `sys.path`).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_HYBRID_PATH = _ROOT / "hybrid.py"


def _load_hybrid_root() -> object:
    spec = importlib.util.spec_from_file_location("finrag_hybrid_root", _HYBRID_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load hybrid implementation from {_HYBRID_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_h = _load_hybrid_root()

BM25Index = _h.BM25Index
DenseRetriever = _h.DenseRetriever
HybridRetriever = _h.HybridRetriever
RetrievalResult = _h.RetrievalResult
RRF_K = _h.RRF_K
TOP_K_BM25 = _h.TOP_K_BM25
TOP_K_DENSE = _h.TOP_K_DENSE
TOP_N_FUSION = _h.TOP_N_FUSION
reciprocal_rank_fusion = _h.reciprocal_rank_fusion

__all__ = [
    "BM25Index",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "RRF_K",
    "TOP_K_BM25",
    "TOP_K_DENSE",
    "TOP_N_FUSION",
    "reciprocal_rank_fusion",
]
