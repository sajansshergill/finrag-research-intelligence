"""
reranker.py
-----------
Cross-encoder re-ranking of hybrid retrieval candidates.

Why re-rank?
Bi-encoder retrieval (dense search) is fast but coarse — it computes
query and document embeddings independently. A cross-encoder jointly
encodes query+document, producing a much more accurate relevance score.

Pipeline position: after RRF fusion, before LLM call.
  Input:  top-20 RetrievalResult objects from hybrid.py
  Output: top-5 RetrievalResult objects sorted by cross-encoder score

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fine-tuned on MS MARCO passage ranking
  - Fast (MiniLM) but accurate enough for production
  - Returns logit scores; higher = more relevant
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.hybrid import RetrievalResult


TOP_K_RERANK = 5   # final candidates returned to LLM


class CrossEncoderReranker:
    """
    Wraps the cross-encoder/ms-marco-MiniLM-L-6-v2 model.
    Falls back to a pass-through (RRF score ordering) if the model
    is not installed — ensures the pipeline still works in test environments.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._model   = None
        if not use_mock:
            self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.MODEL_NAME)
            print(f"✓ Cross-encoder loaded: {self.MODEL_NAME}")
        except ImportError:
            print("⚠ sentence-transformers not installed — using mock re-ranker")
            self.use_mock = True

    def rerank(self,
               query:      str,
               candidates: list[RetrievalResult],
               top_k:      int = TOP_K_RERANK) -> list[RetrievalResult]:
        """
        Score each candidate against the query and return top_k sorted
        by cross-encoder score descending.
        """
        if not candidates:
            return []

        if self.use_mock or self._model is None:
            # Fall back to RRF score ordering
            ranked = sorted(candidates, key=lambda r: r.rrf_score, reverse=True)
            for r in ranked:
                r.rerank_score = r.rrf_score
            return ranked[:top_k]

        pairs  = [(query, r.text) for r in candidates]
        scores = self._model.predict(pairs)

        for result, score in zip(candidates, scores):
            result.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda r: r.rerank_score, reverse=True)
        return ranked[:top_k]


# ── Convenience function ──────────────────────────────────────────────────────

def rerank(query: str, candidates: list[RetrievalResult],
           top_k: int = TOP_K_RERANK,
           use_mock: bool = False) -> list[RetrievalResult]:
    reranker = CrossEncoderReranker(use_mock=use_mock)
    return reranker.rerank(query, candidates, top_k)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os, json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    from src.retrieval.hybrid import BM25Index, reciprocal_rank_fusion

    index = BM25Index("data/processed/chunks.jsonl")
    query = "emerging market buy recommendation rating change"

    bm25_results = index.search(query, top_k=20)
    fused = reciprocal_rank_fusion([], bm25_results, index.chunks, top_n=20)

    reranker = CrossEncoderReranker(use_mock=True)
    top = reranker.rerank(query, fused, top_k=5)

    print(f"\nTop-5 after re-ranking for: '{query}'\n")
    for i, r in enumerate(top, 1):
        print(f"{i}. {r.company} [{r.sector} / {r.region}]")
        print(f"   Rating: {r.rating}  |  Coverage: {r.years_coverage}y  "
              f"|  Change: {r.is_rating_change}")
        print(f"   RRF: {r.rrf_score:.4f}  |  Rerank: {r.rerank_score:.4f}")
        print(f"   {r.text[:100]}...\n")