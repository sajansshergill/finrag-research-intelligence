"""
hybrid.py
---------
Hybrid retrieval combining dense vector search (Qdrant) with sparse
BM25 keyword search, fused via Reciprocal Rank Fusion (RRF).

Why hybrid?
- Dense search excels at semantic similarity ("analyst changed view")
- BM25 excels at exact term matching ("Emerging Markets", ticker symbols)
- RRF fusion outperforms either method alone on financial domain queries

Pipeline:
  1. Apply metadata pre-filter → get eligible report_ids from DuckDB
  2. Dense retrieval  → top-K from Qdrant (payload-filtered by report_ids)
  3. BM25 retrieval   → top-K from in-memory BM25 index
  4. RRF fusion       → merge ranked lists into a single ranked result
  5. Return top-N candidates for re-ranking

RRF formula:
  score(d) = Σ 1 / (k + rank(d, list_i))
  k=60 is standard; reduces the impact of highly-ranked outliers.
"""

import json
from dataclasses import dataclass, field
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

TOP_K_DENSE  = 20   # candidates from vector search
TOP_K_BM25   = 20   # candidates from BM25
TOP_N_FUSION = 20   # output of RRF (goes to re-ranker)
RRF_K        = 60   # RRF constant


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id:          str
    report_id:         str
    text:              str
    analyst_name:      str
    company:           str
    sector:            str
    region:            str
    rating:            str
    old_rating:        Optional[str]
    is_rating_change:  bool
    rating_change_date:Optional[str]
    years_coverage:    int
    publication_date:  str
    target_price:      int
    dense_rank:        Optional[int]   = None
    bm25_rank:         Optional[int]   = None
    rrf_score:         float           = 0.0
    rerank_score:      Optional[float] = None


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index over chunked corpus.
    Built once at startup from the chunks JSONL file.
    """

    def __init__(self, chunks_path: str):
        # `rank-bm25` is optional; in constrained environments (or CI) we fall
        # back to a tiny local BM25 implementation.
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception:  # pragma: no cover
            BM25Okapi = None  # type: ignore[assignment]

        class _SimpleBM25:
            def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
                import math

                self.corpus_tokens = corpus_tokens
                self.k1 = k1
                self.b = b

                self.doc_lens = [len(d) for d in corpus_tokens]
                self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0

                df: dict[str, int] = {}
                for doc in corpus_tokens:
                    for t in set(doc):
                        df[t] = df.get(t, 0) + 1

                self.df = df
                self.N = len(corpus_tokens)
                self._math = math

            def get_scores(self, query_tokens: list[str]) -> list[float]:
                if not self.corpus_tokens:
                    return []
                if not query_tokens:
                    return [0.0] * len(self.corpus_tokens)

                scores = [0.0] * len(self.corpus_tokens)
                for i, doc in enumerate(self.corpus_tokens):
                    if not doc:
                        continue
                    freqs: dict[str, int] = {}
                    for t in doc:
                        freqs[t] = freqs.get(t, 0) + 1

                    dl = self.doc_lens[i]
                    denom_norm = 1.0 - self.b + self.b * (dl / self.avgdl) if self.avgdl else 1.0

                    for t in query_tokens:
                        f = freqs.get(t, 0)
                        if f == 0:
                            continue
                        # Standard BM25 idf with +1 smoothing to keep it positive.
                        n_q = self.df.get(t, 0)
                        idf = self._math.log(1.0 + (self.N - n_q + 0.5) / (n_q + 0.5))
                        scores[i] += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * denom_norm)

                return scores

        self.chunks: list[dict] = []
        with open(chunks_path) as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]

        tokenized = [self._tokenize(c["text"]) for c in self.chunks]
        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = _SimpleBM25(tokenized)
        print(f"✓ BM25 index built: {len(self.chunks)} chunks")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = TOP_K_BM25,
               allowed_report_ids: Optional[set] = None) -> list[tuple[int, float]]:
        """
        Returns list of (chunk_index, bm25_score) sorted descending.
        If allowed_report_ids is set, only returns chunks from those reports.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            if allowed_report_ids and chunk["report_id"] not in allowed_report_ids:
                continue
            results.append((idx, score))
            if len(results) >= top_k:
                break

        return results


# ── Dense retrieval (Qdrant) ──────────────────────────────────────────────────

class DenseRetriever:
    """Wraps Qdrant search with payload filtering."""

    def __init__(self, qdrant_url: str = "http://localhost:6333",
                 collection: str = "finrag"):
        try:
            from qdrant_client import QdrantClient
            self._client     = QdrantClient(url=qdrant_url)
            self.collection  = collection
        except ImportError:
            self._client = None
            print("⚠ qdrant-client not available — dense retrieval disabled")

    def embed_query(self, query: str) -> list[float]:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            return model.encode(query, normalize_embeddings=True).tolist()
        except ImportError:
            import random
            return [random.gauss(0, 0.1) for _ in range(1024)]

    def search(self, query: str, top_k: int = TOP_K_DENSE,
               allowed_report_ids: Optional[set] = None) -> list[dict]:
        """
        Returns list of Qdrant hit payloads, filtered by allowed_report_ids.
        """
        if self._client is None:
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchAny

        vector = self.embed_query(query)
        query_filter = None

        if allowed_report_ids:
            query_filter = Filter(must=[
                FieldCondition(
                    key="report_id",
                    match=MatchAny(any=list(allowed_report_ids)),
                )
            ])

        hits = self._client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [hit.payload for hit in hits]


# ── RRF Fusion ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results:  list[dict],
    bm25_results:   list[tuple[int, float]],
    bm25_chunks:    list[dict],
    top_n:          int = TOP_N_FUSION,
    k:              int = RRF_K,
) -> list[RetrievalResult]:
    """
    Merge dense and BM25 ranked lists using Reciprocal Rank Fusion.
    Returns top_n RetrievalResult objects sorted by RRF score descending.
    """
    scores: dict[str, float]              = {}
    meta:   dict[str, dict]               = {}
    ranks:  dict[str, dict[str, int]]     = {}

    # Score dense results
    for rank, payload in enumerate(dense_results, start=1):
        cid = payload["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        meta[cid]   = payload
        ranks.setdefault(cid, {})["dense"] = rank

    # Score BM25 results
    for rank, (idx, _) in enumerate(bm25_results, start=1):
        chunk = bm25_chunks[idx]
        cid   = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        if cid not in meta:
            meta[cid] = chunk
        ranks.setdefault(cid, {})["bm25"] = rank

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]

    results = []
    for cid in sorted_ids:
        m = meta[cid]
        r = ranks.get(cid, {})
        results.append(RetrievalResult(
            chunk_id          = cid,
            report_id         = m.get("report_id", ""),
            text              = m.get("text", ""),
            analyst_name      = m.get("analyst_name", ""),
            company           = m.get("company", ""),
            sector            = m.get("sector", ""),
            region            = m.get("region", ""),
            rating            = m.get("rating", ""),
            old_rating        = m.get("old_rating"),
            is_rating_change  = m.get("is_rating_change", False),
            rating_change_date= m.get("rating_change_date"),
            years_coverage    = m.get("years_coverage", 0),
            publication_date  = m.get("publication_date", ""),
            target_price      = m.get("target_price", 0),
            dense_rank        = r.get("dense"),
            bm25_rank         = r.get("bm25"),
            rrf_score         = scores[cid],
        ))

    return results


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Orchestrates metadata pre-filtering → dense + BM25 → RRF fusion.
    """

    def __init__(self, chunks_path: str,
                 db_path: str          = "data/finrag.duckdb",
                 qdrant_url: str       = "http://localhost:6333",
                 collection: str       = "finrag"):
        self.bm25_index     = BM25Index(chunks_path)
        self.dense_retriever= DenseRetriever(qdrant_url, collection)
        self.db_path        = db_path

    def retrieve(
        self,
        query:              str,
        metadata_filter:    Optional[str] = None,   # SQL WHERE clause
        top_n:              int           = TOP_N_FUSION,
    ) -> list[RetrievalResult]:
        """
        Full hybrid retrieval pipeline.
        metadata_filter: SQL WHERE clause string from build_metadata_filter()
        """
        # Step 1: metadata pre-filter
        allowed_ids = None
        if metadata_filter and metadata_filter != "1=1":
            from src.ingestion.metadata_loader import get_filtered_report_ids
            ids = get_filtered_report_ids(self.db_path, metadata_filter)
            allowed_ids = set(ids)
            if not allowed_ids:
                return []   # no reports match structural criteria

        # Step 2: dense retrieval
        dense_results = self.dense_retriever.search(
            query, top_k=TOP_K_DENSE, allowed_report_ids=allowed_ids
        )

        # Step 3: BM25 retrieval
        bm25_results = self.bm25_index.search(
            query, top_k=TOP_K_BM25, allowed_report_ids=allowed_ids
        )

        # Step 4: RRF fusion
        return reciprocal_rank_fusion(
            dense_results, bm25_results, self.bm25_index.chunks, top_n
        )


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    # BM25-only test (no Qdrant needed)
    index = BM25Index("data/processed/chunks.jsonl")

    query = "emerging market buy recommendation analyst changed view"
    results = index.search(query, top_k=5)

    print(f"\nBM25 top-5 for: '{query}'\n")
    with open("data/processed/chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    for rank, (idx, score) in enumerate(results, 1):
        c = chunks[idx]
        print(f"{rank}. [{c['sector']} / {c['region']}] {c['company']} "
              f"| {c['rating']} | {c['years_coverage']}y | score={score:.3f}")
        print(f"   {c['text'][:120]}...\n")