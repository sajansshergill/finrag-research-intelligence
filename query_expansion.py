"""
query_expansion.py
------------------
LLM-based query expansion for improved retrieval recall.

Problem:
  A user types: "analyst upgraded emerging market stock recently"
  The corpus may contain: "we are revising our rating to Buy... EM exposure"
  These don't overlap well in either keyword or embedding space.

Solution:
  Ask the LLM to rewrite the query into N diverse variants, covering
  different phrasings, synonyms, and related angles. Run retrieval for
  each variant, then merge and deduplicate results before re-ranking.

This substantially reduces sensitivity to exact wording and improves
recall for paraphrase-heavy financial language.
"""

from __future__ import annotations
import json


# ── Prompt template ───────────────────────────────────────────────────────────

EXPANSION_SYSTEM = (
    "You are a financial research query specialist. "
    "Rewrite user queries to improve document retrieval from a financial research archive."
)

EXPANSION_USER_TEMPLATE = """Original query: "{query}"

Generate {n} alternative phrasings of this query that:
1. Use different financial terminology and synonyms
2. Cover related angles the user might care about
3. Are specific enough to retrieve relevant analyst research notes

Return ONLY a JSON array of strings, no explanation.
Example: ["alternative 1", "alternative 2", "alternative 3"]"""


# ── Query expander ────────────────────────────────────────────────────────────

class QueryExpander:
    """
    Generates N diverse query variants using an LLM, then merges
    retrieval results across all variants using RRF.
    """

    def __init__(self, llm_client=None, n_variants: int = 3,
                 use_mock: bool = False):
        self.llm_client = llm_client
        self.n_variants = n_variants
        self.use_mock   = use_mock

    # ── Mock expansions for testing ──────────────────────────────────────────

    _MOCK_EXPANSIONS = {
        "default": [
            "analyst upgraded rating to buy or strong buy",
            "recommendation change from hold to buy emerging markets",
            "equity research note rating revision bullish outlook",
        ]
    }

    def expand(self, query: str) -> list[str]:
        """Returns [original_query] + N LLM-generated variants."""
        if self.use_mock or self.llm_client is None:
            variants = self._MOCK_EXPANSIONS.get("default", [])
            return [query] + variants[:self.n_variants]

        prompt = EXPANSION_USER_TEMPLATE.format(
            query=query, n=self.n_variants
        )
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXPANSION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=300,
            temperature=0.4,
        )
        raw = response.choices[0].message.content.strip()
        try:
            variants = json.loads(raw)
            if not isinstance(variants, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            variants = []

        return [query] + variants[:self.n_variants]

    def retrieve_expanded(
        self,
        query:              str,
        retriever,          # HybridRetriever instance
        metadata_filter:    str | None    = None,
        top_n_per_variant:  int           = 10,
        final_top_n:        int           = 20,
    ) -> list:
        """
        Expand the query, run retrieval for each variant,
        merge results using RRF, return top final_top_n.
        """
        from src.retrieval.hybrid import reciprocal_rank_fusion, TOP_K_BM25

        queries  = self.expand(query)
        all_bm25 = []

        for q in queries:
            bm25_hits = retriever.bm25_index.search(q, top_k=top_n_per_variant)
            all_bm25.extend(bm25_hits)

        # Deduplicate by chunk index, keep best score
        best: dict[int, float] = {}
        for idx, score in all_bm25:
            best[idx] = max(best.get(idx, 0), score)

        merged = sorted(best.items(), key=lambda x: x[1], reverse=True)

        # Re-apply RRF over merged results
        return reciprocal_rank_fusion(
            dense_results=[],
            bm25_results=merged[:final_top_n * 2],
            bm25_chunks=retriever.bm25_index.chunks,
            top_n=final_top_n,
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    expander = QueryExpander(use_mock=True)
    query = "emerging market analysts who changed their buy recommendation"
    variants = expander.expand(query)
    print(f"Original: {query}\n")
    print("Expanded queries:")
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v}")