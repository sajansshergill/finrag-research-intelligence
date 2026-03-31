"""
hyde.py
-------
HyDE: Hypothetical Document Embeddings retrieval strategy.

Problem it solves:
  User queries are short and underspecified. A query like
  "analysts who changed their view on emerging markets" has a very
  different embedding distribution from the long, detailed research
  notes it should retrieve. This embedding gap hurts recall.

HyDE solution:
  1. Ask the LLM to generate a hypothetical "ideal" research note that
     would perfectly answer the query — as if the analyst had written it.
  2. Embed the hypothetical document (not the original query).
  3. Use that embedding for vector search — it lives in the same embedding
     space as real research notes, dramatically improving recall.

Tradeoff: one extra LLM call per query adds ~300-800ms latency.
Use HyDE for exploratory/broad queries; skip for precise filter queries.
"""

from __future__ import annotations
import os

# ── HyDE prompt template ──────────────────────────────────────────────────────

HYDE_SYSTEM_PROMPT = """You are a senior equity research analyst with 15 years of 
experience covering global financial markets. Write in the precise, formal style 
used in institutional research reports."""

HYDE_USER_TEMPLATE = """A client asked: "{query}"

Write a short excerpt (3-4 sentences) from a financial research note that would 
directly address this query. Include specific details like sector, region, analyst 
rationale, and recommendation language. Write as if this is a real research note."""


# ── HyDE retriever ────────────────────────────────────────────────────────────

class HyDERetriever:
    """
    Generates a hypothetical document for the query, embeds it,
    and performs dense retrieval using the hypothetical embedding.
    """

    def __init__(self, llm_client=None, embedder=None, use_mock: bool = False):
        self.llm_client = llm_client
        self.embedder   = embedder
        self.use_mock   = use_mock

    def generate_hypothetical_doc(self, query: str) -> str:
        """Ask LLM to write a hypothetical research note for the query."""
        if self.use_mock or self.llm_client is None:
            # Mock: return a template-based document for testing
            return (
                f"Following a comprehensive review of market conditions, we are "
                f"upgrading our outlook on {query.split()[0] if query.split() else 'equity'} "
                f"with a Buy rating. The fundamental drivers supporting this view "
                f"include improving macroeconomic conditions, strong earnings momentum, "
                f"and favorable sector rotation trends in emerging markets. "
                f"Our analysis suggests significant upside potential over the next 12 months."
            )

        # OpenAI client
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user",   "content": HYDE_USER_TEMPLATE.format(query=query)},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string using the configured embedder."""
        if self.use_mock or self.embedder is None:
            import random
            return [random.gauss(0, 0.1) for _ in range(1024)]
        return self.embedder.embed_texts([text])[0]

    def retrieve(self, query: str,
                 qdrant_client=None,
                 collection: str     = "finrag",
                 top_k: int          = 10,
                 allowed_report_ids: set | None = None) -> list[dict]:
        """
        Full HyDE pipeline:
          1. Generate hypothetical document
          2. Embed it
          3. Search Qdrant with the hypothetical embedding
        Returns list of payload dicts.
        """
        hyp_doc    = self.generate_hypothetical_doc(query)
        hyp_vector = self.embed_text(hyp_doc)

        if self.use_mock or qdrant_client is None:
            print(f"[HyDE mock] Hypothetical doc: {hyp_doc[:120]}...")
            return []   # no Qdrant in test env

        from qdrant_client.models import Filter, FieldCondition, MatchAny

        query_filter = None
        if allowed_report_ids:
            query_filter = Filter(must=[
                FieldCondition(
                    key="report_id",
                    match=MatchAny(any=list(allowed_report_ids)),
                )
            ])

        hits = qdrant_client.search(
            collection_name=collection,
            query_vector=hyp_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [hit.payload for hit in hits]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hyde = HyDERetriever(use_mock=True)
    query = "emerging market buy recommendations from analysts who changed their view"
    hyp_doc = hyde.generate_hypothetical_doc(query)
    print(f"Query: {query}\n")
    print(f"Hypothetical document:\n{hyp_doc}")