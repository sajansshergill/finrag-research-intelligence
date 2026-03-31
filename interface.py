"""
interface.py
------------
LLM interface for grounded answer generation over retrieved chunks.

Responsibilities:
  1. Build a structured prompt with retrieved context + metadata
  2. Call the LLM (OpenAI GPT-4o or Anthropic Claude)
  3. Parse the response into a structured AnswerResult with citations
  4. Support structured financial queries (rec tracking, analyst Q&A)

Grounding principle:
  The LLM is instructed to answer ONLY from provided context.
  Each claim must reference a source chunk_id. This enables faithfulness
  evaluation downstream and prevents hallucination.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.hybrid import RetrievalResult


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Citation:
    chunk_id:    str
    report_id:   str
    analyst:     str
    company:     str
    sector:      str
    region:      str
    rating:      str
    pub_date:    str
    text_snippet: str


@dataclass
class AnswerResult:
    query:        str
    answer:       str
    citations:    list[Citation]
    model:        str
    prompt_tokens:  int = 0
    answer_tokens:  int = 0
    latency_ms:     float = 0.0


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior financial research analyst assistant. 
Answer questions using ONLY the research excerpts provided below.
For every claim, cite the source using [SOURCE: chunk_id].
If the context does not contain enough information, say so explicitly.
Do not speculate beyond what the sources state."""

def build_context_block(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    lines = []
    for i, r in enumerate(results, 1):
        change_info = ""
        if r.is_rating_change and r.old_rating:
            change_info = f" (changed from {r.old_rating} → {r.rating}"
            if r.rating_change_date:
                change_info += f" on {r.rating_change_date}"
            change_info += ")"

        lines.append(
            f"[{i}] SOURCE ID: {r.chunk_id}\n"
            f"    Analyst: {r.analyst_name} | Company: {r.company} | "
            f"Sector: {r.sector} | Region: {r.region}\n"
            f"    Rating: {r.rating}{change_info} | "
            f"Coverage: {r.years_coverage} years | "
            f"Published: {r.publication_date}\n"
            f"    Excerpt: {r.text}\n"
        )
    return "\n".join(lines)


def build_prompt(query: str, results: list[RetrievalResult]) -> str:
    context = build_context_block(results)
    return (
        f"RESEARCH CONTEXT:\n\n{context}\n\n"
        f"USER QUERY: {query}\n\n"
        f"Provide a concise, grounded answer citing specific sources with "
        f"[SOURCE: chunk_id]. Highlight any rating changes, analyst views, "
        f"and key investment themes relevant to the query."
    )


# ── LLM clients ───────────────────────────────────────────────────────────────

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model  = model

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=800,
            temperature=0.1,
        )
        msg = response.choices[0].message.content
        return msg, response.usage.prompt_tokens, response.usage.completion_tokens


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model  = model

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text
        return text, response.usage.input_tokens, response.usage.output_tokens


class MockLLMClient:
    """Deterministic mock for unit tests — no API calls."""
    model = "mock"

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        # Extract first chunk_id from the prompt for a realistic-looking citation
        import re
        ids = re.findall(r"SOURCE ID: (\S+)", user)
        cite = f"[SOURCE: {ids[0]}]" if ids else "[SOURCE: unknown]"
        answer = (
            f"Based on the provided research context, several analysts have "
            f"recently updated their recommendations on emerging market equities. "
            f"{cite} The analysis indicates strong conviction for buy-rated names "
            f"with improving fundamental momentum."
        )
        return answer, len(user.split()), len(answer.split())


# ── Citation parser ───────────────────────────────────────────────────────────

def parse_citations(answer: str,
                    results: list[RetrievalResult]) -> list[Citation]:
    """
    Extract [SOURCE: chunk_id] references from the answer text
    and match them to RetrievalResult objects.
    """
    import re
    cited_ids = set(re.findall(r"\[SOURCE:\s*(\S+?)\]", answer))
    result_map = {r.chunk_id: r for r in results}

    citations = []
    for cid in cited_ids:
        if cid in result_map:
            r = result_map[cid]
            citations.append(Citation(
                chunk_id     = cid,
                report_id    = r.report_id,
                analyst      = r.analyst_name,
                company      = r.company,
                sector       = r.sector,
                region       = r.region,
                rating       = r.rating,
                pub_date     = r.publication_date,
                text_snippet = r.text[:200],
            ))
    return citations


# ── Main interface ────────────────────────────────────────────────────────────

class FinRAGInterface:
    """
    Orchestrates the final retrieval → prompt → LLM → answer pipeline.
    """

    def __init__(self, llm_client=None, use_mock: bool = False):
        if use_mock or llm_client is None:
            self.llm = MockLLMClient()
        else:
            self.llm = llm_client

    def answer(self, query: str,
               results: list[RetrievalResult],
               latency_ms: float = 0.0) -> AnswerResult:
        """
        Generate a grounded answer from retrieved results.
        """
        import time
        prompt = build_prompt(query, results)
        t0     = time.time()
        text, p_tok, a_tok = self.llm.complete(SYSTEM_PROMPT, prompt)
        elapsed = (time.time() - t0) * 1000

        citations = parse_citations(text, results)

        return AnswerResult(
            query         = query,
            answer        = text,
            citations     = citations,
            model         = self.llm.model,
            prompt_tokens = p_tok,
            answer_tokens = a_tok,
            latency_ms    = elapsed + latency_ms,
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    from src.retrieval.hybrid import BM25Index, reciprocal_rank_fusion

    index    = BM25Index("data/processed/chunks.jsonl")
    query    = "buy recommendations in emerging markets from senior analysts"
    hits     = index.search(query, top_k=10)
    fused    = reciprocal_rank_fusion([], hits, index.chunks, top_n=5)

    iface  = FinRAGInterface(use_mock=True)
    result = iface.answer(query, fused)

    print(f"Query  : {result.query}\n")
    print(f"Answer : {result.answer}\n")
    print(f"Citations ({len(result.citations)}):")
    for c in result.citations:
        print(f"  · {c.analyst} | {c.company} | {c.sector} | {c.rating}")