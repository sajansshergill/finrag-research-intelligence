"""
metrics.py
----------
Retrieval and answer quality evaluation metrics for FinRAG.

Metrics tracked:
  Retrieval:
    - precision@k  : fraction of top-k results that are relevant
    - recall@k     : fraction of relevant docs found in top-k
    - MRR          : mean reciprocal rank of first relevant result
    - NDCG@k       : normalized discounted cumulative gain

  Answer quality:
    - faithfulness : fraction of claims in answer supported by context
                     (LLM-as-judge, no external ground truth needed)
    - answer_relevance : does the answer address the query?

All runs are logged to MLflow for tracking and comparison.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.hybrid import RetrievalResult
    from src.llm.interface import AnswerResult


# ── Metric models ─────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    precision_at_1:  float = 0.0
    precision_at_5:  float = 0.0
    precision_at_10: float = 0.0
    recall_at_5:     float = 0.0
    recall_at_10:    float = 0.0
    mrr:             float = 0.0
    ndcg_at_5:       float = 0.0
    num_relevant:    int   = 0
    num_retrieved:   int   = 0


@dataclass
class AnswerMetrics:
    faithfulness:       float = 0.0   # 0.0 - 1.0
    answer_relevance:   float = 0.0
    citation_coverage:  float = 0.0   # % of answer claims that have citations
    num_citations:      int   = 0


@dataclass
class EvalResult:
    query_id:          str
    query:             str
    strategy:          str            # "hybrid", "hyde", "expanded", etc.
    retrieval_metrics: RetrievalMetrics
    answer_metrics:    AnswerMetrics
    latency_ms:        float = 0.0
    model:             str   = ""
    timestamp:         str   = ""


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str],
                   relevant_ids:  set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for rid in top_k if rid in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str],
                relevant_ids:  set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for rid in top_k if rid in relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str],
                         relevant_ids: set[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str],
              relevant_ids:  set[str], k: int) -> float:
    import math
    dcg  = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(retrieved_ids[:k])
        if rid in relevant_ids
    )
    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant_ids), k))
    )
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(results: list[RetrievalResult],
                              relevant_ids: set[str]) -> RetrievalMetrics:
    retrieved_ids = [r.chunk_id for r in results]
    return RetrievalMetrics(
        precision_at_1  = precision_at_k(retrieved_ids, relevant_ids, 1),
        precision_at_5  = precision_at_k(retrieved_ids, relevant_ids, 5),
        precision_at_10 = precision_at_k(retrieved_ids, relevant_ids, 10),
        recall_at_5     = recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10    = recall_at_k(retrieved_ids, relevant_ids, 10),
        mrr             = mean_reciprocal_rank(retrieved_ids, relevant_ids),
        ndcg_at_5       = ndcg_at_k(retrieved_ids, relevant_ids, 5),
        num_relevant    = len(relevant_ids),
        num_retrieved   = len(retrieved_ids),
    )


# ── Faithfulness eval (LLM-as-judge) ─────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are evaluating whether an answer is grounded in the 
provided context passages.

Context:
{context}

Answer:
{answer}

For each factual claim in the answer, determine if it is directly supported by 
the context. Return a JSON object:
{{
  "supported_claims": <int>,
  "total_claims": <int>,
  "faithfulness_score": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}}
Return ONLY the JSON, no other text."""


def evaluate_faithfulness(answer: str, context_texts: list[str],
                          llm_client=None, use_mock: bool = False) -> float:
    """
    LLM-as-judge faithfulness evaluation.
    Returns a score between 0.0 (not grounded) and 1.0 (fully grounded).
    """
    if use_mock or llm_client is None:
        # Mock: give a random-ish score based on citation presence
        import re
        citations = len(re.findall(r"\[SOURCE:", answer))
        return min(1.0, 0.5 + citations * 0.15)

    context_block = "\n\n".join(
        f"[{i+1}] {t}" for i, t in enumerate(context_texts[:5])
    )
    prompt = FAITHFULNESS_PROMPT.format(
        context=context_block, answer=answer
    )

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        raw  = response.choices[0].message.content.strip()
        data = json.loads(raw)
        return float(data.get("faithfulness_score", 0.0))
    except Exception:
        return 0.0


def compute_answer_metrics(answer_result: AnswerResult,
                           results: list[RetrievalResult],
                           llm_client=None,
                           use_mock: bool = False) -> AnswerMetrics:
    context_texts = [r.text for r in results]
    faithfulness  = evaluate_faithfulness(
        answer_result.answer, context_texts, llm_client, use_mock
    )

    # Citation coverage: citations / number of sentences in answer
    import re
    sentences = [s for s in re.split(r'[.!?]+', answer_result.answer) if s.strip()]
    n_cited   = len(answer_result.citations)
    coverage  = min(1.0, n_cited / max(len(sentences), 1))

    return AnswerMetrics(
        faithfulness      = faithfulness,
        answer_relevance  = 0.0,        # requires embedding comparison; logged as 0 in mock
        citation_coverage = coverage,
        num_citations     = n_cited,
    )


# ── MLflow logger ─────────────────────────────────────────────────────────────

class EvalLogger:
    """Logs eval results to MLflow."""

    def __init__(self, experiment_name: str = "finrag-eval",
                 use_mock: bool = False):
        self.use_mock        = use_mock
        self.experiment_name = experiment_name
        self._mlflow         = None
        if not use_mock:
            self._init_mlflow()

    def _init_mlflow(self):
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            print(f"✓ MLflow experiment: {self.experiment_name}")
        except ImportError:
            print("⚠ mlflow not installed — logging disabled")
            self.use_mock = True

    def log_eval(self, result: EvalResult) -> None:
        from datetime import datetime
        result.timestamp = datetime.utcnow().isoformat()

        if self.use_mock:
            r = result.retrieval_metrics
            a = result.answer_metrics
            print(
                f"[EVAL] {result.query_id} | {result.strategy} | "
                f"P@5={r.precision_at_5:.2f} R@10={r.recall_at_10:.2f} "
                f"MRR={r.mrr:.2f} faith={a.faithfulness:.2f} "
                f"lat={result.latency_ms:.0f}ms"
            )
            return

        with self._mlflow.start_run(run_name=f"{result.query_id}-{result.strategy}"):
            rm = result.retrieval_metrics
            am = result.answer_metrics
            self._mlflow.log_params({
                "query_id":  result.query_id,
                "strategy":  result.strategy,
                "model":     result.model,
                "query":     result.query[:100],
            })
            self._mlflow.log_metrics({
                "precision_at_1":   rm.precision_at_1,
                "precision_at_5":   rm.precision_at_5,
                "precision_at_10":  rm.precision_at_10,
                "recall_at_5":      rm.recall_at_5,
                "recall_at_10":     rm.recall_at_10,
                "mrr":              rm.mrr,
                "ndcg_at_5":        rm.ndcg_at_5,
                "faithfulness":     am.faithfulness,
                "citation_coverage":am.citation_coverage,
                "num_citations":    float(am.num_citations),
                "latency_ms":       result.latency_ms,
            })

    def log_batch(self, results: list[EvalResult]) -> None:
        for r in results:
            self.log_eval(r)
        self._print_summary(results)

    @staticmethod
    def _print_summary(results: list[EvalResult]) -> None:
        if not results:
            return
        n = len(results)
        avg = lambda attr: sum(getattr(r.retrieval_metrics, attr) for r in results) / n
        avg_a = lambda attr: sum(getattr(r.answer_metrics, attr) for r in results) / n
        print(f"\n── Eval summary ({n} queries) {'─'*30}")
        print(f"  Precision@5   : {avg('precision_at_5'):.3f}")
        print(f"  Recall@10     : {avg('recall_at_10'):.3f}")
        print(f"  MRR           : {avg('mrr'):.3f}")
        print(f"  NDCG@5        : {avg('ndcg_at_5'):.3f}")
        print(f"  Faithfulness  : {avg_a('faithfulness'):.3f}")
        print(f"  Latency (avg) : {sum(r.latency_ms for r in results)/n:.0f}ms")