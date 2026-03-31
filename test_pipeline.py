"""
Tests for FinRAG pipeline components.
Run: pytest tests/ -v
"""

import json
import os
import sys

import pytest

# Ensure we import this repo's `src/` package (avoid collisions with any
# unrelated `~/src` directory on the machine).
sys.path.insert(0, os.path.dirname(__file__))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_reports():
    from src.ingestion.synthetic_data import generate_report
    return [generate_report(i) for i in range(1, 21)]


@pytest.fixture(scope="session")
def sample_chunks(sample_reports):
    from src.ingestion.chunker import chunk_report
    chunks = []
    for r in sample_reports:
        chunks.extend(chunk_report(r))
    return chunks


# ── Synthetic data tests ──────────────────────────────────────────────────────

class TestSyntheticData:
    def test_report_has_required_fields(self, sample_reports):
        required = [
            "report_id", "analyst_name", "company", "ticker",
            "sector", "region", "rating", "publication_date",
            "years_coverage", "body",
        ]
        for r in sample_reports:
            for field in required:
                assert field in r, f"Missing field: {field}"

    def test_rating_values_valid(self, sample_reports):
        valid = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
        for r in sample_reports:
            assert r["rating"] in valid

    def test_rating_change_consistency(self, sample_reports):
        for r in sample_reports:
            if r["is_rating_change"]:
                assert r["old_rating"] is not None
                assert r["rating_change_date"] is not None
                assert r["old_rating"] != r["rating"]

    def test_body_is_non_empty(self, sample_reports):
        for r in sample_reports:
            assert len(r["body"]) > 100

    def test_years_coverage_positive(self, sample_reports):
        for r in sample_reports:
            assert r["years_coverage"] > 0


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunks_produced(self, sample_chunks):
        assert len(sample_chunks) > 0

    def test_chunk_has_required_fields(self, sample_chunks):
        required = [
            "chunk_id", "report_id", "window_index", "text",
            "analyst_name", "sector", "region", "rating",
        ]
        for c in sample_chunks:
            d = c if isinstance(c, dict) else vars(c)
            for field in required:
                assert field in d

    def test_chunk_ids_unique(self, sample_chunks):
        ids = [c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]
               for c in sample_chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_text_min_length(self, sample_chunks):
        for c in sample_chunks:
            text = c.text if hasattr(c, "text") else c["text"]
            assert len(text) >= 80

    def test_window_index_sequential(self, sample_chunks):
        from collections import defaultdict
        by_report = defaultdict(list)
        for c in sample_chunks:
            rid = c.report_id if hasattr(c, "report_id") else c["report_id"]
            idx = c.window_index if hasattr(c, "window_index") else c["window_index"]
            by_report[rid].append(idx)
        for rid, indices in by_report.items():
            assert sorted(indices) == list(range(len(indices)))

    def test_metadata_propagated(self, sample_reports, sample_chunks):
        report_map = {r["report_id"]: r for r in sample_reports}
        for c in sample_chunks:
            if isinstance(c, dict):
                rid, sector = c["report_id"], c["sector"]
            else:
                rid, sector = c.report_id, c.sector
            assert sector == report_map[rid]["sector"]


# ── BM25 retrieval tests ──────────────────────────────────────────────────────

class TestBM25Retrieval:
    @pytest.fixture(scope="class")
    def bm25_index(self, tmp_path_factory, sample_chunks):
        from dataclasses import asdict

        from src.retrieval.hybrid import BM25Index

        tmp = tmp_path_factory.mktemp("data")
        chunks_path = tmp / "chunks.jsonl"
        with open(chunks_path, "w") as f:
            for c in sample_chunks:
                row = asdict(c) if hasattr(c, "__dataclass_fields__") else c
                f.write(json.dumps(row) + "\n")
        return BM25Index(str(chunks_path))

    def test_index_loads(self, bm25_index):
        assert len(bm25_index.chunks) > 0

    def test_search_returns_results(self, bm25_index):
        results = bm25_index.search("buy recommendation emerging market", top_k=5)
        assert len(results) > 0

    def test_results_sorted_by_score(self, bm25_index):
        results = bm25_index.search("analyst rating change", top_k=10)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_filter_by_report_ids(self, bm25_index):
        allowed = {bm25_index.chunks[0]["report_id"]}
        results = bm25_index.search("buy", top_k=10, allowed_report_ids=allowed)
        for idx, _ in results:
            assert bm25_index.chunks[idx]["report_id"] in allowed

    def test_empty_query_safe(self, bm25_index):
        results = bm25_index.search("", top_k=5)
        # Should not raise; may return empty or low-scored results
        assert isinstance(results, list)


# ── RRF fusion tests ──────────────────────────────────────────────────────────

class TestRRFFusion:
    def test_rrf_scores_positive(self, sample_chunks):
        from dataclasses import asdict

        from src.retrieval.hybrid import reciprocal_rank_fusion

        chunks = [asdict(c) if hasattr(c, "__dataclass_fields__") else c
                  for c in sample_chunks[:10]]
        bm25_results = [(i, float(10 - i)) for i in range(10)]
        results = reciprocal_rank_fusion([], bm25_results, chunks, top_n=5)
        assert all(r.rrf_score > 0 for r in results)

    def test_rrf_sorted_descending(self, sample_chunks):
        from dataclasses import asdict

        from src.retrieval.hybrid import reciprocal_rank_fusion

        chunks = [asdict(c) if hasattr(c, "__dataclass_fields__") else c
                  for c in sample_chunks[:10]]
        bm25_results = [(i, float(10 - i)) for i in range(10)]
        results = reciprocal_rank_fusion([], bm25_results, chunks, top_n=5)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_top_n_respected(self, sample_chunks):
        from dataclasses import asdict

        from src.retrieval.hybrid import reciprocal_rank_fusion

        chunks = [asdict(c) if hasattr(c, "__dataclass_fields__") else c
                  for c in sample_chunks[:20]]
        bm25_results = [(i, float(20 - i)) for i in range(20)]
        results = reciprocal_rank_fusion([], bm25_results, chunks, top_n=7)
        assert len(results) <= 7


# ── Query expansion tests ─────────────────────────────────────────────────────

class TestQueryExpansion:
    def test_retrieve_expanded_returns_fused_results(self, sample_chunks, tmp_path):
        from dataclasses import asdict

        from src.retrieval.hybrid import BM25Index
        from src.retrieval.query_expansion import QueryExpander

        chunks_path = tmp_path / "chunks.jsonl"
        with open(chunks_path, "w") as f:
            for c in sample_chunks:
                row = asdict(c) if hasattr(c, "__dataclass_fields__") else c
                f.write(json.dumps(row) + "\n")

        index = BM25Index(str(chunks_path))

        class _Retriever:
            def __init__(self):
                self.bm25_index = index

        expander = QueryExpander(use_mock=True, n_variants=2)
        fused = expander.retrieve_expanded(
            "buy recommendation emerging market",
            _Retriever(),
            final_top_n=5,
        )
        assert isinstance(fused, list)
        assert len(fused) <= 5
        assert all(hasattr(r, "chunk_id") and hasattr(r, "rrf_score") for r in fused)


# ── Re-ranker tests ───────────────────────────────────────────────────────────

class TestReranker:
    def test_mock_reranker_sorts(self, sample_chunks):
        from dataclasses import asdict

        from src.retrieval.hybrid import reciprocal_rank_fusion
        from src.retrieval.reranker import CrossEncoderReranker

        chunks = [asdict(c) if hasattr(c, "__dataclass_fields__") else c
                  for c in sample_chunks[:10]]
        bm25_results = [(i, float(10 - i)) for i in range(10)]
        fused = reciprocal_rank_fusion([], bm25_results, chunks, top_n=10)

        reranker = CrossEncoderReranker(use_mock=True)
        top = reranker.rerank("buy recommendation", fused, top_k=5)

        assert len(top) <= 5
        scores = [r.rerank_score for r in top if r.rerank_score is not None]
        assert scores == sorted(scores, reverse=True)

    def test_reranker_empty_input(self):
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(use_mock=True)
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


# ── Eval metrics tests ────────────────────────────────────────────────────────

class TestEvalMetrics:
    def test_precision_at_k_perfect(self):
        from src.eval.metrics import precision_at_k
        ids = ["a", "b", "c", "d", "e"]
        rel = {"a", "b", "c", "d", "e"}
        assert precision_at_k(ids, rel, 5) == 1.0

    def test_precision_at_k_zero(self):
        from src.eval.metrics import precision_at_k
        ids = ["a", "b", "c"]
        rel = {"x", "y", "z"}
        assert precision_at_k(ids, rel, 3) == 0.0

    def test_recall_at_k(self):
        from src.eval.metrics import recall_at_k
        ids = ["a", "b", "c", "d", "e"]
        rel = {"a", "b"}
        assert recall_at_k(ids, rel, 5) == 1.0
        assert recall_at_k(ids, rel, 1) == 0.5

    def test_mrr_first_hit(self):
        from src.eval.metrics import mean_reciprocal_rank
        ids = ["x", "a", "y"]
        rel = {"a"}
        assert mean_reciprocal_rank(ids, rel) == pytest.approx(0.5)

    def test_ndcg_perfect(self):
        from src.eval.metrics import ndcg_at_k
        ids = ["a", "b", "c"]
        rel = {"a", "b", "c"}
        assert ndcg_at_k(ids, rel, 3) == pytest.approx(1.0)

    def test_faithfulness_mock(self):
        from src.eval.metrics import evaluate_faithfulness
        answer = "The analyst upgraded to Buy [SOURCE: RPT-00001_W001]."
        score  = evaluate_faithfulness(answer, ["context text"], use_mock=True)
        assert 0.0 <= score <= 1.0


# ── LLM interface tests ───────────────────────────────────────────────────────

class TestLLMInterface:
    def test_mock_answer_returns_result(self, sample_chunks):
        from dataclasses import asdict

        from src.llm.interface import FinRAGInterface
        from src.retrieval.hybrid import reciprocal_rank_fusion

        chunks = [asdict(c) if hasattr(c, "__dataclass_fields__") else c
                  for c in sample_chunks[:5]]
        bm25_results = [(i, float(5 - i)) for i in range(5)]
        fused = reciprocal_rank_fusion([], bm25_results, chunks, top_n=5)

        iface  = FinRAGInterface(use_mock=True)
        result = iface.answer("buy recommendations", fused)

        assert result.answer
        assert result.query == "buy recommendations"
        assert isinstance(result.citations, list)

    def test_citations_parsed(self, sample_chunks):
        from src.llm.interface import parse_citations
        from src.retrieval.hybrid import RetrievalResult

        # Build a minimal RetrievalResult
        chunk = (sample_chunks[0] if isinstance(sample_chunks[0], dict)
                 else vars(sample_chunks[0]))
        result = RetrievalResult(
            chunk_id=chunk["chunk_id"], report_id=chunk["report_id"],
            text=chunk["text"], analyst_name=chunk["analyst_name"],
            company=chunk["company"], sector=chunk["sector"],
            region=chunk["region"], rating=chunk["rating"],
            old_rating=chunk.get("old_rating"),
            is_rating_change=chunk["is_rating_change"],
            rating_change_date=chunk.get("rating_change_date"),
            years_coverage=chunk["years_coverage"],
            publication_date=chunk["publication_date"],
            target_price=chunk["target_price"],
        )
        answer    = f"Based on research [SOURCE: {chunk['chunk_id']}] we see..."
        citations = parse_citations(answer, [result])
        assert len(citations) == 1
        assert citations[0].chunk_id == chunk["chunk_id"]