"""
ground_truth.py
---------------
Builds a ground-truth evaluation set for retrieval quality measurement.

Each eval item contains:
  - query_id       : unique identifier
  - query          : natural language query
  - metadata_filter: SQL WHERE clause (structural ground truth)
  - relevant_ids   : set of chunk_ids that should be retrieved
  - strategy_tags  : which retrieval strategies this tests

Ground truth is derived programmatically from the corpus metadata
(no human annotation required) — a report is "relevant" if it
satisfies the structural criteria in the metadata filter.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import duckdb


@dataclass
class EvalQuery:
    query_id:         str
    query:            str
    metadata_filter:  str
    relevant_chunk_ids: set[str] = field(default_factory=set)
    description:      str = ""
    strategy_tags:    list[str] = field(default_factory=list)


# ── Eval query definitions ────────────────────────────────────────────────────

def build_eval_queries() -> list[EvalQuery]:
    """
    20 representative eval queries covering:
    - Simple sector/region filters
    - Rating change queries (the core use case)
    - Coverage-depth filters
    - Compound multi-filter queries
    - Adversarial / broad queries
    """
    six_mo   = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    one_yr   = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    return [
        # ── Rating change queries ─────────────────────────────────────────
        EvalQuery(
            query_id="Q001",
            query="Show me all emerging market buy recommendations from analysts "
                  "with 10+ years of coverage who changed their view in the last 6 months",
            metadata_filter=(
                f"region = 'Emerging Markets' "
                f"AND rating IN ('Buy', 'Strong Buy') "
                f"AND is_rating_change = TRUE "
                f"AND years_coverage >= 10 "
                f"AND rating_change_date >= '{six_mo}'"
            ),
            description="Core flagship query from the JD",
            strategy_tags=["metadata_filter", "rating_change", "hybrid"],
        ),
        EvalQuery(
            query_id="Q002",
            query="Which analysts downgraded energy stocks in the last year?",
            metadata_filter=(
                f"sector = 'Energy' "
                f"AND rating IN ('Hold', 'Sell', 'Strong Sell') "
                f"AND is_rating_change = TRUE "
                f"AND rating_change_date >= '{one_yr}'"
            ),
            description="Sector downgrade tracking",
            strategy_tags=["metadata_filter", "rating_change"],
        ),
        EvalQuery(
            query_id="Q003",
            query="Find research notes where analysts upgraded technology stocks to strong buy",
            metadata_filter=(
                "sector = 'Technology' "
                "AND rating = 'Strong Buy' "
                "AND is_rating_change = TRUE "
                "AND old_rating IN ('Buy', 'Hold', 'Sell', 'Strong Sell')"
            ),
            description="Strong buy upgrades in tech",
            strategy_tags=["metadata_filter", "rating_change"],
        ),
        EvalQuery(
            query_id="Q004",
            query="Analysts who reversed a sell recommendation on healthcare",
            metadata_filter=(
                "sector = 'Healthcare' "
                "AND is_rating_change = TRUE "
                "AND old_rating IN ('Sell', 'Strong Sell') "
                "AND rating IN ('Buy', 'Strong Buy', 'Hold')"
            ),
            description="Sell reversal in healthcare",
            strategy_tags=["metadata_filter", "rating_change"],
        ),
        # ── Coverage depth queries ────────────────────────────────────────
        EvalQuery(
            query_id="Q005",
            query="Research from senior analysts with over 15 years of coverage on financials",
            metadata_filter=(
                "sector = 'Financials' "
                "AND years_coverage >= 15"
            ),
            description="Senior analyst coverage filter",
            strategy_tags=["metadata_filter", "coverage_depth"],
        ),
        EvalQuery(
            query_id="Q006",
            query="Buy recommendations from analysts with deep emerging market expertise",
            metadata_filter=(
                "region = 'Emerging Markets' "
                "AND rating IN ('Buy', 'Strong Buy') "
                "AND years_coverage >= 10"
            ),
            description="EM expertise + bullish rating",
            strategy_tags=["metadata_filter", "coverage_depth"],
        ),
        # ── Sector + region compound ──────────────────────────────────────
        EvalQuery(
            query_id="Q007",
            query="Energy sector analysis focused on Asia Pacific markets",
            metadata_filter=(
                "sector = 'Energy' "
                "AND region = 'Asia Pacific'"
            ),
            description="Sector + region compound filter",
            strategy_tags=["metadata_filter", "compound"],
        ),
        EvalQuery(
            query_id="Q008",
            query="Technology research covering North America with bullish outlook",
            metadata_filter=(
                "sector = 'Technology' "
                "AND region = 'North America' "
                "AND rating IN ('Buy', 'Strong Buy')"
            ),
            description="Tech + NA + bullish",
            strategy_tags=["metadata_filter", "compound"],
        ),
        EvalQuery(
            query_id="Q009",
            query="Healthcare investments in Latin America or Middle East",
            metadata_filter=(
                "sector = 'Healthcare' "
                "AND region IN ('Latin America', 'Middle East & Africa')"
            ),
            description="Multi-region healthcare",
            strategy_tags=["metadata_filter", "compound"],
        ),
        EvalQuery(
            query_id="Q010",
            query="Industrials sector buy recommendations from European analysts",
            metadata_filter=(
                "sector = 'Industrials' "
                "AND region = 'Europe' "
                "AND rating IN ('Buy', 'Strong Buy')"
            ),
            description="EU industrials bullish",
            strategy_tags=["metadata_filter", "compound"],
        ),
        # ── Semantic + filter hybrid ──────────────────────────────────────
        EvalQuery(
            query_id="Q011",
            query="Analysts discussing margin expansion as a key driver in materials",
            metadata_filter="sector = 'Materials'",
            description="Semantic query within sector filter — tests hybrid retrieval",
            strategy_tags=["hybrid", "semantic"],
        ),
        EvalQuery(
            query_id="Q012",
            query="Research notes highlighting free cash flow generation in utilities",
            metadata_filter="sector = 'Utilities'",
            description="Semantic within sector",
            strategy_tags=["hybrid", "semantic"],
        ),
        EvalQuery(
            query_id="Q013",
            query="Analysts mentioning regulatory tailwinds for communication services",
            metadata_filter="sector = 'Communication Services'",
            description="Regulatory theme in comms sector",
            strategy_tags=["hybrid", "semantic"],
        ),
        EvalQuery(
            query_id="Q014",
            query="Reports discussing supply chain risks and their impact on consumer stocks",
            metadata_filter="sector = 'Consumer Discretionary'",
            description="Supply chain risk in consumer sector",
            strategy_tags=["hybrid", "semantic"],
        ),
        EvalQuery(
            query_id="Q015",
            query="Real estate analysts discussing rising interest rate sensitivity",
            metadata_filter="sector = 'Real Estate'",
            description="Rate sensitivity in REIT space",
            strategy_tags=["hybrid", "semantic"],
        ),
        # ── Broad / adversarial queries ───────────────────────────────────
        EvalQuery(
            query_id="Q016",
            query="What are the bullish theses across all sectors right now?",
            metadata_filter="rating IN ('Buy', 'Strong Buy')",
            description="Broad bullish query — tests recall",
            strategy_tags=["broad", "recall_test"],
        ),
        EvalQuery(
            query_id="Q017",
            query="Analysts who are most bearish and have high coverage years",
            metadata_filter=(
                "rating IN ('Sell', 'Strong Sell') "
                "AND years_coverage >= 12"
            ),
            description="Senior bear case analysts",
            strategy_tags=["metadata_filter", "bearish"],
        ),
        EvalQuery(
            query_id="Q018",
            query="Emerging market sell recommendations from this year",
            metadata_filter=(
                "region = 'Emerging Markets' "
                "AND rating IN ('Sell', 'Strong Sell')"
            ),
            description="EM bearish",
            strategy_tags=["metadata_filter"],
        ),
        EvalQuery(
            query_id="Q019",
            query="Which sectors have seen the most rating changes recently?",
            metadata_filter=(
                f"is_rating_change = TRUE "
                f"AND rating_change_date >= '{one_yr}'"
            ),
            description="Recent rating flux — broad",
            strategy_tags=["metadata_filter", "broad"],
        ),
        EvalQuery(
            query_id="Q020",
            query="Summarise the bull case for semiconductor and technology stocks "
                  "from analysts who cover both Asia and North America",
            metadata_filter=(
                "sector = 'Technology' "
                "AND rating IN ('Buy', 'Strong Buy')"
            ),
            description="GraphRAG use case — cross-region analyst",
            strategy_tags=["graph_rag", "semantic", "hybrid"],
        ),
    ]


# ── Ground truth builder ──────────────────────────────────────────────────────

def build_ground_truth(eval_queries: list[EvalQuery],
                       db_path:      str,
                       chunks_path:  str) -> list[EvalQuery]:
    """
    Populate relevant_chunk_ids for each eval query using DuckDB.
    A chunk is relevant if its parent report matches the metadata filter.
    """
    con = duckdb.connect(db_path, read_only=True)

    # Build report_id → chunk_ids mapping
    with open(chunks_path) as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    report_to_chunks: dict[str, list[str]] = {}
    for c in all_chunks:
        report_to_chunks.setdefault(c["report_id"], []).append(c["chunk_id"])

    for eq in eval_queries:
        try:
            rows = con.execute(
                f"SELECT report_id FROM reports WHERE {eq.metadata_filter}"
            ).fetchall()
            relevant_report_ids = {r[0] for r in rows}

            # Expand to chunk_ids
            eq.relevant_chunk_ids = set()
            for rid in relevant_report_ids:
                eq.relevant_chunk_ids.update(report_to_chunks.get(rid, []))

        except Exception as e:
            print(f"⚠ Error building ground truth for {eq.query_id}: {e}")
            eq.relevant_chunk_ids = set()

    con.close()
    return eval_queries


def save_ground_truth(eval_queries: list[EvalQuery], output_path: str) -> None:
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for eq in eval_queries:
            row = {
                "query_id":           eq.query_id,
                "query":              eq.query,
                "metadata_filter":    eq.metadata_filter,
                "relevant_chunk_ids": list(eq.relevant_chunk_ids),
                "description":        eq.description,
                "strategy_tags":      eq.strategy_tags,
            }
            f.write(json.dumps(row) + "\n")
    print(f"✓ Saved {len(eval_queries)} eval queries → {output_path}")


def load_ground_truth(path: str) -> list[EvalQuery]:
    queries = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            queries.append(EvalQuery(
                query_id          = d["query_id"],
                query             = d["query"],
                metadata_filter   = d["metadata_filter"],
                relevant_chunk_ids= set(d["relevant_chunk_ids"]),
                description       = d.get("description", ""),
                strategy_tags     = d.get("strategy_tags", []),
            ))
    return queries


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = build_eval_queries()
    queries = build_ground_truth(
        queries,
        db_path     = "data/finrag.duckdb",
        chunks_path = "data/processed/chunks.jsonl",
    )
    save_ground_truth(queries, "data/eval/ground_truth.jsonl")

    print("\nGround truth summary:")
    for q in queries:
        print(f"  {q.query_id} [{', '.join(q.strategy_tags)}] "
              f"→ {len(q.relevant_chunk_ids)} relevant chunks")