"""
streamlit_app.py
----------------
FinRAG interactive demo.

Features:
  - Natural language query input
  - Metadata filter sidebar (sector, region, years coverage, rating change)
  - Strategy selector (Hybrid, HyDE, Query Expansion)
  - Retrieved chunks displayed with metadata badges
  - Re-ranking scores shown
  - LLM-generated grounded answer with citations
  - Latency breakdown

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import os
import json
import time

# Ensure we import this repo's `src/` package (avoid collisions with any
# unrelated `~/src` directory on the machine).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "FinRAG — Financial Research Intelligence",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_bm25_index():
    from src.retrieval.hybrid import BM25Index
    return BM25Index("data/processed/chunks.jsonl")

@st.cache_resource
def load_graph():
    from src.retrieval.graph_rag import AnalystGraph
    g = AnalystGraph()
    g.build_from_corpus("data/raw/corpus.jsonl")
    return g

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Query Settings")

    st.subheader("Metadata Filters")

    sectors = st.multiselect("Sector", [
        "Technology", "Energy", "Healthcare", "Financials",
        "Consumer Discretionary", "Industrials", "Materials",
        "Real Estate", "Utilities", "Communication Services",
    ])

    regions = st.multiselect("Region", [
        "North America", "Europe", "Emerging Markets",
        "Asia Pacific", "Latin America", "Middle East & Africa",
    ])

    ratings = st.multiselect("Rating", [
        "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"
    ])

    min_coverage = st.slider("Min. years coverage", 0, 30, 0)

    rating_change_only = st.checkbox("Rating changes only", value=False)

    changed_after = st.date_input("Changed after (optional)", value=None)

    st.divider()
    st.subheader("Retrieval Strategy")
    strategy = st.radio("Strategy", [
        "Hybrid (BM25 + Dense)",
        "Query Expansion",
        "HyDE",
    ])

    top_k = st.slider("Top-K results", 3, 10, 5)

    st.divider()
    st.subheader("LLM")
    use_mock_llm = st.checkbox("Mock LLM (no API key needed)", value=True)


# ── Build metadata filter ─────────────────────────────────────────────────────

def build_filter() -> str | None:
    from src.ingestion.metadata_loader import build_metadata_filter
    return build_metadata_filter(
        sectors              = sectors or None,
        regions              = regions or None,
        min_years_coverage   = min_coverage if min_coverage > 0 else None,
        ratings              = ratings or None,
        is_rating_change     = True if rating_change_only else None,
        changed_after        = changed_after.isoformat() if changed_after else None,
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("📊 FinRAG — Financial Research Intelligence")
st.caption(
    "Hybrid RAG pipeline over a synthetic financial research archive. "
    "Combines semantic search, BM25 keyword matching, structured metadata filtering, "
    "cross-encoder re-ranking, and LLM answer generation."
)

# Example queries
st.markdown("**Example queries:**")
example_cols = st.columns(3)
examples = [
    "Buy recommendations in emerging markets from senior analysts",
    "Analysts discussing margin expansion in technology",
    "Healthcare stocks where analysts reversed a sell recommendation",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["query"] = ex

query = st.text_input(
    "Enter your research query",
    value=st.session_state.get("query", ""),
    placeholder="e.g. emerging market buy recommendations from analysts with 10+ years coverage",
)

run_btn = st.button("🔍 Search", type="primary", use_container_width=False)

if run_btn and query.strip():
    metadata_filter = build_filter()

    t_total_start = time.time()

    # ── Retrieval ─────────────────────────────────────────────────────────
    with st.spinner("Retrieving relevant research..."):
        index = load_bm25_index()

        t_ret_start = time.time()

        if strategy == "Query Expansion":
            from src.retrieval.query_expansion import QueryExpander

            class _MockRetriever:
                def __init__(self): self.bm25_index = index

            expander = QueryExpander(use_mock=True, n_variants=3)
            with st.expander("🔎 Expanded queries", expanded=False):
                variants = expander.expand(query)
                for v in variants:
                    st.markdown(f"- {v}")
            fused = expander.retrieve_expanded(query, _MockRetriever(),
                                               final_top_n=top_k * 4)
        elif strategy == "HyDE":
            from src.retrieval.hyde import HyDERetriever
            hyde = HyDERetriever(use_mock=True)
            hyp_doc = hyde.generate_hypothetical_doc(query)
            with st.expander("📝 Hypothetical document (HyDE)", expanded=False):
                st.markdown(f"*{hyp_doc}*")
            # Fall back to BM25 with hypothetical doc as query
            bm25_hits = index.search(hyp_doc, top_k=top_k * 4)
            from src.retrieval.hybrid import reciprocal_rank_fusion
            fused = reciprocal_rank_fusion([], bm25_hits, index.chunks, top_n=top_k * 4)
        else:
            bm25_hits = index.search(query, top_k=top_k * 4)
            from src.retrieval.hybrid import reciprocal_rank_fusion
            fused = reciprocal_rank_fusion([], bm25_hits, index.chunks, top_n=top_k * 4)

        # Apply metadata post-filter (simulate pre-filter without Qdrant)
        if metadata_filter and metadata_filter != "1=1":
            import duckdb
            con = duckdb.connect("data/finrag.duckdb", read_only=True)
            allowed = {
                r[0] for r in
                con.execute(f"SELECT report_id FROM reports WHERE {metadata_filter}").fetchall()
            }
            con.close()
            fused = [r for r in fused if r.report_id in allowed]

        ret_latency = (time.time() - t_ret_start) * 1000

    # ── Re-ranking ────────────────────────────────────────────────────────
    with st.spinner("Re-ranking..."):
        t_rerank_start = time.time()
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(use_mock=True)
        top_results = reranker.rerank(query, fused, top_k=top_k)
        rerank_latency = (time.time() - t_rerank_start) * 1000

    # ── LLM answer ────────────────────────────────────────────────────────
    with st.spinner("Generating grounded answer..."):
        t_llm_start = time.time()
        from src.llm.interface import FinRAGInterface
        iface  = FinRAGInterface(use_mock=use_mock_llm)
        answer = iface.answer(query, top_results, latency_ms=ret_latency + rerank_latency)
        llm_latency = (time.time() - t_llm_start) * 1000

    total_latency = (time.time() - t_total_start) * 1000

    # ── Results display ───────────────────────────────────────────────────
    st.divider()

    # Latency breakdown
    lat_cols = st.columns(4)
    lat_cols[0].metric("Retrieval",  f"{ret_latency:.0f}ms")
    lat_cols[1].metric("Re-ranking", f"{rerank_latency:.0f}ms")
    lat_cols[2].metric("LLM",        f"{llm_latency:.0f}ms")
    lat_cols[3].metric("Total",      f"{total_latency:.0f}ms")

    st.divider()

    col_left, col_right = st.columns([1.2, 1])

    # ── Left: retrieved chunks ────────────────────────────────────────────
    with col_left:
        st.subheader(f"📑 Retrieved Chunks ({len(top_results)})")

        if not top_results:
            st.warning("No results matched your query and filters.")
        else:
            for i, r in enumerate(top_results, 1):
                with st.expander(
                    f"{i}. **{r.company}** — {r.analyst_name} "
                    f"| {r.sector} / {r.region}",
                    expanded=i == 1,
                ):
                    badge_cols = st.columns(4)
                    badge_cols[0].markdown(
                        f"<span style='background:#0068c9;color:white;padding:2px 8px;"
                        f"border-radius:4px;font-size:12px'>{r.rating}</span>",
                        unsafe_allow_html=True,
                    )
                    if r.is_rating_change and r.old_rating:
                        badge_cols[1].markdown(
                            f"<span style='background:#ff4b4b;color:white;padding:2px 8px;"
                            f"border-radius:4px;font-size:12px'>"
                            f"↑ {r.old_rating} → {r.rating}</span>",
                            unsafe_allow_html=True,
                        )
                    badge_cols[2].caption(f"📅 {r.publication_date}")
                    badge_cols[3].caption(f"🎓 {r.years_coverage}y coverage")

                    st.markdown(f"> {r.text}")

                    score_cols = st.columns(3)
                    if r.dense_rank:
                        score_cols[0].caption(f"Dense rank: #{r.dense_rank}")
                    if r.bm25_rank:
                        score_cols[1].caption(f"BM25 rank: #{r.bm25_rank}")
                    if r.rerank_score:
                        score_cols[2].caption(f"Rerank score: {r.rerank_score:.4f}")

    # ── Right: LLM answer ─────────────────────────────────────────────────
    with col_right:
        st.subheader("🤖 Grounded Answer")
        st.markdown(answer.answer)

        if answer.citations:
            st.subheader(f"📎 Citations ({len(answer.citations)})")
            for c in answer.citations:
                st.markdown(
                    f"**{c.analyst}** — {c.company} ({c.sector} / {c.region})\n\n"
                    f"Rating: `{c.rating}` | Published: {c.pub_date}\n\n"
                    f"> {c.text_snippet}..."
                )
                st.divider()

        # GraphRAG context
        if sectors or regions:
            st.subheader("🕸️ Related Analysts (GraphRAG)")
            try:
                graph    = load_graph()
                analysts = graph.expand_query_context(
                    sectors=sectors or None,
                    regions=regions or None,
                    min_reports=1,
                )
                if analysts:
                    for a in analysts[:5]:
                        st.caption(f"· {a}")
                else:
                    st.caption("No related analysts found via graph traversal.")
            except Exception:
                st.caption("GraphRAG unavailable.")

elif run_btn:
    st.warning("Please enter a query.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "FinRAG · Built by Sajan Singh Shergill · "
    "Stack: Qdrant · DuckDB · BM25 · CrossEncoder · Airflow · MLflow · Streamlit"
)