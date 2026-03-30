# FinRAG: Hybrid Retrieval System for Financial Research Archives

Production-grade RAG pipeline over 30+ years of institutional financial research – combining sematic search, structured metadata filtering, cross-encoder re-ranking, and advacned retrieval strategies (HyDE, query expansion, GraphRAG) with a full obsrrvability and evaluation layer.

## The Problem
Financial research archives are full of high-signal data clients can't actually reach. A junior analyst wanting to find "all emerging market buy recommendations from analysts with 10+ years of coverage who changed their view in the last 6months" has to manually trawl spreadsheets, PDFs, and distribution platfrom exports – none of which talk to each other.

FinRAG solved by treating the research archive as queryable knowledge layer; ingest once, retrieve with precision, answer with grounding.

## Architecture
┌─────────────────────────────────────────────────────────────┐
│                        Data Sources                         │
│  Research PDFs  ·  Analyst Metadata  ·  Rec. Change Logs    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              Ingestion & Chunking Pipeline                  │
│  Sentence-window chunking · Metadata tagging · Airflow DAG  │
│  Embedding generation (BAAI/bge-large-en-v1.5)              │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
┌──────────────▼──────────┐  ┌────────────▼───────────────────┐
│    Vector Store          │  │   Structured Metadata Store    │
│    Qdrant                │  │   DuckDB                       │
│    Dense embeddings      │  │   analyst · sector · date      │
│    Payload filters       │  │   region · rating · coverage   │
└──────────────┬──────────┘  └────────────┬───────────────────┘
               │                          │
┌──────────────▼──────────────────────────▼───────────────────┐
│                    Hybrid Retrieval Layer                   │
│                                                             │
│  Vector search (Qdrant)  +  BM25 keyword  →  RRF fusion     │
│  Metadata pre-filter (DuckDB)                               │
│  Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2)          │
│                                                             │
│  Advanced strategies:                                       │
│  · HyDE — hypothetical document embedding                   │
│  · Query expansion — LLM rewrites → merged results          │
│  · GraphRAG — analyst × sector × company relationship graph │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   LLM Interface                             │
│  GPT-4o / Claude Sonnet  ·  Grounded answers + citations    │
│  Research summarization  ·  Analyst Q&A  ·  Rec tracking    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│               Observability & Eval (MLflow)                 │
│  Retrieval precision@5  ·  Recall@10  ·  Answer faithfulness│
│  Embedding freshness  ·  Latency alerts  ·  RAGAS metrics   │
└─────────────────────────────────────────────────────────────┘

## Key Features
### Hybrid Retrieval
Combines dense vector search (semantic similarity) with BM25 sparse search (keyword matching), fused via Reciprocal Rank Fusion. This outperforms either approach where exact ticker symbols and analyst names matter as much as semantic meaning.

### Structure Metadata Filtering
DuckDB acts as the structures metadata layer. Befor hitting the vector store, queries like "analysts with 10+ years covarage" or "recommendations changed in last 6 months
 are resolves as exact SQL filters – eliminating irrelevant chunka before semantic scring begins.

 ### Cross-Encoder Re-ranking
 Top-20 candidates from the hybrid retrieval pass are re-scored by a cross-encoder (cross-encoder/ms-macro-MiniLM-L-6-v2) for precise relevance ranking. Bi-encoder retrieval is fast but coarse; cross-encoder re-ranking is slow but highly accurate – this pipeline gets both.

 ## HyDE (Hypothetical Document Embeddings)
For underspecified queries, the LLM generate a hypothetical "ideal" research note that would answer thw query, embeds it, and uses that embedding for retrieval. Substantially improves recall on abstract or broad queries.

### Query Expansion
The LLM rewrites each incoming query into 3 diverse variants (different phrasings, synonyms, related angles). Results from all variants are merged and deduplicated before re-ranking. Reduces sensitivity to exact wording.

## GraphRAG – Anlayst Relationship Mapping
A NetworkX graph encodes Analyst -> Sector -> Company -> Coverage history relationships. Graph traversal enriches retrieval results: a query about a sector also surfaces reports from analysts who cross-cover related sectors, even if the exact keyword match is weak.

## Evaluation Framework
Every retrieval run is logged to MLflow with:
- precision@5 and recall@10 against a ground-truth eval set
- Answer faithfulness via an LLM judge (RAGAS faithfulness score)
- Embeddig freshness (time since last upsert per document)
- p50 / p95 retrieval and LLM call latency

## Repo Structure
<img width="744" height="1076" alt="image" src="https://github.com/user-attachments/assets/dd0e480c-1885-4b04-b535-8ae4d06d1410" />

## Quickstart
**Prerequisites**: Docker, Python 3.11+, OpenAI or Anthropic API key.
bash# Clone and install
git clone https://github.com/sajanshergill/finrag-research-intelligence
cd finrag-research-intelligence
pip install -e ".[dev]"

### Start infrastructure (Qdrant, Airflow, MLflow)
docker compose up -d

### Generate synthetic research corpus (500 analyst reports)
python src/ingestion/synthetic_data.py --count 500 --output data/raw/

### Run ingestion pipeline
python src/ingestion/embedder.py --input data/raw/ --collection finrag

### Launch the Streamlit demo
streamlit run app/streamlit_app.py

## Example Queries
The system is designed to handle faceted financial queries that combine semantic intent with structured constraints:
"Show me all emerging market buy
recommendations from analysts with
10+ years of coverage who changed
their view in the last 6 months"

"Summarise the bull case for
semiconductor stocks from analysts who
cover both Asia and North America"

"Which analysts reversed a sell
recommendation on energy stocks
after Q3 earnings season?"

"Find research notes where the analyst
significantly raised their
price target alongside a rating 
upgrade"


## Retrieval Strategy Comparison
<img width="968" height="634" alt="image" src="https://github.com/user-attachments/assets/c63d394e-ab06-4764-a026-afdd0dd98e6d" />

Benchmarked on 200 synthetic ground-truth query-document pairs.

## Tech Stack
<img width="584" height="1302" alt="image" src="https://github.com/user-attachments/assets/a5a0bbac-55a1-41a9-ac48-82552660cceb" />

## DataOps Practices
- All pipeline runs versioned and logged to MLflow
- Embedding freshness tracked per document – state chunks trigger re-ingestion
- CI/CD via Github Actions: lint -> unit tests -> integration test against Qdrant Docker
- Environment parity: same docker-compose.yml for dev, staging, and production
- IaC-ready: AWS CDK stacks for App Runner + EventBridge deployment (see infra/)

## Roadmap
- AWS EventBridge integration for event-driven re-ingestion on new research publish
- Structured metadata retrieval for faceted search UI (filter by analyst, sector, date range)
- GraphRAG expansion: multi-hop reasoning across analyst -> company -> sector nodes
- A/B evaluation framework: compare retrieval strategies on live query traffic
- Streaming LLM responses in the Streamlit UI




