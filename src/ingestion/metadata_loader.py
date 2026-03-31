"""
metadata_loader.py
------------------
Loads structured report metadata into DuckDB.

DuckDB acts as the structured metadata layer for pre-filtering before
vector retrieval. SQL filters on analyst, sector, region, date, and
rating are applied here — eliminating irrelevant documents before
semantic scoring begins.

Schema design:
  reports    — one row per research report (source of truth)
  analysts   — deduplicated analyst profiles with coverage stats
  chunks     — chunk registry linking chunk_id → report_id

Usage:
    python src/ingestion/metadata_loader.py \
        --corpus data/raw/corpus.jsonl \
        --chunks data/processed/chunks.jsonl \
        --db data/finrag.duckdb
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import duckdb


# ── Schema DDL ────────────────────────────────────────────────────────────────

DDL_REPORTS = """
CREATE TABLE IF NOT EXISTS reports (
    report_id           VARCHAR PRIMARY KEY,
    analyst_name        VARCHAR NOT NULL,
    company             VARCHAR NOT NULL,
    ticker              VARCHAR NOT NULL,
    sector              VARCHAR NOT NULL,
    region              VARCHAR NOT NULL,
    rating              VARCHAR NOT NULL,
    old_rating          VARCHAR,
    is_rating_change    BOOLEAN DEFAULT FALSE,
    rating_change_date  DATE,
    target_price        INTEGER,
    publication_date    DATE NOT NULL,
    years_coverage      INTEGER NOT NULL,
    word_count          INTEGER
);
"""

DDL_ANALYSTS = """
CREATE TABLE IF NOT EXISTS analysts (
    analyst_name        VARCHAR PRIMARY KEY,
    total_reports       INTEGER DEFAULT 0,
    sectors_covered     VARCHAR[],
    regions_covered     VARCHAR[],
    years_coverage      INTEGER DEFAULT 0,
    latest_report_date  DATE,
    rating_changes      INTEGER DEFAULT 0
);
"""

DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            VARCHAR PRIMARY KEY,
    report_id           VARCHAR NOT NULL REFERENCES reports(report_id),
    window_index        INTEGER,
    sentence_count      INTEGER,
    text_preview        VARCHAR    -- first 200 chars for quick inspection
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_reports_analyst   ON reports(analyst_name);",
    "CREATE INDEX IF NOT EXISTS idx_reports_sector    ON reports(sector);",
    "CREATE INDEX IF NOT EXISTS idx_reports_region    ON reports(region);",
    "CREATE INDEX IF NOT EXISTS idx_reports_rating    ON reports(rating);",
    "CREATE INDEX IF NOT EXISTS idx_reports_change    ON reports(is_rating_change);",
    "CREATE INDEX IF NOT EXISTS idx_reports_pubdate   ON reports(publication_date);",
    "CREATE INDEX IF NOT EXISTS idx_reports_coverage  ON reports(years_coverage);",
    "CREATE INDEX IF NOT EXISTS idx_chunks_report     ON chunks(report_id);",
]


# ── Loader ────────────────────────────────────────────────────────────────────

class MetadataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.con.execute(DDL_REPORTS)
        self.con.execute(DDL_ANALYSTS)
        self.con.execute(DDL_CHUNKS)
        for idx in DDL_INDEXES:
            self.con.execute(idx)
        self.con.commit()

    def load_reports(self, corpus_path: str) -> int:
        with open(corpus_path) as f:
            reports = [json.loads(line) for line in f if line.strip()]

        rows = []
        for r in reports:
            rows.append((
                r["report_id"],
                r["analyst_name"],
                r["company"],
                r["ticker"],
                r["sector"],
                r["region"],
                r["rating"],
                r.get("old_rating"),
                r["is_rating_change"],
                r.get("rating_change_date"),
                r["target_price"],
                r["publication_date"][:10],
                r["years_coverage"],
                r.get("word_count"),
            ))

        self.con.executemany("""
            INSERT OR REPLACE INTO reports VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        self.con.commit()
        return len(rows)

    def load_chunks(self, chunks_path: str) -> int:
        with open(chunks_path) as f:
            chunks = [json.loads(line) for line in f if line.strip()]

        rows = [(
            c["chunk_id"],
            c["report_id"],
            c["window_index"],
            c["sentence_count"],
            c["text"][:200],
        ) for c in chunks]

        self.con.executemany("""
            INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)
        """, rows)
        self.con.commit()
        return len(rows)

    def build_analyst_profiles(self):
        """Aggregate analyst-level stats from the reports table."""
        self.con.execute("DELETE FROM analysts;")
        self.con.execute("""
            INSERT INTO analysts
            SELECT
                analyst_name,
                COUNT(*)                                    AS total_reports,
                ARRAY_AGG(DISTINCT sector)                  AS sectors_covered,
                ARRAY_AGG(DISTINCT region)                  AS regions_covered,
                MAX(years_coverage)                         AS years_coverage,
                MAX(publication_date)                       AS latest_report_date,
                SUM(CASE WHEN is_rating_change THEN 1 ELSE 0 END) AS rating_changes
            FROM reports
            GROUP BY analyst_name
        """)
        self.con.commit()

    def query(self, sql: str):
        return self.con.execute(sql).fetchdf()

    def close(self):
        self.con.close()


# ── Metadata filter builder ───────────────────────────────────────────────────

def build_metadata_filter(
    sectors: list[str] | None = None,
    regions: list[str] | None = None,
    min_years_coverage: int | None = None,
    ratings: list[str] | None = None,
    is_rating_change: bool | None = None,
    changed_after: str | None = None,   # ISO date string
    changed_before: str | None = None,
) -> str:
    """
    Build a WHERE clause for the reports table based on filter criteria.
    Used by the retrieval layer to pre-filter before vector search.

    Returns a SQL WHERE clause string (without the WHERE keyword).
    Returns '1=1' if no filters are specified.
    """
    conditions = []

    if sectors:
        quoted = ", ".join(f"'{s}'" for s in sectors)
        conditions.append(f"sector IN ({quoted})")

    if regions:
        quoted = ", ".join(f"'{r}'" for r in regions)
        conditions.append(f"region IN ({quoted})")

    if min_years_coverage is not None:
        conditions.append(f"years_coverage >= {min_years_coverage}")

    if ratings:
        quoted = ", ".join(f"'{r}'" for r in ratings)
        conditions.append(f"rating IN ({quoted})")

    if is_rating_change is not None:
        conditions.append(f"is_rating_change = {str(is_rating_change).upper()}")

    if changed_after:
        conditions.append(f"rating_change_date >= '{changed_after}'")

    if changed_before:
        conditions.append(f"rating_change_date <= '{changed_before}'")

    return " AND ".join(conditions) if conditions else "1=1"


def get_filtered_report_ids(db_path: str, where_clause: str) -> list[str]:
    """
    Execute a metadata filter and return matching report_ids.
    Called by the retrieval layer before vector search.
    """
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute(f"SELECT report_id FROM reports WHERE {where_clause}").fetchall()
    con.close()
    return [r[0] for r in rows]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/raw/corpus.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--db",     default="data/finrag.duckdb")
    args = parser.parse_args()

    loader = MetadataLoader(args.db)

    n_reports = loader.load_reports(args.corpus)
    print(f"✓ Loaded {n_reports} reports into DuckDB")

    n_chunks = loader.load_chunks(args.chunks)
    print(f"✓ Loaded {n_chunks} chunks into DuckDB")

    loader.build_analyst_profiles()
    print(f"✓ Built analyst profiles")

    # Quick sanity checks
    print("\n── Sample queries ──────────────────────────────────────────")

    df = loader.query("""
        SELECT sector, COUNT(*) as reports
        FROM reports GROUP BY sector ORDER BY reports DESC
    """)
    print("\nReports by sector:")
    print(df.to_string(index=False))

    six_months_ago = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    df2 = loader.query(f"""
        SELECT COUNT(*) as count
        FROM reports
        WHERE is_rating_change = TRUE
          AND years_coverage >= 10
          AND region = 'Emerging Markets'
          AND rating IN ('Buy', 'Strong Buy')
          AND rating_change_date >= '{six_months_ago}'
    """)
    print(f"\n'Emerging Markets Buy upgrades, 10+ yrs coverage, last 6 months': "
          f"{df2['count'].iloc[0]} reports")

    loader.close()
    