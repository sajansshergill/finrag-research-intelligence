"""
chunker.py
----------
Sentence-window chunking for financial research reports.

Strategy: split each report body into overlapping sentence windows.
Each chunk carries the full structured metadata from its parent report,
plus chunk-level fields (chunk_id, window_index, sentence_count).

Why sentence-window chunking?
- Financial reports contain dense, multi-clause sentences.
- Fixed-token chunking can split a single recommendation sentence across
  two chunks, losing the rating + company co-occurrence that retrieval needs.
- Overlapping windows ensure boundary sentences appear in multiple chunks,
  improving recall for queries that land on section transitions.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict


# ── Config ────────────────────────────────────────────────────────────────────

WINDOW_SIZE = 4        # sentences per chunk
OVERLAP = 1            # overlapping sentences between consecutive chunks
MIN_CHUNK_CHARS = 80   # discard very short trailing chunks


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    report_id: str
    window_index: int
    text: str
    sentence_count: int
    # Structured metadata (carried from parent report)
    analyst_name: str
    company: str
    ticker: str
    sector: str
    region: str
    rating: str
    old_rating: str | None
    is_rating_change: bool
    rating_change_date: str | None
    target_price: int
    publication_date: str
    years_coverage: int


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'   # period/exclaim/question → space → capital
)

def split_sentences(text: str) -> list[str]:
    """
    Naive but effective sentence splitter for financial prose.
    Avoids splitting on common abbreviations (vs., e.g., FY24.).
    """
    # Protect common abbreviations from splitting
    protected = re.sub(r'\b(vs|e\.g|i\.e|approx|est|FY\d{2,4}|Q[1-4])\.',
                       lambda m: m.group().replace(".", "<!DOT!>"), text)
    raw = _SENT_BOUNDARY.split(protected)
    return [s.replace("<!DOT!>", ".").strip() for s in raw if s.strip()]


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_report(report: dict) -> list[Chunk]:
    """
    Chunk a single report into overlapping sentence windows.
    Returns a list of Chunk objects with full metadata.
    """
    sentences = split_sentences(report["body"])
    if not sentences:
        return []

    chunks = []
    step = WINDOW_SIZE - OVERLAP
    window_index = 0

    for start in range(0, len(sentences), step):
        window = sentences[start: start + WINDOW_SIZE]
        text = " ".join(window)

        if len(text) < MIN_CHUNK_CHARS:
            continue

        chunk = Chunk(
            chunk_id=f"{report['report_id']}_W{window_index:03d}",
            report_id=report["report_id"],
            window_index=window_index,
            text=text,
            sentence_count=len(window),
            analyst_name=report["analyst_name"],
            company=report["company"],
            ticker=report["ticker"],
            sector=report["sector"],
            region=report["region"],
            rating=report["rating"],
            old_rating=report.get("old_rating"),
            is_rating_change=report["is_rating_change"],
            rating_change_date=report.get("rating_change_date"),
            target_price=report["target_price"],
            publication_date=report["publication_date"],
            years_coverage=report["years_coverage"],
        )
        chunks.append(chunk)
        window_index += 1

    return chunks


def chunk_corpus(input_path: str, output_path: str) -> list[dict]:
    """
    Chunk all reports in a JSONL corpus file.
    Writes chunked output to a new JSONL file.
    Returns list of chunk dicts.
    """
    all_chunks = []

    with open(input_path) as f:
        reports = [json.loads(line) for line in f if line.strip()]

    for report in reports:
        chunks = chunk_report(report)
        all_chunks.extend([asdict(c) for c in chunks])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"✓ Chunked {len(reports)} reports → {len(all_chunks)} chunks")
    print(f"  Avg chunks/report : {len(all_chunks)/len(reports):.1f}")
    print(f"  Output            : {output_path}")
    return all_chunks


if __name__ == "__main__":
    chunk_corpus("data/raw/corpus.jsonl", "data/processed/chunks.jsonl")