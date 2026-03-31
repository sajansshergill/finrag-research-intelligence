"""
embedder.py
-----------
Generates embeddings for chunked text and upserts them into Qdrant.

Model: BAAI/bge-large-en-v1.5 (1024-dim, strong on financial domain)
Vector store: Qdrant (local Docker or Qdrant Cloud)

Each Qdrant point stores:
  - id         : deterministic UUID derived from chunk_id
  - vector     : 1024-dim dense embedding
  - payload    : full structured metadata (analyst, sector, region, etc.)
                 enabling payload-level filtering inside Qdrant

Freshness tracking:
  - Each point's payload includes `embedded_at` timestamp
  - The Airflow DAG checks this against report publication_date
  - Stale chunks (publication_date > embedded_at) are re-upserted

Usage:
    python src/ingestion/embedder.py \
        --chunks data/processed/chunks.jsonl \
        --collection finrag \
        --qdrant-url http://localhost:6333
"""

import json
import uuid
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path


# ── UUID helper ───────────────────────────────────────────────────────────────

def chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id string (UUID v5, DNS namespace)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


# ── Embedder class ────────────────────────────────────────────────────────────

class FinRAGEmbedder:
    """
    Handles embedding generation and Qdrant upsert.

    Falls back to a MockEmbedder if sentence-transformers or qdrant-client
    are not installed (useful for unit tests and CI pipelines).
    """

    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM   = 1024
    BATCH_SIZE      = 64

    def __init__(self, qdrant_url: str = "http://localhost:6333",
                 collection: str = "finrag", use_mock: bool = False):
        self.collection = collection
        self.use_mock   = use_mock
        self._model     = None
        self._client    = None

        if not use_mock:
            self._init_model()
            self._init_qdrant(qdrant_url)

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(self.EMBEDDING_MODEL)
            print("✓ Model loaded")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def _init_qdrant(self, url: str):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._client = QdrantClient(url=url)
            # Create collection if not exists
            existing = [c.name for c in self._client.get_collections().collections]
            if self.collection not in existing:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"✓ Created Qdrant collection: {self.collection}")
            else:
                print(f"✓ Using existing Qdrant collection: {self.collection}")
        except ImportError:
            raise RuntimeError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if self.use_mock:
            import random
            return [[random.gauss(0, 0.1) for _ in range(self.EMBEDDING_DIM)]
                    for _ in texts]
        return self._model.encode(
            texts,
            batch_size=self.BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,    # cosine sim via dot product
        ).tolist()

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """
        Embed and upsert a list of chunk dicts into Qdrant.
        Returns number of points upserted.
        """
        from qdrant_client.models import PointStruct

        embedded_at = datetime.now(timezone.utc).isoformat()
        total = 0

        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i: i + self.BATCH_SIZE]
            texts = [c["text"] for c in batch]
            vectors = self.embed_texts(texts)

            points = []
            for chunk, vector in zip(batch, vectors):
                payload = {k: v for k, v in chunk.items() if k != "text"}
                payload["text"]        = chunk["text"]
                payload["embedded_at"] = embedded_at
                points.append(PointStruct(
                    id      = chunk_id_to_uuid(chunk["chunk_id"]),
                    vector  = vector,
                    payload = payload,
                ))

            if not self.use_mock:
                self._client.upsert(
                    collection_name=self.collection,
                    points=points,
                    wait=True,
                )
            total += len(points)

        return total

    def get_collection_info(self) -> dict:
        if self.use_mock:
            return {"status": "mock", "vectors_count": 0}
        info = self._client.get_collection(self.collection)
        return {
            "status":        info.status,
            "vectors_count": info.vectors_count,
            "points_count":  info.points_count,
        }


# ── Freshness check ───────────────────────────────────────────────────────────

def get_stale_chunks(chunks: list[dict],
                     last_embedded: dict[str, str]) -> list[dict]:
    """
    Return chunks that need re-embedding.
    A chunk is stale if it was published after its last embedding timestamp.
    last_embedded: {chunk_id: iso_timestamp_str}
    """
    stale = []
    for c in chunks:
        if c["chunk_id"] not in last_embedded:
            stale.append(c)
            continue
        pub  = datetime.fromisoformat(c["publication_date"])
        emb  = datetime.fromisoformat(last_embedded[c["chunk_id"]][:19])
        if pub.date() > emb.date():
            stale.append(c)
    return stale


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks",      default="data/processed/chunks.jsonl")
    parser.add_argument("--collection",  default="finrag")
    parser.add_argument("--qdrant-url",  default="http://localhost:6333")
    parser.add_argument("--mock",        action="store_true",
                        help="Use mock embeddings (no model required)")
    args = parser.parse_args()

    with open(args.chunks) as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(chunks)} chunks from {args.chunks}")

    embedder = FinRAGEmbedder(
        qdrant_url  = args.qdrant_url,
        collection  = args.collection,
        use_mock    = args.mock,
    )

    t0 = time.time()
    n = embedder.upsert_chunks(chunks)
    elapsed = time.time() - t0

    print(f"✓ Upserted {n} points in {elapsed:.1f}s "
          f"({n/elapsed:.0f} chunks/sec)")

    if not args.mock:
        info = embedder.get_collection_info()
        print(f"✓ Collection '{args.collection}': {info}")