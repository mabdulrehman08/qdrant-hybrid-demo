"""Hybrid search demo: dense + BM25 sparse vectors fused with RRF in Qdrant.

Runs fully local: in-memory Qdrant (no Docker, no server) and fastembed
models on CPU (no API keys). Compares dense-only retrieval against hybrid
retrieval (dense + sparse with Reciprocal Rank Fusion) side by side.

Usage:
    python demo.py "your question"
    python demo.py --examples
"""

import json
import sys
import time
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

DENSE_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, ~34MB, CPU-friendly
SPARSE_MODEL = "Qdrant/bm25"             # BM25-style sparse term weights
COLLECTION = "tech_docs"
TOP_K = 5

EXAMPLE_QUERIES = [
    "my container keeps restarting over and over",
    "what does exit code 137 mean",
    "fix ERR_SSL_VERSION_OR_CIPHER_MISMATCH",
    "how do I handle HTTP 429 responses",
]


def load_corpus():
    path = Path(__file__).parent / "corpus.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(docs, dense_model, sparse_model):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    texts = [d["text"] for d in docs]
    dense_vecs = list(dense_model.embed(texts))
    sparse_vecs = list(sparse_model.embed(texts))

    points = []
    for doc, dv, sv in zip(docs, dense_vecs, sparse_vecs):
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": dv.tolist(),
                    "bm25": models.SparseVector(
                        indices=sv.indices.tolist(), values=sv.values.tolist()
                    ),
                },
                payload={"text": doc["text"], "category": doc["category"]},
            )
        )
    client.upsert(collection_name=COLLECTION, points=points)
    return client


def search_dense(client, query_dense, limit=TOP_K):
    return client.query_points(
        collection_name=COLLECTION,
        query=query_dense,
        using="dense",
        limit=limit,
        with_payload=True,
    ).points


def search_hybrid(client, query_dense, query_sparse, limit=TOP_K):
    return client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense", limit=20),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices.tolist(),
                    values=query_sparse.values.tolist(),
                ),
                using="bm25",
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points


def snippet(text, width=66):
    return text if len(text) <= width else text[: width - 3] + "..."


def run_query(client, dense_model, sparse_model, query):
    query_dense = list(dense_model.query_embed(query))[0].tolist()
    query_sparse = list(sparse_model.query_embed(query))[0]

    t0 = time.perf_counter()
    dense_hits = search_dense(client, query_dense)
    t_dense = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hybrid_hits = search_hybrid(client, query_dense, query_sparse)
    t_hybrid = (time.perf_counter() - t0) * 1000

    print()
    print("=" * 78)
    print(f'QUERY: "{query}"')
    print("=" * 78)
    print(f"\n--- Dense only (semantic)        [{t_dense:.1f} ms] ---")
    for i, hit in enumerate(dense_hits, 1):
        print(f"  {i}. [{hit.payload['category']:<13}] (score {hit.score:.4f})")
        print(f"     {snippet(hit.payload['text'], 72)}")
    print(f"\n--- Hybrid (dense + BM25, RRF)   [{t_hybrid:.1f} ms] ---")
    dense_ids = {h.id for h in dense_hits}
    for i, hit in enumerate(hybrid_hits, 1):
        marker = "  " if hit.id in dense_ids else "+ "  # + = surfaced by hybrid
        print(f"  {i}. {marker}[{hit.payload['category']:<13}] (rrf {hit.score:.4f})")
        print(f"       {snippet(hit.payload['text'], 70)}")
    new = [h for h in hybrid_hits if h.id not in dense_ids]
    if new:
        print(f"\n  -> hybrid surfaced {len(new)} result(s) dense-only missed (marked +)")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    queries = EXAMPLE_QUERIES if args[0] == "--examples" else [" ".join(args)]

    print("Loading embedding models (first run downloads ~35MB, then cached)...")
    dense_model = TextEmbedding(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)

    docs = load_corpus()
    print(f"Indexing {len(docs)} documents into in-memory Qdrant "
          f"(dense {DENSE_MODEL} + sparse {SPARSE_MODEL})...")
    t0 = time.perf_counter()
    client = build_index(docs, dense_model, sparse_model)
    print(f"Indexed in {time.perf_counter() - t0:.1f}s")

    for q in queries:
        run_query(client, dense_model, sparse_model, q)


if __name__ == "__main__":
    main()
