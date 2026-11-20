# Qdrant Hybrid Search Demo

A small, self-contained demo of **hybrid search** (dense + sparse BM25 with
Reciprocal Rank Fusion) using Qdrant's **Query API** with `prefetch` --
everything runs locally in an **in-memory Qdrant instance**. No Docker, no
server, no API keys.

## What it shows

- Indexes 78 short tech-doc / FAQ snippets (`corpus.json`) with:
  - **Dense vectors**: `BAAI/bge-small-en-v1.5` via fastembed (local CPU)
  - **Sparse vectors**: `Qdrant/bm25` via fastembed, with `Modifier.IDF`
- Runs every query twice and prints the results **side by side**:
  1. Dense-only semantic search
  2. Hybrid search: `client.query_points()` with two `prefetch` branches
     (dense + sparse) fused by `FusionQuery(fusion=Fusion.RRF)`
- Marks hits that only hybrid surfaced with a `+`

## Why hybrid beats dense-only

Dense embeddings are great at meaning ("container keeps restarting" finds
CrashLoopBackOff) but weak at **rare exact tokens**: error codes, acronyms,
IDs. BM25 nails those. RRF fusion gives you both without tuning weights.

Real output from this repo (`python demo.py --examples`):

```
==============================================================================
QUERY: "what does exit code 137 mean"
==============================================================================

--- Dense only (semantic)        [0.7 ms] ---
  1. [linux        ] (score 0.8262)
     Exit code 137 means the process was killed by SIGKILL (128 + 9), most...
  2. [kubernetes   ] (score 0.7461)
     An OOMKilled status (exit code 137) means the container exceeded its ...
  3. [http         ] (score 0.5940)
     ETag headers enable conditional requests: the client sends If-None-Ma...
  4. [http         ] (score 0.5851)
     HTTP 503 Service Unavailable signals temporary overload or maintenanc...
  5. [python       ] (score 0.5811)
     Context managers via the with statement guarantee cleanup: files clos...

--- Hybrid (dense + BM25, RRF)   [2.2 ms] ---
  1.   [linux        ] (rrf 1.0000)
       Exit code 137 means the process was killed by SIGKILL (128 + 9), mo...
  2.   [kubernetes   ] (rrf 0.6667)
       An OOMKilled status (exit code 137) means the container exceeded it...
  ...
```

Both retrievers find the right top hit here, but look at ranks 3-5 on the
dense side: ETag headers, HTTP 503, and Python context managers -- semantic
noise that merely "sounds like" the query. BM25's exact match on `137`
anchors the hybrid ranking. The effect is stronger the rarer the token:

```
==============================================================================
QUERY: "my container keeps restarting over and over"
==============================================================================

--- Dense only (semantic)        [4.2 ms] ---
  1. [kubernetes   ] (score 0.6765)
     Pods stuck in CrashLoopBackOff usually indicate the container exits s...
  2. [kubernetes   ] (score 0.6723)
     An OOMKilled status (exit code 137) means the container exceeded its ...
  ...

--- Hybrid (dense + BM25, RRF)   [2.8 ms] ---
  1.   [kubernetes   ] (rrf 0.8333)
       Pods stuck in CrashLoopBackOff usually indicate the container exits...
  ...
  4. + [kubernetes   ] (rrf 0.3929)
       Readiness probes gate traffic: a pod failing its readinessProbe is ...

  -> hybrid surfaced 1 result(s) dense-only missed (marked +)
```

Note the first query shares zero keywords with "CrashLoopBackOff" -- that is
the dense side earning its keep. Hybrid keeps both strengths.

## Quickstart

```
pip install -r requirements.txt
python demo.py --examples            # 4 canned queries
python demo.py "postgres error 40P01"
python demo.py "fix ERR_SSL_VERSION_OR_CIPHER_MISMATCH"
```

First run downloads ~35MB of ONNX models (cached afterwards). Indexing 78
docs takes about 1.5s on CPU; queries run in single-digit milliseconds.

## How it works (the interesting 20 lines)

The collection is created with one named dense vector and one named sparse
vector, then a single `query_points` call does the fusion server-side:

```python
client.query_points(
    collection_name=COLLECTION,
    prefetch=[
        models.Prefetch(query=query_dense, using="dense", limit=20),
        models.Prefetch(query=query_sparse, using="bm25", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=5,
)
```

Swap `QdrantClient(":memory:")` for a URL and the same code runs against a
real Qdrant cluster.

## About

Built by Muhammad Abdulrehman after using Qdrant in production for a RAG
advisor chatbot at Xiente.
