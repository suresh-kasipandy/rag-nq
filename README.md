# RAG NQ Showcase

Lean RAG on `sentence-transformers/NQ-retrieval`: ingestion, dense/sparse indexes, retrieval, reranking, grounded generation, and evaluation (see project milestones).

## Milestone 0

- **Contracts**: `src/contracts/` (`Retriever`, `Reranker`, `Generator` protocols)
- **API-oriented schemas**: `src/models/query_schemas.py` (`PassageHit`, `Citation`, `GroundedAnswer`, `QueryRequest`, `QueryResponse`)
- **Logging**: `src/observability/logging_setup.py` (stage-tagged messages via `extra={"stage": "..."}`)
- **Doctor**: `python -m src.scripts.doctor` — validates `Settings` and imports; exits `0` on success, `1` if any check fails

## Configuration

Optional environment variables use the `RAG_` prefix. See [.env.example](.env.example).

## Milestone 2 — staged indexing (silver → dense → sparse)

**Goal:** Hugging Face is only used during **ingest**; dense/sparse read **`artifacts/passages.jsonl`** (the silver contract). Ingest skips work when `ingest_manifest.json` matches config and line counts.

**Commands**

1. **Ingest** (HF / Hub cache → `artifacts/passages.jsonl` + `ingest_manifest.json`):

   ```bash
   python -m src.scripts.ingest_passages
   ```

2. **Dense** (chunked read of silver → Qdrant; **requires** silver unless you opt in to HF):

   ```bash
   python -m src.scripts.index_dense
   # or: pull from HF into silver first, then dense
   python -m src.scripts.index_dense --from-hf
   ```

3. **Sparse** (Milestone 2.2: two-pass **streaming** read of silver → BM25-style sparse vectors on Qdrant via `update_vectors`; **no** full-corpus `rank_bm25` token materialization; writes `artifacts/sparse_index_manifest.json`). **Requires** Qdrant and an existing **dense** collection created with this repo version (collection includes a named sparse vector slot next to `dense`). Run `index_dense` before `index_sparse`.

   ```bash
   python -m src.scripts.index_sparse
   ```

### Migrating from `nq_passages_dense` to `nq_passages` (one-time)

If you already have a **dense-only** legacy collection (for example `nq_passages_dense` with no sparse vector slot), Qdrant cannot add a new sparse name in place. Copy stored dense vectors + payloads into a new collection that includes a sparse slot (no re-embedding):

```bash
# Optional: RAG_QDRANT_URL=...  (defaults to http://localhost:6333)
python -m src.scripts.migrate_qdrant_passages_collection
# or: --source-collection nq_passages_dense --target-collection nq_passages --batch-size 512
# dry run: preflight + first scroll page size only
python -m src.scripts.migrate_qdrant_passages_collection --dry-run
# if the target already exists and must be replaced (deletes the target collection first):
python -m src.scripts.migrate_qdrant_passages_collection --recreate-target
```

Then set `RAG_QDRANT_COLLECTION=nq_passages` (or rely on the project default), run `python -m src.scripts.index_sparse`, and retire or delete the old collection when satisfied.

**Checkpoint:** `artifacts/dense_checkpoint.json` stores `collection_name`. If a dense rebuild was mid-flight against the old collection name, delete or edit that checkpoint before resuming `index_dense` against the new collection.

4. **All-in-one** (same pipeline as separate stages; honors skip-if-fresh ingest):

   ```bash
   python -m src.scripts.build_indexes
   ```

**Env (see `.env.example`)**

- `RAG_MAX_PASSAGES`, `RAG_FORCE_INGEST`, `RAG_EMBEDDING_BATCH_SIZE`, `RAG_DENSE_READ_BATCH_LINES`
- Sparse / BM25 (Milestone 2.2): `RAG_QDRANT_SPARSE_VECTOR_NAME` (default `sparse`), `RAG_SPARSE_UPSERT_BATCH_SIZE`, `RAG_BM25_K1`, `RAG_BM25_B`, `RAG_BM25_EPSILON`

`RAG_MAX_PASSAGES` now gates both:
- ingest output size (HF → silver line count), and
- dense indexing scope (silver → Qdrant points), which is useful for smoke tests against a larger existing silver file.

**Notes**

- Full **train** dense remains **wall-clock heavy**; chunking bounds RAM and streams upserts—it does not make 40M encodes “fast.”
- `qdrant-client` is pinned near **1.13.x** to match `docker-compose` Qdrant; the client uses `check_compatibility=False` when supported.
- Dense upsert uses a minimal retry policy (small fixed attempts + exponential backoff) for transient timeout/overload cases.
- Dense indexing writes a minimal checkpoint at `artifacts/dense_checkpoint.json` after successful upsert chunks and resumes from it on the next run if inputs still match.
- **Legacy:** `src.retrieval.sparse_index.SparseIndexer` + `artifacts/bm25_index.pkl` remain for tests and historical Milestone 1–2 behavior; the default CLI path is Qdrant sparse (above).

## Milestone 1.5 — local Qdrant

1. Start Qdrant (persists vectors in a Docker named volume across restarts):

   ```bash
   cd rag-nq-showcase
   docker compose up -d
   ```

2. Optional: verify Qdrant from the doctor (requires the server running on `RAG_QDRANT_URL`):

   ```bash
   RAG_DOCTOR_CHECK_QDRANT=1 python -m src.scripts.doctor
   ```

3. Build indexes against the default URL `http://localhost:6333` (override with `RAG_QDRANT_URL` if needed):

   ```bash
   python -m src.scripts.build_indexes
   ```

To stop Qdrant without deleting stored vectors: `docker compose stop`. To remove the volume as well: `docker compose down -v`.

Ingestion follows the HuggingFace `sentence-transformers/NQ-retrieval` layout: each row’s `candidates` list is expanded into one passage per non-empty candidate, with deterministic UUID passage IDs and optional metadata (`title`, `question`, `passage_types` per candidate, `document_url`, `long_answers`).

## Commands

- **Staged (Milestone 2):** `ingest_passages` → `index_dense` → `index_sparse`, or `build_indexes` for one chain (see Milestone 2 above).
- Doctor: `python -m src.scripts.doctor`
- Tests (from repo root): `pytest rag-nq-showcase/tests`
