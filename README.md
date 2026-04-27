# RAG NQ Showcase

Lean RAG on `sentence-transformers/NQ-retrieval`: ingestion, dense/sparse indexes, retrieval, reranking, grounded generation, and evaluation (see project milestones).

## Milestone 0

- **Contracts**: `src/contracts/` (`Retriever`, `Reranker`, `Generator` protocols)
- **API-oriented schemas**: `src/models/query_schemas.py` (`PassageHit`, `Citation`, `GroundedAnswer`, `QueryRequest`, `QueryResponse`)
- **Logging**: `src/observability/logging_setup.py` (stage-tagged messages via `extra={"stage": "..."}`)
- **Doctor**: `python -m src.scripts.doctor` — validates `Settings` and imports; exits `0` on success, `1` if any check fails

## Configuration

Optional application settings use the `RAG_` prefix. See [.env.example](.env.example).
Scripts load a repo-local `.env` automatically via `Settings.from_env()`; explicit shell
environment variables still take precedence. Provider secrets such as `OPENAI_API_KEY`
can also live in `.env` for local development.

## Milestone 3.6 — staged indexing (raw → chunks → dense → sparse)

**Goal:** Hugging Face is only used during **raw ingest**; downstream stages read local artifacts. Ingest now keeps:
- `artifacts/raw_dataset.jsonl` + `artifacts/raw_ingest_manifest.json` (raw row snapshot)
- `artifacts/index_chunks.jsonl` + `artifacts/chunk_manifest.json` (grouped chunked index corpus)

When manifests match current settings, ingest reuses local artifacts and avoids network.

**Commands**

1. **Raw ingest** (HF / Hub cache → `artifacts/raw_dataset.jsonl` + `raw_ingest_manifest.json`):

   ```bash
   python -m src.ingestion.ingest_raw
   # force re-download even when raw manifest matches
   python -m src.ingestion.ingest_raw --force
   ```

2. **Chunk ingest** (raw snapshot → `index_chunks.jsonl`):

   ```bash
   python -m src.scripts.ingest_chunks
   # force rebuild even when manifests match
   python -m src.scripts.ingest_chunks --force
   ```

3. **Dense** (stream `index_chunks.jsonl` → Qdrant; **requires** chunks unless you opt in to HF):

   ```bash
   python -m src.scripts.index_dense
   # or: refresh raw data, rebuild chunks, then dense
   python -m src.scripts.index_dense --from-hf
   ```

4. **Sparse** (two-pass **streaming** read of chunks → BM25-style sparse vectors on Qdrant via `update_vectors`; **no** full-corpus `rank_bm25` token materialization; writes `artifacts/sparse_index_manifest.json`). **Requires** Qdrant and an existing **dense** collection created with this repo version (collection includes a named sparse vector slot next to `dense`). Run `index_dense` before `index_sparse`.

   ```bash
   python -m src.scripts.index_sparse
   ```

   Sparse indexing writes `artifacts/sparse_checkpoint.json` after each successful sparse flush and resumes from it on the next run when chunk path / collection / sparse name / analyzer / BM25 params / max_passages / sparse batch size still match. Delete that checkpoint to force a full sparse rebuild.

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

5. **All-in-one** (same pipeline as separate stages; honors skip-if-fresh chunk ingest):

   ```bash
   python -m src.scripts.build_indexes
   ```

6. **Chunking strategy analysis** (raw rows → strategy comparison report):

   ```bash
   python -m src.scripts.analyze_chunking_strategies --max-rows 1000 --max-queries 100
   ```

   This writes `artifacts/chunking_evaluation_report.md` and
   `artifacts/chunking_evaluation_report.json`. The analysis uses NQ `long_answers` only as
   evaluation labels to compare candidate, row, fixed-window, parent/deduped, minimum-context,
   and parent-child strategies.

**Env (see `.env.example`)**

- `RAG_MAX_PASSAGES`, `RAG_FORCE_INGEST`, `RAG_FORCE_RAW_INGEST`, `RAG_EMBEDDING_BATCH_SIZE`, `RAG_DENSE_READ_BATCH_LINES`
- Chunking: `RAG_INDEX_CHUNKS_JSONL`, `RAG_CHUNK_MANIFEST_FILE`, `RAG_CHUNK_MIN_TOKENS_SOFT`, `RAG_CHUNK_MIN_TOKENS_HARD`, `RAG_CHUNK_TARGET_TOKENS`, `RAG_CHUNK_MAX_TOKENS`, `RAG_CHUNK_CONTEXT_TEXT_TOKEN_CAP`
- Sparse / BM25: `RAG_QDRANT_SPARSE_VECTOR_NAME` (default `sparse`), `RAG_SPARSE_ANALYZER` (default `regex_stem_stop`), `RAG_SPARSE_UPSERT_BATCH_SIZE`, `RAG_SPARSE_WORKERS`, `RAG_SPARSE_WRITE_CONCURRENCY`, `RAG_SPARSE_CHECKPOINT_FILE`, `RAG_BM25_K1`, `RAG_BM25_B`, `RAG_BM25_EPSILON`
- Progress logging: `RAG_PROGRESS_LOG_EVERY_RECORDS` (default `10000`), `RAG_PROGRESS_LOG_EVERY_BATCHES` (default `500`), `RAG_PROGRESS_LOG_EVERY_SECONDS` (default `60.0`)

`RAG_MAX_PASSAGES` now gates raw/chunk output size and dense/sparse indexing scope, which is useful for smoke tests against larger artifacts.

**Notes**

- Full **train** dense remains **wall-clock heavy**; chunking bounds RAM and streams upserts—it does not make 40M encodes “fast.”
- `qdrant-client` is pinned near **1.13.x** to match `docker-compose` Qdrant; the client uses `check_compatibility=False` when supported.
- Dense upsert uses a minimal retry policy (small fixed attempts + exponential backoff) for transient timeout/overload cases.
- Dense indexing writes a minimal checkpoint at `artifacts/dense_checkpoint.json` after successful upsert chunks and resumes from it on the next run if inputs still match.
- Sparse indexing also checkpoints progress at `artifacts/sparse_checkpoint.json`; default concurrency is conservative (`RAG_SPARSE_WORKERS=1`, `RAG_SPARSE_WRITE_CONCURRENCY=1`) so behavior stays equivalent unless you opt in to higher pass-2 concurrency.
- Long-running raw ingest, chunk ingest, dense indexing, and sparse indexing emit throttled start/progress/complete logs. Defaults avoid per-batch spam; lower the progress env vars for smoke tests or debugging.
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

Chunk ingestion follows the HuggingFace `sentence-transformers/NQ-retrieval` layout: each row’s `candidates` list is preprocessed, role-tagged, and merged into structure-aware chunks. `index_chunks.jsonl` stores one JSON object per source row/group with shared metadata (`title`, `question`, `document_url`, `long_answers`), a local `texts` pool, and a nested `chunks` list that references `texts` by `text_idxs` and `context_idxs`. Dense and sparse indexers flatten those references into Qdrant points using deterministic `chunk_id` values.

## Commands

- **Staged (Milestone 3.6):** `ingest_chunks` → `index_dense` → `index_sparse`, or `build_indexes` for one chain (see Milestone 3.6 above).
- Doctor: `python -m src.scripts.doctor`
- Tests (from repo root): `pytest rag-nq-showcase/tests`

## Milestone 7 — local API runbook

Start Qdrant, build or reuse indexes, then run the FastAPI app:

```bash
cd rag-nq-showcase
docker compose up -d
python -m src.scripts.doctor
python -m uvicorn app.api.main:app --reload
```

OpenAPI docs are available at `http://127.0.0.1:8000/docs`.

### API endpoints

- `GET /health` — lightweight API liveness check.
- `GET /config` — safe runtime metadata: dataset/model names, Qdrant collection/vector names, retrieval/generation knobs, and local artifact presence. It does not expose provider secrets, API keys, or raw environment dumps.
- `POST /retrieve` — retrieval-only diagnostics. Body:

  ```json
  {
    "query": "What is the capital of France?",
    "mode": "hybrid",
    "top_k": 10
  }
  ```

- `POST /query` — retrieval plus optional grounded generation. Body:

  ```json
  {
    "query": "What is the capital of France?",
    "mode": "hybrid",
    "top_k": 10,
    "generate": true
  }
  ```

Both retrieval endpoints return typed `QueryResponse` payloads with retrieved `point_id`s,
ranks/scores where available, retrieval timing/dedupe metrics, and generated citations when
generation is enabled.

### Local troubleshooting

- **Qdrant not reachable:** run `docker compose up -d`, confirm `RAG_QDRANT_URL` points to the running service, then run `RAG_DOCTOR_CHECK_QDRANT=1 python -m src.scripts.doctor`.
- **Missing chunk artifact:** rebuild the default corpus with `python -m src.ingestion.ingest_raw` followed by `python -m src.scripts.ingest_chunks`, or run `python -m src.scripts.build_indexes`.
- **Stale `passages.jsonl` indexes:** the default path after Milestone 3.6 is `artifacts/index_chunks.jsonl`; rebuild dense and sparse indexes from chunks before comparing retrieval results.
- **Missing sparse vectors / wrong sparse vector name:** check `RAG_QDRANT_SPARSE_VECTOR_NAME`, confirm `artifacts/sparse_index_manifest.json` matches the target collection, and rerun `python -m src.scripts.index_sparse`.
- **Partial sparse migration or stale collection:** rebuild sparse vectors against the current dense collection; if the collection schema is incompatible, use the migration/recreate workflow documented above.
- **Hybrid query looks worse than sparse:** compare `dense`, `sparse`, and `hybrid` through `/retrieve`; tune `RAG_HYBRID_DENSE_WEIGHT`, `RAG_HYBRID_SPARSE_WEIGHT`, `RAG_RETRIEVE_K`, and rerank settings before treating hybrid as the default.

## Milestone 8 — Streamlit portfolio UI

Start Qdrant and the FastAPI service, then launch the Streamlit demo:

```bash
cd rag-nq-showcase
docker compose up -d
uv run uvicorn app.api.main:app
uv run streamlit run app/streamlit_app.py
```

The UI defaults to `http://127.0.0.1:8000` for the API and
`artifacts/retrieval_eval.json` for the eval dashboard. Both can be changed in
the sidebar without editing `.env`.

### UI panels

- **Query Playground:** sends `/query` requests with retrieval mode, `top_k`, and
  generation controls. Abstained grounded answers are shown as a human-readable
  "not enough evidence" message while preserving the backend `answer` contract.
- **Retrieval Inspector:** sends `/retrieve` requests for dense, sparse, and
  hybrid modes side by side, including point IDs, rank metadata, context, spans,
  and duplicate aliases where available.
- **Eval Dashboard:** reads the persisted retrieval eval JSON artifact and
  summarizes Recall@k, MRR@k, NDCG@k, winners, and run config.
- **System / Runbook:** shows `/health`, safe `/config` metadata, artifact
  status, and copyable local commands.

The Streamlit UI is a presentation layer only. It does not rebuild indexes,
modify environment files, reset Qdrant, or import internal retriever/generator
classes for live behavior.
