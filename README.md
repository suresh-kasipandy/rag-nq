# RAG NQ Showcase

Lean RAG on `sentence-transformers/NQ-retrieval`: ingestion, dense/sparse indexes, retrieval, reranking, grounded generation, and evaluation (see project milestones).

## Milestone 0

- **Contracts**: `src/contracts/` (`Retriever`, `Reranker`, `Generator` protocols)
- **API-oriented schemas**: `src/models/query_schemas.py` (`PassageHit`, `Citation`, `GroundedAnswer`, `QueryRequest`, `QueryResponse`)
- **Logging**: `src/observability/logging_setup.py` (stage-tagged messages via `extra={"stage": "..."}`)
- **Doctor**: `python -m src.scripts.doctor` — validates `Settings` and imports; exits `0` on success, `1` if any check fails

## Configuration

Optional environment variables use the `RAG_` prefix. See [.env.example](.env.example).

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

- Build indexes (Milestone 1 + 1.5): `python -m src.scripts.build_indexes` (expects Qdrant reachable when building the dense index)
- Doctor: `python -m src.scripts.doctor`
- Tests (from repo root): `pytest rag-nq-showcase/tests`
