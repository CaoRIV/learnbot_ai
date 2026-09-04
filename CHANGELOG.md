# Changelog

All notable changes to this project are documented here.

## [2.8.0] - 2026-09-04

### Added

- A `GET /api/documents` endpoint that returns persisted document metadata and chunk counts without exposing content hashes.
- Automatic restoration of the persisted document list in the Next.js sidebar when the application opens.

## [2.7.0] - 2026-09-02

### Added

- Structured citation rows below each assistant answer with document name, page, chunk ID, relevance score, source type, and safe web links.
- A compact structured citation view for the latest answer in the context panel.
- Explicit UI states for insufficient evidence, empty knowledge bases, and answer-generation errors.

### Changed

- The Next.js workspace now consumes `citations` and `answer_status` instead of deriving its citation UI from the legacy `sources` field.
- The latest-source panel now follows the latest assistant response, so an ungrounded response cannot leave stale citations visible.
- Long citation lists collapse behind an accessible disclosure after the first relevant entries.

## [2.6.0] - 2026-09-01

### Added

- Deterministic Vietnamese refusal responses for empty knowledge bases and retrieval results with no evidence above the relevance threshold.
- Machine-readable `answer_status` values: `answered`, `insufficient_evidence`, `empty_knowledge_base`, and `error`.
- Acceptance tests proving out-of-scope questions do not reach answer generation and return no citations.

### Changed

- The answer-generation LLM and conflict detector are skipped when no eligible evidence remains after filtering.
- Gradio-compatible streaming now reports a distinct Vietnamese status for insufficient evidence and empty knowledge bases.

## [2.5.0] - 2026-08-31

### Added

- A configurable `MIN_RELEVANCE_SCORE` quality gate for retrieved evidence, defaulting to `0.35` on a 0–1 scale.
- Relevance-threshold metadata in `/api/status` and `/api/ask` responses.
- Tests for threshold boundaries, invalid configuration, unscored web evidence, and prompt/citation filtering.

### Changed

- Only local evidence meeting the minimum rerank score is included in the LLM context and structured citations.
- Unscored web results remain eligible because web search must be explicitly enabled by the user.

## [2.4.0] - 2026-08-31

### Added

- Structured retrieval citations with document name, page, chunk ID, rerank score, source type, and optional web URL.
- A typed `citations` field in the `/api/ask` response while retaining the legacy `sources` field.
- Tests proving that API citations come from indexed metadata instead of LLM-generated answer text.

### Changed

- The answer pipeline now carries retrieval evidence and scores through generation without changing the legacy string-returning Gradio API.
- Removed regex-based citation extraction from generated answers.

## [2.3.0] - 2026-08-27

### Added

- A versioned Vietnamese retrieval benchmark with 24 labeled questions, document names, pages, and expected chunks.
- A network-independent benchmark runner for BM25, FAISS, hybrid retrieval, and optional local CrossEncoder reranking.
- Recall@K, MRR, mean latency, P95 latency, JSON/Markdown reports, dataset fingerprints, and CI-friendly recall thresholds.
- A checked-in v2.3.0 baseline report for BM25, FAISS, and hybrid retrieval.

### Changed

- Embedding and reranker model names are now centralized in environment-aware configuration.
- Hybrid merging accepts an explicit metadata map so isolated benchmarks do not depend on the runtime singleton.

## [2.2.0] - 2026-08-25

### Added

- Versioned FAISS/BM25 snapshots with a JSON manifest and SHA-256 integrity checks.
- Automatic index restoration when the Gradio application or FastAPI service starts.
- Tests for snapshot round trips, model compatibility, corruption detection, and transactional rollback.

### Changed

- Document ingestion now embeds only new chunks when a compatible snapshot is active.
- SQLite document writes and snapshot activation now commit in the same transaction.
- Corrupt or incompatible snapshots are rejected with Vietnamese diagnostic messages.

## [2.1.0] - 2026-08-12

### Added

- MIT license recognized by GitHub.
- English README and a concise Chinese project guide.
- Automated tests for configuration, document loading, hybrid retrieval, and missing-key behavior.
- GitHub Actions CI for source compilation and tests.
- Contribution, security, conduct, issue, and pull request guidance.
- A current application screenshot and centralized version metadata.

### Changed

- Repositioned the repository as a transparent educational and reference RAG implementation.
- Clarified supported document types, setup steps, provider choices, and known limitations.
- Added the runtime dependencies required for Excel parsing.
- Updated Gradio support to the 6.x line used by the current interface.

### Removed

- Commercial book, course, community, and store promotion from the repository.

## [2.0.0] - 2026-03-18

### Added

- Modular `core/` and `features/` structure.
- Gradio 6.x compatibility updates.
- Configurable model names and provider selection.
- FAISS and BM25 hybrid retrieval pipeline.
