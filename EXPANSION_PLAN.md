# learnbot_ai Expansion Plan

## 1. Objectives

Develop `learnbot_ai` from a RAG prototype into a Vietnamese document question-answering
assistant that can operate reliably, be evaluated for quality, and scale to support
multiple users.

The following principles remain unchanged:

- The LLM that generates answers must always be called through an API (SiliconFlow, OpenAI, or Gemini).
- Lightweight embedding models and rerankers may run locally.
- The entire UI, logs, and system prompts must be in Vietnamese.
- The application must support Python 3.10+ and machines with approximately 8 GB of RAM.

## 2. Current Status and Issues to Address

- The project already has a document ingestion pipeline, FAISS + BM25 hybrid retrieval,
  a reranker, Gradio, FastAPI, and an automated test suite.
- The index currently exists only in process memory; restarting requires reindexing
  all documents.
- There is no benchmark dataset for measuring Vietnamese retrieval quality.
- The suitability of the embedding model and reranker, as well as the quality of source
  citations, requires further evaluation.
- Scanned PDFs, images, and table layouts are not yet handled fully.

## 3. Proposed Roadmap

### Phase 0 — Stabilize the Foundation

**Objective:** Keep the baseline green before every change.

**Tasks:**

- Standardize test commands for Windows and CI.
- Maintain tests for ingestion, retrieval, LLM providers, and the API.
- Add `compileall`, formatting, and secret-scanning checks.
- Record the Python version, embedding model, and reranker model in use.

**Completion criteria:**

- CI runs reliably on Python 3.10.
- The repository contains no API keys or user data.
- Tests run on every pull request before merging.

### Phase 1 — Persistent Indexes and Incremental Updates

**Objective:** Avoid rebuilding the entire document index after each startup.

**Proposed design:**

- Create a dedicated data directory, such as `data/indexes/`.
- Save FAISS indexes with `faiss.write_index` and load them with `faiss.read_index`.
- Store BM25 data, chunk content, metadata, and ID order in a clearly versioned snapshot.
- Store a manifest containing the file name, size, hash, modification time, chunk count,
  and embedding model.
- Use a Windows-safe file-locking mechanism (`filelock`) to prevent read/write race
  conditions when queries and index updates occur concurrently.
- Write snapshots to a temporary directory and then rename them atomically to avoid
  corrupting the active index.
- Support starting from a snapshot, adding new documents, replacing modified documents,
  and deleting documents.

**Required tests:**

- Saving and reloading an index produces equivalent retrieval results.
- Restarting does not invoke the embedding model when the snapshot remains valid.
- Corrupt or incompatible snapshot versions are detected and reported with Vietnamese
  error messages.
- A mid-operation failure does not remove the previous snapshot.

**Completion criteria:**

- Starting the application with a previously indexed collection does not reprocess documents.
- A `rebuild index` button or API is available when a full rebuild is necessary.

### Phase 2 — Evaluate and Improve Vietnamese Retrieval Quality

**Objective:** Measure quality instead of merely checking whether the system runs.

**Tasks:**

- Create a small Vietnamese evaluation dataset in JSONL format containing questions,
  relevant documents, relevant chunks, and source pages.
- Build a lightweight internal benchmark script that calculates `Recall@k`, `MRR`,
  `Hit Rate`, RAM footprint, and query latency without relying on heavy frameworks,
  ensuring smooth operation on machines with 8 GB of RAM.
- Propose and evaluate lightweight Vietnamese-optimized embedding models (< 500 MB):
  `bkai-foundation-models/vietnamese-bi-encoder` and
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Propose and test cross-encoder rerankers that support Vietnamese:
  `amberoad/bert-multilingual-passage-reranking-msmarco` and `BAAI/bge-reranker-base`.
- Compare multilingual embeddings, BM25, and hybrid retrieval on the same dataset.
- Allow embedding and reranker model names to be configured through environment variables.
- Verify that the active reranker uses a cross-encoder checkpoint trained for the intended task.
- Add a small benchmark to track RAM usage and processing time on low-spec machines.

**Initial completion criteria:**

- A reproducible benchmark report can be generated with a single command.
- Minimum Recall@5 and MRR thresholds prevent quality regressions.
- Every retrieval change updates the relevant test or benchmark.

### Phase 3 — Better-Grounded and Safer Answers

**Objective:** Reduce hallucinations and help users verify sources.

**Tasks:**

- Standardize source metadata: document name, page, chunk ID, and URL when available.
- Display citations next to answers and allow users to open chunk details.
- Standardize the context structure with clear delimiters (for example,
  `<tai_lieu>...</tai_lieu>`) in the system prompt to protect against prompt injection
  from document content.
- Add a minimum confidence threshold; when evidence is insufficient, clearly state that
  the information was not found instead of guessing.
- Support streaming responses (Server-Sent Events / SSE) in the FastAPI backend and
  Gradio UI to reduce perceived response latency.
- Distinguish internal document sources from web search results.
- Record users' correct/incorrect feedback as evaluation data.

**Required tests:**

- Answers contain citations to the correct pages.
- When there are no relevant results, the system does not assert information absent from
  the documents.
- Document content is treated as untrusted data and cannot override the system prompt.

### Phase 4 — Expand Supported Document Types

**Objective:** Support more real-world documents while keeping RAM usage under control.

**Priority order:**

1. OCR for scanned PDFs.
2. Preserve the structure of Excel tables and tables in PDFs.
3. Extract text from PowerPoint files with slide metadata.
4. Optionally analyze images when the LLM provider supports vision.

**Requirements:**

- OCR is optional and must not slow down ordinary text documents.
- Provide documentation for installing external dependencies (Tesseract binaries on
  Windows) and a graceful fallback when no OCR engine is installed.
- Standardize table extraction (Excel/PDF) into Markdown tables or JSON to preserve
  row-and-column structure for both BM25 and embeddings.
- Every chunk must retain its page or slide number.
- Documents exceeding the size limit must be rejected with a clear message.
- Add tests for corrupt, empty, and multi-page documents.

### Phase 5 — Multi-User Support and Production Operations (As Needed)

**Objective:** Support multiple users or independent document collections.

**Tasks:**

- Add authentication and authorization.
- Separate workspaces, documents, and indexes by user or group.
- Store metadata in SQLite or PostgreSQL and large files in object storage.
- Add rate limiting, API quotas, and audit logs.
- Add Docker packaging, health checks, backups, and deployment documentation.

**Prerequisites:**

- Begin only after index snapshots and citations are stable.
- Clearly define the deployment model: internal, personal computer, or public service.

## 4. Priorities

| Priority | Item | Rationale |
| --- | --- | --- |
| P0 | Persistent indexes | Addresses the largest current operational limitation |
| P0 | Retrieval evaluation suite | Establishes a measurement basis for all subsequent improvements |
| P1 | Citations and refusal mechanism | Improves answer reliability |
| P1 | OCR and table structure | Expands the range of supported documents |
| P2 | Multi-user/workspaces | Needed only when the application begins serving multiple users |
| P2 | External integrations | Implement after the core API is stable |

## 5. Milestone-Based Release Plan

### Milestone 1 — Stable Personal Release

- FAISS/BM25 snapshots.
- Incremental updates.
- Vietnamese retrieval benchmarks.
- Page-level citations.

### Milestone 2 — Advanced Document Preview

- OCR for scanned PDFs.
- Table and slide metadata.
- Answer feedback mechanism.

### Milestone 3 — Team Release

- Login and workspaces.
- Document-level permissions.
- Backups, audit logs, and cost limits.

## 6. Risks and Mitigations

- **Insufficient RAM:** Limit the number of documents per ingestion operation, use
  lightweight models, process data in batches, and release memory after indexing.
- **Slow model downloads or network errors:** Use lazy loading and local caching, display
  progress, and provide clear error messages.
- **Embedding changes:** Store the model name and version in the manifest; a new model
  must use a new index namespace.
- **Index read/write conflicts on Windows:** Use multi-process file locking when reading
  and updating snapshots.
- **Missing external OCR dependencies:** Provide a graceful fallback for scanned files
  when the Tesseract binary is not installed on Windows.
- **LLM hallucinations:** Restrict prompts to the provided context, require citations,
  set a refusal threshold, and test out-of-scope questions.
- **Sensitive data exposure:** Warn users when sending context to an external API, allow
  web search to be disabled, and clearly document the data scope in the user guide.

## 7. General Definition of Done

An item is considered complete only when:

- Appropriate automated tests exist.
- Error messages and logs are in Vietnamese.
- The README or operations documentation is updated if usage changes.
- No locally running LLM is used.
- Existing data and indexes are not corrupted when an operation fails.
- The item has been tested on Windows and on a Python version supported by the project.
