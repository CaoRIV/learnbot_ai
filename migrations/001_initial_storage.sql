CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    file_size INTEGER NOT NULL CHECK (file_size >= 0),
    status TEXT NOT NULL DEFAULT 'processing'
        CHECK (status IN ('processing', 'ready', 'failed')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS documents_content_hash_idx
    ON documents (content_hash);

CREATE INDEX IF NOT EXISTS documents_status_idx
    ON documents (status);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    page INTEGER CHECK (page IS NULL OR page >= 1),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

CREATE TABLE IF NOT EXISTS index_snapshots (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    embedding_model TEXT NOT NULL,
    snapshot_path TEXT NOT NULL,
    chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
    status TEXT NOT NULL
        CHECK (status IN ('active', 'inactive', 'invalid')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS index_snapshots_one_active_idx
    ON index_snapshots (status)
    WHERE status = 'active';
