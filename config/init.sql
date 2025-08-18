CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) PRIMARY KEY,
    embedding_dim INTEGER NOT NULL
);

CREATE TABLE sources (
    id SERIAL PRIMARY KEY,
    author TEXT,
    title TEXT,
    source_type int,  # fiction, non-fiction, etc.
    url TEXT,
    genre TEXT,
    subgenre TEXT,
    year INTEGER,
    model_id INTEGER REFERENCES models(id)
);

CREATE TABLE sim_chunks (
    id BIGSERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id)
    embedding vector(1536)
);

CREATE TABLE sim_chunk_data (
    id BIGSERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id),
    metadata JSONB,
    compressed_text TEXT
);

CREATE TABLE source_summaries (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id)
    embedding vector(1536)
);

CREATE TABLE source_summary_data (
    id SERIAL PRIMARY KEY,
    summary_id INTEGER REFERENCES source_summaries(id),
    compressed_text TEXT,
    metadata JSONB
);
