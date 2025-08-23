CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    embedding_dim INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_models_name ON models (name);

CREATE TABLE sources (
    id SERIAL PRIMARY KEY,
    author TEXT,
    title TEXT,
    source_type int,  --- fiction, non-fiction, etc.
    url TEXT,
    genre TEXT,
    subgenre TEXT,
    year INTEGER,
    model_id INTEGER REFERENCES models(id),
    UNIQUE (model_id, author, title, year)
);

CREATE TABLE source_chunks (
    id BIGSERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES sources(id),
    model_id INTEGER REFERENCES models(id),
    embedding vector(1536)
);

CREATE TABLE source_chunk_data (
    id BIGSERIAL PRIMARY KEY,
    chunk_id INTEGER REFERENCES source_chunks(id),
    metadata JSONB,
    chunk_text TEXT
);
