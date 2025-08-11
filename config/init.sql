CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) PRIMARY KEY,
    embedding_dim INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS text_embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR({embedding_dim}),
    model_id INTEGER REFERENCES models(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);
