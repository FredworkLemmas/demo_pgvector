from datetime import datetime

import attrs
from pathlib import Path
from typing import Iterator

import yaml
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import BaseChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

from lib.interfaces import PostgresqlConnectionProvider

MAX_CHUNK_TOKENS = 512
TExT_TYPE__FICTION = 1
TEXT_TYPE__NONFICTION = 2


@attrs.define
class SourceDocument(object):
    postgresql_connection_provider: PostgresqlConnectionProvider
    source_filepath: str
    metadata: dict | None = None
    chunker_class: type[BaseChunker] | None = None
    max_chunk_tokens: int | None = None
    model_name: str | None = None

    def __attrs_post_init__(self):
        self.chunker_class = self.chunker_class or HybridChunker()
        self.metadata = self.metadata or self._metadata_from_file()
        self.max_chunk_tokens = self.max_chunk_tokens or MAX_CHUNK_TOKENS
        self.model_name = (
            self.model_name or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
        )

    def enriched_chunks(self):
        import_date = datetime.datetime.now().isoformat()
        for n, chunk in enumerate(self._raw_chunk_iterator()):
            yield SourceDocumentChunk(
                text=chunk.text,
                source_document=self,
                metadata={
                    **self.metadata,
                    'chunk_tokenizer_model': self.model_name,
                    'chunk_size': self.max_chunk_tokens,
                    'chunk_number': n,
                    'import_date': import_date,
                },
            )

    def register_source(self):
        """
        Insert this source into the 'sources' table using self.metadata and ensure
        a corresponding row exists in the 'models' table for self.model_name.

        Returns:
            int: The ID of the inserted (or existing) row in the 'sources' table.
        """
        from typing import Optional

        # Lazy import so the module is only required when this method is called.
        try:
            pass  # psycopg 3.x
        except Exception:
            try:
                pass  # fallback to psycopg2 if available
            except Exception as e:
                raise RuntimeError(
                    'A PostgreSQL client library (psycopg or psycopg2) is required to register sources.'
                ) from e

        # Map text_type to source_type (simple heuristic)
        def _map_source_type(meta: dict) -> int:
            text_type = (meta.get('text_type') or '').strip().lower()
            fiction_indicators = {
                'short story',
                'novel',
                'novella',
                'poem',
                'fiction',
                'science fiction',
                'sci-fi',
                'sf',
                'fantasy',
                'fable',
                'drama',
                'play',
            }
            if text_type in fiction_indicators:
                return TExT_TYPE__FICTION
            # If genre clearly implies fiction, treat it as fiction.
            genre = (meta.get('genre') or '').strip().lower()
            if genre in {
                'science fiction',
                'fantasy',
                'horror',
                'mystery',
                'thriller',
            }:
                return TExT_TYPE__FICTION
            return TEXT_TYPE__NONFICTION

        def _extract_year(meta: dict) -> Optional[int]:
            # Expecting 'publication_date' possibly as int, 'YYYY', or ISO date.
            pub = meta.get('publication_date')
            if pub is None:
                return None
            # Already an int-like
            try:
                return int(str(pub)[:4])
            except Exception:
                return None

        meta = self.metadata or {}

        author = meta.get('author')
        title = meta.get('title')
        url = meta.get('url')
        genre = meta.get('genre')
        subgenre = meta.get('subgenre')
        source_type = _map_source_type(meta)
        year = _extract_year(meta)
        model_name = self.model_name or 'unknown-model'
        # Default embedding dimension; aligns with source_chunks.embedding vector(1536)
        embedding_dim = 1536

        conn = self.postgresql_connection_provider.get_connection()

        conn.autocommit = False
        with conn.cursor() as cur:
            # Ensure model row exists and get model_id
            # Try insert; on conflict, fetch the existing id.
            model_id: Optional[int] = None
            try:
                cur.execute(
                    """
                    INSERT INTO models (name, embedding_dim)
                    VALUES (%s, %s)
                    ON CONFLICT (name)
                    DO UPDATE SET embedding_dim = EXCLUDED.embedding_dim
                    RETURNING id
                    """,
                    (model_name, embedding_dim),
                )
                row = cur.fetchone()
                if row:
                    model_id = row[0]
            except Exception:
                # If RETURNING not supported or schema differs, fall back to select
                conn.rollback()
                with conn.cursor() as cur2:
                    cur2.execute(
                        'SELECT id FROM models WHERE name = %s', (model_name,)
                    )
                    row = cur2.fetchone()
                    if row:
                        model_id = row[0]
                    else:
                        # Try a plain insert without RETURNING and then select id
                        cur2.execute(
                            'INSERT INTO models (name, embedding_dim) VALUES (%s, %s)',
                            (model_name, embedding_dim),
                        )
                        cur2.execute(
                            'SELECT id FROM models WHERE name = %s',
                            (model_name,),
                        )
                        row = cur2.fetchone()
                        model_id = row[0] if row else None

            # Insert into sources
            cur.execute(
                """
                INSERT INTO sources (author, title, source_type, url, genre, subgenre, year, model_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    author,
                    title,
                    source_type,
                    url,
                    genre,
                    subgenre,
                    year,
                    model_id,
                ),
            )
            source_row = cur.fetchone()
            source_id = source_row[0] if source_row else None
        conn.commit()

        return source_id

    def _raw_chunk_iterator(self) -> Iterator:
        """
        split doc into chunks
        """
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            self.model_name, max_tokens=self.max_chunk_tokens
        )

        document = self._as_docling_document()
        chunker = HybridChunker(tokenizer=tokenizer)

        return chunker.chunk(document)

    def _as_docling_document(self):
        # Validate that the file exists
        if not Path(self.source_filepath).exists():
            raise FileNotFoundError(f'File not found: {self.source_filepath}')
        return (
            DocumentConverter()
            .convert(source=Path(self.source_filepath))
            .document
        )

    def _metadata_from_file(self):
        p = Path(self.source_filepath)
        metadata_path = p.with_name(p.name + '.meta.yml')
        if metadata_path.exists():
            return yaml.safe_load(metadata_path.read_text())
        return {}


@attrs.define
class SourceDocumentChunk:
    text: str
    source_document: SourceDocument
    metadata: dict
