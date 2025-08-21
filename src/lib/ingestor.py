import json
import attrs

from .database import SimpleVectorDatabase
from .interfaces import EmbeddingIngestor


@attrs.define
class PgvectorIngestor(EmbeddingIngestor):
    database: SimpleVectorDatabase
    model_name: str
    embedding_dim: int
    model_id: int | None = None

    def __attrs_post_init__(self):
        self.model_id = (
            self.model_id or self.database.create_or_lookup_model_id()
        )

    def ingest(
        self, embedding: list[float], text: str, metadata: dict = None
    ) -> None:
        # get connection
        conn = self.postgresql_connection_provider.get_connection()

        # start transaction
        cursor = conn.cursor()

        # transform json to string
        metadata_json = json.dumps(metadata) if metadata else None

        # insert new model
        cursor.execute(
            """
            INSERT INTO %s (text, embedding, model_id, metadata)
            VALUES (%s, %s, %s, %s)
            """,
            (
                'source_chunks',
                text,
                embedding,
                self.model_id,
                metadata_json,
            ),
        )

        conn.commit()
        cursor.close()

    def bulk_ingest(
        self, data: list[tuple[int, list[float], str, dict]]
    ) -> None:
        raise NotImplementedError('Bulk ingestion is not supported yet.')
