import json
import attrs

from .interfaces import EmbeddingIngestor, PostgresqlConnectionProvider


@attrs.define
class PgvectorIngestor(EmbeddingIngestor):
    postgresql_connection_provider: PostgresqlConnectionProvider
    model_name: str
    embedding_dim: int

    def get_model_id(self) -> int:
        return self.create_or_lookup_model_id(self.model_name)

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
                self.embeddings_tablename(),
                text,
                embedding,
                self.get_model_id(),
                metadata_json,
            ),
        )

        conn.commit()
        cursor.close()

    def bulk_ingest(
        self, data: list[tuple[int, list[float], str, dict]]
    ) -> None:
        raise NotImplementedError('Bulk ingestion is not supported yet.')

    def embeddings_tablename(self, model_id: int | None = None) -> str:
        model_id = model_id or self.get_model_id()
        return f'text_embeddings_{model_id}'

    def create_or_lookup_model_id(self, model_name: str) -> int:
        # get connection
        conn = self.postgresql_connection_provider.get_connection()
        cursor = conn.cursor()

        try:
            # First, try to find existing model
            cursor.execute(
                'SELECT id FROM models WHERE name = %s', (model_name,)
            )
            result = cursor.fetchone()

            if result:
                # Model exists, return its id
                return result[0]
            else:
                # Model doesn't exist, create it
                cursor.execute(
                    'INSERT INTO models (name, embedding_dim) VALUES (%s, %s) RETURNING id',
                    (model_name, self.embedding_dim),
                )
                model_id = cursor.fetchone()[0]

                # # create new table for the vectors
                # cursor.execute(
                #     """
                #     CREATE TABLE IF NOT EXISTS %s (
                #         id SERIAL PRIMARY KEY,
                #         text TEXT NOT NULL,
                #         embedding VECTOR(%s),
                #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                #         metadata JSONB
                #     );
                #     """,
                #     (self.embeddings_tablename(model_id), self.embedding_dim),
                # )
                conn.commit()
                return model_id

        finally:
            cursor.close()

    @classmethod
    def from_postgresql_connection_provider(
        cls, connection_provider: PostgresqlConnectionProvider
    ) -> 'PgvectorIngestor':
        return cls(postgresql_connection_provider=connection_provider)
