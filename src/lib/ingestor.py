import json
import attrs

from .interfaces import EmbeddingIngestor, PostgresqlConnectionProvider


@attrs.define
class PgvectorIngestor(EmbeddingIngestor):
    postgresql_connection_provider: PostgresqlConnectionProvider
    model_id: int
    embedding_dim: int

    def ingest(
        self,
        embedding: list[float],
        text: str,
        metadata: dict = None
    ) -> None:
        # get connection
        conn = self.postgresql_connection_provider.get_connection()

        # start transaction
        cursor = conn.cursor()

        # transform json to string
        metadata_json = json.dumps(metadata) if metadata else None
        cursor.execute(
            """
            INSERT INTO text_embeddings (text, embedding, model_id, metadata)
            VALUES (%s, %s, %s, %s)
            """, (text, embedding, self.model_id, metadata_json))

        conn.commit()
        cursor.close()
        pass

    def bulk_ingest(
        self,
        data: list[tuple[int, list[float], str, dict]]
    ) -> None:
        pass

    def create_or_lookup_model_id(self, model_name: str) -> int:
        # get connection
        conn = self.postgresql_connection_provider.get_connection()
        cursor = conn.cursor()

        try:
            # First, try to find existing model
            cursor.execute(
                "SELECT id FROM models WHERE name = %s",
                (model_name,)
            )
            result = cursor.fetchone()

            if result:
                # Model exists, return its id
                return result[0]
            else:
                # Model doesn't exist, create it
                cursor.execute(
                    "INSERT INTO models (name, embedding_dim) VALUES (%s, %s) RETURNING id",
                    (model_name, self.embedding_dim)
                )
                model_id = cursor.fetchone()[0]
                conn.commit()
                return model_id

        finally:
            cursor.close()

    @classmethod
    def from_postgresql_connection_provider(
            cls, connection_provider: PostgresqlConnectionProvider) \
            -> 'PgvectorIngestor' :
        return cls(postgresql_connection_provider=connection_provider)
