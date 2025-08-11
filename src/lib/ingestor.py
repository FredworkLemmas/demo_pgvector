from .interfaces import EmbeddingIngestor


class PgvectorIngestor(EmbeddingIngestor):
    def ingest(
        self,
        embedding: list[float],
        text: str,
        metadata: dict = None
    ) -> None:
        pass