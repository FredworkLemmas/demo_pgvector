from abc import ABC, abstractmethod
from typing import Iterator


class EmbeddingGenerator(ABC):
    @abstractmethod
    def generate(self) -> Iterator[list[float]]:
        pass


class TextGenerator(ABC):
    @abstractmethod
    def generate(self) -> Iterator[str]:
        pass


class EmbeddingIngestor(ABC):
    @abstractmethod
    def ingest(
        self,
        embedding: list[float],
        text: str,
        metadata: dict = None
    ) -> None:
        pass

    @abstractmethod
    def bulk_ingest(
        self,
        data: list[tuple[int, list[float], str, dict]]
    ) -> None:
        pass


class EmbeddingQueryProvider(ABC):
    @abstractmethod
    def similar(
        self,
        embedding: list[float],
        limit: int = 5,
        fields: Iterator[str] = ('embedding', 'text', 'metadata')
    ) -> Iterator[tuple]:
        pass


class LLMModelProvider(ABC):
    @abstractmethod
    def generate(self) -> Iterator[str]:
        pass


class PostgresqlConnectionProvider(ABC):
    @abstractmethod
    def get_connection(self):
        pass


class SettingsProvider(ABC):
    @abstractmethod
    def get_settings(self) -> dict:
        pass
