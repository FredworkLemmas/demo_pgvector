import datetime
import attrs
import yaml
import typing
from pathlib import Path
from typing import Iterator

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import BaseChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

if typing.TYPE_CHECKING:
    from lib.database import SimpleVectorDatabase
    from lib.sources import SourceCollection

MAX_CHUNK_TOKENS = 512
TEXT_TYPE__FICTION = 1
TEXT_TYPE__NONFICTION = 2


@attrs.define
class SourceDocument(object):
    database: 'SimpleVectorDatabase'
    source: typing.Optional[typing.Union['SourceCollection', None]] = None
    metadata: dict | None = None
    chunker_class: type[BaseChunker] | None = None
    max_chunk_tokens: int | None = None
    model_name: str | None = None
    model_id: int | None = None
    source_id: int | None = None

    def __attrs_post_init__(self):
        self.chunker_class = self.chunker_class or HybridChunker()
        self.metadata = self.metadata or self._metadata_from_file()
        self.max_chunk_tokens = self.max_chunk_tokens or MAX_CHUNK_TOKENS
        self.model_name = (
            self.model_name or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
        )
        self.model_id = (
            self.model_id
            or self.database.create_or_lookup_model_id(self.model_name)
        )
        # register source
        self.source_id = self.database.create_or_lookup_source(self)

    def enriched_chunks(self):
        print(f'source doc metadata: {self.metadata}')
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
        embeddable_filepath = self.source.embeddable_filepath
        if not Path(embeddable_filepath).exists():
            raise FileNotFoundError(f'File not found: {embeddable_filepath}')
        return (
            DocumentConverter()
            .convert(source=Path(embeddable_filepath))
            .document
        )

    def _metadata_from_file(self):
        p = Path(self.source.source_filepath)
        metadata_path = p.with_name(p.name + '.meta.yml')
        print(f'looking for metdata file: {metadata_path}')
        if metadata_path.exists():
            print('...loading metadatafile')
            return yaml.safe_load(metadata_path.read_text())
        return {}


@attrs.define
class SourceDocumentChunk:
    text: str
    source_document: SourceDocument
    metadata: dict
