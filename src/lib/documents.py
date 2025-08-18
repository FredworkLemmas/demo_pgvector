import attrs
from pathlib import Path
from typing import Iterator
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import BaseChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

MAX_CHUNK_TOKENS = 512


@attrs.define
class SourceDocument(object):
    source_filepath: str
    metadata: dict | None = None
    chunker_class: type[BaseChunker] | None = None
    max_chunk_tokens: int | None = None
    model_name: str | None = None

    def __attrs_post_init__(self):
        self.chunker_class = self.chunker_class or HybridChunker()
        self.metadata = self.metadata or {}
        self.max_chunk_tokens = self.max_chunk_tokens or MAX_CHUNK_TOKENS
        self.model_name = (
            self.model_name or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
        )

    def enriched_chunks(self):
        for n, chunk in enumerate(self._chunk_iterator()):
            yield SourceDocumentChunk(
                text=chunk.text,
                source_document=self,
                metadata={
                    **chunk.metadata,
                    'source_filepath': self.source_filepath,
                    'chunk_tokenizer_model': self.model_name,
                    'chunk_size': self.max_chunk_tokens,
                    'chunk_number': n,
                },
            )

    def _as_docling_document(self):
        # Validate that the file exists
        if not Path(self.source_filepath).exists():
            raise FileNotFoundError(f'File not found: {self.source_filepath}')
        return (
            DocumentConverter()
            .convert(source=Path(self.source_filepath))
            .document
        )

    def _chunk_iterator(self) -> Iterator:
        """
        split doc into chunks
        """
        tokenizer = HuggingFaceTokenizer.from_pretrained(
            self.model_name, max_tokens=self.max_chunk_tokens
        )

        document = self._as_docling_document()
        chunker = HybridChunker(tokenizer=tokenizer)

        return chunker.chunk(document)


@attrs.define
class SourceDocumentChunk:
    text: str
    source_document: SourceDocument
    metadata: dict
