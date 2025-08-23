#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import click

from lib.database import SimpleVectorDatabase
from lib.documents import SourceDocument
from lib.embedding import DeepseekQwen15BEmbeddingGenerator
from lib.settings import DemoSettingsProvider
from lib.sources import SourceConverter

DEFAULT_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
DEFAULT_EMBEDDING_DIM = 1536
INTERNAL_WORKDIR = '/work'


@click.command(help='Import demo data into PostgreSQL database')
@click.option(
    '-f',
    '--file',
    'files',
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help='Files to import (may be used multiple times, e.g., --file file1.csv --file file2.csv)',
)
@click.option(
    '--model',
    default=DEFAULT_MODEL,
    show_default=True,
    help='Model to use for importing',
)
@click.option(
    '--embedding-dim',
    'embedding_dim',
    default=DEFAULT_EMBEDDING_DIM,
    show_default=True,
    type=int,
    help='Embedding dimension to use for importing',
)
def main(files: tuple[str, ...], model: str, embedding_dim: int) -> None:
    # Mirror original task behavior for missing files
    if not files:
        click.echo('No files provided. Exiting.')
        sys.exit(0)

    # Initialize database
    settings_provider = DemoSettingsProvider()
    vector_database = SimpleVectorDatabase.from_settings_provider(
        settings_provider
    )

    # init converter
    converter = SourceConverter(sources=list(files))

    for file in converter.ingestion_ready_sources():
        # init document
        document = SourceDocument(
            source_filepath=file,
            max_chunk_tokens=DEFAULT_EMBEDDING_DIM,
            database=vector_database,
        )

        # ingest chunks
        ordered_chunks = list(document.enriched_chunks())
        ordered_texts = [chunk.text for chunk in ordered_chunks]
        ordered_embeddings = list(
            DeepseekQwen15BEmbeddingGenerator(
                texts=ordered_texts,
                model_name=model,
                embedding_dim=embedding_dim,
            ).generate()
        )
        for i, chunk in enumerate(ordered_chunks):
            vector_database.insert_source_chunk(
                source_id=document.source_id,
                model_id=vector_database.create_or_lookup_model_id(
                    model_name=model
                ),
                embedding=ordered_embeddings[i],
                metadata=chunk.metadata,
                text=chunk.text,
            )


if __name__ == '__main__':
    main()
