#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import click

from lib.documents import SourceDocument
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

    # Initialize settings and database/ingestor components
    # settings_provider = DemoSettingsProvider()
    # connection_provider = (
    #     PgvectorDatabaseConnectionProvider.from_settings_provider(
    #         settings_provider
    #     )
    # )
    # ingestor = PgvectorIngestor(
    #     postgresql_connection_provider=connection_provider,
    #     model_name=model,
    #     embedding_dim=embedding_dim,
    # )

    converter = SourceConverter(sources=list(files))

    print(f'Converting files: {list(files)}')
    print(f'Ingestion ready files: {converter.ingestion_ready_sources()}')

    for file in converter.ingestion_ready_sources():
        document = SourceDocument(
            source_filepath=file, max_chunk_tokens=DEFAULT_EMBEDDING_DIM
        )
        for i, chunk in enumerate(document._raw_chunk_iterator()):
            click.echo(f'Chunk {i}: \n{chunk.text}')
            # click.echo(f'chunk dict: {chunk.__dict__}')
            if i > 4:
                break


if __name__ == '__main__':
    main()
