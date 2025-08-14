#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import click

from lib.database import PgvectorDatabaseConnectionProvider
from lib.ingestor import PgvectorIngestor
from lib.settings import DemoSettingsProvider
from lib.sources import SourceConverter


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_EMBEDDING_DIM = 1536
INTERNAL_WORKDIR = '/work'


@click.command(help="Import demo data into PostgreSQL database")
@click.option(
    "-f",
    "--file",
    "files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help="Files to import (may be used multiple times, e.g., --file file1.csv --file file2.csv)",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="Model to use for importing",
)
@click.option(
    "--embedding-dim",
    "embedding_dim",
    default=DEFAULT_EMBEDDING_DIM,
    show_default=True,
    type=int,
    help="Embedding dimension to use for importing",
)
def main(files: tuple[str, ...], model: str, embedding_dim: int) -> None:
    # Mirror original task behavior for missing files
    if not files:
        click.echo("No files provided. Exiting.")
        sys.exit(0)

    # Initialize settings and database/ingestor components
    settings_provider = DemoSettingsProvider()
    connection_provider = PgvectorDatabaseConnectionProvider.from_settings_provider(
        settings_provider
    )
    ingestor = PgvectorIngestor(
        postgresql_connection_provider=connection_provider,
        model_name=model,
        embedding_dim=embedding_dim,
    )

    # Output selected parameters (as in the original task)
    click.echo(f"files: {list(files)}")
    click.echo(f"model: {model}")

    converter = SourceConverter()
    for file in files:
        is_convertible = converter.is_convertible(file)
        print(f"file: {file}, convertible: {is_convertible}")
        if is_convertible:
            print(f"Converting file: {file}")
            converted_file = converter.convert(file)
            print(f'Converted file: {file} to {converted_file}')


    # If ingestion should actually occur and PgvectorIngestor exposes an API like
    # ingest_file(...) or ingest_files(...), you can uncomment and adapt this:
    #
    # for path in files:
    #     ingestor.ingest_file(path)
    # click.echo("Import completed.")


if __name__ == "__main__":
    main()
