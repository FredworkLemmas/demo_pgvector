#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import sys

from lib.database import SimpleVectorDatabase
from lib.embedding import DeepseekQwen15BEmbeddingGenerator
from lib.settings import DemoSettingsProvider

DEFAULT_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
DEFAULT_EMBEDDING_DIM = 1536


@click.command(help='Search for document chunks similar to a given prompt')
@click.option(
    '--prompt',
    required=True,
    help='The text prompt to search for similar chunks',
)
@click.option(
    '--model',
    default=DEFAULT_MODEL,
    show_default=True,
    help='Model to use for generating embeddings',
)
@click.option(
    '--top-k',
    'top_k',
    default=10,
    show_default=True,
    type=int,
    help='Number of top similar chunks to retrieve',
)
@click.option(
    '--similarity-threshold',
    'similarity_threshold',
    default=0.7,
    show_default=True,
    type=float,
    help='Minimum similarity score threshold (0.0 to 1.0)',
)
def main(
    prompt: str,
    model: str,
    top_k: int,
    similarity_threshold: float,
) -> None:
    """Search for document chunks similar to the provided prompt."""

    # Validate similarity threshold
    if not (0.0 <= similarity_threshold <= 1.0):
        click.echo(
            'Error: similarity-threshold must be between 0.0 and 1.0', err=True
        )
        sys.exit(1)

    # Initialize database
    settings_provider = DemoSettingsProvider()
    vector_database = SimpleVectorDatabase.from_settings_provider(
        settings_provider
    )

    # Generate embedding for the prompt
    click.echo(
        f"Generating embedding for prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'"
    )

    try:
        embedding_generator = DeepseekQwen15BEmbeddingGenerator(
            texts=[prompt],
            model_name=model,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
        )

        # Get the embedding for the prompt
        prompt_embedding = next(embedding_generator.generate())

    except Exception as e:
        click.echo(f'Error generating embedding: {e}', err=True)
        sys.exit(1)

    # Search for similar chunks
    click.echo(
        f'Searching for similar chunks (top_k={top_k}, threshold={similarity_threshold})...'
    )

    try:
        similar_chunks = vector_database.retrieve_similar_source_chunks(
            query_embedding=prompt_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if not similar_chunks:
            click.echo(
                'No similar chunks found above the similarity threshold.'
            )
            return

        # Display results
        click.echo(f'\nFound {len(similar_chunks)} similar chunks:\n')

        for i, chunk in enumerate(similar_chunks, 1):
            click.echo(f'--- Chunk {i} ---')
            click.echo(f'Chunk ID: {chunk["chunk_id"]}')
            click.echo(f'Similarity Score: {chunk["similarity_score"]:.4f}')

            # Display metadata if available
            if chunk.get('metadata'):
                metadata = chunk['metadata']
                click.echo('Metadata:')
                for key, value in metadata.items():
                    if value is not None:
                        click.echo(f'  {key}: {value}')

            # Display chunk text (truncated if too long)
            chunk_text = chunk['chunk_text']
            if len(chunk_text) > 500:
                display_text = chunk_text[:500] + '...'
            else:
                display_text = chunk_text

            click.echo(f'Text: {display_text}')
            click.echo()  # Empty line for separation

    except Exception as e:
        click.echo(f'Error searching for similar chunks: {e}', err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
