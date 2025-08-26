import sys
import click
from vllm import SamplingParams

from lib.constants import DEFAULT_MODEL, DEFAULT_EMBEDDING_DIM
from lib.database import SimpleVectorDatabase
from lib.embedding import DeepseekQwen15BEmbeddingGenerator
from lib.llms import LLMManager
from lib.settings import DemoSettingsProvider

SIMILARITY_THRESHOLD = 0.01


def _generate_embedding(prompt, model) -> list[float]:
    """Generate embedding for the prompt."""
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

    return prompt_embedding


def _get_vector_database():
    """Initialize the vector database."""
    # Initialize database
    settings_provider = DemoSettingsProvider()
    vector_database = SimpleVectorDatabase.from_settings_provider(
        settings_provider
    )
    return vector_database


def _fetch_similar_chunks(prompt_embedding, top_k, similarity_threshold):
    """Fetch similar chunks from the database."""
    vector_database = _get_vector_database()
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

        return similar_chunks

    except Exception as e:
        click.echo(f'Error searching for similar chunks: {e}', err=True)
        sys.exit(1)


def _get_aggregated_chunk_text(similar_chunks):
    """Aggregate chunk text from similar chunks."""
    agg_chunk_text = ''
    if similar_chunks:
        for chunk in similar_chunks:
            m = chunk['metadata']
            txt = chunk['chunk_text']

            print(f'chunk metadata: {chunk["metadata"]}')
            agg_chunk_text += f"""
Excerpt from "{m['title']}", by {m['author']}, published in {m['publication_date']}:
>>>
{txt}
<<<


"""
    return agg_chunk_text


def _get_contextualized_prompt(prompt, model):
    """Get contextualized prompt."""
    # generate embedding for prompt
    brief_prompt = prompt[:100] + '...' if len(prompt) > 100 else prompt
    click.echo(f"Generating embedding for prompt: '{brief_prompt}'")
    prompt_embedding = _generate_embedding(prompt, model)

    similar_chunks = _fetch_similar_chunks(
        prompt_embedding, top_k=5, similarity_threshold=SIMILARITY_THRESHOLD
    )

    agg_chunk_text = ''
    if similar_chunks:
        agg_chunk_text = _get_aggregated_chunk_text(similar_chunks)

    if not agg_chunk_text:
        return f"""
You are a helpful assistant with a library that you refer to as "the Archives"
designed to be helpful for the sorts
of questions whose investigations you are likely to be asked to contribute to.

However, when you consulted the library for information related to the prompt
"{prompt}", you found no relevant information.

Please respond to the following prompt with a disclaimer that notes the lack of
information and, if you can appropriately determine the category, genre,
author, etc. for information that might prove helpful, suggest how the library
might be expanded to include more information.

The prompt to which you must respond is:
    "{prompt}"
"""

    contextualized_prompt = f"""
You are a helpful assistant with a library that you refer to as "the Archives"
designed to be helpful for the sorts
of questions whose investigations you are likely to be asked to contribute to.

When you consulted the library for information related to the prompt
"{prompt}", you found the following relevant information:
'''
{agg_chunk_text}
'''

Please respond to the following prompt and, if the information from the library
is relevant, use the information to respond to the prompt. Include references
to the source data by including the few lines of text from
the source data that contain the information you are referring to along with the
author, title of the work and the publication date.

The prompt to which you must respond is:
    "{prompt}"
"""

    return contextualized_prompt


@click.command()
@click.option(
    '--prompt', required=True, help='The prompt to generate text from'
)
@click.option(
    '--model',
    default=DEFAULT_MODEL,
    help='The model to use for text generation (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)',
)
@click.option(
    '--max-tokens',
    default=5000,
    help='Maximum number of tokens to generate (default: 100)',
)
@click.option(
    '--temperature', default=0.8, help='Sampling temperature (default: 0.8)'
)
def generate_text(prompt, model, max_tokens, temperature):
    """Generate text using vLLM with the specified prompt and model."""

    try:
        # Initialize the LLM
        click.echo(f'Loading model: {model}')
        llm = LLMManager(model_name=model).instance()

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens
        )

        contextualized_prompt = _get_contextualized_prompt(prompt, model)

        # Generate text
        click.echo(f"Generating text for prompt: '{contextualized_prompt}'")
        outputs = llm.generate([contextualized_prompt], sampling_params)

        # Display the result
        generated_text = outputs[0].outputs[0].text
        click.echo('\n' + '=' * 50)
        click.echo('GENERATED TEXT:')
        click.echo('=' * 50)
        click.echo(generated_text)
        click.echo('=' * 50)

    except Exception as e:
        click.echo(f'Error({e.__class__.__name__}): {e}', err=True)
        raise click.Abort()


if __name__ == '__main__':
    generate_text()
