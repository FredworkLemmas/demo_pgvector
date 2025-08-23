import click
from vllm import SamplingParams

from lib.llms import LLMManager


@click.command()
@click.option(
    '--prompt', required=True, help='The prompt to generate text from'
)
@click.option(
    '--model',
    default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    help='The model to use for text generation (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)',
)
@click.option(
    '--max-tokens',
    default=500,
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

        # Generate text
        click.echo(f"Generating text for prompt: '{prompt}'")
        outputs = llm.generate([prompt], sampling_params)

        # Display the result
        generated_text = outputs[0].outputs[0].text
        click.echo('\n' + '=' * 50)
        click.echo('GENERATED TEXT:')
        click.echo('=' * 50)
        click.echo(generated_text)
        click.echo('=' * 50)

    except Exception as e:
        click.echo(f'Error: {e}', err=True)
        raise click.Abort()


if __name__ == '__main__':
    generate_text()
