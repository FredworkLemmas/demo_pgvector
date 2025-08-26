import os
from invocate import task


@task(namespace='env', name='init')
def init_environment(c):
    """Initialize environment for running demo."""
    # make sure input file directory exists
    c.run('install -d /tmp/demo_pgvector/files')


@task(namespace='env', name='start', pre=[init_environment])
def start_docker_compose_env(c):
    """Start docker compose environment."""
    c.run('docker compose up -d --remove-orphans')


@task(namespace='env', name='stop')
def stop_docker_compose_env(c):
    """Stop docker compose environment."""
    c.run('docker compose down --remove-orphans')


@task(namespace='env', name='build', pre=[stop_docker_compose_env])
def build_runner_container(c):
    """Build runner container."""
    c.run('docker compose build runner')


@task(namespace='env', name='cleanup')
def cleanup_demo_env(c):
    """Cleanup demo environment."""
    c.run('sudo rm -rf /tmp/demo_pgvector')


@task(
    namespace='demo',
    name='import',
    iterable=['file'],
    pre=[cleanup_demo_env, init_environment],
    help={
        'file': (
            'Files to import (may be used multiple times, e.g., '
            '--file file1.pdf --file file2.epub)'
        ),
        'model': (
            'Model to use for importing (default: '
            'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)'
        ),
    },
)
def import_demo_data(c, file=None, model=None):
    """Import demo data into PostgreSQL database"""
    # sanity check
    if not file:
        print('No files provided. Exiting.')
        return

    # copy source files to input file directory
    container_files = []
    for f in file:
        # copy file and append to container_files
        c.run(f'cp {f} /tmp/demo_pgvector/files/')
        container_files.append('/files/{}'.format(os.path.basename(f)))

        # copy metadata file if it exists
        if os.path.exists(f'{f}.meta.yml'):
            c.run(f'cp {f}.meta.yml /tmp/demo_pgvector/files/')

    # define model, embedding dimensions
    model = model or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

    # build opts and run in container
    file_opts = [f'-f {f}' for f in container_files]
    model_opts = [f'--model {model}'] if model else []
    c.run(
        'docker compose run runner python3 cli/import_doc.py {}'.format(
            ' '.join(list(file_opts + model_opts))
        )
    )


@task(
    namespace='demo',
    name='search',
    help={
        'prompt': ('The text prompt used to search for similar chunks'),
        'model': (
            'Model to use for importing (default: '
            'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)'
        ),
        'limit': 'Max number of results to return (default: 10)',
        'threshold': (
            'Similarity threshold to use for filtering results (default: 0.7)'
        ),
    },
)
def search_demo_data(c, prompt, model=None, limit=10, threshold=0.7):
    """Search for similar chunks in the demo database."""
    model = model or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

    opts = [
        f'--prompt "{prompt}"',
        f'--model {model}',
        f'--top-k {limit}',
        f'--similarity-threshold {threshold}',
    ]

    c.run(
        'docker compose run runner python3 cli/search_doc_chunks.py {}'.format(
            ' '.join(opts)
        )
    )


@task(
    namespace='demo',
    name='generate',
    help={
        'prompt': ('The text prompt used to generate text'),
        'model': (
            'Model to use for generating text (default: '
            'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)'
        ),
    },
)
def generate_text(c, prompt, model=None):
    """Generate text using LLM with the specified prompt and model."""
    model = model or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

    opts = [
        f'--prompt "{prompt}"',
        f'--model {model}',
    ]
    c.run(
        'docker compose run runner python3 cli/generate_text.py {}'.format(
            ' '.join(opts)
        )
    )


@task(namespace='purge', name='db', pre=[stop_docker_compose_env])
def purge_db(c):
    """Purge all data from PostgreSQL database"""
    c.run('docker volume rm demo_pgvector_postgres_data')


@task(namespace='purge', name='vllm', pre=[stop_docker_compose_env])
def purge_vllm_cache(c):
    """Purge all data from PostgreSQL database"""
    c.run('docker volume rm model_cache')


@task(namespace='example', name='load_and_query_1')
def run_example(c):
    """Run example workflow."""
    # purge database
    c.run('nv env.start')
    c.run('nv purge.db')
    c.run('nv env.stop')
    c.run('nv env.start')

    # import epub files to vector database
    epub_files = [
        f
        for f in os.listdir('examples')
        if os.path.isfile(os.path.join('examples', f))
        and f.lower().endswith('.epub')
    ]
    file_opts = ' '.join([f'--file examples/{f}' for f in epub_files])
    c.run(f'nv demo.import {file_opts}')

    c.run(
        'nv demo.generate -p '
        '"Are robots that are depicted in science fiction generally '
        'friendly to humans? If not, why not?"'
    )
