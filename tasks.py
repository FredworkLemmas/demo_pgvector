import os
from invocate import task


@task(namespace='env', name='init')
def init_environment(c):
    # make sure input file directory exists
    c.run('install -d /tmp/demo_pgvector/files')


@task(namespace='env', name='start', pre=[init_environment])
def start_docker_compose_env(c):
    c.run('docker compose up -d --remove-orphans')


@task(namespace='env', name='stop')
def stop_docker_compose_env(c):
    c.run('docker compose down --remove-orphans')


@task(namespace='env', name='build', pre=[stop_docker_compose_env])
def build_runner_container(c):
    c.run('docker compose build runner')


@task(namespace='env', name='cleanup')
def cleanup_demo_env(c):
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
        'embedding-dim': (
            'Embedding dimension to use for importing (default: 1536)'
        ),
    },
)
def import_demo_data(
    c,
    file=None,
    model=None,
    embedding_dim=1536,
):
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
    embedding_dim = embedding_dim or 1536

    # build opts and run in container
    file_opts = [f'-f {f}' for f in container_files]
    model_opts = [f'--model {model}'] if model else []
    dim_opts = [f'--embedding-dim {embedding_dim}'] if embedding_dim else []
    c.run(
        'docker compose run runner python3 cli/import_doc.py {}'.format(
            ' '.join(list(file_opts + model_opts + dim_opts))
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
        'embedding-dim': (
            'Embedding dimension to use for importing (default: 1536)'
        ),
        'limit': 'Max number of results to return (default: 10)',
        'threshold': (
            'Similarity threshold to use for filtering results (default: 0.7)'
        ),
    },
)
def search_demo_data(
    c, prompt, model=None, embedding_dim=1536, limit=10, threshold=0.7
):
    model = model or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    embedding_dim = embedding_dim or 1536

    opts = [
        f'--prompt "{prompt}"',
        f'--model {model}',
        f'--embedding-dim {embedding_dim}',
        f'--top-k {limit}',
        f'--similarity-threshold {threshold}',
    ]

    c.run(
        'docker compose run runner python3 cli/search_doc_chunks.py {}'.format(
            ' '.join(opts)
        )
    )


@task(namespace='db', name='purge', pre=[stop_docker_compose_env])
def purge_db(c):
    """Purge all data from PostgreSQL database"""
    c.run('docker volume rm demo_pgvector_postgres_data')


@task(namespace='vllm', name='purge', pre=[stop_docker_compose_env])
def purge_vllm_cache(c):
    """Purge all data from PostgreSQL database"""
    c.run('docker volume rm model_cache')
