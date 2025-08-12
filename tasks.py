from invocate import task

from lib.database import PgvectorDatabaseConnectionProvider
from lib.ingestor import PgvectorIngestor
from lib.settings import DemoSettingsProvider


@task(namespace='env', name='start')
def start_docker_compose_env(c):
    c.run('docker compose up -d --remove-orphans')


@task(namespace='env', name='stop')
def stop_docker_compose_env(c):
    c.run('docker compose down --remove-orphans')


@task(namespace='env', name='build')
def build_runner_container(c):
    c.run('docker compose build runner')


@task(
    namespace='demo',
    name='import',
    iterable=['file'],
    help={
        'file': (
            'Files to import (may be used multiple times, e.g., '
            '--file file1.csv --file file2.csv)'
        ),
        'model': (
            'Model to use for importing (default: '
            'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)'
        ),
        'embedding-dim': (
            'Embedding dimension to use for importing (default: 1536)'
        )
    }
)
def import_demo_data(c, file=None, model=None, embedding_dim=1536, ):
    """Import demo data into PostgreSQL database"""
    if not file:
        print('No files provided. Exiting.')
        return
    settings_provider = DemoSettingsProvider()
    connection_provider = \
        PgvectorDatabaseConnectionProvider.from_settings_provider(
            settings_provider)
    ingestor = PgvectorIngestor(
        postgresql_connection_provider=connection_provider,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        embedding_dim=1536
    )
    print(f'files: {file}')
    print(f'model: {model}')
