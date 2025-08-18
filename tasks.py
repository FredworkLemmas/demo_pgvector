import os
from invocate import task

# from lib.database import PgvectorDatabaseConnectionProvider
# from lib.ingestor import PgvectorIngestor
# from lib.settings import DemoSettingsProvider


@task(namespace="env", name="start")
def start_docker_compose_env(c):
    c.run("docker compose up -d --remove-orphans")


@task(namespace="env", name="stop")
def stop_docker_compose_env(c):
    c.run("docker compose down --remove-orphans")


@task(namespace="env", name="build", pre=[stop_docker_compose_env])
def build_runner_container(c):
    c.run("docker compose build runner")


@task(
    namespace="demo",
    name="import",
    iterable=["file"],
    help={
        "file": (
            "Files to import (may be used multiple times, e.g., "
            "--file file1.csv --file file2.csv)"
        ),
        "model": (
            "Model to use for importing (default: "
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)"
        ),
        "embedding-dim": (
            "Embedding dimension to use for importing (default: 1536)"
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
        print("No files provided. Exiting.")
        return
    # make sure the input file directory exists and copy source files to it
    c.run("install -d /tmp/demo_pgvector/files")
    container_files = []
    for f in file:
        c.run(f"cp {f} /tmp/demo_pgvector/files/")
        container_files.append("/files/{}".format(os.path.basename(f)))

    # define model, embedding dimensions
    model = model or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    embedding_dim = embedding_dim or 1536

    # build opts and run in container
    file_opts = [f"-f {f}" for f in container_files]
    model_opts = [f"--model {model}"] if model else []
    dim_opts = [f"--embedding-dim {embedding_dim}"] if embedding_dim else []
    c.run(
        "docker compose run runner python3 cli/import_doc.py {}".format(
            " ".join(list(file_opts + model_opts + dim_opts))
        )
    )
