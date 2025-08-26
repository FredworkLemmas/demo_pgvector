# DEMO: pg_vector
My previous experiment with a Retrieval-Augmented Generation system used [Chroma](https://github.com/chroma-core/chroma)
for semantic search and it dealt purely in text files.  It was also slow, it had
to download model weights every time I ran it, and it didn't run in a container.

With this demo, I wanted:

* to work with a database that wasn't constrained by available system memory
* to expand the input formats I could work with
* to run in a container
* to be able to run it without needing to download model weights every time

This is a simple proof-of-concept that will use:

* PostgreSQL as a vector database via the [PgVector](https://github.com/pgvector/pgvector) extension.
* [Docling](https://github.com/docling-project/docling) for its document chunking ability.
* Pandoc for EPUB-to-Markdown conversion.
* [vLLM](https://github.com/vllm-project/vllm) with [DeepSeek Qwen 1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) for embedding vector generation and
  similarity search.
* local caching of model weights for faster inference and embedding
  generation.

Please note that this is a proving ground and is not intended for
production use.  THERE ARE NO TESTS!  HERE BE DRAGONS!

## Usage instructions for Linux users
_NOTE that may only apply for Ubuntu 24.04 users as of August 22, 2025.  It
further may only work for folks who have a couple of Nvidia 3060TIs._

### Prerequisites
There are a few things you need to install before you can run this.
* probably a Nvidia GPU in a card that has at least 8GB ram.
* Docker (the recent-ish version that include docker-compose but make
  you split it up into "docker" and "compose")
* modern Nvidia and CUDA drivers (I'm at v580)

### Installation
NOTE that certain level of familiarity with Python, Docker and Virtualenv
is assumed.

#### To get started:
* clone this repo
* create a virtualenv for the project
* install the developer requirements with:

  ```pip install -r requirements.dev.txt```

#### To play around:
* run `nv -l` to get a list of available commands:

```text
(demo-pgvector) fred@jamma:~/Projects/demo_pgvector$ nv -l
Available tasks:

  demo.generate              Generate text using LLM with the specified prompt and model.
  demo.import                Import demo data into PostgreSQL database
  demo.search                Search for similar chunks in the demo database.
  env.build                  Build runner container.
  env.cleanup                Cleanup demo environment.
  env.init                   Initialize environment for running demo.
  env.start                  Start docker compose environment.
  env.stop                   Stop docker compose environment.
  example.load-and-query-1   Run example workflow.
  purge.db                   Purge all data from PostgreSQL database
  purge.vllm                 Purge all data from PostgreSQL database
```

##### Main demo-ables
The `demo.import` task will import the demo data into the database.

The `demo.search` task will search the database for a given query.

The `demo.generate` task will generate a new document based on a given query,
optionally querying the semantic database for context.

##### Clean up
The `purge.db` task will purge all data from the database.

The `purge.vllm` task will purge all data from the vLLM cache.

##### Environment
The `env.*` tasks control the docker environment.

##### Kitchen Sink
To demostrate the complete workflow, the `example.load-and-query-1` task will:

* purge the database
* import 5 books about robots that I found in [Project Gutenberg](https://www.gutenberg.org/ebooks/search/?query=robot)
* and then do a RAG query about how friendly sci-fi robots are

The relevant bits of a TON of output are:
```text
(demo-pgvector) fred@jamma:~/Projects/demo_pgvector$ nv example.load-and-query-1

. . .

    [ TONS OF OUTPUT ]

. . .

Generating embedding for prompt: 'Are robots that are depicted in science fiction generally friendly to humans? If not, why not?'

. . .

Generating text for prompt: '
You are a helpful assistant with a library that you refer to as "the Archives"
designed to be helpful for the sorts
of questions whose investigations you are likely to be asked to contribute to.

When you consulted the library for information related to the prompt
"Are robots that are depicted in science fiction generally friendly to humans? If not, why not?", you found the following relevant information:
'''

Excerpt from "Second Variety", by Philip K. Dick, published in 1953:
>>>
This ebook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this ebook or online at [www.gutenberg.org]. If you are not located in the United States, you will have to check the laws of the country where you are located before using this eBook.

. . .

==================================================
GENERATED TEXT:
==================================================
</think>

The depiction of robots in science fiction often reflects their origin and
utility. Those robots, such as the "Second Variety" by Philip K. Dick, are
typically autonomous and designed for specific purposes, often out of the
scope of human adaptation. While the original "After World's End" by Jack
Williamson portrays Malgarth, a robot resembling a Maltese human, the characters
generally exhibit a lack of direct interaction with humans. Instead, their
actions are consequences of their design and mission. The "Black Dancer" by
Par macCarron and Technically, in the 1939 "After World's End," portray a robot
manipulative and offensive. While these examples may not directly answer the
query, they suggest that robot developers may aim to create friendly interaction
by structuring their missions around human needs. However, the text in the
articles does not provide clear evidence of friendly relationships, which would
likely depend on the specific robots' designs and missions.
==================================================

```

## Key takeaways
This proof-of-concept proved a bunch of useful concepts:

* PGVector similarity search is plenty fast!
* Qwen 1.5B is impressive in a RAG pipeline and it fits in my 8GB GPU.
* vLLM caching prevented multiple downloads (once I had it configured correctly)
  and it looks like it can store/load the KV cache to disk to speed up inference
  for prompts that begin with the same string of text.
* Docling seems pretty capable for chunking markdown.

# ADDITIONAL NOTES

* there is an error that seems to occur only after inference is complete and,
  so far, I've not had much luck getting rid of it:
    ```text
    ERROR 08-26 03:58:32 [core_client.py:562] Engine core proc EngineCore_0 died unexpectedly, shutting down client.
    ```
