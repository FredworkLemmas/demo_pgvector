# DEMO: pg_vector

This is a simple proof-of-concept that will use:

* PostgreSQL as a vector database via the PgVector extension.
* Docling for its document chunking ability.
* Pandoc for EPUB-to-Markdown conversion.
* vLLM with DeepSeek Qwen 1.5B for embedding vector generation and
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
* probably a nvidia GPU in a card that has at least 8GB ram.
* docker (the recent-ish version that include docker-compose but make
  you split it up into "docker" and "compose")
* modern nvidia and cuda drivers (I'm at v580)

### Installation
* clone this repo
* create a virtualenv for the project
* install the developer requirements with:

  ```pip install -r requirements.dev.txt```

* run `nv -l` to get a list of available commands
