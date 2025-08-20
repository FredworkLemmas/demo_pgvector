from typing import Iterator

import attrs


@attrs.define
class DeepseekQwen15BEmbeddingGenerator:
    def generate(self) -> Iterator[list[float]]:
        pass
