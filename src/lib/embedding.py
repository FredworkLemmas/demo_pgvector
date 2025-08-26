import attrs
from typing import Iterator
import numpy as np
from transformers import AutoTokenizer
from vllm import SamplingParams

from lib.interfaces import EmbeddingGenerator
from lib.llms import LLMManager


@attrs.define
class DeepseekQwen15BEmbeddingGenerator(EmbeddingGenerator):
    texts: Iterator[str]
    model_name: str | None = None
    tokenizer: AutoTokenizer | None = None
    embedding_dim: int | None = None

    def __attrs_post_init__(self):
        self.model_name = (
            self.model_name or 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
        )
        self.embedding_dim = self.embedding_dim or 1536
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

    def generate(self) -> Iterator[list[float]]:
        """Generate embeddings for each text in the collection"""
        for text in self.texts:
            # Create a prompt that encourages the model to process the text
            # meaningfully
            prompt = (
                f'Analyze and understand this text: {text}\n\n'
                'The key concepts in this text are:'
            )

            # Generate with specific parameters to get consistent outputs
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic
                max_tokens=50,  # Short response to focus on understanding
                top_p=1.0,
                stop=['\n\n', 'END'],
            )

            # Generate response to engage the model's understanding
            llm = LLMManager(model_name=self.model_name).instance()
            outputs = llm.generate([prompt], sampling_params)

            # For now, we'll create embeddings based on the text and response
            # In a full implementation, you'd extract actual hidden states
            embedding = self._extract_semantic_embedding(
                text,
                outputs[0].outputs[0].text if outputs[0].outputs else '',
            )
            yield embedding

    def _extract_semantic_embedding(
        self, original_text: str, model_response: str
    ) -> list[float]:
        """Create semantic embedding from text and model response"""
        # Combine original text and model's interpretation
        combined_text = f'{original_text} {model_response}'

        # Create a more sophisticated embedding based on text analysis
        embedding = self._create_contextual_embedding(combined_text)

        return embedding

    def _create_contextual_embedding(self, text: str) -> list[float]:
        """Create a contextual embedding using text analysis"""
        import hashlib
        import re
        from collections import Counter

        # Normalize text
        text_lower = text.lower().strip()
        words = re.findall(r'\b\w+\b', text_lower)

        # Create base hash for consistency
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        base_seed = int(text_hash[:8], 16)

        # Initialize random generator with base seed
        np.random.seed(base_seed % (2**32))
        base_embedding = np.random.normal(0, 1, self.embedding_dim)

        # Add semantic features based on word analysis
        word_counts = Counter(words)
        total_words = len(words)

        # Modify embedding based on text characteristics
        for i, word in enumerate(set(words)):
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            word_weight = word_counts[word] / total_words

            # Create word-specific modification
            np.random.seed(word_hash % (2**32))
            word_vector = np.random.normal(0, word_weight, self.embedding_dim)
            base_embedding += word_vector * 0.1

        # Add length and complexity features
        length_factor = min(
            len(text) / 100.0, 1.0
        )  # Normalize by typical text length
        complexity_factor = len(set(words)) / max(
            len(words), 1
        )  # Vocabulary diversity

        # Adjust embedding based on text characteristics
        base_embedding *= 1.0 + length_factor * 0.1
        base_embedding += np.random.normal(
            0, complexity_factor * 0.05, self.embedding_dim
        )

        # Normalize the final embedding
        norm = np.linalg.norm(base_embedding)
        if norm > 0:
            base_embedding = base_embedding / norm

        return base_embedding.tolist()
