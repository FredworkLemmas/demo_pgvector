# python
from __future__ import annotations

import os
import json
import hashlib
from typing import Optional, Dict, Any, List

from vllm import LLM, SamplingParams


class MultiPerspectiveSummarizer:
    """
    Summarize a single piece of text from different perspectives using vLLM and
    a Qwen model, optimizing for repeated prompts via prefix caching.

    - The model is 'Qwen/Qwen3-4B-Thinking-2507' by default.
    - The shared header + source text are placed at the start of every request
      to enable prefix caching.
    - Call respond(prompt) to get a summary for a given perspective.
    """

    def __init__(
        self,
        header: str,
        text: Optional[str],
        source_path: str,
        model: str = 'Qwen/Qwen3-4B-Thinking-2507',
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        trust_remote_code: bool = True,
        dtype: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the summarizer.

        Args:
            header: A header/instructions that will precede the text for all
                    prompts.
            text: The source document content. If None, it will be read from
                  source_path.
            source_path: Path to the source document (plain text or markdown).
            model: Hugging Face model id for vLLM to load.
            max_new_tokens: Maximum tokens to generate per response.
            temperature, top_p, top_k, repetition_penalty, stop:
                Sampling controls.
            trust_remote_code: Passed to vLLM to trust remote model code
                               (often required for chat templates).
            dtype: Optional dtype override (e.g., "bfloat16", "float16"). If
                   None, vLLM chooses.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory for vLLM.
        """
        self.header = header
        self.source_path = source_path

        if text is None:
            if not os.path.exists(source_path):
                raise FileNotFoundError(f'Source file not found: {source_path}')
            with open(source_path, 'r', encoding='utf-8') as f:
                text = f.read()
        self.text = text

        # Initialize the model with prefix caching enabled to reuse KV cache
        # across requests. This ensures the shared prefix (header + text) is
        # only processed once across prompts.
        self.llm = LLM(
            model=model,
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=True,
            dtype=dtype,  # let vLLM decide if None
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Build the shared conversation messages; these remain identical for
        # every call. Keeping this identical ensures maximum cache hits for the
        # shared prefix. We put the header in system, and the full document in a
        # user message so that later prompts can be separate user turns appended
        # after this.
        self.shared_messages: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.header.strip()},
            {
                'role': 'user',
                'content': (
                    'You will be asked to produce multiple summaries from '
                    'different perspectives.\n'
                    'Here is the source document you should rely on:\n\n'
                    f'{self.text.strip()}'
                ),
            },
        ]

        # Prepare the tokenizer and a serialized shared prefix using the chat
        # template. We don't need to store token ids explicitly; using identical
        # leading text per call is enough for vLLM to reuse the KV cache.
        self.tokenizer = self.llm.get_tokenizer()

        # This string is the serialized chat up to (but not including) the
        # perspective prompt. We'll append each perspective prompt as an
        # additional user message for each call. We don't add a generation
        # prompt here so that the final respond() call can do so.
        self.shared_prefix_text: str = self.tokenizer.apply_chat_template(
            self.shared_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Default sampling params used across calls; can be overridden
        # per-respond if needed.
        self.base_sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )

        # Persisted for state saving
        self._model_id = model
        self._sp_defaults = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )
        self._trust_remote_code = trust_remote_code
        self._dtype = dtype
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization

    def respond(
        self,
        prompt: str,
        sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Ask the model to produce a summary for a given perspective.

        The shared header + text are prepended (as a cached prefix), and the
        prompt is appended as a new user message. vLLM will reuse the KV cache
        for the prefix.

        Args:
            prompt:
                The perspective-specific instruction (e.g., "Summarize from
                a legal angle").
            sampling_overrides:
                Optional dict to override SamplingParams for this call.

        Returns:
            The model's response text.
        """
        # Full messages for this request: shared prefix + new user turn with
        # the prompt.
        messages = list(self.shared_messages) + [
            {'role': 'user', 'content': prompt.strip()}
        ]

        # Serialize messages using the chat template. Using
        # add_generation_prompt=True adds the assistant role cue to start
        # generation.
        full_prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Because shared_prefix is an exact leading substring of
        # full_prompt_text, and enable_prefix_caching=True, vLLM will reuse the
        # KV cache for that prefix. We just pass the full prompt. No special API
        # is required beyond prefix caching.
        sampling_params = self.base_sampling_params
        if sampling_overrides:
            # Create a shallow copy with overrides
            sampling_params = SamplingParams(
                **{**self.base_sampling_params.__dict__, **sampling_overrides}
            )

        outputs = self.llm.generate(
            [full_prompt_text], sampling_params=sampling_params
        )
        # vLLM returns a list of RequestOutput; we take the first, then its
        # top candidate.
        text = outputs[0].outputs[0].text
        return text


class PersistentMultiPerspectiveSummarizer(MultiPerspectiveSummarizer):
    """
    A MultiPerspectiveSummarizer that can save its state to disk and restore it
    in a later run.

    Notes:
    - This persists the exact serialized shared prefix (via the model's chat
      template) and configuration, so that a later run can re-create the
      identical prompt prefix and proactively warm the cache.
    - vLLM's internal KV cache is in-memory; this class does not serialize KV
      tensors to disk. Instead, call warm_cache() after restore to pre-compute
      the prefix in the new engine.
    """

    STATE_VERSION = 1

    def to_state_dict(self) -> Dict[str, Any]:
        prefix_hash = hashlib.sha256(
            self.shared_prefix_text.encode('utf-8')
        ).hexdigest()
        return {
            'version': self.STATE_VERSION,
            'model': self._model_id,
            'header': self.header,
            'text': self.text,
            'shared_messages': self.shared_messages,
            'shared_prefix_text': self.shared_prefix_text,
            'shared_prefix_sha256': prefix_hash,
            'sampling_defaults': self._sp_defaults,
            'engine': {
                'trust_remote_code': self._trust_remote_code,
                'dtype': self._dtype,
                'tensor_parallel_size': self._tensor_parallel_size,
                'gpu_memory_utilization': self._gpu_memory_utilization,
            },
        }

    def save_state(self, path: str) -> None:
        state = self.to_state_dict()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(
        cls,
        path: str,
        *,
        # Allow callers to optionally override engine settings on restore:
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        trust_remote_code: Optional[bool] = None,
        dtype: Optional[str] = None,
        # Control whether to immediately warm the cache for the shared prefix:
        warm_cache: bool = True,
        warm_cache_tokens: int = 1,
    ) -> 'PersistentMultiPerspectiveSummarizer':
        """
        Restore a PersistentMultiPerspectiveSummarizer from a state file created
        by save_state().

        Set warm_cache=True to proactively compute the shared prefix in the new
        engine so that subsequent respond() calls benefit from prefix caching
        immediately.
        """
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        if state.get('version') != cls.STATE_VERSION:
            raise ValueError(
                'Incompatible state version: '
                f'{state.get("version")} != {cls.STATE_VERSION}'
            )

        engine = state['engine']
        sampling = state['sampling_defaults']

        # Build instance with the same header/text/model and defaults.
        inst = cls(
            header=state['header'],
            text=state['text'],
            source_path='<restored>',  # informational; not used since text is
            # provided
            model=state['model'],
            max_new_tokens=sampling['max_new_tokens'],
            temperature=sampling['temperature'],
            top_p=sampling['top_p'],
            top_k=sampling['top_k'],
            repetition_penalty=sampling['repetition_penalty'],
            stop=sampling['stop'],
            trust_remote_code=trust_remote_code
            if trust_remote_code is not None
            else engine['trust_remote_code'],
            dtype=dtype if dtype is not None else engine['dtype'],
            tensor_parallel_size=tensor_parallel_size
            if tensor_parallel_size is not None
            else engine['tensor_parallel_size'],
            gpu_memory_utilization=gpu_memory_utilization
            if gpu_memory_utilization is not None
            else engine['gpu_memory_utilization'],
        )

        # Verify that the serialized prefix matches exactly (template
        # consistency).
        restored_prefix = inst.shared_prefix_text
        restored_hash = hashlib.sha256(
            restored_prefix.encode('utf-8')
        ).hexdigest()
        if restored_hash != state['shared_prefix_sha256']:
            # If the model's chat template changed between runs, the prefix
            # might not match. Persisting the exact text helps, but we assert
            # equivalence to avoid silent cache misses.
            raise ValueError(
                'Restored chat template produced a different shared prefix '
                'than saved state. '
                'Ensure the same model/tokenizer and template are used.'
            )

        if warm_cache:
            inst.warm_cache(max_tokens=warm_cache_tokens)

        return inst

    def warm_cache(self, max_tokens: int = 1) -> None:
        """
        Proactively compute and cache the shared prefix in the current LLM
        engine.

        This sends the serialized shared prefix into the model and generates a
        minimal number of tokens. Subsequent calls whose prompts start with this
        exact prefix can reuse the cached KV state via prefix caching within the
        current process.
        """
        # Generate from the prefix alone to build its KV state. Keep it minimal.
        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )
        _ = self.llm.generate([self.shared_prefix_text], sampling_params=sp)

    def close(self) -> None:
        """
        Free resources associated with the underlying LLM engine.
        """
        try:
            # vLLM LLM exposes llm_engine with a shutdown() in many versions;
            # attempt it if available.
            engine = getattr(self.llm, 'llm_engine', None)
            if engine and hasattr(engine, 'shutdown'):
                engine.shutdown()
        finally:
            # Drop references to encourage GC of CUDA memory if applicable.
            self.llm = None


if __name__ == '__main__':
    # Example usage:
    header = (
        'You are an expert summarizer. Always ground your summaries '
        'in the provided text. '
        'Be concise, faithful, and note uncertainties.'
    )
    source_path = 'source.md'
    # If you already have the text, pass it instead of reading from file:
    text = None

    # Create a persistent summarizer
    summarizer = PersistentMultiPerspectiveSummarizer(
        header=header,
        text=text,
        source_path=source_path,
        model='Qwen/Qwen3-4B-Thinking-2507',
        max_new_tokens=400,
        temperature=0.2,
    )

    # Optionally warm the cache for the shared prefix in this session.
    summarizer.warm_cache()

    # Save state to reuse later (e.g., across processes or after freeing GPU
    # memory).
    state_path = 'summarizer_state.json'
    summarizer.save_state(state_path)

    perspectives = [
        'Write a one-paragraph executive summary.',
        (
            'Summarize from a technical perspective, focusing on architecture'
            ' and algorithms.'
        ),
        (
            'Summarize potential risks and mitigations from a security'
            ' perspective.'
        ),
        (
            'Summarize for a legal/compliance audience, highlighting obligations '
            'and constraints.'
        ),
        'Summarize the key open questions and uncertainties.',
    ]

    for p in perspectives:
        print(f'\n=== Perspective: {p} ===')
        print(summarizer.respond(p))

    # Free resources (e.g., to later restore and reuse the same prefix template)
    summarizer.close()

    # Later run (or in a new process):
    # restored = PersistentMultiPerspectiveSummarizer.load_state(
    #     state_path, warm_cache=True)
    # print(restored.respond("Write a one-sentence TL;DR."))
