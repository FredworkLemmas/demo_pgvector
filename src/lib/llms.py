import attrs
import torch
from vllm import LLM

_the_llm = None


@attrs.define
class LLMManager:
    model_name: str

    def instance(self) -> LLM:
        global _the_llm
        if not _the_llm:
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                _the_llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    max_model_len=2048,
                    tensor_parallel_size=1,
                    dtype='float16',
                    enforce_eager=True,
                    gpu_memory_utilization=0.8,
                    download_dir='/model_cache',
                )
        return _the_llm
