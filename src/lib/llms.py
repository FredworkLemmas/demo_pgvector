import attrs
import torch
from vllm import LLM


@attrs.define
class LLMManager:
    model_name: str
    llm: LLM | None = None

    def instance(self) -> LLM:
        if not self.llm:
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                self.llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    max_model_len=2048,
                    tensor_parallel_size=1,
                    dtype='float16',
                    enforce_eager=True,
                    gpu_memory_utilization=0.8,
                    download_dir='/model_cache',
                )
        return self.llm
