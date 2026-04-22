"""vLLM inference backend — supports speculative decoding and continuous batching."""

from __future__ import annotations

import time
from typing import Any

from safety_probe.backends.base import BaseBackend, GenerationConfig, GenerationResult


class VLLMBackend(BaseBackend):
    """
    Inference backend using vLLM.

    Preferred backend for:
    - Speculative decoding experiments (set speculative_model in GenerationConfig)
    - Continuous batching / throughput experiments
    - Long context experiments

    Requires: pip install safety-probe[vllm]

    Example:
        backend = VLLMBackend(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            gpu_memory_utilization=0.85,
        )
        # With speculative decoding:
        config = GenerationConfig(
            temperature=0.8,
            speculative_model="meta-llama/Meta-Llama-3-8B",  # draft model
            num_speculative_tokens=5,
        )
    """

    def __init__(
        self,
        model_id: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        tensor_parallel_size: int = 1,
        speculative_model: str | None = None,
        num_speculative_tokens: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.speculative_model = speculative_model
        self.num_speculative_tokens = num_speculative_tokens
        self._llm: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Run: pip install safety-probe[vllm]"
            ) from e

        llm_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        if self.speculative_model is not None:
            llm_kwargs["speculative_model"] = self.speculative_model
            llm_kwargs["num_speculative_tokens"] = self.num_speculative_tokens

        self._llm = LLM(**llm_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._loaded = True

    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        if not self._loaded:
            raise RuntimeError("Call load() or use as context manager before generate().")

        from vllm import SamplingParams

        sampling_params = self._build_sampling_params(config)

        # Apply chat template if available
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        t0 = time.perf_counter()
        outputs = self._llm.generate(formatted_prompts, sampling_params)
        total_latency = time.perf_counter() - t0
        per_prompt_latency = total_latency / len(prompts)

        results = []
        for prompt, output in zip(prompts, outputs):
            response = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            results.append(GenerationResult(
                prompt=prompt,
                response=response.strip(),
                config=config,
                model_id=self.model_id,
                num_tokens=num_tokens,
                latency_s=per_prompt_latency,
            ))

        return results

    def unload(self) -> None:
        import gc
        del self._llm
        self._llm = None
        self._loaded = False
        gc.collect()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_prompt(self, prompt: str) -> str:
        if self._tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _build_sampling_params(self, config: GenerationConfig) -> Any:
        from vllm import SamplingParams

        kwargs: dict[str, Any] = {
            "max_tokens": config.max_new_tokens,
            "repetition_penalty": config.repetition_penalty,
        }

        if config.sampling_strategy == "greedy" or config.temperature == 0.0:
            kwargs["temperature"] = 0.0
        else:
            kwargs["temperature"] = config.temperature
            kwargs["top_p"] = config.top_p
            if config.top_k > 0:
                kwargs["top_k"] = config.top_k
            if config.min_p > 0.0:
                kwargs["min_p"] = config.min_p

        if config.seed is not None:
            kwargs["seed"] = config.seed

        # Speculative decoding params (vLLM handles at LLM init, not per-request;
        # we record the config for experiment tracking but don't pass here)
        return SamplingParams(**kwargs)
