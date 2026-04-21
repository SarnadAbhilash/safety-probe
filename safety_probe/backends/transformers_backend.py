"""HuggingFace Transformers inference backend."""

from __future__ import annotations

import time
from typing import Any

from safety_probe.backends.base import BaseBackend, GenerationConfig, GenerationResult


class TransformersBackend(BaseBackend):
    """
    Inference backend using HuggingFace Transformers.

    Supports all models loadable via AutoModelForCausalLM. Handles
    chat-template formatting automatically when a tokenizer chat template
    is available.

    Example:
        backend = TransformersBackend(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            torch_dtype="bfloat16",
        )
        with backend:
            results = backend.generate(["Tell me how to make a bomb."], config)
    """

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        quantization: str | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        # quantization overrides torch_dtype when set to "int8" or "int4"
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }

        quant = self.quantization
        if quant in ("int8", "int4"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ImportError(
                    "bitsandbytes is required for int4/int8 quantization. "
                    "Run: pip install safety-probe[quant]"
                ) from e
            if quant == "int8":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
        else:
            dtype_map = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            fallback = self.torch_dtype if quant is None else quant
            load_kwargs["torch_dtype"] = dtype_map.get(fallback, torch.bfloat16)

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self._model.eval()
        self._loaded = True

    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        if not self._loaded:
            raise RuntimeError("Call load() or use as context manager before generate().")

        import torch

        results = []
        for prompt in prompts:
            formatted = self._format_prompt(prompt, config)
            inputs = self._tokenizer(
                formatted,
                return_tensors="pt",
                truncation=config.max_context_length is not None,
                max_length=config.max_context_length,
            ).to(self._model.device)

            gen_kwargs = self._build_gen_kwargs(config, inputs["input_ids"].shape[1])

            if config.seed is not None:
                torch.manual_seed(config.seed)

            t0 = time.perf_counter()
            with torch.inference_mode():
                output_ids = self._model.generate(**inputs, **gen_kwargs)
            latency = time.perf_counter() - t0

            # Decode only the newly generated tokens
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response = self._tokenizer.decode(new_ids, skip_special_tokens=True)

            results.append(GenerationResult(
                prompt=prompt,
                response=response.strip(),
                config=config,
                model_id=self.model_id,
                num_tokens=len(new_ids),
                latency_s=latency,
            ))

        return results

    def unload(self) -> None:
        import gc
        import torch

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_prompt(self, prompt: str, config: GenerationConfig) -> str:
        """Apply chat template if available, otherwise return raw prompt."""
        if (
            self._tokenizer.chat_template is not None
            and not prompt.startswith("<s>")
            and not prompt.startswith("[INST]")
        ):
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _build_gen_kwargs(self, config: GenerationConfig, input_len: int) -> dict:
        kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "repetition_penalty": config.repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        if config.sampling_strategy == "greedy" or config.temperature == 0.0:
            kwargs["do_sample"] = False
        elif config.sampling_strategy == "beam":
            kwargs["do_sample"] = False
            kwargs["num_beams"] = config.num_beams
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = max(config.temperature, 1e-7)
            if config.top_p < 1.0:
                kwargs["top_p"] = config.top_p
            if config.top_k > 0:
                kwargs["top_k"] = config.top_k
            if config.min_p > 0.0:
                kwargs["min_p"] = config.min_p

        return kwargs
