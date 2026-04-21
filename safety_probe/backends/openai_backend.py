"""OpenAI-compatible API backend (OpenAI, Together, Fireworks, Anyscale, etc.)."""

from __future__ import annotations

import os
import time
import random
from typing import Any

from safety_probe.backends.base import BaseBackend, GenerationConfig, GenerationResult


class OpenAIBackend(BaseBackend):
    """
    Backend for any OpenAI-compatible chat completion API.

    Works with OpenAI, Together AI, Fireworks, Anyscale, and local servers
    like vLLM's OpenAI-compatible endpoint.

    Example:
        backend = OpenAIBackend(
            model_id="gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"],
        )
        # Together AI:
        backend = OpenAIBackend(
            model_id="meta-llama/Llama-3-8b-chat-hf",
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )
    """

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.timeout = timeout
        self._client: Any = None

    def load(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "default_headers": {
                "HTTP-Referer": "https://github.com/safety-probe",
                "X-Title": "safety-probe",
            },
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self._client = OpenAI(**client_kwargs)
        self._loaded = True

    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        if not self._loaded:
            raise RuntimeError("Call load() or use as context manager before generate().")

        results = []
        for prompt in prompts:
            messages = self._build_messages(prompt)
            kwargs = self._build_request_kwargs(config)

            t0 = time.perf_counter()
            response = self._call_with_retry(messages, kwargs)
            latency = time.perf_counter() - t0

            msg = response.choices[0].message
            # Reasoning models (GLM-5, DeepSeek-R1) put final answer in content
            # and chain-of-thought in reasoning_content. Fall back to reasoning
            # content only if content is empty (means it got cut off mid-think).
            content = msg.content or getattr(msg, "reasoning_content", None) or ""
            num_tokens = response.usage.completion_tokens if response.usage else None

            results.append(GenerationResult(
                prompt=prompt,
                response=content.strip(),
                config=config,
                model_id=self.model_id,
                num_tokens=num_tokens,
                latency_s=latency,
            ))

        return results

    def unload(self) -> None:
        self._client = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call_with_retry(self, messages: list, kwargs: dict, max_retries: int = 6) -> Any:
        """Call the API with exponential backoff on 429 / 5xx errors."""
        from openai import RateLimitError, APIStatusError

        wait = 15.0  # start at 15s — upstream provider 429s need longer waits
        for attempt in range(max_retries):
            try:
                return self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    **kwargs,
                )
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                jitter = random.uniform(0, wait * 0.2)
                print(f"\n  [rate limit] waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait + jitter)
                wait = min(wait * 2, 120)  # cap at 2 minutes
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < max_retries - 1:
                    jitter = random.uniform(0, wait * 0.2)
                    print(f"\n  [server error {e.status_code}] waiting {wait:.0f}s before retry...")
                    time.sleep(wait + jitter)
                    wait = min(wait * 2, 120)
                else:
                    raise

    def _build_request_kwargs(self, config: GenerationConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_tokens": config.max_new_tokens}

        if config.sampling_strategy == "greedy" or config.temperature == 0.0:
            kwargs["temperature"] = 0.0
        else:
            kwargs["temperature"] = config.temperature
            kwargs["top_p"] = config.top_p

        if config.seed is not None:
            kwargs["seed"] = config.seed

        return kwargs
