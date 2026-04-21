"""Abstract base class for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationConfig:
    """Normalized inference parameter config passed to every backend."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 256
    # Context window truncation strategy: "left" | "right" | "none"
    context_truncation: str = "left"
    max_context_length: int | None = None
    # Sampling: "greedy" | "multinomial" | "beam"
    sampling_strategy: str = "multinomial"
    num_beams: int = 1
    seed: int | None = None
    # Speculative decoding (vLLM only)
    speculative_model: str | None = None
    num_speculative_tokens: int = 5
    # Quantization precision (transformers backend): "fp32" | "fp16" | "bf16" | "int8" | "int4"
    quantization: str = "bf16"
    # Extra backend-specific kwargs passed through
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def greedy(cls) -> "GenerationConfig":
        """Deterministic greedy config — used as safety baseline."""
        return cls(temperature=0.0, top_p=1.0, sampling_strategy="greedy", seed=42)


@dataclass
class GenerationResult:
    """Output from a single generation call."""

    prompt: str
    response: str
    config: GenerationConfig
    model_id: str
    # Tokens generated (may be None if backend doesn't expose it)
    num_tokens: int | None = None
    # Wall-clock latency in seconds
    latency_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseBackend(ABC):
    """
    Abstract inference backend.

    Subclasses wrap a specific runtime (HF Transformers, vLLM, OpenAI API, etc.)
    and expose a uniform generate() interface.
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self._kwargs = kwargs
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights / initialize client. Called once before generation."""
        ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[GenerationResult]:
        """
        Generate responses for a batch of prompts under the given config.

        Args:
            prompts: List of fully-formatted prompt strings.
            config: Inference parameter configuration.

        Returns:
            List of GenerationResult, one per prompt (preserving order).
        """
        ...

    def unload(self) -> None:
        """Release model resources. Override if backend needs explicit cleanup."""
        pass

    def __enter__(self) -> "BaseBackend":
        self.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.unload()
