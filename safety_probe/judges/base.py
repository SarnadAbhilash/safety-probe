"""Abstract base class for safety judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from safety_probe.probes.base import ProbeCategory


class Verdict(str, Enum):
    REFUSED = "refused"        # Model declined to comply — safe behavior
    COMPLIED = "complied"      # Model complied with the request — unsafe for harm probes
    PARTIAL = "partial"        # Partial compliance or ambiguous
    ERROR = "error"            # Judge could not determine verdict


@dataclass
class JudgementResult:
    """Result of judging a single (prompt, response) pair."""

    verdict: Verdict
    # Confidence in [0, 1] — 1.0 = certain
    confidence: float
    judge_name: str
    # Brief explanation (populated by LLM judge, empty for rule/classifier judges)
    explanation: str = ""
    probe_category: ProbeCategory | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_refused(self) -> bool:
        return self.verdict == Verdict.REFUSED

    @property
    def is_complied(self) -> bool:
        return self.verdict == Verdict.COMPLIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "judge_name": self.judge_name,
            "explanation": self.explanation,
            "probe_category": self.probe_category.value if self.probe_category else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JudgementResult":
        return cls(
            verdict=Verdict(d["verdict"]),
            confidence=d["confidence"],
            judge_name=d["judge_name"],
            explanation=d.get("explanation", ""),
            probe_category=ProbeCategory(d["probe_category"]) if d.get("probe_category") else None,
            metadata=d.get("metadata", {}),
        )


class BaseJudge(ABC):
    """Abstract safety judge."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def judge(
        self,
        prompt: str,
        response: str,
        probe_category: ProbeCategory | None = None,
    ) -> JudgementResult:
        """
        Judge whether a model response is a refusal or compliance.

        Args:
            prompt: The original probe prompt.
            response: The model's response text.
            probe_category: Optional category hint for the judge.

        Returns:
            JudgementResult with verdict and confidence.
        """
        ...
