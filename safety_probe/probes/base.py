"""Core probe data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProbeCategory(str, Enum):
    """Semantic categories aligned with HarmBench taxonomy."""

    # Direct harm requests
    VIOLENCE = "violence"
    WEAPONS = "weapons"
    CYBERATTACKS = "cyberattacks"
    CHEMICAL_BIO = "chemical_bio"
    FRAUD = "fraud"
    HATE_SPEECH = "hate_speech"
    SEXUAL_HARM = "sexual_harm"
    # Adversarial
    JAILBREAK_ROLEPLAY = "jailbreak_roleplay"
    JAILBREAK_MANYSHOT = "jailbreak_manyshot"
    JAILBREAK_GCG = "jailbreak_gcg"
    # Boundary / dual-use
    BORDERLINE = "borderline"
    BENIGN = "benign"  # control group — should NOT be refused


@dataclass
class Probe:
    """
    A single evaluation probe: one prompt and its metadata.

    Probes are the atomic unit of the probe library. Each probe has:
    - text: the raw prompt string sent to the model
    - category: semantic harm category (drives judge behavior)
    - expected_refusal: whether a safe model should refuse
    - tier: "core" | "adversarial" | "boundary"
    - source: where the probe came from
    """

    text: str
    category: ProbeCategory
    expected_refusal: bool
    tier: str = "core"
    source: str = "custom"
    probe_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probe_id:
            import hashlib
            self.probe_id = hashlib.md5(self.text.encode()).hexdigest()[:8]
