"""Canonical behavioral failure mode taxonomy."""

from __future__ import annotations

from enum import Enum


class FailureMode(str, Enum):
    SYCOPHANCY = "sycophancy"
    HALLUCINATION_UNDER_PRESSURE = "hallucination_under_pressure"
    INSTRUCTION_HIERARCHY_FAILURE = "instruction_hierarchy_failure"
    OVERCONFIDENCE_WHEN_WRONG = "overconfidence_when_wrong"
    MULTI_TURN_CONSISTENCY_FAILURE = "multi_turn_consistency_failure"
