"""Tests for safety metrics computation."""

import pytest
from unittest.mock import MagicMock

from safety_probe.metrics.safety_metrics import SafetyMetrics, ConfigMetrics
from safety_probe.backends.base import GenerationConfig
from safety_probe.judges.base import Verdict, JudgementResult
from safety_probe.probes.base import ProbeCategory


def _make_config(temperature: float) -> GenerationConfig:
    return GenerationConfig(temperature=temperature)


def _make_judgement(verdict: Verdict) -> JudgementResult:
    return JudgementResult(verdict=verdict, confidence=0.9, judge_name="test")


def _make_config_metrics(temperature: float, refusal_rate: float) -> ConfigMetrics:
    return ConfigMetrics(
        config=_make_config(temperature),
        refusal_rate=refusal_rate,
        compliance_rate=1.0 - refusal_rate,
        partial_rate=0.0,
        attack_success_rate=1.0 - refusal_rate,
        refusal_rate_std=0.1,
        over_refusal_rate=0.05,
        category_refusal_rates={},
        n_harm_probes=20,
        n_benign_probes=5,
        n_adversarial_probes=8,
    )


class TestSafetyMetrics:
    def test_safety_stability_score_perfect(self):
        # All configs refuse everything
        metrics = [_make_config_metrics(t, 1.0) for t in [0.0, 0.5, 1.0]]
        s3 = SafetyMetrics.safety_stability_score(metrics)
        assert abs(s3 - 1.0) < 1e-6

    def test_safety_stability_score_degrading(self):
        # Refusal rate drops from 1.0 to 0.0 — high variance
        metrics = [_make_config_metrics(t, r) for t, r in [(0.0, 1.0), (1.0, 0.5), (2.0, 0.0)]]
        s3 = SafetyMetrics.safety_stability_score(metrics)
        # mean=0.5, std≈0.408 → s3 ≈ 0.09
        assert s3 < 0.5

    def test_degradation_onset_temperature_detected(self):
        metrics = [
            _make_config_metrics(0.0, 1.0),
            _make_config_metrics(0.5, 0.98),
            _make_config_metrics(1.0, 0.90),
            _make_config_metrics(1.5, 0.80),  # drops >5% from baseline
        ]
        dot = SafetyMetrics.degradation_onset_temperature(metrics, baseline_temp=0.0, threshold=0.05)
        assert dot is not None
        assert dot <= 1.5

    def test_degradation_onset_none_when_stable(self):
        metrics = [_make_config_metrics(t, 0.99) for t in [0.0, 0.5, 1.0, 1.5, 2.0]]
        dot = SafetyMetrics.degradation_onset_temperature(metrics)
        assert dot is None

    def test_attack_surface_area(self):
        metrics = [
            _make_config_metrics(0.0, 1.0),   # ASR = 0.0 → not vulnerable
            _make_config_metrics(1.0, 0.8),   # ASR = 0.2 → vulnerable
            _make_config_metrics(2.0, 0.5),   # ASR = 0.5 → vulnerable
        ]
        asa = SafetyMetrics.attack_surface_area(metrics, asr_threshold=0.1)
        assert abs(asa - 2 / 3) < 1e-6
