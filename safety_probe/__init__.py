"""
safety-probe: Inference-time safety probing framework for LLMs.

Measures how model safety behaviors degrade as a function of inference
parameters — temperature, sampling strategy, context length, speculative
decoding, batching — and characterizes what causes degradation.
"""

from safety_probe.sweep.parameter_sweep import ParameterSweep
from safety_probe.sweep.grid import SweepGrid
from safety_probe.probes.probe_sets import ProbeSet, load_probe_set
from safety_probe.judges.composite import CompositeJudge
from safety_probe.metrics.safety_metrics import SafetyMetrics

__version__ = "0.1.0"
__all__ = [
    "ParameterSweep",
    "SweepGrid",
    "ProbeSet",
    "load_probe_set",
    "CompositeJudge",
    "SafetyMetrics",
]
