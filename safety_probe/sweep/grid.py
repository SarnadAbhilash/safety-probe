"""Parameter grid definitions for sweep experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from safety_probe.backends.base import GenerationConfig


@dataclass
class ParamRange:
    """Defines the values to sweep for a single parameter."""

    name: str
    values: list[Any]

    @classmethod
    def linspace(cls, name: str, start: float, stop: float, num: int) -> "ParamRange":
        return cls(name=name, values=list(np.linspace(start, stop, num).tolist()))

    @classmethod
    def logspace(cls, name: str, start: float, stop: float, num: int) -> "ParamRange":
        return cls(name=name, values=list(np.logspace(start, stop, num).tolist()))


class SweepGrid:
    """
    Defines a multi-dimensional parameter grid for safety sweeps.

    Each axis is one inference parameter; the grid is the Cartesian product
    of all axis values. Pass a SweepGrid to ParameterSweep.

    Example:
        grid = SweepGrid()
        grid.add(ParamRange.linspace("temperature", 0.0, 2.0, num=9))
        grid.add(ParamRange("top_p", [0.9, 0.95, 1.0]))
        # 9 × 3 = 27 configurations
        configs = grid.configs()
    """

    def __init__(self, base: GenerationConfig | None = None) -> None:
        self._base = base or GenerationConfig.greedy()
        self._axes: list[ParamRange] = []

    def add(self, param_range: ParamRange) -> "SweepGrid":
        self._axes.append(param_range)
        return self

    @property
    def size(self) -> int:
        if not self._axes:
            return 1
        result = 1
        for ax in self._axes:
            result *= len(ax.values)
        return result

    def configs(self) -> list[GenerationConfig]:
        """Return the full list of GenerationConfig objects in the grid."""
        if not self._axes:
            return [self._base]

        names = [ax.name for ax in self._axes]
        value_lists = [ax.values for ax in self._axes]

        configs = []
        for combo in product(*value_lists):
            cfg_dict = self._base.to_dict()
            for name, value in zip(names, combo):
                if name in cfg_dict:
                    cfg_dict[name] = value
                else:
                    cfg_dict["extra"][name] = value
            configs.append(GenerationConfig(**{k: v for k, v in cfg_dict.items() if k != "extra"},
                                             extra=cfg_dict.get("extra", {})))
        return configs

    @classmethod
    def temperature_sweep(
        cls,
        temperatures: list[float] | None = None,
    ) -> "SweepGrid":
        """Convenience constructor: sweep temperature only (most common experiment)."""
        temps = temperatures or [0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
        grid = cls()
        grid.add(ParamRange("temperature", temps))
        return grid

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "SweepGrid":
        """Load a grid from a plain dict (e.g., parsed from YAML config)."""
        base_kwargs = spec.get("base", {})
        base = GenerationConfig(**base_kwargs) if base_kwargs else None
        grid = cls(base=base)
        for param_name, values in spec.get("grid", {}).items():
            grid.add(ParamRange(param_name, values))
        return grid
