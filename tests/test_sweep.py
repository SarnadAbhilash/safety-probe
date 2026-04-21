"""Tests for parameter sweep and grid."""

import pytest
from safety_probe.sweep.grid import SweepGrid, ParamRange
from safety_probe.backends.base import GenerationConfig


class TestSweepGrid:
    def test_temperature_sweep_default(self):
        grid = SweepGrid.temperature_sweep()
        configs = grid.configs()
        assert len(configs) == 8  # default 8 temps
        temps = sorted(c.temperature for c in configs)
        assert temps[0] == 0.0
        assert temps[-1] == 2.0

    def test_custom_temperatures(self):
        grid = SweepGrid.temperature_sweep([0.0, 0.5, 1.0])
        configs = grid.configs()
        assert len(configs) == 3

    def test_cartesian_product(self):
        grid = SweepGrid()
        grid.add(ParamRange("temperature", [0.0, 0.5, 1.0]))
        grid.add(ParamRange("top_p", [0.9, 1.0]))
        configs = grid.configs()
        assert len(configs) == 6  # 3 × 2

    def test_grid_size_property(self):
        grid = SweepGrid()
        grid.add(ParamRange("temperature", [0.0, 0.5, 1.0]))
        grid.add(ParamRange("top_p", [0.9, 1.0]))
        assert grid.size == 6

    def test_from_dict(self):
        spec = {
            "base": {"max_new_tokens": 128},
            "grid": {
                "temperature": [0.0, 1.0],
                "top_p": [0.9, 1.0],
            }
        }
        grid = SweepGrid.from_dict(spec)
        configs = grid.configs()
        assert len(configs) == 4
        assert all(c.max_new_tokens == 128 for c in configs)

    def test_greedy_config_deterministic(self):
        cfg = GenerationConfig.greedy()
        assert cfg.temperature == 0.0
        assert cfg.sampling_strategy == "greedy"
        assert cfg.seed == 42

    def test_param_range_linspace(self):
        pr = ParamRange.linspace("temperature", 0.0, 1.0, num=5)
        assert len(pr.values) == 5
        assert abs(pr.values[0] - 0.0) < 1e-6
        assert abs(pr.values[-1] - 1.0) < 1e-6
