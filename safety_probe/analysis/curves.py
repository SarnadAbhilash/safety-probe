"""Sensitivity curves and 2D interaction heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safety_probe.metrics.safety_metrics import ConfigMetrics


class SensitivityCurves:
    """
    Generate 1D sensitivity curves (refusal rate vs. single parameter)
    and 2D interaction heatmaps (refusal rate vs. two parameters).

    Example:
        curves = SensitivityCurves(config_metrics)
        fig = curves.plot_temperature_curve(save_path="outputs/temp_curve.png")
        fig = curves.plot_heatmap("temperature", "top_p", save_path="outputs/heatmap.png")
    """

    def __init__(self, metrics: list[ConfigMetrics]) -> None:
        self.metrics = metrics

    def temperature_curve(self) -> tuple[list[float], list[float]]:
        """Returns (temperatures, refusal_rates) sorted by temperature."""
        pairs = [(m.config.temperature, m.refusal_rate) for m in self.metrics]
        pairs.sort(key=lambda x: x[0])
        temps, rates = zip(*pairs) if pairs else ([], [])
        return list(temps), list(rates)

    def param_curve(self, param: str) -> tuple[list[float], list[float]]:
        """Generic 1D curve for any GenerationConfig parameter."""
        pairs = []
        for m in self.metrics:
            val = getattr(m.config, param, None)
            if val is not None:
                pairs.append((float(val), m.refusal_rate))
        pairs.sort(key=lambda x: x[0])
        if not pairs:
            return [], []
        vals, rates = zip(*pairs)
        return list(vals), list(rates)

    def heatmap_data(
        self, param_x: str, param_y: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (x_vals, y_vals, refusal_rate_matrix) for a 2D heatmap.

        Aggregates refusal rates across all metrics with matching (x, y) values.
        """
        from collections import defaultdict

        cell_rates: dict[tuple, list[float]] = defaultdict(list)
        for m in self.metrics:
            x = getattr(m.config, param_x, None)
            y = getattr(m.config, param_y, None)
            if x is not None and y is not None:
                cell_rates[(float(x), float(y))].append(m.refusal_rate)

        x_vals = sorted(set(k[0] for k in cell_rates))
        y_vals = sorted(set(k[1] for k in cell_rates))

        matrix = np.full((len(y_vals), len(x_vals)), np.nan)
        for (x, y), rates in cell_rates.items():
            xi = x_vals.index(x)
            yi = y_vals.index(y)
            matrix[yi, xi] = np.mean(rates)

        return np.array(x_vals), np.array(y_vals), matrix

    def plot_temperature_curve(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> Any:
        """Plot refusal rate vs. temperature. Returns matplotlib Figure."""
        import matplotlib.pyplot as plt

        temps, rates = self.temperature_curve()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(temps, rates, marker="o", linewidth=2, color="#e74c3c")
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% refusal baseline")
        ax.fill_between(temps, rates, 0.95, where=[r < 0.95 for r in rates],
                        alpha=0.15, color="#e74c3c", label="Safety gap")
        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel("Refusal Rate", fontsize=12)
        ax.set_title("Safety vs. Temperature", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig

    def plot_heatmap(
        self,
        param_x: str,
        param_y: str,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> Any:
        """Plot 2D interaction heatmap for two parameters."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        x_vals, y_vals, matrix = self.heatmap_data(param_x, param_y)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            matrix,
            xticklabels=[f"{v:.2f}" for v in x_vals],
            yticklabels=[f"{v:.2f}" for v in y_vals],
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_xlabel(param_x, fontsize=12)
        ax.set_ylabel(param_y, fontsize=12)
        ax.set_title(f"Refusal Rate: {param_x} × {param_y}", fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        return fig
