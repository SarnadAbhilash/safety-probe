"""Sweep report generation — rich console + JSON summary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table
from rich import box

from safety_probe.metrics.safety_metrics import ConfigMetrics, SafetyMetrics
from safety_probe.analysis.phase_detection import PhaseDetector

if TYPE_CHECKING:
    from safety_probe.sweep.parameter_sweep import SweepResult

console = Console()


class SweepReport:
    """
    Generates a structured report from sweep results.

    Outputs:
    - Rich console table with per-config safety metrics
    - JSON summary with aggregate statistics
    - Saved plots (optional)

    Example:
        report = SweepReport(result, metrics, probe_set)
        report.print_summary()
        report.save("outputs/report.json")
    """

    def __init__(
        self,
        result: "SweepResult",
        config_metrics: list[ConfigMetrics],
        probe_set: Any,
        output_dir: str | Path | None = None,
    ) -> None:
        self.result = result
        self.config_metrics = config_metrics
        self.probe_set = probe_set
        self.output_dir = Path(output_dir) if output_dir else None

    def print_summary(self) -> None:
        console.rule(f"[bold]Safety Sweep Report — {self.result.model_id}[/bold]")

        # Aggregate stats
        s3 = SafetyMetrics.safety_stability_score(self.config_metrics)
        asa = SafetyMetrics.attack_surface_area(self.config_metrics)

        detector = PhaseDetector(self.config_metrics)
        transitions = detector.detect("temperature")
        profile = detector.robustness_profile("temperature")

        console.print(f"\n[bold cyan]Model:[/bold cyan] {self.result.model_id}")
        console.print(f"[bold cyan]Probe set:[/bold cyan] {self.result.probe_set_name}")
        console.print(f"[bold cyan]Configs evaluated:[/bold cyan] {len(self.config_metrics)}")
        console.print(f"\n[bold yellow]Aggregate Metrics[/bold yellow]")
        console.print(f"  Safety Stability Score (S³): [bold]{s3:.3f}[/bold]")
        console.print(f"  Attack Surface Area (ASA):   [bold]{asa:.3f}[/bold]")
        if profile:
            console.print(f"  Robustness AUC:              [bold]{profile['auc']:.3f}[/bold]")
            if profile.get("safe_zone"):
                console.print(f"  Safe temperature zone:       {profile['safe_zone']}")

        if transitions:
            console.print(f"\n[bold red]Phase Transitions Detected[/bold red]")
            for t in transitions:
                console.print(
                    f"  {t.parameter}={t.transition_value:.2f}  "
                    f"refusal drop: {t.refusal_rate_before:.2f} → {t.refusal_rate_after:.2f}  "
                    f"(Δ={t.drop_magnitude:.2f}, conf={t.confidence:.2f})"
                )

        # Per-config table
        console.print("\n[bold yellow]Per-Config Results[/bold yellow]")
        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Temp", style="cyan", width=6)
        table.add_column("Top-P", width=6)
        table.add_column("Refusal%", style="green", width=10)
        table.add_column("ASR%", style="red", width=8)
        table.add_column("Over-Refusal%", style="yellow", width=14)
        table.add_column("Std", width=6)

        for m in self.config_metrics:
            refusal_color = "green" if m.refusal_rate >= 0.9 else "yellow" if m.refusal_rate >= 0.7 else "red"
            table.add_row(
                f"{m.config.temperature:.2f}",
                f"{m.config.top_p:.2f}",
                f"[{refusal_color}]{m.refusal_rate * 100:.1f}[/{refusal_color}]",
                f"{m.attack_success_rate * 100:.1f}",
                f"{m.over_refusal_rate * 100:.1f}",
                f"{m.refusal_rate_std:.3f}",
            )

        console.print(table)

    def to_dict(self) -> dict[str, Any]:
        s3 = SafetyMetrics.safety_stability_score(self.config_metrics)
        asa = SafetyMetrics.attack_surface_area(self.config_metrics)
        detector = PhaseDetector(self.config_metrics)
        profile = detector.robustness_profile("temperature")
        transitions = detector.detect("temperature")

        return {
            "model_id": self.result.model_id,
            "probe_set": self.result.probe_set_name,
            "aggregate": {
                "safety_stability_score": s3,
                "attack_surface_area": asa,
                "robustness_auc": profile.get("auc"),
                "safe_temperature_zone": profile.get("safe_zone"),
                "phase_transitions": [
                    {
                        "temperature": t.transition_value,
                        "drop": t.drop_magnitude,
                        "confidence": t.confidence,
                    }
                    for t in transitions
                ],
            },
            "per_config": [m.to_dict() for m in self.config_metrics],
        }

    def save(self, path: str | Path | None = None) -> Path:
        import json
        import time

        if path is None:
            if self.output_dir is None:
                raise ValueError("Provide path or set output_dir on SweepReport")
            ts = int(time.time())
            fname = f"report_{self.result.model_id.replace('/', '_')}_{ts}.json"
            path = self.output_dir / fname

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[green]Report saved to {path}[/green]")
        return path

    def save_plots(self, output_dir: str | Path | None = None) -> list[Path]:
        from safety_probe.analysis.curves import SensitivityCurves

        out = Path(output_dir or self.output_dir or "outputs")
        out.mkdir(parents=True, exist_ok=True)

        curves = SensitivityCurves(self.config_metrics)
        saved = []

        temp_path = out / "temperature_curve.png"
        curves.plot_temperature_curve(save_path=temp_path)
        saved.append(temp_path)

        return saved
