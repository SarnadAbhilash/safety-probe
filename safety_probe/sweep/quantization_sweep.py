"""QuantizationSweep — measures safety degradation across precision levels."""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from safety_probe.backends.base import GenerationConfig
from safety_probe.backends.transformers_backend import TransformersBackend
from safety_probe.judges.base import BaseJudge
from safety_probe.probes.probe_sets import ProbeSet
from safety_probe.sweep.grid import SweepGrid
from safety_probe.sweep.parameter_sweep import ParameterSweep, SweepResult

console = Console()

# Canonical ordering for display (most memory → least memory)
_QUANT_ORDER = ["fp32", "bf16", "fp16", "int8", "int4"]


@dataclass
class QuantizationSweepResult:
    """
    Results from sweeping a model across quantization levels.

    sweep_results maps quantization label → SweepResult (single-config sweep).
    """

    model_id: str
    quantization_levels: list[str]
    sweep_results: dict[str, SweepResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------

    def refusal_rates(self) -> dict[str, float]:
        """Fraction of harmful probes that were refused, per quantization level."""
        rates: dict[str, float] = {}
        for quant, result in self.sweep_results.items():
            total = refused = 0
            # single-config sweep → index [0]
            for probe_judgements in result.judgements[0]:
                for j in probe_judgements:
                    total += 1
                    if j.is_refused:
                        refused += 1
            rates[quant] = refused / total if total else 0.0
        return rates

    def compliance_rates(self) -> dict[str, float]:
        """Fraction of harmful probes that were complied with, per quantization level."""
        rates: dict[str, float] = {}
        for quant, result in self.sweep_results.items():
            total = complied = 0
            for probe_judgements in result.judgements[0]:
                for j in probe_judgements:
                    total += 1
                    if j.is_complied:
                        complied += 1
            rates[quant] = complied / total if total else 0.0
        return rates

    def print_summary(self) -> None:
        """Print a Rich table comparing safety rates across quantization levels."""
        refusal = self.refusal_rates()
        compliance = self.compliance_rates()

        ordered = [q for q in _QUANT_ORDER if q in refusal] + [
            q for q in refusal if q not in _QUANT_ORDER
        ]

        table = Table(title=f"Quantization Safety Summary — {self.model_id}", show_lines=True)
        table.add_column("Precision", style="cyan", no_wrap=True)
        table.add_column("Refusal Rate", justify="right")
        table.add_column("Compliance Rate", justify="right")
        table.add_column("Delta vs fp16", justify="right")

        baseline = refusal.get("fp16") or refusal.get("bf16") or next(iter(refusal.values()), 0.0)

        for quant in ordered:
            r = refusal[quant]
            c = compliance[quant]
            delta = r - baseline
            delta_str = f"{delta:+.1%}" if quant not in ("fp16", "bf16") else "baseline"
            color = "green" if delta >= 0 else "red"
            table.add_row(
                quant,
                f"{r:.1%}",
                f"{c:.1%}",
                f"[{color}]{delta_str}[/{color}]",
            )

        console.print(table)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "quantization_levels": self.quantization_levels,
            "sweep_results": {k: v.to_dict() for k, v in self.sweep_results.items()},
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[green]Saved quantization sweep to {path}[/green]")

    @classmethod
    def load(cls, path: str | Path) -> "QuantizationSweepResult":
        with open(path) as f:
            data = json.load(f)
        sweep_results = {k: SweepResult.load.__func__(SweepResult, None) for k in data["sweep_results"]}  # type: ignore[attr-defined]
        # Deserialize each SweepResult from its dict
        from safety_probe.backends.base import GenerationConfig
        from safety_probe.judges.base import JudgementResult

        deserialized: dict[str, SweepResult] = {}
        for quant, sr_dict in data["sweep_results"].items():
            configs = [GenerationConfig(**c) for c in sr_dict["configs"]]
            judgements = [
                [
                    [JudgementResult(**j) for j in sample_list]
                    for sample_list in probe_list
                ]
                for probe_list in sr_dict["judgements"]
            ]
            deserialized[quant] = SweepResult(
                model_id=sr_dict["model_id"],
                probe_set_name=sr_dict["probe_set_name"],
                configs=configs,
                raw_results=sr_dict["raw_results"],
                judgements=judgements,
                metadata=sr_dict.get("metadata", {}),
            )

        return cls(
            model_id=data["model_id"],
            quantization_levels=data["quantization_levels"],
            sweep_results=deserialized,
            metadata=data.get("metadata", {}),
        )


class QuantizationSweep:
    """
    Measures how LLM safety changes across quantization precision levels.

    For each level in quantization_levels, loads the model at that precision,
    runs all probes under base_config, then unloads before moving to the next level.
    This avoids OOM from holding multiple quantized copies simultaneously.

    Supported levels: "fp32", "bf16", "fp16", "int8", "int4"
    int8/int4 require bitsandbytes: pip install safety-probe[quant]

    Example:
        sweep = QuantizationSweep(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            quantization_levels=["bf16", "int8", "int4"],
            probe_set=load_probe_set("core"),
            judge=CompositeJudge(),
        )
        result = sweep.run()
        result.print_summary()
        result.save("outputs/quant_sweep.json")
    """

    def __init__(
        self,
        model_id: str,
        quantization_levels: list[str],
        probe_set: ProbeSet,
        judge: BaseJudge,
        base_config: GenerationConfig | None = None,
        n_samples: int = 1,
        batch_size: int = 8,
        output_dir: str | Path | None = None,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model_id = model_id
        self.quantization_levels = quantization_levels
        self.probe_set = probe_set
        self.judge = judge
        self.base_config = base_config or GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=256,
            sampling_strategy="multinomial",
            seed=42,
        )
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.output_dir = Path(output_dir) if output_dir else None
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.verbose = verbose

    def run(self) -> QuantizationSweepResult:
        start = time.perf_counter()
        n = len(self.quantization_levels)

        console.rule(f"[bold blue]Quantization Safety Sweep — {self.model_id}[/bold blue]")
        console.print(f"  Levels ({n}): {', '.join(self.quantization_levels)}")
        console.print(f"  Probes: {len(self.probe_set.probes)}  |  Samples/level: {self.n_samples}\n")

        sweep_results: dict[str, SweepResult] = {}

        for idx, quant in enumerate(self.quantization_levels, 1):
            console.rule(f"[bold]Level {idx}/{n}: {quant}[/bold]")

            backend = TransformersBackend(
                model_id=self.model_id,
                quantization=quant,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
            )

            # Tag the config with the current quantization level for serialization
            config = dataclasses.replace(self.base_config, quantization=quant)
            grid = SweepGrid(base=config)  # single-point grid

            sweep = ParameterSweep(
                backend=backend,
                grid=grid,
                probe_set=self.probe_set,
                judge=self.judge,
                n_samples=self.n_samples,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
            sweep_results[quant] = sweep.run()

        elapsed = time.perf_counter() - start
        console.print(f"\n[green]Quantization sweep complete in {elapsed:.1f}s[/green]")

        result = QuantizationSweepResult(
            model_id=self.model_id,
            quantization_levels=self.quantization_levels,
            sweep_results=sweep_results,
            metadata={"elapsed_s": elapsed, "n_samples": self.n_samples},
        )

        result.print_summary()

        if self.output_dir:
            timestamp = int(time.time())
            fname = f"{self.model_id.replace('/', '_')}_quant_sweep_{timestamp}.json"
            result.save(self.output_dir / fname)

        return result
