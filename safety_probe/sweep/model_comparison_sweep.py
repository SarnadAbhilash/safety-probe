"""ModelComparisonSweep — compare safety across multiple model endpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from safety_probe.backends.base import BaseBackend, GenerationConfig
from safety_probe.judges.base import BaseJudge
from safety_probe.probes.probe_sets import ProbeSet
from safety_probe.sweep.grid import SweepGrid
from safety_probe.sweep.parameter_sweep import ParameterSweep, SweepResult

console = Console()


@dataclass
class ModelSpec:
    """Describes one model endpoint to probe."""

    label: str
    model_id: str
    backend: str = "openai"       # "openai" | "transformers" | "vllm"
    base_url: str | None = None   # for openai backend (local or remote)
    api_key: str | None = None    # for openai backend; falls back to env var
    quantization: str | None = None  # for transformers backend

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        return cls(
            label=d["label"],
            model_id=d["model_id"],
            backend=d.get("backend", "openai"),
            base_url=d.get("base_url"),
            api_key=d.get("api_key"),
            quantization=d.get("quantization"),
        )

    def build_backend(self) -> BaseBackend:
        if self.backend == "openai":
            from safety_probe.backends.openai_backend import OpenAIBackend
            import os
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "no-key")
            return OpenAIBackend(
                model_id=self.model_id,
                base_url=self.base_url,
                api_key=api_key,
            )
        elif self.backend == "transformers":
            from safety_probe.backends.transformers_backend import TransformersBackend
            return TransformersBackend(
                model_id=self.model_id,
                quantization=self.quantization,
            )
        elif self.backend == "vllm":
            from safety_probe.backends.vllm_backend import VLLMBackend
            return VLLMBackend(model_id=self.model_id)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")


@dataclass
class ModelComparisonResult:
    """
    Results from probing multiple model endpoints with the same probe set.

    sweep_results maps label → SweepResult (single-config sweep per model).
    The first label is treated as the baseline for delta calculations.
    """

    model_specs: list[ModelSpec]
    sweep_results: dict[str, SweepResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.model_specs]

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------

    def refusal_rates(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for label, result in self.sweep_results.items():
            total = refused = 0
            for probe_judgements in result.judgements[0]:
                for j in probe_judgements:
                    total += 1
                    if j.is_refused:
                        refused += 1
            rates[label] = refused / total if total else 0.0
        return rates

    def compliance_rates(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for label, result in self.sweep_results.items():
            total = complied = 0
            for probe_judgements in result.judgements[0]:
                for j in probe_judgements:
                    total += 1
                    if j.is_complied:
                        complied += 1
            rates[label] = complied / total if total else 0.0
        return rates

    def print_summary(self) -> None:
        refusal = self.refusal_rates()
        compliance = self.compliance_rates()
        baseline_label = self.labels[0]
        baseline = refusal[baseline_label]

        table = Table(title="Model Comparison — Safety Summary", show_lines=True)
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Backend", style="dim")
        table.add_column("Refusal Rate", justify="right")
        table.add_column("Compliance Rate", justify="right")
        table.add_column(f"Delta vs {baseline_label}", justify="right")

        spec_map = {s.label: s for s in self.model_specs}

        for label in self.labels:
            r = refusal[label]
            c = compliance[label]
            spec = spec_map[label]
            backend_str = spec.backend
            if spec.base_url:
                backend_str += f" ({spec.base_url})"
            elif spec.quantization:
                backend_str += f" ({spec.quantization})"

            if label == baseline_label:
                delta_str = "baseline"
                color = "white"
            else:
                delta = r - baseline
                delta_str = f"{delta:+.1%}"
                color = "green" if delta >= 0 else "red"

            table.add_row(
                label,
                backend_str,
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
            "model_specs": [
                {
                    "label": s.label,
                    "model_id": s.model_id,
                    "backend": s.backend,
                    "base_url": s.base_url,
                    "quantization": s.quantization,
                }
                for s in self.model_specs
            ],
            "sweep_results": {k: v.to_dict() for k, v in self.sweep_results.items()},
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[green]Saved comparison results to {path}[/green]")

    @classmethod
    def load(cls, path: str | Path) -> "ModelComparisonResult":
        from safety_probe.backends.base import GenerationConfig
        from safety_probe.judges.base import JudgementResult

        with open(path) as f:
            data = json.load(f)

        specs = [ModelSpec.from_dict(d) for d in data["model_specs"]]
        sweep_results: dict[str, SweepResult] = {}
        for label, sr_dict in data["sweep_results"].items():
            configs = [GenerationConfig(**c) for c in sr_dict["configs"]]
            judgements = [
                [
                    [JudgementResult(**j) for j in sample_list]
                    for sample_list in probe_list
                ]
                for probe_list in sr_dict["judgements"]
            ]
            sweep_results[label] = SweepResult(
                model_id=sr_dict["model_id"],
                probe_set_name=sr_dict["probe_set_name"],
                configs=configs,
                raw_results=sr_dict["raw_results"],
                judgements=judgements,
                metadata=sr_dict.get("metadata", {}),
            )

        return cls(
            model_specs=specs,
            sweep_results=sweep_results,
            metadata=data.get("metadata", {}),
        )


class ModelComparisonSweep:
    """
    Probes multiple model endpoints with the same probe set and compares safety.

    Each model is described by a ModelSpec — a label, model ID, backend type,
    and optional base_url (for locally or remotely hosted OpenAI-compatible servers).

    The first spec is used as the baseline in the delta column.

    Example — comparing a local vLLM server against Together AI:
        sweep = ModelComparisonSweep(
            specs=[
                ModelSpec(label="local-int4", model_id="llama3-8b", backend="openai",
                          base_url="http://localhost:8000/v1"),
                ModelSpec(label="together-bf16", model_id="meta-llama/Llama-3-8B-Instruct",
                          backend="openai", base_url="https://api.together.xyz/v1"),
            ],
            probe_set=load_probe_set("core"),
            judge=CompositeJudge(),
        )
        result = sweep.run()
        result.print_summary()

    Can also be constructed from a YAML config via ModelComparisonSweep.from_config().
    """

    def __init__(
        self,
        specs: list[ModelSpec],
        probe_set: ProbeSet,
        judge: BaseJudge,
        base_config: GenerationConfig | None = None,
        n_samples: int = 1,
        batch_size: int = 8,
        output_dir: str | Path | None = None,
        rate_limit_rpm: int | None = None,
        verbose: bool = False,
    ) -> None:
        if len(specs) < 2:
            raise ValueError("ModelComparisonSweep requires at least 2 model specs.")
        self.specs = specs
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
        self.rate_limit_rpm = rate_limit_rpm
        self.verbose = verbose

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        probe_set: ProbeSet,
        judge: BaseJudge,
        **kwargs: Any,
    ) -> "ModelComparisonSweep":
        """Load specs and base config from a YAML file."""
        import yaml

        spec = yaml.safe_load(Path(config_path).read_text())
        specs = [ModelSpec.from_dict(d) for d in spec["models"]]
        base_kwargs = spec.get("base", {})
        base_config = GenerationConfig(**base_kwargs) if base_kwargs else None
        return cls(specs=specs, probe_set=probe_set, judge=judge, base_config=base_config, **kwargs)

    def run(self) -> ModelComparisonResult:
        start = time.perf_counter()
        n = len(self.specs)

        console.rule("[bold blue]Model Comparison Safety Sweep[/bold blue]")
        for s in self.specs:
            loc = s.base_url or ("local weights" if s.backend == "transformers" else s.backend)
            console.print(f"  [cyan]{s.label}[/cyan]  {s.model_id}  ({loc})")
        console.print(f"\n  Probes: {len(self.probe_set.probes)}  |  Samples/model: {self.n_samples}\n")

        sweep_results: dict[str, SweepResult] = {}

        for idx, spec in enumerate(self.specs, 1):
            console.rule(f"[bold]Model {idx}/{n}: {spec.label}[/bold]")

            backend = spec.build_backend()
            grid = SweepGrid(base=self.base_config)

            sweep = ParameterSweep(
                backend=backend,
                grid=grid,
                probe_set=self.probe_set,
                judge=self.judge,
                n_samples=self.n_samples,
                batch_size=self.batch_size,
                rate_limit_rpm=self.rate_limit_rpm,
                verbose=self.verbose,
            )
            sweep_results[spec.label] = sweep.run()

        elapsed = time.perf_counter() - start
        console.print(f"\n[green]Comparison complete in {elapsed:.1f}s[/green]")

        result = ModelComparisonResult(
            model_specs=self.specs,
            sweep_results=sweep_results,
            metadata={"elapsed_s": elapsed, "n_samples": self.n_samples},
        )

        result.print_summary()

        if self.output_dir:
            timestamp = int(time.time())
            fname = f"model_comparison_{timestamp}.json"
            result.save(self.output_dir / fname)

        return result
