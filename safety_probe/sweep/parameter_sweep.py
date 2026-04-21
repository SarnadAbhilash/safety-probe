"""Core ParameterSweep orchestrator."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from safety_probe.backends.base import BaseBackend, GenerationConfig
from safety_probe.judges.base import BaseJudge, JudgementResult
from safety_probe.probes.probe_sets import ProbeSet
from safety_probe.sweep.grid import SweepGrid

console = Console()


@dataclass
class SweepResult:
    """Container for all results from a single sweep run."""

    model_id: str
    probe_set_name: str
    configs: list[GenerationConfig]
    # Outer list: configs; inner list: probes; innermost list: n_samples
    raw_results: list[list[list[str]]]  # [config][probe][sample] -> response text
    judgements: list[list[list[JudgementResult]]]  # [config][probe][sample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "probe_set_name": self.probe_set_name,
            "configs": [c.to_dict() for c in self.configs],
            "raw_results": self.raw_results,
            "judgements": [
                [
                    [j.to_dict() for j in sample_judgements]
                    for sample_judgements in probe_judgements
                ]
                for probe_judgements in self.judgements
            ],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[green]Saved sweep results to {path}[/green]")

    @classmethod
    def load(cls, path: str | Path) -> "SweepResult":
        from safety_probe.backends.base import GenerationConfig
        from safety_probe.judges.base import JudgementResult

        with open(path) as f:
            data = json.load(f)

        configs = [GenerationConfig(**c) for c in data["configs"]]
        judgements = [
            [
                [JudgementResult(**j) for j in sample_list]
                for sample_list in probe_list
            ]
            for probe_list in data["judgements"]
        ]
        return cls(
            model_id=data["model_id"],
            probe_set_name=data["probe_set_name"],
            configs=configs,
            raw_results=data["raw_results"],
            judgements=judgements,
            metadata=data.get("metadata", {}),
        )


class ParameterSweep:
    """
    Orchestrates a full safety sweep across a parameter grid.

    For each configuration in the grid, generates n_samples responses for
    every probe in the probe set, judges each response, and collects results
    into a SweepResult for downstream analysis.

    Example:
        sweep = ParameterSweep(
            backend=TransformersBackend("meta-llama/Llama-3-8B-Instruct"),
            grid=SweepGrid.temperature_sweep(),
            probe_set=load_probe_set("core"),
            judge=CompositeJudge(),
            n_samples=3,
        )
        results = sweep.run()
        results.save("outputs/llama3_temperature_sweep.json")
    """

    def __init__(
        self,
        backend: BaseBackend,
        grid: SweepGrid,
        probe_set: ProbeSet,
        judge: BaseJudge,
        n_samples: int = 1,
        batch_size: int = 8,
        output_dir: str | Path | None = None,
        rate_limit_rpm: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.backend = backend
        self.grid = grid
        self.probe_set = probe_set
        self.judge = judge
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.output_dir = Path(output_dir) if output_dir else None
        # Minimum seconds to wait between generate() calls (derived from rpm limit)
        self._min_gap_s: float = (60.0 / rate_limit_rpm) if rate_limit_rpm else 0.0
        self.verbose = verbose

    def run(self) -> SweepResult:
        configs = self.grid.configs()
        probes = self.probe_set.probes
        n_configs = len(configs)
        n_probes = len(probes)
        total_generations = n_configs * n_probes * self.n_samples

        console.rule(f"[bold blue]Safety Sweep — {self.backend.model_id}[/bold blue]")
        console.print(
            f"  Configs: {n_configs}  |  Probes: {n_probes}  |  Samples/config: {self.n_samples}"
        )
        console.print(f"  Total generations: {total_generations}")
        if self._min_gap_s > 0:
            est_wait = self._min_gap_s * (total_generations / self.batch_size)
            console.print(
                f"  Rate limit: {60.0 / self._min_gap_s:.0f} req/min "
                f"(+~{est_wait:.0f}s throttle overhead)\n"
            )
        else:
            console.print()

        all_raw: list[list[list[str]]] = []
        all_judgements: list[list[list[JudgementResult]]] = []

        start_time = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Sweeping...", total=n_configs)

            with self.backend:
                for cfg_idx, config in enumerate(configs):
                    progress.update(
                        task,
                        description=f"Config {cfg_idx + 1}/{n_configs} | temp={config.temperature:.2f}",
                    )

                    config_raw: list[list[str]] = []
                    config_judgements: list[list[JudgementResult]] = []

                    # Batch probes for efficiency
                    for batch_start in range(0, n_probes, self.batch_size):
                        batch_probes = probes[batch_start: batch_start + self.batch_size]
                        probe_texts = [p.text for p in batch_probes]

                        probe_raw: list[list[str]] = [[] for _ in batch_probes]
                        probe_judgements: list[list[JudgementResult]] = [[] for _ in batch_probes]

                        for sample_idx in range(self.n_samples):
                            if self.verbose:
                                for p in batch_probes:
                                    console.print(
                                        f"  [dim]→ [{p.category.value}] {p.text[:80]}{'...' if len(p.text) > 80 else ''}[/dim]"
                                    )
                                console.print(f"  [dim]  calling API...[/dim]")

                            _call_start = time.perf_counter()
                            gen_results = self.backend.generate(probe_texts, config)
                            elapsed_call = time.perf_counter() - _call_start

                            if self.verbose:
                                console.print(f"  [dim]  ✓ {len(gen_results)} responses in {elapsed_call:.1f}s[/dim]")

                            if self._min_gap_s > 0:
                                sleep_s = max(0.0, self._min_gap_s - elapsed_call)
                                if sleep_s > 0:
                                    if self.verbose:
                                        console.print(f"  [dim]  throttling {sleep_s:.1f}s...[/dim]")
                                    time.sleep(sleep_s)

                            for i, gen in enumerate(gen_results):
                                probe_raw[i].append(gen.response)
                                judgement = self.judge.judge(
                                    prompt=gen.prompt,
                                    response=gen.response,
                                    probe_category=batch_probes[i].category,
                                )
                                probe_judgements[i].append(judgement)
                                if self.verbose:
                                    verdict_color = "green" if judgement.is_refused else "red" if judgement.is_complied else "yellow"
                                    console.print(
                                        f"  [dim]  [{verdict_color}]{judgement.verdict.value}[/{verdict_color}] "
                                        f"(conf={judgement.confidence:.2f}) "
                                        f"← {gen.response[:100].strip()}{'...' if len(gen.response) > 100 else ''}[/dim]"
                                    )

                        config_raw.extend(probe_raw)
                        config_judgements.extend(probe_judgements)

                    all_raw.append(config_raw)
                    all_judgements.append(config_judgements)
                    progress.advance(task)

        elapsed = time.perf_counter() - start_time
        console.print(f"\n[green]Sweep complete in {elapsed:.1f}s[/green]")

        result = SweepResult(
            model_id=self.backend.model_id,
            probe_set_name=self.probe_set.name,
            configs=configs,
            raw_results=all_raw,
            judgements=all_judgements,
            metadata={
                "elapsed_s": elapsed,
                "n_samples": self.n_samples,
                "total_generations": total_generations,
            },
        )

        if self.output_dir:
            timestamp = int(time.time())
            fname = f"{self.backend.model_id.replace('/', '_')}_{timestamp}.json"
            result.save(self.output_dir / fname)

        return result
