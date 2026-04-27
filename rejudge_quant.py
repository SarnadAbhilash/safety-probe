"""Rejudge a QuantizationSweepResult file using the rule+llm cascade."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from safety_probe.judges.composite import CompositeJudge
from safety_probe.judges.base import JudgementResult
from safety_probe.backends.base import GenerationConfig
from safety_probe.probes.probe_sets import load_probe_set
from safety_probe.sweep.parameter_sweep import SweepResult
from safety_probe.sweep.quantization_sweep import QuantizationSweepResult

load_dotenv()
console = Console()

INPUT = Path("outputs/meta-llama_Meta-Llama-3-8B-Instruct_quant_sweep_1776836096.json")
OUTPUT = Path("outputs/meta-llama_Meta-Llama-3-8B-Instruct_quant_sweep_rejudged.json")
PROBE_SET = "core"
JUDGE_PROVIDER = "together"
JUDGE_MODEL = "deepseek-ai/DeepSeek-V3.1"
RPM = 25  # conservative rate limit

probe_set = load_probe_set(PROBE_SET)
judge = CompositeJudge(levels=["rule", "llm"], llm_provider=JUDGE_PROVIDER, llm_model=JUDGE_MODEL)
min_gap_s = 60.0 / RPM

with open(INPUT) as f:
    data = json.load(f)

quant_levels = data["quantization_levels"]
total_calls = len(quant_levels) * len(probe_set.probes)

console.print(f"[bold]Rejudging quantization sweep[/bold]")
console.print(f"  Levels: {quant_levels}")
console.print(f"  Probes: {len(probe_set.probes)}  |  Total judge calls: {total_calls}\n")

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TimeElapsedColumn(),
    console=console,
) as progress:
    task = progress.add_task("Rejudging...", total=total_calls)

    for quant in quant_levels:
        sr_dict = data["sweep_results"][quant]
        configs = [GenerationConfig(**c) for c in sr_dict["configs"]]
        new_judgements = []

        for cfg_idx in range(len(configs)):
            config_judgements = []
            for probe_idx, probe in enumerate(probe_set.probes):
                if probe_idx >= len(sr_dict["raw_results"][cfg_idx]):
                    continue
                sample_judgements = []
                for response in sr_dict["raw_results"][cfg_idx][probe_idx]:
                    t0 = time.perf_counter()
                    j = judge.judge(prompt=probe.text, response=response, probe_category=probe.category)
                    sample_judgements.append(j.__dict__)
                    elapsed = time.perf_counter() - t0
                    sleep_s = max(0.0, min_gap_s - elapsed)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                config_judgements.append(sample_judgements)
                progress.advance(task)
            new_judgements.append(config_judgements)

        data["sweep_results"][quant]["judgements"] = new_judgements
        console.print(f"  [green]✓[/green] {quant} rejudged")

with open(OUTPUT, "w") as f:
    json.dump(data, f, indent=2)

console.print(f"\n[bold green]Done.[/bold green] Saved to {OUTPUT}")
