"""safety-probe CLI — probe, sweep, analyze, report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(
    name="safety-probe",
    help="Inference-time safety probing framework for LLMs.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# sweep subcommand
# ---------------------------------------------------------------------------

@app.command()
def sweep(
    model: Annotated[str, typer.Option("--model", "-m", help="Model ID (HF hub or OpenAI name)")],
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="YAML sweep config")] = None,
    probe_set: Annotated[str, typer.Option("--probe-set", "-p", help="Probe set name")] = "core",
    backend: Annotated[str, typer.Option("--backend", "-b", help="Backend: transformers|vllm|openai")] = "transformers",
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", help="Samples per config")] = 1,
    temperatures: Annotated[Optional[str], typer.Option("--temperatures", help="Comma-separated temps, e.g. 0.0,0.5,1.0")] = None,
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory")] = "outputs",
    judge: Annotated[str, typer.Option("--judge", "-j", help="Judge level: rule|rule+classifier|full")] = "rule",
    base_url: Annotated[Optional[str], typer.Option("--base-url", help="API base URL (e.g. https://api.groq.com/openai/v1)")] = None,
    api_key_env: Annotated[str, typer.Option("--api-key-env", help="Env var name holding the API key")] = "OPENAI_API_KEY",
    rate_limit: Annotated[Optional[int], typer.Option("--rate-limit", "-r", help="Max requests per minute (for hosted APIs)")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print each probe, response, and verdict in real time")] = False,
) -> None:
    """Run a parameter sweep and save results."""
    from safety_probe.backends.transformers_backend import TransformersBackend
    from safety_probe.backends.openai_backend import OpenAIBackend
    from safety_probe.judges.composite import CompositeJudge
    from safety_probe.probes.probe_sets import load_probe_set
    from safety_probe.sweep.grid import SweepGrid
    from safety_probe.sweep.parameter_sweep import ParameterSweep

    # Build backend
    if backend == "transformers":
        be = TransformersBackend(model)
    elif backend == "openai":
        import os
        be = OpenAIBackend(model, base_url=base_url, api_key=os.environ.get(api_key_env))
    elif backend == "vllm":
        from safety_probe.backends.vllm_backend import VLLMBackend
        be = VLLMBackend(model)
    else:
        console.print(f"[red]Unknown backend: {backend}[/red]")
        raise typer.Exit(1)

    # Build grid
    if config:
        import yaml
        spec = yaml.safe_load(config.read_text())
        grid = SweepGrid.from_dict(spec)
    elif temperatures:
        temps = [float(t) for t in temperatures.split(",")]
        grid = SweepGrid.temperature_sweep(temps)
    else:
        grid = SweepGrid.temperature_sweep()

    # Build judge
    if judge == "rule":
        levels = ["rule"]
    elif judge == "rule+classifier":
        levels = ["rule", "classifier"]
    else:
        levels = ["rule", "classifier", "llm"]
    j = CompositeJudge(levels=levels)

    probe_set_obj = load_probe_set(probe_set)

    sweep_obj = ParameterSweep(
        backend=be,
        grid=grid,
        probe_set=probe_set_obj,
        judge=j,
        n_samples=n_samples,
        output_dir=output_dir,
        rate_limit_rpm=rate_limit,
        verbose=verbose,
    )

    results = sweep_obj.run()
    console.print(f"\n[bold green]Sweep complete.[/bold green] Results saved to [cyan]{output_dir}[/cyan]")
    return results


# ---------------------------------------------------------------------------
# quant-sweep subcommand
# ---------------------------------------------------------------------------

@app.command(name="quant-sweep")
def quant_sweep(
    model: Annotated[str, typer.Option("--model", "-m", help="HuggingFace model ID")],
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="YAML quant sweep config")] = None,
    quantization: Annotated[Optional[str], typer.Option("--quantization", "-q", help="Comma-separated levels, e.g. bf16,int8,int4")] = None,
    probe_set: Annotated[str, typer.Option("--probe-set", "-p", help="Probe set name")] = "core",
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", help="Samples per quantization level")] = 1,
    temperature: Annotated[float, typer.Option("--temperature", "-t", help="Base generation temperature")] = 0.7,
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory")] = "outputs",
    judge: Annotated[str, typer.Option("--judge", "-j", help="Judge level: rule|rule+classifier|full")] = "rule",
    device_map: Annotated[str, typer.Option("--device-map", help="Device map for model loading")] = "auto",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Sweep a model across quantization precision levels (bf16→fp16→int8→int4)."""
    import yaml
    from safety_probe.backends.base import GenerationConfig
    from safety_probe.judges.composite import CompositeJudge
    from safety_probe.probes.probe_sets import load_probe_set
    from safety_probe.sweep.quantization_sweep import QuantizationSweep

    # Resolve quantization levels and base config
    if config:
        spec = yaml.safe_load(config.read_text())
        levels = spec.get("quantization_levels", ["bf16", "fp16", "int8", "int4"])
        base_kwargs = spec.get("base", {})
        base_cfg = GenerationConfig(**base_kwargs) if base_kwargs else None
    else:
        levels = quantization.split(",") if quantization else ["bf16", "fp16", "int8", "int4"]
        base_cfg = GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=256,
            sampling_strategy="multinomial",
            seed=42,
        )

    # Build judge
    if judge == "rule":
        judge_levels = ["rule"]
    elif judge == "rule+classifier":
        judge_levels = ["rule", "classifier"]
    else:
        judge_levels = ["rule", "classifier", "llm"]
    j = CompositeJudge(levels=judge_levels)

    probe_set_obj = load_probe_set(probe_set)

    sweep = QuantizationSweep(
        model_id=model,
        quantization_levels=levels,
        probe_set=probe_set_obj,
        judge=j,
        base_config=base_cfg,
        n_samples=n_samples,
        output_dir=output_dir,
        device_map=device_map,
        verbose=verbose,
    )
    sweep.run()


# ---------------------------------------------------------------------------
# compare subcommand
# ---------------------------------------------------------------------------

@app.command()
def compare(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="YAML model comparison config")] = None,
    model: Annotated[Optional[list[str]], typer.Option("--model", "-m", help="label:model_id:base_url  (repeat for each model)")] = None,
    probe_set: Annotated[str, typer.Option("--probe-set", "-p")] = "core",
    n_samples: Annotated[int, typer.Option("--n-samples", "-n")] = 1,
    temperature: Annotated[float, typer.Option("--temperature", "-t")] = 0.7,
    output_dir: Annotated[str, typer.Option("--output-dir", "-o")] = "outputs",
    judge: Annotated[str, typer.Option("--judge", "-j", help="rule|rule+classifier|full")] = "rule",
    rate_limit: Annotated[Optional[int], typer.Option("--rate-limit", "-r", help="Max requests/min per model")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Compare safety across multiple model endpoints with the same probe set.

    \b
    Using a config file (recommended):
        safety-probe compare --config configs/model_comparison.yaml

    \b
    Inline (label:model_id:base_url, repeat --model for each):
        safety-probe compare \\
          --model "local-int4:llama3:http://localhost:8000/v1" \\
          --model "local-bf16:llama3:http://localhost:9000/v1"
    """
    import yaml
    from safety_probe.backends.base import GenerationConfig
    from safety_probe.judges.composite import CompositeJudge
    from safety_probe.probes.probe_sets import load_probe_set
    from safety_probe.sweep.model_comparison_sweep import ModelComparisonSweep, ModelSpec

    # Build specs
    if config:
        spec_data = yaml.safe_load(config.read_text())
        specs = [ModelSpec.from_dict(d) for d in spec_data["models"]]
        base_kwargs = spec_data.get("base", {})
        base_cfg = GenerationConfig(**base_kwargs) if base_kwargs else None
    elif model:
        specs = []
        for entry in model:
            parts = entry.split(":", 2)
            if len(parts) < 2:
                console.print(f"[red]Invalid --model format (expected label:model_id[:base_url]): {entry}[/red]")
                raise typer.Exit(1)
            label, model_id = parts[0], parts[1]
            base_url = parts[2] if len(parts) == 3 else None
            specs.append(ModelSpec(label=label, model_id=model_id, backend="openai", base_url=base_url))
        base_cfg = GenerationConfig(temperature=temperature, top_p=0.95, max_new_tokens=256, seed=42)
    else:
        console.print("[red]Provide --config or at least two --model entries.[/red]")
        raise typer.Exit(1)

    if len(specs) < 2:
        console.print("[red]Need at least 2 models to compare.[/red]")
        raise typer.Exit(1)

    # Build judge
    if judge == "rule":
        judge_levels = ["rule"]
    elif judge == "rule+classifier":
        judge_levels = ["rule", "classifier"]
    else:
        judge_levels = ["rule", "classifier", "llm"]
    j = CompositeJudge(levels=judge_levels)

    sweep = ModelComparisonSweep(
        specs=specs,
        probe_set=load_probe_set(probe_set),
        judge=j,
        base_config=base_cfg,
        n_samples=n_samples,
        output_dir=output_dir,
        rate_limit_rpm=rate_limit,
        verbose=verbose,
    )
    sweep.run()


# ---------------------------------------------------------------------------
# analyze subcommand
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    results_path: Annotated[Path, typer.Argument(help="Path to sweep results JSON")],
    probe_set: Annotated[str, typer.Option("--probe-set", "-p")] = "core",
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", "-o")] = None,
    plots: Annotated[bool, typer.Option("--plots/--no-plots")] = True,
    rejudge: Annotated[bool, typer.Option("--rejudge", help="Re-run judge on saved responses (picks up judge fixes)")] = False,
    judge: Annotated[str, typer.Option("--judge", "-j", help="Judge level: rule|rule+llm")] = "rule",
    judge_provider: Annotated[str, typer.Option("--judge-provider", help="LLM judge provider: together|groq|anthropic|openai|openrouter")] = "together",
    judge_model: Annotated[str, typer.Option("--judge-model", help="LLM judge model ID")] = "deepseek-ai/DeepSeek-V3.1",
) -> None:
    """Analyze saved sweep results and print a report."""
    from safety_probe.sweep.parameter_sweep import SweepResult
    from safety_probe.probes.probe_sets import load_probe_set
    from safety_probe.metrics.safety_metrics import SafetyMetrics
    from safety_probe.analysis.report import SweepReport
    from safety_probe.judges.composite import CompositeJudge
    from safety_probe.judges.base import JudgementResult

    result = SweepResult.load(results_path)
    probe_set_obj = load_probe_set(probe_set)

    if rejudge:
        import time as _time
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        if judge == "rule+llm":
            judge_obj = CompositeJudge(levels=["rule", "llm"], llm_provider=judge_provider, llm_model=judge_model)
            min_gap_s = 60.0 / 25  # 25 rpm conservative for LLM judge
        else:
            judge_obj = CompositeJudge(levels=["rule"])
            min_gap_s = 0.0
        probes = probe_set_obj.probes
        n_configs = len(result.configs)
        n_probes = len(probes)
        total = n_configs * n_probes
        new_judgements = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Re-judging...", total=total)
            for cfg_idx in range(n_configs):
                config_judgements = []
                for probe_idx, probe in enumerate(probes):
                    if probe_idx >= len(result.raw_results[cfg_idx]):
                        continue
                    sample_judgements = []
                    for response in result.raw_results[cfg_idx][probe_idx]:
                        t0 = _time.perf_counter()
                        j = judge_obj.judge(
                            prompt=probe.text,
                            response=response,
                            probe_category=probe.category,
                        )
                        sample_judgements.append(j)
                        if min_gap_s > 0:
                            elapsed = _time.perf_counter() - t0
                            sleep_s = max(0.0, min_gap_s - elapsed)
                            if sleep_s > 0:
                                _time.sleep(sleep_s)
                    config_judgements.append(sample_judgements)
                    progress.advance(task)
                new_judgements.append(config_judgements)
        result.judgements = new_judgements
        console.print("[green]Re-judging complete.[/green]\n")

    metrics_engine = SafetyMetrics(result, probe_set_obj)
    config_metrics = metrics_engine.compute()

    out_dir = output_dir or results_path.parent
    report = SweepReport(result, config_metrics, probe_set_obj, output_dir=out_dir)
    report.print_summary()
    report.save(out_dir / "report.json")

    if plots:
        saved = report.save_plots()
        for p in saved:
            console.print(f"[green]Plot saved: {p}[/green]")


# ---------------------------------------------------------------------------
# probes subcommand
# ---------------------------------------------------------------------------

@app.command()
def probes(
    list_sets: Annotated[bool, typer.Option("--list", "-l", help="List available probe sets")] = False,
    show: Annotated[Optional[str], typer.Option("--show", "-s", help="Show probes in a set")] = None,
    category: Annotated[Optional[str], typer.Option("--category", "-c")] = None,
) -> None:
    """List or inspect probe sets."""
    from safety_probe.probes.probe_sets import _REGISTRY, load_probe_set

    if list_sets or (show is None):
        console.print("\n[bold]Available probe sets:[/bold]")
        for name, ps in _REGISTRY.items():
            console.print(f"  [cyan]{name}[/cyan] — {len(ps)} probes — {ps.description}")
        return

    ps = load_probe_set(show)
    filtered = ps.probes
    if category:
        filtered = [p for p in filtered if p.category.value == category]

    console.print(f"\n[bold]{ps.name}[/bold] ({len(filtered)} probes)\n")
    for probe in filtered:
        expected = "[green]refusal[/green]" if probe.expected_refusal else "[blue]comply[/blue]"
        console.print(f"  [{probe.tier}] {expected} [{probe.category.value}]")
        console.print(f"    {probe.text[:100]}{'...' if len(probe.text) > 100 else ''}\n")


# ---------------------------------------------------------------------------
# profiles subcommand
# ---------------------------------------------------------------------------

@app.command()
def profiles(
    use_case: Annotated[Optional[str], typer.Argument(help="Show config for a use case")] = None,
) -> None:
    """Show recommended safety parameter profiles."""
    from safety_probe.mitigations.profiles import PROFILES

    if use_case:
        if use_case not in PROFILES:
            console.print(f"[red]Unknown use case: {use_case}[/red]")
            raise typer.Exit(1)
        p = PROFILES[use_case]
        console.print(f"\n[bold]{p.name}[/bold] — {p.description}")
        console.print(f"Safety threshold: {p.safety_threshold}")
        console.print(f"Config: {json.dumps(p.config.to_dict(), indent=2)}")
    else:
        console.print("\n[bold]Safety Profiles:[/bold]")
        for name, p in PROFILES.items():
            console.print(
                f"  [cyan]{name:10}[/cyan] threshold={p.safety_threshold:.2f}  "
                f"temp={p.config.temperature}  {p.description[:60]}"
            )


if __name__ == "__main__":
    app()
