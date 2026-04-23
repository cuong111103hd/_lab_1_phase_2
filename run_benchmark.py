"""
run_benchmark.py — Main benchmark runner with Rich logging.

Rich logging shows per-question:
  - Question text
  - Agent flow: Attempt N → Evaluate N → (Reflect N → Attempt N+1 ...)
  - Score (✅ / ❌), token count, latency
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint
from rich.text import Text
from rich.rule import Rule

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
from src.reflexion_lab.schemas import QAExample, RunRecord

console = Console()
app = typer.Typer(add_completion=False)


def _difficulty_color(difficulty: str) -> str:
    return {"easy": "green", "medium": "yellow", "hard": "red"}.get(difficulty, "white")


def _log_record(idx: int, total: int, record: RunRecord, example: QAExample) -> None:
    """Pretty-print one record after it's done."""
    diff_color = _difficulty_color(example.difficulty)

    # Header
    console.print(Rule(
        f"[bold]{idx}/{total}[/bold] [{diff_color}]{example.difficulty.upper()}[/{diff_color}] "
        f"[cyan]{example.qid}[/cyan] | [bold]{record.agent_type.upper()}[/bold]",
        style="dim"
    ))

    # Question
    console.print(f"  [bold white]Q:[/bold white] {example.question}")
    console.print(f"  [dim]Gold:[/dim] [green]{example.gold_answer}[/green]")

    # Per-attempt trace
    for trace in record.traces:
        score_icon = "✅" if trace.score == 1 else "❌"
        score_color = "green" if trace.score == 1 else "red"
        console.print(
            f"  ┌─ [bold]Attempt {trace.attempt_id}[/bold] → "
            f"[{score_color}]{score_icon} score={trace.score}[/{score_color}]  "
            f"[yellow]tokens={trace.token_estimate}[/yellow]  "
            f"[dim]{trace.latency_ms}ms[/dim]"
        )
        console.print(f"  │  Predicted: [italic]{trace.answer}[/italic]")
        console.print(f"  │  Reason   : [dim]{trace.reason[:120]}[/dim]")

        if trace.reflection:
            r = trace.reflection
            console.print(f"  ├─ [magenta]Reflect {trace.attempt_id}[/magenta]")
            console.print(f"  │  Lesson  : [dim]{r.lesson[:120]}[/dim]")
            console.print(f"  │  Strategy: [italic cyan]{r.next_strategy[:120]}[/italic cyan]")

    # Final summary line
    final_icon = "✅ CORRECT" if record.is_correct else "❌ WRONG"
    final_color = "green" if record.is_correct else "red"
    console.print(
        f"  └─ Final: [{final_color}]{final_icon}[/{final_color}]  "
        f"attempts={record.attempts}  "
        f"total_tokens=[yellow]{record.token_estimate}[/yellow]  "
        f"failure_mode=[dim]{record.failure_mode}[/dim]"
    )
    console.print()


def _run_agent_with_logging(
    agent_label: str,
    agent,
    examples: list[QAExample],
    start_idx: int,
    total: int,
) -> list[RunRecord]:
    records = []
    for i, example in enumerate(examples, start_idx):
        record = agent.run(example)
        _log_record(i, total, record, example)
        records.append(record)
    return records


@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/real_run",
    reflexion_attempts: int = 3,
    limit: int = 0,  # 0 = no limit (run all)
) -> None:
    console.print(Panel.fit(
        "[bold cyan]🚀 Reflexion Agent Benchmark[/bold cyan]\n"
        f"Dataset: [yellow]{dataset}[/yellow]  |  "
        f"Reflexion attempts: [yellow]{reflexion_attempts}[/yellow]  |  "
        f"Output: [yellow]{out_dir}[/yellow]",
        border_style="cyan",
    ))

    examples = load_dataset(dataset)
    if limit > 0:
        examples = examples[:limit]

    n = len(examples)
    total_runs = n * 2  # ReAct + Reflexion

    # ── ReAct ──────────────────────────────────────────────
    console.print(Rule("[bold green]ReAct Agent[/bold green]", style="green"))
    react = ReActAgent()
    react_records = _run_agent_with_logging("ReAct", react, examples, 1, n)

    # ── Reflexion ───────────────────────────────────────────
    console.print(Rule("[bold magenta]Reflexion Agent[/bold magenta]", style="magenta"))
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    reflexion_records = _run_agent_with_logging("Reflexion", reflexion, examples, 1, n)

    # ── Save outputs ────────────────────────────────────────
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    report = build_report(all_records, dataset_name=Path(dataset).name, mode="real")
    json_path, md_path = save_report(report, out_path)

    # ── Final summary table ─────────────────────────────────
    table = Table(title="📊 Benchmark Summary", border_style="cyan", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("ReAct", justify="right")
    table.add_column("Reflexion", justify="right")
    table.add_column("Δ", justify="right")

    s = report.summary
    react_s = s.get("react", {})
    refl_s  = s.get("reflexion", {})
    delta_s = s.get("delta_reflexion_minus_react", {})

    table.add_row("Exact Match (EM)",   str(react_s.get("em", 0)),                str(refl_s.get("em", 0)),                f"{delta_s.get('em_abs', 0):+.4f}")
    table.add_row("Avg Attempts",       str(react_s.get("avg_attempts", 0)),       str(refl_s.get("avg_attempts", 0)),       f"{delta_s.get('attempts_abs', 0):+.4f}")
    table.add_row("Avg Tokens",         str(react_s.get("avg_token_estimate", 0)), str(refl_s.get("avg_token_estimate", 0)), f"{delta_s.get('tokens_abs', 0):+.2f}")
    table.add_row("Avg Latency (ms)",   str(react_s.get("avg_latency_ms", 0)),     str(refl_s.get("avg_latency_ms", 0)),     f"{delta_s.get('latency_abs', 0):+.2f}")

    console.print(table)

    rprint(f"\n[green]✓ Saved[/green] {json_path}")
    rprint(f"[green]✓ Saved[/green] {md_path}")
    rprint(f"\n[bold]Full summary JSON:[/bold]")
    rprint(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
