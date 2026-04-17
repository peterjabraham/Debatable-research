"""
Blog Agent Pipeline — CLI entry point.

Usage:
  python main.py run \\
    --topic "The future of email marketing" \\
    --audience "Senior marketing leaders" \\
    --tone "Direct and analytical" \\
    --words 900 \\
    --sources "https://example.com/research1" "https://example.com/blog2" \\
    --cluster-angle "AI personalisation in email"

  python main.py resume <run-id>

  python main.py dry-run \\
    --topic "..." --audience "..." --tone "..."
    # Prints all 6 prompts with state filled in. No API calls.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv(override=True)

app = typer.Typer(help="Blog Agent Pipeline — research, debate, and write with Claude.")
console = Console()


def _get_llm_client() -> "LLMClient":  # noqa: F821
    from src.llm.client import LLMClient

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY environment variable not set.")
        raise typer.Exit(1)
    return LLMClient(api_key=api_key)


def _get_research_client():
    """Perplexity client for A1 if PERPLEXITY_API_KEY is set; else None."""
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        return None
    from src.llm.perplexity import PerplexityClient
    return PerplexityClient(api_key=api_key)


@app.command()
def run(
    topic: str = typer.Option(..., help="The topic to research and debate"),
    audience: str = typer.Option(..., help="Target audience for the blog post"),
    tone: str = typer.Option(..., help="Writing tone (e.g. 'Direct and analytical')"),
    words: int = typer.Option(900, help="Target word count for the blog post"),
    sources: list[str] = typer.Option([], help="URLs to include as sources"),
    cluster_angle: str = typer.Option(None, "--cluster-angle", help="Optional content cluster angle"),
) -> None:
    """Start a new pipeline run."""
    from src.io.input_parser import parse_run_args
    from src.pipeline.runner import PipelineRunner

    state = parse_run_args(
        topic=topic,
        audience=audience,
        tone=tone,
        words=words,
        sources=list(sources),
        cluster_angle=cluster_angle,
    )

    console.print(Panel(f"[bold]Run ID:[/bold] {state.run_id}", title="Starting pipeline"))
    console.print(f"Topic: {topic}")
    console.print(f"Audience: {audience}")
    console.print(f"Tone: {tone}")
    console.print(f"Words: {words}")

    llm = _get_llm_client()
    research = _get_research_client()
    runner = PipelineRunner(llm, research_client=research)

    try:
        result = asyncio.run(runner.run(state))
        console.print(f"\n[green]✓ Pipeline completed.[/green] Run ID: {result.run_id}")
        console.print(f"Output in: output/{result.run_id}/")
    except Exception as exc:
        console.print(f"\n[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume from checkpoint"),
) -> None:
    """Resume a pipeline run from checkpoint."""
    from src.pipeline.checkpoints import load
    from src.pipeline.runner import PipelineRunner
    from src.utils.errors import CheckpointCorruptError

    try:
        state = load(run_id)
    except CheckpointCorruptError as exc:
        console.print(f"[red]Checkpoint error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]Resuming run:[/bold] {run_id}", title="Resume"))

    llm = _get_llm_client()
    research = _get_research_client()
    runner = PipelineRunner(llm, research_client=research)

    try:
        result = asyncio.run(runner.run(state))
        console.print(f"\n[green]✓ Pipeline completed.[/green]")
        console.print(f"Output in: output/{result.run_id}/")
    except Exception as exc:
        console.print(f"\n[red]Pipeline failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command(name="dry-run")
def dry_run(
    topic: str = typer.Option(..., help="The topic"),
    audience: str = typer.Option(..., help="Target audience"),
    tone: str = typer.Option(..., help="Writing tone"),
    words: int = typer.Option(900, help="Target word count"),
    sources: list[str] = typer.Option([], help="URLs to include as sources"),
    cluster_angle: str = typer.Option(None, "--cluster-angle", help="Content cluster angle"),
) -> None:
    """Print all 6 prompts and exit without calling the API."""
    from src.io.input_parser import parse_run_args
    from src.llm.prompts import ALL_PROMPT_FUNCTIONS
    from src.pipeline.state import AGENT_ORDER, AgentStatus

    state = parse_run_args(
        topic=topic,
        audience=audience,
        tone=tone,
        words=words,
        sources=list(sources),
        cluster_angle=cluster_angle,
    )

    console.print(Panel("[bold]Dry Run — No API calls will be made[/bold]", title="Dry Run"))
    console.print(f"Topic: {topic}\nAudience: {audience}\nTone: {tone}\nWords: {words}\n")

    # For dry-run, we fill in placeholder outputs so later agents can render prompts
    placeholder = "(placeholder — would be filled by previous agent)"
    for agent_id in AGENT_ORDER:
        fn = ALL_PROMPT_FUNCTIONS[agent_id]
        prompt = fn(state)
        console.print(Panel(prompt, title=f"[bold cyan]{agent_id} Prompt[/bold cyan]"))
        # Set placeholder output so subsequent agents can build their prompts
        state.agents[agent_id].output = placeholder
        state.agents[agent_id].status = AgentStatus.COMPLETED

    console.print("\n[green]✓ Dry run complete.[/green] 6 prompts printed. No API calls made.")


if __name__ == "__main__":
    app()
