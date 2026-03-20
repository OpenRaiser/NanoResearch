"""CLI commands for paper editing, cleanup, and health check."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from nanoresearch.cli import (
    app,
    console,
    _load_config_safe,
    _load_workspace_safe,
)
from nanoresearch.pipeline.workspace import Workspace


@app.command("paper")
def paper_edit(
    workspace: Path = typer.Option(
        ..., "--workspace", "-w",
        help="Path to research workspace directory",
    ),
    config_path: Path = typer.Option(None, "--config", "-c"),
    instruction: str = typer.Option(
        None, "--instruction", "-i",
        help="One-shot instruction (non-interactive mode)",
    ),
) -> None:
    """Interactive paper editor --- describe changes in natural language.

    Reads the workspace, classifies your instruction, and dispatches to
    the right module (WritingAgent, ReviewAgent, FigureAgent, LaTeX fixer).
    Auto-backs up before every change.

    Commands:
      undo        --- rollback the last change
      rollback    --- pick a snapshot to restore
      review      --- run full review cycle
      recompile   --- recompile PDF from current tex
      status      --- show paper info (score, sections, figures)
      history     --- show edit history
      exit        --- quit
    """
    from nanoresearch.agents.paper_editor import PaperEditor

    ws = _load_workspace_safe(workspace)
    config = _load_config_safe(config_path)
    editor = PaperEditor(
        ws, config,
        log_fn=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
    )

    # Check paper.tex exists
    tex_path = ws.path / "drafts" / "paper.tex"
    if not tex_path.exists():
        console.print(f"[red]No paper.tex found in {ws.path / 'drafts'}[/red]")
        console.print("Run the pipeline first to generate a paper.")
        raise typer.Exit(1)

    # One-shot mode
    if instruction:
        result = asyncio.run(_paper_apply(editor, instruction))
        _print_paper_result(result)
        return

    # Paper status
    _show_paper_status(ws)

    console.print(Panel(
        "[bold]Describe what you want to change.[/bold]\n\n"
        "Examples:\n"
        "  Introduction\u592a\u957f\u4e86\uff0c\u7cbe\u7b80\u52301\u9875\n"
        "  Figure 3\u7684\u6570\u636e\u4e0d\u5bf9\uff0c\u91cd\u65b0\u751f\u6210\n"
        "  \u52a0\u4e00\u6bb5Related Work\u5173\u4e8exxx\u7684\u8ba8\u8bba\n"
        "  \u8dd1\u4e00\u6b21\u5b8c\u6574Review\n\n"
        "Commands: [cyan]undo[/cyan] | [cyan]rollback[/cyan] | [cyan]diff[/cyan] | "
        "[cyan]status[/cyan] | [cyan]history[/cyan] | [cyan]exit[/cyan]",
        title="NanoResearch Paper Editor",
        border_style="blue",
    ))

    while True:
        try:
            user_input = console.input("\n[bold green]paper> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("exit", "quit", "q"):
            console.print("[yellow]Exiting.[/yellow]")
            break

        if cmd == "undo":
            snap = editor.undo()
            if snap:
                console.print(f"[green]Rolled back to: {snap}[/green]")
            else:
                console.print("[yellow]No snapshots available.[/yellow]")
            continue

        if cmd == "rollback":
            snaps = editor.snapshot_mgr.list_snapshots()
            if not snaps:
                console.print("[yellow]No snapshots.[/yellow]")
                continue
            table = Table(title="Paper Snapshots")
            table.add_column("#", style="bold cyan", justify="right")
            table.add_column("ID", style="green")
            table.add_column("Time", style="yellow")
            for i, s in enumerate(snaps, 1):
                table.add_row(str(i), s["id"], s["time"])
            console.print(table)
            try:
                choice = int(console.input(f"Restore [1-{len(snaps)}], 0 to cancel: "))
            except (ValueError, EOFError, KeyboardInterrupt):
                continue
            if 1 <= choice <= len(snaps):
                sid = snaps[choice - 1]["id"]
                if editor.rollback_to(sid):
                    console.print(f"[green]Restored to {sid}[/green]")
                else:
                    console.print("[red]Rollback failed[/red]")
            continue

        if cmd == "status":
            _show_paper_status(ws)
            continue

        if cmd == "history":
            if not editor._history:
                console.print("[dim]No edits yet.[/dim]")
            else:
                for i, h in enumerate(editor._history, 1):
                    tools = h.get("tools_called", [])
                    tool_summary = f"{len(tools)} tools" if tools else "no tools"
                    console.print(
                        f"  {i}. [cyan]{h['instruction'][:60]}[/cyan] "
                        f"({tool_summary}, backup: {h['snapshot_id']})"
                    )
            continue

        if cmd == "diff" or cmd.startswith("diff "):
            _show_paper_diff(editor)
            continue

        # Normal instruction -> ReAct agent loop
        console.print("[dim]Working on it...[/dim]")
        result = asyncio.run(_paper_apply(editor, user_input))
        _print_paper_result(result)


async def _paper_apply(editor: "PaperEditor", instruction: str) -> dict:  # noqa: F821
    return await editor.apply_instruction(instruction)


def _print_paper_result(result: dict) -> None:
    """Pretty-print paper edit result."""
    from collections import Counter

    snap = result.get("snapshot_id", "")
    tools = result.get("tools_called", [])
    summary = result.get("summary", "")

    console.print(f"\n  [dim]backup: {snap} | tools used: {len(tools)}[/dim]")

    if tools:
        # Show unique tools called
        counts = Counter(tools)
        tool_str = ", ".join(f"{name}({c})" if c > 1 else name for name, c in counts.items())
        console.print(f"  [cyan]Tools:[/cyan] {tool_str}")

    if summary:
        console.print()
        console.print(Panel(summary, title="Agent Response", border_style="green"))
    else:
        console.print("  [yellow]No response from agent.[/yellow]")

    console.print("  [dim]Type 'undo' to rollback all changes.[/dim]")


def _show_paper_diff(editor: Any) -> None:
    """Compare latest snapshot with current state."""
    diff = editor.diff_snapshots()
    if "error" in diff:
        console.print(f"  [yellow]{diff['error']}[/yellow]")
        return

    console.print(f"\n  [bold]Comparing:[/bold] {diff['snap_a']} -> {diff['snap_b']}")

    total = diff.get("total_chars", {})
    delta = total.get("after", 0) - total.get("before", 0)
    console.print(f"  [bold]Total chars:[/bold] {total.get('before', '?')} -> {total.get('after', '?')} ({delta:+d})")

    sections = diff.get("sections", [])
    if sections:
        table = Table(title="Section Changes")
        table.add_column("Section", style="bold")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Change", justify="right")
        for s in sections:
            d = s["delta"]
            color = "green" if d < 0 else ("red" if d > 0 else "dim")
            table.add_row(
                s["section"][:35], str(s["before"]), str(s["after"]),
                f"[{color}]{d:+d}[/{color}]", s["change"],
            )
        console.print(table)

    figs = diff.get("figures", {})
    cites = diff.get("citations", {})
    console.print(
        f"  [bold]Figures:[/bold] {figs.get('before', '?')} -> {figs.get('after', '?')}  |  "
        f"[bold]Citations:[/bold] {cites.get('before', '?')} -> {cites.get('after', '?')}"
    )


def _show_paper_status(ws: Workspace) -> None:
    """Show paper overview: sections, figures, review score."""
    import re

    tex_path = ws.path / "drafts" / "paper.tex"
    if not tex_path.exists():
        console.print("[yellow]No paper.tex found.[/yellow]")
        return

    tex = tex_path.read_text(encoding="utf-8", errors="replace")

    # Extract title
    title_m = re.search(r"\\title\{([^}]+)\}", tex)
    title = title_m.group(1).strip() if title_m else "?"

    # Extract sections
    sections = re.findall(r"\\section\{([^}]+)\}", tex)

    # Count figures
    fig_count = len(re.findall(r"\\begin\{figure\}", tex))

    # Count citations
    cite_count = len(re.findall(r"\\cite\{", tex))

    # Review score
    score = "?"
    try:
        review = ws.read_json("drafts/review_output.json")
        score = review.get("overall_score", "?")
    except FileNotFoundError:
        pass

    # Figures on disk
    fig_dir = ws.path / "figures"
    fig_files = list(fig_dir.glob("*.png")) + list(fig_dir.glob("*.pdf")) if fig_dir.is_dir() else []

    console.print(Panel(
        f"[bold]Title:[/bold] {title}\n"
        f"[bold]Sections:[/bold] {', '.join(sections) or 'none'}\n"
        f"[bold]Inline figures:[/bold] {fig_count}  |  "
        f"[bold]Figure files:[/bold] {len(fig_files)}\n"
        f"[bold]Citations:[/bold] {cite_count}\n"
        f"[bold]Review score:[/bold] {score}/10",
        title="Paper Status",
        border_style="cyan",
    ))


@app.command("cleanup-envs")
def cleanup_envs(
    dry_run: bool = typer.Option(False, "--dry-run", help="Only list envs, don't remove"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt"),
) -> None:
    """Remove per-session conda environments (nanoresearch_*).

    Lists all nanoresearch_* conda environments and optionally removes them
    to reclaim disk space.
    """
    from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager

    envs = RuntimeEnvironmentManager.list_nanoresearch_conda_envs()
    if not envs:
        console.print("[green]No nanoresearch_* conda environments found.[/green]")
        raise typer.Exit()

    table = Table(title="Per-session conda environments")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")
    for env in envs:
        table.add_row(env["name"], env["path"])
    console.print(table)
    console.print(f"\n[bold]{len(envs)}[/bold] environment(s) found.")

    if dry_run:
        console.print("[yellow]Dry-run mode --- no environments removed.[/yellow]")
        raise typer.Exit()

    if not yes:
        confirm = typer.confirm(f"Remove all {len(envs)} environment(s)?")
        if not confirm:
            raise typer.Exit()

    removed = 0
    for env in envs:
        console.print(f"  Removing {env['name']} ...", end=" ")
        ok = RuntimeEnvironmentManager.remove_conda_env(env["name"])
        if ok:
            console.print("[green]OK[/green]")
            removed += 1
        else:
            console.print("[red]FAILED[/red]")
    console.print(f"\n[bold]{removed}/{len(envs)}[/bold] environments removed.")


@app.command("health")
def health_check(
    config_path: Path = typer.Option(None, "--config", "-c"),
) -> None:
    """System health check: API, models, environments, disk usage."""
    import shutil
    from nanoresearch.agents.runtime_env import discover_environments

    checks: list[tuple[str, str, str]] = []  # (label, status_icon, detail)

    # 1. Config loadable
    try:
        config = _load_config_safe(config_path)
        checks.append(("Config", "[green]OK[/green]", "Loaded successfully"))
    except SystemExit:
        checks.append(("Config", "[red]FAIL[/red]", "Cannot load config"))
        _print_health(checks)
        raise typer.Exit(1)

    # 2. API key
    api_key = config.api_key
    if api_key and len(api_key) > 4:
        checks.append(("API Key", "[green]OK[/green]", f"****{api_key[-4:]}"))
    else:
        checks.append(("API Key", "[red]MISS[/red]", "Set NANORESEARCH_API_KEY or api_key in config.json"))

    # 3. Base URL reachable
    try:
        import urllib.request
        req = urllib.request.Request(config.base_url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
        checks.append(("Base URL", "[green]OK[/green]", config.base_url[:50]))
    except Exception as exc:
        # Many LLM APIs don't support HEAD, just check TCP
        import socket
        from urllib.parse import urlparse
        parsed = urlparse(config.base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            checks.append(("Base URL", "[green]OK[/green]", f"{host}:{port} reachable"))
        except Exception:
            checks.append(("Base URL", "[red]FAIL[/red]", f"Cannot reach {config.base_url[:40]}"))

    # 4. S2 / OpenAlex keys
    s2_key = os.environ.get("S2_API_KEY", "")
    checks.append((
        "S2 API Key",
        "[green]OK[/green]" if s2_key else "[yellow]MISS[/yellow]",
        "Configured" if s2_key else "Optional --- apply at semanticscholar.org",
    ))
    oa_key = os.environ.get("OPENALEX_API_KEY", "")
    checks.append((
        "OpenAlex Key",
        "[green]OK[/green]" if oa_key else "[yellow]MISS[/yellow]",
        "Configured" if oa_key else "Optional --- speeds up paper search",
    ))

    # 5. Python environments
    envs = discover_environments()
    user_python = (config.experiment_python or "").strip()
    if user_python:
        checks.append(("Experiment Python", "[green]SET[/green]", user_python))
    elif envs:
        checks.append(("Experiment Python", "[yellow]AUTO[/yellow]", f"{len(envs)} envs found, run select-env to pick"))
    else:
        checks.append(("Experiment Python", "[red]NONE[/red]", "No Python environments detected"))

    # 6. LaTeX compiler
    latex = config.latex_compiler if hasattr(config, "latex_compiler") else ""
    if latex and Path(latex).exists():
        checks.append(("LaTeX Compiler", "[green]OK[/green]", str(latex)))
    elif shutil.which("tectonic"):
        checks.append(("LaTeX Compiler", "[green]OK[/green]", "tectonic (in PATH)"))
    elif shutil.which("pdflatex"):
        checks.append(("LaTeX Compiler", "[green]OK[/green]", "pdflatex (in PATH)"))
    else:
        checks.append(("LaTeX Compiler", "[yellow]MISS[/yellow]", "Install tectonic or pdflatex for PDF output"))

    # 7. GPU
    try:
        import subprocess as _sp
        r = _sp.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            checks.append(("GPU", "[green]OK[/green]", r.stdout.strip().split("\n")[0]))
        else:
            checks.append(("GPU", "[yellow]NONE[/yellow]", "No GPU detected"))
    except Exception:
        checks.append(("GPU", "[yellow]NONE[/yellow]", "nvidia-smi not found"))

    # 8. Disk usage
    ws_root = Path.home() / ".nanobot" / "workspace" / "research"
    if ws_root.is_dir():
        total_size = sum(f.stat().st_size for f in ws_root.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        size_str = f"{size_mb:.0f} MB" if size_mb < 1024 else f"{size_mb / 1024:.1f} GB"
        session_count = sum(1 for d in ws_root.iterdir() if (d / "manifest.json").is_file())
        checks.append(("Workspace", "[green]OK[/green]", f"{session_count} sessions, {size_str}"))
    else:
        checks.append(("Workspace", "[dim]EMPTY[/dim]", "No sessions yet"))

    # 9. Models
    models_info = []
    for stage_name in ["ideation", "writing", "review"]:
        sc = config.for_stage(stage_name)
        models_info.append(f"{stage_name}: {sc.model}")
    checks.append(("Models", "[green]OK[/green]", " | ".join(models_info)))

    _print_health(checks)


def _print_health(checks: list[tuple[str, str, str]]) -> None:
    """Print health check results as a table."""
    table = Table(title="NanoResearch Health Check")
    table.add_column("Component", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Detail")
    for label, icon, detail in checks:
        table.add_row(label, icon, detail)
    console.print(table)
