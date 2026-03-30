"""CLI commands: export, config, delete, deep, inspect, feishu, select-env."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from nanoresearch.config import ExecutionProfile
from nanoresearch.pipeline.unified_orchestrator import UnifiedPipelineOrchestrator
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage

from nanoresearch.cli import (
    app,
    console,
    _DEFAULT_ROOT,
    _setup_logging,
    _load_config_safe,
    _load_workspace_safe,
    _cli_progress,
    _print_result,
    _run_deep_pipeline,
)


@app.command()
def export(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace directory"),
    output: Path = typer.Option(None, "--output", "-o", help="Output directory (default: current dir)"),
) -> None:
    """Export a completed session to a clean output folder."""
    ws = _load_workspace_safe(workspace)
    if ws.manifest.current_stage != PipelineStage.DONE:
        console.print(f"[yellow]Warning:[/yellow] Pipeline status is {ws.manifest.current_stage.value}, not DONE")

    try:
        export_path = ws.export(output)
        console.print(f"[green]Exported to:[/green] {export_path}")
    except RuntimeError as exc:
        console.print(f"[red]Export failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command("config")
def show_config(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Show the current configuration (API keys are masked)."""
    config = _load_config_safe(config_path)
    snapshot = config.snapshot()

    # Mask the base_url partially
    base_url = config.base_url
    if len(base_url) > 20:
        base_url = base_url[:20] + "..."

    table = Table(title="Configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Base URL", base_url)
    table.add_row("API Key", "****" + config.api_key[-4:] if len(config.api_key) > 4 else "***")
    table.add_row("Global Timeout", f"{config.timeout}s")
    table.add_row("Max Retries", str(config.max_retries))
    table.add_row("Template Format", config.template_format)
    table.add_row("Execution Profile", config.execution_profile.value)
    table.add_row("Writing Mode", config.writing_mode.value)
    table.add_row("Auto Create Env", str(config.auto_create_env))
    table.add_row("Auto Download Resources", str(config.auto_download_resources))

    console.print(table)

    # Per-stage models
    stage_table = Table(title="Per-Stage Models")
    stage_table.add_column("Stage", style="bold")
    stage_table.add_column("Model")
    stage_table.add_column("Temperature")
    stage_table.add_column("Max Tokens")

    for stage_name in ["ideation", "planning", "experiment", "writing",
                       "code_gen", "figure_prompt", "figure_code", "figure_gen",
                       "evidence_extraction", "review"]:
        sc = config.for_stage(stage_name)
        stage_table.add_row(
            stage_name,
            sc.model,
            str(sc.temperature) if sc.temperature is not None else "None",
            str(sc.max_tokens),
        )

    console.print(stage_table)


@app.command()
def delete(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    root: Path = typer.Option(_DEFAULT_ROOT, "--root", "-r"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a research session and its workspace."""
    import shutil

    ws_path = root / session_id
    if not ws_path.is_dir():
        console.print(f"[red]Session not found:[/red] {ws_path}")
        raise typer.Exit(1)

    # Show what will be deleted
    manifest_path = ws_path / "manifest.json"
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            console.print(f"  Topic: {data.get('topic', '?')}")
            console.print(f"  Stage: {data.get('current_stage', '?')}")
        except (json.JSONDecodeError, OSError):
            pass

    if not force:
        confirm = typer.confirm(f"Delete session {session_id} at {ws_path}?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    try:
        shutil.rmtree(ws_path)
        console.print(f"[green]Deleted:[/green] {ws_path}")
    except OSError as exc:
        console.print(f"[red]Delete failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def deep(
    topic: str = typer.Option(..., "--topic", "-t", help="Research topic"),
    format: str = typer.Option("neurips", "--format", "-f", help="Paper format"),
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config file"),
    profile: ExecutionProfile | None = typer.Option(
        None,
        "--profile",
        help="Unified execution profile",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    dev: bool = typer.Option(False, "--dev", help="Dev mode: skip experiment stages"),
) -> None:
    """Compatibility alias for the unified deep-backbone pipeline."""
    _setup_logging(verbose)

    console.print("[yellow]`deep` is now a compatibility alias of `run`.[/yellow]")
    config = _load_config_safe(config_path)
    config.template_format = format
    if profile is not None:
        config.execution_profile = profile
    if dev:
        for st in ["setup", "coding", "execution", "analysis"]:
            if st not in config.skip_stages:
                config.skip_stages.append(st)
        console.print("[bold #d29922]DEV mode:[/bold #d29922] skipping setup/coding/execution/analysis")

    workspace = Workspace.create(
        topic=topic,
        config_snapshot=config.snapshot(),
        pipeline_mode=PipelineMode.DEEP,
    )
    console.print(Panel(
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Mode:[/bold] Unified deep backbone\n"
        f"[bold]Profile:[/bold] {config.execution_profile.value}\n"
        f"[bold]Session:[/bold] {workspace.manifest.session_id}\n"
        f"[bold]Workspace:[/bold] {workspace.path}\n"
        f"[bold]Format:[/bold] {format}\n\n"
        f"Pipeline: IDEATION -> PLANNING -> SETUP -> CODING -> EXECUTION -> ANALYSIS -> FIGURE_GEN -> WRITING -> REVIEW",
        title="NanoResearch Deep Mode",
        border_style="magenta",
    ))

    orchestrator = UnifiedPipelineOrchestrator(
        workspace, config, progress_callback=_cli_progress,
    )
    try:
        result = asyncio.run(_run_deep_pipeline(orchestrator, topic))
        _print_result(result, workspace)
    except Exception as e:
        console.print(f"[red]Deep pipeline failed:[/red] {e}")
        console.print(f"[bold]Workspace:[/bold] {workspace.path}")
        raise typer.Exit(1)


@app.command()
def inspect(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace directory"),
    stage: str = typer.Option(None, "--stage", "-s", help="Stage to inspect (e.g., ideation, planning)"),
) -> None:
    """Inspect individual stage outputs from a session."""
    ws = _load_workspace_safe(workspace)

    file_map = {
        "ideation": "papers/ideation_output.json",
        "planning": "plans/experiment_blueprint.json",
        "experiment": "logs/experiment_output.json",
        "setup": "plans/setup_output.json",
        "coding": "plans/coding_output.json",
        "execution": "plans/execution_output.json",
        "analysis": "plans/analysis_output.json",
        "figure_gen": "drafts/figure_output.json",
        "writing": "drafts/paper_skeleton.json",
        "review": "drafts/review_output.json",
    }

    if stage:
        stage = stage.lower()
        if stage not in file_map:
            console.print(f"[red]Unknown stage:[/red] {stage}. Available: {list(file_map)}")
            raise typer.Exit(1)
        try:
            data = ws.read_json(file_map[stage])
            console.print_json(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        except FileNotFoundError:
            console.print(f"[yellow]No output found for stage '{stage}'[/yellow]")
    else:
        # Show overview of all available outputs
        console.print(f"[bold]Workspace:[/bold] {ws.path}\n")
        for name, path in file_map.items():
            exists = (ws.path / path).is_file()
            icon = "[green]exists[/green]" if exists else "[dim]missing[/dim]"
            console.print(f"  {name:12s} {icon}  ({path})")


@app.command()
def feishu(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Launch Feishu bot for triggering NanoResearch pipeline via Feishu messages.

    Requires a Feishu app with App ID/Secret configured via
    FEISHU_APP_ID/FEISHU_APP_SECRET env vars or ~/.nanoresearch/config.json.
    """
    _setup_logging(verbose)
    from nanoresearch.feishu_bot import main as feishu_main
    console.print(Panel(
        "[bold]NanoResearch \u98de\u4e66\u673a\u5668\u4eba[/bold]\n\n"
        "\u5728\u98de\u4e66\u4e2d\u7ed9\u673a\u5668\u4eba\u53d1\u6d88\u606f\u5373\u53ef\u542f\u52a8 pipeline\u3002\n"
        "\u652f\u6301\u7684\u547d\u4ee4\uff1a/run <\u4e3b\u9898>\u3001/status\u3001/list\u3001/stop\u3001/help\n"
        "\u6216\u76f4\u63a5\u53d1\u9001\u7814\u7a76\u4e3b\u9898\u3002\n\n"
        "\u6309 Ctrl+C \u505c\u6b62\u3002",
        title="Feishu Bot",
        border_style="blue",
    ))
    feishu_main()


@app.command("select-env")
def select_env(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Scan available Python environments and select one for experiments.

    Auto-detects conda envs, system pythons, and pyenv versions.
    Saves the selection to config.json so all future runs use it.
    """
    from nanoresearch.agents.runtime_env import discover_environments

    console.print("[bold]Scanning Python environments...[/bold]\n")
    envs = discover_environments()

    if not envs:
        console.print("[red]No Python environments found on this system.[/red]")
        raise typer.Exit(1)

    # Build table
    table = Table(title="Available Python Environments")
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Python", style="dim")
    table.add_column("Version", style="yellow")
    table.add_column("Key Packages", style="magenta")

    for i, env in enumerate(envs, 1):
        pkgs = ", ".join(env["packages"]) if env["packages"] else "-"
        table.add_row(str(i), env["name"], env["python"], env["version"], pkgs)

    console.print(table)
    console.print()

    # Prompt user to pick
    choice = typer.prompt(
        f"Select environment [1-{len(envs)}], or 0 to cancel",
        type=int,
        default=0,
    )
    if choice == 0 or choice < 1 or choice > len(envs):
        console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit()

    selected = envs[choice - 1]
    python_path = selected["python"]

    # Save to config.json
    cfg_path = config_path or Path.home() / ".nanoresearch" / "config.json"
    cfg_data: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    research = cfg_data.setdefault("research", {})
    research["experiment_python"] = python_path
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_data, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(Panel(
        f"[bold]Selected:[/bold] {selected['name']}\n"
        f"[bold]Python:[/bold]   {python_path}\n"
        f"[bold]Version:[/bold]  {selected['version']}\n"
        f"[bold]Packages:[/bold] {', '.join(selected['packages']) or 'none detected'}\n\n"
        f"[green]Saved to {cfg_path}[/green]",
        title="Environment Selected",
        border_style="green",
    ))


# ─── Skill management ───

@app.command("skills")
def skills_cmd(
    action: str = typer.Argument("list", help="Action: list, install, uninstall, info"),
    name: str = typer.Argument("", help="Skill name or GitHub URL"),
    tag: str = typer.Option(None, "--tag", "-t", help="Filter by tag"),
) -> None:
    """Manage NanoResearch skills (list, install, uninstall)."""
    from nanoresearch.skill_registry import SkillRegistry

    registry = SkillRegistry()
    registry.discover()

    if action == "list":
        skills = registry.list_skills(tag=tag)
        if not skills:
            console.print("[dim]No skills found.[/dim]")
            if tag:
                console.print(f"[dim](filtered by tag: {tag})[/dim]")
            return
        table = Table(title="NanoResearch Skills")
        table.add_column("Name", style="bold cyan")
        table.add_column("Source", width=8)
        table.add_column("Tags")
        table.add_column("Description")
        for s in skills:
            table.add_row(
                s.name,
                f"[green]{s.source}[/green]" if s.source == "builtin" else f"[yellow]{s.source}[/yellow]",
                ", ".join(s.tags) if s.tags else "[dim]--[/dim]",
                s.description[:60] if s.description else "[dim]--[/dim]",
            )
        console.print(table)

    elif action == "install":
        if not name:
            console.print("[red]Usage: nanoresearch skills install <github-url-or-path>[/red]")
            return
        if name.startswith(("http://", "https://", "github.com")):
            url = name if name.startswith("http") else f"https://{name}"
            console.print(f"Installing from [cyan]{url}[/cyan]...")
            result = registry.install_from_github(url)
        else:
            console.print(f"[red]Local install not yet supported. Use a GitHub URL.[/red]")
            return
        if result.get("success"):
            console.print(f"[green]Installed:[/green] {', '.join(result.get('installed', []))}")
        else:
            console.print(f"[red]Failed:[/red] {result.get('message', 'Unknown error')}")

    elif action == "uninstall":
        if not name:
            console.print("[red]Usage: nanoresearch skills uninstall <name>[/red]")
            return
        if registry.uninstall(name):
            console.print(f"[green]Uninstalled:[/green] {name}")
        else:
            console.print(f"[red]Not found or cannot uninstall:[/red] {name}")

    elif action == "info":
        if not name:
            console.print("[red]Usage: nanoresearch skills info <name>[/red]")
            return
        skill = registry.get_skill(name)
        if not skill:
            console.print(f"[red]Skill not found:[/red] {name}")
            return
        console.print(Panel(
            f"[bold]Name:[/bold] {skill.name}\n"
            f"[bold]Version:[/bold] {skill.version}\n"
            f"[bold]Source:[/bold] {skill.source}\n"
            f"[bold]Tags:[/bold] {', '.join(skill.tags) or 'none'}\n"
            f"[bold]Path:[/bold] {skill.path}\n\n"
            f"{skill.description}",
            title=f"Skill: {skill.name}",
            border_style="cyan",
        ))
