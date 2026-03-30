"""CLI entry point for NanoResearch."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path

# Fix Windows encoding: force UTF-8 for stdout/stderr to prevent
# UnicodeEncodeError when Rich prints non-ASCII characters (e.g. ö, é)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace"
            )

import time

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nanoresearch import __version__
from nanoresearch.config import ExecutionProfile, ResearchConfig
from nanoresearch.pipeline.orchestrator import PipelineOrchestrator
from nanoresearch.pipeline.unified_orchestrator import UnifiedPipelineOrchestrator
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PaperMode, PipelineMode, PipelineStage

app = typer.Typer(
    name="nanoresearch",
    help="Minimal AI-driven research engine: idea → paper draft",
    add_completion=False,
)
console = Console()

_DEFAULT_ROOT = Path.home() / ".nanoresearch" / "workspace" / "research"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"nanoresearch v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """NanoResearch — AI-powered research paper generation pipeline."""
    # Auto-create ~/.nanoresearch directory structure if it doesn't exist
    _ensure_nanoresearch_home()


def _ensure_nanoresearch_home() -> None:
    """Create ~/.nanoresearch and its subdirectories if they don't exist."""
    nanoresearch_home = Path.home() / ".nanoresearch"
    subdirs = ["workspace/research", "chat_memory", "cache/models", "cache/data"]

    nanoresearch_home.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (nanoresearch_home / subdir).mkdir(parents=True, exist_ok=True)


def _setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = []
    if log_file is not None:
        # File-only logging (used when Live UI is active to avoid terminal noise)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _load_config_safe(config_path: Path | None) -> ResearchConfig:
    """Load config with user-friendly error messages."""
    try:
        cfg = ResearchConfig.load(config_path)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    # Propagate optional third-party API keys from config.json → env vars
    _propagate_api_keys(config_path)
    return cfg


def _propagate_api_keys(config_path: Path | None) -> None:
    """Read optional API keys from config.json and set as env vars."""
    path = config_path or Path.home() / ".nanoresearch" / "config.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    research = data.get("research", {})
    key_map = {
        "openalex_api_key": "OPENALEX_API_KEY",
        "s2_api_key": "S2_API_KEY",
    }
    for json_key, env_key in key_map.items():
        val = research.get(json_key, "")
        if val and not os.environ.get(env_key):
            os.environ[env_key] = str(val)


def _load_workspace_safe(path: Path) -> Workspace:
    """Load workspace with user-friendly error messages."""
    try:
        return Workspace.load(path)
    except FileNotFoundError:
        console.print(f"[red]Workspace not found:[/red] {path}")
        raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"[red]Workspace error:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def run(
    topic: str = typer.Option(..., "--topic", "-t", help="Research topic"),
    format: str = typer.Option(None, "--format", "-f", help="Paper format (auto-discovered from templates directory)"),
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config file"),
    profile: ExecutionProfile | None = typer.Option(
        None,
        "--profile",
        help="Unified execution profile",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and exit without running"),
    dev: bool = typer.Option(False, "--dev", help="Dev mode: skip experiment stages (setup/coding/execution/analysis)"),
    tui: bool = typer.Option(False, "--tui", help="Use full-screen TUI instead of inline progress panel"),
) -> None:
    """Run the unified research pipeline from topic to paper draft."""
    _setup_logging(verbose)

    # Validate topic
    if not topic or not topic.strip():
        console.print("[red]Error:[/red] --topic must be a non-empty string")
        raise typer.Exit(1)
    topic = topic.strip()

    # Parse paper_mode from topic prefix (e.g. "survey:short: LLM Reasoning")
    paper_mode = PaperMode.from_string(topic)
    if paper_mode.is_survey:
        # Strip the prefix from topic to get clean topic string
        for prefix in ["survey:short:", "survey:standard:", "survey:long:", "original:"]:
            if topic.lower().startswith(prefix):
                topic = topic[len(prefix):].strip()
                break

    config = _load_config_safe(config_path)
    if profile is not None:
        config.execution_profile = profile

    # Only override template_format if user explicitly passed --format
    if format is not None:
        from nanoresearch.templates import get_available_formats
        valid_formats = get_available_formats()
        if format not in valid_formats:
            console.print(f"[red]Error:[/red] --format must be one of {valid_formats}")
            raise typer.Exit(1)
        config.template_format = format

    # --dev: skip experiment stages, go straight from planning to figures/writing
    _DEV_SKIP = ["setup", "coding", "execution", "analysis"]
    if dev:
        for st in _DEV_SKIP:
            if st not in config.skip_stages:
                config.skip_stages.append(st)
        console.print("[bold #d29922]DEV mode:[/bold #d29922] skipping setup/coding/execution/analysis")

    if dry_run:
        console.print(Panel(
            f"[bold]Topic:[/bold] {topic}\n"
            f"[bold]Format:[/bold] {format}\n"
            f"[bold]Base URL:[/bold] {config.base_url}\n"
            f"[bold]Ideation model:[/bold] {config.ideation.model}\n"
            f"[bold]Writing model:[/bold] {config.writing.model}\n"
            f"[bold]Execution profile:[/bold] {config.execution_profile.value}\n"
            f"[bold]Writing mode:[/bold] {config.writing_mode.value}\n"
            f"[bold]Max retries:[/bold] {config.max_retries}\n"
            f"[bold]Skip stages:[/bold] {config.skip_stages}\n"
            f"\n[green]Configuration is valid.[/green]",
            title="Dry Run",
            border_style="cyan",
        ))
        return

    workspace = Workspace.create(
        topic=topic,
        config_snapshot=config.snapshot(),
        pipeline_mode=PipelineMode.DEEP,
        paper_mode=paper_mode,
    )
    if console.is_terminal:
        console.print(_build_welcome_banner(
            topic, workspace.manifest.session_id, str(workspace.path), config,
        ))
    else:
        console.print(Panel(
            f"[bold]Topic:[/bold] {topic}\n"
            f"[bold]Pipeline:[/bold] Unified deep backbone\n"
            f"[bold]Profile:[/bold] {config.execution_profile.value}\n"
            f"[bold]Session:[/bold] {workspace.manifest.session_id}\n"
            f"[bold]Workspace:[/bold] {workspace.path}",
            title="NanoResearch",
            border_style="blue",
        ))

    # Interactive env selection MUST happen before Live display starts
    _ensure_env_selected(config)

    _run_with_live_progress(
        lambda cb: UnifiedPipelineOrchestrator(workspace, config, progress_callback=cb),
        _run_deep_pipeline, topic, workspace, use_tui=tui,
    )


def _cli_progress(stage: str, status: str, message: str) -> None:
    """Fallback progress callback for non-TTY or piped output."""
    icons = {
        "started": "[cyan]>>>[/cyan]",
        "completed": "[green] OK[/green]",
        "skipped": "[dim] --[/dim]",
        "retrying": "[yellow] !![/yellow]",
        "failed": "[red]ERR[/red]",
    }
    if status != "substep":
        console.print(f"  {icons.get(status, '   ')} {message}")


# Deep pipeline stages in order
_DEEP_STAGES = [
    "ideation", "planning", "setup", "coding", "execution",
    "analysis", "figure_gen", "writing", "review",
]

# ─── Visual constants (ASCII-safe only — no emoji to avoid terminal width bugs) ───
_SPINNERS = ("|", "/", "-", "\\")


def _build_welcome_banner(topic: str, session_id: str, workspace: str, config: ResearchConfig) -> Panel:
    """Build a styled welcome banner."""
    header = Text.from_markup(
        "  [bold #58a6ff]Nano[/bold #58a6ff][bold #79c0ff]Research[/bold #79c0ff]"
        f"  [dim #484f58]v{__version__}[/dim #484f58]"
    )

    info_table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    info_table.add_column("key", style="bold #6e7681", width=10, justify="right")
    info_table.add_column("val")
    info_table.add_row("Topic", f"[bold white]{topic}[/bold white]")
    info_table.add_row("Model", f"[#c9d1d9]{config.ideation.model}[/#c9d1d9]  [dim]|[/dim]  [#c9d1d9]{config.execution_profile.value}[/#c9d1d9]")
    info_table.add_row("Session", f"[#58a6ff]{session_id}[/#58a6ff]")
    info_table.add_row("Dir", f"[dim]{workspace}[/dim]")

    return Panel(
        Group(header, Text(), info_table),
        border_style="#30363d",
        box=box.ROUNDED,
        padding=(1, 3),
    )


class LiveProgressDisplay:
    """Rich Live progress display — per-stage substep tracking."""

    _STAGE_META = {
        "ideation": ("Ideation", [
            "Generating queries", "Searching literature", "Enriching citations",
            "Downloading papers", "Analyzing gaps", "Extracting evidence",
        ]),
        "planning": ("Planning", [
            "Designing blueprint", "Defining metrics", "Planning ablations",
        ]),
        "setup": ("Setup", [
            "Preparing environment", "Installing dependencies",
        ]),
        "coding": ("Coding", [
            "Generating code files", "Validating imports",
        ]),
        "execution": ("Execution", [
            "Running experiment", "Collecting results",
        ]),
        "analysis": ("Analysis", [
            "Computing metrics", "Building comparison", "Analyzing ablations",
        ]),
        "figure_gen": ("Figures", [
            "Planning figures", "Generating figures",
        ]),
        "writing": ("Writing", [
            "Building grounding", "Writing: Introduction", "Writing: Related Work",
            "Writing: Method", "Writing: Experiments", "Writing: Conclusion",
            "Rendering LaTeX", "Compiling PDF",
        ]),
        "review": ("Review", [
            "Reviewing paper", "Applying revisions", "Re-compiling PDF",
        ]),
    }

    def __init__(self, stages: list[str] | None = None) -> None:
        self._stages = stages or _DEEP_STAGES
        self._states: dict[str, dict] = {
            s: {
                "status": "pending", "start": 0.0, "elapsed": 0.0,
                "substep": "", "substep_idx": 0,
            }
            for s in self._stages
        }
        self._live: Live | None = None
        self._pipeline_start = time.monotonic()
        self._tick = 0

    def __call__(self, stage: str, status: str, message: str) -> None:
        if stage not in self._states:
            self._states[stage] = {
                "status": "pending", "start": 0.0, "elapsed": 0.0,
                "substep": "", "substep_idx": 0,
            }
        s = self._states[stage]
        if status == "substep":
            s["substep"] = message
            meta = self._STAGE_META.get(stage)
            if meta:
                known_steps = meta[1]
                for i, step_name in enumerate(known_steps):
                    if step_name.lower() in message.lower() or message.lower() in step_name.lower():
                        s["substep_idx"] = i + 1
                        break
                else:
                    s["substep_idx"] = min(s["substep_idx"] + 1, len(known_steps))
        elif status == "started":
            s["status"] = "running"
            s["start"] = time.monotonic()
            s["substep"] = ""
            s["substep_idx"] = 0
        elif status == "completed":
            s["status"] = "completed"
            s["elapsed"] = time.monotonic() - s["start"] if s["start"] else 0.0
            s["substep"] = ""
        elif status == "failed":
            s["status"] = "failed"
            s["elapsed"] = time.monotonic() - s["start"] if s["start"] else 0.0
        elif status == "skipped":
            s["status"] = "skipped"
        elif status == "retrying":
            s["status"] = "retrying"
            s["substep"] = message
        # Don't manually refresh — rely on Live auto-refresh to avoid
        # race conditions with the refresh thread on Windows.

    def __rich_console__(self, rconsole, options):
        self._tick += 1
        yield self._render()

    def _spinner(self) -> str:
        return _SPINNERS[self._tick % len(_SPINNERS)]

    def _fmt_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"

    def _render(self) -> Panel:
        table = Table(
            box=None, show_header=True, expand=True,
            padding=(0, 1), pad_edge=False,
            header_style="bold #6e7681",
        )
        table.add_column("#", width=4, justify="right")
        table.add_column("Stage", width=14)
        table.add_column("Status", width=10, justify="center")
        table.add_column("Time", width=7, justify="right")
        table.add_column("Detail", ratio=1, no_wrap=True, overflow="ellipsis")

        completed_count = 0
        total = len(self._stages)

        for idx, stage in enumerate(self._stages, 1):
            s = self._states.get(stage, {})
            st = s.get("status", "pending")
            meta = self._STAGE_META.get(stage, (stage, []))
            label = meta[0]
            known_steps = meta[1]
            num = f"{idx}."

            if st == "running":
                spinner = self._spinner()
                elapsed = time.monotonic() - s.get("start", time.monotonic())
                substep = s.get("substep", "")
                substep_idx = s.get("substep_idx", 0)
                if known_steps:
                    filled = int(substep_idx / len(known_steps) * 8)
                    mini = "#" * filled + "-" * (8 - filled)
                    detail_str = substep if substep else known_steps[min(substep_idx, len(known_steps) - 1)]
                    detail = f"[#58a6ff][{mini}][/#58a6ff] [#c9d1d9]{detail_str}[/#c9d1d9]"
                else:
                    detail = f"[#c9d1d9]{substep}[/#c9d1d9]" if substep else ""
                table.add_row(
                    f"[bold #58a6ff]{num}[/bold #58a6ff]",
                    f"[bold #58a6ff]{label}[/bold #58a6ff]",
                    f"[bold #58a6ff]{spinner} RUN[/bold #58a6ff]",
                    f"[#58a6ff]{self._fmt_time(elapsed)}[/#58a6ff]",
                    detail,
                )

            elif st == "completed":
                completed_count += 1
                table.add_row(
                    f"[#3fb950]{num}[/#3fb950]",
                    f"[#3fb950]{label}[/#3fb950]",
                    "[#3fb950]+ DONE[/#3fb950]",
                    f"[#3fb950]{self._fmt_time(s.get('elapsed', 0))}[/#3fb950]",
                    "",
                )

            elif st == "failed":
                table.add_row(
                    f"[#f85149]{num}[/#f85149]",
                    f"[#f85149]{label}[/#f85149]",
                    "[bold #f85149]x FAIL[/bold #f85149]",
                    f"[#f85149]{self._fmt_time(s.get('elapsed', 0))}[/#f85149]",
                    f"[#f85149]{s.get('substep', '')}[/#f85149]",
                )

            elif st == "retrying":
                spinner = self._spinner()
                elapsed = time.monotonic() - s.get("start", time.monotonic())
                table.add_row(
                    f"[#d29922]{num}[/#d29922]",
                    f"[#d29922]{label}[/#d29922]",
                    f"[#d29922]{spinner} RETRY[/#d29922]",
                    f"[#d29922]{self._fmt_time(elapsed)}[/#d29922]",
                    f"[#d29922]{s.get('substep', '')}[/#d29922]",
                )

            elif st == "skipped":
                completed_count += 1
                table.add_row(
                    f"[#484f58]{num}[/#484f58]",
                    f"[#484f58]{label}[/#484f58]",
                    "[#484f58]- SKIP[/#484f58]",
                    "[#484f58]--[/#484f58]",
                    "",
                )

            else:  # pending
                table.add_row(
                    f"[#6e7681]{num}[/#6e7681]",
                    f"[#8b949e]{label}[/#8b949e]",
                    "[#484f58]...[/#484f58]",
                    "",
                    "",
                )

        # ── Overall progress bar ──
        total_elapsed = time.monotonic() - self._pipeline_start
        pct = completed_count / total * 100 if total else 0
        bar_w = 30
        filled = int(pct / 100 * bar_w)
        bar_str = "[#3fb950]" + "=" * filled + "[/#3fb950]" + "[#21262d]" + "-" * (bar_w - filled) + "[/#21262d]"

        footer = Text.from_markup(
            f"  [{bar_str}]  "
            f"[bold white]{pct:.0f}%[/bold white]  "
            f"[#8b949e]{completed_count}/{total} stages[/#8b949e] [#484f58]|[/#484f58] "
            f"[#8b949e]{self._fmt_time(total_elapsed)}[/#8b949e]"
        )

        return Panel(
            Group(table, Text(), footer),
            title="[bold #58a6ff]NanoResearch Pipeline[/bold #58a6ff]",
            subtitle="[#484f58]Ctrl+C to stop[/#484f58]",
            border_style="#30363d",
            box=box.ROUNDED,
            padding=(1, 2),
        )


_VALID_STAGES = [
    "ideation", "planning", "setup", "coding", "execution",
    "analysis", "figure_gen", "writing", "review",
]


@app.command()
def resume(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace directory"),
    config_path: Path = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    tui: bool = typer.Option(False, "--tui", help="Use full-screen TUI"),
    from_stage: str = typer.Option(None, "--from", "-F", help=f"Restart from a specific stage ({', '.join(_VALID_STAGES)})"),
    dev: bool = typer.Option(False, "--dev", help="Dev mode: skip experiment stages"),
    skip: str = typer.Option(None, "--skip", "-s", help="Comma-separated stages to skip (e.g. 'setup,coding')"),
) -> None:
    """Resume a pipeline from its last checkpoint."""
    _setup_logging(verbose)

    ws = _load_workspace_safe(workspace)
    manifest = ws.manifest
    config = _load_config_safe(config_path)

    # --dev: skip experiment stages
    if dev:
        for st in ["setup", "coding", "execution", "analysis"]:
            if st not in config.skip_stages:
                config.skip_stages.append(st)

    # --skip: skip specific stages
    if skip:
        for st in skip.split(","):
            st = st.strip().lower()
            if st and st not in config.skip_stages:
                if st not in _VALID_STAGES:
                    console.print(f"[red]Unknown stage to skip:[/red] {st}")
                    console.print(f"[dim]Valid stages: {', '.join(_VALID_STAGES)}[/dim]")
                    raise typer.Exit(1)
                config.skip_stages.append(st)

    # Show detailed status table before resuming
    status_table = Table(box=box.SIMPLE, title="Session Status")
    status_table.add_column("Stage", width=14)
    status_table.add_column("Status", width=10)
    status_table.add_column("Info", ratio=1)
    status_colors = {
        "pending": "dim", "running": "yellow",
        "completed": "green", "failed": "red",
    }
    for stage_name, rec in manifest.stages.items():
        color = status_colors.get(rec.status, "white")
        skip_note = " [dim](will skip)[/dim]" if stage_name.lower() in config.skip_stages else ""
        status_table.add_row(
            stage_name,
            f"[{color}]{rec.status}[/{color}]",
            skip_note,
        )
    console.print(status_table)
    console.print(f"  [bold]Topic:[/bold] {manifest.topic}")
    console.print(f"  [bold]Session:[/bold] {manifest.session_id}")
    if config.skip_stages:
        console.print(f"  [bold #d29922]Skipping:[/bold #d29922] {', '.join(config.skip_stages)}")
    console.print()

    # --from: restart from a specific stage (reset it and all following to pending)
    if from_stage:
        from_stage = from_stage.strip().lower()
        if from_stage not in _VALID_STAGES:
            console.print(f"[red]Unknown stage:[/red] {from_stage}")
            console.print(f"[dim]Valid stages: {', '.join(_VALID_STAGES)}[/dim]")
            raise typer.Exit(1)
        found = False
        for stage_name, rec in manifest.stages.items():
            if stage_name.lower() == from_stage:
                found = True
            if found:
                rec.status = "pending"
        target_stage = PipelineStage(from_stage.upper())
        manifest.current_stage = target_stage
        ws.update_manifest(current_stage=manifest.current_stage, stages=manifest.stages)
        console.print(f"  [cyan]Restarting from stage:[/cyan] [bold]{from_stage.upper()}[/bold]")

    elif manifest.current_stage in (PipelineStage.DONE, PipelineStage.FAILED):
        if manifest.current_stage == PipelineStage.FAILED:
            found_failed = False
            for stage_name, rec in manifest.stages.items():
                if rec.status == "failed":
                    rec.status = "pending"
                    manifest.current_stage = rec.stage
                    ws.update_manifest(
                        current_stage=manifest.current_stage,
                        stages=manifest.stages,
                    )
                    console.print(
                        f"  Resetting failed stage [yellow]{stage_name}[/yellow] to pending"
                    )
                    found_failed = True
                    break
            if not found_failed:
                console.print(
                    "[yellow]Pipeline is FAILED but no failed stage found. "
                    "Check manifest manually.[/yellow]"
                )
                raise typer.Exit(1)
        else:
            console.print("[green]Pipeline already completed.[/green]")
            return

    is_deep = manifest.pipeline_mode == PipelineMode.DEEP
    _ensure_env_selected(config)

    if is_deep:
        _run_with_live_progress(
            lambda cb: UnifiedPipelineOrchestrator(ws, config, progress_callback=cb),
            _run_deep_pipeline, manifest.topic, ws, use_tui=tui,
        )
    else:
        _run_with_live_progress(
            lambda cb: PipelineOrchestrator(ws, config, progress_callback=cb),
            _run_pipeline, manifest.topic, ws, use_tui=tui,
        )


@app.command()
def status(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Path to workspace directory"),
) -> None:
    """Show the status of a research session."""
    ws = _load_workspace_safe(workspace)
    manifest = ws.manifest

    table = Table(title=f"Session: {manifest.session_id}")
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Completed")
    table.add_column("Retries")

    status_colors = {
        "pending": "dim",
        "running": "yellow",
        "completed": "green",
        "failed": "red",
    }

    for stage_name, rec in manifest.stages.items():
        color = status_colors.get(rec.status, "white")
        started = rec.started_at.strftime("%H:%M:%S") if rec.started_at else "-"
        completed = rec.completed_at.strftime("%H:%M:%S") if rec.completed_at else "-"
        table.add_row(
            stage_name,
            f"[{color}]{rec.status}[/{color}]",
            started,
            completed,
            str(rec.retries),
        )

    console.print(table)
    console.print(f"\n[bold]Topic:[/bold] {manifest.topic}")
    console.print(f"[bold]Mode:[/bold] {manifest.pipeline_mode.value}")
    execution_profile = manifest.config_snapshot.get("execution_profile", "?")
    console.print(f"[bold]Profile:[/bold] {execution_profile}")
    console.print(f"[bold]Current Stage:[/bold] {manifest.current_stage.value}")
    console.print(f"[bold]Artifacts:[/bold] {len(manifest.artifacts)}")
    for art in manifest.artifacts:
        console.print(f"  - {art.name}: {art.path}")


@app.command("list")
def list_sessions(
    root: Path = typer.Option(_DEFAULT_ROOT, "--root", "-r"),
) -> None:
    """List all research sessions."""
    if not root.is_dir():
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Research Sessions")
    table.add_column("Session ID", style="bold")
    table.add_column("Topic")
    table.add_column("Stage")
    table.add_column("Created")

    for session_dir in sorted(root.iterdir()):
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            created = str(data.get("created_at", "?"))
            table.add_row(
                data.get("session_id", "?"),
                str(data.get("topic", "?"))[:50],
                data.get("current_stage", "?"),
                created[:19] if len(created) >= 19 else created,
            )
        except (json.JSONDecodeError, OSError) as exc:
            console.print(
                f"[dim]Skipping {session_dir.name}: corrupted manifest ({exc})[/dim]"
            )
            continue

    console.print(table)


class _StderrSilencer:
    """Redirect stderr at OS file-descriptor level (catches C extensions too)."""

    def __init__(self, log_file: Path) -> None:
        self._log_file = log_file
        self._saved_fd: int | None = None
        self._log_fh = None

    def __enter__(self):
        import warnings
        # 1. Suppress ALL Python warnings globally
        warnings.filterwarnings("ignore")
        # 1b. Override showwarning to prevent any leakage (some packages bypass filters)
        self._orig_showwarning = warnings.showwarning
        warnings.showwarning = lambda *a, **kw: None
        # 2. Redirect fd-level stderr → log file (catches C extensions, warnings.warn, etc.)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._saved_fd = os.dup(2)  # save original stderr fd
        self._log_fh = open(str(self._log_file), "a", encoding="utf-8")
        os.dup2(self._log_fh.fileno(), 2)  # redirect fd 2 → log file
        # 3. Also redirect Python-level sys.stderr
        self._orig_stderr = sys.stderr
        sys.stderr = self._log_fh
        return self

    def __exit__(self, *exc):
        import warnings
        # Restore everything
        if self._saved_fd is not None:
            os.dup2(self._saved_fd, 2)
            os.close(self._saved_fd)
        if self._orig_stderr:
            sys.stderr = self._orig_stderr
        if self._log_fh:
            self._log_fh.close()
        if hasattr(self, "_orig_showwarning"):
            warnings.showwarning = self._orig_showwarning
        warnings.resetwarnings()


def _ensure_env_selected(config: ResearchConfig) -> None:
    """If no experiment_python is configured, prompt user to select BEFORE Live starts.

    The interactive input() must happen before Rich Live takes over the terminal,
    otherwise the prompt is hidden behind the progress panel.
    """
    if config.experiment_python:
        return  # already configured
    if not sys.stdin.isatty():
        return  # non-interactive, let auto-create handle it
    # Check if all experiment stages are skipped (e.g. --dev mode)
    exp_stages = {"setup", "coding", "execution", "analysis"}
    if exp_stages.issubset(set(config.skip_stages)):
        return  # no experiment stages, no env needed

    try:
        from nanoresearch.agents.runtime_env._discovery import discover_environments
        envs = discover_environments()
    except Exception:
        return  # discovery failed, let pipeline handle it

    if not envs:
        return  # no envs found, auto-create will handle it

    console.print()
    table = Table(title="Available Python Environments", box=box.SIMPLE)
    table.add_column("#", justify="right", width=4)
    table.add_column("Environment", width=30)
    table.add_column("Python", width=12)
    table.add_column("Packages")
    for i, env in enumerate(envs, 1):
        pkgs = ", ".join(env["packages"]) if env["packages"] else ""
        table.add_row(str(i), env["name"], env["version"], pkgs)
    table.add_row("0", "[dim]Skip (auto-create)[/dim]", "", "")
    console.print(table)

    try:
        raw = input(f"  Select environment [0-{len(envs)}]: ").strip()
        choice = int(raw) if raw else 0
    except (ValueError, EOFError, KeyboardInterrupt):
        console.print()
        return

    if choice < 1 or choice > len(envs):
        return

    selected = envs[choice - 1]
    python_path = selected["python"]
    config.experiment_python = python_path

    # Save to config.json
    cfg_path = Path.home() / ".nanoresearch" / "config.json"
    cfg_data: dict = {}
    if cfg_path.exists():
        try:
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    cfg_data.setdefault("research", {})["experiment_python"] = python_path
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_data, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"  [green]Selected:[/green] {selected['name']} -> {python_path}")
    console.print(f"  [dim]Saved to {cfg_path}[/dim]\n")


def _run_with_live_progress(
    make_orchestrator, run_fn, topic: str, ws: Workspace, *, use_tui: bool = False,
) -> None:
    """Run a pipeline with TUI, Live panel, or fallback progress display."""
    log_file = ws.path / "logs" / "pipeline.log"

    # ── TUI mode: full-screen Textual app ──
    if use_tui and console.is_terminal:
        _setup_logging(log_file=log_file)
        try:
            from nanoresearch.tui import PipelineTUI

            tui_app = PipelineTUI()
            orchestrator = make_orchestrator(tui_app.progress_callback)

            async def _tui_pipeline():
                try:
                    return await run_fn(orchestrator, topic)
                finally:
                    await orchestrator.close()

            tui_app._run_coro = _tui_pipeline
            result = tui_app.run()

            if tui_app._error:
                console.print(f"\n[red]Pipeline failed:[/red] {tui_app._error}")
                console.print(f"[dim]Full log: {log_file}[/dim]")
                raise typer.Exit(1)
            if result:
                _print_result(result, ws)
            console.print(f"[dim]Full log: {log_file}[/dim]")
        except ImportError:
            console.print("[yellow]textual not installed, falling back to inline progress.[/yellow]")
            console.print("[dim]Install with: pip install textual[/dim]\n")
            _run_with_live_progress(make_orchestrator, run_fn, topic, ws, use_tui=False)
        return

    # ── Inline Rich Live mode ──
    if console.is_terminal:
        _setup_logging(log_file=log_file)

        # Silence ALL stderr noise (logging, warnings, C extensions) at fd level
        with _StderrSilencer(log_file):
            display = LiveProgressDisplay()
            orchestrator = make_orchestrator(display)
            try:
                with Live(display, console=console, refresh_per_second=2) as live:
                    display._live = live
                    result = asyncio.run(run_fn(orchestrator, topic))
                _print_result(result, ws)
                console.print(f"[dim]Full log: {log_file}[/dim]")
            except Exception as e:
                console.print(f"\n[red]Pipeline failed:[/red] {e}")
                console.print(f"[dim]Full log: {log_file}[/dim]")
                raise typer.Exit(1)
    else:
        orchestrator = make_orchestrator(_cli_progress)
        try:
            result = asyncio.run(run_fn(orchestrator, topic))
            _print_result(result, ws)
        except Exception as e:
            console.print(f"[red]Pipeline failed:[/red] {e}")
            raise typer.Exit(1)


async def _run_pipeline(orchestrator: PipelineOrchestrator, topic: str) -> dict:
    try:
        return await orchestrator.run(topic)
    finally:
        await orchestrator.close()


async def _run_deep_pipeline(orchestrator, topic: str) -> dict:
    try:
        return await orchestrator.run(topic)
    finally:
        await orchestrator.close()


def _print_result(result: dict, workspace: Workspace) -> None:
    console.print("\n[bold green]Pipeline completed![/bold green]\n")

    # Auto-export to a clean output folder
    try:
        export_path = workspace.export()
        console.print(Panel(
            f"[bold]Output folder:[/bold] {export_path}\n\n"
            f"  paper.pdf        — Compiled paper\n"
            f"  paper.tex        — LaTeX source\n"
            f"  references.bib   — Bibliography\n"
            f"  figures/         — All figures\n"
            f"  code/            — Experiment code skeleton\n"
            f"  data/            — Structured research data\n"
            f"  manifest.json    — Pipeline execution record",
            title="[green]Exported[/green]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[yellow]Export failed:[/yellow] {e}")
        console.print(f"[bold]Raw workspace:[/bold] {workspace.path}")


# Import command modules to register their @app.command() decorators
import nanoresearch.cli_commands  # noqa: F401, E402
import nanoresearch.cli_code_edit  # noqa: F401, E402
import nanoresearch.cli_paper_edit  # noqa: F401, E402


if __name__ == "__main__":
    app()
