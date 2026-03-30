"""Textual TUI for NanoResearch pipeline — full-screen terminal interface.

Replaces Rich Live with a proper TUI that:
- Left panel: stage list with live status + timer
- Right panel: scrolling real-time log
- Bottom: progress bar + stats + toggleable shortcut help
- No ANSI cursor hacks → no stacking bug on Windows
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Label, ProgressBar, RichLog, Static

# ── Pipeline stage definitions ──
_DEEP_STAGES = [
    "IDEATION", "PLANNING", "SETUP", "CODING", "EXECUTION",
    "ANALYSIS", "FIGURE_GEN", "WRITING", "REVIEW",
]

_STAGE_LABELS = {
    "IDEATION": "Ideation",
    "PLANNING": "Planning",
    "SETUP": "Setup",
    "CODING": "Coding",
    "EXECUTION": "Execution",
    "ANALYSIS": "Analysis",
    "FIGURE_GEN": "Figures",
    "WRITING": "Writing",
    "REVIEW": "Review",
}

_HELP_TEXT = """\
[bold]Keyboard Shortcuts[/bold]

  [cyan]q[/cyan]        Quit pipeline
  [cyan]Ctrl+C[/cyan]   Quit pipeline
  [cyan]Ctrl+P[/cyan]   Command palette
  [cyan]?[/cyan]        Toggle this help
  [cyan]Tab[/cyan]      Focus next panel
  [cyan]PgUp/Dn[/cyan]  Scroll log"""


def _norm(stage: str) -> str:
    """Normalize stage name to uppercase to match PipelineStage enum."""
    return stage.strip().upper()


class StageListWidget(Static):
    """Left panel: list of pipeline stages with status indicators."""

    def __init__(self, stages: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stages = stages or _DEEP_STAGES
        self._states: dict[str, dict] = {
            s: {"status": "pending", "start": 0.0, "elapsed": 0.0, "substep": ""}
            for s in self._stages
        }

    def update_stage(self, stage: str, status: str, message: str = "") -> None:
        key = _norm(stage)
        if key not in self._states:
            self._states[key] = {"status": "pending", "start": 0.0, "elapsed": 0.0, "substep": ""}
        s = self._states[key]
        if status == "substep":
            s["substep"] = message
        elif status == "started":
            s["status"] = "running"
            s["start"] = time.monotonic()
            s["substep"] = ""
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
        self._refresh_display()

    @property
    def completed_count(self) -> int:
        return sum(
            1 for s in self._states.values()
            if s["status"] in ("completed", "skipped")
        )

    @property
    def has_running(self) -> bool:
        return any(s["status"] == "running" for s in self._states.values())

    def _fmt_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"

    def _refresh_display(self) -> None:
        lines: list[str] = []
        for idx, stage in enumerate(self._stages, 1):
            s = self._states.get(stage, {})
            st = s.get("status", "pending")
            label = _STAGE_LABELS.get(stage, stage.title())

            if st == "running":
                elapsed = time.monotonic() - s.get("start", time.monotonic())
                time_str = self._fmt_time(elapsed)
                substep = s.get("substep", "")
                line = f"  [bold cyan]>  {idx}. {label:<11s}  RUN   {time_str:>6s}[/bold cyan]"
                if substep:
                    sub = substep[:24]
                    line += f"\n     [italic cyan]{sub}[/italic cyan]"
                lines.append(line)
            elif st == "completed":
                time_str = self._fmt_time(s.get("elapsed", 0))
                lines.append(f"  [green]+  {idx}. {label:<11s}  DONE  {time_str:>6s}[/green]")
            elif st == "failed":
                time_str = self._fmt_time(s.get("elapsed", 0))
                lines.append(f"  [red]x  {idx}. {label:<11s}  FAIL  {time_str:>6s}[/red]")
            elif st == "retrying":
                elapsed = time.monotonic() - s.get("start", time.monotonic())
                time_str = self._fmt_time(elapsed)
                lines.append(f"  [yellow]!  {idx}. {label:<11s}  RETRY {time_str:>6s}[/yellow]")
            elif st == "skipped":
                lines.append(f"  [dim]-  {idx}. {label:<11s}  SKIP[/dim]")
            else:
                lines.append(f"  [#6e7681].  {idx}. {label:<11s}  ...[/#6e7681]")

        self.update("\n".join(lines))


class PipelineTUI(App):
    """Full-screen TUI for NanoResearch pipeline."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        height: 1fr;
    }
    #stages-panel {
        width: 38;
        border: round $primary;
        padding: 1 0;
    }
    #stages-title {
        text-style: bold;
        color: $text;
        text-align: center;
        padding: 0 1;
    }
    #stage-list {
        margin-top: 1;
    }
    #log-panel {
        border: round $secondary;
    }
    #log-title {
        text-style: bold;
        color: $text;
        padding: 0 1;
    }
    #log {
        margin: 0 1;
    }
    #footer-bar {
        height: 3;
        padding: 0 2;
        layout: horizontal;
        background: $surface;
    }
    #progress {
        width: 1fr;
        margin-top: 1;
    }
    #progress-stats {
        width: auto;
        min-width: 28;
        text-align: right;
        margin-top: 1;
        padding-left: 2;
    }
    #hint-bar {
        height: 1;
        background: $primary-background;
        color: $text-muted;
        padding: 0 2;
    }
    #help-panel {
        height: auto;
        max-height: 12;
        background: $surface;
        border-top: solid $primary;
        padding: 1 3;
        display: none;
    }
    #help-panel.visible {
        display: block;
    }
    """

    TITLE = "NanoResearch Pipeline"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("question_mark", "toggle_help", "Help"),
    ]

    def __init__(
        self,
        stages: list[str] | None = None,
        run_coro: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stages_list = stages or _DEEP_STAGES
        self._run_coro = run_coro
        self._pipeline_start = time.monotonic()
        self._result: dict | None = None
        self._error: Exception | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="stages-panel"):
                yield Label("Stages", id="stages-title")
                yield StageListWidget(self._stages_list, id="stage-list")
            with Vertical(id="log-panel"):
                yield Label(" Log", id="log-title")
                yield RichLog(id="log", highlight=True, markup=True, wrap=True, auto_scroll=True)
        with Horizontal(id="footer-bar"):
            yield ProgressBar(id="progress", total=len(self._stages_list), show_eta=False, show_percentage=True)
            yield Label("0/9 stages | 0s", id="progress-stats")
        yield Static(_HELP_TEXT, id="help-panel")
        yield Label(
            " [dim]q[/dim] Quit  [dim]?[/dim] Help  [dim]^P[/dim] Palette",
            id="hint-bar",
        )

    def action_toggle_help(self) -> None:
        """Toggle the keyboard shortcuts help panel."""
        panel = self.query_one("#help-panel")
        panel.toggle_class("visible")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick_update)
        if self._run_coro is not None:
            self._run_pipeline()

    @work(thread=False)
    async def _run_pipeline(self) -> None:
        try:
            self._result = await self._run_coro()
        except Exception as e:
            self._error = e
            self.log_message(f"[bold red]PIPELINE FAILED: {e}[/bold red]")
        finally:
            self._update_stats()
            await asyncio.sleep(2.0)
            self.exit(self._result)

    def progress_callback(self, stage: str, status: str, message: str) -> None:
        """Drop-in replacement for LiveProgressDisplay.__call__."""
        try:
            stage_list: StageListWidget = self.query_one("#stage-list", StageListWidget)
            stage_list.update_stage(stage, status, message)
            progress: ProgressBar = self.query_one("#progress", ProgressBar)
            progress.progress = stage_list.completed_count
            self._update_stats()
        except Exception:
            pass

        if status == "substep":
            self.log_message(f"[dim]{_norm(stage)}:[/dim] {message}")
        elif status == "started":
            label = _STAGE_LABELS.get(_norm(stage), stage)
            self.log_message(f"[bold cyan]>> {label} started[/bold cyan]")
        elif status == "completed":
            label = _STAGE_LABELS.get(_norm(stage), stage)
            self.log_message(f"[bold green]++ {label} completed[/bold green]")
        elif status == "failed":
            label = _STAGE_LABELS.get(_norm(stage), stage)
            self.log_message(f"[bold red]!! {label} failed: {message}[/bold red]")
        elif status == "retrying":
            label = _STAGE_LABELS.get(_norm(stage), stage)
            self.log_message(f"[yellow]~~ {label} retrying: {message}[/yellow]")
        elif status == "skipped":
            label = _STAGE_LABELS.get(_norm(stage), stage)
            self.log_message(f"[dim]-- {label} skipped[/dim]")

    def log_message(self, msg: str) -> None:
        try:
            log_widget: RichLog = self.query_one("#log", RichLog)
            elapsed = time.monotonic() - self._pipeline_start
            m, s = divmod(int(elapsed), 60)
            log_widget.write(f"[dim]{m:02d}:{s:02d}[/dim]  {msg}")
        except Exception:
            pass

    def _update_stats(self) -> None:
        try:
            stage_list: StageListWidget = self.query_one("#stage-list", StageListWidget)
            total = len(self._stages_list)
            completed = stage_list.completed_count
            elapsed = time.monotonic() - self._pipeline_start
            m, s = divmod(int(elapsed), 60)
            label: Label = self.query_one("#progress-stats", Label)
            label.update(f"{completed}/{total} stages | {m}m{s:02d}s")
        except Exception:
            pass

    def _tick_update(self) -> None:
        try:
            stage_list: StageListWidget = self.query_one("#stage-list", StageListWidget)
            if stage_list.has_running:
                stage_list._refresh_display()
            self._update_stats()
        except Exception:
            pass
