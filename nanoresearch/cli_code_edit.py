"""CLI commands for interactive code editing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from nanoresearch.cli import app, console, _load_config_safe


@app.command("code")
def code_edit(
    dir: Path = typer.Option(
        ..., "--dir", "-d",
        help="Path to the code directory to edit",
    ),
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config file"),
    instruction: str = typer.Option(
        None, "--instruction", "-i",
        help="One-shot instruction (non-interactive mode)",
    ),
) -> None:
    """Interactive code editor with LLM assistance and auto-backup.

    Enter instructions in natural language. The LLM reads your code,
    makes changes, and auto-backs up before each edit.

    Commands during interactive mode:
      preview <...> --- dry-run: show proposed edits without applying
      undo          --- rollback to the state before the last edit
      rollback      --- show all snapshots and pick one to restore
      history       --- show edit history
      files         --- list current code files
      exit / quit   --- exit the editor
    """
    from rich.panel import Panel

    from nanoresearch.agents.code_editor import (
        CodeSnapshotManager,
        InteractiveCodeEditor,
        read_code_context,
    )

    code_dir = Path(dir).resolve()
    if not code_dir.is_dir():
        console.print(f"[red]Directory not found:[/red] {code_dir}")
        raise typer.Exit(1)

    config = _load_config_safe(config_path)
    editor = InteractiveCodeEditor(
        code_dir, config, log_fn=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
    )

    # -- One-shot mode --
    if instruction:
        result = asyncio.run(_code_apply(editor, instruction))
        _print_edit_result(result)
        return

    # -- Interactive mode --
    console.print(Panel(
        f"[bold]Code directory:[/bold] {code_dir}\n"
        f"[bold]Files:[/bold] {len(read_code_context(code_dir))}\n\n"
        "Type your instructions in natural language.\n"
        "Commands: [cyan]undo[/cyan] | [cyan]rollback[/cyan] | "
        "[cyan]history[/cyan] | [cyan]files[/cyan] | [cyan]exit[/cyan]",
        title="NanoResearch Code Editor",
        border_style="blue",
    ))

    while True:
        try:
            user_input = console.input("\n[bold green]> [/bold green]").strip()
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
                console.print(f"[green]Rolled back to snapshot:[/green] {snap}")
            else:
                console.print("[yellow]No snapshots to roll back to.[/yellow]")
            continue

        if cmd == "rollback":
            snaps = editor.snapshot_mgr.list_snapshots()
            if not snaps:
                console.print("[yellow]No snapshots available.[/yellow]")
                continue
            table = Table(title="Snapshots")
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

        if cmd == "history":
            if not editor._history:
                console.print("[dim]No edits yet.[/dim]")
            else:
                for i, h in enumerate(editor._history, 1):
                    console.print(
                        f"  {i}. [cyan]{h['instruction'][:80]}[/cyan] "
                        f"({h['applied']} edits, backup: {h['snapshot_id']})"
                    )
            continue

        if cmd == "files":
            files = read_code_context(code_dir)
            for f in files:
                lines = f["content"].count("\n") + 1
                console.print(f"  {f['path']:<40s}  {lines:>5} lines")
            console.print(f"\n  [bold]{len(files)}[/bold] file(s) total")
            continue

        # -- Dry-run: preview proposed edits before applying --
        if cmd.startswith("preview ") or cmd.startswith("dry-run "):
            instr = user_input.split(" ", 1)[1] if " " in user_input else ""
            if not instr:
                console.print("[yellow]Usage: preview <instruction>[/yellow]")
                continue
            console.print("[dim]Generating edit plan (not applying)...[/dim]")
            preview = asyncio.run(_code_preview(editor, instr))
            errs = preview.get("errors", [])
            edits = preview.get("edits", [])
            if errs:
                for e in errs:
                    console.print(f"  [red]Error:[/red] {e}")
            if edits:
                console.print(f"\n  [bold]{len(edits)} proposed edit(s):[/bold]")
                for j, ed in enumerate(edits, 1):
                    action = ed.get("action", "edit")
                    path = ed.get("path", "?")
                    if action == "edit":
                        old_preview = ed.get("old", "")[:80].replace("\n", "\\n")
                        console.print(f"  {j}. [cyan]edit[/cyan] {path}: {old_preview!r} -> ...")
                    elif action == "create":
                        console.print(f"  {j}. [green]create[/green] {path}")
                    elif action == "delete":
                        console.print(f"  {j}. [red]delete[/red] {path}")
                console.print("\n  [dim]To apply, run the same instruction without 'preview'.[/dim]")
            else:
                console.print("[yellow]No edits proposed.[/yellow]")
            continue

        # -- Normal instruction -> apply via LLM --
        console.print("[dim]Applying changes...[/dim]")
        result = asyncio.run(_code_apply(editor, user_input))
        _print_edit_result(result)


async def _code_apply(
    editor: "InteractiveCodeEditor",  # noqa: F821
    instruction: str,
) -> dict:
    return await editor.apply_instruction(instruction)


async def _code_preview(
    editor: "InteractiveCodeEditor",  # noqa: F821
    instruction: str,
) -> dict:
    return await editor.preview_instruction(instruction)


def _print_edit_result(result: dict) -> None:
    """Pretty-print the result of an edit operation."""
    snap = result.get("snapshot_id", "")
    applied = result.get("edits_applied", 0)
    errors = result.get("errors", [])
    changed = result.get("files_changed", [])

    if applied > 0:
        console.print(f"[green]Applied {applied} edit(s)[/green]  (backup: {snap})")
        for f in changed:
            console.print(f"  [cyan]{f}[/cyan]")
    else:
        console.print("[yellow]No edits applied.[/yellow]")

    if errors:
        for err in errors:
            console.print(f"  [red]Error:[/red] {err}")

    if applied > 0:
        console.print("[dim]Type 'undo' to rollback this change.[/dim]")
