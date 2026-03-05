#!/usr/bin/env python3
"""Standalone figure regeneration script.

Only re-runs the FIGURE_GEN stage using existing workspace data (blueprint,
ideation_output).  Generated figure filenames are kept identical so they
drop straight into the existing LaTeX document for before/after comparison.

Usage:
    python regen_figures.py                       # auto-pick latest workspace
    python regen_figures.py <session_id>          # specific workspace
    python regen_figures.py --workspace <path>    # explicit directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nanoresearch.agents.figure_gen import FigureAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("regen_figures")

DEFAULT_WS_ROOT = Path.home() / ".nanobot" / "workspace" / "research"


# ── helpers ───────────────────────────────────────────────────────────

def find_workspace(session_id: str | None, explicit_path: str | None) -> Path:
    """Resolve to a workspace directory."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            sys.exit(f"Workspace path does not exist: {p}")
        return p

    if session_id:
        p = DEFAULT_WS_ROOT / session_id
        if not p.exists():
            sys.exit(f"Session {session_id} not found under {DEFAULT_WS_ROOT}")
        return p

    # Auto-pick the most recently modified workspace
    if not DEFAULT_WS_ROOT.exists():
        sys.exit(f"Default workspace root not found: {DEFAULT_WS_ROOT}")
    candidates = sorted(
        DEFAULT_WS_ROOT.iterdir(),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        sys.exit("No workspaces found.")
    return candidates[0]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def backup_figures(figures_dir: Path) -> Path | None:
    """Back up existing figures/ to figures_backup/ for comparison."""
    if not figures_dir.exists() or not any(figures_dir.iterdir()):
        return None
    backup = figures_dir.parent / "figures_backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(figures_dir, backup)
    log.info("Old figures backed up to %s", backup)
    return backup


# ── main ──────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate FIGURE_GEN stage")
    parser.add_argument("session_id", nargs="?", default=None,
                        help="Session ID (folder name under ~/.nanobot/workspace/research/)")
    parser.add_argument("--workspace", "-w", default=None,
                        help="Explicit workspace directory path")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backing up old figures")
    args = parser.parse_args()

    ws_path = find_workspace(args.session_id, args.workspace)
    log.info("Using workspace: %s", ws_path)

    # Load workspace
    workspace = Workspace.load(ws_path)
    topic = workspace.manifest.topic
    log.info("Topic: %s", topic)

    # Load config
    config = ResearchConfig.load()

    # ── Locate input artifacts ────────────────────────────────────────
    # ideation_output: could be in papers/ or data/
    ideation_path = ws_path / "papers" / "ideation_output.json"
    if not ideation_path.is_file():
        ideation_path = ws_path / "data" / "ideation_output.json"
    if not ideation_path.is_file():
        sys.exit(f"ideation_output.json not found in {ws_path}")
    ideation_output = load_json(ideation_path)
    log.info("Loaded ideation_output from %s", ideation_path)

    # blueprint: could be in plans/ or data/
    bp_path = ws_path / "plans" / "experiment_blueprint.json"
    if not bp_path.is_file():
        bp_path = ws_path / "data" / "experiment_blueprint.json"
    if not bp_path.is_file():
        sys.exit(f"experiment_blueprint.json not found in {ws_path}")
    blueprint = load_json(bp_path)
    log.info("Loaded blueprint from %s", bp_path)

    # experiment results (may be empty — that's fine, synthetic fallback kicks in)
    experiment_results: dict = {}
    experiment_status: str = "skipped"
    exp_out_path = ws_path / "logs" / "experiment_output.json"
    if exp_out_path.is_file():
        exp_out = load_json(exp_out_path)
        experiment_results = exp_out.get("experiment_results", {})
        experiment_status = exp_out.get("experiment_status", "skipped")
    log.info("Experiment status: %s  (has results: %s)",
             experiment_status, bool(experiment_results))

    # ── Back up old figures ───────────────────────────────────────────
    if not args.no_backup:
        backup_figures(ws_path / "figures")

    # ── Run FIGURE_GEN ────────────────────────────────────────────────
    agent = FigureAgent(workspace, config)
    log.info("=" * 60)
    log.info("Starting FIGURE_GEN (synthetic fallback enabled)")
    log.info("=" * 60)

    try:
        result = await agent.run(
            experiment_blueprint=blueprint,
            ideation_output=ideation_output,
            experiment_results=experiment_results,
            experiment_status=experiment_status,
        )
    finally:
        await agent.close()

    figures = result.get("figures", {})
    log.info("=" * 60)
    log.info("Done! Generated %d figures:", len(figures))
    for key, info in figures.items():
        out = info.get("output_path", "?")
        log.info("  %s  ->  %s", key, out)
    log.info("=" * 60)

    # ── Also copy into the export directory if it exists ──────────────
    export_dirs = list(ws_path.parent.glob("nanoresearch_*"))
    for edir in export_dirs:
        efig = edir / "figures"
        if efig.is_dir():
            for key, info in figures.items():
                src = Path(info.get("output_path", ""))
                if src.is_file():
                    dst = efig / src.name
                    shutil.copy2(src, dst)
                    log.info("Copied %s -> %s", src.name, dst)
                # Also copy PDF if exists
                pdf_src = src.with_suffix(".pdf")
                if pdf_src.is_file():
                    shutil.copy2(pdf_src, efig / pdf_src.name)


if __name__ == "__main__":
    asyncio.run(main())
