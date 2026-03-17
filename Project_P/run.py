"""Project_P entry point.

Usage:
    python run.py path/to/paper/          # Full fix + compile
    python run.py path/to/paper/ --no-llm # Without LLM
    python run.py path/to/paper/ --no-compile  # Skip PDF compilation
    python run.py path/to/paper/ --dry-run # Only report, don't write
    python run.py path/to/paper/ --restore # Restore from .bak/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Project_P: Paper formatting fix agent")
    parser.add_argument("paper_dir", type=Path, help="Directory containing paper.tex")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM features")
    parser.add_argument("--no-compile", action="store_true", help="Skip PDF compilation")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write")
    parser.add_argument("--restore", action="store_true", help="Restore from .bak/ backup")
    parser.add_argument("--config", type=Path, default=None, help="Config file path")

    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()

    if not paper_dir.is_dir():
        print(f"Error: {paper_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.restore:
        from project_p.pipeline import restore_paper
        if restore_paper(paper_dir):
            print("Restored successfully.")
        else:
            print("Restore failed.", file=sys.stderr)
            sys.exit(1)
        return

    from project_p.config import Config
    from project_p.pipeline import fix_paper

    config = Config.load(args.config) if args.config else Config.load()

    result = fix_paper(
        paper_dir,
        use_llm=not args.no_llm,
        do_compile=not args.no_compile,
        dry_run=args.dry_run,
        config=config,
    )

    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
