"""Interactive code editing assistant with backup/rollback support.

Allows users to describe changes in natural language. The LLM reads the
current code, produces file edits, and applies them — with automatic
snapshots so the user can roll back at any time.
"""

from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.multi_model import ModelDispatcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot / rollback manager
# ---------------------------------------------------------------------------

class CodeSnapshotManager:
    """Manages zip-based snapshots of a code directory."""

    def __init__(self, code_dir: Path) -> None:
        self.code_dir = code_dir.resolve()
        self.backup_dir = self.code_dir / ".code_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # -- create --------------------------------------------------------

    def create_snapshot(self, label: str = "") -> str:
        """Create a zip snapshot. Returns the snapshot id."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)[:40]
        snap_id = f"{ts}_{safe_label}" if safe_label else ts
        zip_path = self.backup_dir / f"{snap_id}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(self.code_dir.rglob("*")):
                if fp.is_file() and ".code_backups" not in fp.parts:
                    arcname = fp.relative_to(self.code_dir)
                    zf.write(fp, arcname)

        return snap_id

    # -- list ----------------------------------------------------------

    def list_snapshots(self) -> list[dict[str, str]]:
        """Return list of ``{"id": ..., "time": ..., "path": ...}``."""
        snaps: list[dict[str, str]] = []
        for zp in sorted(self.backup_dir.glob("*.zip")):
            snap_id = zp.stem
            parts = snap_id.split("_", 2)
            if len(parts) >= 2:
                time_str = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:8]} {parts[1][:2]}:{parts[1][2:4]}:{parts[1][4:6]}"
            else:
                time_str = snap_id
            snaps.append({"id": snap_id, "time": time_str, "path": str(zp)})
        return snaps

    # -- rollback ------------------------------------------------------

    def rollback(self, snapshot_id: str) -> bool:
        """Restore code from a snapshot. Returns True on success."""
        zip_path = self.backup_dir / f"{snapshot_id}.zip"
        if not zip_path.exists():
            return False

        # Remove current files (except .code_backups)
        for fp in list(self.code_dir.rglob("*")):
            if fp.is_file() and ".code_backups" not in fp.parts:
                fp.unlink()

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.code_dir)
        return True

    def rollback_latest(self) -> str | None:
        """Rollback to the most recent snapshot. Returns snapshot id or None."""
        snaps = self.list_snapshots()
        if not snaps:
            return None
        latest = snaps[-1]
        if self.rollback(latest["id"]):
            return latest["id"]
        return None


# ---------------------------------------------------------------------------
# Code context builder
# ---------------------------------------------------------------------------

_CODE_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".sh", ".txt", ".md"}
_MAX_FILE_SIZE = 50_000  # chars


def read_code_context(code_dir: Path) -> list[dict[str, str]]:
    """Read all code files in the directory, return list of {path, content}."""
    files: list[dict[str, str]] = []
    for fp in sorted(code_dir.rglob("*")):
        if not fp.is_file():
            continue
        if ".code_backups" in fp.parts or "__pycache__" in fp.parts:
            continue
        if fp.suffix not in _CODE_EXTENSIONS:
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            if len(content) > _MAX_FILE_SIZE:
                content = content[:_MAX_FILE_SIZE] + "\n... [truncated]"
            files.append({
                "path": str(fp.relative_to(code_dir)),
                "content": content,
            })
        except Exception:
            continue
    return files


def _format_code_context(files: list[dict[str, str]]) -> str:
    """Format code files into a single context string."""
    parts: list[str] = []
    for f in files:
        parts.append(f"=== {f['path']} ===\n{f['content']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM-powered code editor
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise code editor. The user will describe changes to make to their \
experiment project. You have the full code context below.

Your task: produce a JSON array of file edits. Each edit is an object:

```json
[
  {
    "action": "edit",
    "path": "relative/path/to/file.py",
    "old": "exact string to find (multi-line ok)",
    "new": "replacement string"
  },
  {
    "action": "create",
    "path": "relative/path/to/new_file.py",
    "content": "full file content"
  },
  {
    "action": "delete",
    "path": "relative/path/to/remove.py"
  }
]
```

Rules:
- "old" must be an EXACT substring of the current file content (including whitespace/indentation).
- Keep "old" long enough to be unique in the file.
- For large changes, use multiple edit objects on the same file.
- Only output the JSON array, no markdown fences, no explanation.
- If you need to create a new file, use "create" action.
- If you need to delete a file, use "delete" action.
- Preserve existing code style and indentation.
- Make minimal changes — do not refactor unrelated code.
"""


class InteractiveCodeEditor:
    """LLM-powered interactive code editor with backup/rollback."""

    def __init__(
        self,
        code_dir: Path,
        config: ResearchConfig,
        log_fn: Any = None,
    ) -> None:
        self.code_dir = code_dir.resolve()
        self.snapshot_mgr = CodeSnapshotManager(self.code_dir)
        self.config = config
        self.dispatcher = ModelDispatcher(config)
        self._log = log_fn or (lambda msg: None)
        self._history: list[dict[str, str]] = []

    async def preview_instruction(self, instruction: str) -> dict[str, Any]:
        """Dry-run: ask LLM for edits but do NOT apply them.

        Returns dict with keys: edits, errors (same format as apply_instruction
        but with the raw edit plan so the user can review before applying).
        """
        files = read_code_context(self.code_dir)
        if not files:
            return {"edits": [], "errors": ["No code files found"]}

        code_ctx = _format_code_context(files)
        user_prompt = (
            f"## Current Code\n\n{code_ctx}\n\n"
            f"## User Instruction\n\n{instruction}"
        )
        stage_cfg = self.config.for_stage("code_gen")
        try:
            raw = await self.dispatcher.generate(
                stage_cfg, _SYSTEM_PROMPT, user_prompt, json_mode=True,
            )
        except Exception as exc:
            return {"edits": [], "errors": [f"LLM call failed: {exc}"]}

        edits, errors = self._parse_edits(raw)
        return {"edits": edits, "errors": errors}

    async def apply_instruction(self, instruction: str) -> dict[str, Any]:
        """Apply a natural-language instruction to the code.

        1. Auto-snapshot before changes
        2. Read all code files
        3. Ask LLM for edits
        4. Apply edits
        5. Return summary

        Returns dict with keys: snapshot_id, edits_applied, errors, files_changed.
        """
        # Step 1: backup
        snap_id = self.snapshot_mgr.create_snapshot(label="before_edit")
        self._log(f"Backup created: {snap_id}")

        # Step 2: read code
        files = read_code_context(self.code_dir)
        if not files:
            return {
                "snapshot_id": snap_id,
                "edits_applied": 0,
                "errors": ["No code files found in directory"],
                "files_changed": [],
            }

        code_ctx = _format_code_context(files)

        # Step 3: ask LLM
        user_prompt = (
            f"## Current Code\n\n{code_ctx}\n\n"
            f"## User Instruction\n\n{instruction}"
        )
        stage_cfg = self.config.for_stage("code_gen")
        try:
            raw = await self.dispatcher.generate(
                stage_cfg, _SYSTEM_PROMPT, user_prompt, json_mode=True,
            )
        except Exception as exc:
            return {
                "snapshot_id": snap_id,
                "edits_applied": 0,
                "errors": [f"LLM call failed: {exc}"],
                "files_changed": [],
            }

        # Step 4: parse and apply
        edits, parse_errors = self._parse_edits(raw)
        applied = 0
        errors = list(parse_errors)
        files_changed: list[str] = []

        for edit in edits:
            action = edit.get("action", "edit")
            path = edit.get("path", "")
            if not path:
                errors.append(f"Edit missing 'path': {edit}")
                continue

            target = self.code_dir / path

            if action == "delete":
                if target.exists():
                    target.unlink()
                    files_changed.append(f"deleted: {path}")
                    applied += 1
                else:
                    errors.append(f"File not found for delete: {path}")
                continue

            if action == "create":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(edit.get("content", ""), encoding="utf-8")
                files_changed.append(f"created: {path}")
                applied += 1
                continue

            # action == "edit"
            old = edit.get("old", "")
            new = edit.get("new", "")
            if not old:
                errors.append(f"Edit missing 'old' for {path}")
                continue

            if not target.exists():
                errors.append(f"File not found: {path}")
                continue

            content = target.read_text(encoding="utf-8", errors="replace")
            if old not in content:
                errors.append(
                    f"old string not found in {path} "
                    f"(first 60 chars: {old[:60]!r})"
                )
                continue

            content = content.replace(old, new, 1)
            target.write_text(content, encoding="utf-8")
            files_changed.append(f"edited: {path}")
            applied += 1

        self._history.append({
            "instruction": instruction,
            "snapshot_id": snap_id,
            "applied": str(applied),
        })

        return {
            "snapshot_id": snap_id,
            "edits_applied": applied,
            "errors": errors,
            "files_changed": files_changed,
        }

    def undo(self) -> str | None:
        """Rollback to the latest snapshot. Returns snapshot id or None."""
        result = self.snapshot_mgr.rollback_latest()
        if result:
            self._log(f"Rolled back to {result}")
        return result

    def rollback_to(self, snapshot_id: str) -> bool:
        """Rollback to a specific snapshot."""
        return self.snapshot_mgr.rollback(snapshot_id)

    @staticmethod
    def _parse_edits(raw: str) -> tuple[list[dict[str, str]], list[str]]:
        """Parse LLM response into edit objects."""
        errors: list[str] = []
        text = raw.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"JSON parse error: {exc}")
            return [], errors

        if isinstance(data, dict):
            # Sometimes LLM wraps in {"edits": [...]}
            for key in ("edits", "changes", "files"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                data = [data]

        if not isinstance(data, list):
            errors.append(f"Expected JSON array, got {type(data).__name__}")
            return [], errors

        return data, errors
