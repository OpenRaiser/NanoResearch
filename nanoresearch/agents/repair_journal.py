"""Utilities for recording reversible experiment-file mutations."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPAIR_SNAPSHOT_JOURNAL_PATH = "logs/repair_snapshot_journal.json"
REPAIR_SNAPSHOT_ARCHIVE_DIR = "logs/repair_snapshots"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative_display_path(target_path: Path, root_dir: Path | None) -> str:
    resolved_target = target_path.resolve(strict=False)
    if root_dir is not None:
        try:
            return str(resolved_target.relative_to(root_dir.resolve(strict=False))).replace("\\", "/")
        except ValueError:
            pass
    return str(resolved_target)


def _safe_snapshot_name(path_text: str) -> str:
    cleaned = path_text.replace("\\", "__").replace("/", "__").replace(":", "_")
    return cleaned.replace("..", "__")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_repair_snapshot(
    workspace_root: Path,
    target_path: Path,
    *,
    namespace: str,
    root_dir: Path | None = None,
    existed_before: bool | None = None,
    operation: str = "",
) -> dict[str, Any]:
    """Capture the pre-mutation state for a target file."""

    previous_state = target_path.exists() if existed_before is None else bool(existed_before)
    resolved_target = target_path.resolve(strict=False)
    relative_path = _relative_display_path(resolved_target, root_dir)

    snapshot: dict[str, Any] = {
        "captured_at": _utc_now(),
        "path": relative_path,
        "absolute_path": str(resolved_target),
        "existed_before": previous_state,
        "operation": operation or ("rewrite" if previous_state else "create"),
        "snapshot_path": "",
        "size_bytes": 0,
        "sha256": "",
    }

    if not previous_state:
        return snapshot

    snapshot_root = workspace_root / REPAIR_SNAPSHOT_ARCHIVE_DIR / namespace
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_name = f"{uuid.uuid4().hex[:12]}__{_safe_snapshot_name(relative_path)}"
    snapshot_file = snapshot_root / snapshot_name
    shutil.copy2(target_path, snapshot_file)

    snapshot["snapshot_path"] = str(snapshot_file.relative_to(workspace_root)).replace("\\", "/")
    snapshot["size_bytes"] = int(target_path.stat().st_size)
    snapshot["sha256"] = _sha256_file(target_path)
    return snapshot


def rollback_snapshot(
    workspace_root: Path,
    target_path: Path,
    snapshot: dict[str, Any],
) -> None:
    """Restore a file to its captured pre-mutation state."""

    if bool(snapshot.get("existed_before")):
        snapshot_path = str(snapshot.get("snapshot_path", "")).strip()
        if not snapshot_path:
            raise FileNotFoundError(f"Snapshot path missing for {target_path}")
        source = workspace_root / snapshot_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        return

    if target_path.exists():
        target_path.unlink()


def append_snapshot_journal(
    workspace_root: Path,
    *,
    agent: str,
    mutation_kind: str,
    scope: str,
    snapshots: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    journal_path: str = REPAIR_SNAPSHOT_JOURNAL_PATH,
) -> dict[str, Any]:
    """Append a batch of snapshots to the workspace-level repair journal."""

    journal_file = workspace_root / journal_path
    payload: dict[str, Any] = {
        "entry_count": 0,
        "entries": [],
    }
    if journal_file.is_file():
        try:
            existing = json.loads(journal_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            entries = existing.get("entries")
            if isinstance(entries, list):
                payload["entries"] = entries

    entries = list(payload["entries"])
    entry: dict[str, Any] = {
        "entry_id": len(entries) + 1,
        "recorded_at": _utc_now(),
        "agent": agent,
        "mutation_kind": mutation_kind,
        "scope": scope,
        "snapshot_count": len(snapshots),
        "snapshots": list(snapshots),
    }
    if metadata:
        entry["metadata"] = dict(metadata)

    entries.append(entry)
    payload["entries"] = entries
    payload["entry_count"] = len(entries)

    journal_file.parent.mkdir(parents=True, exist_ok=True)
    journal_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return entry
