"""Progress streaming — writes real-time status to a JSON file."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_EVENTS = 50  # rolling window of events kept in file


class ProgressEmitter:
    """Emits pipeline progress events to a JSON file.

    The file is atomically overwritten on each event so that external
    consumers can poll it safely.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._events: list[dict[str, Any]] = []
        self._pipeline_start = time.monotonic()
        self._stage_start: float | None = None

    def _emit(self, event: dict[str, Any]) -> None:
        try:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
            event["elapsed_s"] = round(time.monotonic() - self._pipeline_start, 1)
            self._events.append(event)
            # Keep only last N events
            if len(self._events) > MAX_EVENTS:
                self._events = self._events[-MAX_EVENTS:]
            self._write()
        except Exception as exc:
            # Progress is cosmetic — never crash the pipeline
            logger.debug("Progress emit failed (non-fatal): %s", exc)

    def _write(self) -> None:
        """Atomic write of the progress file."""
        data = {
            "events": self._events,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp"
            )
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                fd = -1
                os.replace(tmp, str(self._path))
            except BaseException:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            # Fallback: direct write
            try:
                self._path.write_text(content, encoding="utf-8")
            except OSError as exc:
                logger.debug("Failed to write progress file: %s", exc)

    def stage_start(
        self,
        stage: str,
        total_stages: int,
        stage_index: int,
        message: str = "",
    ) -> None:
        self._stage_start = time.monotonic()
        pct = round(stage_index / total_stages * 100) if total_stages else 0
        self._emit({
            "type": "stage_start",
            "stage": stage,
            "progress_pct": pct,
            "message": message or f"Starting {stage}...",
            "stage_index": stage_index,
            "total_stages": total_stages,
        })

    def stage_complete(
        self,
        stage: str,
        total_stages: int,
        stage_index: int,
        message: str = "",
    ) -> None:
        stage_elapsed = 0.0
        if self._stage_start is not None:
            stage_elapsed = round(time.monotonic() - self._stage_start, 1)
        pct = round((stage_index + 1) / total_stages * 100) if total_stages else 0
        self._emit({
            "type": "stage_complete",
            "stage": stage,
            "progress_pct": pct,
            "message": message or f"{stage} completed",
            "stage_elapsed_s": stage_elapsed,
        })

    def substep(self, stage: str, message: str) -> None:
        self._emit({
            "type": "substep",
            "stage": stage,
            "message": message,
        })

    def error(self, stage: str, message: str) -> None:
        self._emit({
            "type": "error",
            "stage": stage,
            "message": message,
        })

    def pipeline_complete(self, success: bool, message: str = "") -> None:
        self._emit({
            "type": "pipeline_complete",
            "success": success,
            "progress_pct": 100 if success else -1,
            "message": message or ("Pipeline completed" if success else "Pipeline failed"),
        })
