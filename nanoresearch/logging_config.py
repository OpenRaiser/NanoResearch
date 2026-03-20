"""Structured logging setup — console (human) + file (JSON lines)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Capture extra structured fields set via `extra={}` on log calls
        for key in ("stage", "session_id", "model", "tokens", "latency_ms",
                     "error_type", "agent"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(
    log_path: Path | None = None,
    level: int = logging.INFO,
    console_level: int | None = None,
) -> None:
    """Configure the ``nanoresearch`` logger with dual output.

    Args:
        log_path: Path for the JSON-lines log file.  ``None`` disables file logging.
        level: Root log level for the ``nanoresearch`` logger.
        console_level: Override console handler level (defaults to *level*).
    """
    root = logging.getLogger("nanoresearch")
    root.setLevel(level)

    # Close and remove existing handlers to avoid fd leaks on repeated calls
    for h in root.handlers[:]:
        try:
            h.close()
        except Exception:
            pass  # closing old handlers — safe to ignore
    root.handlers.clear()

    # Console handler — human-readable
    console = logging.StreamHandler()
    console.setLevel(console_level if console_level is not None else level)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console)

    # File handler — JSON lines
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(JSONFormatter())
        root.addHandler(fh)

    # Prevent propagation to root logger (avoids double output)
    root.propagate = False
