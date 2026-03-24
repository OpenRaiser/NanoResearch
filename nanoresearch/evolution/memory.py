"""Long-term memory store for cross-workspace research context."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")


class MemoryType(str, Enum):
    USER_PROFILE = "user_profile"
    PROJECT_CONTEXT = "project_context"
    DECISION_HISTORY = "decision_history"


class MemoryScope(str, Enum):
    GLOBAL_USER = "global_user"
    PROJECT = "project"
    WORKSPACE_DERIVED = "workspace_derived"


class MemoryRecord(BaseModel):
    memory_id: str
    memory_type: MemoryType
    scope: MemoryScope = MemoryScope.WORKSPACE_DERIVED
    source: str = ""
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_weight: float = Field(default=1.0, ge=0.0, le=1.5)
    tags: list[str] = Field(default_factory=list)
    project_key: str = ""
    workspace_id: str = ""


_TASK_TYPE_WEIGHTS: dict[str, dict[MemoryType, float]] = {
    "literature": {
        MemoryType.USER_PROFILE: 1.35,
        MemoryType.PROJECT_CONTEXT: 1.0,
        MemoryType.DECISION_HISTORY: 0.95,
    },
    "planning": {
        MemoryType.USER_PROFILE: 0.95,
        MemoryType.PROJECT_CONTEXT: 1.35,
        MemoryType.DECISION_HISTORY: 1.15,
    },
    "experiment": {
        MemoryType.USER_PROFILE: 1.0,
        MemoryType.PROJECT_CONTEXT: 1.3,
        MemoryType.DECISION_HISTORY: 1.25,
    },
    "writing": {
        MemoryType.USER_PROFILE: 1.35,
        MemoryType.PROJECT_CONTEXT: 1.0,
        MemoryType.DECISION_HISTORY: 1.2,
    },
    "review": {
        MemoryType.USER_PROFILE: 1.1,
        MemoryType.PROJECT_CONTEXT: 1.0,
        MemoryType.DECISION_HISTORY: 1.35,
    },
}


class MemoryStore:
    """Persistent long-term memory store under ``~/.nanoresearch/memory``."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        enabled: bool = True,
        top_k: int = 5,
        decay_factor: float = 0.08,
    ) -> None:
        self.enabled = enabled
        self.top_k = max(1, top_k)
        self.decay_factor = max(0.0, decay_factor)
        self.root = root or (Path.home() / ".nanoresearch" / "memory")
        self.file = self.root / "records.json"
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _load_records(self) -> list[MemoryRecord]:
        if not self.enabled or not self.file.is_file():
            return []
        try:
            raw = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load memory store %s: %s", self.file, exc)
            return []
        records: list[MemoryRecord] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                records.append(MemoryRecord.model_validate(item))
            except Exception as exc:
                logger.debug("Skipping malformed memory record: %s", exc)
        return records

    def _save_records(self, records: list[MemoryRecord]) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        payload = [record.model_dump(mode="json") for record in records]
        self.file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _normalize_tags(tags: list[str] | None) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for tag in tags or []:
            tag_norm = re.sub(r"\s+", " ", str(tag).strip().lower())
            if not tag_norm or tag_norm in seen:
                continue
            seen.add(tag_norm)
            normalized.append(tag_norm)
        return normalized

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_WORD_RE.findall((text or "").lower()))

    @staticmethod
    def _make_memory_id(memory_type: MemoryType, scope: MemoryScope, content: str) -> str:
        digest = hashlib.sha1(f"{memory_type.value}|{scope.value}|{content}".encode("utf-8")).hexdigest()
        return f"mem-{digest[:12]}"

    def remember(
        self,
        memory_type: MemoryType | str,
        content: str,
        *,
        scope: MemoryScope | str = MemoryScope.WORKSPACE_DERIVED,
        source: str = "",
        importance: float = 0.6,
        recency_weight: float = 1.0,
        tags: list[str] | None = None,
        project_key: str = "",
        workspace_id: str = "",
    ) -> MemoryRecord | None:
        if not self.enabled:
            return None
        content = (content or "").strip()
        if not content:
            return None
        memory_type = MemoryType(memory_type)
        scope = MemoryScope(scope)
        tags = self._normalize_tags(tags)
        records = self._load_records()
        memory_id = self._make_memory_id(memory_type, scope, content[:400])
        now = datetime.now(timezone.utc)
        for index, record in enumerate(records):
            if record.memory_id != memory_id:
                continue
            updated = record.model_copy(update={
                "timestamp": now,
                "importance": max(record.importance, min(max(importance, 0.0), 1.0)),
                "recency_weight": min(1.5, max(record.recency_weight, recency_weight)),
                "tags": self._normalize_tags(record.tags + tags),
                "source": source or record.source,
                "project_key": project_key or record.project_key,
                "workspace_id": workspace_id or record.workspace_id,
            })
            records[index] = updated
            self._save_records(records)
            return updated
        record = MemoryRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            scope=scope,
            source=source,
            content=content,
            importance=min(max(importance, 0.0), 1.0),
            recency_weight=min(max(recency_weight, 0.0), 1.5),
            tags=tags,
            project_key=project_key,
            workspace_id=workspace_id,
        )
        records.append(record)
        self._save_records(records)
        return record

    def decay(self, *, project_key: str = "", amount: float | None = None) -> int:
        if not self.enabled:
            return 0
        amount = self.decay_factor if amount is None else max(0.0, amount)
        records = self._load_records()
        changed = 0
        updated: list[MemoryRecord] = []
        for record in records:
            if project_key and record.project_key != project_key:
                updated.append(record)
                continue
            new_weight = max(0.1, record.recency_weight - amount)
            if abs(new_weight - record.recency_weight) > 1e-6:
                changed += 1
            updated.append(record.model_copy(update={"recency_weight": new_weight}))
        if changed:
            self._save_records(updated)
        return changed

    def retrieve(
        self,
        task_type: str,
        *,
        topic: str = "",
        tags: list[str] | None = None,
        text: str = "",
        project_key: str = "",
        top_k: int | None = None,
    ) -> list[MemoryRecord]:
        if not self.enabled:
            return []
        weights = _TASK_TYPE_WEIGHTS.get(task_type, _TASK_TYPE_WEIGHTS.get("planning", {}))
        query_tags = set(self._normalize_tags(tags))
        query_tokens = self._tokenize(" ".join([topic, text, " ".join(query_tags)]))
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self._load_records():
            if project_key and record.project_key and record.project_key != project_key:
                if record.scope == MemoryScope.PROJECT:
                    continue
            memory_weight = weights.get(record.memory_type, 1.0)
            token_overlap = len(self._tokenize(record.content) & query_tokens)
            tag_overlap = len(set(record.tags) & query_tags)
            project_bonus = 0.45 if project_key and record.project_key == project_key else 0.0
            score = (
                memory_weight * (1.2 + record.importance + record.recency_weight)
                + 0.55 * token_overlap
                + 0.9 * tag_overlap
                + project_bonus
            )
            if score >= 1.45:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
        limit = max(1, top_k or self.top_k)
        return [record for _, record in scored[:limit]]

    def render_prompt_context(
        self,
        task_type: str,
        *,
        topic: str = "",
        tags: list[str] | None = None,
        text: str = "",
        project_key: str = "",
        top_k: int | None = None,
    ) -> str:
        records = self.retrieve(
            task_type,
            topic=topic,
            tags=tags,
            text=text,
            project_key=project_key,
            top_k=top_k,
        )
        if not records:
            return ""
        lines = []
        for record in records:
            source = f" [{record.source}]" if record.source else ""
            lines.append(f"- ({record.memory_type.value}){source} {record.content}")
        return (
            "\n\n=== LONG-TERM RESEARCH MEMORY ===\n"
            "Use these durable preferences, prior decisions, and project facts when making choices. "
            "Prefer recent high-importance memories, but do not hard-delete older context.\n"
            + "\n".join(lines)
            + "\n=== END LONG-TERM RESEARCH MEMORY ===\n"
        )
