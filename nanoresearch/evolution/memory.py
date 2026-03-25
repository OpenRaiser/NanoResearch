"""Long-term memory stores for cross-workspace research context."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

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


class ResearchMemoryKind(str, Enum):
    PROMISING_DIRECTION = "promising_direction"
    FAILED_DIRECTION = "failed_direction"
    DATA_STRATEGY = "data_strategy"
    TRAINING_STRATEGY = "training_strategy"


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


class ResearchMemoryRecord(BaseModel):
    memory_id: str
    memory_kind: ResearchMemoryKind
    content: str
    task_family: str = ""
    proposal_ref: str = ""
    direction_ref: str = ""
    conditions: dict[str, str] = Field(default_factory=dict)
    evidence_summary: str = ""
    trajectory_summary: list[str] = Field(default_factory=list)
    uncertainty_note: str = ""
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    support_count: int = Field(default=1, ge=1)
    contradiction_count: int = Field(default=0, ge=0)
    source: str = ""
    source_stage: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
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


_RESEARCH_KIND_PRIORS: dict[str, dict[ResearchMemoryKind, float]] = {
    "literature": {
        ResearchMemoryKind.PROMISING_DIRECTION: 1.55,
        ResearchMemoryKind.FAILED_DIRECTION: 1.35,
        ResearchMemoryKind.DATA_STRATEGY: 0.65,
        ResearchMemoryKind.TRAINING_STRATEGY: 0.6,
    },
    "planning": {
        ResearchMemoryKind.PROMISING_DIRECTION: 1.6,
        ResearchMemoryKind.FAILED_DIRECTION: 1.45,
        ResearchMemoryKind.DATA_STRATEGY: 0.7,
        ResearchMemoryKind.TRAINING_STRATEGY: 0.65,
    },
    "experiment": {
        ResearchMemoryKind.PROMISING_DIRECTION: 0.7,
        ResearchMemoryKind.FAILED_DIRECTION: 0.95,
        ResearchMemoryKind.DATA_STRATEGY: 1.6,
        ResearchMemoryKind.TRAINING_STRATEGY: 1.55,
    },
    "writing": {
        ResearchMemoryKind.PROMISING_DIRECTION: 0.85,
        ResearchMemoryKind.FAILED_DIRECTION: 0.9,
        ResearchMemoryKind.DATA_STRATEGY: 0.7,
        ResearchMemoryKind.TRAINING_STRATEGY: 0.7,
    },
    "review": {
        ResearchMemoryKind.PROMISING_DIRECTION: 0.9,
        ResearchMemoryKind.FAILED_DIRECTION: 1.0,
        ResearchMemoryKind.DATA_STRATEGY: 0.7,
        ResearchMemoryKind.TRAINING_STRATEGY: 0.75,
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
        self.research_file = self.root / "research_records.json"
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

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

    def _load_research_records(self) -> list[ResearchMemoryRecord]:
        if not self.enabled or not self.research_file.is_file():
            return []
        try:
            raw = json.loads(self.research_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load research memory store %s: %s", self.research_file, exc)
            return []
        records: list[ResearchMemoryRecord] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                records.append(ResearchMemoryRecord.model_validate(item))
            except Exception as exc:
                logger.debug("Skipping malformed research memory record: %s", exc)
        return records

    def _save_research_records(self, records: list[ResearchMemoryRecord]) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        payload = [record.model_dump(mode="json") for record in records]
        self.research_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

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
    def _normalize_conditions(conditions: dict[str, Any] | None) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in (conditions or {}).items():
            key_norm = re.sub(r"\s+", "_", str(key).strip().lower())
            if not key_norm:
                continue
            if isinstance(value, (list, tuple, set)):
                value_norm = ", ".join(str(v).strip() for v in value if str(v).strip())
            elif isinstance(value, dict):
                value_norm = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                value_norm = str(value).strip()
            if value_norm:
                normalized[key_norm] = value_norm
        return normalized

    @staticmethod
    def _merge_trajectory(existing: list[str], incoming: list[str] | None, *, limit: int = 6) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for item in [*(existing or []), *((incoming or [])[:limit])]:
            item_norm = " ".join(str(item).strip().split())
            if not item_norm or item_norm in seen:
                continue
            seen.add(item_norm)
            merged.append(item_norm)
            if len(merged) >= limit:
                break
        return merged

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_WORD_RE.findall((text or "").lower()))

    @staticmethod
    def _make_memory_id(memory_type: MemoryType, scope: MemoryScope, content: str) -> str:
        digest = hashlib.sha1(f"{memory_type.value}|{scope.value}|{content}".encode("utf-8")).hexdigest()
        return f"mem-{digest[:12]}"

    @staticmethod
    def _make_research_memory_id(
        memory_kind: ResearchMemoryKind,
        task_family: str,
        content: str,
        conditions: dict[str, str],
        anchor: str,
    ) -> str:
        serialized_conditions = json.dumps(conditions, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha1(
            f"{memory_kind.value}|{task_family}|{anchor}|{serialized_conditions}|{content}".encode("utf-8")
        ).hexdigest()
        return f"rmem-{digest[:14]}"

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
                "importance": max(record.importance, self._clamp(importance, 0.0, 1.0)),
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
            importance=self._clamp(importance, 0.0, 1.0),
            recency_weight=self._clamp(recency_weight, 0.0, 1.5),
            tags=tags,
            project_key=project_key,
            workspace_id=workspace_id,
        )
        records.append(record)
        self._save_records(records)
        return record

    def remember_research(
        self,
        memory_kind: ResearchMemoryKind | str,
        content: str,
        *,
        task_family: str = "",
        proposal_ref: str = "",
        direction_ref: str = "",
        conditions: dict[str, Any] | None = None,
        evidence_summary: str = "",
        trajectory_summary: list[str] | None = None,
        uncertainty_note: str = "",
        confidence: float = 0.6,
        support_count: int = 1,
        contradiction_count: int = 0,
        source: str = "",
        source_stage: str = "",
        importance: float = 0.7,
        recency_weight: float = 1.0,
        tags: list[str] | None = None,
        project_key: str = "",
        workspace_id: str = "",
    ) -> ResearchMemoryRecord | None:
        if not self.enabled:
            return None
        content = (content or "").strip()
        if not content:
            return None
        memory_kind = ResearchMemoryKind(memory_kind)
        tags = self._normalize_tags(tags)
        conditions_norm = self._normalize_conditions(conditions)
        anchor = proposal_ref or direction_ref or task_family or content[:120]
        memory_id = self._make_research_memory_id(
            memory_kind,
            task_family,
            content[:400],
            conditions_norm,
            anchor,
        )
        records = self._load_research_records()
        now = datetime.now(timezone.utc)
        for index, record in enumerate(records):
            if record.memory_id != memory_id:
                continue
            merged_conditions = dict(record.conditions)
            merged_conditions.update(conditions_norm)
            updated = record.model_copy(update={
                "timestamp": now,
                "content": content if len(content) >= len(record.content) else record.content,
                "evidence_summary": evidence_summary or record.evidence_summary,
                "trajectory_summary": self._merge_trajectory(record.trajectory_summary, trajectory_summary),
                "uncertainty_note": uncertainty_note or record.uncertainty_note,
                "confidence": max(record.confidence, self._clamp(confidence, 0.0, 1.0)),
                "support_count": max(1, record.support_count + max(1, support_count)),
                "contradiction_count": max(0, record.contradiction_count + max(0, contradiction_count)),
                "importance": max(record.importance, self._clamp(importance, 0.0, 1.0)),
                "recency_weight": min(1.5, max(record.recency_weight, recency_weight)),
                "conditions": merged_conditions,
                "tags": self._normalize_tags(record.tags + tags),
                "source": source or record.source,
                "source_stage": source_stage or record.source_stage,
                "project_key": project_key or record.project_key,
                "workspace_id": workspace_id or record.workspace_id,
            })
            records[index] = updated
            self._save_research_records(records)
            return updated
        record = ResearchMemoryRecord(
            memory_id=memory_id,
            memory_kind=memory_kind,
            content=content,
            task_family=task_family,
            proposal_ref=proposal_ref,
            direction_ref=direction_ref,
            conditions=conditions_norm,
            evidence_summary=evidence_summary,
            trajectory_summary=self._merge_trajectory([], trajectory_summary),
            uncertainty_note=uncertainty_note,
            confidence=self._clamp(confidence, 0.0, 1.0),
            support_count=max(1, support_count),
            contradiction_count=max(0, contradiction_count),
            source=source,
            source_stage=source_stage,
            importance=self._clamp(importance, 0.0, 1.0),
            recency_weight=self._clamp(recency_weight, 0.0, 1.5),
            tags=tags,
            project_key=project_key,
            workspace_id=workspace_id,
        )
        records.append(record)
        self._save_research_records(records)
        return record

    def decay(self, *, project_key: str = "", amount: float | None = None) -> int:
        if not self.enabled:
            return 0
        amount = self.decay_factor if amount is None else max(0.0, amount)
        changed = 0

        records = self._load_records()
        updated_generic: list[MemoryRecord] = []
        for record in records:
            if project_key and record.project_key != project_key:
                updated_generic.append(record)
                continue
            new_weight = max(0.1, record.recency_weight - amount)
            if abs(new_weight - record.recency_weight) > 1e-6:
                changed += 1
            updated_generic.append(record.model_copy(update={"recency_weight": new_weight}))
        if changed and updated_generic:
            self._save_records(updated_generic)

        research_records = self._load_research_records()
        research_changed = 0
        updated_research: list[ResearchMemoryRecord] = []
        for record in research_records:
            if project_key and record.project_key != project_key:
                updated_research.append(record)
                continue
            new_weight = max(0.1, record.recency_weight - amount)
            if abs(new_weight - record.recency_weight) > 1e-6:
                research_changed += 1
            updated_research.append(record.model_copy(update={"recency_weight": new_weight}))
        if research_changed and updated_research:
            self._save_research_records(updated_research)
        return changed + research_changed

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

    def retrieve_research(
        self,
        task_type: str,
        *,
        topic: str = "",
        tags: list[str] | None = None,
        text: str = "",
        conditions: dict[str, Any] | None = None,
        project_key: str = "",
        top_k: int | None = None,
    ) -> list[ResearchMemoryRecord]:
        if not self.enabled:
            return []
        priors = _RESEARCH_KIND_PRIORS.get(task_type, _RESEARCH_KIND_PRIORS.get("planning", {}))
        query_tags = set(self._normalize_tags(tags))
        query_conditions = self._normalize_conditions(conditions)
        query_tokens = self._tokenize(
            " ".join(
                [
                    topic,
                    text,
                    " ".join(query_tags),
                    " ".join(f"{key} {value}" for key, value in query_conditions.items()),
                ]
            )
        )
        query_condition_keys = set(query_conditions)
        scored: list[tuple[float, ResearchMemoryRecord]] = []
        for record in self._load_research_records():
            if project_key and record.project_key and record.project_key != project_key:
                continue
            prior = priors.get(record.memory_kind, 0.75)
            content_tokens = self._tokenize(record.content)
            evidence_tokens = self._tokenize(record.evidence_summary)
            trajectory_tokens = self._tokenize(" ".join(record.trajectory_summary))
            condition_tokens = self._tokenize(" ".join(f"{key} {value}" for key, value in record.conditions.items()))
            token_overlap = len((content_tokens | evidence_tokens | trajectory_tokens) & query_tokens)
            condition_overlap = len(condition_tokens & query_tokens)
            tag_overlap = len(set(record.tags) & query_tags)
            exact_condition_matches = sum(
                1 for key, value in query_conditions.items()
                if record.conditions.get(key, "").lower() == value.lower()
            )
            missing_condition_keys = sum(1 for key in query_condition_keys if key not in record.conditions)
            mismatch_count = sum(
                1 for key, value in query_conditions.items()
                if key in record.conditions and record.conditions.get(key, "").lower() != value.lower()
            )
            project_bonus = 0.55 if project_key and record.project_key == project_key else 0.0
            support_bonus = min(record.support_count, 5) * 0.2
            contradiction_penalty = 0.35 * min(record.contradiction_count, 4)
            condition_bonus = 0.9 * exact_condition_matches + 0.75 * condition_overlap
            mismatch_penalty = 0.55 * mismatch_count + 0.15 * missing_condition_keys
            if record.memory_kind == ResearchMemoryKind.FAILED_DIRECTION:
                mismatch_penalty *= 1.35
                if exact_condition_matches:
                    condition_bonus += 0.45
            score = (
                prior * (1.15 + record.importance + record.recency_weight + record.confidence)
                + 0.45 * token_overlap
                + 0.7 * tag_overlap
                + condition_bonus
                + support_bonus
                + project_bonus
                - contradiction_penalty
                - mismatch_penalty
            )
            if score >= 1.45:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].support_count, item[1].timestamp), reverse=True)
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

    def render_research_context(
        self,
        task_type: str,
        *,
        topic: str = "",
        tags: list[str] | None = None,
        text: str = "",
        conditions: dict[str, Any] | None = None,
        project_key: str = "",
        top_k: int | None = None,
    ) -> str:
        records = self.retrieve_research(
            task_type,
            topic=topic,
            tags=tags,
            text=text,
            conditions=conditions,
            project_key=project_key,
            top_k=top_k,
        )
        if not records:
            return ""
        if task_type in {"literature", "planning"}:
            title = "DIRECTION MEMORY"
            instruction = (
                "Use these promising and failed direction summaries to prioritize feasible directions "
                "and avoid repeating directions that have already failed under similar conditions."
            )
        elif task_type == "experiment":
            title = "STRATEGY MEMORY"
            instruction = (
                "Use these experiment strategies to improve data handling, preflight validation, "
                "and training stability before making new implementation choices."
            )
        else:
            title = "RESEARCH MEMORY"
            instruction = "Use these evolved research memories when they are relevant to the current task."

        lines = []
        for record in records:
            source = f" [{record.source_stage or record.source}]" if (record.source_stage or record.source) else ""
            condition_bits = ", ".join(f"{key}={value}" for key, value in list(record.conditions.items())[:4])
            evidence = f" | evidence: {record.evidence_summary}" if record.evidence_summary else ""
            trajectory = f" | trajectory: {'; '.join(record.trajectory_summary[:2])}" if record.trajectory_summary else ""
            uncertainty = f" | uncertainty: {record.uncertainty_note}" if record.uncertainty_note else ""
            suffix = f" | conditions: {condition_bits}" if condition_bits else ""
            lines.append(f"- ({record.memory_kind.value}){source} {record.content}{suffix}{evidence}{trajectory}{uncertainty}")
        return (
            f"\n\n=== {title} ===\n"
            f"{instruction}\n"
            + "\n".join(lines)
            + f"\n=== END {title} ===\n"
        )
