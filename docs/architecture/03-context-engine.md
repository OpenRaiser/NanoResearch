# 5. P0: Context Engine (OpenClaw-Inspired)

## What OpenClaw Does

OpenClaw implements a **pluggable ContextEngine interface** with lifecycle hooks:

```
bootstrap → ingest → assemble → compact → afterTurn
                                           ↓
                                   prepareSubagentSpawn → onSubagentEnded
```

**Key files in OpenClaw** (cloned to `D:/openclaw`):
- `src/context-engine/types.ts` — ContextEngine interface (168 lines)
- `src/context-engine/registry.ts` — Factory registry + slot resolution (86 lines)
- `src/context-engine/legacy.ts` — Backward-compatible wrapper (117 lines)
- `src/memory/hybrid.ts` — Hybrid vector+keyword search with temporal decay (156 lines)
- `src/memory/temporal-decay.ts` — Exponential decay scoring (168 lines)

**Key design insights**:
1. **Pluggable slot**: Only ONE context engine active at a time (exclusive slot pattern)
2. **LegacyEngine as fallback**: Always registered, preserves existing behavior
3. **Lifecycle hooks**: `bootstrap` (init), `ingest` (per-message), `assemble` (pre-LLM), `afterTurn` (post-LLM), `compact` (overflow)
4. **Hybrid search**: `score = vectorWeight * vectorScore + textWeight * textScore` (default 0.7/0.3)
5. **Temporal decay**: `multiplier = e^(-ln2/halfLifeDays * ageInDays)` — older memories score lower
6. **MMR diversity**: Maximal Marginal Relevance re-ranking prevents redundant retrieval

## 5.1 What NanoResearch Should Borrow

NanoResearch is NOT a chatbot — it's a pipeline. The direct OpenClaw model doesn't apply.
But the **concepts** are transformative for our use case:

### Concept 1: Cross-Stage Memory with Decay

Currently each stage gets its inputs from the previous stage's JSON output. There's no
"memory" across pipeline runs or across stages within a run beyond what's explicitly
passed in the stage input dict.

**Problem examples**:
- REVIEW doesn't know which specific experiments failed in EXECUTION (only gets final metrics)
- WRITING doesn't remember which search queries found the best papers (only gets papers)
- Across runs: the system doesn't remember "last time topic X failed because of Y"

**Solution**: Implement a `ResearchMemory` class:

```python
"""nanoresearch/memory/research_memory.py"""
import json
import hashlib
import math
import time
from pathlib import Path
from typing import Optional


class MemoryEntry:
    """A single memory unit with temporal decay."""
    __slots__ = ("key", "content", "source_stage", "created_at",
                 "access_count", "tags", "importance")

    def __init__(self, key: str, content: str, source_stage: str,
                 tags: list[str] = None, importance: float = 0.5):
        self.key = key
        self.content = content
        self.source_stage = source_stage
        self.created_at = time.time()
        self.access_count = 0
        self.tags = tags or []
        self.importance = importance  # 0.0 to 1.0


class ResearchMemory:
    """Cross-stage memory with hybrid search and temporal decay.

    Inspired by OpenClaw's ContextEngine, adapted for pipeline use.

    Lifecycle:
        1. bootstrap(session_id) — load from disk on resume
        2. ingest(entry) — store after each stage completes
        3. query(query, tags, budget) — retrieve relevant memories
        4. compact(max_entries) — prune old/low-value entries
        5. persist(path) — save to disk

    Usage in pipeline:
        memory = ResearchMemory()
        memory.bootstrap(session_id)

        # After IDEATION:
        memory.ingest(MemoryEntry(
            key="search_strategy",
            content="OpenAlex returned best results for 'transformer attention'",
            source_stage="IDEATION",
            tags=["search", "strategy"],
            importance=0.7
        ))

        # In WRITING, retrieve relevant memories:
        relevant = memory.query("writing method section", tags=["method"],
                                budget=5)
    """
    HALF_LIFE_HOURS = 24.0  # Memories decay with 24h half-life within a run
    CROSS_RUN_HALF_LIFE_DAYS = 30.0  # Cross-run memories decay with 30-day half-life

    def __init__(self):
        self._entries: dict[str, MemoryEntry] = {}
        self._session_id: Optional[str] = None

    def bootstrap(self, session_id: str, persist_path: Optional[Path] = None):
        """Load existing memories from disk."""
        self._session_id = session_id
        if persist_path and persist_path.exists():
            try:
                data = json.loads(persist_path.read_text("utf-8"))
                for item in data.get("entries", []):
                    entry = MemoryEntry(
                        key=item["key"],
                        content=item["content"],
                        source_stage=item["source_stage"],
                        tags=item.get("tags", []),
                        importance=item.get("importance", 0.5),
                    )
                    entry.created_at = item.get("created_at", time.time())
                    entry.access_count = item.get("access_count", 0)
                    self._entries[entry.key] = entry
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # Start fresh on corruption

    def ingest(self, entry: MemoryEntry):
        """Store a memory entry. Overwrites if key exists."""
        self._entries[entry.key] = entry

    def query(self, query: str, tags: list[str] = None,
              budget: int = 10, stage_filter: str = None) -> list[dict]:
        """Retrieve relevant memories with temporal decay scoring.

        NOTE: This is a read-only operation. access_count is tracked
        separately to avoid side effects during queries.

        Args:
            query: Natural language query (matched against content via word overlap).
            tags: Filter to entries with at least one matching tag.
            budget: Maximum number of results.
            stage_filter: Only return entries from this stage.

        Returns:
            List of {"key", "content", "score", "source_stage"} sorted by score.
        """
        now = time.time()
        query_words = set(query.lower().split())

        scored = []
        for entry in self._entries.values():
            # Tag filter
            if tags and not set(tags) & set(entry.tags):
                continue
            # Stage filter
            if stage_filter and entry.source_stage != stage_filter:
                continue

            # Word overlap score (simple BM25-like)
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            if not query_words:
                text_score = 0.0
            else:
                text_score = overlap / len(query_words)

            # Temporal decay
            age_hours = (now - entry.created_at) / 3600.0
            decay = math.exp(-math.log(2) / self.HALF_LIFE_HOURS * age_hours)

            # Importance boost
            importance_boost = 0.5 + 0.5 * entry.importance

            # Combined score
            score = text_score * decay * importance_boost

            if score > 0.01:
                scored.append({
                    "key": entry.key,
                    "content": entry.content,
                    "score": round(score, 4),
                    "source_stage": entry.source_stage,
                    "tags": entry.tags,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:budget]

    def compact(self, max_entries: int = 200):
        """Prune lowest-scored entries to stay under budget."""
        if len(self._entries) <= max_entries:
            return
        now = time.time()
        scored = []
        for key, entry in self._entries.items():
            age_hours = (now - entry.created_at) / 3600.0
            decay = math.exp(-math.log(2) / self.HALF_LIFE_HOURS * age_hours)
            score = entry.importance * decay * (1 + entry.access_count * 0.1)
            scored.append((key, score))
        scored.sort(key=lambda x: x[1])
        # Remove lowest-scored entries
        to_remove = len(self._entries) - max_entries
        for key, _ in scored[:to_remove]:
            del self._entries[key]

    def persist(self, path: Path):
        """Save all memories to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self._session_id,
            "entry_count": len(self._entries),
            "entries": [
                {
                    "key": e.key,
                    "content": e.content,
                    "source_stage": e.source_stage,
                    "created_at": e.created_at,
                    "access_count": e.access_count,
                    "tags": e.tags,
                    "importance": e.importance,
                }
                for e in self._entries.values()
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_context_for_stage(self, stage: str, budget: int = 10) -> str:
        """Build a context string for a specific stage from memories.

        This replaces the ad-hoc context passing between stages.
        """
        # Stage-specific memory queries
        stage_queries = {
            "WRITING": "key findings experiment results method design",
            "REVIEW": "paper structure claims evidence figures",
            "FIGURE_GEN": "experiment results training metrics ablation",
            "ANALYSIS": "execution status metrics errors debug",
        }
        query = stage_queries.get(stage, stage.lower())
        results = self.query(query, budget=budget)
        if not results:
            return ""
        lines = [f"[Memory from {r['source_stage']}] {r['content']}" for r in results]
        return "\n".join(lines)
```

### Concept 2: Cross-Run Learning

```python
"""nanoresearch/memory/cross_run_memory.py"""
import json
import math
import time
from pathlib import Path


class CrossRunMemory:
    """Persistent memory across pipeline runs.

    Stored at ~/.nanobot/memory/cross_run.json

    Examples of what gets stored:
    - "Topic 'attention mechanisms' failed at EXECUTION because PyTorch 2.x
       changed the autograd API" → future runs can preempt this
    - "OpenAlex returns better results than S2 for NLP topics" → search strategy
    - "User's cluster needs 'module load cuda/12.1' before training" → env setup
    """
    PERSIST_PATH = Path.home() / ".nanobot" / "memory" / "cross_run.json"

    def __init__(self):
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        if self.PERSIST_PATH.exists():
            try:
                data = json.loads(self.PERSIST_PATH.read_text("utf-8"))
                self._entries = data.get("entries", [])
            except (json.JSONDecodeError, OSError):
                self._entries = []

    def record_outcome(self, topic: str, stage: str, success: bool,
                       lesson: str, tags: list[str] = None):
        """Record a lesson learned from a pipeline run."""
        self._entries.append({
            "topic": topic,
            "stage": stage,
            "success": success,
            "lesson": lesson,
            "tags": tags or [],
            "timestamp": time.time(),
        })
        self._persist()

    def get_lessons(self, topic: str, stage: str, limit: int = 5) -> list[str]:
        """Retrieve relevant lessons for a given topic and stage."""
        now = time.time()
        scored = []
        topic_words = set(topic.lower().split())
        for entry in self._entries:
            entry_words = set(entry.get("topic", "").lower().split())
            overlap = len(topic_words & entry_words) / max(len(topic_words), 1)

            # Temporal decay (30-day half-life)
            age_days = (now - entry.get("timestamp", now)) / 86400.0
            decay = math.exp(-math.log(2) / 30.0 * age_days)

            # Stage match boost
            stage_match = 1.5 if entry.get("stage") == stage else 1.0

            score = overlap * decay * stage_match
            if score > 0.01:
                scored.append((entry["lesson"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [lesson for lesson, _ in scored[:limit]]

    def _persist(self):
        self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Keep only last 500 entries
        if len(self._entries) > 500:
            self._entries = self._entries[-500:]
        data = {"entries": self._entries, "count": len(self._entries)}
        self.PERSIST_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

### Concept 3: Pluggable Context Assembly (inspired by OpenClaw slots)

```python
"""nanoresearch/context_engine/interface.py"""
from abc import ABC, abstractmethod


class ContextEngine(ABC):
    """Pluggable context engine interface (inspired by OpenClaw).

    Only ONE engine is active per pipeline run (exclusive slot pattern).
    Default: LegacyContextEngine (current behavior).
    """

    @abstractmethod
    async def assemble(self, stage: str, base_context: dict,
                       budget_chars: int = 15000) -> str:
        """Assemble context for an LLM call within a character budget.

        Args:
            stage: Current pipeline stage name.
            base_context: Dict of available context blocks.
            budget_chars: Maximum total characters.

        Returns:
            Assembled context string.
        """
        ...

    @abstractmethod
    async def compact(self, messages: list[dict],
                      budget_chars: int = 100000) -> list[dict]:
        """Compact message history to fit within budget.

        Args:
            messages: Current conversation messages.
            budget_chars: Maximum total characters.

        Returns:
            Compacted message list.
        """
        ...

    async def after_stage(self, stage: str, output: dict):
        """Hook called after each stage completes. Override to persist state."""
        pass

    async def dispose(self):
        """Cleanup resources."""
        pass


class LegacyContextEngine(ContextEngine):
    """Backward-compatible engine wrapping existing behavior.

    NOTE: assemble() iterates dict items in insertion order (Python 3.7+).
    If priority matters, ensure base_context is an OrderedDict or the caller
    inserts keys in priority order.
    """

    async def assemble(self, stage, base_context, budget_chars=15000):
        # Existing per-section context builder logic
        # This preserves 100% backward compatibility
        blocks = []
        total = 0
        for key, value in base_context.items():
            text = str(value) if not isinstance(value, str) else value
            if total + len(text) > budget_chars:
                remaining = budget_chars - total
                if remaining > 200:
                    blocks.append(f"[{key}] {text[:remaining]}...[truncated]")
                break
            blocks.append(f"[{key}] {text}")
            total += len(text)
        return "\n\n".join(blocks)

    async def compact(self, messages, budget_chars=100000):
        # Existing head/tail compaction from base.py
        total = sum(len(str(m)) for m in messages)
        if total <= budget_chars:
            return messages
        # Keep first 2 + last 6 messages, trim middle
        if len(messages) <= 8:
            return messages
        head = messages[:2]
        tail = messages[-6:]
        return head + [{"role": "system",
                        "content": f"[{len(messages)-8} messages compacted]"}] + tail


class SmartContextEngine(ContextEngine):
    """Advanced engine with memory-aware context assembly.

    Uses ResearchMemory to inject relevant cross-stage context.
    Uses priority scoring to select most important blocks.

    NOTE: Requires ResearchMemory and MemoryEntry from
    nanoresearch.memory.research_memory to be imported by the caller.
    """

    def __init__(self, memory: 'ResearchMemory'):
        self.memory = memory

    async def assemble(self, stage, base_context, budget_chars=15000):
        # 1. Get memory-based context additions
        memory_context = self.memory.get_context_for_stage(stage, budget=5)

        # 2. Priority-score each context block
        priorities = self._score_blocks(stage, base_context)

        # 3. Greedily select blocks by priority within budget
        selected = []
        remaining = budget_chars
        if memory_context:
            mem_block = f"[Cross-Stage Memory]\n{memory_context}"
            if len(mem_block) < remaining * 0.2:  # Max 20% budget for memory
                selected.append(mem_block)
                remaining -= len(mem_block)

        for key, priority in sorted(priorities, key=lambda x: x[1], reverse=True):
            text = str(base_context[key])
            if len(text) <= remaining:
                selected.append(f"[{key}]\n{text}")
                remaining -= len(text)
            elif remaining > 500:
                selected.append(f"[{key}]\n{text[:remaining-50]}...[truncated]")
                remaining = 0
                break

        return "\n\n".join(selected)

    async def compact(self, messages, budget_chars=100000):
        # Same as Legacy for now; can be enhanced with summarization later
        total = sum(len(str(m)) for m in messages)
        if total <= budget_chars:
            return messages
        if len(messages) <= 8:
            return messages
        head = messages[:2]
        tail = messages[-6:]
        return head + [{"role": "system",
                        "content": f"[{len(messages)-8} messages compacted]"}] + tail

    async def after_stage(self, stage, output):
        """Auto-extract key facts from stage output into memory.

        NOTE: Uses MemoryEntry from nanoresearch.memory.research_memory.
        The caller must ensure this import is available.
        """
        from nanoresearch.memory.research_memory import MemoryEntry

        if isinstance(output, dict):
            if "error" in output:
                self.memory.ingest(MemoryEntry(
                    key=f"{stage}_error",
                    content=f"Stage {stage} encountered: {str(output['error'])[:500]}",
                    source_stage=stage,
                    tags=["error", "debug"],
                    importance=0.8,
                ))
            if "key_findings" in output:
                findings = output["key_findings"]
                if isinstance(findings, list):
                    for i, f in enumerate(findings[:5]):
                        self.memory.ingest(MemoryEntry(
                            key=f"{stage}_finding_{i}",
                            content=str(f)[:500],
                            source_stage=stage,
                            tags=["finding"],
                            importance=0.7,
                        ))

    def _score_blocks(self, stage: str, blocks: dict) -> list[tuple[str, float]]:
        """Score context blocks by relevance to current stage."""
        # Stage-specific relevance weights
        relevance = {
            "WRITING": {
                "hypotheses": 0.9, "evidence": 0.9, "method": 0.8,
                "results": 1.0, "papers": 0.5, "figures": 0.7,
            },
            "REVIEW": {
                "claims": 0.9, "evidence": 0.8, "results": 0.9,
                "papers": 0.6, "method": 0.7,
            },
            "FIGURE_GEN": {
                "results": 1.0, "training_log": 0.9, "ablation": 0.8,
                "method": 0.5,
            },
        }
        stage_weights = relevance.get(stage, {})
        scored = []
        for key in blocks:
            weight = stage_weights.get(key, 0.5)
            scored.append((key, weight))
        return scored
```

## 5.2 Integration into Orchestrator

```python
# In orchestrator.py run():

from nanoresearch.memory.research_memory import ResearchMemory
from nanoresearch.context_engine.interface import SmartContextEngine

memory = ResearchMemory()
memory.bootstrap(session_id, workspace.path / "memory.json")
context_engine = SmartContextEngine(memory)

for stage in stages:
    # ... existing stage execution ...
    result = await agent.run(**inputs)

    # NEW: post-stage memory hook
    await context_engine.after_stage(stage.name, result)

    # ... existing output saving ...

# At end of pipeline:
memory.persist(workspace.path / "memory.json")
```

## 5.3 Cross-Run Integration

```python
# In orchestrator.py, at pipeline start:
from nanoresearch.memory.cross_run_memory import CrossRunMemory
cross_run = CrossRunMemory()

# Get lessons from previous runs
lessons = cross_run.get_lessons(topic, "EXECUTION")
if lessons:
    log.info(f"Lessons from previous runs: {lessons}")

# At pipeline end:
cross_run.record_outcome(
    topic=topic,
    stage=final_stage,
    success=pipeline_success,
    lesson=f"Pipeline {'completed' if pipeline_success else 'failed at ' + failed_stage}",
    tags=["pipeline_outcome"],
)
```

## OpenClaw Reference Files

The OpenClaw source code is cloned to `D:/openclaw` for reference. Key files:

| File | What to Study |
|------|---------------|
| `src/context-engine/types.ts` | ContextEngine interface contract |
| `src/context-engine/registry.ts` | Factory registration + slot resolution pattern |
| `src/context-engine/legacy.ts` | Backward-compatible wrapper pattern |
| `src/memory/hybrid.ts` | Hybrid vector+keyword merge algorithm |
| `src/memory/temporal-decay.ts` | Exponential decay scoring formula |
| `src/memory/mmr.ts` | MMR diversity re-ranking |
| `src/plugins/slots.ts` | Exclusive plugin slot pattern |

**Key formulas from OpenClaw**:
- Hybrid score: `score = vectorWeight * vectorScore + textWeight * textScore` (0.7/0.3 default)
- Temporal decay: `multiplier = e^(-ln2/halfLifeDays * ageInDays)` (30-day half-life default)
- BM25 rank to score: `score = relevance / (1 + relevance)` where `relevance = -rank`
