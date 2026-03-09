# 11. P2: Cost Tracking

## Problem

Users have no visibility into API costs per run.

## Solution

Modify `ModelDispatcher.generate()` to return usage data:

```python
# In multi_model.py, modify generate() to return a richer result:

@dataclass
class LLMResult:
    content: str
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    model: str
    latency_ms: int

# In generate():
start = time.monotonic()
response = client.chat.completions.create(...)
latency_ms = int((time.monotonic() - start) * 1000)

usage = {}
if hasattr(response, 'usage') and response.usage:
    usage = {
        "prompt_tokens": response.usage.prompt_tokens or 0,
        "completion_tokens": response.usage.completion_tokens or 0,
        "total_tokens": response.usage.total_tokens or 0,
    }

return LLMResult(
    content=response.choices[0].message.content or "",
    usage=usage,
    model=config.model,
    latency_ms=latency_ms,
)
```

> **IMPORTANT**: This is a breaking change to `generate()` return type.
> Migration: All callers currently expect `str`. Change them to use `result.content`.
> Do this file by file, not all at once. Consider keeping `generate()` returning `str`
> and adding a separate `generate_with_usage()` method to avoid breaking changes.

### Cost Aggregation in Orchestrator

```python
# In orchestrator, accumulate per-stage costs:
stage_costs = {}
# After each stage:
stage_costs[stage.name] = {
    "total_tokens": accumulated_tokens,
    "num_calls": call_count,
    "total_latency_ms": accumulated_latency,
}
# Save to manifest:
workspace.update_manifest(cost_tracking=stage_costs)
```

---

# 12. P2: Constants Centralization

Create `nanoresearch/constants.py`:

```python
"""Centralized constants for the NanoResearch pipeline.

All magic numbers should be defined here. Import from this module,
never hardcode numeric literals in agent code.
"""

# === Literature Search ===
TARGET_CITATION_COUNT = 50
MIN_HIGH_CITED_PAPERS = 10
SNOWBALL_MAX_NEW_PAPERS = 15
SNOWBALL_TOP_K = 5
SEARCH_COVERAGE_THRESHOLD = 8  # out of 10
MAX_SUPPLEMENTARY_SEARCH_ROUNDS = 2

# === Code Generation ===
MAX_IMPORT_FIX_RETRIES = 2
MAX_CODE_GEN_RETRIES = 3
MAX_REFERENCE_REPOS = 3

# === Execution ===
MAX_DEBUG_ROUNDS = 20
QUICK_EVAL_TIMEOUT_S = 1200
DRY_RUN_TIMEOUT_S = 60
SUBPROCESS_OUTPUT_LIMIT = 5000

# === Writing ===
MAX_SECTION_CONTEXT_CHARS = 15000
MAX_LATEX_FIX_ROUNDS = 5
MAX_CONTRIBUTION_ITEMS = 3

# === Review ===
MIN_ACCEPTABLE_SECTION_SCORE = 8.0
MAX_REVISION_ROUNDS = 5
CONVERGENCE_THRESHOLD = 0.3  # stop if improvement < this

# === Analysis ===
MAX_ANALYSIS_FIGURES = 5

# === Figure Generation ===
MAX_IMAGE_RETRIES = 2
MAX_CODE_CHART_RETRIES = 3

# === API ===
MAX_API_RETRIES = 5
RETRY_BASE_DELAY_S = 3.0
RETRY_BACKOFF_FACTOR = 2.0

# === Context Management ===
TOOL_RESULT_MAX_CHARS = 6000
TOOL_RESULT_HEAD_CHARS = 2000
TOOL_RESULT_TAIL_CHARS = 1500
CONTEXT_COMPACTION_THRESHOLD = 100000
PROTECTED_TAIL_TURNS = 6
COMPACTED_PREVIEW_CHARS = 400

# === Metrics ===
LOWER_IS_BETTER_PATTERNS = frozenset({
    "loss", "error", "perplexity", "cer", "wer", "fer",
    "mae", "mse", "rmse", "mape", "fid", "kid", "ece",
    "latency", "inference_time",
})
```

### Migration

For each constant:
1. Add to `constants.py`
2. Replace the hardcoded value in the agent file with an import
3. Verify the value matches the original exactly
4. Run tests

---

# 13. P2: DAG Parallel Stage Scheduling

## Problem

All 9 stages run strictly in serial. But some stages have no dependencies between them.

## Solution

Define a dependency DAG and run independent stages in parallel:

```python
# In orchestrator.py:

STAGE_DEPENDENCIES = {
    "IDEATION": [],
    "PLANNING": ["IDEATION"],
    "SETUP": ["PLANNING"],
    "CODING": ["SETUP", "PLANNING"],
    "EXECUTION": ["CODING"],
    "ANALYSIS": ["EXECUTION"],
    "FIGURE_GEN": ["ANALYSIS"],
    "WRITING": ["ANALYSIS", "FIGURE_GEN"],
    "REVIEW": ["WRITING"],
}

async def _run_dag(self):
    """Run stages respecting dependencies, parallelizing where possible."""
    completed = set()
    results = {}

    while True:
        # Find stages that can run now
        runnable = []
        for stage in self._processing_stages():
            if stage.name in completed:
                continue
            deps = STAGE_DEPENDENCIES.get(stage.name, [])
            if all(d in completed for d in deps):
                runnable.append(stage)

        if not runnable:
            break  # All done or deadlocked

        if len(runnable) == 1:
            # Single stage: run directly (saves overhead)
            result = await self._run_stage(runnable[0], results)
            results[runnable[0].name] = result
            completed.add(runnable[0].name)
        else:
            # Multiple independent stages: run in parallel
            tasks = [self._run_stage(s, results) for s in runnable]
            stage_results = await asyncio.gather(*tasks, return_exceptions=True)
            for stage, result in zip(runnable, stage_results):
                if isinstance(result, Exception):
                    self.log(f"Stage {stage.name} failed: {result}")
                    raise result
                results[stage.name] = result
                completed.add(stage.name)
```

> **Safety**: This is an opt-in feature. Add `parallel_stages: true` to config.
> Default: `false` (keeps existing serial behavior).

---

# 14. P2: Progress Streaming

## Problem

Users wait 30-60 minutes with no feedback beyond log files.

## Solution

Add a `ProgressEmitter` that writes to a JSON file, updated in real-time:

```python
"""nanoresearch/pipeline/progress.py"""
import json
import time
from pathlib import Path
from typing import Optional


class ProgressEmitter:
    """Emit progress events to a JSON file for UI consumption."""

    def __init__(self, progress_path: Path):
        self.path = progress_path
        self._events: list[dict] = []
        self._current_stage: Optional[str] = None
        self._start_time = time.time()

    def stage_start(self, stage: str, total_stages: int, current_index: int):
        self._current_stage = stage
        self._emit({
            "type": "stage_start",
            "stage": stage,
            "progress_pct": round(current_index / total_stages * 100),
            "message": f"Starting {stage}...",
        })

    def stage_progress(self, message: str, detail: str = ""):
        self._emit({
            "type": "stage_progress",
            "stage": self._current_stage,
            "message": message,
            "detail": detail,
        })

    def stage_complete(self, stage: str, summary: str = ""):
        self._emit({
            "type": "stage_complete",
            "stage": stage,
            "message": f"{stage} completed",
            "summary": summary,
        })

    def pipeline_complete(self, success: bool, summary: str = ""):
        self._emit({
            "type": "pipeline_complete",
            "success": success,
            "total_time_s": round(time.time() - self._start_time),
            "summary": summary,
        })

    def _emit(self, event: dict):
        event["timestamp"] = time.time()
        self._events.append(event)
        # Atomic write
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({
            "events": self._events[-50:],  # Keep last 50 events
            "current": event,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
```

### Integration

```python
# In orchestrator.py:
progress = ProgressEmitter(workspace.path / "progress.json")

for i, stage in enumerate(stages):
    progress.stage_start(stage.name, len(stages), i)
    result = await self._run_stage(stage, ...)
    progress.stage_complete(stage.name, f"Generated {len(result)} items")

progress.pipeline_complete(True, "Paper generated successfully")
```

---

# 15. P3: Structured Logging

Replace ad-hoc `self.log()` with structured logging:

```python
"""nanoresearch/logging_config.py"""
import logging
import json
import sys


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "session_id": getattr(record, "session_id", None),
            "message": record.getMessage(),
        }
        # Add any extra fields
        for key in ("model", "tokens", "latency_ms", "error_type"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(log_path=None, level=logging.INFO):
    logger = logging.getLogger("nanoresearch")
    logger.setLevel(level)

    # Console handler (human-readable)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    # File handler (structured JSON)
    if log_path:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger
```
