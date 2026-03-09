# 21. Graceful Shutdown & Signal Handling

## Problem

The pipeline has **no signal handlers**. If the process receives SIGTERM/SIGINT:
- SLURM jobs are left running on the cluster (orphaned)
- Workspace manifest may be in inconsistent state (partial write)
- Temporary files are not cleaned up
- OpenAI/API client connections are not closed

## Solution

```python
"""nanoresearch/pipeline/shutdown.py"""
import signal
import asyncio
import logging
from typing import Optional

log = logging.getLogger("nanoresearch")


class GracefulShutdown:
    """Graceful shutdown manager for the pipeline.

    Registers signal handlers that set a flag and trigger cleanup.
    Agents should check `is_shutting_down` periodically.
    """

    def __init__(self):
        self._shutting_down = False
        self._cleanup_callbacks: list = []
        self._original_handlers: dict = {}

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def register(self):
        """Register signal handlers. Call once at pipeline start."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except (OSError, ValueError):
                pass  # Not all signals available on Windows

    def on_cleanup(self, callback):
        """Register a cleanup callback (sync or async)."""
        self._cleanup_callbacks.append(callback)

    def _handle_signal(self, signum, frame):
        if self._shutting_down:
            # Second signal: force exit
            log.warning("Forced shutdown (second signal)")
            raise SystemExit(1)
        self._shutting_down = True
        log.info(f"Shutdown signal received ({signal.Signals(signum).name}), "
                 f"cleaning up...")

    async def run_cleanup(self):
        """Run all registered cleanup callbacks."""
        for cb in reversed(self._cleanup_callbacks):
            try:
                result = cb()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                log.warning(f"Cleanup callback failed: {e}")

    def restore(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
```

### Integration

```python
# In orchestrator.py run():
shutdown = GracefulShutdown()
shutdown.register()

# Register cleanup: cancel SLURM jobs, close clients, persist memory
shutdown.on_cleanup(lambda: dispatcher.close())
shutdown.on_cleanup(lambda: memory.persist(workspace.path / "memory.json"))
shutdown.on_cleanup(lambda: workspace.update_manifest(
    current_stage=PipelineStage.FAILED.value))

try:
    for stage in stages:
        if shutdown.is_shutting_down:
            log.info("Shutdown requested, stopping pipeline")
            break
        # ... run stage ...
finally:
    await shutdown.run_cleanup()
    shutdown.restore()
```

---

# 22. Resource Cleanup & Connection Pool Safety

## Problem

- `ModelDispatcher._clients` dict grows unbounded — a new OpenAI client is created
  for each unique timeout value and never evicted.
- Subprocess instances from code execution may become zombies if parent crashes.
- No `async with` pattern for the pipeline orchestrator.

## Solution

### 22.1 Client Pool with Max Size

```python
# In multi_model.py, modify _get_client():

_MAX_CLIENTS = 5  # Maximum concurrent API clients

def _get_client(self, timeout: float) -> OpenAI:
    rounded = round(timeout / 10) * 10  # Round to nearest 10s
    if rounded not in self._clients:
        # Evict oldest if at capacity
        if len(self._clients) >= _MAX_CLIENTS:
            oldest_key = next(iter(self._clients))
            try:
                self._clients[oldest_key].close()
            except Exception:
                pass
            del self._clients[oldest_key]
        self._clients[rounded] = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=rounded,
        )
    return self._clients[rounded]
```

### 22.2 Subprocess Cleanup Guard

```python
# In execution agents, wrap subprocess calls:

async def _run_with_cleanup(cmd: list[str], timeout: int, cwd: Path) -> tuple[str, str, int]:
    """Run subprocess with guaranteed cleanup on cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd))
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout)
        return stdout.decode("utf-8", errors="replace"), \
               stderr.decode("utf-8", errors="replace"), \
               proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()  # Reap zombie
        raise
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
```

### 22.3 Orchestrator as Async Context Manager

```python
class PipelineOrchestrator:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.dispatcher.close()
        if self.memory:
            self.memory.persist(self.workspace.path / "memory.json")
        return False  # Don't suppress exceptions
```

---

# 23. Bare Exception Hardening

## Problem

20+ instances of `except Exception:` with `pass` or minimal handling. These swallow
real errors and make debugging extremely difficult.

## Rule for Developers

```
NEVER write `except Exception: pass`
ALWAYS at minimum: `except Exception as e: self.log(f"... failed: {e}")`
PREFER specific exceptions: `except (json.JSONDecodeError, KeyError) as e:`
```

## Specific Fixes Needed

| File | Line (approx) | Current | Fix |
|------|---------------|---------|-----|
| base.py:156 | `except Exception: pass` | `except Exception as e: log.debug(f"JSON escape fix failed: {e}")` |
| execution.py:4158 | `except Exception:` silent | Add `log.warning(f"Log stream error: {e}")` |
| execution.py:4317 | `except Exception:` silent | Add `log.warning(f"Metrics recovery: {e}")` |
| debug.py:156 | `except Exception:` in file read | `except (OSError, UnicodeDecodeError) as e:` |
| coding.py:354 | `except Exception:` in JSON parse | `except (json.JSONDecodeError, ValueError) as e:` |
| figure_gen.py:1730 | `except Exception:` on image close | `except (OSError, AttributeError) as e:` |

## General Pattern to Follow

```python
# BAD:
try:
    result = risky_operation()
except Exception:
    pass

# GOOD:
try:
    result = risky_operation()
except SpecificError as e:
    self.log(f"Operation failed (non-fatal): {e}")
    result = fallback_value
```

---

# 24. Concurrency Safety

## Problem

Several global singletons are initialized lazily without thread safety:

1. `ideation.py:41-72` — `_arxiv_search`, `_s2_search` etc. are global `None` variables
   initialized on first call. If two coroutines call simultaneously, both see `None` and
   both initialize.

2. `multi_model.py:44-56` — `_clients` dict is shared but has no lock.

## 24.1 Fix for Ideation Search Functions

The project already has `_lazy_lock = asyncio.Lock()` but it's only used for some
functions. Ensure ALL lazy init functions use it:

```python
# Verify every _ensure_*() function acquires _lazy_lock:
async def _ensure_all_search_functions():
    async with _lazy_lock:
        if _arxiv_search is not None:
            return  # Already initialized
        # ... initialize all at once ...
```

## 24.2 Fix for Client Dict

```python
import threading

class ModelDispatcher:
    def __init__(self, config):
        self._clients = {}
        self._client_lock = threading.Lock()

    def _get_client(self, timeout: float) -> OpenAI:
        rounded = round(timeout / 10) * 10
        with self._client_lock:
            if rounded not in self._clients:
                self._clients[rounded] = OpenAI(...)
            return self._clients[rounded]
```

---

# 25. Checkpoint Transactionality

## Problem

- Workspace manifest is written with atomic rename (good), but stage output JSON files
  are written directly — if crash happens mid-write, the JSON is corrupted.
- On resume, the orchestrator loads the output file without validating it's complete JSON.

## Solution

```python
# In workspace.py write_json(), use same atomic pattern as _write_manifest:

def write_json(self, subpath: str, data: dict):
    """Atomically write JSON to workspace."""
    target = self.path / subpath
    target.parent.mkdir(parents=True, exist_ok=True)
    import tempfile, os
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(target))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

```python
# In orchestrator.py, validate loaded output:

def _load_stage_output(self, stage: PipelineStage) -> dict:
    """Load and validate stage output JSON."""
    path = self._output_file_path(stage)
    if not path.exists():
        raise StageError(f"Output file missing for {stage.name}: {path}")
    try:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        raise StageError(
            f"Corrupted output for {stage.name}: {e}. "
            f"Delete {path} and re-run this stage."
        )
```

---

# 26. SLURM Edge Cases

## Problem

Several SLURM integration edge cases are unhandled:

1. Job enters `COMPLETING` state (between RUNNING and COMPLETED) — not recognized
2. `sacct` query can hang indefinitely — no timeout
3. Job ID file (`active_job_id.txt`) corruption → can't recover job state
4. `scancel` is fire-and-forget — no verification that job actually stopped

## Solution

```python
# 1. Add COMPLETING to recognized states:
SLURM_RUNNING_STATES = {"RUNNING", "COMPLETING", "REQUEUED", "RESIZING"}
SLURM_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
                          "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}

# 2. Add timeout to sacct:
async def _get_job_status(job_id: str, timeout: int = 30) -> str:
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "sacct", "-j", job_id, "--format=State", "--noheader",
                "--parsable2", stdout=asyncio.subprocess.PIPE),
            timeout=timeout)
        stdout, _ = await proc.communicate()
        status = stdout.decode().strip().split("\n")[0].strip()
        return status if status else "UNKNOWN"
    except asyncio.TimeoutError:
        return "UNKNOWN"

# 3. Validate job ID on load:
def _read_job_id(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text("utf-8").strip()
    if not text.isdigit():
        log.warning(f"Corrupted job ID file: {path} contains '{text}'")
        return None
    return text

# 4. Verify scancel:
async def _cancel_job(job_id: str, timeout: int = 15) -> bool:
    proc = await asyncio.create_subprocess_exec("scancel", job_id)
    await asyncio.wait_for(proc.wait(), timeout=timeout)
    # Verify
    await asyncio.sleep(2)
    status = await _get_job_status(job_id)
    if status in SLURM_TERMINAL_STATES or status == "UNKNOWN":
        return True
    log.warning(f"Job {job_id} still in state {status} after scancel")
    return False
```
