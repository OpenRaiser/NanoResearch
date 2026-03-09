# 6. P1: File Splitting & Module Restructuring

## 6.1 execution.py (4749 lines → 7 files)

```
agents/execution/
├── __init__.py              # ExecutionAgent.run() entry point
├── local_runner.py          # Local subprocess execution
├── cluster_runner.py        # SLURM/cluster execution
├── debug_loop.py           # 20-round iterative debug
├── resource_matcher.py     # GPU/CPU resource matching
├── repair_strategies.py    # Error repair strategy registry
└── result_collector.py     # Metrics collection & normalization
```

**How to split**: Start with `result_collector.py` — extract `_collect_metrics()`,
`_parse_stdout_metrics()`, and `_normalize_metrics()`. These have no dependencies on
agent state and are pure functions.

## 6.2 writing.py (3578 lines → 7 files)

```
agents/writing/
├── __init__.py             # WritingAgent.run() entry point
├── context_builder.py      # Per-section context builders
├── grounding.py            # Evidence grounding & citation placement
├── section_writer.py       # Section-by-section generation
├── table_builder.py        # Comparison tables
├── citation_manager.py     # BibTeX assembly & dedup
└── latex_assembler.py      # Final LaTeX document assembly
```

## 6.3 experiment.py (3346 lines → 4 files)

```
agents/experiment/
├── __init__.py             # Entry point, mode dispatcher
├── pipeline_mode.py        # Structured phase execution
├── react_mode.py           # ReAct tool-use loop
└── edit_apply.py           # File edit utilities
```

---

# 7. P1: Shared LaTeX Fixer Module

## Problem

LaTeX fix logic is duplicated between `writing.py` and `review.py`:
- Level 1 (deterministic fixes) ~200 lines each
- Level 2 (LLM search-replace) ~100 lines each

Bugs in one copy don't get fixed in the other. The fix for `_ensure_packages` duplication
was already discovered during the V1 audit.

## Solution

Create `nanoresearch/latex/fixer.py` as the single source of truth:

```python
"""nanoresearch/latex/fixer.py — Shared 2-level LaTeX fix pipeline.

Level 1: Deterministic fixes (Unicode, packages, preamble, env matching).
Level 2: LLM search-replace using JSON [{old, new}] format.

Both writing.py and review.py should import from this module.
"""
import re
import shutil
from pathlib import Path
from typing import Optional, Callable, Awaitable

UNICODE_REPLACEMENTS = {
    '\u2013': '--',
    '\u2014': '---',
    '\u2018': '`',
    '\u2019': "'",
    '\u201c': '``',
    '\u201d': "''",
    '\u00e9': "\\'e",
    '\u00e8': "\\`e",
    '\u00f6': '\\"o',
    '\u00fc': '\\"u',
    '\u00e4': '\\"a',
    # Add more as encountered
}

REQUIRED_PACKAGES = [
    "inputenc", "fontenc", "amsmath", "amssymb", "graphicx",
    "booktabs", "hyperref", "natbib", "placeins",
]


def deterministic_fix(tex: str) -> str:
    """Level 1: Apply all deterministic fixes. No LLM needed.

    Safe to call multiple times (idempotent).
    """
    # 1. Remove junk before \documentclass
    dc_match = re.search(r'\\documentclass', tex)
    if dc_match and dc_match.start() > 0:
        tex = tex[dc_match.start():]

    # 2. Unicode replacements
    for char, replacement in UNICODE_REPLACEMENTS.items():
        tex = tex.replace(char, replacement)

    # 3. Remove control characters (except newline, tab)
    tex = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', tex)

    # 4. Fix @{} doubling at table line ends
    tex = re.sub(r'@\{\}(\s*\\\\)', r'\1', tex)

    # 5. Ensure required packages
    tex = _ensure_packages(tex)

    # 6. Fix mismatched \begin{}/\end{} pairs
    tex = _fix_mismatched_envs(tex)

    # 7. Ensure \end{document} exists
    if '\\end{document}' not in tex:
        tex = tex.rstrip() + '\n\\end{document}\n'

    return tex


async def llm_search_replace_fix(
    tex: str,
    error_log: str,
    llm_call: Callable[[str, str], Awaitable[str]],
    max_rounds: int = 3,
) -> tuple[str, bool]:
    """Level 2: LLM-driven search-replace fix.

    Args:
        tex: LaTeX source with compilation errors.
        error_log: tectonic/pdflatex error output.
        llm_call: async function(system, user) -> str (raw LLM response).
        max_rounds: Maximum fix attempts.

    Returns:
        (fixed_tex, success) tuple. success=False if no fixes were applied.
    """
    seen_errors = set()
    any_applied = False

    for round_num in range(max_rounds):
        # Detect infinite loop
        error_sig = _error_signature(error_log)
        if error_sig in seen_errors:
            break
        seen_errors.add(error_sig)

        system = (
            "You are a LaTeX debugging expert. Given compilation errors, "
            "return a JSON array of search-replace fixes.\n"
            "Format: [{\"old\": \"exact text to find\", \"new\": \"replacement text\"}]\n"
            "Rules:\n"
            "- old must be an EXACT substring of the LaTeX source\n"
            "- Each fix should be minimal (change as little as possible)\n"
            "- Return [] if you cannot determine a fix"
        )
        user = (
            f"LaTeX compilation error:\n```\n{error_log[-3000:]}\n```\n\n"
            f"LaTeX source (first 8000 chars):\n```\n{tex[:8000]}\n```"
        )

        try:
            raw = await llm_call(system, user)
            fixes = _parse_fixes_json(raw)
        except Exception:
            break

        if not fixes:
            break

        applied = 0
        for fix in fixes:
            old = fix.get("old", "")
            new = fix.get("new", "")
            if not old or old == new:
                continue
            if old in tex:
                tex = tex.replace(old, new, 1)
                applied += 1
            else:
                # Try whitespace-normalized match
                normalized_old = ' '.join(old.split())
                normalized_tex = ' '.join(tex.split())
                if normalized_old in normalized_tex:
                    # Find the actual position and replace
                    tex = _whitespace_aware_replace(tex, old, new)
                    applied += 1

        if applied == 0:
            break
        any_applied = True

    # FIX: return whether any fix was actually applied
    return tex, any_applied


def backup_and_fix(tex_path: Path,
                   compile_fn,
                   llm_call,
                   max_rounds: int = 3) -> tuple[bool, Optional[str]]:
    """Full fix pipeline with backup/restore.

    Args:
        tex_path: Path to .tex file.
        compile_fn: Callable that returns (success: bool, error_log: str).
        llm_call: Async LLM call function.
        max_rounds: Maximum fix rounds.

    Returns:
        (success, error_or_none)
    """
    # Backup
    backup_path = tex_path.with_suffix('.tex.bak')
    shutil.copy2(tex_path, backup_path)

    tex = tex_path.read_text(encoding='utf-8')

    # Level 1: Deterministic
    tex = deterministic_fix(tex)
    tex_path.write_text(tex, encoding='utf-8')

    success, error = compile_fn(tex_path)
    if success:
        return True, None

    # Level 2: LLM search-replace
    # (caller must run this in async context)
    # Return the current state for async processing
    return False, error


# --- Internal helpers ---

def _ensure_packages(tex: str) -> str:
    """Add missing required packages after \\documentclass line.

    Only adds if the package name does not appear anywhere in the preamble
    (covers \\usepackage{pkg}, \\usepackage[opts]{pkg}, and style files
    that internally \\RequirePackage{pkg}).
    """
    # Extract preamble (everything before \begin{document})
    begin_doc = tex.find('\\begin{document}')
    preamble = tex[:begin_doc] if begin_doc > 0 else tex[:2000]

    for pkg in REQUIRED_PACKAGES:
        # Check if package name appears anywhere in preamble
        if pkg not in preamble:
            dc_end = tex.find('\n', tex.find('\\documentclass'))
            if dc_end > 0:
                tex = tex[:dc_end+1] + f'\\usepackage{{{pkg}}}\n' + tex[dc_end+1:]
    return tex


def _fix_mismatched_envs(tex: str) -> str:
    """Fix unmatched \\begin{}/\\end{} pairs."""
    begins = re.findall(r'\\begin\{(\w+)\}', tex)
    ends = re.findall(r'\\end\{(\w+)\}', tex)
    begin_counts = {}
    end_counts = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1

    for env in begin_counts:
        diff = begin_counts[env] - end_counts.get(env, 0)
        if diff > 0:
            # Missing \end{env} — append before \end{document}
            end_doc = tex.rfind('\\end{document}')
            if end_doc > 0:
                tex = tex[:end_doc] + f'\\end{{{env}}}\n' * diff + tex[end_doc:]
    return tex


def _error_signature(error: str) -> str:
    """Extract a stable signature from an error log for dedup."""
    import hashlib
    # Keep last 500 chars of error (most specific)
    return hashlib.md5(error[-500:].encode()).hexdigest()[:8]


def _parse_fixes_json(raw: str) -> list[dict]:
    """Parse LLM response as JSON array of {old, new} fixes."""
    import json
    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip())
    raw = re.sub(r'\n?```\s*$', '', raw.strip())
    try:
        fixes = json.loads(raw)
        if isinstance(fixes, list):
            return [f for f in fixes if isinstance(f, dict) and "old" in f and "new" in f]
    except json.JSONDecodeError:
        pass
    return []


def _whitespace_aware_replace(tex: str, old: str, new: str) -> str:
    """Replace old with new, normalizing whitespace for matching.

    FIX: Use flexible whitespace between all line parts, not just for
    single-line patterns.
    """
    old_lines = old.strip().split('\n')
    pattern_parts = []
    for line in old_lines:
        escaped = re.escape(line.strip())
        pattern_parts.append(escaped)
    # Join all parts with flexible whitespace (works for both single and multi-line)
    pattern = r'\s+'.join(pattern_parts)

    try:
        result = re.sub(pattern, new, tex, count=1)
        return result
    except re.error:
        return tex  # Safety: don't modify if regex fails
```

### Migration Steps

1. Create `nanoresearch/latex/fixer.py` with the code above
2. In `writing.py`, replace all deterministic fix calls with:
   ```python
   from nanoresearch.latex.fixer import deterministic_fix, llm_search_replace_fix
   ```
3. In `review.py`, replace `_try_deterministic_fix()` and `_search_replace_llm_fix()` with imports
4. Delete the duplicated implementations from both files
5. Run LaTeX compilation tests to verify identical behavior
