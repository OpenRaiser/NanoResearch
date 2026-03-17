"""Iteration helpers part 2: full-write fallback, history, imports, syntax, best round."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents._code_utils import _strip_code_fences
from nanoresearch.agents.repair_journal import (
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.schemas.iteration import (
    ExperimentHypothesis,
    IterationState,
    RoundResult,
)

logger = logging.getLogger(__name__)


class _IterationHelpersMixin:
    """Mixin — full-write fallback, history, imports, syntax, best round."""

    async def _apply_iteration_changes_fullwrite(
        self,
        hypothesis: ExperimentHypothesis,
        code_dir: Path,
    ) -> list[str]:
        """Fallback: when search-replace fails, ask LLM to rewrite the target file entirely."""
        self._remember_mutation_snapshot_entry(None)
        # Find the primary target file from planned_changes
        target_rel = None
        for change_desc in hypothesis.planned_changes:
            # Extract file path from descriptions like "src/trainer.py: fix ..."
            for part in change_desc.replace(":", " ").split():
                candidate = code_dir / part
                try:
                    # Security: ensure candidate is within code_dir (no path traversal)
                    candidate.resolve().relative_to(code_dir.resolve())
                except ValueError:
                    continue
                if candidate.exists() and candidate.is_file():
                    target_rel = part
                    break
            if target_rel:
                break

        if not target_rel:
            # Default to main.py
            if (code_dir / "main.py").exists():
                target_rel = "main.py"
            else:
                return []

        target = code_dir / target_rel
        try:
            current = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        # Build file context: head + tail for large files to stay within LLM limits
        total_lines = len(current.splitlines())
        if len(current) <= 12000:
            file_block = current
        else:
            # Show first 8K chars + last 4K chars with a separator
            head = current[:8000]
            tail = current[-4000:]
            file_block = (
                f"{head}\n\n... [{total_lines} lines total, middle section omitted for brevity] ...\n\n{tail}"
            )

        prompt = f"""Rewrite the file `{target_rel}` to implement this change:

== Hypothesis ==
{hypothesis.hypothesis}

== Planned Changes ==
{chr(10).join(hypothesis.planned_changes)}

== Current File ({total_lines} lines) ==
```python
{file_block}
```

Output the COMPLETE new file content. No markdown fences, no explanation — ONLY the Python code.
The output MUST be a complete, runnable file — do NOT omit any functions or classes from the original."""

        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                f"You are an ML code editor. Rewrite {target_rel} to implement the requested change. "
                f"Output ONLY the complete file. Do NOT truncate or omit any part of the original code.",
                prompt,
            )
            # Robust fence stripping — handles LLM self-correction and multiple blocks
            new_content = _strip_code_fences(raw)

            # Safety: reject if the rewrite looks truncated (LLM hit max_tokens)
            if new_content and len(new_content) > 50:
                # Truncation heuristic: a valid Python file should end with a
                # complete statement — not mid-line or mid-string.
                _last_line = new_content.rstrip().rsplit("\n", 1)[-1].strip()
                _looks_truncated = (
                    # Ends with open string/paren/bracket
                    _last_line.endswith(("(", "[", "{", ",", "\\", '"""', "'''"))
                    # Or ends mid-expression (no closing quote, has unbalanced quotes)
                    or _last_line.count('"') % 2 == 1
                    or _last_line.count("'") % 2 == 1
                    # Or suspiciously short AND the file was large (likely max_tokens cutoff)
                    or (len(new_content) < len(current) * 0.3 and len(current) > 1000)
                )
                if _looks_truncated:
                    logger.warning(
                        "Full-file rewrite for %s looks truncated (%d vs %d chars, last: %s), skipping",
                        target_rel, len(new_content), len(current), _last_line[-60:],
                    )
                    return []
                snapshot = capture_repair_snapshot(
                    self.workspace.path,
                    target,
                    namespace="iteration_fullwrite",
                    root_dir=self.workspace.path,
                    operation="rewrite",
                )
                target.write_text(new_content, encoding="utf-8")
                if target.suffix.lower() == ".py" and not self._check_syntax(target):
                    self.log(f"  Full-file rewrite produced invalid Python in {target_rel}, rolling back")
                    rollback_snapshot(self.workspace.path, target, snapshot)
                    snapshot["rolled_back"] = True
                    snapshot["rollback_reason"] = "syntax_error"
                    entry = append_snapshot_journal(
                        self.workspace.path,
                        agent=self.__class__.__name__,
                        mutation_kind="iteration_fullwrite",
                        scope="legacy_iteration_fullwrite",
                        snapshots=[snapshot],
                        metadata={"modified_files": []},
                    )
                    self._remember_mutation_snapshot_entry(entry)
                    return []

                entry = append_snapshot_journal(
                    self.workspace.path,
                    agent=self.__class__.__name__,
                    mutation_kind="iteration_fullwrite",
                    scope="legacy_iteration_fullwrite",
                    snapshots=[snapshot],
                    metadata={"modified_files": [target_rel]},
                )
                self._remember_mutation_snapshot_entry(entry)
                self.log(f"  Rewrote {target_rel} (full-file fallback, {len(new_content)} chars)")
                return [target_rel]
        except Exception as exc:
            logger.warning("Full-file rewrite fallback failed for %s: %s", target_rel, exc)

        return []

    @staticmethod
    def _build_history_summary(rounds: list[RoundResult]) -> str:
        """Compress historical rounds into a compact summary (~100 chars each)."""
        if not rounds:
            return ""
        lines = []
        for r in rounds:
            metrics_str = ""
            if r.analysis and r.analysis.metric_summary:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in r.analysis.metric_summary.items()
                )
            hyp_short = r.hypothesis.hypothesis[:80]
            attribution = r.analysis.attribution if r.analysis else "n/a"
            lines.append(
                f"R{r.round_number}: [{r.quick_eval_status}] {hyp_short} "
                f"| metrics: {metrics_str or 'none'} | attr: {attribution}"
            )
        return "\n".join(lines)

    @staticmethod
    def _check_import_consistency(code_dir: Path) -> list[dict]:
        """Scan all generated files for cross-file import mismatches via AST.

        Checks two patterns:
        1. `from X import Y` where Y doesn't exist in X
        2. `import X; X.func()` where func doesn't exist in X

        Returns list of mismatch dicts.
        """
        from nanoresearch.agents.import_checker import ImportChecker

        checker = ImportChecker(code_dir)
        return checker.check_imports()

    async def _fix_import_mismatches(
        self, code_dir: Path, mismatches: list[dict],
    ) -> None:
        """Ask LLM to fix cross-file import mismatches via search-replace patches."""
        # Read all source files
        all_sources = {}
        for py_file in code_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                try:
                    all_sources[py_file.name] = py_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

        source_listing = ""
        for fname, content in sorted(all_sources.items()):
            source_listing += f"\n# FILE: {fname}\n{content}\n"

        system_prompt = (
            "You are fixing cross-file interface mismatches between Python files in a project. "
            "Some files reference names that don't exist in the target module. "
            "Fix by EITHER adding the missing function/class to the target module, "
            "OR renaming the call to match what's already defined. "
            "Return JSON with patches."
        )

        mismatch_desc = json.dumps(mismatches[:10], indent=2)  # cap at 10
        user_prompt = f"""Import mismatches found:
{mismatch_desc}

Source files:
{source_listing[:15000]}

Return JSON:
{{
  "patches": [
    {{
      "file": "filename.py",
      "old": "exact text to replace",
      "new": "replacement text"
    }}
  ]
}}"""

        try:
            result = await self.generate_json(system_prompt, user_prompt)
            patches = result.get("patches", []) if isinstance(result, dict) else []

            fixed = 0
            for patch in patches:
                filepath = code_dir / patch.get("file", "")
                try:
                    filepath.resolve().relative_to(code_dir.resolve())
                except ValueError:
                    continue
                old_text = patch.get("old", "")
                new_text = patch.get("new", "")
                if filepath.exists() and old_text and new_text:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                    if old_text in content:
                        filepath.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
                        fixed += 1
                        self.log(f"  Fixed import mismatch in {patch['file']}")
            self.log(f"Import fix: {fixed}/{len(patches)} patches applied")
        except Exception as e:
            self.log(f"Import fix failed (non-fatal): {e}")

    @staticmethod
    def _check_syntax(filepath: Path) -> bool:
        """Check if a Python file has valid syntax via py_compile."""
        try:
            import py_compile
            py_compile.compile(str(filepath), doraise=True)
            return True
        except py_compile.PyCompileError:
            return False
        except Exception:
            return True  # assume OK if check itself fails

    @staticmethod
    def _get_best_round(state: IterationState) -> dict:
        """Return result data from the best round, or the last round as fallback."""
        if not state.rounds:
            return {
                "execution_status": "skipped",
                "quick_eval_status": "skipped",
                "metrics": {},
            }
        # Find best round by index
        best_idx = None
        if state.best_round is not None:
            for i, r in enumerate(state.rounds):
                if r.round_number == state.best_round:
                    best_idx = i
                    break
        # Fallback to last round
        if best_idx is None:
            best_idx = len(state.rounds) - 1

        best = state.rounds[best_idx]
        return {
            "execution_status": best.execution_status,
            "quick_eval_status": best.quick_eval_status,
            "metrics": best.metrics,
        }
