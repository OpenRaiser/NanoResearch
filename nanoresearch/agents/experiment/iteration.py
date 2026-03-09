"""Iteration helpers: checkpoint, hypothesis, changes, history, imports, syntax."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from nanoresearch.agents.repair_journal import (
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.schemas.iteration import (
    ExperimentHypothesis,
    FeedbackAnalysis,
    IterationState,
    RoundResult,
)

logger = logging.getLogger(__name__)


class _IterationMixin:
    """Mixin — iteration checkpoint, hypothesis, changes, history."""

    def _save_iteration_checkpoint(
        self,
        state: IterationState,
        checkpoint_path: str = "logs/iteration_checkpoint.json",
    ) -> None:
        """Save iteration state checkpoint for crash recovery."""
        self.workspace.write_json(
            checkpoint_path,
            state.model_dump(),
        )

    def _load_iteration_checkpoint(
        self,
        default_state: IterationState,
        checkpoint_path: str = "logs/iteration_checkpoint.json",
    ) -> tuple[IterationState, int]:
        """Load iteration checkpoint if available.

        Returns (state, start_round) where start_round is the round to
        resume from (1 if no checkpoint exists).
        """
        try:
            data = self.workspace.read_json(checkpoint_path)
            if isinstance(data, dict) and data.get("rounds"):
                state = IterationState.model_validate(data)
                completed_rounds = len(state.rounds)
                start_round = completed_rounds + 1
                if start_round <= state.max_rounds:
                    logger.info(
                        "Resuming experiment from round %d (checkpoint has %d completed rounds)",
                        start_round, completed_rounds,
                    )
                    return state, start_round
                else:
                    logger.info(
                        "Checkpoint shows all %d rounds completed, starting fresh",
                        completed_rounds,
                    )
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to load iteration checkpoint: %s", exc)
        return default_state, 1

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    async def _generate_iteration_hypothesis(
        self,
        analysis: FeedbackAnalysis | None,
        history_summary: str,
        blueprint: str,
        preflight_error_ctx: str = "",
        code_dir: Path | None = None,
    ) -> ExperimentHypothesis:
        """LLM generates the next iteration hypothesis from feedback."""
        analysis_text = ""
        if analysis:
            analysis_text = (
                f"Attribution: {analysis.attribution}\n"
                f"Recommended action: {analysis.recommended_action}\n"
                f"Metrics: {json.dumps(analysis.metric_summary)}\n"
                f"Training dynamics: convergence={analysis.training_dynamics.convergence_speed}, "
                f"overfitting={analysis.training_dynamics.overfitting_detected}, "
                f"stability={analysis.training_dynamics.loss_stability}\n"
                f"Error categories: {analysis.error_categories}"
        )

        # Collect actual file list from code_dir for the LLM
        code_dir = code_dir or (self.workspace.path / "code")
        actual_files = []
        if code_dir.exists():
            for f in sorted(code_dir.rglob("*")):
                if f.is_file() and "__pycache__" not in str(f) and ".pyc" not in str(f):
                    actual_files.append(str(f.relative_to(code_dir)).replace("\\", "/"))
        file_list = "\n".join(f"  - {f}" for f in actual_files) if actual_files else "  (no files yet)"

        # Build list of previously tried hypotheses to prevent repetition
        prev_hypotheses = []
        if history_summary:
            for line in history_summary.split("\n"):
                if line.strip():
                    prev_hypotheses.append(line.strip())
        prev_hyp_block = "\n".join(prev_hypotheses) if prev_hypotheses else "None"

        prompt = f"""Based on the previous experiment round's feedback, generate a hypothesis for the next improvement iteration.
{preflight_error_ctx}
== Previous Analysis ==
{analysis_text or "No analysis available."}

== History ==
{history_summary or "No previous rounds."}

== PREVIOUSLY TRIED HYPOTHESES (DO NOT REPEAT) ==
{prev_hyp_block}

== Experiment Blueprint ==
{blueprint[:2000]}

== Actual Project Files ==
{file_list}

IMPORTANT RULES:
1. Only reference files that exist in the list above. Do NOT invent new file paths.
2. Use the EXACT paths shown above in your planned_changes.
3. The `--quick-eval` mode HARDCODES a small model and 3-5 epochs regardless of config.
   Changing epochs/batch_size/num_runs in config/default.yaml has NO EFFECT on quick-eval.
   DO NOT suggest increasing epochs or changing hyperparameters in config — it is USELESS.
4. Instead, focus on changes that actually affect quick-eval behavior:
   - Fix bugs in model architecture (src/model.py)
   - Fix bugs in training loop (src/trainer.py)
   - Fix evaluation/metrics collection (src/evaluate.py, src/utils.py)
   - Fix data loading/preprocessing (src/dataset.py)
   - Fix the quick-eval code path in main.py directly
   - Improve model architecture (e.g., add batch norm, better init, residual connections)
5. DO NOT repeat any hypothesis from the list above. Each round must try something DIFFERENT.
   If you cannot think of a genuinely new improvement, set "no_new_ideas": true.

Output a JSON object with:
{{
  "hypothesis": "<what you will change and why>",
  "planned_changes": ["<EXACT_FILE_PATH: specific change>", ...],
  "expected_signal": "<what metric improvement you expect>",
  "rationale": "<reasoning>",
  "no_new_ideas": false
}}"""

        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                "You are an ML experiment iteration planner. Generate a focused hypothesis for the next improvement round. Output ONLY valid JSON.",
                prompt,
                json_mode=True,
            )
            data = self._parse_llm_json_payload(raw)

            # If LLM says no new ideas, signal early stop
            if data.get("no_new_ideas"):
                logger.info("LLM reports no new ideas — will signal early stop")
                return ExperimentHypothesis(
                    round_number=0,
                    hypothesis="__NO_NEW_IDEAS__",
                    planned_changes=[],
                    expected_signal="",
                    rationale="LLM exhausted improvement ideas",
                )

            return ExperimentHypothesis(
                round_number=0,  # caller sets this
                hypothesis=data.get("hypothesis", "Iterative improvement"),
                planned_changes=data.get("planned_changes", []),
                expected_signal=data.get("expected_signal", ""),
                rationale=data.get("rationale", ""),
            )
        except Exception as exc:
            logger.warning("Failed to generate hypothesis: %s", exc)
            return ExperimentHypothesis(
                round_number=0,
                hypothesis="Retry with general improvements based on error feedback",
                planned_changes=["Fix errors from previous round"],
                expected_signal="Successful execution",
                rationale="Fallback hypothesis after LLM generation failure",
            )

    async def _apply_iteration_changes(
        self,
        hypothesis: ExperimentHypothesis,
        code_dir: Path,
        venv_python: str,
    ) -> list[str]:
        """LLM modifies specific files using search-replace edits (OpenClaw style).

        Uses precise search-replace blocks instead of full file rewrites to:
        1. Reduce token usage (LLM only outputs the diff, not entire files)
        2. Avoid accidental deletion of unchanged code
        3. Make changes auditable
        """
        self._remember_mutation_snapshot_entry(None)
        # Collect current file contents for context
        file_contents: dict[str, str] = {}
        for py_file in code_dir.rglob("*.py"):
            parts = py_file.relative_to(code_dir).parts
            if any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue
            try:
                rel = str(py_file.relative_to(code_dir)).replace("\\", "/")
                content = py_file.read_text(encoding="utf-8", errors="replace")
                file_contents[rel] = content
            except OSError:
                continue

        # Also include config and other non-py files
        for pattern in ("config/*.yaml", "config/*.yml", "*.txt", "*.sh"):
            for f in code_dir.glob(pattern):
                try:
                    rel = str(f.relative_to(code_dir)).replace("\\", "/")
                    file_contents[rel] = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

        files_summary = "\n".join(
            f"--- {path} ---\n{content[:2000]}\n"
            for path, content in file_contents.items()
        )

        prompt = f"""Apply the following changes to the experiment code project using SEARCH-REPLACE edits.

== Hypothesis ==
{hypothesis.hypothesis}

== Planned Changes ==
{json.dumps(hypothesis.planned_changes, indent=2)}

== Rationale ==
{hypothesis.rationale}

== Current Files ==
{files_summary[:15000]}

Output a JSON array of edit operations. Two types are supported:

1. **Search-replace edit** (preferred for modifying existing files):
{{
  "path": "relative/path.py",
  "action": "edit",
  "edits": [
    {{"old": "exact text to find", "new": "replacement text"}}
  ]
}}

2. **Full file write** (only for NEW files that don't exist yet):
{{
  "path": "relative/new_file.py",
  "action": "write",
  "content": "full file content"
}}

IMPORTANT RULES:
- "old" must be an EXACT substring of the current file content (including whitespace/indentation)
- Each "old" string must be unique within its file
- Use search-replace for ALL modifications to existing files
- Only use "write" action for creating brand new files
- Multiple edits per file are fine — they are applied sequentially

Output ONLY valid JSON array."""

        modified_files: list[str] = []
        snapshot_batch: list[dict[str, Any]] = []
        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                "You are an ML code editor. Apply precise search-replace edits to implement the hypothesis. Output ONLY a JSON array.",
                prompt,
            )
            changes = self._parse_llm_json_payload(raw)
            if not isinstance(changes, list):
                changes = [changes]

            for change in changes:
                if not isinstance(change, dict) or "path" not in change:
                    continue
                file_path = change["path"]
                # Security: prevent directory traversal
                try:
                    (code_dir / file_path).resolve().relative_to(code_dir.resolve())
                except ValueError:
                    logger.warning("Skipping unsafe iteration path: %s", file_path)
                    continue

                action = change.get("action", "write")  # backwards compat

                if action == "edit":
                    # Search-replace mode
                    edits = change.get("edits", [])
                    if not edits:
                        continue
                    # Read current content
                    target = code_dir / file_path
                    if not target.exists():
                        logger.warning("Edit target does not exist: %s", file_path)
                        continue
                    try:
                        current = target.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue

                    applied = 0
                    for edit in edits:
                        if not isinstance(edit, dict):
                            continue
                        old = edit.get("old", "")
                        new = edit.get("new", "")
                        if not old:
                            continue
                        current, matched, match_strategy = self._apply_search_replace_edit(
                            current,
                            old,
                            new,
                        )
                        if matched:
                            applied += 1
                            self.log(f"  Matched edit in {file_path} via {match_strategy}")
                        else:
                            logger.warning(
                                "Edit old text not found in %s: %s",
                                file_path, old[:80],
                            )

                    if applied > 0:
                        target_path = code_dir / file_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        snapshot = capture_repair_snapshot(
                            self.workspace.path,
                            target_path,
                            namespace="iteration_changes",
                            root_dir=self.workspace.path,
                            operation="rewrite",
                        )
                        target_path.write_text(current, encoding="utf-8")
                        if target_path.suffix.lower() == ".py" and not self._check_syntax(target_path):
                            self.log(f"  Edited file became invalid Python in {file_path}, rolling back")
                            rollback_snapshot(self.workspace.path, target_path, snapshot)
                            snapshot["rolled_back"] = True
                            snapshot["rollback_reason"] = "syntax_error"
                            snapshot_batch.append(snapshot)
                            continue

                        modified_files.append(file_path)
                        snapshot_batch.append(snapshot)
                        self.log(f"  Edited: {file_path} ({applied}/{len(edits)} edits applied)")
                else:
                    # Full write mode (new files or backwards compat)
                    content = change.get("content", "")
                    if not content:
                        continue
                    target_path = code_dir / file_path
                    existed_before = target_path.exists()
                    snapshot = capture_repair_snapshot(
                        self.workspace.path,
                        target_path,
                        namespace="iteration_changes",
                        root_dir=self.workspace.path,
                        existed_before=existed_before,
                        operation="rewrite" if existed_before else "create",
                    )
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(content, encoding="utf-8")
                    if target_path.suffix.lower() == ".py" and not self._check_syntax(target_path):
                        self.log(f"  Wrote invalid Python in {file_path}, rolling back")
                        rollback_snapshot(self.workspace.path, target_path, snapshot)
                        snapshot["rolled_back"] = True
                        snapshot["rollback_reason"] = "syntax_error"
                        snapshot_batch.append(snapshot)
                        continue

                    modified_files.append(file_path)
                    snapshot_batch.append(snapshot)
                    self.log(f"  Wrote: {file_path}")

        except Exception as exc:
            logger.warning("Failed to apply iteration changes: %s", exc)

        if snapshot_batch:
            entry = append_snapshot_journal(
                self.workspace.path,
                agent=self.__class__.__name__,
                mutation_kind="iteration_changes",
                scope="legacy_iteration_search_replace",
                snapshots=snapshot_batch,
                metadata={"modified_files": list(modified_files)},
            )
            self._remember_mutation_snapshot_entry(entry)
        return modified_files

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
            new_content = (raw or "").strip()
            # Strip markdown fences
            if new_content.startswith("```"):
                lines = new_content.split("\n")[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                new_content = "\n".join(lines)

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
        """Scan all generated files for cross-file import mismatches.

        Borrowed from Deep Pipeline's CodingAgent — checks two patterns:
        1. `from X import Y` where Y doesn't exist in X
        2. `import X; X.func()` where func doesn't exist in X

        Returns list of mismatch dicts.
        """
        import re as _re

        definitions: dict[str, list[str]] = {}  # module -> [defined names]
        imports: list[dict] = []
        module_accesses: list[dict] = []
        local_modules = {f.stem for f in code_dir.rglob("*.py")}

        for py_file in code_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            module_name = py_file.stem

            # Find class and top-level function definitions
            defs = [m.group(1) for m in _re.finditer(r"^(?:class|def)\s+(\w+)", content, _re.MULTILINE)]
            definitions[module_name] = defs

            # Find cross-file imports: from X import Y, Z
            for m in _re.finditer(r"^from\s+(?:src\.)?(\w+)\s+import\s+(.+)$", content, _re.MULTILINE):
                src_module = m.group(1)
                # Strip inline comments before parsing names
                import_text = m.group(2).split("#")[0]
                imported_names = [n.strip().split(" as ")[0].strip() for n in import_text.split(",")]
                imported_names = [n for n in imported_names if n]  # drop empty after comment strip
                imports.append({"importer": py_file.name, "module": src_module, "names": imported_names})

            # Find `import X` for local modules, then scan for X.attr() calls
            imported_modules: dict[str, str] = {}
            for m in _re.finditer(r"^import\s+(?:src\.)?(\w+)(?:\s+as\s+(\w+))?$", content, _re.MULTILINE):
                real_name = m.group(1)
                alias = m.group(2) or real_name
                if real_name in local_modules:
                    imported_modules[alias] = real_name

            for alias, real_name in imported_modules.items():
                for m in _re.finditer(rf"\b{_re.escape(alias)}\.(\w+)\s*\(", content):
                    attr = m.group(1)
                    if not attr.startswith("_"):
                        module_accesses.append({
                            "importer": py_file.name, "module": real_name, "attr": attr,
                        })

        # Check mismatches
        mismatches = []
        for imp in imports:
            module = imp["module"]
            if module not in definitions:
                continue
            defined = set(definitions[module])
            for name in imp["names"]:
                if name and name not in defined:
                    mismatches.append({
                        "importer": imp["importer"], "module": module,
                        "missing_name": name, "available": sorted(defined),
                    })

        seen_access: set[tuple[str, str, str]] = set()
        for acc in module_accesses:
            module = acc["module"]
            if module not in definitions:
                continue
            attr = acc["attr"]
            key = (acc["importer"], module, attr)
            if key in seen_access:
                continue
            seen_access.add(key)
            defined = set(definitions[module])
            if attr not in defined:
                mismatches.append({
                    "importer": acc["importer"], "module": module,
                    "missing_name": attr, "available": sorted(defined),
                    "usage_pattern": f"import {module}; {module}.{attr}()",
                })

        return mismatches

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
            result = subprocess.run(
                [sys.executable, "-c",
                 f"import py_compile; py_compile.compile(r'{filepath}', doraise=True)"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
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

