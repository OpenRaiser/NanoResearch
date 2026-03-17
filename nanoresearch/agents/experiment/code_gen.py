"""Code generation: project plan, individual files, verification."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents._code_utils import _strip_code_fences
from . import (
    MAX_REFERENCE_REPOS,
    MAX_FILE_TREE_ENTRIES,
    MAX_README_EXCERPT_LENGTH,
    PROJECT_PLAN_SYSTEM_PROMPT,
    FILE_GEN_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class _CodeGenMixin:
    """Mixin — code generation and verification."""

    @staticmethod
    def _build_repo_context(reference_repos: list[dict]) -> str:
        """Build a context string from reference GitHub repos."""
        if not reference_repos:
            return ""

        lines = ["=== REFERENCE GITHUB REPOSITORIES ==="]
        lines.append("Use these real open-source projects as structural references.")
        lines.append("Mirror their project layout, naming conventions, and best practices.\n")

        for repo in reference_repos[:MAX_REFERENCE_REPOS]:
            name = repo.get("full_name", "unknown")
            desc = repo.get("description", "")
            stars = repo.get("stars", 0)
            tree = repo.get("file_tree", [])
            readme = repo.get("readme_excerpt", "")

            lines.append(f"--- {name} ({stars} stars) ---")
            if desc:
                lines.append(f"Description: {desc}")

            if tree:
                lines.append("File structure:")
                for path in tree[:MAX_FILE_TREE_ENTRIES]:
                    lines.append(f"  {path}")
                if len(tree) > MAX_FILE_TREE_ENTRIES:
                    lines.append(f"  ... ({len(tree) - MAX_FILE_TREE_ENTRIES} more files)")

            if readme:
                lines.append(f"README excerpt:\n  {readme[:MAX_README_EXCERPT_LENGTH]}")

            lines.append("")

        lines.append("=== END REFERENCE REPOS ===")
        return "\n".join(lines)

    async def _generate_project_plan(self, blueprint_summary: str, repo_context: str = "") -> dict:
        """Phase 1: Generate the project plan JSON via Codex."""
        repo_section = ""
        if repo_context:
            repo_section = (
                f"\n{repo_context}\n\n"
                "IMPORTANT: Model your project structure after the reference repos above.\n"
                "Use similar directory layouts, naming conventions, and design patterns.\n"
                "The generated code should look like it belongs in one of these real repos.\n"
            )

        prompt = f"""Design a complete Python ML project for this experiment:

{blueprint_summary}
{repo_section}
The project must be a self-contained, runnable research codebase with:
- Full model architecture implementation
- Data loading and preprocessing pipeline
- Training loop with checkpoint saving and logging
- Evaluation with all specified metrics
- Ablation experiment support
- Configuration via YAML
- Shell scripts for launching experiments

Output the project plan as a JSON object."""

        code_gen_config = self.config.for_stage("code_gen")
        raw = await self._dispatcher.generate(
            code_gen_config, PROJECT_PLAN_SYSTEM_PROMPT, prompt, json_mode=True
        )

        try:
            return self._parse_llm_json_payload(raw)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse project plan JSON. First 500 chars: %s",
                self._strip_json_fence(raw)[:500],
            )
            raise RuntimeError(
                f"Project plan is not valid JSON: {exc}"
            ) from exc

    async def _generate_file(
        self,
        file_spec: dict,
        interface_contract: str,
        blueprint_summary: str,
        repo_context: str = "",
    ) -> str:
        """Phase 2: Generate a single file via Codex."""
        file_path = file_spec["path"]
        description = file_spec.get("description", "")
        interfaces = file_spec.get("interfaces", [])
        depends_on = file_spec.get("depends_on", [])

        repo_section = ""
        if repo_context:
            repo_section = (
                f"\n{repo_context}\n\n"
                "Write code that follows the patterns and conventions of the reference repos above.\n"
            )

        prompt = f"""Generate the file: {file_path}
Description: {description}

This file must implement these interfaces:
{json.dumps(interfaces, indent=2)}

Dependencies (other project files this imports from):
{json.dumps(depends_on, indent=2)}

=== FULL PROJECT INTERFACE CONTRACT ===
{interface_contract}
=== END CONTRACT ===

=== EXPERIMENT BLUEPRINT ===
{blueprint_summary}
=== END BLUEPRINT ===
{repo_section}
Generate the COMPLETE file content. Follow the interface contract exactly."""

        code_gen_config = self.config.for_stage("code_gen")
        content = await self._dispatcher.generate(
            code_gen_config, FILE_GEN_SYSTEM_PROMPT, prompt
        )

        # Robust fence stripping — handles LLM self-correction and multiple blocks
        content = _strip_code_fences(content)

        return content

    def _verify_code(self, generated_files: list[str]) -> dict:
        """Verify generated Python files have valid syntax via compile()."""
        results = []
        passed = 0
        total = 0

        for fp in generated_files:
            if not fp.endswith(".py"):
                continue

            total += 1
            file_path = self.workspace.path / "code" / fp
            if not file_path.exists():
                results.append({
                    "file": fp,
                    "status": "missing",
                    "error": "File not found",
                })
                continue

            try:
                source = file_path.read_text(encoding="utf-8")
                compile(source, fp, "exec")
                results.append({"file": fp, "status": "ok", "error": None})
                passed += 1
            except SyntaxError as e:
                results.append({
                    "file": fp,
                    "status": "syntax_error",
                    "error": f"Line {e.lineno}: {e.msg}",
                })

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "files": results,
        }
