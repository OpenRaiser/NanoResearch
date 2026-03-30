"""EvoMemory-inspired cross-session persistent memory system.

Two layers:
  1. **Injection**: reads MEMORY.md and injects into LLM system prompts
  2. **Extraction**: after each pipeline run, extracts structured facts and merges

Storage: ~/.nanoresearch/memory/MEMORY.md
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_DIR = Path.home() / ".nanoresearch" / "memory"
_MEMORY_FILE = _MEMORY_DIR / "MEMORY.md"

# ─── Structured schema ───

_EXTRACTION_SYSTEM = """\
You are a memory extraction assistant. Given the research pipeline output,
extract ONLY genuinely new information into the JSON schema below.
Leave fields as null if no new info is available. Do NOT fabricate.

Return JSON:
{
  "user_profile": {
    "preferred_language": "<zh/en or null>",
    "research_domain": "<primary domain or null>",
    "hardware": "<GPU/CPU info or null>"
  },
  "research_preferences": {
    "preferred_models": ["<model names>"],
    "preferred_frameworks": ["<pytorch/tensorflow/etc>"],
    "preferred_template": "<neurips/icml/arxiv or null>",
    "writing_style": "<formal/concise/etc or null>"
  },
  "experiment_conclusion": {
    "topic": "<research topic>",
    "method": "<proposed method name>",
    "key_result": "<best metric achieved>",
    "conclusion": "<1-sentence takeaway>",
    "date": "<YYYY-MM-DD>"
  },
  "learned_preferences": ["<any user preference or habit observed>"]
}"""

_INJECTION_TEMPLATE = """\
=== PERSISTENT MEMORY (from prior sessions) ===
{content}
=== END MEMORY ===

Use this context to personalize your output. Do NOT mention the memory system to the user."""


class ResearchMemory:
    """Cross-session persistent memory for NanoResearch."""

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._dir = memory_dir or _MEMORY_DIR
        self._file = self._dir / "MEMORY.md"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ─── Layer 1: Injection ───

    def read(self) -> str:
        """Read the current memory content."""
        if self._file.is_file():
            try:
                return self._file.read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return ""

    def inject_into_prompt(self, system_prompt: str) -> str:
        """Prepend memory context to an LLM system prompt."""
        content = self.read()
        if not content:
            return system_prompt
        injection = _INJECTION_TEMPLATE.format(content=content)
        return injection + "\n\n" + system_prompt

    # ─── Layer 2: Extraction & Merge ───

    async def extract_and_merge(self, pipeline_result: dict, dispatcher) -> None:
        """Extract structured facts from a pipeline result and merge into memory.

        Args:
            pipeline_result: The full pipeline output dict
            dispatcher: ModelDispatcher instance for LLM calls
        """
        summary = self._build_extraction_input(pipeline_result)
        if not summary:
            return

        try:
            from nanoresearch.config import StageModelConfig
            # Use a cheap/fast config for extraction
            cfg = StageModelConfig(
                model=dispatcher._config.ideation.model,
                temperature=0.0,
                max_tokens=2000,
                base_url=dispatcher._config.ideation.base_url,
                api_key=dispatcher._config.ideation.api_key,
            )
            raw = await dispatcher.generate(cfg, _EXTRACTION_SYSTEM, summary, json_mode=True)
            extracted = json.loads(raw)
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return

        self._merge(extracted)
        logger.info("Memory updated: %s", self._file)

    def _build_extraction_input(self, result: dict) -> str:
        """Build a concise summary of the pipeline result for extraction."""
        parts = []

        # Topic
        topic = result.get("topic", "")
        if topic:
            parts.append(f"Research topic: {topic}")

        # Ideation
        ideation = result.get("ideation_output", {})
        if isinstance(ideation, dict):
            hyps = ideation.get("hypotheses", [])
            if hyps:
                parts.append(f"Hypotheses: {len(hyps)} generated")

        # Blueprint
        bp = result.get("experiment_blueprint", {})
        if isinstance(bp, dict):
            method = (bp.get("proposed_method") or {}).get("name", "")
            if method:
                parts.append(f"Proposed method: {method}")
            metrics = [m.get("name", "") for m in (bp.get("metrics") or [])]
            if metrics:
                parts.append(f"Metrics: {', '.join(metrics)}")

        # Experiment results
        exp = result.get("experiment_output", {})
        if isinstance(exp, dict):
            main_results = (exp.get("experiment_results") or {}).get("main_results", [])
            if main_results:
                parts.append(f"Main results: {len(main_results)} entries")

        # Cost
        cost = result.get("cost_summary", {})
        if isinstance(cost, dict) and cost.get("total_tokens"):
            parts.append(f"Total tokens: {cost['total_tokens']}")

        return "\n".join(parts) if parts else ""

    def _merge(self, extracted: dict) -> None:
        """Merge extracted facts into MEMORY.md, deduplicating."""
        existing = self.read()
        sections: dict[str, str] = {}

        # Parse existing sections
        if existing:
            current_heading = ""
            current_lines: list[str] = []
            for line in existing.split("\n"):
                if line.startswith("## "):
                    if current_heading:
                        sections[current_heading] = "\n".join(current_lines).strip()
                    current_heading = line[3:].strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            if current_heading:
                sections[current_heading] = "\n".join(current_lines).strip()

        # Merge user profile
        profile = extracted.get("user_profile") or {}
        if any(v for v in profile.values() if v):
            existing_profile = sections.get("User Profile", "")
            for key, val in profile.items():
                if val and str(val).lower() not in existing_profile.lower():
                    existing_profile += f"\n- **{key}**: {val}"
            sections["User Profile"] = existing_profile.strip()

        # Merge research preferences
        prefs = extracted.get("research_preferences") or {}
        if any(v for v in prefs.values() if v):
            existing_prefs = sections.get("Research Preferences", "")
            for key, val in prefs.items():
                if not val:
                    continue
                val_str = ", ".join(val) if isinstance(val, list) else str(val)
                if val_str.lower() not in existing_prefs.lower():
                    existing_prefs += f"\n- **{key}**: {val_str}"
            sections["Research Preferences"] = existing_prefs.strip()

        # Append experiment conclusion (history)
        conclusion = extracted.get("experiment_conclusion") or {}
        if conclusion.get("topic"):
            history = sections.get("Experiment History", "")
            date = conclusion.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            entry = (
                f"\n### {date}: {conclusion.get('topic', 'Unknown')}\n"
                f"- **Method**: {conclusion.get('method', 'N/A')}\n"
                f"- **Key Result**: {conclusion.get('key_result', 'N/A')}\n"
                f"- **Conclusion**: {conclusion.get('conclusion', 'N/A')}"
            )
            # Dedup by topic+date
            if conclusion["topic"].lower() not in history.lower():
                history += entry
            sections["Experiment History"] = history.strip()

        # Merge learned preferences (dedup)
        learned = extracted.get("learned_preferences") or []
        if learned:
            existing_learned = sections.get("Learned Preferences", "")
            existing_lower = existing_learned.lower()
            for pref in learned:
                if pref and pref.strip().lower() not in existing_lower:
                    existing_learned += f"\n- {pref.strip()}"
                    existing_lower = existing_learned.lower()
            sections["Learned Preferences"] = existing_learned.strip()

        # Write back
        self._write(sections)

    def _write(self, sections: dict[str, str]) -> None:
        """Write sections back to MEMORY.md."""
        ordered_keys = ["User Profile", "Research Preferences", "Experiment History", "Learned Preferences"]
        lines = ["# NanoResearch Memory\n"]
        for key in ordered_keys:
            content = sections.get(key, "").strip()
            if content:
                lines.append(f"## {key}\n{content}\n")
        # Any extra sections
        for key, content in sections.items():
            if key not in ordered_keys and content.strip():
                lines.append(f"## {key}\n{content.strip()}\n")

        try:
            self._file.write_text("\n".join(lines), encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to write memory: %s", e)

    def record_fact(self, fact: str) -> None:
        """Manually record a fact to the Learned Preferences section."""
        existing = self.read()
        sections: dict[str, str] = {}
        # Quick parse
        if existing:
            current_heading = ""
            current_lines: list[str] = []
            for line in existing.split("\n"):
                if line.startswith("## "):
                    if current_heading:
                        sections[current_heading] = "\n".join(current_lines).strip()
                    current_heading = line[3:].strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            if current_heading:
                sections[current_heading] = "\n".join(current_lines).strip()

        learned = sections.get("Learned Preferences", "")
        if fact.strip().lower() not in learned.lower():
            learned += f"\n- {fact.strip()}"
            sections["Learned Preferences"] = learned.strip()
            self._write(sections)
