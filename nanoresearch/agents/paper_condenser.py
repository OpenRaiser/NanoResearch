"""Conversation condenser + system prompt for paper editing agent.

Event-sourced conversation memory with LLM-based condensation,
inspired by OpenHands context condensation.  Also hosts the
PAPER_AGENT_SYSTEM prompt and helper functions used by PaperEditor.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.pipeline.workspace import Workspace

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Conversation Condenser  (inspired by OpenHands context condensation)
# ═══════════════════════════════════════════════════════════════════════════

_CONDENSE_SYSTEM = """\
You are a conversation summarizer for a research paper editing agent.
Compress the conversation events into a structured summary.
Be concise but preserve ALL actionable information.
Respond in the same language as the conversation (Chinese if Chinese)."""

_CONDENSE_TEMPLATE = """\
{prev_context}
## Events to Summarize

{events_text}

## Instructions

Produce a structured summary using these EXACT section headers.
Keep each section to 2-4 bullet points max.  Write "None" if empty.

### PAPER_STATE
Current paper status: pages, sections modified, figure count, review score.

### REVISION_HISTORY
Changes made and why.  Section names, direction (shortened/expanded/rewritten), impact.

### PENDING_ITEMS
Remaining work the user mentioned or issues identified but not yet addressed.

### KEY_DECISIONS
User preferences: style, length targets, sections to preserve, figure aesthetics."""


class ConversationCondenser:
    """Event-sourced conversation memory with LLM-based condensation.

    Design principles (inspired by OpenHands context condenser):
      1. Append-only event log -- history is never mutated
      2. Condensation events mark forgotten indices + store structured summary
      3. ``to_messages()`` builds a *view*: summary + surviving events
      4. Atomic boundaries -- user+assistant pairs never split
      5. Dual trigger -- soft (message count) + hard (context overflow)
    """

    SOFT_THRESHOLD = 16   # ~8 user+assistant pairs
    KEEP_TAIL = 6         # last 3 turns always visible

    def __init__(
        self, dispatcher: ModelDispatcher, config: ResearchConfig,
    ) -> None:
        self._events: list[dict[str, Any]] = []
        # Each condensation: (set of forgotten event indices, summary text)
        self._condensations: list[tuple[set[int], str]] = []
        self._dispatcher = dispatcher
        self._config = config

    # ── append ─────────────────────────────────────────────────────

    def append(self, message: dict[str, Any]) -> None:
        """Append a message to the event log (never mutated later)."""
        self._events.append(message)

    # ── read ───────────────────────────────────────────────────────

    def to_messages(self) -> list[dict[str, Any]]:
        """Build the current view: condensation summary + surviving events."""
        forgotten: set[int] = set()
        for fids, _ in self._condensations:
            forgotten.update(fids)

        messages: list[dict[str, Any]] = []
        # Prepend latest condensation summary (if any)
        if self._condensations:
            _, summary = self._condensations[-1]
            messages.append({
                "role": "system",
                "content": f"[Previous conversation context]\n{summary}",
            })

        for i, msg in enumerate(self._events):
            if i not in forgotten:
                messages.append(msg)
        return messages

    @property
    def active_count(self) -> int:
        """Number of non-forgotten events."""
        forgotten: set[int] = set()
        for fids, _ in self._condensations:
            forgotten.update(fids)
        return sum(1 for i in range(len(self._events)) if i not in forgotten)

    def needs_condensation(self) -> bool:
        return self.active_count > self.SOFT_THRESHOLD

    # ── condense ───────────────────────────────────────────────────

    async def condense(self, hard: bool = False) -> bool:
        """Compress old events into a structured summary.

        Args:
            hard: if True, keep fewer tail events (context overflow recovery).
        Returns:
            True if condensation was performed.
        """
        keep_tail = 2 if hard else self.KEEP_TAIL
        if len(self._events) <= keep_tail:
            return False

        tail_start = len(self._events) - keep_tail
        tail_start = self._snap_to_pair_boundary(tail_start)
        if tail_start <= 0:
            return False

        # Collect non-forgotten indices in the forgettable range
        already_forgotten: set[int] = set()
        for fids, _ in self._condensations:
            already_forgotten.update(fids)

        to_compress = [
            (i, self._events[i])
            for i in range(tail_start)
            if i not in already_forgotten
        ]
        if len(to_compress) < 2:
            return False

        # Pre-pass: truncate verbose content for the summariser
        formatted: list[str] = []
        for _, msg in to_compress:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if len(content) > 600:
                content = content[:600] + "..."
            if content:
                formatted.append(f"[{role}] {content}")
        events_text = "\n\n".join(formatted)

        prev_context = ""
        if self._condensations:
            _, prev = self._condensations[-1]
            prev_context = f"## Previous Summary\n\n{prev}"

        # LLM-based structured summarisation
        prompt = _CONDENSE_TEMPLATE.format(
            prev_context=prev_context,
            events_text=events_text,
        )
        cfg = self._config.for_stage("code_gen")
        try:
            summary = await self._dispatcher.generate(
                cfg, _CONDENSE_SYSTEM, prompt,
            )
            summary = summary.strip()
        except Exception as exc:
            # Fallback: mechanical bullet-point summary
            logger.warning("Condensation LLM failed (%s), using fallback", exc)
            lines: list[str] = []
            for _, msg in to_compress:
                if msg.get("role") == "user":
                    lines.append(f"- User: {msg.get('content', '')[:120]}")
                elif msg.get("role") == "assistant":
                    lines.append(f"- Agent: {msg.get('content', '')[:120]}")
            summary = "### REVISION_HISTORY\n" + "\n".join(lines)

        self._condensations.append(({i for i, _ in to_compress}, summary))
        return True

    # ── helpers ────────────────────────────────────────────────────

    def _snap_to_pair_boundary(self, idx: int) -> int:
        """Move *idx* forward so it lands on a user message (pair start)."""
        while 0 < idx < len(self._events):
            if self._events[idx].get("role") == "user":
                break
            idx += 1
        return min(idx, len(self._events))


# ═══════════════════════════════════════════════════════════════════════════
# Paper agent system prompt + helpers used by PaperEditor
# ═══════════════════════════════════════════════════════════════════════════

PAPER_AGENT_SYSTEM = """\
You are an expert research paper editing assistant.  You have access to tools
that let you inspect and modify every aspect of the paper: sections, figures,
citations, LaTeX source, and PDF compilation.

## Workflow — ALWAYS follow this pattern

1. **Inspect first** — call get_paper_info to understand the current state
   (pages, section lengths, figures, score) BEFORE making any changes.
2. **Plan** — for complex requests, break them into concrete steps.
   Think about which sections/figures are involved and what each step achieves.
3. **Execute** — use tools to make changes.  For multi-section edits, process
   them one by one so you can verify each.
4. **Verify** — after changes, re-inspect (get_paper_info, read_section) to
   confirm they had the desired effect.
5. **Compile** — call compile_pdf at the end if you made any content changes.
   Check the resulting page count.
6. **Report** — summarize what you changed, before/after metrics (page count,
   section lengths, etc.).

## Strategy guides

### Reducing page count
1. get_paper_info → note pages and section_char_lengths
2. Identify the longest sections (usually Experiments, Method, Related Work)
3. rewrite_section each with instruction to "shorten by ~30%, remove redundant
   paragraphs, keep all \\cite and \\ref, keep all figures"
4. compile_pdf → check pages
5. If still too many pages, trim more aggressively or remove less important content
6. Report before/after page counts

### Changing figure style
1. list_figures → see current figures
2. regenerate_figure with precise style_instruction
   (e.g. "blue-green color palette, sans-serif labels, clean white background")
3. compile_pdf to include new figures
4. Report which figures were regenerated

### Improving quality
1. run_review → get scores and issues
2. For each low-scoring section, rewrite_section with the review feedback
3. compile_pdf
4. Optionally run_review again to verify improvement

### Small targeted edits
- Use edit_tex for typos, numbers, single-sentence changes
- Use rewrite_section only when substantial rewriting is needed

## Important rules

- NEVER modify content without reading it first.
- Preserve all \\cite{{}}, \\ref{{}}, and \\begin{{figure}}...\\end{{figure}} blocks
  unless the user explicitly asks to remove them.
- When rewriting sections, always include the target length or change percentage
  in the instruction (e.g. "shorten by 40%", "expand to ~500 words").
- Always compile PDF after content changes.
- If a tool returns an error, diagnose the problem and try an alternative approach.

## Language

Respond in the same language as the user (Chinese if they write in Chinese).
"""


def _count_pdf_pages_quick(pdf_path: Path) -> int | None:
    """Count pages in a PDF file. Returns None if unavailable."""
    if not pdf_path.exists():
        return None
    try:
        content = pdf_path.read_bytes()
        pages = len(re.findall(rb"/Type\s*/Page[^s]", content))
        return pages if pages > 0 else None
    except Exception:
        return None


def build_paper_context(ws: Workspace) -> str:
    """Build a concise paper status summary to inject into the conversation."""
    tex_path = ws.path / "drafts" / "paper.tex"
    if not tex_path.exists():
        return ""
    tex = tex_path.read_text(encoding="utf-8", errors="replace")

    title_m = re.search(r"\\title\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)\}", tex)
    sections = re.findall(r"\\section\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)\}", tex)
    fig_count = len(re.findall(r"\\begin\{figure\}", tex))
    cite_count = len(re.findall(r"\\cite\{", tex))
    pages = _count_pdf_pages_quick(ws.path / "drafts" / "paper.pdf")

    score = None
    try:
        review = ws.read_json("drafts/review_output.json")
        score = review.get("overall_score")
    except FileNotFoundError:
        pass

    lines = [
        f"## Current Paper State",
        f"- Title: {title_m.group(1).strip() if title_m else '?'}",
        f"- Sections: {', '.join(sections)}",
        f"- Pages: {pages or '?'}",
        f"- Inline figures: {fig_count}",
        f"- Citations: {cite_count}",
    ]
    if score is not None:
        lines.append(f"- Review score: {score}/10")
    return "\n".join(lines)


def enrich_tool_error(tool_name: str, exc: Exception) -> dict[str, Any]:
    """Build an error dict with context-aware recovery hints."""
    error_msg = str(exc)
    result: dict[str, Any] = {"error": error_msg}

    hints: dict[str, str] = {
        "regenerate_figure": "Try list_figures first, or check experiment results with read_experiment_results",
        "compile_pdf": "Try fix_latex to auto-diagnose and repair LaTeX errors",
        "fix_latex": "Check paper.tex for \\end{document} placement or broken \\cite{} references",
        "run_review": "Ensure paper.tex exists and compiles. Try compile_pdf first",
        "rewrite_section": "Check exact section name with get_paper_info first",
        "read_section": "Use get_paper_info to list all section names",
    }
    if tool_name in hints:
        result["recovery_hint"] = hints[tool_name]

    return result
