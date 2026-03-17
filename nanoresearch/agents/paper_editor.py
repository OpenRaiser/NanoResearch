"""Paper autopilot: ReAct agent that understands user intent and orchestrates
multiple modules to edit research papers.

Split into 3 modules:
  - paper_snapshot.py: PaperSnapshotManager + tool handlers + system prompt
  - paper_condenser.py: ConversationCondenser
  - paper_editor.py: PaperEditor (this file)
"""

from __future__ import annotations

import functools
import json
import logging
import re
import zipfile
from collections import Counter
from typing import Any

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.pipeline.workspace import Workspace

# ── Re-exports for backward compatibility ──────────────────────────────
from nanoresearch.agents.paper_snapshot import (  # noqa: F401
    PaperSnapshotManager,
    _tool_get_paper_info,
    _tool_read_section,
    _tool_rewrite_section,
    _tool_edit_tex,
    _tool_list_figures,
    _tool_regenerate_figure,
    _tool_run_review,
    _tool_compile_pdf,
    _tool_fix_latex,
    _tool_read_experiment_results,
    _count_pdf_pages,
    _load_stage_data,
)
from nanoresearch.agents.paper_condenser import (  # noqa: F401
    ConversationCondenser,
    PAPER_AGENT_SYSTEM,
    build_paper_context,
    enrich_tool_error,
)

logger = logging.getLogger(__name__)

# Legacy aliases (private-name convention from original monolith)
_PAPER_AGENT_SYSTEM = PAPER_AGENT_SYSTEM
_build_paper_context = build_paper_context
_enrich_tool_error = enrich_tool_error


class PaperEditor:
    """ReAct agent that orchestrates paper editing via tool calls."""

    def __init__(
        self,
        workspace: Workspace,
        config: ResearchConfig,
        log_fn: Any = None,
    ) -> None:
        self.ws = workspace
        self.config = config
        self.dispatcher = ModelDispatcher(config)
        self.snapshot_mgr = PaperSnapshotManager(workspace)
        self._log = log_fn or (lambda msg: None)
        self._history: list[dict[str, Any]] = []
        self._tools = self._build_tools()
        # Event-sourced conversation memory with LLM condensation
        self._condenser = ConversationCondenser(self.dispatcher, self.config)

    def _build_tools(self) -> ToolRegistry:
        """Register all paper editing tools."""
        reg = ToolRegistry()
        ws = self.ws
        disp = self.dispatcher
        cfg = self.config

        reg.register(ToolDefinition(
            name="get_paper_info",
            description=(
                "Get paper overview: title, sections (with character lengths), "
                "page count, figure count, citation count, review score."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=functools.partial(_tool_get_paper_info, ws),
        ))

        reg.register(ToolDefinition(
            name="read_section",
            description="Read the full LaTeX content of a specific section.",
            parameters={
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "description": "Section heading (e.g. 'Introduction', 'Method')",
                    },
                },
                "required": ["section_name"],
            },
            handler=functools.partial(_tool_read_section, ws),
        ))

        reg.register(ToolDefinition(
            name="rewrite_section",
            description=(
                "Rewrite a section with specific instructions. "
                "E.g. shorten it, change focus, add content, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "description": "Section heading to rewrite",
                    },
                    "instruction": {
                        "type": "string",
                        "description": (
                            "Detailed instruction: what to change, target length, "
                            "style, what to preserve or remove"
                        ),
                    },
                },
                "required": ["section_name", "instruction"],
            },
            handler=functools.partial(
                _tool_rewrite_section, ws, disp, cfg,
            ),
        ))

        reg.register(ToolDefinition(
            name="edit_tex",
            description=(
                "Direct search/replace on paper.tex. Use for small, precise edits "
                "(fix a typo, change a number, swap a word)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "old": {
                        "type": "string",
                        "description": "Exact substring to find in paper.tex",
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                },
                "required": ["old", "new"],
            },
            handler=functools.partial(_tool_edit_tex, ws),
        ))

        reg.register(ToolDefinition(
            name="list_figures",
            description="List all figure files with metadata (filename, size, caption, chart type).",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=functools.partial(_tool_list_figures, ws),
        ))

        reg.register(ToolDefinition(
            name="regenerate_figure",
            description=(
                "Regenerate paper figures using FigureAgent. "
                "Can target a specific figure or regenerate all. "
                "Provide style instructions for visual changes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fig_description": {
                        "type": "string",
                        "description": "What the figures should show / what to change",
                    },
                    "style_instruction": {
                        "type": "string",
                        "description": "Visual style (e.g. 'blue-green palette, clean white background')",
                    },
                    "target_figure": {
                        "type": "string",
                        "description": "Specific figure key to regenerate (e.g. 'fig2_main_results'). Leave empty to regenerate all.",
                    },
                },
                "required": ["fig_description"],
            },
            handler=functools.partial(
                _tool_regenerate_figure, ws, disp, cfg,
            ),
        ))

        reg.register(ToolDefinition(
            name="fix_latex",
            description=(
                "Run LaTeX error diagnosis + auto-fix + recompile. "
                "Use when compile_pdf fails. Automatically diagnoses errors "
                "and applies fixes (up to 3 retry rounds)."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=functools.partial(_tool_fix_latex, ws, disp, cfg),
        ))

        reg.register(ToolDefinition(
            name="run_review",
            description=(
                "Run the full ReviewAgent: scores each section, identifies issues, "
                "and auto-revises sections with low scores."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=functools.partial(_tool_run_review, ws, disp, cfg),
        ))

        reg.register(ToolDefinition(
            name="compile_pdf",
            description="Compile paper.tex to PDF and return the page count.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=functools.partial(_tool_compile_pdf, ws),
        ))

        reg.register(ToolDefinition(
            name="read_experiment_results",
            description="Read experiment results JSON for fact-checking or data inspection.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=functools.partial(_tool_read_experiment_results, ws),
        ))

        return reg

    # ── main entry ────────────────────────────────────────────────────

    async def apply_instruction(self, instruction: str) -> dict[str, Any]:
        """Run the ReAct agent loop to fulfill the user's instruction."""
        # Step 1: backup
        snap_id = self.snapshot_mgr.create_snapshot(label="before_edit")
        self._log(f"Backup: {snap_id}")

        # Step 2: build messages with condensed memory
        cfg = self.config.for_stage("code_gen")
        openai_tools = self._tools.to_openai_tools()

        paper_ctx = build_paper_context(self.ws)
        system_content = PAPER_AGENT_SYSTEM
        if paper_ctx:
            system_content += "\n\n" + paper_ctx

        def _build_messages() -> list[dict[str, Any]]:
            msgs: list[dict[str, Any]] = [
                {"role": "system", "content": system_content},
            ]
            msgs.extend(self._condenser.to_messages())
            msgs.append({"role": "user", "content": instruction})
            return msgs

        messages = _build_messages()

        max_rounds = 25
        tools_called: list[str] = []
        final_text = ""
        _context_retried = False

        for round_idx in range(max_rounds):
            try:
                msg = await self.dispatcher.generate_with_tools(
                    cfg, messages, openai_tools,
                )
            except Exception as exc:
                err_str = str(exc).lower()
                if not _context_retried and (
                    "context" in err_str or "token" in err_str
                    or "too long" in err_str
                ):
                    self._log("Context overflow -- hard-condensing memory...")
                    condensed = await self._condenser.condense(hard=True)
                    if condensed:
                        messages = _build_messages()
                        _context_retried = True
                        continue
                raise

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                final_text = msg.content or ""
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            if msg.content:
                assistant_msg["content"] = msg.content
            messages.append(assistant_msg)

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                self._log(f"Tool: {name}({', '.join(f'{k}={v!r}' for k, v in list(args.items())[:3])})")
                tools_called.append(name)

                try:
                    result = await self._tools.call(name, args)
                except Exception as exc:
                    result = enrich_tool_error(name, exc)

                result_str = json.dumps(result, ensure_ascii=False, default=str)
                if len(result_str) > 15000:
                    result_str = result_str[:15000] + "\n... [truncated]"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
        else:
            final_text = "(Agent reached max rounds)"

        # Step 3: save to condenser
        self._condenser.append({"role": "user", "content": instruction})
        if tools_called:
            tc_counts = Counter(tools_called)
            tool_str = ", ".join(
                f"{n}(x{c})" if c > 1 else n for n, c in tc_counts.items()
            )
            assistant_content = f"[Tools used: {tool_str}]\n{final_text}"
        else:
            assistant_content = final_text or "(no response)"
        self._condenser.append({"role": "assistant", "content": assistant_content})

        if self._condenser.needs_condensation():
            self._log("Auto-condensing conversation memory...")
            await self._condenser.condense()

        self._history.append({
            "instruction": instruction,
            "snapshot_id": snap_id,
            "tools_called": tools_called,
        })

        return {
            "snapshot_id": snap_id,
            "tools_called": tools_called,
            "tool_count": len(tools_called),
            "summary": final_text,
        }

    # ── undo ──────────────────────────────────────────────────────────

    def undo(self) -> str | None:
        return self.snapshot_mgr.rollback_latest()

    def rollback_to(self, snapshot_id: str) -> bool:
        return self.snapshot_mgr.rollback(snapshot_id)

    def diff_snapshots(
        self, snap_a: str | None = None, snap_b: str | None = None,
    ) -> dict[str, Any]:
        """Compare two snapshots (or latest snapshot vs current state)."""
        def _read_tex_from_zip(snap_id: str) -> str | None:
            zp = self.snapshot_mgr.backup_dir / f"{snap_id}.zip"
            if not zp.exists():
                return None
            with zipfile.ZipFile(zp, "r") as zf:
                for name in zf.namelist():
                    if name.endswith("paper.tex"):
                        return zf.read(name).decode("utf-8", errors="replace")
            return None

        def _section_lengths(tex: str) -> dict[str, int]:
            sections = re.findall(r"\\section\{([^}]+)\}", tex)
            lengths: dict[str, int] = {}
            for i, sec in enumerate(sections):
                start = tex.find(f"\\section{{{sec}}}")
                if start < 0:
                    continue
                end = len(tex)
                for j in range(i + 1, len(sections)):
                    nxt = tex.find(f"\\section{{{sections[j]}}}", start + 1)
                    if nxt > start:
                        end = nxt
                        break
                ed = tex.find("\\end{document}", start)
                if 0 < ed < end:
                    end = ed
                lengths[sec] = end - start
            return lengths

        if snap_a is None:
            snaps = self.snapshot_mgr.list_snapshots()
            if not snaps:
                return {"error": "No snapshots available"}
            snap_a = snaps[-1]["id"]

        tex_a = _read_tex_from_zip(snap_a)
        if tex_a is None:
            return {"error": f"Snapshot '{snap_a}' not found or has no paper.tex"}

        if snap_b is None:
            tex_path = self.ws.path / "drafts" / "paper.tex"
            if not tex_path.exists():
                return {"error": "No paper.tex in current workspace"}
            tex_b = tex_path.read_text(encoding="utf-8", errors="replace")
            snap_b_label = "current"
        else:
            tex_b = _read_tex_from_zip(snap_b)
            if tex_b is None:
                return {"error": f"Snapshot '{snap_b}' not found"}
            snap_b_label = snap_b

        len_a = _section_lengths(tex_a)
        len_b = _section_lengths(tex_b)
        all_sections = list(dict.fromkeys(list(len_a) + list(len_b)))

        section_diff: list[dict[str, Any]] = []
        for sec in all_sections:
            a = len_a.get(sec, 0)
            b = len_b.get(sec, 0)
            delta = b - a
            pct = f"{delta / max(a, 1) * 100:+.0f}%" if a > 0 else ("new" if b > 0 else "removed")
            section_diff.append({
                "section": sec, "before": a, "after": b,
                "delta": delta, "change": pct,
            })

        fig_a = len(re.findall(r"\\begin\{figure\}", tex_a))
        fig_b = len(re.findall(r"\\begin\{figure\}", tex_b))
        cite_a = len(re.findall(r"\\cite\{", tex_a))
        cite_b = len(re.findall(r"\\cite\{", tex_b))

        return {
            "snap_a": snap_a,
            "snap_b": snap_b_label,
            "total_chars": {"before": len(tex_a), "after": len(tex_b)},
            "sections": section_diff,
            "figures": {"before": fig_a, "after": fig_b},
            "citations": {"before": cite_a, "after": cite_b},
        }
