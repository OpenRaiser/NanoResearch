"""Snapshot manager + tool handlers for paper editing.

PaperSnapshotManager provides zip-based backup for drafts/ + figures/.
Tool handler functions wrap pipeline modules for ReAct tool dispatch.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.multi_model import ModelDispatcher
from nanoresearch.pipeline.workspace import Workspace

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot manager  (zip-based backup for drafts/ + figures/)
# ═══════════════════════════════════════════════════════════════════════════

class PaperSnapshotManager:
    """Zip-based backup for the drafts/ + figures/ directories."""

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace
        self.backup_dir = workspace.path / "snapshots"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self, label: str = "") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_id = f"{ts}_{label}" if label else ts
        zp = self.backup_dir / f"{snap_id}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for subdir in ("drafts", "figures"):
                d = self.ws.path / subdir
                if d.is_dir():
                    for fp in d.rglob("*"):
                        if fp.is_file():
                            zf.write(fp, f"{subdir}/{fp.relative_to(d)}")
        return snap_id

    def list_snapshots(self) -> list[dict[str, Any]]:
        snaps: list[dict[str, Any]] = []
        for zp in sorted(self.backup_dir.glob("*.zip")):
            snaps.append({
                "id": zp.stem,
                "size_kb": round(zp.stat().st_size / 1024, 1),
                "created": datetime.fromtimestamp(zp.stat().st_mtime).isoformat(),
            })
        return snaps

    def rollback(self, snapshot_id: str) -> bool:
        zp = self.backup_dir / f"{snapshot_id}.zip"
        if not zp.exists():
            return False
        with zipfile.ZipFile(zp, "r") as zf:
            for name in zf.namelist():
                target = self.ws.path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
        return True

    def rollback_latest(self) -> str | None:
        snaps = self.list_snapshots()
        if not snaps:
            return None
        return snaps[-1]["id"] if self.rollback(snaps[-1]["id"]) else None


# ═══════════════════════════════════════════════════════════════════════════
# Tool handlers — each wraps a pipeline module
# ═══════════════════════════════════════════════════════════════════════════

async def _tool_get_paper_info(ws: Workspace, **_kw: Any) -> dict[str, Any]:
    """Read paper metadata: title, sections, page count, figure count, citations."""
    tex_path = ws.path / "drafts" / "paper.tex"
    if not tex_path.exists():
        return {"error": "paper.tex not found"}
    tex = tex_path.read_text(encoding="utf-8", errors="replace")

    title_m = re.search(r"\\title\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)\}", tex)
    sections = re.findall(r"\\section\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+)\}", tex)
    fig_count = len(re.findall(r"\\begin\{figure\}", tex))
    cite_count = len(re.findall(r"\\cite\{", tex))

    # Section lengths (approx chars)
    sec_lengths: dict[str, int] = {}
    for i, sec in enumerate(sections):
        pat = re.escape(sec)
        start = re.search(r"\\section\{" + pat + r"\}", tex)
        if start:
            end_pos = len(tex)
            for j, next_sec in enumerate(sections):
                if j > i:
                    nxt = re.search(r"\\section\{" + re.escape(next_sec) + r"\}", tex)
                    if nxt and nxt.start() > start.start():
                        end_pos = nxt.start()
                        break
            ed = re.search(r"\\end\{document\}", tex)
            if ed and ed.start() < end_pos:
                end_pos = ed.start()
            sec_lengths[sec] = end_pos - start.start()

    # Page count from PDF
    pages = _count_pdf_pages(ws.path / "drafts" / "paper.pdf")

    # Review score
    score = None
    try:
        review = ws.read_json("drafts/review_output.json")
        score = review.get("overall_score")
    except FileNotFoundError:
        pass

    return {
        "title": title_m.group(1).strip() if title_m else "?",
        "sections": sections,
        "section_char_lengths": sec_lengths,
        "inline_figures": fig_count,
        "citations": cite_count,
        "pages": pages,
        "review_score": score,
        "tex_total_chars": len(tex),
    }


async def _tool_read_section(
    ws: Workspace, section_name: str, **_kw: Any,
) -> dict[str, Any]:
    """Read the LaTeX content of a specific section."""
    tex = (ws.path / "drafts" / "paper.tex").read_text("utf-8", errors="replace")
    pattern = (
        r"(\\section\{" + re.escape(section_name) + r"\})"
        r"(.*?)"
        r"(?=\\section\{|\\end\{document\})"
    )
    m = re.search(pattern, tex, re.DOTALL)
    if not m:
        # case-insensitive partial
        pattern2 = (
            r"(\\section\{[^}]*" + re.escape(section_name) + r"[^}]*\})"
            r"(.*?)"
            r"(?=\\section\{|\\end\{document\})"
        )
        m = re.search(pattern2, tex, re.DOTALL | re.IGNORECASE)
    if not m:
        return {"error": f"Section '{section_name}' not found"}
    content = m.group(2).strip()
    return {
        "heading": m.group(1),
        "content": content,
        "char_length": len(content),
        "approx_lines": content.count("\n") + 1,
    }


async def _tool_rewrite_section(
    ws: Workspace, dispatcher: ModelDispatcher, config: ResearchConfig,
    section_name: str, instruction: str, **_kw: Any,
) -> dict[str, Any]:
    """Rewrite a section using the LLM with specific instructions."""
    tex = (ws.path / "drafts" / "paper.tex").read_text("utf-8", errors="replace")
    pattern = (
        r"(\\section\{[^}]*" + re.escape(section_name) + r"[^}]*\})"
        r"(.*?)"
        r"(?=\\section\{|\\end\{document\})"
    )
    m = re.search(pattern, tex, re.DOTALL | re.IGNORECASE)
    if not m:
        return {"error": f"Section '{section_name}' not found"}

    heading = m.group(1)
    old_content = m.group(2)

    system = (
        f"You are rewriting the \"{section_name}\" section of a research paper.\n"
        f"Follow the user's instruction precisely.\n"
        f"Output ONLY the section body LaTeX (no \\section heading, no \\end{{document}}).\n"
        f"Preserve all \\cite{{}} and \\ref{{}} commands.\n"
        f"Preserve \\begin{{figure}}...\\end{{figure}} blocks unless told to remove.\n"
        f"Academic tone, concise."
    )
    user = (
        f"## Current content\n\n{old_content.strip()}\n\n"
        f"## Instruction\n\n{instruction}"
    )
    cfg = config.for_stage("writing")
    new_content = await dispatcher.generate(cfg, system, user)
    # Strip any accidental fences
    new_content = re.sub(r"^```\w*\n?", "", new_content)
    new_content = re.sub(r"\n?```$", "", new_content)

    old_block = heading + old_content
    new_block = heading + "\n" + new_content.strip() + "\n\n"
    tex = tex.replace(old_block, new_block, 1)
    (ws.path / "drafts" / "paper.tex").write_text(tex, encoding="utf-8")

    return {
        "status": "ok",
        "section": section_name,
        "old_chars": len(old_content),
        "new_chars": len(new_content),
        "change_pct": f"{(len(new_content) - len(old_content)) / max(len(old_content), 1) * 100:+.0f}%",
    }


async def _tool_edit_tex(
    ws: Workspace, old: str, new: str, **_kw: Any,
) -> dict[str, Any]:
    """Direct search/replace on paper.tex."""
    tex_path = ws.path / "drafts" / "paper.tex"
    tex = tex_path.read_text("utf-8", errors="replace")
    if old not in tex:
        return {"error": f"String not found (first 60 chars): {old[:60]!r}"}
    tex = tex.replace(old, new, 1)
    tex_path.write_text(tex, encoding="utf-8")
    return {"status": "ok", "replaced": True}


async def _tool_list_figures(ws: Workspace, **_kw: Any) -> dict[str, Any]:
    """List all figure files and their info."""
    fig_dir = ws.path / "figures"
    if not fig_dir.is_dir():
        return {"figures": []}
    figs = []
    for fp in sorted(fig_dir.iterdir()):
        if fp.suffix in (".png", ".pdf", ".jpg", ".jpeg"):
            figs.append({
                "filename": fp.name,
                "size_kb": round(fp.stat().st_size / 1024, 1),
                "type": fp.suffix,
            })
    # Also check figure_output.json for metadata
    try:
        fig_output = ws.read_json("drafts/figure_output.json")
        figures_data = fig_output.get("figures", {})
        if isinstance(figures_data, dict):
            for key, info in figures_data.items():
                if not isinstance(info, dict):
                    continue
                for f in figs:
                    if key in f["filename"]:
                        f["fig_key"] = key
                        f["caption"] = info.get("caption", "")[:100]
                        f["chart_type"] = info.get("chart_type", info.get("fig_type", ""))
    except (FileNotFoundError, Exception):
        pass
    return {"figures": figs, "count": len(figs)}


async def _tool_regenerate_figure(
    ws: Workspace, dispatcher: ModelDispatcher, config: ResearchConfig,
    fig_description: str, style_instruction: str = "",
    target_figure: str = "",
    **_kw: Any,
) -> dict[str, Any]:
    """Regenerate figures using FigureAgent with style/content guidance.

    If target_figure is set, only that figure key is regenerated.
    Otherwise all figures are regenerated.
    """
    try:
        from nanoresearch.agents.figure_gen import FigureAgent
    except ImportError as exc:
        return {"error": f"FigureAgent not available: {exc}"}

    stage_data = _load_stage_data(ws)
    agent = FigureAgent(ws, config)
    agent._dispatcher = dispatcher

    # Inject style guidance into the blueprint
    bp = dict(stage_data.get("experiment_blueprint", {}))
    if style_instruction:
        bp["figure_style_override"] = style_instruction
    if fig_description:
        bp["figure_description_override"] = fig_description
    if target_figure:
        bp["target_figure_key"] = target_figure

    try:
        output = await agent.run(
            experiment_blueprint=bp,
            ideation_output=stage_data.get("ideation_output", {}),
            experiment_results=stage_data.get("experiment_results", {}),
            experiment_status="completed",
        )
    except Exception as exc:
        return {"error": f"FigureAgent failed: {exc}"}

    figs = list(output.get("figures", {}).keys())
    return {
        "status": "ok",
        "figures_generated": figs,
        "count": len(figs),
    }


async def _tool_run_review(
    ws: Workspace, dispatcher: ModelDispatcher, config: ResearchConfig,
    **_kw: Any,
) -> dict[str, Any]:
    """Run ReviewAgent: score + revise the paper."""
    try:
        from nanoresearch.agents.review import ReviewAgent
    except ImportError as exc:
        return {"error": f"ReviewAgent not available: {exc}"}

    tex = (ws.path / "drafts" / "paper.tex").read_text("utf-8", errors="replace")
    stage_data = _load_stage_data(ws)
    agent = ReviewAgent(ws, config)
    agent._dispatcher = dispatcher

    try:
        output = await agent.run(
            paper_tex=tex,
            ideation_output=stage_data.get("ideation_output", {}),
            experiment_blueprint=stage_data.get("experiment_blueprint", {}),
            experiment_results=stage_data.get("experiment_results", {}),
        )
    except Exception as exc:
        return {"error": f"ReviewAgent failed: {exc}"}

    return {
        "status": "ok",
        "overall_score": output.get("overall_score"),
        "revision_rounds": output.get("revision_rounds", 0),
        "sections_revised": list(output.get("revised_sections", {}).keys()),
        "major_revisions": output.get("major_revisions", []),
    }


async def _tool_compile_pdf(ws: Workspace, **_kw: Any) -> dict[str, Any]:
    """Compile paper.tex to PDF and return page count."""
    try:
        from mcp_server.tools.pdf_compile import compile_pdf
        tex_path = str(ws.path / "drafts" / "paper.tex")
        result = compile_pdf(tex_path)
        if result.get("pdf_path"):
            pages = _count_pdf_pages(Path(result["pdf_path"]))
            return {"status": "ok", "pdf_path": result["pdf_path"], "pages": pages}
        return {"error": result.get("error", "Compilation failed")}
    except Exception as exc:
        return {"error": f"Compile error: {exc}"}


async def _tool_fix_latex(
    ws: Workspace, dispatcher: ModelDispatcher, config: ResearchConfig,
    **_kw: Any,
) -> dict[str, Any]:
    """Run LaTeX sanitiser + compile fix loop (auto-diagnose and repair errors)."""
    try:
        from nanoresearch.agents.writing import WritingAgent
    except ImportError as exc:
        return {"error": f"WritingAgent not available: {exc}"}

    agent = WritingAgent(ws, config)
    agent._dispatcher = dispatcher
    tex_path = ws.path / "drafts" / "paper.tex"
    try:
        result = await agent._compile_pdf(
            tex_path, max_fix_attempts=3,
            template_format=config.template_format,
        )
    except Exception as exc:
        return {"error": f"LaTeX fix failed: {exc}"}

    if "pdf_path" in result:
        pages = _count_pdf_pages(Path(result["pdf_path"]))
        return {"status": "ok", "pdf_path": result["pdf_path"], "pages": pages}
    return {"error": result.get("error", "Fix+compile failed")}


async def _tool_read_experiment_results(ws: Workspace, **_kw: Any) -> dict[str, Any]:
    """Read experiment results for grounding."""
    for rpath in [
        "drafts/experiment_results.json",
        "experiment/results.json",
        "plans/execution_results.json",
    ]:
        try:
            data = ws.read_json(rpath)
            # Truncate if too large
            text = json.dumps(data, indent=2, ensure_ascii=False)
            if len(text) > 10000:
                text = text[:10000] + "\n... [truncated]"
            return {"source": rpath, "data": text}
        except FileNotFoundError:
            continue
    return {"error": "No experiment results found"}


# ── helper ────────────────────────────────────────────────────────────────

def _count_pdf_pages(pdf_path: Path) -> int | None:
    """Count pages in a PDF file. Returns None if unavailable."""
    if not pdf_path.exists():
        return None
    try:
        content = pdf_path.read_bytes()
        # Quick regex count of /Type /Page entries (approximate)
        pages = len(re.findall(rb"/Type\s*/Page[^s]", content))
        return pages if pages > 0 else None
    except Exception:
        return None


def _load_stage_data(ws: Workspace) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, path in [
        ("ideation_output", "papers/ideation_output.json"),
        ("experiment_blueprint", "plans/experiment_blueprint.json"),
        ("figure_output", "drafts/figure_output.json"),
    ]:
        try:
            data[key] = ws.read_json(path)
        except FileNotFoundError:
            data[key] = {}
    for rpath in [
        "drafts/experiment_results.json",
        "experiment/results.json",
        "plans/execution_results.json",
    ]:
        try:
            data["experiment_results"] = ws.read_json(rpath)
            break
        except FileNotFoundError:
            continue
    if "experiment_results" not in data:
        data["experiment_results"] = {}
    return data
