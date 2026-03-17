"""FORMAT_FIX agent — wraps Project_P to fix LaTeX formatting after REVIEW."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Locate Project_P package relative to nanoresearch repo root
_PROJECT_P_DIR = Path(__file__).resolve().parents[2] / "Project_P"


def _ensure_project_p_importable() -> bool:
    """Add Project_P to sys.path if available."""
    if _PROJECT_P_DIR.is_dir() and str(_PROJECT_P_DIR) not in sys.path:
        sys.path.insert(0, str(_PROJECT_P_DIR))
    try:
        import project_p  # noqa: F401
        return True
    except ImportError:
        logger.warning("Project_P not found at %s, FORMAT_FIX will be skipped", _PROJECT_P_DIR)
        return False


class FormatFixAgent(BaseResearchAgent):
    """Runs Project_P deterministic + optional LLM fixes on the paper."""

    stage = PipelineStage.FORMAT_FIX

    async def run(self, **inputs: Any) -> dict[str, Any]:
        if not _ensure_project_p_importable():
            return {"skipped": True, "reason": "Project_P not available"}

        drafts_dir = self.workspace.path / "drafts"
        figures_dir = self.workspace.path / "figures"
        tex_path = drafts_dir / "paper.tex"

        if not tex_path.exists():
            logger.warning("No paper.tex found in drafts/, skipping FORMAT_FIX")
            return {"skipped": True, "reason": "paper.tex not found"}

        # Prepare a temp working directory with the structure Project_P expects
        with tempfile.TemporaryDirectory(prefix="format_fix_") as tmp:
            work_dir = Path(tmp)
            fig_subdir = work_dir / "figures"
            fig_subdir.mkdir()

            # Copy paper.tex
            tex_content = tex_path.read_text(errors="replace")
            # Ensure \graphicspath is present so tectonic finds figures/
            if r"\graphicspath" not in tex_content:
                tex_content = tex_content.replace(
                    r"\begin{document}",
                    r"\graphicspath{{figures/}}" + "\n" + r"\begin{document}",
                )
            (work_dir / "paper.tex").write_text(tex_content)

            # Copy references.bib
            bib_path = drafts_dir / "references.bib"
            if bib_path.exists():
                shutil.copy2(bib_path, work_dir / "references.bib")

            # Copy style file
            sty_path = drafts_dir / "neurips_2025.sty"
            if sty_path.exists():
                shutil.copy2(sty_path, work_dir / "neurips_2025.sty")

            # Copy figures from both figures/ and drafts/ (some pipelines put them in drafts/)
            for src_dir in [figures_dir, drafts_dir]:
                if not src_dir.exists():
                    continue
                for fig_file in src_dir.iterdir():
                    if fig_file.suffix.lower() in (".png", ".pdf", ".jpg", ".jpeg", ".eps"):
                        dest = fig_subdir / fig_file.name
                        if not dest.exists():
                            shutil.copy2(fig_file, dest)

            # Build Project_P config
            from project_p.config import Config, LLMConfig
            tectonic_path = shutil.which("tectonic") or "tectonic"
            pp_config = Config(
                tectonic_path=tectonic_path,
                llm=LLMConfig(),  # empty api_key → LLM disabled
            )

            # Run fix_paper in a thread (it's synchronous)
            from project_p.pipeline import fix_paper
            result = await asyncio.to_thread(
                fix_paper,
                work_dir,
                use_llm=False,
                do_compile=True,
                dry_run=False,
                config=pp_config,
            )

            if "error" in result:
                logger.error("Project_P fix_paper failed: %s", result["error"])
                return {"error": result["error"], "fixes": []}

            fixes = result.get("fixes", [])
            logger.info("Project_P applied %d fixes: %s", len(fixes), fixes)

            # Copy fixed files back to workspace
            fixed_tex = work_dir / "paper.tex"
            if fixed_tex.exists():
                shutil.copy2(fixed_tex, tex_path)
                self.workspace.register_artifact("paper_tex", tex_path, self.stage)

            fixed_bib = work_dir / "references.bib"
            if fixed_bib.exists():
                shutil.copy2(fixed_bib, bib_path)
                self.workspace.register_artifact("references_bib", bib_path, self.stage)

            fixed_pdf = work_dir / "paper.pdf"
            pdf_dest = drafts_dir / "paper.pdf"
            compile_success = False
            if fixed_pdf.exists():
                shutil.copy2(fixed_pdf, pdf_dest)
                self.workspace.register_artifact("paper_pdf", pdf_dest, self.stage)
                compile_success = True
                logger.info("FORMAT_FIX: PDF compiled successfully")

            # Build output
            output = {
                "fixes": fixes,
                "compile_success": compile_success,
                "num_fixes": len(fixes),
            }

            # Save output JSON
            output_path = drafts_dir / "format_fix_output.json"
            self.workspace.write_json("drafts/format_fix_output.json", output)
            self.workspace.register_artifact("format_fix_output", output_path, self.stage)

            return output
