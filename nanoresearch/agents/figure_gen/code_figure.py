"""Code-based chart generation (matplotlib) mixin."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

from ._constants import (
    CHART_CODE_SYSTEM,
    CHART_EXEC_TIMEOUT,
    MAX_CODE_CHART_RETRIES,
    MAX_FIG_ASPECT_RATIO,
    MAX_FIG_HEIGHT_PX,
    _FIGURE_CODE_PREAMBLE,
    _run_chart_subprocess,
)

logger = logging.getLogger(__name__)


class _CodeFigureMixin:
    """Mixin — matplotlib code-based chart generation."""

    async def _generate_code_figure(
        self,
        fig_key: str,
        output_path: str,
        user_prompt: str,
        caption: str,
    ) -> dict[str, Any]:
        """Have LLM generate plotting code, then execute it to create the chart.

        Retries up to MAX_CODE_CHART_RETRIES times, feeding error messages
        back to the LLM so it can fix matplotlib API issues, missing imports, etc.
        """
        filename_stem = fig_key
        figure_code_config = self.config.for_stage("figure_code")
        png_path = Path(output_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        last_error = ""
        prev_error = ""

        for attempt in range(MAX_CODE_CHART_RETRIES):
            # Early-exit if the same error repeats (LLM can't fix it)
            if attempt >= 2 and last_error and last_error == prev_error:
                self.log(f"  {fig_key} same error repeated — stopping retry loop")
                break
            prev_error = last_error

            # Build prompt — on retry, include the error feedback
            current_prompt = user_prompt
            if last_error:
                current_prompt += (
                    f"\n\n=== PREVIOUS ATTEMPT FAILED (attempt {attempt}) ===\n"
                    f"Error:\n{last_error[:1500]}\n\n"
                    f"Common fixes:\n"
                    f"- 'capthick' does NOT exist in matplotlib — remove it entirely\n"
                    f"- Check that all kwargs are valid for your matplotlib version\n"
                    f"- Ensure the output path is exactly: {output_path}\n"
                    f"- Use fig.tight_layout() before saving\n"
                    f"=== FIX THE ERROR AND REGENERATE THE COMPLETE CODE ==="
                )

            # Step 1: LLM generates the plotting script
            try:
                code = await self._dispatcher.generate(
                    figure_code_config, CHART_CODE_SYSTEM, current_prompt
                )
            except Exception as e:
                last_error = f"LLM generation error: {e}"
                self.log(f"  {fig_key} attempt {attempt + 1}/{MAX_CODE_CHART_RETRIES} LLM failed: {e}")
                continue

            code = code.strip()
            # Strip markdown fences if present
            if code.startswith("```"):
                lines = code.split("\n")
                lines = [l for l in lines[1:] if not l.strip().startswith("```")]
                code = "\n".join(lines)

            # Inject preamble: enforce sane rcParams in the subprocess
            # Strip any imports the LLM wrote that conflict with the preamble
            # (matplotlib, numpy, seaborn, ticker are all provided by preamble)
            code = re.sub(
                r"^import matplotlib(?:\.\w+)? as .*$|"
                r"^import matplotlib$|"
                r"^from matplotlib(?:\.\w+)? import .*$|"
                r"^matplotlib\.use\(.*\)$|"
                r"^mpl\.use\(.*\)$|"
                r"^import matplotlib\.pyplot as plt$|"
                r"^import numpy as np$|"
                r"^import seaborn as sns$",
                "", code, flags=re.MULTILINE,
            )
            code = _FIGURE_CODE_PREAMBLE + code

            # Save the generated code for debugging/reproducibility
            code_path = self.workspace.write_text(
                f"figures/{filename_stem}_plot.py", code
            )
            self.log(f"  {fig_key} attempt {attempt + 1} code generated ({len(code)} chars)")

            # Step 2: Execute the plotting script
            try:
                loop = asyncio.get_running_loop()
                python_exe = self._resolve_experiment_python()
                result = await loop.run_in_executor(
                    None,
                    partial(
                        _run_chart_subprocess,
                        [python_exe, str(code_path)],
                        timeout=CHART_EXEC_TIMEOUT,
                        cwd=str(self.workspace.path),
                    ),
                )
                if result["returncode"] != 0:
                    last_error = result["stderr"][:1500]
                    self.log(f"  {fig_key} attempt {attempt + 1} execution failed: {last_error[:300]}")
                    self.workspace.write_text(
                        f"logs/{filename_stem}_error.log",
                        f"STDOUT:\n{result['stdout']}\n\nSTDERR:\n{result['stderr']}",
                    )
                    continue
            except subprocess.TimeoutExpired:
                last_error = f"Execution timed out after {CHART_EXEC_TIMEOUT}s"
                self.log(f"  {fig_key} attempt {attempt + 1} timed out")
                continue
            except Exception as exc:
                last_error = str(exc)
                self.log(f"  {fig_key} attempt {attempt + 1} error: {exc}")
                continue

            # Step 3: Verify PNG was created
            # LLMs often ignore absolute output_path and save to relative
            # path instead.  Search likely locations before giving up.
            if not png_path.exists():
                _ws = Path(self.workspace.path)
                # LLMs often ignore the absolute output_path and use a
                # bare filename in plt.savefig().  With cwd=workspace the
                # PNG lands in the workspace root instead of figures/.
                _alt_candidates = [
                    _ws / f"{fig_key}.png",                   # cwd-relative (most common)
                    _ws / "experiment" / f"{fig_key}.png",     # saved in experiment dir
                    _ws / "experiment" / "results" / f"{fig_key}.png",
                ]
                _found_alt = None
                for _alt in _alt_candidates:
                    if _alt.exists() and _alt != png_path:
                        _found_alt = _alt
                        break
                if _found_alt:
                    import shutil as _shutil
                    _shutil.move(str(_found_alt), str(png_path))
                    self.log(
                        f"  {fig_key} attempt {attempt + 1}: PNG found at "
                        f"{_found_alt.name}, moved to figures/"
                    )
                    # Also move companion PDF if it exists
                    _alt_pdf = _found_alt.with_suffix(".pdf")
                    if _alt_pdf.exists():
                        _shutil.move(
                            str(_alt_pdf),
                            str(png_path.with_suffix(".pdf")),
                        )
                else:
                    last_error = (
                        f"Code ran successfully but PNG not generated at "
                        f"{output_path}. IMPORTANT: You MUST use this exact "
                        f"output path in plt.savefig()."
                    )
                    self.log(f"  {fig_key} attempt {attempt + 1}: {last_error}")
                    continue

            # Step 3b: Validate image dimensions — reject absurd sizes
            try:
                from PIL import Image as _PILImage
                with _PILImage.open(png_path) as _img:
                    _w, _h = _img.size
                self.log(f"  {fig_key} output size: {_w}x{_h}")
                aspect = _h / max(_w, 1)
                if _h > MAX_FIG_HEIGHT_PX and aspect > MAX_FIG_ASPECT_RATIO:
                    last_error = (
                        f"Figure too tall: {_w}x{_h} pixels "
                        f"(aspect {aspect:.1f} > {MAX_FIG_ASPECT_RATIO}). "
                        f"Use a smaller figsize like (7, 4.3) or (7, 5) "
                        f"and call fig.tight_layout(). "
                        f"Do NOT use figsize with height > 8 inches."
                    )
                    self.log(f"  {fig_key} attempt {attempt + 1} rejected: {last_error}")
                    png_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass  # PIL not available or file invalid — let it through

            self.log(f"  {fig_key} saved (attempt {attempt + 1})")
            return await self._save_figure_files(fig_key, filename_stem, caption,
                                                 png_path.read_bytes(), already_saved=True,
                                                 code_generated=True)

        # All retries exhausted — use fallback placeholder
        self.log(f"  {fig_key} all {MAX_CODE_CHART_RETRIES} attempts failed, using fallback")
        result = await self._generate_fallback_chart(fig_key, filename_stem, caption)
        result["is_fallback"] = True
        return result

    async def _generate_fallback_chart(
        self, fig_key: str, filename_stem: str, caption: str,
    ) -> dict[str, Any]:
        """Return a failed status dict — do NOT generate a placeholder image.

        Previously this generated a "Chart generation failed" placeholder PNG.
        Now we simply mark the figure as failed so the LaTeX assembler skips it
        entirely, rather than embedding a useless placeholder in the paper.
        """
        self.log(f"  {fig_key} chart generation failed — marking as failed (no placeholder)")
        return {
            "fig_key": fig_key,
            "caption": caption,
            "status": "failed",
            "error": "Chart generation failed after all retries",
        }

    # -----------------------------------------------------------------------
    # Shared: save PNG + PDF + register artifacts
    # -----------------------------------------------------------------------
