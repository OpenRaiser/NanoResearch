"""Figure trimming and cropping mixin."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _TrimMixin:
    """Mixin — figure trimming (deterministic + LLM-assisted)."""

    def _deterministic_trim(self, fig_key: str, png_path: Path) -> None:
        """Deterministic whitespace trimming — no LLM, always reliable.

        Two-pass approach:
        1. Basic bbox crop (remove outer whitespace margins)
        2. Interior gap detection: if a huge vertical gap (>30% of height)
           exists between content regions, crop to the largest contiguous
           content block — this handles matplotlib multi-subplot figures
           where only some subplots have content.
        """
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return

        img = Image.open(png_path)
        w, h = img.size

        arr = np.array(img.convert("RGB"))
        gray = np.mean(arr, axis=2)
        non_white = gray < 245
        rows_mask = np.any(non_white, axis=1)
        cols_mask = np.any(non_white, axis=0)

        row_indices = np.where(rows_mask)[0]
        col_indices = np.where(cols_mask)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return

        margin = 30

        # --- Pass 2: detect large interior vertical gaps ---
        # If there's a gap > 30% of image height, the figure likely has
        # empty subplot areas. Keep only the largest contiguous content block.
        gaps = np.diff(row_indices)
        large_gap_threshold = h * 0.3
        large_gap_indices = np.where(gaps > large_gap_threshold)[0]

        if len(large_gap_indices) > 0:
            # Split row_indices into contiguous segments at large gaps
            split_points = [0] + [i + 1 for i in large_gap_indices] + [len(row_indices)]
            segments = []
            for i in range(len(split_points) - 1):
                seg = row_indices[split_points[i]:split_points[i + 1]]
                if len(seg) > 0:
                    segments.append((seg[0], seg[-1], len(seg)))

            # Keep the segment with the most content rows
            best_seg = max(segments, key=lambda s: s[2])
            top = max(0, int(best_seg[0]) - margin)
            bottom = min(h, int(best_seg[1]) + margin)
            left = max(0, int(col_indices[0]) - margin)
            right = min(w, int(col_indices[-1]) + margin)
        else:
            # No large gaps — standard bbox crop
            top = max(0, int(row_indices[0]) - margin)
            bottom = min(h, int(row_indices[-1]) + margin)
            left = max(0, int(col_indices[0]) - margin)
            right = min(w, int(col_indices[-1]) + margin)

        # Only trim if we'd remove at least 5% of area
        cropped_area = (right - left) * (bottom - top)
        original_area = w * h
        if cropped_area >= original_area * 0.95:
            self.log(f"  {fig_key} deterministic trim: no significant whitespace")
            return

        cropped = img.crop((left, top, right, bottom))
        cropped.save(png_path)
        cw, ch = cropped.size
        self.log(f"  {fig_key} deterministic trim: {w}x{h} -> {cw}x{ch}")

    async def _smart_trim_figure(self, fig_key: str, png_path: Path) -> None:
        """LLM-driven figure trimming.

        Flow:
        1. Send original image to LLM → LLM decides if trim needed
        2. If yes, LLM writes Python cropping code → execute it
        3. Send result to LLM for verification → APPROVE or REJECT with fix
        4. Max 2 rounds; on any failure fall back to original
        """
        import io
        from PIL import Image

        original_bytes = png_path.read_bytes()
        img = Image.open(io.BytesIO(original_bytes))
        w, h = img.size
        self.log(f"  {fig_key} trim check: {w}x{h}")

        # Round 1: LLM analyzes original image
        try:
            trim_plan = await self._llm_analyze_trim(fig_key, original_bytes, w, h)
        except Exception as e:
            self.log(f"  {fig_key} LLM trim analysis failed: {e}")
            return

        if not trim_plan.get("needs_trim"):
            self.log(f"  {fig_key} LLM says no trim needed")
            return

        code = trim_plan.get("code", "")
        if not code.strip():
            self.log(f"  {fig_key} LLM returned needs_trim but no code")
            return

        # Execute LLM's cropping code
        trimmed_path = png_path.parent / f"{png_path.stem}_trimmed.png"
        success = self._exec_trim_code(code, str(png_path), str(trimmed_path))

        if not success or not trimmed_path.exists():
            self.log(f"  {fig_key} trim code execution failed")
            return

        # Round 2: LLM verifies the trimmed result (max 2 rounds)
        import shutil
        trimmed_bytes = trimmed_path.read_bytes()
        accepted = False

        for verify_round in range(2):
            try:
                verdict = await self._llm_verify_trim(fig_key, trimmed_bytes)
            except Exception as e:
                self.log(f"  {fig_key} LLM verify failed: {e}, accepting trim")
                accepted = True  # LLM wrote the code; trust it on API failure
                break

            if verdict.get("verdict", "").upper() == "APPROVE":
                accepted = True
                break

            # REJECT — try the fix code if provided
            fix_code = verdict.get("code", "")
            reason = verdict.get("reason", "unknown")
            self.log(f"  {fig_key} LLM REJECTED (round {verify_round + 1}): {reason}")

            if not fix_code.strip() or verify_round >= 1:
                break  # no fix code or last round — give up

            # Execute fix code
            fix_output = png_path.parent / f"{png_path.stem}_fix.png"
            success = self._exec_trim_code(
                fix_code, str(trimmed_path), str(fix_output),
            )
            if success and fix_output.exists():
                trimmed_bytes = fix_output.read_bytes()
                shutil.copy2(str(fix_output), str(trimmed_path))
                fix_output.unlink(missing_ok=True)
            else:
                self.log(f"  {fig_key} fix code execution failed")
                break

        if accepted:
            shutil.copy2(str(trimmed_path), str(png_path))
            with Image.open(png_path) as _img:
                tw, th = _img.size
            self.log(f"  {fig_key} trim ACCEPTED: {w}x{h} -> {tw}x{th}")
        else:
            self.log(f"  {fig_key} keeping original (LLM did not approve trim)")

        trimmed_path.unlink(missing_ok=True)

    async def _llm_analyze_trim(
        self, fig_key: str, image_bytes: bytes, width: int, height: int,
    ) -> dict:
        """Send image to LLM; get back trim decision + code.

        Uses figure_code stage (vision-capable model like Claude Sonnet),
        NOT figure_gen (image generation model like Gemini).
        """
        # Use vision-capable model, not the image-generation model
        vision_config = self.config.for_stage("figure_code")
        response = await self.generate_with_image(
            self._TRIM_ANALYZE_SYSTEM,
            f"Figure '{fig_key}', dimensions: {width}x{height} pixels.\n"
            f"Analyze this figure. Is there excess whitespace that should "
            f"be cropped? If yes, write Python code to crop it properly.\n"
            f"Remember: preserve ALL chart content (axes, labels, legends, data).",
            image_bytes,
            json_mode=True,
            stage_override=vision_config,
        )
        return self._safe_parse_json(response, {"needs_trim": False})

    async def _llm_verify_trim(
        self, fig_key: str, image_bytes: bytes,
    ) -> dict:
        """Send trimmed image to LLM for visual verification.

        Uses figure_code stage (vision-capable model like Claude Sonnet),
        NOT figure_gen (image generation model like Gemini).
        """
        # Use vision-capable model, not the image-generation model
        vision_config = self.config.for_stage("figure_code")
        response = await self.generate_with_image(
            self._TRIM_VERIFY_SYSTEM,
            f"This is a cropped version of figure '{fig_key}'. "
            f"Is the crop correct? Is all content preserved?",
            image_bytes,
            json_mode=True,
            stage_override=vision_config,
        )
        return self._safe_parse_json(response, {"verdict": "APPROVE"})

    def _exec_trim_code(self, code: str, input_path: str, output_path: str) -> bool:
        """Execute LLM-written trim code in a subprocess.

        Pre-defines INPUT_PATH and OUTPUT_PATH variables for the code.
        Returns True if execution succeeded and output file exists.
        """
        import os
        import subprocess
        import sys
        import textwrap

        preamble = textwrap.dedent("""\
            import os, sys
            INPUT_PATH = %s
            OUTPUT_PATH = %s
            from PIL import Image, ImageChops
            import numpy as np
        """) % (repr(input_path), repr(output_path))
        wrapper = preamble + "\n" + code

        # BUG-33 fix: use venv Python (where PIL is installed) instead of
        # sys.executable (the orchestrator Python, which may lack PIL).
        python_exe = self._resolve_experiment_python()
        try:
            result = subprocess.run(
                [python_exe, "-c", wrapper],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.warning("Trim code failed: %s", result.stderr[:500])
                return False
            return os.path.exists(output_path)
        except subprocess.TimeoutExpired:
            logger.warning("Trim code timed out (30s)")
            return False
        except Exception as e:
            logger.warning("Trim code execution error: %s", e)
            return False

