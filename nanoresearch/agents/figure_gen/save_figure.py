"""Figure file saving and utility methods."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _SaveFigureMixin:
    """Mixin — figure file saving and JSON parsing."""

    async def _save_figure_files(
        self,
        fig_key: str,
        filename_stem: str,
        caption: str,
        image_bytes: bytes,
        already_saved: bool = False,
        code_generated: bool = False,
    ) -> dict[str, Any]:
        """Save PNG (if not already saved) + convert to PDF + register artifacts.

        Args:
            code_generated: True for matplotlib/code-generated charts.
                Only code-generated figures go through LLM-driven trim,
                because API-generated figures (DALL-E, Gemini) are already
                properly sized by the image model.
        """
        png_path = self.workspace.path / "figures" / f"{filename_stem}.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)

        if not already_saved:
            png_path.write_bytes(image_bytes)

        # Deterministic whitespace trim — always runs for code-generated charts.
        # This is the primary defense; LLM trim below is a secondary refinement.
        if code_generated:
            try:
                self._deterministic_trim(fig_key, png_path)
            except Exception as e:
                self.log(f"  {fig_key} deterministic trim failed (non-fatal): {e}")
            # LLM-driven trim: secondary refinement pass
            try:
                await self._smart_trim_figure(fig_key, png_path)
            except Exception as e:
                self.log(f"  {fig_key} smart-trim failed (non-fatal): {e}")

        # Convert to PDF via Pillow
        pdf_path = self.workspace.path / "figures" / f"{filename_stem}.pdf"
        try:
            from PIL import Image
            img = Image.open(png_path)
            try:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(str(pdf_path), "PDF", resolution=300.0)
            finally:
                img.close()
            self.log(f"  {fig_key} saved: PNG + PDF")
        except Exception as e:
            self.log(f"  {fig_key} PDF conversion failed: {e}")
            pdf_path = None

        # Register artifacts
        self.workspace.register_artifact(f"{fig_key}_png", png_path, self.stage)
        if pdf_path is not None and pdf_path.exists():
            self.workspace.register_artifact(f"{fig_key}_pdf", pdf_path, self.stage)

        return {
            "png_path": str(png_path),
            "pdf_path": str(pdf_path) if pdf_path else None,
            "caption": caption,
        }

    # ------------------------------------------------------------------
    # LLM-driven figure trim: LLM sees image → writes code → executes
    # → LLM verifies result → approve / iterate (max 2 rounds)
    # ------------------------------------------------------------------

    _TRIM_ANALYZE_SYSTEM = (
        "You are a figure layout expert for academic papers. "
        "You will see a scientific figure image. Analyze it and decide "
        "whether it needs cropping to remove excess whitespace.\n\n"
        "RULES:\n"
        "- Academic figures MUST be compact — crop AGGRESSIVELY\n"
        "- Remove ALL blank/whitespace regions beyond a small margin\n"
        "- Orphaned text fragments (stray 'N/A', watermarks) floating in "
        "whitespace far from charts are NOT meaningful — crop them away\n"
        "- Keep ~20-30px margin around actual chart content\n\n"
        "OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:\n"
        '{"needs_trim": false}\n'
        "OR\n"
        '{"needs_trim": true, "code": "<python code>"}\n\n'
        "If needs_trim is true, write Python code using this PROVEN algorithm:\n"
        "```\n"
        "from PIL import Image\n"
        "import numpy as np\n"
        "img = Image.open(INPUT_PATH)\n"
        "arr = np.array(img)\n"
        "# Detect non-white pixels (threshold 245 catches light gray too)\n"
        "if arr.ndim == 3:\n"
        "    gray = np.mean(arr[:,:,:3], axis=2)\n"
        "else:\n"
        "    gray = arr.astype(float)\n"
        "non_white = gray < 245\n"
        "rows_mask = np.any(non_white, axis=1)\n"
        "cols_mask = np.any(non_white, axis=0)\n"
        "row_indices = np.where(rows_mask)[0]\n"
        "col_indices = np.where(cols_mask)[0]\n"
        "margin = 25\n"
        "top = max(0, row_indices[0] - margin)\n"
        "bottom = min(arr.shape[0], row_indices[-1] + margin)\n"
        "left = max(0, col_indices[0] - margin)\n"
        "right = min(arr.shape[1], col_indices[-1] + margin)\n"
        "cropped = img.crop((left, top, right, bottom))\n"
        "cropped.save(OUTPUT_PATH)\n"
        "print(f'Cropped: {img.size} -> {cropped.size}')\n"
        "```\n"
        "You may adapt this algorithm (e.g., adjust margin, threshold) but "
        "the core approach of detecting content via non-white pixel boundaries "
        "is REQUIRED. Do NOT use hardcoded pixel coordinates.\n"
        'Variables INPUT_PATH and OUTPUT_PATH are pre-defined strings.'
    )

    _TRIM_VERIFY_SYSTEM = (
        "You are a figure quality inspector for academic papers. "
        "You will see a cropped scientific figure. "
        "Check if the cropping is correct.\n\n"
        "APPROVE if:\n"
        "- All MAIN chart/graph content is fully visible: axes, tick marks, "
        "axis labels, legends, titles, data (bars/lines/points), and "
        "data annotations (value labels above bars, arrows, etc.)\n"
        "- Margins are compact (small gap around the content is fine)\n"
        "- The figure looks clean and publication-ready\n\n"
        "REJECT ONLY if:\n"
        "- A chart axis, axis label, or tick mark is visibly clipped\n"
        "- A legend entry is cut off or missing\n"
        "- Data (bars, lines, points) is partially clipped\n"
        "- A subplot panel is missing or cut off\n\n"
        "DO NOT reject for:\n"
        "- Removal of blank whitespace (that is the GOAL)\n"
        "- Removal of orphaned text fragments (stray 'N/A' etc.) that were "
        "floating in whitespace far from the chart\n"
        "- Tight margins — compact is good for papers\n\n"
        "OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:\n"
        '{"verdict": "APPROVE"}\n'
        "OR\n"
        '{"verdict": "REJECT", "reason": "...", "code": "<fix code>"}\n\n'
        "If REJECT, provide Python code that fixes the crop (same format: "
        "reads INPUT_PATH, saves to OUTPUT_PATH, uses PIL/numpy)."
    )


    @staticmethod
    def _safe_parse_json(text: str, default: dict) -> dict:
        """Parse JSON from LLM response, stripping markdown fences."""
        import json as _json
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return _json.loads(text[start:end])
                except _json.JSONDecodeError:
                    pass
            return default
