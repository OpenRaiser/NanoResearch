"""Figure whitespace trimming — deterministic + optional LLM vision."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ── LLM prompts ──────────────────────────────────────────────────────────────

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
    "```\n"
    "You may adapt this algorithm but the core non-white pixel boundary "
    "approach is REQUIRED. Do NOT use hardcoded pixel coordinates.\n"
    "Variables INPUT_PATH and OUTPUT_PATH are pre-defined strings."
)

_TRIM_VERIFY_SYSTEM = (
    "You are a figure quality inspector for academic papers. "
    "You will see a cropped scientific figure. "
    "Check if the cropping is correct.\n\n"
    "APPROVE if:\n"
    "- All MAIN chart/graph content is fully visible: axes, tick marks, "
    "axis labels, legends, titles, data, and annotations\n"
    "- Margins are compact\n"
    "- The figure looks clean and publication-ready\n\n"
    "REJECT ONLY if:\n"
    "- A chart axis, label, or tick mark is visibly clipped\n"
    "- A legend entry is cut off\n"
    "- Data is partially clipped\n"
    "- A subplot panel is missing or cut off\n\n"
    "DO NOT reject for:\n"
    "- Removal of blank whitespace (that is the GOAL)\n"
    "- Tight margins — compact is good for papers\n\n"
    "OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences:\n"
    '{"verdict": "APPROVE"}\n'
    "OR\n"
    '{"verdict": "REJECT", "reason": "...", "code": "<fix code>"}\n\n'
    "If REJECT, provide Python code that fixes the crop."
)


# ── Deterministic auto-trim ──────────────────────────────────────────────────

def auto_trim(image_path: Path) -> bool:
    """PIL + numpy whitespace detection and cropping.

    Supports .png, .jpg, .jpeg files.
    Returns True if the image was cropped.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        logger.warning("PIL/numpy not available, skipping auto_trim")
        return False

    try:
        img = Image.open(image_path)
    except Exception as exc:
        logger.warning("Cannot open %s: %s", image_path, exc)
        return False

    # Convert to RGB if needed (e.g., RGBA PNGs, palette images)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    arr = np.array(img)
    if arr.ndim == 3:
        gray = np.mean(arr[:, :, :3], axis=2)
    else:
        gray = arr.astype(float)

    non_white = gray < 245
    rows = np.where(np.any(non_white, axis=1))[0]
    cols = np.where(np.any(non_white, axis=0))[0]

    if len(rows) == 0:
        return False  # All white — skip

    margin = 25
    crop_box = (
        max(0, int(cols[0]) - margin),
        max(0, int(rows[0]) - margin),
        min(arr.shape[1], int(cols[-1]) + margin),
        min(arr.shape[0], int(rows[-1]) + margin),
    )

    original_area = arr.shape[0] * arr.shape[1]
    cropped_area = (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])

    # Only crop if removing >10% of area
    if cropped_area >= 0.90 * original_area:
        return False

    cropped = img.crop(crop_box)
    # Preserve quality for JPEG
    save_kwargs = {}
    if image_path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 95
    cropped.save(image_path, **save_kwargs)
    logger.info(
        "Auto-trimmed %s: %dx%d -> %dx%d (%.0f%% area removed)",
        image_path.name, img.size[0], img.size[1],
        cropped.size[0], cropped.size[1],
        (1 - cropped_area / original_area) * 100,
    )
    return True


# ── LLM-driven trim ─────────────────────────────────────────────────────────

def llm_trim(image_path: Path, llm_client) -> bool:
    """LLM-driven figure trimming with verification loop.

    Returns True if the image was trimmed.
    """
    original_bytes = image_path.read_bytes()
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(original_bytes))
        w, h = img.size
    except Exception:
        return False

    logger.info("LLM trim check: %s (%dx%d)", image_path.name, w, h)

    # Round 1: analyze
    try:
        response = llm_client.generate_with_image(
            _TRIM_ANALYZE_SYSTEM,
            f"Figure '{image_path.stem}', dimensions: {w}x{h} pixels.\n"
            f"Analyze this figure. Is there excess whitespace?",
            original_bytes,
            json_mode=True,
        )
        trim_plan = llm_client.safe_parse_json(response, {"needs_trim": False})
    except Exception as e:
        logger.warning("LLM trim analysis failed: %s", e)
        return False

    if not trim_plan.get("needs_trim"):
        logger.info("LLM says no trim needed for %s", image_path.name)
        return False

    code = trim_plan.get("code", "")
    if not code.strip():
        return False

    # Execute cropping code
    trimmed_path = image_path.parent / f"{image_path.stem}_trimmed.png"
    success = _exec_trim_code(code, str(image_path), str(trimmed_path))

    if not success or not trimmed_path.exists():
        logger.warning("Trim code execution failed for %s", image_path.name)
        return False

    # Round 2: verify (max 2 rounds)
    trimmed_bytes = trimmed_path.read_bytes()
    accepted = False

    for verify_round in range(2):
        try:
            response = llm_client.generate_with_image(
                _TRIM_VERIFY_SYSTEM,
                f"This is a cropped version of '{image_path.stem}'. "
                f"Is the crop correct?",
                trimmed_bytes,
                json_mode=True,
            )
            verdict = llm_client.safe_parse_json(response, {"verdict": "APPROVE"})
        except Exception:
            accepted = True  # Trust LLM's code on API failure
            break

        if verdict.get("verdict", "").upper() == "APPROVE":
            accepted = True
            break

        fix_code = verdict.get("code", "")
        reason = verdict.get("reason", "unknown")
        logger.info("LLM REJECTED %s (round %d): %s", image_path.name, verify_round + 1, reason)

        if not fix_code.strip() or verify_round >= 1:
            break

        fix_output = image_path.parent / f"{image_path.stem}_fix.png"
        success = _exec_trim_code(fix_code, str(trimmed_path), str(fix_output))
        if success and fix_output.exists():
            trimmed_bytes = fix_output.read_bytes()
            shutil.copy2(str(fix_output), str(trimmed_path))
            fix_output.unlink(missing_ok=True)
        else:
            break

    if accepted:
        shutil.copy2(str(trimmed_path), str(image_path))
        logger.info("LLM trim ACCEPTED for %s", image_path.name)
        trimmed_path.unlink(missing_ok=True)
        return True
    else:
        logger.info("Keeping original %s (LLM did not approve)", image_path.name)
        trimmed_path.unlink(missing_ok=True)
        return False


def _exec_trim_code(code: str, input_path: str, output_path: str) -> bool:
    """Execute LLM-written trim code in a subprocess."""
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

    try:
        result = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("Trim code stderr: %s", result.stderr[:500])
            return False
        return True
    except Exception as exc:
        logger.warning("Trim code execution error: %s", exc)
        return False


# ── PDF figure trimming via PyMuPDF ──────────────────────────────────────────

def auto_trim_pdf(pdf_path: Path) -> bool:
    """Trim whitespace from a single-page PDF figure using PyMuPDF.

    Renders the PDF to a raster image, detects non-white boundaries,
    and applies a cropbox to the PDF page.
    Returns True if the PDF was cropped.
    """
    try:
        import fitz
        import numpy as np
    except ImportError:
        logger.warning("PyMuPDF/numpy not available, skipping PDF trim")
        return False

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning("Cannot open PDF %s: %s", pdf_path, exc)
        return False

    if len(doc) == 0:
        doc.close()
        return False

    page = doc[0]
    # Render at 150 DPI for whitespace detection
    pix = page.get_pixmap(dpi=150)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    # Grayscale conversion
    if arr.shape[2] >= 3:
        gray = np.mean(arr[:, :, :3], axis=2)
    else:
        gray = arr[:, :, 0].astype(float)

    non_white = gray < 245
    rows = np.where(np.any(non_white, axis=1))[0]
    cols = np.where(np.any(non_white, axis=0))[0]

    if len(rows) == 0:
        doc.close()
        return False

    margin = 25  # pixels at 150 DPI
    top = max(0, int(rows[0]) - margin)
    bottom = min(pix.h, int(rows[-1]) + margin)
    left = max(0, int(cols[0]) - margin)
    right = min(pix.w, int(cols[-1]) + margin)

    # Check if removing >10% of area
    original_area = pix.h * pix.w
    cropped_area = (right - left) * (bottom - top)
    if cropped_area >= 0.90 * original_area:
        doc.close()
        return False

    # Convert pixel coords back to PDF points
    page_rect = page.rect
    scale_x = page_rect.width / pix.w
    scale_y = page_rect.height / pix.h

    crop_rect = fitz.Rect(
        page_rect.x0 + left * scale_x,
        page_rect.y0 + top * scale_y,
        page_rect.x0 + right * scale_x,
        page_rect.y0 + bottom * scale_y,
    )

    page.set_cropbox(crop_rect)
    doc.save(str(pdf_path), incremental=True, encryption=0)
    doc.close()

    logger.info(
        "Auto-trimmed PDF %s: %.0f%% area removed",
        pdf_path.name,
        (1 - cropped_area / original_area) * 100,
    )
    return True


# ── Master trim function ─────────────────────────────────────────────────────

def trim_all_figures(figures_dir: Path, llm_client=None) -> list[str]:
    """Trim whitespace from all figures in the directory.

    Supports .png, .jpg, .jpeg, and .pdf files.
    Returns list of trimmed file names.
    """
    trimmed: list[str] = []

    # Raster images
    image_files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend(figures_dir.glob(ext))

    for img_file in sorted(image_files):
        if auto_trim(img_file):
            trimmed.append(img_file.name)
            continue
        if llm_client is not None:
            if llm_trim(img_file, llm_client):
                trimmed.append(img_file.name)

    # PDF figures
    pdf_files = list(figures_dir.glob("*.pdf"))
    for pdf_file in sorted(pdf_files):
        # Skip paper.pdf itself
        if pdf_file.stem.lower() == "paper":
            continue
        if auto_trim_pdf(pdf_file):
            trimmed.append(pdf_file.name)

    if trimmed:
        logger.info("Trimmed %d figures: %s", len(trimmed), ", ".join(trimmed))
    return trimmed
