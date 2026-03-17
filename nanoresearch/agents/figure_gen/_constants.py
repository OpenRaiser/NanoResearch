"""Figure generation agent — dynamic figure planning + hybrid AI/code charts.

Instead of hardcoding 3 identical-pattern figures, this agent:
  1. Asks the LLM to plan which figures to generate based on the research context
  2. Generates each figure using the appropriate method (AI image or LLM code)
  3. Supports diverse chart types: bar, line, heatmap, scatter, radar, box, etc.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.prompts import load_prompt
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Configurable limits
CHART_EXEC_TIMEOUT = 60  # seconds for subprocess chart execution


def _run_chart_subprocess(
    command: list[str],
    *,
    timeout: int = 60,
    cwd: str | None = None,
) -> dict[str, str | int]:
    """Run a chart-plotting subprocess with proper process-tree cleanup on timeout."""
    from nanoresearch.agents.execution.cluster_runner import _kill_process_tree

    proc = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc.pid)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
        raise  # re-raise so caller catches TimeoutExpired
    return {"returncode": proc.returncode or 0, "stdout": stdout, "stderr": stderr}
MAX_IMAGE_PROMPT_LEN = 3800
MAX_EVIDENCE_TRAINING_LOG_ENTRIES = 50  # cap training log in evidence block
MAX_EVIDENCE_BLOCK_LEN = 8000  # cap total evidence block length
MAX_IMAGE_RETRIES = 2  # retries before LLM diagnosis
MAX_OPTIMIZED_PROMPT_LEN = 1500  # shorter prompt for retry after diagnosis
MAX_CODE_CHART_RETRIES = 3  # retries for code chart generation (with error feedback)

# Maximum allowed figure dimensions (pixels at 300 DPI).
# A4 width = 8.27in → at 300 DPI = 2481px.  Max height = 1.5x width.
MAX_FIG_WIDTH_PX = 2600
MAX_FIG_HEIGHT_PX = 3000
MAX_FIG_ASPECT_RATIO = 1.8  # height / width — reject if taller than this

# Preamble injected before EVERY LLM-generated figure script.
# This ensures matplotlib is properly configured regardless of what the LLM writes.
_FIGURE_CODE_PREAMBLE = """\
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None

# === Enforced rcParams (injected by NanoResearch) ===
mpl.rcParams.update({
    'figure.autolayout': True,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.figsize': (7, 4.3),     # sane default: ~golden ratio
    'figure.max_open_warning': 5,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'pdf.fonttype': 42,
})
# === End enforced rcParams ===
"""

# ---------------------------------------------------------------------------
# Prompts (loaded from nanoresearch/prompts/figure_gen/*.yaml)
# ---------------------------------------------------------------------------

FIGURE_PLAN_SYSTEM = load_prompt("figure_gen", "planning")
FIGURE_PROMPT_SYSTEM = load_prompt("figure_gen", "prompt_engineering")
CHART_CODE_SYSTEM = load_prompt("figure_gen", "chart_code")
PROMPT_CORE_PRINCIPLES = load_prompt("figure_gen", "core_principles")

# Chart type specific prompts — dict[str, str]
CHART_TYPE_PROMPTS: dict[str, str] = {
    ct: load_prompt("figure_gen/chart_types", ct)
    for ct in [
        "grouped_bar", "line_plot", "heatmap", "radar", "scatter",
        "box_plot", "stacked_bar", "violin", "horizontal_bar",
        "scaling_law", "confusion_matrix", "embedding_scatter",
    ]
}

# AI figure templates — dict[str, str]
AI_FIGURE_TEMPLATES: dict[str, str] = {
    tmpl: load_prompt("figure_gen/ai_templates", tmpl)
    for tmpl in [
        "system_overview", "transformer_arch", "encoder_decoder",
        "multi_stage", "comparison_framework", "attention_map",
        "embedding_viz", "qualitative_comparison", "data_pipeline",
        "loss_landscape", "generic",
    ]
}

# ---------------------------------------------------------------------------
# Caption cleaning for AI-generated images
# ---------------------------------------------------------------------------

# Maximum length for a paper-ready caption.  Anything longer is likely the
# image-generation prompt leaking through.
MAX_ACADEMIC_CAPTION_LEN = 200


def _clean_ai_image_caption(caption: str, title: str = "") -> str:
    """Shorten an AI-image caption that is actually a generation prompt.

    If the caption is <= MAX_ACADEMIC_CAPTION_LEN characters it is returned
    as-is.  Otherwise we extract a short, academic-style caption by:
      1. Taking the first sentence (up to the first period followed by a space
         or end-of-string).
      2. If the first sentence is still too long, fall back to a generic
         caption built from the *title* field.
    """
    if not caption or len(caption) <= MAX_ACADEMIC_CAPTION_LEN:
        return caption

    # Try extracting the first sentence
    match = re.match(r"^(.+?\.)\s", caption)
    if match:
        first_sentence = match.group(1).strip()
        if len(first_sentence) <= MAX_ACADEMIC_CAPTION_LEN:
            return first_sentence

    # First sentence still too long or no period found — build from title
    if title:
        clean_title = title.strip().rstrip(".")
        return f"{clean_title} architecture and data flow."
    # Last resort: hard-truncate at the first comma/colon boundary
    for sep in (",", ":", ";"):
        idx = caption.find(sep)
        if 20 < idx <= MAX_ACADEMIC_CAPTION_LEN:
            return caption[:idx].rstrip() + "."
    return caption[:MAX_ACADEMIC_CAPTION_LEN].rsplit(" ", 1)[0].rstrip(".,;: ") + "."


# ---------------------------------------------------------------------------
# FigureAgent — dynamic figure planning + hybrid AI/code generation
# ---------------------------------------------------------------------------
