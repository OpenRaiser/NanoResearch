"""Writing agent — assembles the final paper draft using LaTeX templates.

Generates each section independently via separate LLM calls to avoid
truncated JSON and escape issues with large monolithic outputs.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.latex import fixer as latex_fixer
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.schemas.paper import PaperSkeleton, Section

from nanoresearch.skill_prompts import (
    get_writing_system_prompt,
    ABSTRACT_SYSTEM,
    TITLE_SYSTEM,
)

from ._types import (  # noqa: F401 — re-exported for backward compat
    GroundingPacket,
    ContributionClaim,
    ContributionContract,
    ResultCompleteness,
)

logger = logging.getLogger(__name__)


# Configurable limits
MAX_PAPERS_FOR_CITATIONS = 50
MAX_LATEX_FIX_ATTEMPTS = 3

# Legacy aliases — now each section gets its own system prompt via
# get_writing_system_prompt(heading), but some internal methods still
# need a generic prompt for non-section tasks (e.g., LaTeX fix).
SECTION_SYSTEM_PROMPT = get_writing_system_prompt("_default")
ABSTRACT_SYSTEM_PROMPT = ABSTRACT_SYSTEM
TITLE_SYSTEM_PROMPT = TITLE_SYSTEM

# Section specs: (heading, label, writing_instructions, related_figures)
# related_figures: list of figure keys to embed WITHIN this section
PAPER_SECTIONS = [
    ("Introduction", "sec:intro",
     "Write 4-5 paragraphs following the classic three-move structure:\n"
     "MOVE 1 --- Establish importance (1-2 paragraphs):\n"
     "  Open with a concrete, compelling motivation for the research problem. "
     "Cite key works that establish the problem's importance. "
     "Spend more space on what is novel, not on well-known background.\n"
     "MOVE 2 --- Identify the gap (1 paragraph):\n"
     "  Describe what current methods do and their SPECIFIC, QUANTITATIVE limitations "
     "(e.g. 'achieves only 85\\% on X', 'scales as O(n^2)'). "
     "Cite 3-5 representative works using \\citet{} and \\citep{}.\n"
     "MOVE 3 --- State contribution (1-2 paragraphs):\n"
     "  State the key insight that motivates your approach. "
     "Describe your proposed method at a high level --- name it, list core components, "
     "explain why this design addresses the identified limitations.\n"
     "  End with a contributions list using \\begin{itemize} with EXACTLY 2 or 3 \\item entries:\n"
     "  - 'We propose [METHOD], a ... that ...'\n"
     "  - 'We introduce [COMPONENT], which ...'\n"
     "  - 'Experiments on [DATASETS] demonstrate that [METHOD] achieves ...'\n"
     "  Do NOT exceed 3 contribution bullets --- merge related points if needed.\n"
     "Contributions should appear as early as possible. Use assertive language throughout.",
     []),

    ("Related Work", "sec:related",
     "Write 4-5 paragraphs organized by 3-4 THEMATIC clusters (not chronologically).\n"
     "For each cluster: summarize 3-5 papers, show evolution of ideas, then state "
     "how the proposed method differs or improves.\n"
     "Respect prior work before noting limitations --- do not dismiss previous contributions.\n"
     "End with a paragraph that explicitly positions this work: "
     "'Unlike [prior work] which ..., our method ...'.\n"
     "Cite quantitative results from prior work where available from the evidence data.\n"
     "IMPORTANT: Every \\cite{} or \\citet{} key MUST be from the provided CITATION KEYS list. "
     "Do NOT invent citation keys. If uncertain about a key, omit the citation.\n"
     "MUST-CITE ENFORCEMENT: If the context includes a 'MUST-CITE PAPERS' section, "
     "you MUST cite ALL of those papers in this section. Organize them into your thematic "
     "clusters naturally. Do not skip any must-cite paper.\n"
     "FAIRNESS: discuss the STRONGEST baselines, not just weak ones. "
     "Acknowledge prior contributions before noting limitations.\n"
     "Use \\citet{key} when author is subject, \\citep{key} for parenthetical.",
     []),

    ("Method", "sec:method",
     "Write 5-7 paragraphs with full technical detail of the proposed method.\n"
     "Structure:\n"
     "(1) Overview paragraph: state the problem formulation with formal notation "
     "(input space \\mathcal{X}, output space \\mathcal{Y}). "
     "Give a high-level description. Reference Figure~\\ref{fig:architecture} if available.\n"
     "(2-4) One \\subsection{} per major component/submodule. For each:\n"
     "  - State its purpose and how it connects to other components\n"
     "  - Provide mathematical formulation with numbered equations (\\begin{equation})\n"
     "  - Reference every equation in text: 'as defined in Eq.~\\eqref{eq:loss}'\n"
     "  - Explain design choices and why alternatives were rejected\n"
     "  - Use consistent notation: \\mathbf{x} for vectors, \\mathbf{W} for matrices, "
     "\\mathcal{L} for loss functions\n"
     "(5) Training/optimization: loss function, optimizer, learning rate schedule.\n"
     "(6) Complexity analysis: report time and space complexity (Big-O). "
     "Compare against baseline complexity. State FLOPs or inference time if available.\n"
     "Use \\begin{align} for multi-line equations (NEVER eqnarray).\n"
     "Do NOT include \\begin{figure} blocks yourself --- figures are inserted automatically "
     "near their \\ref{fig:...} references.",
     ["architecture"]),

    ("Experiments", "sec:experiments",
     "Write 6-8 paragraphs covering:\n"
     "(1) Experimental Setup: datasets (with statistics: \\# samples, splits, domain), "
     "evaluation metrics (define each), baselines (cite source of each).\n"
     "(2) Implementation Details: hyperparameters (final values AND search ranges), "
     "hardware (GPU type, count), training time, random seeds, software versions.\n"
     "(3) Main Results: you MUST include a LaTeX table (Table~\\ref{tab:main_results}) using "
     "\\begin{table}[t!] with booktabs comparing all methods across all metrics.\n"
     "  - Bold the best result in each column using \\textbf{}\n"
     "  - Include standard deviations or confidence intervals (e.g., $\\pm$ 0.3)\n"
     "  - Use \\citet{key} to reference baseline sources\n"
     "  - Avoid '--' in tables; fill ALL cells with concrete numbers\n"
     "(4) Analysis: explain WHY the method works --- what specific component leads to gains. "
     "Support with evidence, not speculation.\n"
     "(5) Ablation Study: you MUST include a LaTeX ablation table (Table~\\ref{tab:ablation}) "
     "using \\begin{table}[t!] with booktabs. Each row removes or replaces one component "
     "(e.g., 'w/o module A', 'replace B with C'). Columns are evaluation metrics. "
     "The full model should be the last row with best results in \\textbf{}. "
     "Discuss what each ablation reveals about the component's contribution.\n"
     "(6) Additional analysis as appropriate: efficiency comparison (FLOPs, inference time), "
     "qualitative examples, case studies, error analysis.\n"
     "Escape percent signs as \\%. Use -- for en-dashes in number ranges.\n\n"
     "TABLE FORMATTING RULES (apply to ALL tables):\n"
     "  - Maximum 6 metric columns per table. If more metrics exist, select the 6 most\n"
     "    important ones and mention others in text.\n"
     "  - Use SHORT column headers: 1-3 words or standard abbreviations (e.g., 'Acc',\n"
     "    'F1', 'BLEU', 'mAP'). Never use full sentences as column headers.\n"
     "  - For tables with 5+ data columns, wrap the tabular in\n"
     "    \\resizebox{\\textwidth}{!}{...} to prevent overflow beyond page margins.\n"
     "  - Escape ALL percent signs as \\% in table cells and headers.\n\n"
     "CRITICAL — RESULTS IN TABLES:\n"
     "If the context contains REAL EXPERIMENT RESULTS (marked as such above), you MUST use\n"
     "those exact numbers in Table~\\ref{tab:main_results} and Table~\\ref{tab:ablation}.\n"
     "Do NOT round, adjust, or modify them.\n"
     "If no results are available for the PROPOSED METHOD because the experiment FAILED,\n"
     "use '--' in the proposed method's table cells. For BASELINE methods, fill in\n"
     "published numbers from their original papers (cite the source).\n"
     "Add a note: 'Results for our method are pending due to execution issues.'\n"
     "Do NOT skip or omit the tables — always include Table 1 and Table 2.\n\n"
     "Do NOT include \\begin{figure} blocks yourself --- figures are inserted automatically "
     "near their \\ref{fig:...} references.",
     []),

    ("Conclusion", "sec:conclusion",
     "Write 2-3 paragraphs:\n"
     "(1) Summarize the method name, core idea, and key quantitative results in 2-3 sentences.\n"
     "(2) Discuss limitations honestly --- what scenarios, data types, or scale might be "
     "challenging. Honest acknowledgment is valued, not penalized.\n"
     "(3) Future work: 2-3 concrete, specific research directions (not vague).\n"
     "Do NOT introduce new results or citations here.",
     []),
]


_LATEX_TEXT_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "^": r"\^{}",
    "~": r"\~{}",
}


_IDENTIFIER_COMMANDS = frozenset({
    "ref", "eqref", "autoref", "nameref", "pageref",
    "label",
    "cite", "citet", "citep", "citealp", "citeauthor", "citeyear",
    "bibliography", "bibliographystyle",
    "input", "include", "includegraphics",
    "url",
})


def _escape_latex_text(text: str) -> str:
    """Escape LaTeX special characters in plain text (captions, method names, etc.).

    Preserves existing LaTeX commands and already-escaped sequences while still
    escaping bare special characters in surrounding prose.
    Reference-type commands (\\ref, \\label, \\cite, etc.) have their braced
    arguments preserved verbatim since they contain identifiers, not prose.
    """
    if not isinstance(text, str):
        text = str(text)

    result: list[str] = []
    i = 0
    in_math = False
    preservable_after_backslash = set(r"\$%#&_{}~^()[]")

    while i < len(text):
        ch = text[i]

        if ch == "\\":
            if i + 1 >= len(text):
                result.append(r"\textbackslash{}")
                break

            next_char = text[i + 1]
            if next_char.isalpha():
                j = i + 2
                while j < len(text) and text[j].isalpha():
                    j += 1
                cmd_name = text[i + 1:j]
                result.append(text[i:j])
                i = j

                # For identifier commands, preserve {...} arguments verbatim
                if cmd_name in _IDENTIFIER_COMMANDS:
                    # Skip optional [...]
                    while i < len(text) and text[i] == '[':
                        close_bracket = text.find(']', i)
                        if close_bracket == -1:
                            break
                        result.append(text[i:close_bracket + 1])
                        i = close_bracket + 1
                    # Preserve {...} argument
                    if i < len(text) and text[i] == '{':
                        depth = 0
                        k = i
                        while k < len(text):
                            if text[k] == '{':
                                depth += 1
                            elif text[k] == '}':
                                depth -= 1
                                if depth == 0:
                                    result.append(text[i:k + 1])
                                    i = k + 1
                                    break
                            k += 1
                        else:
                            result.append(text[i:])
                            i = len(text)
                continue

            if next_char in preservable_after_backslash:
                result.append(text[i:i + 2])
                i += 2
                # BUG-39 fix: track \(...\) inline math mode
                if next_char == '(' or next_char == '[':
                    in_math = True
                elif next_char == ')' or next_char == ']':
                    in_math = False
                continue

            result.append(r"\textbackslash{}")
            i += 1
            continue

        if ch == "$":
            if in_math:
                result.append(ch)
                in_math = False
            else:
                has_closing_dollar = False
                j = i + 1
                escaped = False
                while j < len(text):
                    lookahead = text[j]
                    if escaped:
                        escaped = False
                    elif lookahead == "\\":
                        escaped = True
                    elif lookahead == "$":
                        has_closing_dollar = True
                        break
                    j += 1
                if has_closing_dollar:
                    result.append(ch)
                    in_math = True
                else:
                    result.append(r"\$")
            i += 1
            continue

        if in_math:
            result.append(ch)
            i += 1
            continue

        result.append(_LATEX_TEXT_ESCAPES.get(ch, ch))
        i += 1

    return "".join(result)



def _check_global_consistency(
    latex_content: str,
    abstract: str,
    sections: list[Section],
) -> list[str]:
    """Post-generation consistency check across all sections."""
    issues: list[str] = []

    refs = set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', latex_content))
    labels = set(re.findall(r'\\label\{([^}]+)\}', latex_content))
    for ref in sorted(refs - labels):
        issues.append(f"\\ref{{{ref}}} has no matching \\label (will show '??' in PDF)")

    all_labels = re.findall(r'\\label\{([^}]+)\}', latex_content)
    seen: set[str] = set()
    for lbl in all_labels:
        if lbl in seen:
            issues.append(f"Duplicate \\label{{{lbl}}} -- LaTeX will error or mis-link")
        seen.add(lbl)

    if abstract:
        abstract_pcts = set(re.findall(r'(\d+\.?\d*)\s*\\?%', abstract))
        body_text = "\n".join(sec.content for sec in sections)
        for num in sorted(abstract_pcts):
            if num not in body_text:
                issues.append(
                    f"Abstract claims {num}\\% but this number does not appear "
                    f"in any body section -- possible fabrication"
                )

    for sec in sections:
        if sec.label == "sec:intro":
            itemize_blocks = re.findall(
                r'\\begin\{itemize\}(.*?)\\end\{itemize\}',
                sec.content, re.DOTALL,
            )
            for block in itemize_blocks:
                n_items = len(re.findall(r'\\item', block))
                if n_items > 5:
                    issues.append(
                        f"Introduction has {n_items} \\item entries -- "
                        f"consider merging to 2-4 contributions"
                    )
            break

    for env in ("figure", "figure*", "table", "table*"):
        escaped = re.escape(env)
        blocks = re.findall(
            rf'\\begin\{{{escaped}\}}(.*?)\\end\{{{escaped}\}}',
            latex_content, re.DOTALL,
        )
        for block in blocks:
            if r'\label{' not in block:
                cap = re.search(r'\\caption\{([^}]{0,60})', block)
                hint = cap.group(1) if cap else "(no caption)"
                issues.append(
                    f"A {env} environment has no \\label -- cannot be cross-referenced: "
                    f"{hint}..."
                )

    return issues


from .context_builder import _ContextBuilderMixin
from .grounding import _GroundingMixin
from .section_writer import _SectionWriterMixin
from .citation_manager import _CitationManagerMixin
from .latex_assembler import _LaTeXAssemblerMixin
from .writing_agent import _WritingAgentMixin

__all__ = ["WritingAgent", "GroundingPacket", "ContributionClaim", "ContributionContract"]


class WritingAgent(
    _WritingAgentMixin,
    _ContextBuilderMixin,
    _GroundingMixin,
    _SectionWriterMixin,
    _CitationManagerMixin,
    _LaTeXAssemblerMixin,
    BaseResearchAgent,
):
    """Generates a full LaTeX research paper from experiment results."""

    stage = PipelineStage.WRITING

    # ---- cite key management ------------------------------------------------

    # Surname prefixes that should be merged (e.g., "van der Waals" -> "vanderwaals")
    _NAME_PREFIXES = frozenset({
        "van", "von", "de", "del", "della", "di", "du", "el", "le", "la",
        "bin", "ibn", "al", "das", "dos", "den", "der", "het", "ten",
    })
