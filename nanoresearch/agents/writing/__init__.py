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
     "CRITICAL — RESULTS IN TABLES:\n"
     "If the context contains REAL EXPERIMENT RESULTS (marked as such above), you MUST use\n"
     "those exact numbers in Table~\\ref{tab:main_results} and Table~\\ref{tab:ablation}.\n"
     "Do NOT round, adjust, or modify them.\n"
     "If results are marked SYNTHETIC or come from figure data, use those numbers in the tables.\n"
     "NEVER leave the proposed method rows as '--'. Always fill tables with concrete numbers.\n"
     "If no results are available because the experiment FAILED or did not run,\n"
     "you MUST clearly state this in the text. Write: 'Due to [reason], we were unable to\n"
     "obtain experimental results. We present our methodology and leave empirical validation\n"
     "for future work.' Do NOT fabricate or invent any experimental numbers.\n\n"
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

                # For identifier commands, preserve {…} arguments verbatim
                # (underscores in \ref{fig:my_fig} are identifiers, not prose)
                if cmd_name in _IDENTIFIER_COMMANDS:
                    # Skip optional [...]
                    while i < len(text) and text[i] == '[':
                        close_bracket = text.find(']', i)
                        if close_bracket == -1:
                            break
                        result.append(text[i:close_bracket + 1])
                        i = close_bracket + 1
                    # Preserve {…} argument
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
                            # Unmatched brace — just append what we have
                            result.append(text[i:])
                            i = len(text)
                continue

            if next_char in preservable_after_backslash:
                result.append(text[i:i + 2])
                i += 2
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
    """Post-generation consistency check across all sections.

    Returns list of issue strings.  Non-blocking — issues are logged
    and passed to REVIEW for optional fixing.
    """
    issues: list[str] = []

    # ── 1. Broken cross-references (\ref without matching \label) ──
    refs = set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', latex_content))
    labels = set(re.findall(r'\\label\{([^}]+)\}', latex_content))
    for ref in sorted(refs - labels):
        issues.append(f"\\ref{{{ref}}} has no matching \\label (will show '??' in PDF)")

    # ── 2. Duplicate labels (causes LaTeX error) ──
    all_labels = re.findall(r'\\label\{([^}]+)\}', latex_content)
    seen: set[str] = set()
    for lbl in all_labels:
        if lbl in seen:
            issues.append(f"Duplicate \\label{{{lbl}}} — LaTeX will error or mis-link")
        seen.add(lbl)

    # ── 3. Abstract percentage numbers must appear somewhere in body ──
    if abstract:
        abstract_pcts = set(re.findall(r'(\d+\.?\d*)\s*\\?%', abstract))
        # Get body text (everything except abstract)
        body_text = "\n".join(sec.content for sec in sections)
        for num in sorted(abstract_pcts):
            if num not in body_text:
                issues.append(
                    f"Abstract claims {num}\\% but this number does not appear "
                    f"in any body section — possible fabrication"
                )

    # ── 4. Contribution bullet count sanity ──
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
                        f"Introduction has {n_items} \\item entries — "
                        f"consider merging to 2-4 contributions"
                    )
            break

    # ── 5. Floats (figure/table) without \label are unreferenceable ──
    for env in ("figure", "figure*", "table", "table*"):
        escaped = re.escape(env)
        blocks = re.findall(
            rf'\\begin\{{{escaped}\}}(.*?)\\end\{{{escaped}\}}',
            latex_content, re.DOTALL,
        )
        for block in blocks:
            if r'\label{' not in block:
                # Extract caption for identification
                cap = re.search(r'\\caption\{([^}]{0,60})', block)
                hint = cap.group(1) if cap else "(no caption)"
                issues.append(
                    f"A {env} environment has no \\label — cannot be cross-referenced: "
                    f"{hint}..."
                )

    return issues


from .context_builder import _ContextBuilderMixin
from .grounding import _GroundingMixin
from .section_writer import _SectionWriterMixin
from .citation_manager import _CitationManagerMixin
from .latex_assembler import _LaTeXAssemblerMixin

__all__ = ["WritingAgent", "GroundingPacket", "ContributionClaim", "ContributionContract"]


class WritingAgent(
    _ContextBuilderMixin,
    _GroundingMixin,
    _SectionWriterMixin,
    _CitationManagerMixin,
    _LaTeXAssemblerMixin,
    BaseResearchAgent,
):
    """Generates a full LaTeX research paper from experiment results."""

    stage = PipelineStage.WRITING

    async def run(self, **inputs: Any) -> dict[str, Any]:
        ideation: dict = inputs.get("ideation_output", {})
        blueprint: dict = inputs.get("experiment_blueprint", {})
        figure_output: dict = inputs.get("figure_output", {})
        template_format: str = inputs.get("template_format", self.config.template_format)
        experiment_results: dict = inputs.get("experiment_results", {})
        experiment_analysis: dict = inputs.get("experiment_analysis", {})
        experiment_summary: str = inputs.get("experiment_summary", "")
        experiment_status: str = inputs.get("experiment_status", "pending")
        authors: list[str] = inputs.get("authors", None) or ["NanoResearch"]

        self.log("Starting paper writing")

        # Step 0a: Build grounding packet — single source of truth for evidence
        grounding = self._build_grounding_packet(
            experiment_results,
            experiment_status,
            experiment_analysis,
            experiment_summary,
            blueprint,
        )
        self.log(
            f"Grounding: completeness={grounding.result_completeness}, "
            f"main_results={len(grounding.main_results)}, "
            f"ablations={len(grounding.ablation_results)}, "
            f"baselines={'yes' if grounding.comparison_with_baselines else 'no'}"
        )
        if grounding.evidence_gaps:
            self.log(f"Evidence gaps: {grounding.evidence_gaps}")

        # Step 0b: Build cite key mapping from papers
        papers = ideation.get("papers", [])
        cite_keys = self._build_cite_keys(papers)
        bibtex = self._build_bibtex(papers, cite_keys)

        # Build per-section context primitives (P0-A)
        core_ctx = self._build_core_context(ideation, blueprint, cite_keys)

        # Title & abstract need a broad context — use Introduction-like context
        title_abstract_ctx = self._ctx_introduction(
            core_ctx, grounding=grounding,
        )

        # Step 1: Generate title
        title = await self._generate_title(title_abstract_ctx)
        self.log(f"Title: {title}")

        # Step 2: Generate abstract (with result number binding)
        abstract = await self._generate_abstract(title_abstract_ctx, grounding)
        self.log("Abstract generated")

        # Step 3: Build figures & table data from blueprint
        figure_blocks = self._build_figure_blocks(blueprint, figure_output)

        # Step 4: Generate each section independently, embed figures inline
        # Track which figure blocks have been placed
        placed_figures: set[str] = set()

        # P0-B: Contribution contract — extracted after Introduction is written
        contribution_contract: ContributionContract | None = None
        method_name = (blueprint.get("proposed_method") or {}).get("name", "")

        sections = []
        prior_sections_summary: list[str] = []
        for heading, label, instructions, fig_keys in PAPER_SECTIONS:
            self.log(f"Writing section: {heading}")

            # P0-A: Build tailored context for this section
            section_ctx = self._build_section_context(
                label,
                core_ctx,
                grounding=grounding,
                experiment_results=experiment_results,
                experiment_status=experiment_status,
                experiment_analysis=experiment_analysis,
                experiment_summary=experiment_summary,
            )

            # P0-B: Inject contribution contract for post-Introduction sections
            if contribution_contract and label != "sec:intro":
                contract_block = contribution_contract.for_section(label)
                if contract_block:
                    section_ctx = section_ctx + "\n\n" + contract_block

            # Build per-section figure context: only show UN-placed figures
            remaining_figs = [k for k in figure_blocks if k not in placed_figures]
            fig_list_text = "\n".join(
                f"  - \\ref{{fig:{k}}}: {k}" for k in remaining_figs
            )
            placed_note = ""
            if placed_figures:
                placed_list = ", ".join(sorted(placed_figures))
                placed_note = (
                    f"\nFigures ALREADY placed in previous sections (do NOT include again): "
                    f"{placed_list}\n"
                )
            # Inject pre-built tables for Experiments section
            table_injection = ""
            if label == "sec:experiments" and grounding.has_real_results:
                table_parts = []
                if grounding.main_table_latex:
                    table_parts.append(
                        "=== PRE-BUILT MAIN RESULTS TABLE (use this EXACTLY, do NOT rebuild) ===\n"
                        + grounding.main_table_latex
                        + "\n=== END PRE-BUILT TABLE ==="
                    )
                if grounding.ablation_table_latex:
                    table_parts.append(
                        "=== PRE-BUILT ABLATION TABLE (use this EXACTLY, do NOT rebuild) ===\n"
                        + grounding.ablation_table_latex
                        + "\n=== END PRE-BUILT TABLE ==="
                    )
                if table_parts:
                    table_injection = "\n\n" + "\n\n".join(table_parts)

            # Conclusion-specific result-number binding
            conclusion_binding = ""
            if label == "sec:conclusion":
                if grounding.has_real_results and grounding.final_metrics:
                    metric_strs = [f"{k}={v}" for k, v in list(grounding.final_metrics.items())[:5]]
                    conclusion_binding = (
                        "\n\n=== CONCLUSION RESULT BINDING ===\n"
                        f"Real metrics to reference: {', '.join(metric_strs)}\n"
                        "Mention key results quantitatively when summarizing contributions. "
                        "Use the exact numbers above.\n"
                        "=== END BINDING ==="
                    )
                elif not grounding.has_real_results:
                    conclusion_binding = (
                        "\n\n=== CONCLUSION RESULT BINDING ===\n"
                        "No real experiment results. Do NOT cite specific performance numbers. "
                        "Focus on method design and future work.\n"
                        "=== END BINDING ==="
                    )

            context_with_figs = (
                f"{section_ctx}\n\n"
                f"=== AVAILABLE FIGURES (use \\ref{{fig:NAME}} to reference) ===\n"
                f"{fig_list_text}\n"
                f"{placed_note}"
                f"=== END FIGURES ==="
                f"{table_injection}"
                f"{conclusion_binding}"
            )

            content = await self._generate_section(
                context_with_figs, heading, instructions, prior_sections_summary
            )

            # ── Post-generation table verification for Experiments ──
            if label == "sec:experiments" and grounding.has_real_results:
                content = self._verify_and_inject_tables(
                    content, grounding, heading,
                )

            # ── Detect figures the LLM already embedded in section content ──
            # This prevents the code from placing the same figures again.
            # Match by \label{fig:XXX}
            llm_placed_labels = re.findall(
                r'\\begin\{figure\*?\}.*?\\label\{fig:([^}]+)\}.*?\\end\{figure\*?\}',
                content, re.DOTALL,
            )
            for fig_label in llm_placed_labels:
                if fig_label in figure_blocks and fig_label not in placed_figures:
                    placed_figures.add(fig_label)
                    self.log(f"  LLM already placed fig:{fig_label} in {heading}")
            # Also match by \includegraphics filename (e.g., fig3_main_results.pdf)
            llm_placed_files = re.findall(
                r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',
                content,
            )
            for fname in llm_placed_files:
                # Map filename back to figure key: "fig3_main_results.pdf" → "main_results"
                stem = fname.rsplit(".", 1)[0]  # remove extension
                for fk in figure_blocks:
                    if fk in placed_figures:
                        continue
                    # Check if the filename contains the figure key
                    if fk in stem or stem.endswith(fk):
                        placed_figures.add(fk)
                        self.log(f"  LLM already included {fname} → marking fig:{fk} as placed in {heading}")

            # ── Smart figure placement ──
            if label == "sec:intro":
                # Task illustration / qualitative / motivation figures → Intro
                intro_keywords = ("qualitative", "example", "motivation", "task",
                                  "illustration", "counterfactual", "demo", "sample",
                                  "teaser", "intuition")
                for fk in list(figure_blocks.keys()):
                    if fk in placed_figures:
                        continue
                    if any(kw in fk for kw in intro_keywords):
                        content += "\n\n" + figure_blocks[fk]
                        placed_figures.add(fk)
                        break  # only one intro figure

            if label == "sec:method":
                # Architecture / framework / pipeline figure → top of Method section
                arch_keywords = ("overview", "framework", "pipeline", "architecture", "model")

                # First: if LLM placed an arch figure but NOT at the top, move it
                for fk in list(figure_blocks.keys()):
                    if fk not in placed_figures:
                        continue
                    if not any(kw in fk for kw in arch_keywords):
                        continue
                    # This arch figure was already placed by LLM — check position
                    # Extract the figure block from content and move to top
                    fig_pattern = re.compile(
                        r'\n*\\begin\{figure\*?\}.*?\\label\{fig:'
                        + re.escape(fk)
                        + r'\}.*?\\end\{figure\*?\}\n*',
                        re.DOTALL,
                    )
                    match = fig_pattern.search(content)
                    if match and match.start() > 200:
                        # Figure is not near the top — move it
                        content = content[:match.start()] + content[match.end():]
                        content = figure_blocks[fk] + "\n\n" + content.lstrip("\n")
                        self.log(f"  Moved LLM-placed fig:{fk} to top of Method")
                    break  # only handle one arch figure

                # Then: place unplaced arch figure at top
                for fk in list(figure_blocks.keys()):
                    if fk in placed_figures:
                        continue
                    if any(kw in fk for kw in arch_keywords):
                        content = figure_blocks[fk] + "\n\n" + content
                        placed_figures.add(fk)
                        break  # only one architecture figure at top

            # 2) For ALL sections: insert remaining figures near their \ref
            # But respect reserved placement: arch figures belong in Method, not Intro
            _arch_kws = ("overview", "framework", "pipeline", "architecture", "model")
            _intro_kws = ("qualitative", "example", "motivation", "task",
                          "illustration", "counterfactual", "demo", "teaser")
            for fk, blk in figure_blocks.items():
                if fk in placed_figures:
                    continue
                # Skip architecture figures outside Method
                if label != "sec:method" and any(kw in fk for kw in _arch_kws):
                    continue
                # Skip intro/qualitative figures outside Intro
                if label != "sec:intro" and any(kw in fk for kw in _intro_kws):
                    continue
                content, inserted = self._insert_figure_near_ref(content, fk, blk)
                if inserted:
                    placed_figures.add(fk)

            sections.append(Section(heading=heading, label=label, content=content))
            snippet = content[:200].replace("\n", " ").strip()
            prior_sections_summary.append(f"[{heading}]: {snippet}...")

            # P0-B: Extract contribution contract after Introduction is written
            if label == "sec:intro" and not contribution_contract:
                contribution_contract = self._extract_contribution_contract(
                    content, method_name,
                )
                if contribution_contract.claims:
                    self.log(
                        f"Contribution contract: {len(contribution_contract.claims)} claims "
                        f"({', '.join(c.claim_type for c in contribution_contract.claims)})"
                    )
                else:
                    self.log("No contribution claims extracted from Introduction")

        # Fallback: distribute remaining figures to appropriate sections
        remaining = [k for k in figure_blocks if k not in placed_figures]
        if remaining:
            self.log(f"Fallback placement for {len(remaining)} unplaced figures: {remaining}")
            # Map figure keywords to preferred sections
            section_hints = {
                "sec:intro": ("qualitative", "example", "motivation", "task",
                              "illustration", "counterfactual", "demo", "teaser",
                              "intuition", "sample"),
                "sec:experiments": ("result", "comparison", "performance", "main", "latency",
                                    "tradeoff", "trade_off", "efficiency", "scalab"),
                "sec:method": ("architecture", "framework", "pipeline", "overview", "model",
                               "diagram", "workflow"),
                "sec:conclusion": ("ablation", "analysis", "error", "contradiction"),
            }
            for fk in remaining:
                target_label = "sec:experiments"  # default
                for sec_label, keywords in section_hints.items():
                    if any(kw in fk for kw in keywords):
                        target_label = sec_label
                        break
                for sec in sections:
                    if sec.label == target_label:
                        sec.content += "\n\n" + figure_blocks[fk]
                        placed_figures.add(fk)
                        self.log(f"  Placed '{fk}' → {target_label}")
                        break
                else:
                    # target section not found, append to Experiments
                    for sec in sections:
                        if sec.label == "sec:experiments":
                            sec.content += "\n\n" + figure_blocks[fk]
                            placed_figures.add(fk)
                            self.log(f"  Placed '{fk}' → sec:experiments (fallback)")
                            break

        # Post-assembly validation: ensure ALL figure blocks are in the sections
        final_missing = [k for k in figure_blocks if k not in placed_figures]
        if final_missing:
            self.log(f"CRITICAL: {len(final_missing)} figures still unplaced after all passes: {final_missing}")
            # Force-inject into Experiments as last resort
            for sec in sections:
                if sec.label == "sec:experiments":
                    for fk in final_missing:
                        sec.content += "\n\n" + figure_blocks[fk]
                        self.log(f"  Force-injected '{fk}' → sec:experiments")
                    break

        self.log(f"Figure placement complete: {len(figure_blocks)} blocks, "
                 f"{len(placed_figures)} placed")

        # ── Post-assembly deduplication: remove duplicate figure blocks ──
        # If the same figure appears in multiple sections, keep only the first occurrence.
        seen_fig_labels: set[str] = set()
        seen_fig_files: set[str] = set()
        for sec in sections:
            def _dedup_figure(m: re.Match) -> str:
                block = m.group(0)
                # Check by label
                label_m = re.search(r'\\label\{(fig:[^}]+)\}', block)
                lbl = label_m.group(1) if label_m else None
                if lbl and lbl in seen_fig_labels:
                    self.log(f"  Removed duplicate figure {lbl} from {sec.heading}")
                    return ""
                # Check by includegraphics filename
                file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
                if file_m:
                    fname = file_m.group(1)
                    if fname in seen_fig_files:
                        self.log(f"  Removed duplicate figure file {fname} from {sec.heading}")
                        return ""
                    seen_fig_files.add(fname)
                # Register label AFTER both checks pass (avoid phantom labels)
                if lbl:
                    seen_fig_labels.add(lbl)
                return block
            sec.content = re.sub(
                r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
                _dedup_figure, sec.content, flags=re.DOTALL,
            )
            # Clean up leftover blank lines from removed figures
            sec.content = re.sub(r'\n{3,}', '\n\n', sec.content)

        # Step 5: Build skeleton
        skeleton = PaperSkeleton(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            figures=[],
            template_format=template_format,
            references_bibtex=bibtex,
        )

        # Step 6: Render LaTeX + sanitize
        latex_content = self._render_latex(skeleton)
        latex_content = self._sanitize_latex(latex_content)

        # Step 6b-pre: Full-document figure dedup (safety net after assembly)
        # Per-section dedup ran earlier, but assembly/render might re-introduce duplicates
        seen_labels: set[str] = set()
        seen_files: set[str] = set()
        def _dedup_assembled(m: re.Match) -> str:
            block = m.group(0)
            lbl_m = re.search(r'\\label\{(fig:[^}]+)\}', block)
            if lbl_m:
                lbl = lbl_m.group(1)
                if lbl in seen_labels:
                    self.log(f"  Full-doc dedup: removed duplicate figure {lbl}")
                    return ""
                seen_labels.add(lbl)
            file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
            if file_m:
                fname = file_m.group(1)
                if fname in seen_files:
                    self.log(f"  Full-doc dedup: removed duplicate figure file {fname}")
                    return ""
                seen_files.add(fname)
            return block
        latex_content = re.sub(
            r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
            _dedup_assembled, latex_content, flags=re.DOTALL,
        )
        latex_content = re.sub(r'\n{3,}', '\n\n', latex_content)

        # Step 6b: Final LaTeX-level figure validation
        # Check that every figure file from figure_output has an \includegraphics in the tex
        latex_content = self._validate_figures_in_latex(latex_content, figure_output)

        # Step 6c: Resolve missing citations — find \cite keys not in bib, auto-fill
        bibtex = await self._resolve_missing_citations(latex_content, bibtex)

        # Step 6d: Citation coverage validation + must-cite enforcement
        citation_report = self._validate_citation_coverage(
            latex_content, ideation, cite_keys
        )
        if citation_report.get("missing_must_cites"):
            self.log(f"Must-cite enforcement: {len(citation_report['missing_must_cites'])} "
                     f"must-cite papers not referenced, injecting into Related Work")
            latex_content = self._inject_must_cites(
                latex_content, citation_report["missing_must_cites"], cite_keys, ideation
            )
            # Re-resolve in case new cite keys were introduced
            bibtex = await self._resolve_missing_citations(latex_content, bibtex)

        # Log citation quality report
        self._log_citation_report(citation_report)

        # Step 6e: Global consistency check
        consistency_issues = _check_global_consistency(
            latex_content, abstract, sections,
        )
        if consistency_issues:
            self.log(f"Consistency check: {len(consistency_issues)} issue(s) found")
            for issue in consistency_issues:
                self.log(f"  - {issue}")

        # Save outputs
        tex_path = self.workspace.write_text("drafts/paper.tex", latex_content)
        bib_content = self._sanitize_bibtex(bibtex)
        bib_path = self.workspace.write_text("drafts/references.bib", bib_content)
        skeleton_path = self.workspace.write_json(
            "drafts/paper_skeleton.json",
            skeleton.model_dump(mode="json"),
        )

        self.workspace.register_artifact("paper_tex", tex_path, self.stage)
        self.workspace.register_artifact("references_bib", bib_path, self.stage)
        self.workspace.register_artifact("paper_skeleton", skeleton_path, self.stage)

        # Step 7: Try to compile PDF
        pdf_result = await self._compile_pdf(tex_path, template_format=template_format)

        result = {
            "tex_path": str(tex_path),
            "bib_path": str(bib_path),
            "grounding": grounding.to_output_dict(),
            "consistency_issues": consistency_issues,
        }
        if "pdf_path" in pdf_result:
            result["pdf_path"] = pdf_result["pdf_path"]
            self.workspace.register_artifact(
                "paper_pdf", self.workspace.path / "drafts" / "paper.pdf", self.stage
            )
        else:
            result["pdf_error"] = pdf_result.get("error", "Unknown error")
            self.log(f"PDF compilation failed: {result['pdf_error']}")

        self.log("Writing stage complete")
        return result

    # ---- cite key management ------------------------------------------------

    # Surname prefixes that should be merged (e.g., "van der Waals" → "vanderwaals")
    _NAME_PREFIXES = frozenset({
        "van", "von", "de", "del", "della", "di", "du", "el", "le", "la",
        "bin", "ibn", "al", "das", "dos", "den", "der", "het", "ten",
    })

