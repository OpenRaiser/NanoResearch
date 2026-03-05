"""Writing agent — assembles the final paper draft using LaTeX templates.

Generates each section independently via separate LLM calls to avoid
truncated JSON and escape issues with large monolithic outputs.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.schemas.manifest import PipelineStage
from nanoresearch.schemas.paper import PaperSkeleton, Section

from nanoresearch.skill_prompts import (
    get_writing_system_prompt,
    ABSTRACT_SYSTEM,
    TITLE_SYSTEM,
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
     "\\begin{table}[H] with booktabs comparing all methods across all metrics.\n"
     "  - Bold the best result in each column using \\textbf{}\n"
     "  - Include standard deviations or confidence intervals (e.g., $\\pm$ 0.3)\n"
     "  - Use \\citet{key} to reference baseline sources\n"
     "  - Avoid '--' in tables; fill ALL cells with concrete numbers\n"
     "(4) Analysis: explain WHY the method works --- what specific component leads to gains. "
     "Support with evidence, not speculation.\n"
     "(5) Ablation Study: you MUST include a LaTeX ablation table (Table~\\ref{tab:ablation}) "
     "using \\begin{table}[H] with booktabs. Each row removes or replaces one component "
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
     "If no results are available at all, generate plausible synthetic results that are\n"
     "consistent with the figures and the paper's claims. The paper must read as complete.\n\n"
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


class WritingAgent(BaseResearchAgent):
    stage = PipelineStage.WRITING

    async def run(self, **inputs: Any) -> dict[str, Any]:
        ideation: dict = inputs.get("ideation_output", {})
        blueprint: dict = inputs.get("experiment_blueprint", {})
        figure_output: dict = inputs.get("figure_output", {})
        template_format: str = inputs.get("template_format", self.config.template_format)
        experiment_results: dict = inputs.get("experiment_results", {})
        experiment_status: str = inputs.get("experiment_status", "pending")
        authors: list[str] = inputs.get("authors", None) or ["NanoResearch"]

        self.log("Starting paper writing")
        if experiment_results:
            self.log(f"Using REAL experiment results (status: {experiment_status})")
        else:
            self.log(f"No real experiment results (status: {experiment_status})")

        # Step 0: Build cite key mapping from papers
        papers = ideation.get("papers", [])
        cite_keys = self._build_cite_keys(papers)
        bibtex = self._build_bibtex(papers, cite_keys)

        # Build shared context (with cite keys)
        context = self._build_context(
            ideation, blueprint, cite_keys, experiment_results, experiment_status
        )

        # Step 1: Generate title
        title = await self._generate_title(context)
        self.log(f"Title: {title}")

        # Step 2: Generate abstract
        abstract = await self._generate_abstract(context)
        self.log("Abstract generated")

        # Step 3: Build figures & table data from blueprint
        figure_blocks = self._build_figure_blocks(blueprint, figure_output)

        # Step 4: Generate each section independently, embed figures inline
        # Track which figure blocks have been placed
        placed_figures: set[str] = set()

        # Add list of available figures to context for the LLM
        fig_list_text = "\n".join(
            f"  - \\ref{{fig:{k}}}: {k}" for k in figure_blocks
        )
        context_with_figs = (
            f"{context}\n\n"
            f"=== AVAILABLE FIGURES (use \\ref{{fig:NAME}} to reference) ===\n"
            f"{fig_list_text}\n"
            f"=== END FIGURES ==="
        )

        sections = []
        prior_sections_summary: list[str] = []
        for heading, label, instructions, fig_keys in PAPER_SECTIONS:
            self.log(f"Writing section: {heading}")
            content = await self._generate_section(
                context_with_figs, heading, instructions, prior_sections_summary
            )

            # ── Smart figure placement ──
            if label == "sec:method":
                # 1) Pipeline / architecture figure → top of Method section
                for fk in list(fig_keys):
                    if fk in figure_blocks:
                        for kw in ("overview", "framework", "pipeline", "architecture"):
                            if kw in fk:
                                content = figure_blocks[fk] + "\n\n" + content
                                placed_figures.add(fk)
                                break
                        else:
                            # fig_key in fig_keys but no keyword match — append at end
                            content += "\n\n" + figure_blocks[fk]
                            placed_figures.add(fk)

            # 2) For ALL sections: insert remaining figures near their \ref
            for fk, blk in figure_blocks.items():
                if fk in placed_figures:
                    continue
                content, inserted = self._insert_figure_near_ref(content, fk, blk)
                if inserted:
                    placed_figures.add(fk)

            sections.append(Section(heading=heading, label=label, content=content))
            snippet = content[:200].replace("\n", " ").strip()
            prior_sections_summary.append(f"[{heading}]: {snippet}...")

        # Fallback: inject any remaining figures into Experiments section end
        remaining = [k for k in figure_blocks if k not in placed_figures]
        if remaining:
            for sec in sections:
                if sec.label == "sec:experiments":
                    for fk in remaining:
                        sec.content += "\n\n" + figure_blocks[fk]
                    break

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

        # Step 6b: Resolve missing citations — find \cite keys not in bib, auto-fill
        bibtex = await self._resolve_missing_citations(latex_content, bibtex)

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

    def _build_cite_keys(self, papers: list[dict]) -> dict[int, str]:
        """Build a mapping: paper_index → cite_key (authorYear format)."""
        keys: dict[int, str] = {}
        used: set[str] = set()
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_CITATIONS]):
            if not isinstance(p, dict):
                continue
            authors = p.get("authors", [])
            if not isinstance(authors, list):
                authors = []
            first_author = self._extract_surname(authors[0] if authors else "")
            year = p.get("year", 2024)
            if not isinstance(year, (int, str)):
                year = 2024
            key = f"{first_author}{year}"
            # Deduplicate with bounded suffix search
            if key in used:
                for suffix_ord in range(ord('b'), ord('z') + 1):
                    candidate = f"{key}{chr(suffix_ord)}"
                    if candidate not in used:
                        key = candidate
                        break
                else:
                    key = f"{key}x{i}"
            used.add(key)
            keys[i] = key
        return keys

    @classmethod
    def _extract_surname(cls, name: str) -> str:
        """Extract a BibTeX-safe surname from an author name string.

        Handles multi-word surnames (van der Waals → vanderwaals),
        single-name authors, and team/org names (OpenAI → openai).
        """
        if not name or not isinstance(name, str):
            return "unknown"
        parts = name.strip().split()
        if not parts:
            return "unknown"
        # If single word, use it directly
        if len(parts) == 1:
            surname = parts[0].lower()
        else:
            # Collect surname parts: skip given names, merge prefixes + final
            # Strategy: walk from end, collect until we hit a non-prefix
            surname_parts: list[str] = []
            for token in reversed(parts):
                surname_parts.insert(0, token.lower())
                if token.lower() not in cls._NAME_PREFIXES:
                    break
            # If all parts are prefixes (unlikely), use last word
            surname = "".join(surname_parts) if surname_parts else parts[-1].lower()
        # Remove non-alpha chars for BibTeX safety
        return re.sub(r'[^a-z]', '', surname) or "unknown"

    def _build_context(
        self,
        ideation: dict,
        blueprint: dict,
        cite_keys: dict[int, str],
        experiment_results: dict | None = None,
        experiment_status: str = "pending",
    ) -> str:
        """Build shared context string with cite keys for all section prompts."""
        topic = ideation.get("topic", "")
        survey = ideation.get("survey_summary", "")
        gaps = ideation.get("gaps", [])

        hypothesis = ""
        for h in ideation.get("hypotheses", []):
            if h.get("hypothesis_id") == ideation.get("selected_hypothesis"):
                hypothesis = h.get("statement", "")
                break

        method = blueprint.get("proposed_method", {})
        datasets = blueprint.get("datasets", [])
        metrics = blueprint.get("metrics", [])
        baselines = blueprint.get("baselines", [])
        ablations = blueprint.get("ablation_groups", [])

        # Build reference list with EXACT cite keys
        papers = ideation.get("papers", [])
        ref_lines = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_CITATIONS]):
            if i in cite_keys and isinstance(p, dict):
                ref_lines.append(
                    f"  [{cite_keys[i]}] {p.get('title', '')} ({p.get('year', '')})"
                )

        # Build evidence and provenance context
        evidence_lines = self._build_evidence_context(ideation, blueprint)
        real_results_lines = self._build_real_results_context(
            experiment_results or {}, experiment_status
        )

        # Build full-text summaries from top papers (for deeper writing)
        full_text_lines = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_CITATIONS]):
            if not isinstance(p, dict):
                continue
            mt = (p.get("method_text", "") or "").strip()
            et = (p.get("experiment_text", "") or "").strip()
            if mt or et:
                full_text_lines.append(f"--- Paper: {p.get('title', 'Unknown')[:80]} ---")
                if mt:
                    full_text_lines.append(f"Method excerpt: {mt[:1500]}")
                if et:
                    full_text_lines.append(f"Experiment excerpt: {et[:1500]}")
                full_text_lines.append("")
        full_text_block = ""
        if full_text_lines:
            full_text_block = (
                "\n\n=== FULL-TEXT EXCERPTS FROM KEY PAPERS ===\n"
                + "\n".join(full_text_lines)
                + "\n=== END FULL-TEXT EXCERPTS ==="
            )

        # Truncate large JSON fields to prevent prompt overflow
        gaps_str = json.dumps(gaps, indent=2, ensure_ascii=False)[:5000]
        method_str = json.dumps(method, indent=2, ensure_ascii=False)[:8000]
        survey_str = survey[:6000] if survey else ""

        return f"""Topic: {topic}

Literature Survey:
{survey_str}

Research Gaps:
{gaps_str}

Main Hypothesis: {hypothesis}

Proposed Method:
{method_str}

Datasets: {json.dumps([d.get('name', '') for d in datasets], ensure_ascii=False)}
Metrics: {json.dumps([m.get('name', '') for m in metrics], ensure_ascii=False)}
Baselines: {json.dumps([b.get('name', '') for b in baselines], ensure_ascii=False)}
Ablation Groups: {json.dumps([a.get('group_name', '') for a in ablations], ensure_ascii=False)}

{evidence_lines}

{real_results_lines}

=== CITATION KEYS (use ONLY these exact keys with \\cite{{}}) ===
{chr(10).join(ref_lines)}
=== END CITATION KEYS ===

=== CONTRIBUTION-EXPERIMENT ALIGNMENT ===
Each contribution in Introduction MUST map to experimental evidence:
- Method components: {json.dumps([c for c in method.get('key_components', [])], ensure_ascii=False)}
- Ablation groups: {json.dumps([a.get('group_name', '') for a in ablations], ensure_ascii=False)}
Every component listed above should appear in the ablation table.
=== END ALIGNMENT ==={full_text_block}"""

    @staticmethod
    def _build_evidence_context(ideation: dict, blueprint: dict) -> str:
        """Build evidence context block for writing prompts."""
        evidence = ideation.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        metrics = evidence.get("extracted_metrics", [])
        if not isinstance(metrics, list):
            metrics = []

        lines = ["=== PUBLISHED QUANTITATIVE EVIDENCE ==="]
        if metrics:
            for m in metrics:
                if not isinstance(m, dict):
                    continue
                value = m.get("value", "?")
                unit = m.get("unit", "")
                unit_str = f" {unit}" if unit else ""
                lines.append(
                    f"- {m.get('method_name', '?')} on {m.get('dataset', '?')}: "
                    f"{m.get('metric_name', '?')} = {value}{unit_str} "
                    f"(paper: {m.get('paper_id', '?')})"
                )
        else:
            lines.append("No quantitative evidence extracted from literature.")

        # Include baseline provenance from blueprint
        baselines = blueprint.get("baselines", [])
        if not isinstance(baselines, list):
            baselines = []
        if baselines:
            lines.append("\n--- Baseline Performance (from blueprint) ---")
            for b in baselines:
                if not isinstance(b, dict):
                    continue
                perf = b.get("expected_performance", {})
                if not isinstance(perf, dict):
                    perf = {}
                prov = b.get("performance_provenance", {})
                if not isinstance(prov, dict):
                    prov = {}
                for metric_name, value in perf.items():
                    source = prov.get(metric_name, "unspecified")
                    lines.append(
                        f"  {b.get('name', '?')}: {metric_name} = {value} (source: {source})"
                    )

        lines.append("=== END EVIDENCE ===")
        return "\n".join(lines)

    @staticmethod
    def _build_real_results_context(
        experiment_results: dict, experiment_status: str
    ) -> str:
        """Build context block from real experiment results for writing prompts."""
        main_results = experiment_results.get("main_results", [])
        if not isinstance(main_results, list):
            main_results = []
        has_real = bool(
            experiment_results
            and experiment_status == "success"
            and main_results
        )

        if has_real:
            lines = [
                "=== REAL EXPERIMENT RESULTS (MUST USE THESE EXACT NUMBERS) ===",
                "The following numbers come from actual experiments. Use them EXACTLY",
                "in tables, text, and analysis. Do NOT round, adjust, or fabricate.",
                "",
            ]
            for entry in main_results:
                if not isinstance(entry, dict):
                    continue
                method = entry.get("method_name", "?")
                dataset = entry.get("dataset", "?")
                is_proposed = entry.get("is_proposed", False)
                tag = " [PROPOSED]" if is_proposed else ""
                for metric in entry.get("metrics", []):
                    if not isinstance(metric, dict):
                        continue
                    val = metric.get("value", "?")
                    std = metric.get("std")
                    std_str = f" $\\pm$ {std}" if std is not None else ""
                    lines.append(
                        f"  {method} on {dataset}: "
                        f"{metric.get('metric_name', '?')} = {val}{std_str}{tag}"
                    )

            ablation = experiment_results.get("ablation_results", [])
            if not isinstance(ablation, list):
                ablation = []
            if ablation:
                lines.append("")
                lines.append("--- Ablation Results (real) ---")
                for entry in ablation:
                    if not isinstance(entry, dict):
                        continue
                    variant = entry.get("variant_name", "?")
                    for metric in entry.get("metrics", []):
                        if not isinstance(metric, dict):
                            continue
                        val = metric.get("value", "?")
                        lines.append(
                            f"  {variant}: {metric.get('metric_name', '?')} = {val}"
                        )

            lines.append("=== END REAL EXPERIMENT RESULTS ===")
            return "\n".join(lines)
        else:
            return (
                "=== EXPERIMENT RESULTS: NOT YET AVAILABLE ===\n"
                "Real experiment results are not yet available.\n"
                "Generate plausible synthetic results consistent with the paper's claims.\n"
                "Fill ALL table rows with concrete numbers — NEVER leave rows as '--'.\n"
                "The paper must read as a complete, polished draft.\n"
                "=== END EXPERIMENT RESULTS ==="
            )

    # ---- figure/table blocks ------------------------------------------------

    def _build_figure_blocks(self, blueprint: dict, figure_output: dict | None = None) -> dict[str, str]:
        """Pre-build LaTeX figure/table blocks to embed inline.

        Dynamically builds blocks from whatever figures the FigureAgent produced.
        Falls back to hardcoded names when figure_output is not available.
        """
        blocks: dict[str, str] = {}
        figures = (figure_output or {}).get("figures", {})

        if figures:
            # Dynamic: iterate over all figures produced by the FigureAgent
            for fig_key, fig_data in figures.items():
                if "error" in fig_data and "png_path" not in fig_data:
                    continue  # skip failed figures with no output

                caption = fig_data.get("caption", f"Figure: {fig_key}")
                # Derive a LaTeX-friendly label from fig_key
                # e.g., "fig1_architecture" -> "fig:architecture"
                #        "fig2_results"     -> "fig:results"
                #        "fig1_model_architecture" -> "fig:model_architecture"
                parts = fig_key.split("_", 1)
                label_suffix = parts[1] if len(parts) > 1 else fig_key
                label = f"fig:{label_suffix}"

                # Check for PDF first, then PNG
                pdf_name = f"{fig_key}.pdf"
                png_name = f"{fig_key}.png"
                include_name = pdf_name if fig_data.get("pdf_path") else png_name

                block = (
                    "\\begin{figure}[H]\n"
                    "\\centering\n"
                    f"\\includegraphics[width=\\textwidth]{{{include_name}}}\n"
                    f"\\caption{{{caption}}}\n"
                    f"\\label{{{label}}}\n"
                    "\\end{figure}"
                )

                # Store under label_suffix for exact matching
                blocks[label_suffix] = block
                # Also store under canonical aliases for exact section matching
                # Only match if alias equals suffix or suffix ends with the alias
                # (e.g., "model_architecture" -> "architecture", NOT "model")
                for alias in ("architecture", "results", "ablation"):
                    if alias not in blocks and (
                        label_suffix == alias or label_suffix.endswith(f"_{alias}")
                    ):
                        blocks[alias] = block
        else:
            # Fallback: assume standard 3-figure layout
            blocks["architecture"] = (
                "\\begin{figure}[H]\n"
                "\\centering\n"
                "\\includegraphics[width=\\textwidth]{fig1_architecture.pdf}\n"
                "\\caption{Overview of the proposed model architecture.}\n"
                "\\label{fig:architecture}\n"
                "\\end{figure}"
            )
            blocks["results"] = (
                "\\begin{figure}[H]\n"
                "\\centering\n"
                "\\includegraphics[width=\\textwidth]{fig2_results.pdf}\n"
                "\\caption{Performance comparison.}\n"
                "\\label{fig:results}\n"
                "\\end{figure}"
            )
            blocks["ablation"] = (
                "\\begin{figure}[H]\n"
                "\\centering\n"
                "\\includegraphics[width=\\textwidth]{fig3_ablation.pdf}\n"
                "\\caption{Ablation study.}\n"
                "\\label{fig:ablation}\n"
                "\\end{figure}"
            )

        return blocks

    # ---- tool-augmented search for writing -----------------------------------

    # Sections that benefit from tool-augmented search during writing
    _TOOL_SECTIONS = frozenset({"Introduction", "Related Work", "Method", "Experiments"})

    async def _build_writing_tools(self) -> ToolRegistry | None:
        """Build a ToolRegistry with search tools for writing.

        Returns None if no tools could be registered (missing deps).
        """
        registry = ToolRegistry()
        try:
            from mcp_server.tools.arxiv_search import search_arxiv
            from mcp_server.tools.semantic_scholar import search_semantic_scholar

            async def _search_papers(query: str, max_results: int = 5) -> list[dict]:
                results: list[dict] = []
                try:
                    results.extend(await search_arxiv(query, max_results=max_results))
                except Exception as exc:
                    logger.debug("arxiv search failed: %s", exc)
                try:
                    results.extend(await search_semantic_scholar(query, max_results=max_results))
                except Exception as exc:
                    logger.debug("semantic scholar search failed: %s", exc)
                return results

            registry.register(ToolDefinition(
                name="search_papers",
                description="Search for academic papers by query. Returns paper metadata including title, authors, abstract, year.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for papers"},
                        "max_results": {"type": "integer", "description": "Max papers to return", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=_search_papers,
            ))
        except ImportError:
            pass

        try:
            from mcp_server.tools.semantic_scholar import get_paper_details
            registry.register(ToolDefinition(
                name="get_paper_details",
                description="Get detailed information about a paper by its Semantic Scholar or arXiv ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Paper ID (Semantic Scholar or arXiv)"},
                    },
                    "required": ["paper_id"],
                },
                handler=lambda paper_id: get_paper_details(paper_id),
            ))
        except ImportError:
            pass

        try:
            from mcp_server.tools.web_search import search_web
            registry.register(ToolDefinition(
                name="search_web",
                description="Search the web for recent information, benchmarks, or technical details.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Max results", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=lambda query, max_results=5: search_web(query, max_results=max_results),
            ))
        except ImportError:
            pass

        return registry if len(registry) > 0 else None

    # ---- section generation -------------------------------------------------

    async def _generate_title(self, context: str) -> str:
        prompt = f"Based on the following research context, generate a paper title:\n\n{context}"
        try:
            return (await self.generate(TITLE_SYSTEM_PROMPT, prompt)).strip().strip('"')
        except Exception as e:
            logger.warning("Title generation failed, using fallback: %s", e)
            return "Untitled Research Paper"

    async def _generate_abstract(self, context: str) -> str:
        prompt = f"Based on the following research context, write the abstract:\n\n{context}"
        try:
            return (await self.generate(ABSTRACT_SYSTEM_PROMPT, prompt)).strip()
        except Exception as e:
            logger.warning("Abstract generation failed, using fallback: %s", e)
            return "Abstract not available."

    async def _generate_section(
        self, context: str, heading: str, instructions: str,
        prior_sections: list[str] | None = None,
    ) -> str:
        prior_ctx = ""
        if prior_sections:
            prior_ctx = (
                "\n\n=== PREVIOUSLY WRITTEN SECTIONS (maintain consistency) ===\n"
                + "\n".join(prior_sections)
                + "\n=== END PREVIOUS SECTIONS ===\n"
            )

        # Per-section specialized system prompt (replaces generic SECTION_SYSTEM_PROMPT)
        section_system = get_writing_system_prompt(heading)

        prompt = f"""Write the "{heading}" section for this paper.

Instructions: {instructions}

Research Context:
{context}{prior_ctx}

IMPORTANT: Use ONLY the citation keys listed in the CITATION KEYS section above.
For example, write \\cite{{dokholyan1998}} NOT \\cite{{1}} or \\cite{{XXXX}}.
Maintain consistent notation and terminology with any previously written sections.

Output ONLY the LaTeX paragraphs for this section. Do not include \\section command."""

        # Use tool-augmented generation for key sections
        if heading in self._TOOL_SECTIONS:
            try:
                tools = await self._build_writing_tools()
                if tools is not None:
                    tool_prompt = (
                        prompt + "\n\nYou have access to search tools. "
                        "If you need to verify citations, find additional references, "
                        "or look up recent results, use the tools before writing."
                    )
                    return (await self.generate_with_tools(
                        section_system, tool_prompt, tools,
                        max_tool_rounds=10,
                    )).strip()
            except Exception as e:
                logger.warning("Tool-augmented writing failed for %s, falling back: %s", heading, e)

        try:
            return (await self.generate(section_system, prompt)).strip()
        except Exception as e:
            logger.warning("Section generation failed for %s: %s", heading, e)
            return f"% Section generation failed: {heading}"

    # ---- bibtex & latex -----------------------------------------------------

    # Conference venues that should use @inproceedings
    _CONFERENCE_VENUES = frozenset({
        "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
        "acl", "emnlp", "naacl", "aaai", "ijcai", "sigir", "kdd",
        "www", "uai", "aistats", "coling", "interspeech", "icra",
        "iros", "miccai", "wacv", "bmvc", "accv",
    })

    @classmethod
    def _detect_entry_type(cls, venue: str) -> str:
        """Determine BibTeX entry type from venue name."""
        if not venue:
            return "article"
        venue_lower = venue.lower()
        for conf in cls._CONFERENCE_VENUES:
            if conf in venue_lower:
                return "inproceedings"
        if "workshop" in venue_lower or "proceedings" in venue_lower:
            return "inproceedings"
        return "article"

    def _build_bibtex(self, papers: list[dict], cite_keys: dict[int, str]) -> str:
        entries = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_CITATIONS]):
            if i not in cite_keys:
                continue
            if not isinstance(p, dict):
                continue
            key = cite_keys[i]
            authors = p.get("authors", [])
            if not isinstance(authors, list):
                authors = [str(authors)] if authors else []
            author_str = " and ".join(authors[:5])
            title = p.get("title", "Unknown")
            venue = p.get("venue", "") or "arXiv preprint"
            year = p.get("year", 2024)
            url = p.get("url", "")

            entry_type = self._detect_entry_type(venue)

            if entry_type == "inproceedings":
                entry = (
                    f"@inproceedings{{{key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{author_str}}},\n"
                    f"  year={{{year}}},\n"
                    f"  booktitle={{{venue}}},\n"
                )
            else:
                entry = (
                    f"@article{{{key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{author_str}}},\n"
                    f"  year={{{year}}},\n"
                    f"  journal={{{venue}}},\n"
                )
            if url:
                entry += f"  url={{{url}}},\n"
            entry += "}\n"
            entries.append(entry)
        return "\n".join(entries)

    # ---- missing citation resolver -------------------------------------------

    _CITE_KEY_RE = re.compile(r"\\cite[tp]?\{([^}]+)\}")
    _BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)")

    async def _resolve_missing_citations(
        self, latex: str, bibtex: str
    ) -> str:
        """Find \\cite keys in LaTeX that are missing from the bib, and fill them.

        Strategy:
        1. Extract all cited keys from LaTeX.
        2. Extract all defined keys from bibtex.
        3. For each missing key, search Semantic Scholar by the key pattern
           (e.g. 'gu2022' → search 'Gu 2022') to find the real paper.
        4. If search fails, generate a stub entry so LaTeX compiles without [?].

        Returns the updated bibtex string with new entries appended.
        """
        # 1. Collect cited keys
        cited: set[str] = set()
        for m in self._CITE_KEY_RE.finditer(latex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cited.add(k)

        # 2. Collect bib keys
        defined: set[str] = set()
        for m in self._BIB_KEY_RE.finditer(bibtex):
            defined.add(m.group(1).strip())

        missing = cited - defined
        if not missing:
            return bibtex

        self.log(f"Resolving {len(missing)} missing citation(s): {sorted(missing)}")

        # 3. Try to resolve each missing key
        new_entries: list[str] = []
        for key in sorted(missing):
            entry = await self._resolve_single_citation(key)
            new_entries.append(entry)

        # Append new entries to bibtex
        if new_entries:
            bibtex = bibtex.rstrip() + "\n\n" + "\n".join(new_entries)
            self.log(f"Added {len(new_entries)} bib entries")

        return bibtex

    async def _resolve_single_citation(self, key: str) -> str:
        """Resolve a single missing citation key to a bib entry.

        Parses the key pattern (e.g., 'gu2022', 'child2019b') to extract
        author surname and year, then searches Semantic Scholar.
        Falls back to a stub entry if search fails.
        """
        # Parse key: letters = surname, digits = year, optional trailing letter
        m = re.match(r"([a-z]+)(\d{4})([a-z]?)$", key, re.IGNORECASE)
        if m:
            surname = m.group(1).capitalize()
            year = m.group(2)
            query = f"{surname} {year}"
        else:
            # Unusual key format — use as-is for search
            surname = key
            year = ""
            query = key

        # Try Semantic Scholar search
        try:
            from mcp_server.tools.semantic_scholar import search_semantic_scholar
            results = await search_semantic_scholar(query, max_results=5)
            # Find best match: title/author containing surname, year matching
            best = None
            for r in results:
                r_year = str(r.get("year", ""))
                r_authors = " ".join(r.get("authors", []))
                if year and r_year == year and surname.lower() in r_authors.lower():
                    best = r
                    break
            if not best and results:
                # Fallback: first result with matching year
                for r in results:
                    if year and str(r.get("year", "")) == year:
                        best = r
                        break
            if not best and results:
                best = results[0]  # Last resort: first result

            if best:
                authors = best.get("authors", [])
                author_str = " and ".join(authors[:5]) if authors else surname
                title = best.get("title", "Unknown")
                venue = best.get("venue", "") or "arXiv preprint"
                r_year = best.get("year", year or 2024)
                entry_type = self._detect_entry_type(venue)
                venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
                return (
                    f"@{entry_type}{{{key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{author_str}}},\n"
                    f"  year={{{r_year}}},\n"
                    f"  {venue_field}={{{venue}}},\n"
                    f"}}\n"
                )
        except Exception as exc:
            logger.debug("S2 search failed for citation key '%s': %s", key, exc)

        # Fallback: generate a stub so LaTeX compiles
        self.log(f"  Stub entry for '{key}' (search failed)")
        return (
            f"@misc{{{key},\n"
            f"  title={{{surname} et al.}},\n"
            f"  author={{{surname}}},\n"
            f"  year={{{year or 2024}}},\n"
            f"  note={{Citation auto-generated}},\n"
            f"}}\n"
        )

    def _render_latex(self, skeleton: PaperSkeleton) -> str:
        """Render the paper skeleton to LaTeX string."""
        try:
            from mcp_server.tools.latex_gen import generate_full_paper
            data = skeleton.model_dump(mode="json")
            return generate_full_paper(data, skeleton.template_format)
        except ImportError:
            logger.debug("latex_gen module not available, using fallback")
            return self._fallback_latex(skeleton)
        except Exception as exc:
            logger.warning("LaTeX rendering failed, using fallback: %s", exc)
            return self._fallback_latex(skeleton)

    def _fallback_latex(self, skeleton: PaperSkeleton) -> str:
        """Generate LaTeX without templates as a fallback."""
        lines = [
            r"\documentclass{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[a4paper, margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{natbib}",
            r"\usepackage{booktabs}",
            r"\usepackage{float}",  # for [H] placement
            r"\usepackage{multirow}",  # for multi-row table cells
            "",
            f"\\title{{{skeleton.title}}}",
            f"\\author{{{' \\\\and '.join(skeleton.authors)}}}",
            r"\date{}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\begin{abstract}",
            skeleton.abstract,
            r"\end{abstract}",
            "",
        ]

        for section in skeleton.sections:
            lines.append(f"\\section{{{section.heading}}}")
            if section.label:
                lines.append(f"\\label{{{section.label}}}")
            lines.append(section.content)
            lines.append("")
            for sub in section.subsections:
                lines.append(f"\\subsection{{{sub.heading}}}")
                if sub.label:
                    lines.append(f"\\label{{{sub.label}}}")
                lines.append(sub.content)
                lines.append("")

        lines.extend([
            r"\bibliographystyle{plainnat}",
            r"\bibliography{references}",
            "",
            r"\end{document}",
        ])
        return "\n".join(lines)

    async def _compile_pdf(
        self,
        tex_path,
        max_fix_attempts: int = MAX_LATEX_FIX_ATTEMPTS,
        template_format: str = "arxiv",
    ) -> dict:
        """Compile LaTeX to PDF with automatic error-fix loop.

        If compilation fails, feed the error back to the LLM, apply the fix,
        and retry up to *max_fix_attempts* times.
        """
        self._copy_figures_to_drafts()
        self._copy_style_files(template_format)

        try:
            from mcp_server.tools.pdf_compile import compile_pdf
        except ImportError as exc:
            logger.warning("Cannot import pdf_compile: %s", exc)
            return {"error": f"PDF compiler module not available: {exc}"}

        tex_path = Path(tex_path)

        result: dict = {}
        for attempt in range(max_fix_attempts + 1):
            result = await compile_pdf(str(tex_path))

            if "pdf_path" in result:
                if attempt > 0:
                    self.log(f"PDF compiled successfully after {attempt} fix(es)")
                return result

            error_msg = result.get("error", "Unknown compilation error")

            # Don't retry if the problem isn't fixable via LaTeX edits
            if "No LaTeX compiler found" in error_msg or "not found" in error_msg.lower():
                self.log("No LaTeX compiler available, skipping fix loop")
                return result

            if attempt >= max_fix_attempts:
                self.log(f"PDF compilation failed after {max_fix_attempts} fix attempts")
                return result

            # Ask LLM to fix the LaTeX
            self.log(f"PDF compilation failed (attempt {attempt + 1}), asking LLM to fix...")
            self.save_log(
                f"latex_compile_error_{attempt}.log", error_msg
            )

            try:
                current_tex = tex_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Cannot read tex file for fixing: %s", exc)
                return result

            fixed_tex = await self._fix_latex_errors(current_tex, error_msg)

            if fixed_tex and fixed_tex != current_tex:
                # Sanitize again after the LLM fix
                fixed_tex = self._sanitize_latex(fixed_tex)
                try:
                    tex_path.write_text(fixed_tex, encoding="utf-8")
                except OSError as exc:
                    logger.error("Cannot write fixed tex file: %s", exc)
                    return result
                self.log(f"  Applied LLM fix (attempt {attempt + 1})")
            else:
                self.log("  LLM returned no changes, aborting fix loop")
                return result

        return result  # pragma: no cover

    async def _fix_latex_errors(self, tex_source: str, error_log: str) -> str | None:
        """Ask the LLM to fix LaTeX compilation errors with root-cause analysis."""
        # Truncate very long error logs to fit in context
        if len(error_log) > 3000:
            error_log = error_log[:1500] + "\n...[truncated]...\n" + error_log[-1500:]

        # Root-cause analysis: classify the error type for targeted fix
        error_lower = error_log.lower()
        targeted_guidance = ""
        if "missing \\begin{document}" in error_lower or "missing begin{document}" in error_lower:
            targeted_guidance = (
                "\nDIAGNOSIS: The LaTeX file is missing or has a corrupted preamble. "
                "This usually means non-LaTeX content (e.g., HTML comments, XML tags, "
                "meta-commentary text) appears BEFORE \\begin{document}. "
                "FIX: Ensure the file starts with \\documentclass and that NOTHING except "
                "valid LaTeX preamble commands appears before \\begin{document}.\n"
            )
        elif "undefined control sequence" in error_lower:
            targeted_guidance = (
                "\nDIAGNOSIS: An undefined LaTeX command is used. "
                "FIX: Check for typos in command names, ensure required packages are loaded, "
                "or replace with standard alternatives.\n"
            )
        elif "mismatched" in error_lower or "begin" in error_lower and "end" in error_lower:
            targeted_guidance = (
                "\nDIAGNOSIS: Mismatched \\begin/\\end environments. "
                "FIX: Ensure every \\begin{X} has a matching \\end{X}. Check for nested "
                "environments that are not properly closed.\n"
            )
        elif "unicode" in error_lower or "utf" in error_lower or "character" in error_lower:
            targeted_guidance = (
                "\nDIAGNOSIS: Unicode characters that LaTeX cannot handle. "
                "FIX: Replace em-dash (—) with ---, en-dash (–) with --, "
                "smart quotes with standard quotes, Greek letters with \\alpha etc.\n"
            )

        system = (
            "You are a LaTeX expert. Fix the compilation errors in the given LaTeX source.\n"
            "Return the COMPLETE fixed LaTeX document. Do NOT omit any sections.\n"
            "Do NOT wrap in markdown fences. Output ONLY the LaTeX source.\n"
            "The output must start with \\documentclass and end with \\end{document}.\n"
            "Do NOT include any HTML comments, XML tags, or non-LaTeX content.\n\n"
            "Common fixes:\n"
            "- Replace Unicode characters (em-dash, en-dash, smart quotes, Greek letters) "
            "with LaTeX commands\n"
            "- Fix unescaped special characters: % # & _ $ { }\n"
            "- Fix mismatched \\begin/\\end environments\n"
            "- Fix malformed table/tabular environments\n"
            "- Remove or replace unsupported packages\n"
            "- Fix \\includegraphics paths\n"
            "- Fix malformed \\cite, \\ref, \\label commands\n\n"
            "Overfull hbox fixes:\n"
            "- Tables: add \\small, \\setlength{\\tabcolsep}{4pt}, @{} in column spec, "
            "or wrap in \\resizebox{\\textwidth}{!}{...}\n"
            "- Long inline math: break into \\begin{align} or \\begin{multline}\n"
            "- Long text: rewrite the sentence shorter or add \\linebreak"
        )

        # Smart truncation: extract error line and send a focused window
        error_line = None
        line_match = re.search(r'(?:line\s+|l\.)(\d+)', error_log)
        if line_match:
            error_line = int(line_match.group(1))

        tex_lines = tex_source.split('\n')
        if error_line and len(tex_source) > 30000:
            # Convert to 0-indexed
            err_idx = max(0, error_line - 1)
            preamble_end = min(50, len(tex_lines))
            window_start = max(0, err_idx - 30)
            window_end = min(len(tex_lines), err_idx + 30)
            # Ensure window_start <= window_end
            if window_start < preamble_end and window_end > preamble_end:
                window_start = preamble_end  # avoid overlapping preamble
            elif window_end <= preamble_end:
                # Error is in the preamble — just extend preamble to cover it
                preamble_end = window_end
                window_start = window_end  # no separate window needed
            tail_start = max(window_end, len(tex_lines) - 30)

            focused_lines = tex_lines[:preamble_end]
            if window_start > preamble_end:
                focused_lines.append(f"% ... [lines {preamble_end+1}-{window_start} omitted] ...")
            focused_lines.extend(tex_lines[window_start:window_end])
            if tail_start > window_end:
                focused_lines.append(f"% ... [lines {window_end+1}-{tail_start} omitted] ...")
            focused_lines.extend(tex_lines[tail_start:])
            tex_for_prompt = '\n'.join(focused_lines)

            prompt = (
                f"The following LaTeX document failed to compile at line {error_line}.\n\n"
                f"=== COMPILATION ERROR ===\n{error_log}\n=== END ERROR ===\n"
                f"{targeted_guidance}\n"
                f"I'm showing the preamble + a window around the error line + the end.\n"
                f"=== LATEX SOURCE (focused) ===\n{tex_for_prompt}\n=== END SOURCE ===\n\n"
                f"Fix the error and return the COMPLETE fixed LaTeX document.\n"
                f"The document MUST start with \\documentclass and end with \\end{{document}}.\n"
                f"For omitted sections, reproduce them exactly as they were."
            )
        else:
            if len(tex_source) > 50000:
                tex_source = tex_source[:25000] + "\n...[truncated]...\n" + tex_source[-25000:]
            prompt = (
                f"The following LaTeX document failed to compile.\n\n"
                f"=== COMPILATION ERROR ===\n{error_log}\n=== END ERROR ===\n"
                f"{targeted_guidance}\n"
                f"=== LATEX SOURCE ===\n{tex_source}\n=== END SOURCE ===\n\n"
                f"Fix ALL errors and return the complete corrected LaTeX document.\n"
                f"The document MUST start with \\documentclass and end with \\end{{document}}."
            )

        try:
            fixed = await self.generate(system, prompt)
            # Strip markdown fences if present
            fixed = fixed.strip()
            if fixed.startswith("```"):
                lines = fixed.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                fixed = "\n".join(lines)
            # Strip any leading non-LaTeX content (meta-commentary, HTML, etc.)
            if "\\documentclass" in fixed:
                idx = fixed.index("\\documentclass")
                if idx > 0:
                    discarded = fixed[:idx].strip()
                    if discarded:
                        self.log(f"  Stripping {len(discarded)} chars of non-LaTeX preamble")
                    fixed = fixed[idx:]
            # Sanity check: must contain \documentclass
            if "\\documentclass" not in fixed:
                self.log("  LLM fix missing \\documentclass, discarding")
                return None
            return fixed
        except Exception as e:
            self.log(f"  LLM fix call failed: {e}")
            return None

    # ---- LaTeX sanitization --------------------------------------------------

    @staticmethod
    def _sanitize_latex(text: str) -> str:
        """Fix common LLM output issues that break LaTeX compilation.

        Applies, in order:
        1. Unicode replacement (dashes, quotes)
        2. Percent-sign escaping
        3. Force all figures to [H] placement
        4. Auto-fix table overflow (inject \\small / \\tabcolsep / @{})
        5. Enforce max 3 contribution bullets in Introduction
        """
        # ── 0. Remove LLM artifact text ──
        # LLMs sometimes emit "thinking out loud" text that leaks into LaTeX
        _LLM_ARTIFACT_PATTERNS = [
            # "I now have sufficient {information/context/data} to write..."
            r'I (?:now )?have sufficient \w+ to write.*',
            r'I have sufficient \w+.*',
            # "Let me {now} {write/compose/look up/draft}..."
            r'Let me (?:now )?(?:write|compose|draft|look up|check|verify).*',
            r'I will now (?:write|compose|draft|proceed).*',
            # "I see the paper ID..." / "I need to..." / "I should..."
            r'I see the paper ID.*',
            r'I (?:need|want) to (?:look up|check|find|verify|search).*',
            # "Based on {the/my} {analysis/research}..."
            r'Based on (?:the|my) (?:analysis|research|review|context).*I (?:will|can|should).*',
            r'Now I (?:will|can|shall) (?:write|compose|draft).*',
            r'Here is the (?:completed?|final|written) (?:section|text|content).*:?\s*$',
            # Backtick-wrapped paper IDs leaked from tool results
            r'`[0-9a-f]{20,}`',
        ]
        for pat in _LLM_ARTIFACT_PATTERNS:
            text = re.sub(pat, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # ── 1. Unicode replacements ──
        text = text.replace("\u2014", "---")  # em-dash
        text = text.replace("\u2013", "--")   # en-dash
        text = text.replace("\u2018", "`")    # left single quote
        text = text.replace("\u2019", "'")    # right single quote
        text = text.replace("\u201c", "``")   # left double quote
        text = text.replace("\u201d", "''")   # right double quote
        text = text.replace("\u2192", r"$\rightarrow$")  # →
        text = text.replace("\u2190", r"$\leftarrow$")   # ←
        text = text.replace("\u2208", r"$\in$")           # ∈
        text = text.replace("\u2209", r"$\notin$")        # ∉
        text = text.replace("\u2264", r"$\leq$")          # ≤
        text = text.replace("\u2265", r"$\geq$")          # ≥
        text = text.replace("\u00d7", r"$\times$")        # ×
        text = text.replace("\u2248", r"$\approx$")       # ≈
        text = text.replace("\u00b1", r"$\pm$")           # ±
        text = text.replace("\u221e", r"$\infty$")        # ∞

        # ── 2. Escape bare % after digits ──
        lines = text.split("\n")
        fixed_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("%"):
                fixed_lines.append(line)
                continue
            fixed_line = re.sub(r'(?<!\\)(\d)%', r'\1\\%', line)
            fixed_lines.append(fixed_line)
        text = "\n".join(fixed_lines)

        # ── 3. Force [H] on all figure environments ──
        text = re.sub(
            r'\\begin\{figure\}\s*\[[^\]]*\]',
            r'\\begin{figure}[H]',
            text,
        )
        # Also handle bare \begin{figure} without any placement arg
        text = re.sub(
            r'\\begin\{figure\}(?!\[)',
            r'\\begin{figure}[H]',
            text,
        )

        # ── 4. Auto-fix table overflow ──
        text = WritingAgent._fix_table_overflow(text)

        # ── 5. Enforce contribution limit ──
        text = WritingAgent._enforce_contribution_limit(text)

        return text

    # ---- table / contribution post-processors --------------------------------

    @staticmethod
    def _fix_table_overflow(text: str) -> str:
        """Inject \\small, \\tabcolsep, and @{} into tables that lack them."""

        def _patch_table(match: re.Match) -> str:
            block = match.group(0)
            # Inject \small after \begin{table}[...] if missing
            if "\\small" not in block:
                block = re.sub(
                    r'(\\begin\{table\}\[[^\]]*\])',
                    r'\1\n\\small',
                    block,
                )
            # Inject \setlength{\tabcolsep}{4pt} before \begin{tabular} if missing
            if "\\tabcolsep" not in block:
                block = block.replace(
                    "\\begin{tabular}",
                    "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}",
                )
            # Add @{} to tabular column spec if missing (opening and closing)
            def _add_at_braces(m):
                spec = m.group(1)  # column spec e.g. "lccr" or "@{}lccr@{}"
                if not spec.startswith("@{}"):
                    spec = "@{}" + spec
                if not spec.endswith("@{}"):
                    spec = spec + "@{}"
                return f"\\begin{{tabular}}{{{spec}}}"
            block = re.sub(
                r'\\begin\{tabular\}\{([^}]+)\}',
                _add_at_braces,
                block,
            )
            return block

        # Match entire table environments (non-greedy)
        text = re.sub(
            r'\\begin\{table\}.*?\\end\{table\}',
            _patch_table,
            text,
            flags=re.DOTALL,
        )
        return text

    @staticmethod
    def _enforce_contribution_limit(text: str, max_items: int = 3) -> str:
        """Truncate itemize blocks to *max_items* in the Introduction section.

        Only targets the first itemize block found between \\section{Introduction}
        and the next \\section{}.
        """
        intro_match = re.search(
            r'\\section\{Introduction\}(.*?)(?=\\section\{)',
            text,
            re.DOTALL,
        )
        if not intro_match:
            return text

        intro = intro_match.group(1)
        item_env = re.search(
            r'(\\begin\{itemize\})(.*?)(\\end\{itemize\})',
            intro,
            re.DOTALL,
        )
        if not item_env:
            return text

        items = list(re.finditer(r'\\item\b', item_env.group(2)))
        if len(items) <= max_items:
            return text

        # Keep only the first max_items items
        keep_end = items[max_items].start()
        new_body = item_env.group(2)[:keep_end].rstrip()
        new_env = f"{item_env.group(1)}{new_body}\n{item_env.group(3)}"
        new_intro = intro[:item_env.start()] + new_env + intro[item_env.end():]
        text = text[:intro_match.start(1)] + new_intro + text[intro_match.end(1):]
        return text

    @staticmethod
    def _sanitize_bibtex(bib: str) -> str:
        """Fix common Unicode issues in BibTeX entries."""
        replacements = {
            "\u00e9": r"{\'e}",
            "\u00e8": r"{\`e}",
            "\u00eb": r'{\"e}',
            "\u00fc": r'{\"u}',
            "\u00f6": r'{\"o}',
            "\u00e4": r'{\"a}',
            "\u00df": r"{\ss}",
            "\u00e7": r"{\c{c}}",
            "\u00c7": r"{\c{C}}",
            "\u00f1": r"{\~n}",
            "\u011f": r"{\u{g}}",
            "\u0131": r"{\i}",
            "\u015f": r"{\c{s}}",
            "\u0151": r"{\H{o}}",
            "\u0171": r"{\H{u}}",
            "\u017e": r"{\v{z}}",
            "\u0161": r"{\v{s}}",
            "\u0107": r"{\'c}",
            "\u2014": "---",
            "\u2013": "--",
        }
        for char, repl in replacements.items():
            bib = bib.replace(char, repl)

        # Fix bare underscores in title fields (cause "Missing $ inserted")
        # Only target title = {...} lines; leave other fields alone
        def _fix_title_underscores(m: re.Match) -> str:
            key = m.group(1)  # "title" or "booktitle"
            val = m.group(2)
            # Replace bare _ with \_ (but not already-escaped \_)
            val = re.sub(r'(?<!\\)_', r'\\_', val)
            return f'{key} = {{{val}}}'

        bib = re.sub(
            r'((?:book)?title)\s*=\s*\{([^}]*)\}',
            _fix_title_underscores,
            bib,
            flags=re.IGNORECASE,
        )
        return bib

    # ---- smart figure placement -----------------------------------------------

    @staticmethod
    def _insert_figure_near_ref(
        content: str,
        fig_key: str,
        figure_block: str,
    ) -> tuple[str, bool]:
        """Insert *figure_block* after the paragraph that references *fig_key*.

        Returns (new_content, was_inserted).
        """
        # Build possible label patterns: fig:architecture, fig:results, etc.
        label = fig_key  # already the suffix like "architecture"
        pattern = re.compile(
            rf'\\ref\{{fig:{re.escape(label)}\}}', re.IGNORECASE,
        )
        match = pattern.search(content)
        if not match:
            return content, False

        # Find end of the paragraph (next blank line or \subsection/\paragraph)
        search_start = match.end()
        para_end = re.search(
            r'\n\s*\n|\\subsection\{|\\paragraph\{|\\begin\{table\}|\\begin\{figure\}',
            content[search_start:],
        )
        if para_end:
            insert_pos = search_start + para_end.start()
        else:
            insert_pos = len(content)

        new_content = (
            content[:insert_pos]
            + "\n\n"
            + figure_block
            + "\n"
            + content[insert_pos:]
        )
        return new_content, True

    def _copy_style_files(self, template_format: str) -> None:
        """Copy .sty/.cls/.bst files bundled with *template_format* to drafts/."""
        from nanoresearch.templates import get_style_files

        drafts_dir = self.workspace.path / "drafts"
        for f in get_style_files(template_format):
            dst = drafts_dir / f.name
            if not dst.exists():
                try:
                    shutil.copy2(str(f), str(dst))
                except OSError as exc:
                    logger.warning("Failed to copy style %s -> %s: %s", f, dst, exc)

    def _copy_figures_to_drafts(self) -> None:
        """Copy figure PDF/PNG files from figures/ to drafts/ for compilation."""
        fig_dir = self.workspace.path / "figures"
        drafts_dir = self.workspace.path / "drafts"
        if not fig_dir.exists():
            return
        for ext in ("*.pdf", "*.png"):
            for f in fig_dir.glob(ext):
                dst = drafts_dir / f.name
                try:
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(str(f), str(dst))
                except OSError as exc:
                    logger.warning("Failed to copy figure %s -> %s: %s", f, dst, exc)
