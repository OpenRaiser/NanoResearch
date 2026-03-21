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
from nanoresearch.schemas.manifest import PaperMode, PipelineStage
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

# Survey section definitions: (heading, label, writing_instructions, fig_keys)
# short: 5 sections | standard: 7 sections | long: 9 sections
SURVEY_SECTIONS = {
    "short": [
        ("Introduction", "sec:intro",
         "Write 4-5 paragraphs:\n"
         "(1) Opening: establish the importance and scope of the topic. "
         "Cover the breath of work in this area and why it matters.\n"
         "(2) Coverage: state the number of papers reviewed, the year range, "
         "and the number of thematic clusters identified.\n"
         "(3) Taxonomy brief: briefly preview the main categories of methods/approaches "
         "covered in this survey.\n"
         "(4) Contributions: state what this survey provides (e.g., systematic review, "
         "unified taxonomy, comparative analysis, open challenges). Use a bullet list "
         "with 2-3 \\item entries.\n"
         "Use assertive language throughout. Cite key papers establishing importance.",
         []),

        ("Related Work", "sec:related",
         "Write 5-7 paragraphs organized by THEMATIC CLUSTERS.\n"
         "For each cluster: summarize 3-5 representative papers, show the evolution of ideas, "
         "and highlight the key differences between approaches.\n"
         "Organize by theme (e.g., by architecture, by task, by training paradigm), "
         "NOT chronologically.\n"
         "IMPORTANT: Every \\cite{} or \\citet{} key MUST be from the provided CITATION KEYS list. "
         "Do NOT invent citation keys.\n"
         "MUST-CITE ENFORCEMENT: If the context includes a 'MUST-CITE PAPERS' section, "
         "you MUST cite ALL of those papers in this section.\n"
         "FAIRNESS: discuss the STRONGEST and most influential works, not just peripheral ones.\n"
         "Use \\citet{key} when author is subject, \\citep{key} for parenthetical.",
         []),

        ("Taxonomy", "sec:taxonomy",
         "Write 6-8 paragraphs presenting a structured taxonomy of methods/approaches "
         "in this area.\n"
         "Structure:\n"
         "(1) Overview: describe the organizing principles of the taxonomy and list "
         "the top-level categories.\n"
         "(2-5) One \\subsection{} per major category. For each:\n"
         "  - Define the category and its distinguishing characteristics\n"
         "  - List sub-categories or variants with representative works (cite them)\n"
         "  - Explain the relationship between this category and others in the taxonomy\n"
         "  - Note strengths and weaknesses of approaches in this category\n"
         "(6) Comparative overview: a summary table or paragraph contrasting categories "
         "on key dimensions (e.g., scalability, data efficiency, generalization).\n"
         "Cite representative papers using \\citep{key}. Use the theme clusters from "
         "the context as your organizing structure.",
         []),

        ("Methods", "sec:method",
         "Write 6-8 paragraphs systematically analyzing the METHODOLOGICAL designs "
         "across reviewed papers.\n"
         "Structure:\n"
         "(1) Overview: frame the problem formally and describe the space of solution designs.\n"
         "(2-4) Detailed analysis of key technical components:\n"
         "  - For each major component, compare how different categories implement it\n"
         "  - Use numbered equations to illustrate key formulations where illuminating\n"
         "  - Reference equations in text: 'as defined in Eq.~\\eqref{eq:xxx}'\n"
         "  - Explain design trade-offs and why different schools of thought make different choices\n"
         "(5) Training and optimization: compare training objectives, optimizers, and "
         "regularization strategies across approaches.\n"
         "(6) Complexity and efficiency: discuss computational costs, FLOPs, or inference "
         "time comparisons where available.\n"
         "Use \\begin{align} for multi-line equations (NEVER eqnarray).\n"
         "Do NOT include \\begin{figure} blocks --- figures are inserted automatically.",
         []),

        ("Conclusion", "sec:conclusion",
         "Write 3-4 paragraphs:\n"
         "(1) Summarize the overall landscape --- main themes, methodological trends, "
         "and key findings from the survey.\n"
         "(2) Open challenges: 3-4 concrete, specific research problems that remain unsolved "
         "or underexplored (draw from the key_challenges in context).\n"
         "(3) Future directions: 2-3 promising research trajectories based on the "
         "identified gaps and future_directions from the literature.\n"
         "(4) Closing: briefly position the survey's contribution to the field.\n"
         "Do NOT introduce new results or citations here.",
         []),
    ],

    "standard": [
        # short sections +
        ("Introduction", "sec:intro",
         "Write 4-5 paragraphs:\n"
         "(1) Opening: establish the importance and scope of the topic. "
         "Cover the breath of work in this area and why it matters.\n"
         "(2) Coverage: state the number of papers reviewed, the year range, "
         "and the number of thematic clusters identified.\n"
         "(3) Taxonomy brief: briefly preview the main categories of methods/approaches "
         "covered in this survey.\n"
         "(4) Contributions: state what this survey provides (e.g., systematic review, "
         "unified taxonomy, comparative analysis, open challenges). Use a bullet list "
         "with 2-3 \\item entries.\n"
         "Use assertive language throughout. Cite key papers establishing importance.",
         []),

        ("Related Work", "sec:related",
         "Write 5-7 paragraphs organized by THEMATIC CLUSTERS.\n"
         "For each cluster: summarize 3-5 representative papers, show the evolution of ideas, "
         "and highlight the key differences between approaches.\n"
         "Organize by theme (e.g., by architecture, by task, by training paradigm), "
         "NOT chronologically.\n"
         "IMPORTANT: Every \\cite{} or \\citet{} key MUST be from the provided CITATION KEYS list. "
         "Do NOT invent citation keys.\n"
         "MUST-CITE ENFORCEMENT: If the context includes a 'MUST-CITE PAPERS' section, "
         "you MUST cite ALL of those papers in this section.\n"
         "FAIRNESS: discuss the STRONGEST and most influential works, not just peripheral ones.\n"
         "Use \\citet{key} when author is subject, \\citep{key} for parenthetical.",
         []),

        ("Taxonomy", "sec:taxonomy",
         "Write 6-8 paragraphs presenting a structured taxonomy of methods/approaches "
         "in this area.\n"
         "Structure:\n"
         "(1) Overview: describe the organizing principles of the taxonomy and list "
         "the top-level categories.\n"
         "(2-5) One \\subsection{} per major category. For each:\n"
         "  - Define the category and its distinguishing characteristics\n"
         "  - List sub-categories or variants with representative works (cite them)\n"
         "  - Explain the relationship between this category and others in the taxonomy\n"
         "  - Note strengths and weaknesses of approaches in this category\n"
         "(6) Comparative overview: a summary table or paragraph contrasting categories "
         "on key dimensions (e.g., scalability, data efficiency, generalization).\n"
         "Cite representative papers using \\citep{key}. Use the theme clusters from "
         "the context as your organizing structure.",
         []),

        ("Methods", "sec:method",
         "Write 6-8 paragraphs systematically analyzing the METHODOLOGICAL designs "
         "across reviewed papers.\n"
         "Structure:\n"
         "(1) Overview: frame the problem formally and describe the space of solution designs.\n"
         "(2-4) Detailed analysis of key technical components:\n"
         "  - For each major component, compare how different categories implement it\n"
         "  - Use numbered equations to illustrate key formulations where illuminating\n"
         "  - Reference equations in text: 'as defined in Eq.~\\eqref{eq:xxx}'\n"
         "  - Explain design trade-offs and why different schools of thought make different choices\n"
         "(5) Training and optimization: compare training objectives, optimizers, and "
         "regularization strategies across approaches.\n"
         "(6) Complexity and efficiency: discuss computational costs, FLOPs, or inference "
         "time comparisons where available.\n"
         "Use \\begin{align} for multi-line equations (NEVER eqnarray).\n"
         "Do NOT include \\begin{figure} blocks --- figures are inserted automatically.",
         []),

        ("Applications", "sec:applications",
         "Write 5-6 paragraphs covering where and how these methods have been applied.\n"
         "Structure:\n"
         "(1) Overview: summarize the range of application domains and tasks addressed "
         "by methods in this survey.\n"
         "(2-3) Domain-specific analysis: for each major application domain, "
         "discuss which methods have been applied, what datasets are used, "
         "and what performance levels are achieved (cite original papers).\n"
         "(4) Task taxonomy: classify the types of tasks addressed (e.g., classification, "
         "generation, prediction, optimization) and which methodological categories "
         "excel at each.\n"
         "(5) Emerging applications: note newly emerging or underexplored application "
         "areas identified across the literature.\n"
         "IMPORTANT: Do NOT fabricate numbers. Only cite results that appear in the "
         "provided evidence or citation data. Mark areas with insufficient data as 'limited "
         "empirical evaluation' rather than citing vague performance claims.",
         []),

        ("Challenges", "sec:challenges",
         "Write 5-6 paragraphs analyzing the key challenges and open problems.\n"
         "Structure:\n"
         "(1) Overview: synthesize the main difficulties identified across the literature "
         "into 3-5 high-level challenge categories.\n"
         "(2-4) One \\subsection{} per major challenge category. For each:\n"
         "  - Describe the challenge in technical terms\n"
         "  - Show how different methods attempt to address it (and where they fall short)\n"
         "  - Illustrate with specific examples from reviewed papers (cite them)\n"
         "  - Discuss trade-offs between addressing this challenge and others\n"
         "(5) Cross-cutting challenges: discuss challenges that span multiple categories "
         "or application domains.\n"
         "Draw directly from the key_challenges provided in the context. "
         "Do NOT invent challenges not supported by the literature.",
         []),

        ("Conclusion", "sec:conclusion",
         "Write 3-4 paragraphs:\n"
         "(1) Summarize the overall landscape --- main themes, methodological trends, "
         "and key findings from the survey.\n"
         "(2) Open challenges: 3-4 concrete, specific research problems that remain unsolved "
         "or underexplored (draw from the key_challenges in context).\n"
         "(3) Future directions: 2-3 promising research trajectories based on the "
         "identified gaps and future_directions from the literature.\n"
         "(4) Closing: briefly position the survey's contribution to the field.\n"
         "Do NOT introduce new results or citations here.",
         []),
    ],

    "long": [
        # standard sections +
        ("Introduction", "sec:intro",
         "Write 4-5 paragraphs:\n"
         "(1) Opening: establish the importance and scope of the topic. "
         "Cover the breath of work in this area and why it matters.\n"
         "(2) Coverage: state the number of papers reviewed, the year range, "
         "and the number of thematic clusters identified.\n"
         "(3) Taxonomy brief: briefly preview the main categories of methods/approaches "
         "covered in this survey.\n"
         "(4) Contributions: state what this survey provides (e.g., systematic review, "
         "unified taxonomy, comparative analysis, open challenges). Use a bullet list "
         "with 2-3 \\item entries.\n"
         "Use assertive language throughout. Cite key papers establishing importance.",
         []),

        ("Related Work", "sec:related",
         "Write 5-7 paragraphs organized by THEMATIC CLUSTERS.\n"
         "For each cluster: summarize 3-5 representative papers, show the evolution of ideas, "
         "and highlight the key differences between approaches.\n"
         "Organize by theme (e.g., by architecture, by task, by training paradigm), "
         "NOT chronologically.\n"
         "IMPORTANT: Every \\cite{} or \\citet{} key MUST be from the provided CITATION KEYS list. "
         "Do NOT invent citation keys.\n"
         "MUST-CITE ENFORCEMENT: If the context includes a 'MUST-CITE PAPERS' section, "
         "you MUST cite ALL of those papers in this section.\n"
         "FAIRNESS: discuss the STRONGEST and most influential works, not just peripheral ones.\n"
         "Use \\citet{key} when author is subject, \\citep{key} for parenthetical.",
         []),

        ("Taxonomy", "sec:taxonomy",
         "Write 6-8 paragraphs presenting a structured taxonomy of methods/approaches "
         "in this area.\n"
         "Structure:\n"
         "(1) Overview: describe the organizing principles of the taxonomy and list "
         "the top-level categories.\n"
         "(2-5) One \\subsection{} per major category. For each:\n"
         "  - Define the category and its distinguishing characteristics\n"
         "  - List sub-categories or variants with representative works (cite them)\n"
         "  - Explain the relationship between this category and others in the taxonomy\n"
         "  - Note strengths and weaknesses of approaches in this category\n"
         "(6) Comparative overview: a summary table or paragraph contrasting categories "
         "on key dimensions (e.g., scalability, data efficiency, generalization).\n"
         "Cite representative papers using \\citep{key}. Use the theme clusters from "
         "the context as your organizing structure.",
         []),

        ("Methods", "sec:method",
         "Write 6-8 paragraphs systematically analyzing the METHODOLOGICAL designs "
         "across reviewed papers.\n"
         "Structure:\n"
         "(1) Overview: frame the problem formally and describe the space of solution designs.\n"
         "(2-4) Detailed analysis of key technical components:\n"
         "  - For each major component, compare how different categories implement it\n"
         "  - Use numbered equations to illustrate key formulations where illuminating\n"
         "  - Reference equations in text: 'as defined in Eq.~\\eqref{eq:xxx}'\n"
         "  - Explain design trade-offs and why different schools of thought make different choices\n"
         "(5) Training and optimization: compare training objectives, optimizers, and "
         "regularization strategies across approaches.\n"
         "(6) Complexity and efficiency: discuss computational costs, FLOPs, or inference "
         "time comparisons where available.\n"
         "Use \\begin{align} for multi-line equations (NEVER eqnarray).\n"
         "Do NOT include \\begin{figure} blocks --- figures are inserted automatically.",
         []),

        ("Applications", "sec:applications",
         "Write 5-6 paragraphs covering where and how these methods have been applied.\n"
         "Structure:\n"
         "(1) Overview: summarize the range of application domains and tasks addressed "
         "by methods in this survey.\n"
         "(2-3) Domain-specific analysis: for each major application domain, "
         "discuss which methods have been applied, what datasets are used, "
         "and what performance levels are achieved (cite original papers).\n"
         "(4) Task taxonomy: classify the types of tasks addressed (e.g., classification, "
         "generation, prediction, optimization) and which methodological categories "
         "excel at each.\n"
         "(5) Emerging applications: note newly emerging or underexplored application "
         "areas identified across the literature.\n"
         "IMPORTANT: Do NOT fabricate numbers. Only cite results that appear in the "
         "provided evidence or citation data. Mark areas with insufficient data as 'limited "
         "empirical evaluation' rather than citing vague performance claims.",
         []),

        ("Challenges", "sec:challenges",
         "Write 5-6 paragraphs analyzing the key challenges and open problems.\n"
         "Structure:\n"
         "(1) Overview: synthesize the main difficulties identified across the literature "
         "into 3-5 high-level challenge categories.\n"
         "(2-4) One \\subsection{} per major challenge category. For each:\n"
         "  - Describe the challenge in technical terms\n"
         "  - Show how different methods attempt to address it (and where they fall short)\n"
         "  - Illustrate with specific examples from reviewed papers (cite them)\n"
         "  - Discuss trade-offs between addressing this challenge and others\n"
         "(5) Cross-cutting challenges: discuss challenges that span multiple categories "
         "or application domains.\n"
         "Draw directly from the key_challenges provided in the context. "
         "Do NOT invent challenges not supported by the literature.",
         []),

        ("Systematic Analysis", "sec:systematic",
         "Write 6-8 paragraphs providing a deep critical analysis of trends, assumptions, "
         "and methodological quality across the literature.\n"
         "Structure:\n"
         "(1) Overview: frame what this analysis covers and its goals.\n"
         "(2) Trend analysis: identify 2-3 major evolutionary trends in methodology "
         "(e.g., how problem formulation, model architecture, or evaluation practice "
         "has changed over time).\n"
         "(3) Evaluation practices: analyze how methods are evaluated across the literature "
         "- are benchmarks consistent? Are metrics meaningful? Are comparisons fair?\n"
         "  - Identify cases where evaluation is inconsistent or incomparable across papers\n"
         "  - Note gaps in empirical coverage (e.g., methods only tested on small-scale data)\n"
         "(4) Reproducibility: assess the reproducibility of reported results based on "
         "methodological detail provided in original papers.\n"
         "(5) Theoretical foundations: analyze the theoretical grounding of different "
         "approaches — which are well-motivated vs. purely empirical?\n"
         "(6) Synthesis: draw overall lessons about the state of the field.\n"
         "Support claims with specific citations from the literature. Be critical but fair.",
         []),

        ("Future Directions", "sec:future",
         "Write 4-5 paragraphs outlining promising future research directions.\n"
         "Structure:\n"
         "(1) Overview: frame future directions based on identified gaps and open challenges.\n"
         "(2-3) Specific directions (one \\subsection{} per direction): for each direction\n"
         "  - State the specific problem or opportunity\n"
         "  - Explain why it is promising or necessary given current limitations\n"
         "  - Connect it to specific trends or gaps identified in the Systematic Analysis\n"
         "  - Mention relevant prior work that motivates this direction (cite it)\n"
         "(4) High-impact opportunities: highlight 1-2 directions that could have "
         "broad impact across multiple application areas.\n"
         "Draw directly from the future_directions provided in the context. "
         "Be concrete and specific — vague directions are not useful.",
         []),

        ("Conclusion", "sec:conclusion",
         "Write 3-4 paragraphs:\n"
         "(1) Summarize the overall landscape --- main themes, methodological trends, "
         "and key findings from the survey.\n"
         "(2) Open challenges: 3-4 concrete, specific research problems that remain unsolved "
         "or underexplored (draw from the key_challenges in context).\n"
         "(3) Future directions: 2-3 promising research trajectories based on the "
         "identified gaps and future_directions from the literature.\n"
         "(4) Closing: briefly position the survey's contribution to the field.\n"
         "Do NOT introduce new results or citations here.",
         []),
    ],
}

# Map paper_mode -> size key for SURVEY_SECTIONS
PAPER_MODE_SECTIONS = {
    PaperMode.SURVEY_SHORT: SURVEY_SECTIONS["short"],
    PaperMode.SURVEY_STANDARD: SURVEY_SECTIONS["standard"],
    PaperMode.SURVEY_LONG: SURVEY_SECTIONS["long"],
}


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
