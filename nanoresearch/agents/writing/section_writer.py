"""Section writing: title, abstract, section generation, tool builder."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_PAPERS_FOR_CITATIONS = 50

# Survey section prompts: maps section label -> writing instructions string.
# Used by run() to look up instructions when dispatching based on paper_mode.
SURVEY_SECTION_PROMPTS: dict[str, str] = {
    "sec:intro": (
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
        "Use assertive language throughout. Cite key papers establishing importance."
    ),
    "sec:related": (
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
        "Use \\citet{key} when author is subject, \\citep{key} for parenthetical."
    ),
    "sec:taxonomy": (
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
        "the context as your organizing structure."
    ),
    "sec:method": (
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
        "Do NOT include \\begin{figure} blocks --- figures are inserted automatically."
    ),
    "sec:applications": (
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
        "empirical evaluation' rather than citing vague performance claims."
    ),
    "sec:challenges": (
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
        "Do NOT invent challenges not supported by the literature."
    ),
    "sec:systematic": (
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
        "Support claims with specific citations from the literature. Be critical but fair."
    ),
    "sec:future": (
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
        "Be concrete and specific — vague directions are not useful."
    ),
    "sec:conclusion": (
        "Write 3-4 paragraphs:\n"
        "(1) Summarize the overall landscape --- main themes, methodological trends, "
        "and key findings from the survey.\n"
        "(2) Open challenges: 3-4 concrete, specific research problems that remain unsolved "
        "or underexplored (draw from the key_challenges in context).\n"
        "(3) Future directions: 2-3 promising research trajectories based on the "
        "identified gaps and future_directions from the literature.\n"
        "(4) Closing: briefly position the survey's contribution to the field.\n"
        "Do NOT introduce new results or citations here."
    ),
}

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.agents.writing.latex_assembler import _strip_llm_thinking
from nanoresearch.skill_prompts import get_writing_system_prompt, ABSTRACT_SYSTEM, TITLE_SYSTEM

ABSTRACT_SYSTEM_PROMPT = ABSTRACT_SYSTEM
TITLE_SYSTEM_PROMPT = TITLE_SYSTEM

_LEADING_FILLER_RE = re.compile(
    r"^(?:Now|Next|Then|Here|Let us|Let me|In this section|We now|"
    r"I have enough context[^.]*\.\s*|I will (?:now )?write[^.]*\.\s*|"
    r"I(?:'ll| will| can| shall) (?:now )?(?:write|draft|generate|compose|produce)[^.]*\.\s*|"
    r"Having described[^,]*,\s*|Having established[^,]*,\s*|"
    r"With this in mind[,;:\s]*|Building on this[,;:\s]*)[,;:\s]*",
    re.IGNORECASE,
)


class _SectionWriterMixin:
    """Mixin — section generation methods."""

    @staticmethod
    def _strip_leading_filler(text: str) -> str:
        """Remove LLM filler words/phrases at the start of generated section content."""
        # Only strip if the remainder still starts with a word (not a LaTeX command)
        m = _LEADING_FILLER_RE.match(text)
        if m:
            rest = text[m.end():]
            if rest and rest[0].isalpha():
                # Capitalize first letter of remaining text
                return rest[0].upper() + rest[1:]
        return text

    async def _build_writing_tools(self) -> ToolRegistry | None:
        """Build a ToolRegistry with search tools for writing.

        Returns None if no tools could be registered (missing deps).
        """
        registry = ToolRegistry()
        try:
            from mcp_server.tools.arxiv_search import search_arxiv
            from mcp_server.tools.openalex import search_openalex

            async def _search_papers(query: str, max_results: int = 5) -> list[dict]:
                results: list[dict] = []
                try:
                    results.extend(await search_arxiv(query, max_results=max_results))
                except Exception as exc:
                    logger.debug("arxiv search failed: %s", exc)
                try:
                    results.extend(await search_openalex(query, max_results=max_results))
                except Exception as exc:
                    logger.debug("openalex search failed: %s", exc)
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
            from mcp_server.tools.openalex import search_openalex as _search_openalex_detail
            registry.register(ToolDefinition(
                name="get_paper_details",
                description="Get detailed information about a paper by title or query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "Paper title or query to look up"},
                    },
                    "required": ["paper_id"],
                },
                handler=lambda paper_id: _search_openalex_detail(paper_id, max_results=1),
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

        # RAG tool: read full-text from a paper's PDF
        try:
            from mcp_server.tools.pdf_reader import download_and_extract

            async def _read_paper_pdf(pdf_url: str, section: str = "") -> dict:
                """Download a paper PDF and extract its full text or a specific section."""
                result = await download_and_extract(pdf_url, max_pages=20)
                if section:
                    # Return specific section if requested
                    sections = result.get("sections", {})
                    for name, content in sections.items():
                        if section.lower() in name.lower():
                            return {"section": name, "content": content[:5000]}
                    return {"error": f"Section '{section}' not found. Available: {list(sections.keys())}"}
                # Return method + experiment + abstract (most useful for writing)
                out: dict = {}
                if result.get("method_text"):
                    out["method"] = result["method_text"][:4000]
                if result.get("experiment_text"):
                    out["experiments"] = result["experiment_text"][:4000]
                sections = result.get("sections", {})
                if "Abstract" in sections:
                    out["abstract"] = sections["Abstract"][:1000]
                if not out:
                    out["full_text"] = result.get("full_text", "")[:6000]
                out["page_count"] = result.get("page_count", 0)
                out["sections_available"] = list(sections.keys())
                return out

            registry.register(ToolDefinition(
                name="read_paper_pdf",
                description=(
                    "Download and read a paper's PDF to get its full text. "
                    "Use this to get detailed method descriptions, experiment setups, "
                    "or specific results from a paper. Provide the PDF URL "
                    "(e.g., https://arxiv.org/pdf/2301.12345). "
                    "Optionally specify a section name to read only that section."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pdf_url": {
                            "type": "string",
                            "description": "URL to the PDF file (e.g., https://arxiv.org/pdf/XXXX.XXXXX)",
                        },
                        "section": {
                            "type": "string",
                            "description": "Optional section name to extract (e.g., 'Method', 'Experiments')",
                            "default": "",
                        },
                    },
                    "required": ["pdf_url"],
                },
                handler=_read_paper_pdf,
            ))
        except ImportError:
            pass

        return registry if len(registry) > 0 else None

    # ---- section generation -------------------------------------------------

    async def _generate_title(self, context: str) -> str:
        prompt = f"Based on the following research context, generate a paper title:\n\n{context}"
        try:
            return ((await self.generate(TITLE_SYSTEM_PROMPT, prompt)) or "").strip().strip('"')
        except Exception as e:
            logger.warning("Title generation failed, using fallback: %s", e)
            return "Untitled Research Paper"

    async def _generate_abstract(
        self, context: str, grounding: GroundingPacket | None = None,
    ) -> str:
        number_binding = ""
        if grounding and grounding.has_real_results and grounding.final_metrics:
            metric_strs = [f"{k}={v}" for k, v in list(grounding.final_metrics.items())[:5]]
            number_binding = (
                "\n\nIMPORTANT — RESULT NUMBERS IN ABSTRACT:\n"
                f"Real experiment metrics: {', '.join(metric_strs)}\n"
                "You MUST mention at least the primary metric in the abstract. "
                "Use the exact value from above. Do NOT fabricate different numbers."
            )
        elif grounding and not grounding.has_real_results:
            number_binding = (
                "\n\nIMPORTANT: No real experiment results are available. "
                "Do NOT mention any specific accuracy/F1/performance numbers in the abstract. "
                "Focus on the method and its design instead."
            )
        prompt = f"Based on the following research context, write the abstract:\n\n{context}{number_binding}"
        try:
            abstract = ((await self.generate(ABSTRACT_SYSTEM_PROMPT, prompt)) or "").strip()
            # Enforce word limit: if too long, ask LLM to compress (not hard-truncate)
            words = abstract.split()
            if len(words) > 260:
                logger.info("Abstract too long (%d words), asking LLM to compress to ~250", len(words))
                compress_prompt = (
                    f"The following abstract is {len(words)} words. "
                    f"Compress it to 200-250 words while preserving ALL key information: "
                    f"the problem, method name, all components, dataset names, and metric values. "
                    f"Do NOT drop the experimental results sentence. "
                    f"Output ONLY the compressed abstract text.\n\n{abstract}"
                )
                compressed = ((await self.generate(ABSTRACT_SYSTEM_PROMPT, compress_prompt)) or "").strip()
                if compressed and 100 < len(compressed.split()) <= 280:
                    abstract = compressed
                    logger.info("Abstract compressed to %d words", len(abstract.split()))
                else:
                    logger.warning("LLM compression returned bad length (%d words), keeping original",
                                   len((compressed or "").split()))
            return abstract
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

Output ONLY the LaTeX paragraphs for this section. Do not include \\section command.

FORMAT RULES:
- Do NOT wrap your output in Markdown code fences (```latex ... ``` or ``` ... ```).
- Do NOT insert \\begin{{figure}}...\\end{{figure}} environments; use \\ref{{fig:xxx}}
  to reference figures only — figures are inserted automatically by the pipeline.
- Use bare ~ for non-breaking space before \\ref (e.g. Figure~\\ref{{fig:arch}}),
  NOT \\~{{}} which renders as a tilde accent character in LaTeX."""

        # Use tool-augmented generation for key sections
        if self.config.should_use_writing_tools(heading):
            try:
                tools = await self._build_writing_tools()
                if tools is not None:
                    tool_prompt = (
                        prompt + "\n\nYou have access to search tools. "
                        "If you need to verify citations, find additional references, "
                        "or look up recent results, use the tools before writing. "
                        "Ground claims in retrieved evidence and actual experiment outputs."
                    )
                    content = (await self.generate_with_tools(
                        section_system, tool_prompt, tools,
                        max_tool_rounds=self.config.writing_tool_max_rounds,
                    ) or "").strip()
                    if not content:
                        self.log(f"  ReAct loop returned empty content for {heading}, retrying without tools")
                        content = ((await self.generate(section_system, prompt)) or "").strip()
                    # Defense-in-depth: strip stray \end{document}
                    content = re.sub(r'\\end\{document\}\s*', '', content).strip()
                    # Strip LLM filler at start of section
                    content = self._strip_leading_filler(content)
                    content = _strip_llm_thinking(content)
                    # Fix component count mismatches (e.g. "four components" but lists 5)
                    content = self._fix_component_count_mismatch(content)
                    return content
            except Exception as e:
                logger.warning("Tool-augmented writing failed for %s, falling back: %s", heading, e)

        try:
            content = ((await self.generate(section_system, prompt)) or "").strip()
            # Defense-in-depth: LLMs sometimes emit \end{document} inside
            # section content. Strip it so it doesn't terminate the document
            # prematurely (causing all \cite{} to become (?)).
            content = re.sub(r'\\end\{document\}\s*', '', content).strip()
            # Strip LLM filler at start of section
            content = self._strip_leading_filler(content)
            content = _strip_llm_thinking(content)
            # Fix component count mismatches (e.g. "four components" but lists 5)
            content = self._fix_component_count_mismatch(content)
            return content
        except Exception as e:
            logger.warning("Section generation failed for %s: %s", heading, e)
            return f"% Section generation failed: {heading}"

    # ---- post-processing: component count fix --------------------------------

    # Map number words to digits (and back)
    _NUM_WORDS = {
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    _DIGIT_WORDS = {v: k for k, v in _NUM_WORDS.items()}

    @classmethod
    def _fix_component_count_mismatch(cls, content: str) -> str:
        r"""Fix 'N components: (1)...(2)...' where N doesn't match actual count.

        Detects patterns like 'four key components' followed by enumerated items
        (either \item, (1)/(2)/(3), or \subsection) and corrects the number word.
        """
        # Pattern: "NUM_WORD key/main/core? components/modules/stages/parts"
        num_claim_re = re.compile(
            r'\b(two|three|four|five|six|seven|eight|nine|ten)\b'
            r'(\s+(?:key|main|core|primary|major|fundamental|critical|novel|distinct))?'
            r'\s+(components?|modules?|stages?|parts?|contributions?|elements?|blocks?)',
            re.IGNORECASE,
        )
        # Find all claims in the content
        for m in num_claim_re.finditer(content):
            claimed_word = m.group(1).lower()
            claimed_n = cls._NUM_WORDS.get(claimed_word, 0)
            if claimed_n == 0:
                continue

            # Count actual items after this claim
            after = content[m.end():]
            # Strategy 1: count \item entries (in itemize/enumerate)
            items_match = re.search(
                r'\\begin\{(?:itemize|enumerate)\}(.*?)\\end\{(?:itemize|enumerate)\}',
                after[:3000], re.DOTALL,
            )
            if items_match:
                actual_n = len(re.findall(r'\\item\b', items_match.group(1)))
            else:
                # Strategy 2: count (1), (2), (3) style enumeration
                # Only match sequential numbering starting from (1) to avoid
                # false positives from equation numbers, years, etc.
                paren_items = re.findall(r'\((\d+)\)', after[:2000])
                # Filter: keep only small sequential numbers (1-10 range)
                paren_small = [int(x) for x in paren_items if 1 <= int(x) <= 10]
                # Require (1) to be present (sequential enumeration starts at 1)
                if paren_small and 1 in paren_small:
                    actual_n = max(paren_small)
                else:
                    # Strategy 3: count \subsection entries
                    subsections = re.findall(r'\\subsection\{', after[:5000])
                    actual_n = len(subsections) if subsections else 0

            if actual_n > 0 and actual_n != claimed_n and actual_n in cls._DIGIT_WORDS:
                correct_word = cls._DIGIT_WORDS[actual_n]
                # Preserve original capitalization
                if m.group(1)[0].isupper():
                    correct_word = correct_word.capitalize()
                old_span = content[m.start():m.end()]
                new_span = old_span[:m.start(1) - m.start()] + correct_word + old_span[m.end(1) - m.start():]
                content = content[:m.start()] + new_span + content[m.end():]
                logger.info(
                    "Fixed component count mismatch: '%s' → '%s' (actual: %d items)",
                    claimed_word, correct_word, actual_n,
                )
                break  # Only fix the first mismatch per call to avoid cascading

        return content

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
