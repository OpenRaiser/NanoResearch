"""Section writing: title, abstract, section generation, tool builder."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_PAPERS_FOR_CITATIONS = 50

from nanoresearch.agents.tools import ToolDefinition, ToolRegistry
from nanoresearch.skill_prompts import get_writing_system_prompt, ABSTRACT_SYSTEM, TITLE_SYSTEM

ABSTRACT_SYSTEM_PROMPT = ABSTRACT_SYSTEM
TITLE_SYSTEM_PROMPT = TITLE_SYSTEM

class _SectionWriterMixin:
    """Mixin — section generation methods."""

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
            return ((await self.generate(ABSTRACT_SYSTEM_PROMPT, prompt)) or "").strip()
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
                    return content
            except Exception as e:
                logger.warning("Tool-augmented writing failed for %s, falling back: %s", heading, e)

        try:
            content = ((await self.generate(section_system, prompt)) or "").strip()
            # Defense-in-depth: LLMs sometimes emit \end{document} inside
            # section content. Strip it so it doesn't terminate the document
            # prematurely (causing all \cite{} to become (?)).
            content = re.sub(r'\\end\{document\}\s*', '', content).strip()
            return content
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
