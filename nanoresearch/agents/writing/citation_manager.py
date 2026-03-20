"""Citation management: BibTeX resolution, coverage checking, must-cite injection."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _CitationManagerMixin:
    """Mixin — citation management methods."""

    # Match \cite, \citet, \citep, \citeauthor, \citealp, \citealt, \Citet, etc.
    _CITE_KEY_RE = re.compile(r"\\[Cc]ite(?:t|p|author|year|alp|alt|num)?(?:\*)?(?:\[[^\]]*\])*\{([^}]+)\}")
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

        # 3. Try to resolve each missing key (skip if already added by a prior call)
        new_entries: list[str] = []
        for key in sorted(missing):
            # Double-check key isn't already in bibtex (guards against duplicate calls)
            if re.search(r'@\w+\s*\{\s*' + re.escape(key) + r'\s*,', bibtex):
                continue
            entry = await self._resolve_single_citation(key)
            new_entries.append(entry)

        # Append new entries to bibtex
        if new_entries:
            bibtex = bibtex.rstrip() + "\n\n" + "\n".join(new_entries)
            self.log(f"Added {len(new_entries)} bib entries")

        return bibtex

    async def _resolve_single_citation(self, key: str) -> str:
        """Resolve a single missing citation key to a bib entry.

        Parses the key pattern (e.g., 'gu2022', 'child2019b',
        'beltagy2020longformer') to extract author surname, year, and
        optional method/paper name, then searches OpenAlex.
        Falls back to a stub entry if search fails.
        """
        # Parse key: letters = surname, digits = year, optional trailing letters (method name)
        m = re.match(r"([a-z]+)(\d{4})([a-z]*)$", key, re.IGNORECASE)
        if m:
            surname = m.group(1).capitalize()
            year = m.group(2)
            method_hint = m.group(3)  # e.g. "longformer", "bigbird", "b"
            # Build search query: include method name if it's more than a
            # single disambiguator letter (e.g. "longformer" but not "b")
            if len(method_hint) > 1:
                query = f"{surname} {year} {method_hint}"
            else:
                query = f"{surname} {year}"
        else:
            # Unusual key format — use as-is for search
            surname = key
            year = ""
            method_hint = ""
            query = key

        # Try OpenAlex search (free, no API key application needed)
        try:
            from mcp_server.tools.openalex import search_openalex
            results = await search_openalex(query, max_results=5)
            # Find best match with prioritized matching
            best = None
            # Priority 1: year + author match + method hint in title
            for r in results:
                r_year = str(r.get("year", ""))
                r_authors = " ".join(r.get("authors", []))
                r_title = (r.get("title") or "").lower()
                if (year and r_year == year
                        and surname.lower() in r_authors.lower()):
                    if method_hint and len(method_hint) > 1:
                        if method_hint.lower() in r_title:
                            best = r
                            break
                    else:
                        best = r
                        break
            # Priority 2: year + author match (without method hint check)
            if not best:
                for r in results:
                    r_year = str(r.get("year", ""))
                    r_authors = " ".join(r.get("authors", []))
                    if year and r_year == year and surname.lower() in r_authors.lower():
                        best = r
                        break
            # Priority 3: year + method hint in title (author name may differ)
            if not best and method_hint and len(method_hint) > 1:
                for r in results:
                    r_year = str(r.get("year", ""))
                    r_title = (r.get("title") or "").lower()
                    if year and r_year == year and method_hint.lower() in r_title:
                        best = r
                        break
            # Priority 4: year match only (no blind first-result fallback)
            if not best and results:
                for r in results:
                    if year and str(r.get("year", "")) == year:
                        best = r
                        break

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
            logger.debug("OpenAlex search failed for citation key '%s': %s", key, exc)

        # BUG-22 fix: instead of a fake stub with title={Surname et al.},
        # generate an honest @misc entry without a fabricated title.
        self.log(f"  Unresolved entry for '{key}' (search failed)")
        return (
            f"@misc{{{key},\n"
            f"  author={{{surname}}},\n"
            f"  year={{{year or 2024}}},\n"
            f"  note={{Could not retrieve full metadata. "
            f"Please replace with the correct reference.}},\n"
            f"}}\n"
        )

    def _cleanup_unused_bibtex(self, latex: str, bibtex: str) -> str:
        """Remove BibTeX entries that are not cited anywhere in the LaTeX source.

        This prevents the .bib file from accumulating unused entries
        collected during ideation but never referenced in the final paper.
        """
        # 1. Collect all cited keys from LaTeX
        cited: set[str] = set()
        for m in self._CITE_KEY_RE.finditer(latex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cited.add(k)

        if not cited:
            return bibtex  # No citations at all — don't touch bib

        # 2. Parse bib entries and filter
        # Match @type{key, ... } blocks
        entry_re = re.compile(
            r'(@\w+\s*\{)\s*([^,\s]+)\s*,(.*?)(?=\n@|\Z)',
            re.DOTALL,
        )
        kept: list[str] = []
        removed_count = 0
        for m in entry_re.finditer(bibtex):
            key = m.group(2).strip()
            if key in cited:
                kept.append(m.group(0).rstrip())
            else:
                removed_count += 1

        if removed_count > 0:
            self.log(f"Removed {removed_count} unused BibTeX entries (kept {len(kept)})")
            return "\n\n".join(kept) + "\n"
        return bibtex

    # ---- citation coverage validation ----------------------------------------

    def _validate_citation_coverage(
        self, latex: str, ideation: dict, cite_keys: dict[int, str]
    ) -> dict:
        """Validate citation quality of the written paper.

        Checks:
        1. Total citation count
        2. Must-cite papers referenced
        3. Citation distribution across sections
        4. High-cited papers coverage
        """
        # 1. Extract all cited keys
        cited: set[str] = set()
        for m in self._CITE_KEY_RE.finditer(latex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    cited.add(k)

        total_citations = len(cited)

        # 2. Check must-cite coverage
        must_cites = ideation.get("must_cites", [])
        must_cite_matches = ideation.get("must_cite_matches", [])
        papers = ideation.get("papers", [])

        missing_must_cites: list[dict] = []
        cited_must_cites: list[str] = []
        for mc in must_cite_matches:
            idx = mc.get("paper_index")
            matched = mc.get("matched", False)
            title = mc.get("title", "")
            if matched and idx is not None and idx in cite_keys:
                key = cite_keys[idx]
                if key in cited:
                    cited_must_cites.append(title)
                else:
                    missing_must_cites.append({"title": title, "cite_key": key, "paper_index": idx})
            else:
                # Unmatched must-cite — can't check, but flag it
                missing_must_cites.append({"title": title, "cite_key": None, "paper_index": None})

        # 3. Citation distribution by section
        section_cites: dict[str, int] = {}
        # Split by \section
        section_pattern = re.compile(r'\\section\{((?:[^{}]|\{[^{}]*\})+)\}')
        parts = section_pattern.split(latex)
        for i in range(1, len(parts), 2):
            sec_name = parts[i] if i < len(parts) else "Unknown"
            sec_content = parts[i + 1] if i + 1 < len(parts) else ""
            sec_cited = set()
            for m in self._CITE_KEY_RE.finditer(sec_content):
                for k in m.group(1).split(","):
                    k = k.strip()
                    if k:
                        sec_cited.add(k)
            section_cites[sec_name] = len(sec_cited)

        # 4. High-cited papers coverage
        high_cited_keys = set()
        for i, p in enumerate(papers):
            if not isinstance(p, dict):
                continue
            if (p.get("citation_count", 0) or 0) >= 100 and i in cite_keys:
                high_cited_keys.add(cite_keys[i])

        cited_high = high_cited_keys & cited
        uncited_high = high_cited_keys - cited

        return {
            "total_citations": total_citations,
            "must_cite_total": len(must_cites),
            "must_cite_found": len(cited_must_cites),
            "missing_must_cites": missing_must_cites,
            "cited_must_cites": cited_must_cites,
            "section_cites": section_cites,
            "high_cited_total": len(high_cited_keys),
            "high_cited_referenced": len(cited_high),
            "uncited_high_keys": sorted(uncited_high),
        }

    def _inject_must_cites(
        self, latex: str, missing: list[dict],
        cite_keys: dict[int, str], ideation: dict,
    ) -> str:
        """Inject missing must-cite papers into the Related Work section.

        Adds a brief sentence for each must-cite paper that was identified
        but not referenced in the paper.
        """
        papers = ideation.get("papers", [])

        # Build injection lines
        inject_lines = []
        for mc in missing:
            key = mc.get("cite_key")
            idx = mc.get("paper_index")
            if not key:
                continue  # Can't inject without a cite key

            # Get paper title for context
            title = mc.get("title", "")
            if idx is not None and idx < len(papers):
                p = papers[idx]
                if isinstance(p, dict):
                    title = p.get("title", title)

            inject_lines.append(f"\\citet{{{key}}} is also relevant to this line of research.")

        if not inject_lines:
            return latex

        injection = "\n".join(inject_lines)

        # Find the Related Work section and inject before its end
        # Handle common heading variants (Related Works, Prior Work, etc.)
        rw_pattern = re.compile(
            r'(\\section\{(?:Related Works?|Prior Work|Literature Review'
            r'|Background(?:\s+and\s+Related\s+Work)?)\}.*?)'
            r'(\\section\{)',
            re.DOTALL | re.IGNORECASE,
        )
        m = rw_pattern.search(latex)
        if m:
            # Insert before next \section
            insert_pos = m.start(2)
            latex = (
                latex[:insert_pos]
                + "\n\n" + injection + "\n\n"
                + latex[insert_pos:]
            )
            self.log(f"Injected {len(inject_lines)} must-cite references into Related Work")
        else:
            # Fallback: try inserting after Introduction
            intro_pattern = re.compile(
                r'(\\section\{Introduction\}.*?)(\\section\{)',
                re.DOTALL
            )
            m = intro_pattern.search(latex)
            if m:
                insert_pos = m.start(2)
                latex = (
                    latex[:insert_pos]
                    + "\n\n" + injection + "\n\n"
                    + latex[insert_pos:]
                )
                self.log(f"Injected {len(inject_lines)} must-cite references after Introduction")

        return latex

    def _log_citation_report(self, report: dict) -> None:
        """Log a citation quality report."""
        self.log("=== Citation Quality Report ===")
        self.log(f"  Total unique citations: {report.get('total_citations', 0)}")
        self.log(f"  Must-cite coverage: {report.get('must_cite_found', 0)}/{report.get('must_cite_total', 0)}")
        self.log(f"  High-cited papers referenced: {report.get('high_cited_referenced', 0)}/{report.get('high_cited_total', 0)}")

        section_cites = report.get("section_cites", {})
        if section_cites:
            self.log("  Per-section citations:")
            for sec, count in section_cites.items():
                self.log(f"    {sec}: {count}")

        uncited_high = report.get("uncited_high_keys", [])
        if uncited_high:
            self.log(f"  Uncited high-cited papers: {', '.join(uncited_high[:10])}")

        missing = report.get("missing_must_cites", [])
        if missing:
            self.log(f"  Missing must-cites: {[m['title'][:50] for m in missing[:5]]}")

        # Save report to workspace
        self.save_log("citation_quality_report.json",
                      json.dumps(report, indent=2, ensure_ascii=False, default=str))
        self.log("=== End Citation Report ===")

