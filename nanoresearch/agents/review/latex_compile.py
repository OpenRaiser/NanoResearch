"""LaTeX compilation and citation resolution mixin."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from nanoresearch.latex import fixer as latex_fixer

from ._constants import MAX_LATEX_FIX_ATTEMPTS, _detect_bib_entry_type

logger = logging.getLogger(__name__)


class _LaTeXCompileMixin:
    """Mixin — citation resolution, sanitization, PDF compilation, LaTeX error fixing."""

    async def _resolve_missing_citations(
        self, latex: str, bib_path: Path
    ) -> tuple[str, bool]:
        """Find \\cite keys missing from bib and auto-fill them.

        Delegates to WritingAgent's resolver. Returns (latex, changed).
        """
        try:
            from nanoresearch.agents.writing import WritingAgent
            bib_content = bib_path.read_text(encoding="utf-8")

            # Reuse WritingAgent's citation regex
            cited: set[str] = set()
            for m in WritingAgent._CITE_KEY_RE.finditer(latex):
                for k in m.group(1).split(","):
                    k = k.strip()
                    if k:
                        cited.add(k)

            defined: set[str] = set()
            for m in WritingAgent._BIB_KEY_RE.finditer(bib_content):
                defined.add(m.group(1).strip())

            missing = cited - defined
            if not missing:
                return latex, False

            self.log(f"Resolving {len(missing)} missing citation(s) in review: {sorted(missing)}")

            # Create a temporary WritingAgent-like resolver
            new_entries: list[str] = []
            for key in sorted(missing):
                # Skip if already present (guards against duplicate calls)
                if re.search(r'@\w+\s*\{\s*' + re.escape(key) + r'\s*,', bib_content):
                    continue
                entry = await self._resolve_single_citation_key(key)
                new_entries.append(entry)

            if new_entries:
                bib_content = bib_content.rstrip() + "\n\n" + "\n".join(new_entries)
                # Atomic write: write to temp file then rename
                tmp_path = bib_path.with_suffix(".bib.tmp")
                tmp_path.write_text(bib_content, encoding="utf-8")
                tmp_path.replace(bib_path)
                self.log(f"Added {len(new_entries)} bib entries during review")

            return latex, bool(new_entries)
        except Exception as exc:
            logger.warning("Citation resolution failed: %s", exc)
            return latex, False

    async def _resolve_single_citation_key(self, key: str) -> str:
        """Resolve a missing citation key via S2 search or stub."""
        m = re.match(r"([a-z]+)(\d{4})([a-z]?)$", key, re.IGNORECASE)
        surname = m.group(1).capitalize() if m else key
        year = m.group(2) if m else ""
        query = f"{surname} {year}" if m else key

        try:
            from mcp_server.tools.openalex import search_openalex
            results = await search_openalex(query, max_results=5)
            best = None
            for r in results:
                r_year = str(r.get("year", ""))
                r_authors = " ".join(a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in r.get("authors", []))
                if year and r_year == year and surname.lower() in r_authors.lower():
                    best = r
                    break
            if not best:
                for r in results:
                    if year and str(r.get("year", "")) == year:
                        best = r
                        break
            if not best and results:
                best = results[0]

            if best:
                authors = best.get("authors", [])
                author_str = " and ".join(a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in authors[:5]) if authors else surname
                title = best.get("title", "Unknown")
                venue = best.get("venue", "") or "arXiv preprint"
                r_year = best.get("year", year or 2024)
                # BUG-26 fix: detect conference vs journal from venue name
                entry_type, venue_field = _detect_bib_entry_type(venue)
                return (
                    f"@{entry_type}{{{key},\n"
                    f"  title={{{title}}},\n"
                    f"  author={{{author_str}}},\n"
                    f"  year={{{r_year}}},\n"
                    f"  {venue_field}={{{venue}}},\n"
                    f"}}\n"
                )
        except Exception as exc:
            logger.debug("S2 search failed for '%s': %s", key, exc)

        # BUG-4 fix: instead of generating a fake stub with
        # title={Surname et al.}, generate a minimal @misc entry that
        # honestly marks itself as unresolved. The note field explains
        # the situation rather than masquerading as a real reference.
        return (
            f"@misc{{{key},\n"
            f"  author={{{surname}}},\n"
            f"  year={{{year or 2024}}},\n"
            f"  note={{Could not retrieve full metadata. "
            f"Please replace with the correct reference.}},\n"
            f"}}\n"
        )

    @staticmethod
    def _sanitize_revised_tex(tex: str) -> str:
        """Sanitize revised LaTeX using WritingAgent's sanitizer.

        Falls back to inline critical fixes if WritingAgent import fails.
        """
        try:
            from nanoresearch.agents.writing import WritingAgent
            tex = WritingAgent._sanitize_latex(tex)
        except Exception as exc:
            logger.warning("WritingAgent._sanitize_latex failed (%s), applying inline fixes", exc)
            # Inline fallback: at minimum fix the most critical issues
            import re as _re
            # [H]/[h]/[h!] -> [t!] (preserve * for column-spanning variants)
            tex = _re.sub(r'\\begin\{figure\}\s*\[[Hh]!?\]', r'\\begin{figure}[t!]', tex)
            tex = _re.sub(r'\\begin\{figure\*\}\s*\[[Hh]!?\]', r'\\begin{figure*}[t!]', tex)
            tex = _re.sub(r'\\begin\{table\}\s*\[[Hh]!?\]', r'\\begin{table}[t!]', tex)
            tex = _re.sub(r'\\begin\{table\*\}\s*\[[Hh]!?\]', r'\\begin{table*}[t!]', tex)
            # Unicode dashes
            tex = tex.replace("\u2014", "---").replace("\u2013", "--")
            tex = tex.replace("\u201c", "``").replace("\u201d", "''")
        return tex

    async def _compile_pdf_with_fix_loop(self, tex_path: Path) -> dict:
        """Compile LaTeX to PDF with automatic error-fix loop.

        If compilation fails, feed the error back to the LLM, apply the fix,
        and retry up to MAX_LATEX_FIX_ATTEMPTS times.

        Safety features (OpenClaw-inspired):
        - Backs up original tex before fix loop; restores on total failure
        - Post-write verification: re-reads file to confirm write succeeded
        """
        import shutil

        try:
            from mcp_server.tools.pdf_compile import compile_pdf
        except ImportError:
            return {"error": "PDF compiler module not available"}

        tex_path = Path(tex_path)

        # Backup original tex before any fix attempts
        backup_path = tex_path.with_suffix('.tex.bak')
        try:
            shutil.copy2(tex_path, backup_path)
        except OSError:
            pass  # non-fatal

        import hashlib
        result: dict = {}
        seen_error_sigs: set[str] = set()
        for attempt in range(MAX_LATEX_FIX_ATTEMPTS + 1):
            try:
                result = await compile_pdf(str(tex_path))
                if not isinstance(result, dict):
                    result = {"error": f"Unexpected compile_pdf return type: {type(result).__name__}"}
            except Exception as e:
                result = {"error": str(e)}

            if "pdf_path" in result:
                if attempt > 0:
                    self.log(f"PDF compiled successfully after {attempt} fix(es)")
                return result

            error_msg = result.get("error", "Unknown compilation error")

            # Detect repeated identical errors to avoid infinite loops
            error_sig = hashlib.md5(error_msg[-500:].encode()).hexdigest()[:8]
            if error_sig in seen_error_sigs:
                self.log("LaTeX fix loop: same error repeated, stopping")
                return result
            seen_error_sigs.add(error_sig)

            # Don't retry if the problem isn't fixable via LaTeX edits
            if "not found" in error_msg.lower() or "not available" in error_msg.lower():
                self.log("No LaTeX compiler available, skipping fix loop")
                return result

            if attempt >= MAX_LATEX_FIX_ATTEMPTS:
                self.log(f"PDF compilation failed after {MAX_LATEX_FIX_ATTEMPTS} fix attempts")
                # Restore backup on total failure
                if backup_path.exists():
                    try:
                        shutil.copy2(backup_path, tex_path)
                        self.log("  Restored original tex from backup")
                    except OSError:
                        pass
                return result

            # ── Check if error originates from .bbl (BibTeX) ──
            # Errors like "paper.bbl:64: Misplaced alignment tab character &"
            # can only be fixed by editing references.bib, not paper.tex.
            if '.bbl' in error_msg or 'alignment tab' in error_msg.lower():
                bib_path = tex_path.parent / "references.bib"
                if bib_path.exists():
                    try:
                        from nanoresearch.agents.writing import WritingAgent
                        bib_content = bib_path.read_text(encoding="utf-8")
                        fixed_bib = WritingAgent._sanitize_bibtex(bib_content)
                        if fixed_bib != bib_content:
                            bib_path.write_text(fixed_bib, encoding="utf-8")
                            self.log(f"  Fixed BibTeX file (attempt {attempt + 1})")
                            continue  # retry compilation with fixed .bib
                    except Exception as bib_exc:
                        self.log(f"  BibTeX fix failed: {bib_exc}")

            # Feed error to LLM and fix
            self.log(
                f"PDF compilation failed (attempt {attempt + 1}), "
                f"feeding error to LLM for fix..."
            )

            try:
                current_tex = tex_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Cannot read tex file for fixing: %s", exc)
                return result

            fixed_tex = await self._fix_latex_errors(current_tex, error_msg)

            if fixed_tex and fixed_tex != current_tex:
                fixed_tex = self._sanitize_revised_tex(fixed_tex)
                try:
                    tex_path.write_text(fixed_tex, encoding="utf-8")
                except OSError as exc:
                    logger.error("Cannot write fixed tex file: %s", exc)
                    return result
                # Post-write verification
                try:
                    verify = tex_path.read_text(encoding="utf-8")
                    if verify != fixed_tex:
                        self.log("  WARNING: post-write verification failed, reverting")
                        tex_path.write_text(current_tex, encoding="utf-8")
                        return result
                except OSError:
                    pass
                self.log(f"  Applied LLM fix (attempt {attempt + 1})")
            else:
                self.log("  LLM returned no changes, aborting fix loop")
                return result

        return result

    async def _fix_latex_errors(self, tex_source: str, error_log: str) -> str | None:
        """Fix LaTeX compilation errors using a 2-level strategy.

        Level 1: Deterministic fixes (no LLM) — via shared latex.fixer module.
        Level 2: Search-replace LLM fix — LLM outputs {"old":"...","new":"..."} pairs.

        Inspired by OpenClaw's edit tool: minimal LLM output, exact text matching.
        NEVER sends the full document to the LLM for rewriting.
        """
        error_log = latex_fixer.truncate_error_log(error_log)

        error_lines = latex_fixer.extract_error_lines(error_log)
        error_line = error_lines[0] if error_lines else None

        tex_lines = tex_source.split('\n')
        error_lower = error_log.lower()

        # ──────────── Level 1: Deterministic fixes ────────────
        fixed = latex_fixer.deterministic_fix(
            tex_source, error_log, error_line, log_fn=self.log,
        )
        if fixed and fixed != tex_source:
            self.log("  Level 1: deterministic fix applied")
            return fixed

        # Classify error for LLM hint
        targeted_hint = latex_fixer.classify_error(error_lower)

        # ──────────── Level 2: Search-replace LLM fix ────────────
        result = await self._search_replace_llm_fix(
            tex_source, tex_lines, error_line, error_log, targeted_hint
        )
        if result:
            return result

        self.log("  All fix levels exhausted, no fix found")
        return None

    def _try_deterministic_fix(
        self,
        tex_source: str,
        tex_lines: list[str],
        error_log: str,
        error_lower: str,
        error_line: int | None,
    ) -> str | None:
        """Level 1: Delegate to shared latex_fixer.deterministic_fix()."""
        return latex_fixer.deterministic_fix(
            tex_source, error_log, error_line, log_fn=self.log,
        )

    @staticmethod
    def _classify_error(error_lower: str) -> str:
        """Delegate to shared latex_fixer.classify_error()."""
        return latex_fixer.classify_error(error_lower)

    async def _search_replace_llm_fix(
        self,
        tex_source: str,
        tex_lines: list[str],
        error_line: int | None,
        error_log: str,
        targeted_hint: str,
    ) -> str | None:
        """Level 2: Search-replace fix via shared latex_fixer module."""
        win_start, win_end, numbered = latex_fixer.build_error_snippet(
            tex_lines, error_line,
        )
        prompt = latex_fixer.build_search_replace_prompt(
            error_log, error_line, targeted_hint,
            win_start, win_end, numbered,
        )

        revision_config = self.config.for_stage("revision")
        try:
            raw = await self.generate(
                latex_fixer.SEARCH_REPLACE_SYSTEM_PROMPT, prompt,
                stage_override=revision_config,
            )
            edits = latex_fixer.parse_edit_json(raw)
            if not edits:
                self.log("  Level 2: LLM returned no valid edits")
                return None
            return latex_fixer.apply_edits(
                tex_source, edits, log_fn=self.log,
                search_window=(win_start, win_end),
            )
        except Exception as exc:
            self.log(f"  Level 2 search-replace fix failed: {exc}")

        return None

    @staticmethod
    def _parse_edit_json(raw: str) -> list[dict]:
        """Delegate to shared latex_fixer.parse_edit_json()."""
        return latex_fixer.parse_edit_json(raw)
