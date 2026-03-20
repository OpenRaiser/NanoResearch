"""Apply-revisions mixin — meta-refine, smart truncation, grounding, and revision application."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from nanoresearch.schemas.review import SectionReview

from ._constants import _ABSTRACT_PATTERN

logger = logging.getLogger(__name__)


class _ApplyRevisionsMixin:
    """Mixin — revision helpers: apply, meta-refine, truncation, grounding, guidance."""

    def _build_revision_grounding_block(self) -> str:
        """Build a grounding context block for revision prompts.

        Tells the revision LLM which numbers are real and must be preserved.
        """
        grounding = getattr(self, '_writing_grounding', {})
        analysis = getattr(self, '_experiment_analysis', {})
        status = getattr(self, '_experiment_status', 'pending')

        completeness = grounding.get('result_completeness', 'none') if grounding else 'none'
        has_real = grounding.get('has_real_results', False) if grounding else False

        if not has_real:
            return (
                "=== GROUNDING STATUS: NO REAL RESULTS ===\n"
                "This paper has NO real experiment results. During revision:\n"
                "- Do NOT add any experimental numbers\n"
                "- Do NOT fill empty table cells with fabricated values\n"
                "- Preserve any 'results pending' or 'future work' language\n"
                "=== END GROUNDING ==="
            )

        lines = [
            f"=== GROUNDING STATUS: {completeness.upper()} RESULTS ===",
            "This paper contains REAL experiment results. During revision:",
            "- PRESERVE all numbers in tables (they come from real experiments)",
            "- Do NOT round, adjust, or 'improve' any metric values",
            "- Do NOT add new baseline numbers that weren't in the original",
        ]

        # Add real metric values for reference
        final_metrics = analysis.get('final_metrics', {}) if isinstance(analysis, dict) else {}
        if isinstance(final_metrics, dict) and final_metrics:
            lines.append("Real metrics (for reference, do NOT modify):")
            for k, v in list(final_metrics.items())[:10]:
                lines.append(f"  {k} = {v}")

        lines.append("=== END GROUNDING ===")
        return "\n".join(lines)

    async def _meta_refine_revision(
        self,
        paper_tex: str,
        old_review: SectionReview,
        new_review: SectionReview,
        failed_revision: str,
        ideation_output: dict,
    ) -> str | None:
        """Diagnose why a revision degraded quality, then retry with extra constraints.

        This is the runtime prompt self-optimization loop:
        1. LLM analyzes old review vs new review to find what the revision broke
        2. Generates extra constraints to prevent the same mistakes
        3. Retries revision with the augmented prompt
        """
        section_name = old_review.section
        old_strengths = old_review.strengths

        # Guard: if failed_revision is empty, nothing to diagnose
        if not failed_revision or not failed_revision.strip():
            logger.warning("Meta-refine: empty failed revision for '%s', skipping", section_name)
            return None

        # Step 1: Diagnose with review-stage LLM (cheap + fast)
        diagnosis_prompt = f"""A revision of the "{section_name}" section made the paper WORSE.

BEFORE revision (score {old_review.score}/10):
- Strengths: {json.dumps(old_strengths, ensure_ascii=False)}
- Issues: {json.dumps(old_review.issues[:5], ensure_ascii=False)}

AFTER revision (score {new_review.score}/10):
- New issues: {json.dumps(new_review.issues[:5], ensure_ascii=False)}

Failed revision text (first 3000 chars):
{failed_revision[:3000]}

Analyze what the revision did WRONG. Common mistakes:
- Removed specific data/numbers and replaced with vague language
- Deleted citations or technical details
- Introduced generic filler ("it is worth noting", "interestingly")
- Broke LaTeX formatting or removed figures/tables
- Over-simplified technical content
- Changed notation inconsistently

Return JSON:
{{
    "diagnosis": "What specifically went wrong with this revision",
    "extra_constraints": [
        "Constraint 1: Do NOT ...",
        "Constraint 2: MUST keep ...",
        "Constraint 3: ..."
    ]
}}"""

        review_config = self.config.for_stage("review")
        try:
            result = await self.generate_json(
                "You are a meta-reviewer analyzing why a paper revision failed. "
                "Be specific about what went wrong and provide actionable constraints.",
                diagnosis_prompt,
                stage_override=review_config,
            )
        except Exception as exc:
            logger.warning("Meta-refine diagnosis failed for '%s': %s", section_name, exc)
            return None

        if isinstance(result, list):
            result = {"extra_constraints": result}
        diagnosis = result.get("diagnosis", "unknown")
        extra_constraints = result.get("extra_constraints", [])
        self.log(f"  '{section_name}' diagnosis: {diagnosis[:120]}")

        if not extra_constraints:
            return None

        # Step 2: Retry revision with extra constraints appended
        constraints_block = "\n".join(f"- {c}" for c in extra_constraints[:5])
        augmented_review = SectionReview(
            section=old_review.section,
            score=old_review.score,
            issues=old_review.issues,
            suggestions=old_review.suggestions + [
                f"[META-REFINE] Previous revision failed because: {diagnosis}. "
                f"Extra constraints:\n{constraints_block}"
            ],
        )
        # Carry over strengths and justification
        augmented_review.strengths = old_strengths
        augmented_review.score_justification = old_review.score_justification

        self.log(f"  '{section_name}' retrying revision with {len(extra_constraints)} extra constraints")
        try:
            return await self._revise_section(
                paper_tex, augmented_review, ideation_output,
            )
        except Exception as exc:
            logger.warning("Meta-refine retry failed for '%s': %s", section_name, exc)
            return None

    @staticmethod
    def _smart_truncate(text: str, max_chars: int = 20000) -> str:
        """Truncate paper text preserving high-priority sections.

        Instead of naive head/tail split (which often drops Method/Experiment),
        keep preamble + sections by priority order.
        """
        if len(text) <= max_chars:
            return text

        # Find section boundaries
        # BUG-14 fix: support one level of nested braces in section titles
        # e.g. \section{Method for \textbf{Hard} Cases}
        section_starts = [
            (m.start(), m.group(1))
            for m in re.finditer(r'\\section\{((?:[^{}]|\{[^{}]*\})+)\}', text)
        ]
        if not section_starts:
            # No sections found, fall back to head/tail
            return text[:12000] + "\n\n[...truncated...]\n\n" + text[-8000:]

        # Always keep preamble (title, abstract, etc.) up to first \section
        preamble = text[:section_starts[0][0]]
        remaining = max_chars - len(preamble)
        if remaining <= 0:
            return preamble[:max_chars]

        # Collect sections by priority, record their original position for ordering
        priority = ["introduction", "method", "experiment",
                     "result", "conclusion", "related"]
        # (original_start_pos, section_text) — for document-order output
        kept: list[tuple[int, str]] = []
        added_indices: set[int] = set()

        for pname in priority:
            if remaining <= 0:
                break
            for i, (start, title) in enumerate(section_starts):
                if i in added_indices:
                    continue
                if pname in title.lower():
                    end = (section_starts[i + 1][0]
                           if i + 1 < len(section_starts)
                           else len(text))
                    content = text[start:end]
                    if len(content) <= remaining:
                        kept.append((start, content))
                        remaining -= len(content)
                        added_indices.add(i)
                    elif remaining > 500:
                        kept.append((start,
                                     content[:remaining] + "\n[...truncated...]"))
                        remaining = 0
                        added_indices.add(i)
                    break

        if not kept:
            return text[:max_chars]

        # Sort by original document position so LLM sees correct order
        kept.sort(key=lambda x: x[0])
        return preamble + "\n\n".join(s for _, s in kept)

    @staticmethod
    def _get_section_revision_guidance(section_name: str) -> str:
        """Return section-specific revision guidance for top-tier venue standards."""
        section_lower = section_name.lower()
        if "related" in section_lower or "prior" in section_lower or "background" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Related Work):\n"
                "- Organize by THEME/APPROACH, not chronologically\n"
                "- Each paragraph should cover one research direction with 3-5 citations\n"
                "- For each cited work, briefly state its approach AND its limitation\n"
                "- End each paragraph by explaining how your work addresses these limitations\n"
                "- The final paragraph should clearly differentiate your approach from ALL prior work\n"
                "- Use \\citet{} when the author is the subject, \\citep{} for parenthetical\n"
                "- Minimum 15 citations total for a strong Related Work section\n"
                "- Cover at minimum: (1) the main task, (2) the key technique you use, "
                "(3) closely related approaches you improve upon"
            )
        elif "intro" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Introduction):\n"
                "- Start with the broader problem and its importance (1 paragraph)\n"
                "- Describe the specific challenge your work addresses (1 paragraph)\n"
                "- Briefly outline your approach and key contributions (1 paragraph)\n"
                "- List 3-4 concrete contributions as a bulleted list\n"
                "- Include at least 5-8 citations to establish context"
            )
        elif "method" in section_lower or "approach" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Method):\n"
                "- Include a formal problem definition with mathematical notation\n"
                "- Describe each component with equations\n"
                "- Use \\begin{equation} for key formulas, number them for reference\n"
                "- Explain the intuition behind each design choice\n"
                "- Reference the architecture figure if available (Figure~\\ref{fig:framework})"
            )
        elif "experiment" in section_lower or "result" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Experiments):\n"
                "- Describe datasets with statistics (size, splits, metrics)\n"
                "- List all baselines with brief descriptions and citations\n"
                "- Present main results in a table (Table~\\ref{tab:main_results})\n"
                "- Include ablation study results\n"
                "- Discuss why your method outperforms baselines\n"
                "- ALL numeric values in tables must be concrete, never '--' or 'N/A'"
            )
        elif "conclusion" in section_lower:
            return (
                "SECTION-SPECIFIC GUIDANCE (Conclusion):\n"
                "- Summarize the key contributions (2-3 sentences)\n"
                "- State the main experimental findings with numbers\n"
                "- Discuss limitations honestly\n"
                "- Suggest 2-3 specific future work directions"
            )
        return ""

    @staticmethod
    def _apply_revisions(paper_tex: str, revised_sections: dict[str, str]) -> str:
        """Apply revised sections back into the paper LaTeX source.

        Handles \\section{}, \\subsection{}, and \\subsubsection{}.
        Searches only after \\begin{document} to avoid matching ToC entries.
        Preserves \\begin{figure}...\\end{figure} blocks from the original
        when the revised content doesn't include them.
        """
        result = paper_tex

        # Find body start to avoid ToC matches
        body_marker = r"\begin{document}"
        body_start = result.find(body_marker)
        if body_start >= 0:
            body_start += len(body_marker)
        else:
            body_start = 0

        for heading, new_content in revised_sections.items():
            # BUG-41 fix: strip stray \end{document} from LLM-revised
            # section content — it would terminate the document prematurely
            # and cause all \cite{} after it to become (?).
            new_content = re.sub(r'\\end\{document\}\s*', '', new_content)
            # Recalculate body_start each iteration because prior replacements
            # shift all character offsets in `result`
            body_start = result.find(body_marker)
            if body_start >= 0:
                body_start += len(body_marker)
            else:
                body_start = 0

            # Special case: Abstract lives in \begin{abstract}...\end{abstract}
            if heading == "Abstract":
                abs_m = _ABSTRACT_PATTERN.search(result)
                if abs_m:
                    result = (
                        result[:abs_m.start(2)]
                        + "\n" + new_content.strip() + "\n"
                        + result[abs_m.end(2):]
                    )
                    logger.info("Applied abstract revision")
                else:
                    logger.warning("Cannot find abstract in paper — revision discarded")
                continue

            # Determine the level of the section being revised
            # by finding its command in the document
            esc_heading = re.escape(heading)
            heading_match = re.search(
                r"\\((?:sub){0,2})section\*?\{" + esc_heading + r"\}",
                result[body_start:],
            )
            if not heading_match:
                logger.warning(
                    "Cannot find section '%s' in paper — revision discarded", heading
                )
                continue

            section_level = heading_match.group(1).count("sub")

            # For top-level sections (\section{}), match everything up to the
            # next \section{} (same level), \end{document}, or \bibliography.
            # This includes all subsections within it.
            # BUG-35 fix: lookahead also matches \begin{thebibliography}
            # (inline bibliography style) in addition to \bibliography{}.
            _BIB_LA = r"\\bibliography(?:style)?\{|\\begin\{thebibliography\}"
            if section_level == 0:
                pattern = (
                    r"(\\section\*?\{" + esc_heading + r"\})"
                    r"(.*?)"
                    r"(?=\\section\*?\{|\\end\{document\}|" + _BIB_LA + r")"
                )
            elif section_level == 1:
                # BUG-7 fix: for \subsection{}, match up to the next
                # \section{} or \subsection{} — NOT \subsubsection{}.
                # This ensures subsubsections within this subsection are
                # included in the replacement range, preventing duplication.
                pattern = (
                    r"(\\subsection\*?\{" + esc_heading + r"\})"
                    r"(.*?)"
                    r"(?=\\(?:sub)?section\*?\{|\\end\{document\}|" + _BIB_LA + r")"
                )
            else:
                # For \subsubsection{}, match up to next section at any level
                pattern = (
                    r"(\\subsubsection\*?\{" + esc_heading + r"\})"
                    r"(.*?)"
                    r"(?=\\(?:sub){0,2}section\*?\{|\\end\{document\}|" + _BIB_LA + r")"
                )
            match = re.search(pattern, result[body_start:], re.DOTALL)
            if not match:
                logger.warning(
                    "Cannot find section '%s' in paper — revision discarded", heading
                )
                continue

            old_content = match.group(2)
            abs_start = body_start + match.start(2)
            abs_end = body_start + match.end(2)

            # Preserve figure/table environments from old content
            # that may have been dropped by the revision LLM
            old_figures = re.findall(
                r'(\\begin\{figure\*?\}.*?\\end\{figure\*?\})',
                old_content, re.DOTALL,
            )
            old_tables = re.findall(
                r'(\\begin\{table\*?\}.*?\\end\{table\*?\})',
                old_content, re.DOTALL,
            )
            preserved = []
            for fig_block in old_figures:
                # Only preserve if new content doesn't already have this figure
                label_match = re.search(r'\\label\{([^}]+)\}', fig_block)
                if label_match:
                    label = label_match.group(1)
                    # Exact label match (not substring) to avoid fig:method matching fig:method_base
                    if not re.search(r'\\label\{' + re.escape(label) + r'\}', new_content):
                        preserved.append(fig_block)
                else:
                    # No label — check if specific includegraphics file is already present
                    file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', fig_block)
                    if file_m:
                        fname = re.escape(file_m.group(1))
                        if not re.search(r'\\includegraphics(?:\[[^\]]*\])?\{' + fname + r'\}', new_content):
                            preserved.append(fig_block)
                    elif 'includegraphics' not in new_content:
                        preserved.append(fig_block)
            for tbl_block in old_tables:
                label_match = re.search(r'\\label\{([^}]+)\}', tbl_block)
                if label_match:
                    label = label_match.group(1)
                    # Exact label match (not substring)
                    if not re.search(r'\\label\{' + re.escape(label) + r'\}', new_content):
                        preserved.append(tbl_block)
                elif 'caption' in tbl_block and '\\begin{tabular}' not in new_content:
                    preserved.append(tbl_block)

            suffix = ""
            if preserved:
                # Ensure preserved figures/tables are injected BEFORE any
                # bibliography commands in new_content (not after them).
                bib_anchor_re = re.compile(
                    r'\\bibliography(?:style)?\{|\\begin\{thebibliography\}'
                )
                bib_in_new = bib_anchor_re.search(new_content)
                if bib_in_new:
                    # Insert preserved blocks before bibliography
                    inject_pos = bib_in_new.start()
                    new_content = (
                        new_content[:inject_pos]
                        + "\n\n" + "\n\n".join(preserved) + "\n\n"
                        + new_content[inject_pos:]
                    )
                else:
                    suffix = "\n\n" + "\n\n".join(preserved)

            result = (
                result[:abs_start]
                + "\n" + new_content + suffix + "\n\n"
                + result[abs_end:]
            )
        return result
