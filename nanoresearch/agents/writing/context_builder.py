"""Context builders: core context, cite keys, full context (legacy)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_PAPERS_FOR_CITATIONS = 50

from nanoresearch.skill_prompts import get_writing_system_prompt
from ._types import ContributionContract, ContributionClaim, GroundingPacket
from .context_sections import _ContextSectionsMixin


class _ContextBuilderMixin(_ContextSectionsMixin):
    """Mixin — context building methods."""

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

    def _build_full_context(
        self,
        ideation: dict,
        blueprint: dict,
        cite_keys: dict[int, str],
        experiment_results: dict | None = None,
        experiment_status: str = "pending",
        experiment_analysis: dict | None = None,
        experiment_summary: str = "",
        grounding: GroundingPacket | None = None,
    ) -> str:
        """Build full context string (legacy, used as fallback). Prefer _build_section_context()."""
        topic = ideation.get("topic", "")
        survey = ideation.get("survey_summary", "")
        gaps = ideation.get("gaps", [])

        hypothesis = ""
        for h in ideation.get("hypotheses", []):
            if not isinstance(h, dict):
                continue
            if h.get("hypothesis_id") == ideation.get("selected_hypothesis"):
                hypothesis = h.get("statement", "")
                break

        method = blueprint.get("proposed_method") or {}
        if not isinstance(method, dict):
            method = {}
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
        normalized_results = self._normalize_experiment_results(
            experiment_results or {},
            blueprint,
            experiment_analysis or {},
        )
        evidence_lines = self._build_evidence_context(ideation, blueprint)
        real_results_lines = self._build_real_results_context(
            normalized_results,
            experiment_status,
        )
        analysis_lines = self._build_experiment_analysis_context(
            experiment_analysis or {},
            experiment_summary,
            experiment_status,
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

Datasets: {json.dumps([d.get('name', '') for d in datasets if isinstance(d, dict)], ensure_ascii=False)}
Metrics: {json.dumps([m.get('name', '') for m in metrics if isinstance(m, dict)], ensure_ascii=False)}
Baselines: {json.dumps([b.get('name', '') for b in baselines if isinstance(b, dict)], ensure_ascii=False)}
Ablation Groups: {json.dumps([a.get('group_name', '') for a in ablations if isinstance(a, dict)], ensure_ascii=False)}

{evidence_lines}

{real_results_lines}

{analysis_lines}

{self._build_baseline_comparison_context(grounding)}

{self._build_grounding_status_context(grounding)}

=== CITATION KEYS (use ONLY these exact keys with \\cite{{}}) ===
{chr(10).join(ref_lines)}
=== END CITATION KEYS ===

=== CONTRIBUTION-EXPERIMENT ALIGNMENT ===
Each contribution in Introduction MUST map to experimental evidence:
- Method components: {json.dumps([c for c in method.get('key_components', [])], ensure_ascii=False)}
- Ablation groups: {json.dumps([a.get('group_name', '') for a in ablations if isinstance(a, dict)], ensure_ascii=False)}
Every component listed above should appear in the ablation table.
=== END ALIGNMENT ===

{self._build_must_cite_context(ideation, cite_keys)}{full_text_block}"""

    def _build_must_cite_context(self, ideation: dict, cite_keys: dict[int, str]) -> str:
        """Build a must-cite instruction block for writing prompts.

        Maps must-cite titles to their actual cite_keys so the LLM
        knows exactly which keys to use.
        """
        must_cites = ideation.get("must_cites", [])
        must_cite_matches = ideation.get("must_cite_matches", [])
        if not must_cites:
            return ""

        lines = ["=== MUST-CITE PAPERS (these MUST appear in the paper, especially Related Work) ==="]
        lines.append("The following papers are essential references identified from survey analysis.")
        lines.append("You MUST cite each of these at least once in the paper.\n")

        papers = ideation.get("papers", [])
        cited_keys = []
        for mc in must_cite_matches:
            title = mc.get("title", "")
            idx = mc.get("paper_index")
            matched = mc.get("matched", False)
            if matched and idx is not None and idx in cite_keys:
                key = cite_keys[idx]
                lines.append(f"  - \\cite{{{key}}}: {title}")
                cited_keys.append(key)
            else:
                lines.append(f"  - [no key available]: {title} (cite by searching if possible)")

        # If no matches but we have titles, still list them
        if not must_cite_matches:
            for mc_entry in must_cites[:15]:
                # must_cites can be list[str] or list[dict]
                title = mc_entry.get("title", "") if isinstance(mc_entry, dict) else str(mc_entry)
                if not title:
                    continue
                # Try to find by title in papers
                for i, p in enumerate(papers):
                    if not isinstance(p, dict):
                        continue
                    p_title = (p.get("title") or "").lower().strip()
                    mc_lower = title.lower().strip()
                    mc_words = set(mc_lower.split())
                    p_words = set(p_title.split())
                    if mc_words and p_words:
                        overlap = len(mc_words & p_words) / min(len(mc_words), len(p_words))
                        if overlap > 0.5 and i in cite_keys:
                            lines.append(f"  - \\cite{{{cite_keys[i]}}}: {title}")
                            cited_keys.append(cite_keys[i])
                            break
                else:
                    lines.append(f"  - [unmatched]: {title}")

        lines.append("\n=== END MUST-CITE ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # P0-A: Per-section context builder
    # ------------------------------------------------------------------
    # Instead of sending the same ~30-40K context to every section, each
    # section gets a tailored context containing only the blocks it needs.
    # This reduces token waste and lets each section focus on relevant info.
    # ------------------------------------------------------------------

    def _build_core_context(
        self,
        ideation: dict,
        blueprint: dict,
        cite_keys: dict[int, str],
    ) -> dict[str, Any]:
        """Extract shared primitives once; return a dict consumed by section builders.

        This is called ONCE in run(), and the resulting dict is passed to
        _build_section_context() for each section.
        """
        topic = ideation.get("topic", "")

        hypothesis = ""
        for h in ideation.get("hypotheses", []):
            if not isinstance(h, dict):
                continue
            if h.get("hypothesis_id") == ideation.get("selected_hypothesis"):
                hypothesis = h.get("statement", "")
                break

        method = blueprint.get("proposed_method") or {}
        if not isinstance(method, dict):
            method = {}
        datasets = blueprint.get("datasets", [])
        metrics = blueprint.get("metrics", [])
        baselines = blueprint.get("baselines", [])
        ablations = blueprint.get("ablation_groups", [])

        # Pre-build citation key reference lines (with author for disambiguation)
        papers = ideation.get("papers", [])
        ref_lines = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_CITATIONS]):
            if i in cite_keys and isinstance(p, dict):
                authors = p.get("authors", [])
                first_author = authors[0] if isinstance(authors, list) and authors else ""
                ref_lines.append(
                    f'  [{cite_keys[i]}] "{p.get("title", "")}" '
                    f'by {first_author} ({p.get("year", "")})'
                )

        # B2: Pre-build baseline→cite key mapping
        # Fuzzy-match baseline names from blueprint against paper titles from ideation
        baseline_cite_map = self._match_baselines_to_cite_keys(baselines, papers, cite_keys)

        # Pre-build full-text excerpt lines
        full_text_lines: list[str] = []
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

        return {
            "topic": topic,
            "hypothesis": hypothesis,
            "method": method,
            "method_str": json.dumps(method, indent=2, ensure_ascii=False)[:8000],
            "method_name": method.get("name", ""),
            "method_brief": method.get("description", "")[:500],
            "key_components": method.get("key_components", []),
            "survey": ideation.get("survey_summary", ""),
            "gaps": ideation.get("gaps", []),
            "datasets": datasets,
            "metrics": metrics,
            "baselines": baselines,
            "ablations": ablations,
            "dataset_names": json.dumps(
                [d.get("name", "") for d in datasets if isinstance(d, dict)],
                ensure_ascii=False,
            ),
            "metric_names": json.dumps(
                [m.get("name", "") for m in metrics if isinstance(m, dict)],
                ensure_ascii=False,
            ),
            "baseline_names": json.dumps(
                [b.get("name", "") for b in baselines if isinstance(b, dict)],
                ensure_ascii=False,
            ),
            "ablation_names": json.dumps(
                [a.get("group_name", "") for a in ablations if isinstance(a, dict)],
                ensure_ascii=False,
            ),
            "ref_lines": ref_lines,
            "full_text_lines": full_text_lines,
            "baseline_cite_map": baseline_cite_map,
            "ideation": ideation,
            "blueprint": blueprint,
            "cite_keys": cite_keys,
        }

    @staticmethod
    def _extract_method_impl_details(method_content: str) -> str:
        """Extract implementation details (epochs, GPU, lr, loss weights) from Method section.

        These are passed to Experiments section context so it doesn't contradict
        the numbers already committed in the Method section.
        """
        if not method_content or len(method_content) < 100:
            return ""
        # Extract key implementation patterns
        patterns = {
            "epochs": r'(?:train(?:ing)?|run)s?\s+(?:for\s+)?(\d+)\s+epochs?',
            "batch_size": r'batch\s+size\s+(?:is\s+|of\s+)?(\d+)',
            "learning_rate": r'learning\s+rate\s+(?:of\s+)?([0-9.]+\s*[×x]?\s*10\^?\{?-?\d+\}?|\d+[eE]-?\d+)',
            "gpu": r'((?:NVIDIA\s+)?(?:A100|V100|H100|RTX\s*\d+)[^.]*?)(?:\.|,|;)',
            "optimizer": r'(?:optimise?|train)\s+.*?(?:with|using)\s+(Adam[Ww]?|SGD|RMSProp)',
            "loss_weights": r'(?:λ|\\lambda)[_\{]?\d*\}?\s*=\s*([0-9.]+)',
            "dropout": r'(?:dropout\s+(?:with\s+|rate\s+|of\s+)?(?:p\s*=\s*)?|p\s*=\s*)([0-9.]+)',
        }
        found = []
        for name, pat in patterns.items():
            matches = re.findall(pat, method_content, re.IGNORECASE)
            if matches:
                found.append(f"  - {name}: {', '.join(str(m).strip() for m in matches)}")
        if not found:
            return ""
        return (
            "\n\n=== IMPLEMENTATION DETAILS COMMITTED IN METHOD SECTION ===\n"
            "The Method section already states the following implementation specifics.\n"
            "You MUST use these EXACT values — do NOT introduce different numbers.\n"
            + "\n".join(found)
            + "\n=== END COMMITTED DETAILS ==="
        )

    @staticmethod
    def _match_baselines_to_cite_keys(
        baselines: list[dict],
        papers: list[dict],
        cite_keys: dict[int, str],
    ) -> dict[str, str]:
        """Fuzzy-match baseline names from blueprint against paper titles.

        Returns a mapping: baseline_name → cite_key.
        This prevents the LLM from guessing which cite key belongs to which baseline.
        """
        if not baselines or not papers:
            return {}

        result: dict[str, str] = {}
        for b in baselines:
            if not isinstance(b, dict):
                continue
            bname = (b.get("name") or "").strip()
            if not bname:
                continue
            bname_lower = bname.lower()
            # Build tokens from baseline name (split on spaces, hyphens, underscores)
            bname_tokens = set(re.split(r'[\s\-_]+', bname_lower))
            bname_tokens.discard("")

            best_score = 0.0
            best_idx = -1

            for i, p in enumerate(papers[:MAX_PAPERS_FOR_CITATIONS]):
                if i not in cite_keys or not isinstance(p, dict):
                    continue
                title = (p.get("title") or "").lower()
                if not title:
                    continue

                # Score 1: exact baseline name in title (strongest signal)
                if bname_lower in title:
                    score = 1.0
                # Score 2: acronym or short name appears as word boundary in title
                elif len(bname_lower) >= 2 and re.search(
                    r'\b' + re.escape(bname_lower) + r'\b', title
                ):
                    score = 0.9
                # Score 3: token overlap (for multi-word baseline names)
                elif len(bname_tokens) >= 2:
                    title_tokens = set(re.split(r'[\s\-_:,]+', title))
                    overlap = bname_tokens & title_tokens
                    score = len(overlap) / min(len(bname_tokens), len(title_tokens))
                    score *= 0.7  # scale down token overlap
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score
                    best_idx = i

            # Only assign if we have a reasonable match
            if best_score >= 0.5 and best_idx >= 0:
                result[bname] = cite_keys[best_idx]

        return result

    # Methods moved to context_sections.py: _build_section_context,
    # _ctx_introduction, _ctx_related_work, _ctx_method, _ctx_experiments,
    # _ctx_conclusion, _ctx_default, _cite_keys_block, _baseline_cite_block,
    # _extract_contribution_contract, _build_evidence_context
