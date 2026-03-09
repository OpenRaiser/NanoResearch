"""Context builders: per-section context, citation keys, contribution contract."""
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

class _ContextBuilderMixin:
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
                        overlap = len(mc_words & p_words) / max(len(mc_words), len(p_words))
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

        # Pre-build citation key reference lines
        papers = ideation.get("papers", [])
        ref_lines = []
        for i, p in enumerate(papers[:MAX_PAPERS_FOR_CITATIONS]):
            if i in cite_keys and isinstance(p, dict):
                ref_lines.append(
                    f"  [{cite_keys[i]}] {p.get('title', '')} ({p.get('year', '')})"
                )

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
            "ideation": ideation,
            "blueprint": blueprint,
            "cite_keys": cite_keys,
        }

    def _build_section_context(
        self,
        section_label: str,
        core: dict[str, Any],
        grounding: GroundingPacket | None = None,
        experiment_results: dict | None = None,
        experiment_status: str = "pending",
        experiment_analysis: dict | None = None,
        experiment_summary: str = "",
    ) -> str:
        """Build a tailored context string for a specific section.

        Each section gets only the context blocks it actually needs,
        reducing prompt size from ~30-40K to ~8-15K per section.
        """
        dispatcher = {
            "sec:intro": self._ctx_introduction,
            "sec:related": self._ctx_related_work,
            "sec:method": self._ctx_method,
            "sec:experiments": self._ctx_experiments,
            "sec:conclusion": self._ctx_conclusion,
        }
        builder = dispatcher.get(section_label, self._ctx_default)
        return builder(
            core,
            grounding=grounding,
            experiment_results=experiment_results,
            experiment_status=experiment_status,
            experiment_analysis=experiment_analysis,
            experiment_summary=experiment_summary,
        )

    # --- Section-specific context builders ---

    def _ctx_introduction(
        self,
        core: dict[str, Any],
        grounding: GroundingPacket | None = None,
        **_kwargs: Any,
    ) -> str:
        """Introduction: topic, gaps, hypothesis, method brief, cite keys."""
        gaps_str = json.dumps(core["gaps"], indent=2, ensure_ascii=False)[:3000]
        survey_brief = core["survey"][:2000] if core["survey"] else ""

        parts = [
            f"Topic: {core['topic']}",
            "",
            f"Literature Survey (brief):\n{survey_brief}" if survey_brief else "",
            "",
            f"Research Gaps:\n{gaps_str}",
            "",
            f"Main Hypothesis: {core['hypothesis']}",
            "",
            f"Proposed Method: {core['method_name']}",
            f"Method Overview: {core['method_brief']}",
            f"Key Components: {json.dumps(core['key_components'], ensure_ascii=False)}",
            "",
            f"Datasets: {core['dataset_names']}",
            f"Metrics: {core['metric_names']}",
            "",
            self._cite_keys_block(core["ref_lines"]),
        ]
        return "\n".join(p for p in parts if p is not None)

    def _ctx_related_work(
        self,
        core: dict[str, Any],
        **_kwargs: Any,
    ) -> str:
        """Related Work: full survey, gaps, evidence, cite keys, must-cites, full-text."""
        survey_str = core["survey"][:6000] if core["survey"] else ""
        gaps_str = json.dumps(core["gaps"], indent=2, ensure_ascii=False)[:5000]
        evidence_lines = self._build_evidence_context(core["ideation"], core["blueprint"])

        full_text_block = ""
        if core["full_text_lines"]:
            full_text_block = (
                "\n\n=== FULL-TEXT EXCERPTS FROM KEY PAPERS ===\n"
                + "\n".join(core["full_text_lines"])
                + "\n=== END FULL-TEXT EXCERPTS ==="
            )

        parts = [
            f"Topic: {core['topic']}",
            "",
            f"Literature Survey:\n{survey_str}",
            "",
            f"Research Gaps:\n{gaps_str}",
            "",
            f"Proposed Method: {core['method_name']}",
            "",
            evidence_lines,
            "",
            self._cite_keys_block(core["ref_lines"]),
            "",
            self._build_must_cite_context(core["ideation"], core["cite_keys"]),
            full_text_block,
        ]
        return "\n".join(p for p in parts if p is not None)

    def _ctx_method(
        self,
        core: dict[str, Any],
        **_kwargs: Any,
    ) -> str:
        """Method: full method detail, hypothesis, ablations, cite keys, full-text."""
        full_text_block = ""
        if core["full_text_lines"]:
            full_text_block = (
                "\n\n=== FULL-TEXT EXCERPTS FROM KEY PAPERS ===\n"
                + "\n".join(core["full_text_lines"])
                + "\n=== END FULL-TEXT EXCERPTS ==="
            )

        parts = [
            f"Topic: {core['topic']}",
            "",
            f"Main Hypothesis: {core['hypothesis']}",
            "",
            f"Proposed Method:\n{core['method_str']}",
            "",
            f"Datasets: {core['dataset_names']}",
            f"Metrics: {core['metric_names']}",
            f"Ablation Groups: {core['ablation_names']}",
            "",
            self._cite_keys_block(core["ref_lines"]),
            full_text_block,
        ]
        return "\n".join(p for p in parts if p is not None)

    def _ctx_experiments(
        self,
        core: dict[str, Any],
        grounding: GroundingPacket | None = None,
        experiment_results: dict | None = None,
        experiment_status: str = "pending",
        experiment_analysis: dict | None = None,
        experiment_summary: str = "",
        **_kwargs: Any,
    ) -> str:
        """Experiments: method brief, datasets/metrics/baselines full, results, analysis, grounding."""
        normalized_results = self._normalize_experiment_results(
            experiment_results or {},
            core["blueprint"],
            experiment_analysis or {},
        )
        evidence_lines = self._build_evidence_context(core["ideation"], core["blueprint"])
        real_results_lines = self._build_real_results_context(
            normalized_results, experiment_status,
        )
        analysis_lines = self._build_experiment_analysis_context(
            experiment_analysis or {}, experiment_summary, experiment_status,
        )

        method = core["method"]
        ablations = core["ablations"]

        parts = [
            f"Topic: {core['topic']}",
            "",
            f"Main Hypothesis: {core['hypothesis']}",
            "",
            f"Proposed Method: {core['method_name']}",
            f"Method Overview: {core['method_brief']}",
            "",
            f"Datasets: {json.dumps(core['datasets'], indent=2, ensure_ascii=False)[:4000]}",
            f"Metrics: {json.dumps(core['metrics'], indent=2, ensure_ascii=False)[:2000]}",
            f"Baselines: {json.dumps(core['baselines'], indent=2, ensure_ascii=False)[:3000]}",
            f"Ablation Groups: {json.dumps(ablations, indent=2, ensure_ascii=False)[:2000]}",
            "",
            evidence_lines,
            "",
            real_results_lines,
            "",
            analysis_lines,
            "",
            self._build_baseline_comparison_context(grounding),
            "",
            self._build_grounding_status_context(grounding),
            "",
            self._cite_keys_block(core["ref_lines"]),
            "",
            "=== CONTRIBUTION-EXPERIMENT ALIGNMENT ===",
            "Each contribution in Introduction MUST map to experimental evidence:",
            f"- Method components: {json.dumps([c for c in method.get('key_components', [])], ensure_ascii=False)}",
            f"- Ablation groups: {json.dumps([a.get('group_name', '') for a in ablations if isinstance(a, dict)], ensure_ascii=False)}",
            "Every component listed above should appear in the ablation table.",
            "=== END ALIGNMENT ===",
        ]
        return "\n".join(p for p in parts if p is not None)

    def _ctx_conclusion(
        self,
        core: dict[str, Any],
        grounding: GroundingPacket | None = None,
        experiment_results: dict | None = None,
        experiment_status: str = "pending",
        experiment_analysis: dict | None = None,
        experiment_summary: str = "",
        **_kwargs: Any,
    ) -> str:
        """Conclusion: topic, hypothesis, method brief, results summary, grounding."""
        normalized_results = self._normalize_experiment_results(
            experiment_results or {},
            core["blueprint"],
            experiment_analysis or {},
        )
        real_results_lines = self._build_real_results_context(
            normalized_results, experiment_status,
        )
        analysis_lines = self._build_experiment_analysis_context(
            experiment_analysis or {}, experiment_summary, experiment_status,
        )

        parts = [
            f"Topic: {core['topic']}",
            "",
            f"Main Hypothesis: {core['hypothesis']}",
            "",
            f"Proposed Method: {core['method_name']}",
            f"Method Overview: {core['method_brief']}",
            f"Key Components: {json.dumps(core['key_components'], ensure_ascii=False)}",
            "",
            f"Datasets: {core['dataset_names']}",
            f"Metrics: {core['metric_names']}",
            "",
            real_results_lines,
            "",
            analysis_lines,
            "",
            self._build_grounding_status_context(grounding),
            "",
            self._cite_keys_block(core["ref_lines"]),
        ]
        return "\n".join(p for p in parts if p is not None)

    def _ctx_default(
        self,
        core: dict[str, Any],
        grounding: GroundingPacket | None = None,
        experiment_results: dict | None = None,
        experiment_status: str = "pending",
        experiment_analysis: dict | None = None,
        experiment_summary: str = "",
        **_kwargs: Any,
    ) -> str:
        """Fallback: build full context for unknown section labels."""
        return self._build_full_context(
            core["ideation"],
            core["blueprint"],
            core["cite_keys"],
            experiment_results,
            experiment_status,
            experiment_analysis,
            experiment_summary,
            grounding,
        )

    @staticmethod
    def _cite_keys_block(ref_lines: list[str]) -> str:
        """Format citation keys block."""
        return (
            "=== CITATION KEYS (use ONLY these exact keys with \\cite{}) ===\n"
            + "\n".join(ref_lines)
            + "\n=== END CITATION KEYS ==="
        )

    # ------------------------------------------------------------------
    # P0-B: Contribution Contract extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_contribution_contract(
        intro_content: str,
        method_name: str = "",
    ) -> ContributionContract:
        r"""Extract structured contribution claims from Introduction LaTeX content.

        Parses \begin{itemize}...\end{itemize} blocks to find \item entries
        that represent the paper's contribution claims.

        Args:
            intro_content: The generated Introduction section LaTeX.
            method_name: The proposed method name from blueprint.

        Returns:
            ContributionContract with extracted claims.
        """
        contract = ContributionContract(method_name=method_name)

        # Find itemize/enumerate blocks (contributions are usually in the last one)
        list_blocks = re.findall(
            r'\\begin\{(?:itemize|enumerate)\}(.*?)\\end\{(?:itemize|enumerate)\}',
            intro_content, re.DOTALL,
        )
        if not list_blocks:
            return contract

        # Use the last list block (contributions typically appear at the end)
        contrib_block = list_blocks[-1]
        items = re.split(r'\\item\s*', contrib_block)
        items = [it.strip() for it in items if it.strip()]

        for item_text in items:
            # Clean LaTeX for analysis (remove cite, ref, textbf, etc.)
            clean = re.sub(r'\\(?:cite[tp]?|ref|eqref|label)\{[^}]*\}', '', item_text)
            clean = re.sub(r'\\(?:textbf|textit|emph)\{([^}]*)\}', r'\1', clean)
            clean = re.sub(r'[~\\]', ' ', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()

            # Classify claim type
            lower = clean.lower()
            if any(kw in lower for kw in (
                "experiment", "demonstrate", "achieve", "outperform",
                "state-of-the-art", "sota", "benchmark", "empirical",
                "show that", "shows that",
            )):
                claim_type = "empirical"
            elif any(kw in lower for kw in (
                "introduce", "design", "develop", "novel",
                "module", "component", "mechanism", "layer",
            )):
                claim_type = "component"
            else:
                claim_type = "method"

            # Extract key terms: capitalized multi-word names, or text in \textbf{}
            key_terms: list[str] = []
            # From \textbf{...} in original
            bold_terms = re.findall(r'\\textbf\{([^}]+)\}', item_text)
            key_terms.extend(t.strip() for t in bold_terms if len(t.strip()) > 1)
            # Capitalized phrases (2+ words starting with caps, e.g. "Adaptive Channel Attention")
            cap_phrases = re.findall(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', clean,
            )
            for phrase in cap_phrases:
                if len(phrase) <= 5:
                    continue
                # Skip if this phrase is a substring of an existing term
                if any(phrase in existing for existing in key_terms):
                    continue
                # If an existing term is a substring of this phrase, replace it
                key_terms = [t for t in key_terms if t not in phrase]
                key_terms.append(phrase)
            # Method name if mentioned
            if method_name and method_name.lower() in lower:
                if method_name not in key_terms:
                    key_terms.insert(0, method_name)

            # Use first ~200 chars of item text (cleaned)
            claim_text = clean[:200].rstrip()

            contract.claims.append(ContributionClaim(
                text=claim_text,
                claim_type=claim_type,
                key_terms=key_terms[:5],  # cap at 5 terms
            ))

        return contract

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
