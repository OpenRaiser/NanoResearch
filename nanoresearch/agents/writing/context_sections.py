"""Per-section context builders and contribution contract extraction."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ._types import ContributionContract, ContributionClaim, GroundingPacket

logger = logging.getLogger(__name__)


class _ContextSectionsMixin:
    """Mixin — section-specific context builders and contribution contract."""

    def _build_section_context(
        self,
        section_label: str,
        core: dict[str, Any],
        grounding: GroundingPacket | None = None,
        experiment_results: dict | None = None,
        experiment_status: str = "pending",
        experiment_analysis: dict | None = None,
        experiment_summary: str = "",
        prior_sections: dict[str, str] | None = None,
    ) -> str:
        """Build a tailored context string for a specific section."""
        dispatcher = {
            "sec:intro": self._ctx_introduction,
            "sec:related": self._ctx_related_work,
            "sec:method": self._ctx_method,
            "sec:experiments": self._ctx_experiments,
            "sec:conclusion": self._ctx_conclusion,
        }
        builder = dispatcher.get(section_label, self._ctx_default)
        ctx = builder(
            core,
            grounding=grounding,
            experiment_results=experiment_results,
            experiment_status=experiment_status,
            experiment_analysis=experiment_analysis,
            experiment_summary=experiment_summary,
        )
        # For Experiments and Conclusion: inject Method's committed implementation details
        if section_label in ("sec:experiments", "sec:conclusion") and prior_sections:
            method_content = prior_sections.get("Method", "") or prior_sections.get("sec:method", "")
            impl_block = self._extract_method_impl_details(method_content)
            if impl_block:
                ctx += impl_block
        return ctx

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
            self._baseline_cite_block(core.get("baseline_cite_map", {})),
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
            + "\n\nCITATION VERIFICATION RULE (CRITICAL):\n"
            "Before writing \\cite{key}, CHECK that the key's TITLE above matches "
            "what you are attributing. For example:\n"
            "  - If you mention AdamW optimizer, find the key whose title is about AdamW "
            "(Loshchilov & Hutter), NOT a different paper.\n"
            "  - If you mention a baseline method (e.g. MuLOT), cite the key whose "
            "title IS that method, NOT a different paper by similar authors.\n"
            "  - If no key matches, do NOT cite — leave uncited rather than misattribute.\n"
            "=== END CITATION KEYS ==="
        )

    @staticmethod
    def _baseline_cite_block(baseline_cite_map: dict[str, str]) -> str:
        """Format baseline->cite key mapping block for experiments context."""
        if not baseline_cite_map:
            return ""
        lines = [
            "=== BASELINE -> CITATION KEY MAPPING ===",
            "When mentioning these baselines, use EXACTLY these citation keys:",
        ]
        for bname, ckey in baseline_cite_map.items():
            lines.append(f"  {bname} -> \\cite{{{ckey}}}")
        lines.append(
            "If a baseline is NOT listed above, look up its key in the CITATION KEYS "
            "section by matching the paper title. Do NOT guess."
        )
        lines.append("=== END BASELINE MAPPING ===")
        return "\n".join(lines)

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
        """
        contract = ContributionContract(method_name=method_name)

        list_blocks = re.findall(
            r'\\begin\{(?:itemize|enumerate)\}(.*?)\\end\{(?:itemize|enumerate)\}',
            intro_content, re.DOTALL,
        )
        if not list_blocks:
            return contract

        contrib_block = list_blocks[-1]
        items = re.split(r'\\item\s*', contrib_block)
        items = [it.strip() for it in items if it.strip()]

        for item_text in items:
            clean = re.sub(r'\\(?:cite[tp]?|ref|eqref|label)\{[^}]*\}', '', item_text)
            clean = re.sub(r'\\(?:textbf|textit|emph)\{([^}]*)\}', r'\1', clean)
            clean = re.sub(r'[~\\]', ' ', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()

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

            key_terms: list[str] = []
            bold_terms = re.findall(r'\\textbf\{([^}]+)\}', item_text)
            key_terms.extend(t.strip() for t in bold_terms if len(t.strip()) > 1)
            cap_phrases = re.findall(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', clean,
            )
            for phrase in cap_phrases:
                if len(phrase) <= 5:
                    continue
                if any(phrase in existing for existing in key_terms):
                    continue
                key_terms = [t for t in key_terms if t not in phrase]
                key_terms.append(phrase)
            if method_name and method_name.lower() in lower:
                if method_name not in key_terms:
                    key_terms.insert(0, method_name)

            claim_text = clean[:200].rstrip()

            contract.claims.append(ContributionClaim(
                text=claim_text,
                claim_type=claim_type,
                key_terms=key_terms[:5],
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
