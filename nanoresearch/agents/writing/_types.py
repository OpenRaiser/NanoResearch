"""Shared data types for the writing package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResultCompleteness = Literal["full", "partial", "quick_eval", "none"]


@dataclass
class GroundingPacket:
    """Structured summary of all evidence available for paper writing.

    Built once from execution/analysis outputs, consumed by every section
    generator so they share a consistent view of what evidence exists.
    """

    experiment_status: str = "pending"
    result_completeness: ResultCompleteness = "none"

    # Structured results
    main_results: list[dict] = field(default_factory=list)
    ablation_results: list[dict] = field(default_factory=list)
    comparison_with_baselines: dict = field(default_factory=dict)
    final_metrics: dict = field(default_factory=dict)

    # Narrative evidence
    key_findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    training_dynamics: str = ""
    analysis_summary: str = ""
    experiment_summary_md: str = ""

    # Evidence gaps — things the paper cannot ground
    evidence_gaps: list[str] = field(default_factory=list)

    # Pre-built LaTeX table scaffolds (empty string if not available)
    main_table_latex: str = ""
    ablation_table_latex: str = ""

    @property
    def has_real_results(self) -> bool:
        return self.result_completeness != "none"

    def to_output_dict(self) -> dict:
        """Serialize for inclusion in writing output metadata."""
        return {
            "experiment_status": self.experiment_status,
            "result_completeness": self.result_completeness,
            "has_real_results": self.has_real_results,
            "num_main_results": len(self.main_results),
            "num_ablation_results": len(self.ablation_results),
            "has_baseline_comparison": bool(self.comparison_with_baselines),
            "evidence_gaps": self.evidence_gaps,
        }


@dataclass
class ContributionClaim:
    """A single contribution claim extracted from the Introduction."""
    text: str                # The claim text (e.g., "We propose X, a ... that ...")
    claim_type: str          # "method" | "component" | "empirical"
    key_terms: list[str]     # Key terms/names mentioned (e.g., method name, module name)


@dataclass
class ContributionContract:
    """Structured claims extracted from Introduction, injected into later sections.

    Ensures consistency: every claim in Introduction has matching content
    in Method (technical detail), Experiments (evidence), and Conclusion (summary).
    """
    claims: list[ContributionClaim] = field(default_factory=list)
    method_name: str = ""

    def for_section(self, section_label: str) -> str:
        """Build a contract guidance block for a specific section."""
        if not self.claims:
            return ""

        lines = ["=== CONTRIBUTION CONTRACT ==="]
        lines.append("The Introduction made the following contribution claims.")
        lines.append("This section MUST be consistent with these claims.\n")

        if section_label == "sec:method":
            lines.append("For EACH claim below, provide full technical detail:")
            for i, c in enumerate(self.claims, 1):
                if c.claim_type in ("method", "component"):
                    lines.append(f"  {i}. {c.text}")
                    if c.key_terms:
                        lines.append(f"     → Describe: {', '.join(c.key_terms)}")
                else:
                    lines.append(f"  {i}. [empirical — will be addressed in Experiments]")
            lines.append("\nEvery method/component claim MUST have a corresponding "
                         "\\subsection{} with equations and design rationale.")

        elif section_label == "sec:experiments":
            lines.append("For EACH claim below, provide experimental EVIDENCE:")
            for i, c in enumerate(self.claims, 1):
                lines.append(f"  {i}. {c.text}")
                if c.claim_type == "empirical":
                    lines.append("     → MUST show quantitative results supporting this claim")
                elif c.claim_type == "component":
                    lines.append("     → MUST have ablation row removing/replacing this component")
                else:
                    lines.append("     → MUST show overall method performance")
            lines.append("\nEvery contribution MUST map to either main results or ablation results.")

        elif section_label == "sec:conclusion":
            lines.append("Summarize each contribution with evidence:")
            for i, c in enumerate(self.claims, 1):
                lines.append(f"  {i}. {c.text}")
            lines.append("\nRestate each contribution with the supporting evidence found. "
                         "Do NOT introduce new claims not in Introduction.")

        elif section_label == "sec:related":
            lines.append("Position this work relative to prior art for each claim:")
            for i, c in enumerate(self.claims, 1):
                if c.claim_type in ("method", "component"):
                    lines.append(f"  {i}. {c.text}")
                    lines.append("     → Discuss how prior methods address this differently")

        lines.append("=== END CONTRIBUTION CONTRACT ===")
        return "\n".join(lines)
