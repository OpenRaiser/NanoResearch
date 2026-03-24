"""WritingAgent main run method and figure placement logic."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from nanoresearch.evolution.memory import MemoryType
from ._types import ContributionContract, GroundingPacket
from .import _check_global_consistency, PAPER_SECTIONS, PAPER_MODE_SECTIONS
from .section_writer import SURVEY_SECTION_PROMPTS
from nanoresearch.schemas.paper import PaperSkeleton, Section

logger = logging.getLogger(__name__)


class _WritingAgentMixin:
    """Mixin — WritingAgent.run() and figure placement logic."""

    async def run(self, **inputs: Any) -> dict[str, Any]:
        ideation: dict = inputs.get("ideation_output", {})
        blueprint: dict = inputs.get("experiment_blueprint", {})
        figure_output: dict = inputs.get("figure_output", {})
        template_format: str = inputs.get("template_format", self.config.template_format)
        experiment_results: dict = inputs.get("experiment_results", {})
        experiment_analysis: dict = inputs.get("experiment_analysis", {})
        experiment_summary: str = inputs.get("experiment_summary", "")
        experiment_status: str = inputs.get("experiment_status", "pending")
        authors: list[str] = inputs.get("authors", None) or ["NanoResearch"]

        # Detect paper mode from ideation output
        paper_mode_str: str = ideation.get("paper_mode", "original_research")
        is_survey = paper_mode_str != "original_research"
        self.log(f"Paper mode: {paper_mode_str} (survey={is_survey})")

        # Select section list based on paper_mode
        if is_survey:
            # Resolve paper_mode string to PaperMode enum for PAPER_MODE_SECTIONS lookup
            from nanoresearch.schemas.manifest import PaperMode
            try:
                paper_mode_enum = PaperMode(paper_mode_str)
            except ValueError:
                paper_mode_enum = PaperMode.ORIGINAL_RESEARCH
            section_list = PAPER_MODE_SECTIONS.get(paper_mode_enum, PAPER_SECTIONS)
        else:
            section_list = PAPER_SECTIONS

        self.log("Starting paper writing")
        adaptive_context = self.build_adaptive_context(
            "writing",
            topic=ideation.get("topic", ""),
            blueprint=blueprint,
            text=json.dumps({
                "paper_mode": paper_mode_str,
                "topic": ideation.get("topic", ""),
                "selected_hypothesis": ideation.get("selected_hypothesis", ""),
            }, ensure_ascii=False),
            tags=[ideation.get("topic", ""), paper_mode_str, template_format],
            template_format=template_format,
            include_script_recommendations=True,
        )
        retry_error = str(inputs.get("_retry_error", "")).strip()
        if retry_error:
            self.learn_from_trace(
                "writing",
                "writing_retry",
                retry_error,
                tags=[ideation.get("topic", ""), paper_mode_str, "retry"],
            )

        # Step 0a: Build grounding packet
        grounding = self._build_grounding_packet(
            experiment_results, experiment_status,
            experiment_analysis, experiment_summary, blueprint,
        )
        self.log(
            f"Grounding: completeness={grounding.result_completeness}, "
            f"main_results={len(grounding.main_results)}, "
            f"ablations={len(grounding.ablation_results)}, "
            f"baselines={'yes' if grounding.comparison_with_baselines else 'no'}"
        )
        if grounding.evidence_gaps:
            self.log(f"Evidence gaps: {grounding.evidence_gaps}")

        # Step 0b: Build cite key mapping from papers
        papers = ideation.get("papers", [])
        cite_keys = self._build_cite_keys(papers)
        bibtex = self._build_bibtex(papers, cite_keys)

        # Build per-section context primitives (P0-A)
        core_ctx = self._build_core_context(ideation, blueprint, cite_keys)
        if adaptive_context:
            core_ctx = f"{core_ctx}\n\n{adaptive_context}"

        # Title & abstract need a broad context
        title_abstract_ctx = self._ctx_introduction(core_ctx, grounding=grounding)

        # Step 1: Generate title
        title = await self._generate_title(title_abstract_ctx)
        self.log(f"Title: {title}")

        # Step 2: Generate abstract
        abstract = await self._generate_abstract(title_abstract_ctx, grounding)
        self.log("Abstract generated")

        # Step 3: Build figures & table data from blueprint
        figure_blocks = self._build_figure_blocks(blueprint, figure_output)

        # Step 4: Generate each section independently, embed figures inline
        placed_figures: set[str] = set()

        # P0-B: Contribution contract
        contribution_contract: ContributionContract | None = None
        method_name = (blueprint.get("proposed_method") or {}).get("name", "")

        sections = []
        prior_sections_summary: list[str] = []
        for heading, label, section_instructions, fig_keys in section_list:
            # For surveys: use survey-specific prompts (stored separately) and skip experiment results
            if is_survey:
                instructions = SURVEY_SECTION_PROMPTS.get(label, section_instructions)
                # Surveys do not have experiment results — pass None/defaults to context builders
                ctx_experiment_results: dict | None = None
                ctx_experiment_status: str = "pending"
                ctx_experiment_analysis: dict | None = None
                ctx_experiment_summary: str = ""
            else:
                instructions = section_instructions
                ctx_experiment_results = experiment_results
                ctx_experiment_status = experiment_status
                ctx_experiment_analysis = experiment_analysis
                ctx_experiment_summary = experiment_summary

            self.log(f"Writing section: {heading}")

            _prior_content = {s.heading: s.content for s in sections}
            section_ctx = self._build_section_context(
                label, core_ctx, grounding=grounding,
                experiment_results=ctx_experiment_results,
                experiment_status=ctx_experiment_status,
                experiment_analysis=ctx_experiment_analysis,
                experiment_summary=ctx_experiment_summary,
                prior_sections=_prior_content,
            )

            # Contribution contract is only for original research (not surveys)
            if not is_survey and contribution_contract and label != "sec:intro":
                contract_block = contribution_contract.for_section(label)
                if contract_block:
                    section_ctx = section_ctx + "\n\n" + contract_block

            remaining_figs = [k for k in figure_blocks if k not in placed_figures]
            fig_list_text = "\n".join(
                f"  - \\ref{{fig:{k}}}: {k}" for k in remaining_figs
            )
            placed_note = ""
            if placed_figures:
                placed_list = ", ".join(sorted(placed_figures))
                placed_note = (
                    f"\nFigures ALREADY placed in previous sections (do NOT include again): "
                    f"{placed_list}\n"
                )

            table_injection = ""
            if label == "sec:experiments":
                table_parts = []
                if grounding.main_table_latex:
                    if grounding.has_real_results:
                        header = "=== PRE-BUILT MAIN RESULTS TABLE (use this EXACTLY, do NOT rebuild) ==="
                    else:
                        header = (
                            "=== SCAFFOLD MAIN RESULTS TABLE ===\n"
                            "Use this table structure. Fill baseline cells with numbers from "
                            "their original papers (cite sources). Keep proposed method cells as '--'."
                        )
                    table_parts.append(
                        header + "\n" + grounding.main_table_latex + "\n=== END PRE-BUILT TABLE ==="
                    )
                if grounding.ablation_table_latex:
                    if grounding.has_real_results:
                        header = "=== PRE-BUILT ABLATION TABLE (use this EXACTLY, do NOT rebuild) ==="
                    else:
                        header = (
                            "=== SCAFFOLD ABLATION TABLE ===\n"
                            "Use this table structure. Keep all cells as '--' since no "
                            "ablation data is available."
                        )
                    table_parts.append(
                        header + "\n" + grounding.ablation_table_latex + "\n=== END PRE-BUILT TABLE ==="
                    )
                if table_parts:
                    table_injection = "\n\n" + "\n\n".join(table_parts)

            conclusion_binding = ""
            if label == "sec:conclusion":
                if is_survey:
                    # Surveys bind to key_challenges and future_directions instead of experiment results
                    ideation = inputs.get("ideation_output", {})
                    key_challenges = ideation.get("key_challenges", []) if isinstance(ideation, dict) else []
                    future_directions = ideation.get("future_directions", []) if isinstance(ideation, dict) else []
                    if key_challenges or future_directions:
                        challenges_str = "\n".join(f"  - {t}" for t in key_challenges) if key_challenges else "  (none provided)"
                        directions_str = "\n".join(f"  - {t}" for t in future_directions) if future_directions else "  (none provided)"
                        conclusion_binding = (
                            "\n\n=== CONCLUSION RESULT BINDING (SURVEY) ===\n"
                            "Key Challenges:\n" + challenges_str + "\n\n"
                            "Future Directions:\n" + directions_str + "\n\n"
                            "Use these to summarize open challenges and future research trajectories.\n"
                            "Do NOT cite specific experiment performance numbers.\n"
                            "=== END BINDING ==="
                        )
                elif grounding.has_real_results and grounding.final_metrics:
                    metric_strs = [f"{k}={v}" for k, v in list(grounding.final_metrics.items())[:5]]
                    conclusion_binding = (
                        "\n\n=== CONCLUSION RESULT BINDING ===\n"
                        f"Real metrics to reference: {', '.join(metric_strs)}\n"
                        "Mention key results quantitatively when summarizing contributions. "
                        "Use the exact numbers above.\n"
                        "=== END BINDING ==="
                    )
                elif not grounding.has_real_results:
                    conclusion_binding = (
                        "\n\n=== CONCLUSION RESULT BINDING ===\n"
                        "No real experiment results. Do NOT cite specific performance numbers. "
                        "Focus on method design and future work.\n"
                        "=== END BINDING ==="
                    )

            context_with_figs = (
                f"{section_ctx}\n\n"
                f"=== AVAILABLE FIGURES (use \\ref{{fig:NAME}} to reference) ===\n"
                f"{fig_list_text}\n"
                f"{placed_note}"
                f"=== END FIGURES ==="
                f"{table_injection}"
                f"{conclusion_binding}"
            )

            content = await self._generate_section(
                context_with_figs, heading, instructions, prior_sections_summary
            )

            # Post-generation table verification for Experiments
            if label == "sec:experiments" and (
                grounding.main_table_latex or grounding.ablation_table_latex
            ):
                content = self._verify_and_inject_tables(content, grounding, heading)

            # Detect figures the LLM already embedded
            llm_placed_labels = re.findall(
                r'\\begin\{figure\*?\}.*?\\label\{fig:([^}]+)\}.*?\\end\{figure\*?\}',
                content, re.DOTALL,
            )
            for fig_label in llm_placed_labels:
                if fig_label in figure_blocks and fig_label not in placed_figures:
                    placed_figures.add(fig_label)
                    self.log(f"  LLM already placed fig:{fig_label} in {heading}")
            llm_placed_files = re.findall(
                r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content,
            )
            for fname in llm_placed_files:
                stem = fname.rsplit(".", 1)[0]
                for fk in figure_blocks:
                    if fk in placed_figures:
                        continue
                    if fk in stem or stem.endswith(fk):
                        placed_figures.add(fk)
                        self.log(f"  LLM already included {fname} -> marking fig:{fk} as placed in {heading}")

            # Smart figure placement
            content, placed_figures = self._place_section_figures(
                content, label, heading, figure_blocks, placed_figures,
            )

            sections.append(Section(heading=heading, label=label, content=content))
            snippet = content[:200].replace("\n", " ").strip()
            prior_sections_summary.append(f"[{heading}]: {snippet}...")

            # P0-B: Extract contribution contract after Introduction (original research only)
            if not is_survey and label == "sec:intro" and not contribution_contract:
                contribution_contract = self._extract_contribution_contract(content, method_name)
                if contribution_contract.claims:
                    self.log(
                        f"Contribution contract: {len(contribution_contract.claims)} claims "
                        f"({', '.join(c.claim_type for c in contribution_contract.claims)})"
                    )
                else:
                    self.log("No contribution claims extracted from Introduction")

        # Fallback: distribute remaining figures
        remaining = [k for k in figure_blocks if k not in placed_figures]
        if remaining:
            self.log(f"Fallback placement for {len(remaining)} unplaced figures: {remaining}")
            section_hints = {
                "sec:intro": ("qualitative", "example", "motivation", "task",
                              "illustration", "counterfactual", "demo", "teaser",
                              "intuition", "sample"),
                "sec:experiments": ("result", "comparison", "performance", "main", "latency",
                                    "tradeoff", "trade_off", "efficiency", "scalab"),
                "sec:method": ("architecture", "framework", "pipeline", "overview", "model",
                               "diagram", "workflow"),
                "sec:conclusion": ("ablation", "analysis", "error", "contradiction"),
            }
            for fk in remaining:
                target_label = "sec:experiments"
                for sec_label, keywords in section_hints.items():
                    if any(kw in fk for kw in keywords):
                        target_label = sec_label
                        break
                for sec in sections:
                    if sec.label == target_label:
                        sec.content += "\n\n" + figure_blocks[fk]
                        placed_figures.add(fk)
                        self.log(f"  Placed '{fk}' -> {target_label}")
                        break
                else:
                    for sec in sections:
                        if sec.label == "sec:experiments":
                            sec.content += "\n\n" + figure_blocks[fk]
                            placed_figures.add(fk)
                            self.log(f"  Placed '{fk}' -> sec:experiments (fallback)")
                            break

        # Post-assembly validation
        final_missing = [k for k in figure_blocks if k not in placed_figures]
        if final_missing:
            self.log(f"CRITICAL: {len(final_missing)} figures still unplaced after all passes: {final_missing}")
            for sec in sections:
                if sec.label == "sec:experiments":
                    for fk in final_missing:
                        sec.content += "\n\n" + figure_blocks[fk]
                        self.log(f"  Force-injected '{fk}' -> sec:experiments")
                    break

        self.log(f"Figure placement complete: {len(figure_blocks)} blocks, "
                 f"{len(placed_figures)} placed")

        # Per-section dedup
        self._dedup_section_figures(sections)

        # Step 5: Build skeleton
        skeleton = PaperSkeleton(
            title=title, authors=authors, abstract=abstract,
            sections=sections, figures=[],
            template_format=template_format, references_bibtex=bibtex,
        )

        # Step 6: Render LaTeX + sanitize
        latex_content = self._render_latex(skeleton)
        latex_content = self._sanitize_latex(latex_content)

        # Step 6b-pre: Full-document figure dedup
        latex_content = self._dedup_full_doc_figures(latex_content)

        # Step 6b: Final LaTeX-level figure validation
        latex_content = self._validate_figures_in_latex(latex_content, figure_output)

        # Step 6c: Resolve missing citations
        bibtex = await self._resolve_missing_citations(latex_content, bibtex)

        # Step 6d: Citation coverage validation
        citation_report = self._validate_citation_coverage(latex_content, ideation, cite_keys)
        if citation_report.get("missing_must_cites"):
            self.log(f"Must-cite enforcement: {len(citation_report['missing_must_cites'])} "
                     f"must-cite papers not referenced, injecting into Related Work")
            latex_content = self._inject_must_cites(
                latex_content, citation_report["missing_must_cites"], cite_keys, ideation
            )
            bibtex = await self._resolve_missing_citations(latex_content, bibtex)

        self._log_citation_report(citation_report)

        # Step 6d.5: Cleanup unused BibTeX entries
        bibtex = self._cleanup_unused_bibtex(latex_content, bibtex)

        # Step 6e: Global consistency check
        consistency_issues = _check_global_consistency(latex_content, abstract, sections)
        if consistency_issues:
            self.log(f"Consistency check: {len(consistency_issues)} issue(s) found")
            for issue in consistency_issues:
                self.log(f"  - {issue}")

        # Save outputs
        tex_path = self.workspace.write_text("drafts/paper.tex", latex_content)
        bib_content = self._sanitize_bibtex(bibtex)
        bib_path = self.workspace.write_text("drafts/references.bib", bib_content)
        skeleton_path = self.workspace.write_json(
            "drafts/paper_skeleton.json", skeleton.model_dump(mode="json"),
        )

        self.workspace.register_artifact("paper_tex", tex_path, self.stage)
        self.workspace.register_artifact("references_bib", bib_path, self.stage)
        self.workspace.register_artifact("paper_skeleton", skeleton_path, self.stage)

        # Step 7: Try to compile PDF
        pdf_result = await self._compile_pdf(tex_path, template_format=template_format)

        result = {
            "tex_path": str(tex_path),
            "bib_path": str(bib_path),
            "grounding": grounding.to_output_dict(),
            "consistency_issues": consistency_issues,
        }
        if "pdf_path" in pdf_result:
            result["pdf_path"] = pdf_result["pdf_path"]
            self.workspace.register_artifact(
                "paper_pdf", self.workspace.path / "drafts" / "paper.pdf", self.stage
            )
        else:
            result["pdf_error"] = pdf_result.get("error", "Unknown error")
            self.log(f"PDF compilation failed: {result['pdf_error']}")

        topic_name = ideation.get("topic", "unknown topic")
        pdf_ready = "yes" if "pdf_path" in result else "no"
        self.remember_context(
            MemoryType.PROJECT_CONTEXT,
            f"Writing completed for {topic_name} in mode {paper_mode_str} with template {template_format}. PDF={pdf_ready}.",
            importance=0.7,
            tags=[ideation.get("topic", ""), paper_mode_str, "writing", template_format],
            source="writing_output",
            topic=ideation.get("topic", ""),
        )
        self.log("Writing stage complete")
        return result

    def _place_section_figures(
        self,
        content: str,
        label: str,
        heading: str,
        figure_blocks: dict[str, str],
        placed_figures: set[str],
    ) -> tuple[str, set[str]]:
        """Smart figure placement for a section. Returns (content, placed_figures)."""
        _arch_kws = ("overview", "framework", "pipeline", "architecture", "model")
        _intro_kws = ("qualitative", "example", "motivation", "task",
                       "illustration", "counterfactual", "demo", "teaser")

        if label == "sec:intro":
            intro_keywords = _intro_kws + ("intuition", "sample")
            for fk in list(figure_blocks.keys()):
                if fk in placed_figures:
                    continue
                if any(kw in fk for kw in intro_keywords):
                    content += "\n\n" + figure_blocks[fk]
                    placed_figures.add(fk)
                    break

        if label == "sec:method":
            arch_keywords = _arch_kws
            for fk in list(figure_blocks.keys()):
                if fk not in placed_figures:
                    continue
                if not any(kw in fk for kw in arch_keywords):
                    continue
                fig_pattern = re.compile(
                    r'\n*\\begin\{figure\*?\}.*?\\label\{fig:'
                    + re.escape(fk)
                    + r'\}.*?\\end\{figure\*?\}\n*',
                    re.DOTALL,
                )
                match = fig_pattern.search(content)
                if match and match.start() > 200:
                    content = content[:match.start()] + content[match.end():]
                    content = figure_blocks[fk] + "\n\n" + content.lstrip("\n")
                    self.log(f"  Moved LLM-placed fig:{fk} to top of Method")
                break

            for fk in list(figure_blocks.keys()):
                if fk in placed_figures:
                    continue
                if any(kw in fk for kw in arch_keywords):
                    content = figure_blocks[fk] + "\n\n" + content
                    placed_figures.add(fk)
                    break

        # Insert remaining figures near their \ref
        for fk, blk in figure_blocks.items():
            if fk in placed_figures:
                continue
            if label != "sec:method" and any(kw in fk for kw in _arch_kws):
                continue
            if label != "sec:intro" and any(kw in fk for kw in _intro_kws):
                continue
            content, inserted = self._insert_figure_near_ref(content, fk, blk)
            if inserted:
                placed_figures.add(fk)

        return content, placed_figures

    def _dedup_section_figures(self, sections: list[Section]) -> None:
        """Remove duplicate figure blocks across sections (keep first occurrence)."""
        seen_fig_labels: set[str] = set()
        seen_fig_files: set[str] = set()
        for sec in sections:
            def _dedup_figure(m: re.Match) -> str:
                block = m.group(0)
                label_m = re.search(r'\\label\{(fig:[^}]+)\}', block)
                lbl = label_m.group(1) if label_m else None
                if lbl and lbl in seen_fig_labels:
                    self.log(f"  Removed duplicate figure {lbl} from {sec.heading}")
                    return ""
                file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
                if file_m:
                    fname = file_m.group(1)
                    if fname in seen_fig_files:
                        self.log(f"  Removed duplicate figure file {fname} from {sec.heading}")
                        return ""
                    seen_fig_files.add(fname)
                if lbl:
                    seen_fig_labels.add(lbl)
                return block
            sec.content = re.sub(
                r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
                _dedup_figure, sec.content, flags=re.DOTALL,
            )
            sec.content = re.sub(r'\n{3,}', '\n\n', sec.content)

    def _dedup_full_doc_figures(self, latex_content: str) -> str:
        """Full-document figure dedup (safety net after assembly)."""
        seen_labels: set[str] = set()
        seen_files: set[str] = set()

        def _dedup_assembled(m: re.Match) -> str:
            block = m.group(0)
            lbl_m = re.search(r'\\label\{(fig:[^}]+)\}', block)
            if lbl_m:
                lbl = lbl_m.group(1)
                if lbl in seen_labels:
                    self.log(f"  Full-doc dedup: removed duplicate figure {lbl}")
                    return ""
                seen_labels.add(lbl)
            file_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
            if file_m:
                fname = file_m.group(1)
                if fname in seen_files:
                    self.log(f"  Full-doc dedup: removed duplicate figure file {fname}")
                    return ""
                seen_files.add(fname)
            return block

        latex_content = re.sub(
            r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
            _dedup_assembled, latex_content, flags=re.DOTALL,
        )
        latex_content = re.sub(r'\n{3,}', '\n\n', latex_content)
        return latex_content
