"""Evolved natural-language and script skill stores with review and artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")
_ALLOWED_SCRIPT_CATEGORIES = {
    "environment_setup",
    "literature_tracking",
    "figure_formatting",
}


class SkillDomain(str, Enum):
    LITERATURE = "literature"
    PLANNING = "planning"
    CODING = "coding"
    WRITING = "writing"
    REVIEW = "review"
    EXPERIMENT = "experiment"


class SkillReviewDecision(str, Enum):
    ADD_NEW = "add_new"
    MERGE_INTO_EXISTING = "merge_into_existing"
    REVISE_THEN_RETRY = "revise_then_retry"
    DISCARD = "discard"


class ScriptSkillCategory(str, Enum):
    ENVIRONMENT_SETUP = "environment_setup"
    LITERATURE_TRACKING = "literature_tracking"
    FIGURE_FORMATTING = "figure_formatting"


class ScriptTestStatus(str, Enum):
    PROPOSED = "proposed"
    PASSED = "passed"
    FAILED = "failed"


class SkillCandidate(BaseModel):
    candidate_id: str
    domain: SkillDomain
    trigger_pattern: str
    name: str
    description: str = ""
    when_to_use: str = ""
    instructions: list[str] = Field(default_factory=list)
    source_trace: str
    source_stage: str = ""
    confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SkillReview(BaseModel):
    review_id: str
    candidate_id: str
    decision: SkillReviewDecision
    matched_skill_ids: list[str] = Field(default_factory=list)
    rationale: str
    safety_note: str
    revised_name: str = ""
    revised_when_to_use: str = ""
    revised_instructions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NaturalLanguageSkill(BaseModel):
    skill_id: str
    stable_id: str = ""
    version: str = "0.1.0"
    skill_type: str = "natural_language"
    domain: SkillDomain
    trigger_pattern: str
    name: str = ""
    description: str = ""
    when_to_use: str = ""
    instructions: list[str] = Field(default_factory=list)
    rule_text: str = ""
    source_trace: str = ""
    source_stage: str = ""
    confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)
    last_applied_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)
    review_history: list[dict[str, Any]] = Field(default_factory=list)


class ScriptSkill(BaseModel):
    skill_id: str
    skill_type: str = "python_script"
    category: ScriptSkillCategory
    name: str
    description: str
    input_contract: str = ""
    output_contract: str = ""
    safe_to_autorun: bool = False
    test_status: ScriptTestStatus = ScriptTestStatus.PROPOSED
    script_path: str
    scope: str = "project"
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SkillLifecycleResult(BaseModel):
    candidate: SkillCandidate
    review: SkillReview
    skill: NaturalLanguageSkill | None = None
    action: str


class SkillEvolutionStore:
    """Persistent adaptive skill registry under ``~/.nanoresearch/skills``."""

    def __init__(self, root: Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.root = root or (Path.home() / ".nanoresearch" / "skills")
        self.nl_file = self.root / "natural_language.json"
        self.script_file = self.root / "script_registry.json"
        self.candidate_file = self.root / "candidate_log.json"
        self.review_file = self.root / "review_log.json"
        self.users_root = self.root / "Users" / "default"
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self.users_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_tags(tags: list[str] | None) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for tag in tags or []:
            norm = re.sub(r"\s+", " ", str(tag).strip().lower())
            if not norm or norm in seen:
                continue
            seen.add(norm)
            normalized.append(norm)
        return normalized

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_WORD_RE.findall((text or "").lower()))

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
        return slug or "adaptive-skill"

    @staticmethod
    def _skill_id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
        return f"{prefix}-{digest[:12]}"

    @staticmethod
    def _parse_patch(version: str) -> tuple[int, int, int]:
        try:
            major, minor, patch = [int(part) for part in str(version).split(".")[:3]]
            return major, minor, patch
        except Exception:
            return 0, 1, 0

    def _bump_patch(self, version: str) -> str:
        major, minor, patch = self._parse_patch(version)
        return f"{major}.{minor}.{patch + 1}"

    def _load_models(self, path: Path, model_cls: type[BaseModel]) -> list[BaseModel]:
        if not self.enabled or not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load skill store %s: %s", path, exc)
            return []
        models: list[BaseModel] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                models.append(model_cls.model_validate(item))
            except Exception as exc:
                logger.debug("Skipping malformed skill entry: %s", exc)
        return models

    def _save_models(self, path: Path, models: list[BaseModel]) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        payload = [model.model_dump(mode="json") for model in models]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_model(self, path: Path, model: BaseModel) -> None:
        models = self._load_models(path, type(model))
        models.append(model)
        self._save_models(path, models)

    def _artifact_dir(self, skill: NaturalLanguageSkill) -> Path:
        return self.users_root / self._slug(skill.stable_id or skill.name or skill.skill_id)

    def _render_skill_artifact(self, skill: NaturalLanguageSkill) -> str:
        instructions = skill.instructions or ([skill.rule_text] if skill.rule_text else [])
        instruction_lines = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(instructions)) or "1. Follow the preserved guidance from this skill."
        tags = ", ".join(skill.tags)
        return (
            f"# {skill.name or skill.stable_id}\n\n"
            f"- id: {skill.stable_id or skill.skill_id}\n"
            f"- version: {skill.version}\n"
            f"- domain: {skill.domain.value}\n"
            f"- trigger_pattern: {skill.trigger_pattern}\n"
            f"- last_updated: {skill.updated_at.isoformat()}\n"
            f"- tags: {tags}\n\n"
            f"## Description\n{skill.description or skill.rule_text}\n\n"
            f"## When To Use\n{skill.when_to_use or 'Use when the described recurring pattern appears again.'}\n\n"
            f"## Instructions\n{instruction_lines}\n\n"
            f"## Provenance\n- source_stage: {skill.source_stage or 'unknown'}\n- source_trace: {skill.source_trace[:1000] or 'n/a'}\n"
        )

    def _write_skill_artifact(self, skill: NaturalLanguageSkill, review: SkillReview | None = None) -> None:
        if not self.enabled:
            return
        artifact_dir = self._artifact_dir(skill)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "SKILL.md").write_text(self._render_skill_artifact(skill), encoding="utf-8")
        (artifact_dir / "skill.json").write_text(json.dumps(skill.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        if review is not None:
            review_log = artifact_dir / "review_log.json"
            reviews = []
            if review_log.is_file():
                try:
                    reviews = json.loads(review_log.read_text(encoding="utf-8"))
                except Exception:
                    reviews = []
            if not isinstance(reviews, list):
                reviews = []
            reviews.append(review.model_dump(mode="json"))
            review_log.write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_artifact_skills(self) -> list[NaturalLanguageSkill]:
        skills: list[NaturalLanguageSkill] = []
        if not self.enabled or not self.users_root.is_dir():
            return skills
        for skill_json in sorted(self.users_root.rglob("skill.json")):
            try:
                payload = json.loads(skill_json.read_text(encoding="utf-8"))
                skills.append(NaturalLanguageSkill.model_validate(payload))
            except Exception as exc:
                logger.debug("Skipping malformed skill artifact %s: %s", skill_json, exc)
        return skills

    def _load_all_nl_skills(self) -> list[NaturalLanguageSkill]:
        merged: dict[str, NaturalLanguageSkill] = {}
        for skill in self._load_artifact_skills():
            merged[skill.stable_id or skill.skill_id] = skill
        for skill in self._load_models(self.nl_file, NaturalLanguageSkill):
            if not isinstance(skill, NaturalLanguageSkill):
                continue
            key = skill.stable_id or skill.skill_id
            merged.setdefault(key, skill)
        return list(merged.values())

    def _persist_nl_skills(self, skills: list[NaturalLanguageSkill]) -> None:
        self._save_models(self.nl_file, skills)
        for skill in skills:
            self._write_skill_artifact(skill)

    def _infer_name(self, domain: SkillDomain, trigger_pattern: str, instructions: list[str]) -> str:
        head = instructions[0] if instructions else trigger_pattern.replace("_", " ")
        head = re.sub(r"\s+", " ", head).strip().rstrip(".")
        if len(head) > 64:
            head = head[:61].rstrip() + "..."
        return f"{domain.value.title()} Skill: {head}"

    def _heuristic_instruction_list(self, domain: SkillDomain | str, trace: str) -> list[str]:
        domain = SkillDomain(domain)
        trace_lower = (trace or "").lower()
        if any(token in trace_lower for token in ("seed", "variance", "std", "standard deviation")):
            return ["Report aggregate metrics with at least three random seeds.", "Include both mean and standard deviation before claiming improvements."]
        if "ablation" in trace_lower:
            return ["Require one ablation per core module.", "Make each claimed component correspond to an isolated removal study."]
        if any(token in trace_lower for token in ("oom", "cuda out of memory", "memoryerror", "out of memory")):
            return ["Run a reduced-scale dry run before launching full experiments.", "Validate memory usage and logging artifacts before scaling up."]
        if any(token in trace_lower for token in ("citation", "bibtex", "cite")) and domain in {SkillDomain.WRITING, SkillDomain.REVIEW}:
            return ["Use only citation keys that exist in the bibliography.", "Tie each strong factual claim to an available source before revision."]
        if any(token in trace_lower for token in ("environment", "dependency", "module not found", "importerror", "loader", "preflight")):
            return ["Validate the environment with a lightweight preflight dependency check.", "Check dataset loader contracts, paths, and schema assumptions before full execution."]
        if domain == SkillDomain.LITERATURE:
            return ["Broaden literature search queries with synonyms and adjacent task names.", "Use review papers before committing to a novelty claim."]
        if domain == SkillDomain.PLANNING:
            return ["Convert repeated failures into explicit planning constraints.", "Bake reviewer and retry lessons into the next experiment blueprint."]
        if domain == SkillDomain.WRITING:
            return ["Structure each section as context, limitation, and project-specific distinction.", "Preserve citation correctness and avoid unsupported claims."]
        if domain == SkillDomain.REVIEW:
            return ["When a section scores poorly, name the concrete problem and exact fix.", "Preserve strong claims but revise unsupported or unclear statements."]
        return ["Turn repeated failure patterns into an explicit checklist before the next attempt."]

    def extract_skill_candidate(
        self,
        *,
        domain: SkillDomain | str,
        trigger_pattern: str,
        source_trace: str,
        tags: list[str] | None = None,
        confidence: float = 0.55,
        name: str = "",
        description: str = "",
        when_to_use: str = "",
        instructions: list[str] | None = None,
        source_stage: str = "",
    ) -> SkillCandidate | None:
        if not self.enabled:
            return None
        domain = SkillDomain(domain)
        trace = " ".join((source_trace or "").strip().split())
        if not trace:
            return None
        instructions = [str(item).strip() for item in (instructions or []) if str(item).strip()] or self._heuristic_instruction_list(domain, trace)
        if not instructions:
            return None
        trigger_pattern = (trigger_pattern or "adaptive_rule").strip()
        name = (name or self._infer_name(domain, trigger_pattern, instructions)).strip()
        description = (description or instructions[0]).strip()
        when_to_use = (when_to_use or f"Use when '{trigger_pattern}' or a similar recurring pattern appears in {domain.value} work.").strip()
        candidate = SkillCandidate(
            candidate_id=self._skill_id("cand", domain.value, trigger_pattern, trace[:300]),
            domain=domain,
            trigger_pattern=trigger_pattern,
            name=name,
            description=description,
            when_to_use=when_to_use,
            instructions=instructions,
            source_trace=trace[:4000],
            source_stage=source_stage,
            confidence=min(max(confidence, 0.0), 1.0),
            tags=self._normalize_tags(tags),
        )
        candidates = [item for item in self._load_models(self.candidate_file, SkillCandidate) if isinstance(item, SkillCandidate)]
        if all(existing.candidate_id != candidate.candidate_id for existing in candidates):
            candidates.append(candidate)
            self._save_models(self.candidate_file, candidates)
        return candidate

    def _candidate_tokens(self, candidate: SkillCandidate) -> set[str]:
        return self._tokenize(" ".join([candidate.name, candidate.description, candidate.when_to_use, *candidate.instructions, candidate.trigger_pattern]))

    def _skill_tokens(self, skill: NaturalLanguageSkill) -> set[str]:
        return self._tokenize(" ".join([skill.name, skill.description, skill.when_to_use, *skill.instructions, skill.rule_text, skill.trigger_pattern]))

    def _score_similarity(self, candidate: SkillCandidate, skill: NaturalLanguageSkill) -> float:
        if candidate.domain != skill.domain:
            return 0.0
        cand_tokens = self._candidate_tokens(candidate)
        skill_tokens = self._skill_tokens(skill)
        overlap = len(cand_tokens & skill_tokens)
        if not overlap:
            return 0.0
        coverage = overlap / max(1, len(cand_tokens))
        trigger_bonus = 0.8 if candidate.trigger_pattern == skill.trigger_pattern else 0.0
        tag_bonus = 0.35 * len(set(candidate.tags) & set(skill.tags))
        return overlap * 0.35 + coverage * 2.2 + trigger_bonus + tag_bonus

    def review_skill_candidate(self, candidate: SkillCandidate, *, top_k: int = 3) -> SkillReview:
        existing = self._load_all_nl_skills()
        matches = sorted(
            ((self._score_similarity(candidate, skill), skill) for skill in existing),
            key=lambda item: item[0],
            reverse=True,
        )
        matches = [(score, skill) for score, skill in matches if score > 0][:max(1, top_k)]
        matched_ids = [skill.stable_id or skill.skill_id for _, skill in matches]
        decision = SkillReviewDecision.ADD_NEW
        rationale = "Candidate is sufficiently distinct from existing skills and is reusable enough to become a new skill."
        safety_note = "No existing skill is modified by this decision."
        revised_instructions: list[str] = []
        revised_name = ""
        revised_when_to_use = ""

        if len(candidate.instructions) < 1 or len(" ".join(candidate.instructions).split()) < 6:
            decision = SkillReviewDecision.REVISE_THEN_RETRY
            rationale = "Candidate is too terse to safely store as a reusable skill without revision."
            safety_note = "Revise candidate before any change to existing skills."
            revised_instructions = self._heuristic_instruction_list(candidate.domain, candidate.source_trace)
            revised_name = candidate.name
            revised_when_to_use = candidate.when_to_use
        elif matches:
            top_score, top_skill = matches[0]
            candidate_text = " ".join(candidate.instructions).lower()
            top_text = " ".join(top_skill.instructions or [top_skill.rule_text]).lower()
            if top_score >= 3.8 and (candidate_text in top_text or candidate_text == top_text or len(candidate_text) <= len(top_text)):
                decision = SkillReviewDecision.DISCARD
                rationale = f"Candidate substantially overlaps with existing skill {top_skill.stable_id or top_skill.skill_id} and does not add enough new behavior."
                safety_note = "Discarding avoids overwriting an existing stable skill."
            elif top_score >= 2.4:
                decision = SkillReviewDecision.MERGE_INTO_EXISTING
                rationale = f"Candidate is best treated as an incremental improvement to existing skill {top_skill.stable_id or top_skill.skill_id}."
                safety_note = "Merge must preserve the existing skill's core when_to_use and instructions before appending new constraints."
        review = SkillReview(
            review_id=self._skill_id("review", candidate.candidate_id, decision.value, "|".join(matched_ids)),
            candidate_id=candidate.candidate_id,
            decision=decision,
            matched_skill_ids=matched_ids,
            rationale=rationale,
            safety_note=safety_note,
            revised_name=revised_name,
            revised_when_to_use=revised_when_to_use,
            revised_instructions=revised_instructions,
        )
        reviews = [item for item in self._load_models(self.review_file, SkillReview) if isinstance(item, SkillReview)]
        reviews.append(review)
        self._save_models(self.review_file, reviews)
        return review

    def _merge_instructions(self, existing: list[str], incoming: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for instruction in [*(existing or []), *(incoming or [])]:
            norm = " ".join(str(instruction).strip().split())
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(norm)
        return merged

    def maintain_skill(self, candidate: SkillCandidate, *, review: SkillReview | None = None) -> SkillLifecycleResult:
        review = review or self.review_skill_candidate(candidate)
        if review.decision == SkillReviewDecision.REVISE_THEN_RETRY:
            revised_candidate = candidate.model_copy(update={
                "instructions": review.revised_instructions or candidate.instructions,
                "name": review.revised_name or candidate.name,
                "when_to_use": review.revised_when_to_use or candidate.when_to_use,
            })
            second_review = self.review_skill_candidate(revised_candidate)
            if second_review.decision == SkillReviewDecision.REVISE_THEN_RETRY:
                second_review = second_review.model_copy(update={
                    "decision": SkillReviewDecision.DISCARD,
                    "rationale": second_review.rationale + " Candidate still remains too weak after one revision pass.",
                })
            return self.maintain_skill(revised_candidate, review=second_review)

        skills = self._load_all_nl_skills()
        if review.decision == SkillReviewDecision.DISCARD:
            return SkillLifecycleResult(candidate=candidate, review=review, skill=None, action="discard")

        if review.decision == SkillReviewDecision.MERGE_INTO_EXISTING and review.matched_skill_ids:
            target_id = review.matched_skill_ids[0]
            updated_skills: list[NaturalLanguageSkill] = []
            merged_skill: NaturalLanguageSkill | None = None
            for skill in skills:
                key = skill.stable_id or skill.skill_id
                if key != target_id:
                    updated_skills.append(skill)
                    continue
                merged_skill = skill.model_copy(update={
                    "version": self._bump_patch(skill.version),
                    "description": skill.description or candidate.description,
                    "when_to_use": skill.when_to_use or candidate.when_to_use,
                    "instructions": self._merge_instructions(skill.instructions or [skill.rule_text], candidate.instructions),
                    "rule_text": " ".join(self._merge_instructions(skill.instructions or [skill.rule_text], candidate.instructions)),
                    "confidence": max(skill.confidence, candidate.confidence),
                    "updated_at": datetime.now(timezone.utc),
                    "tags": self._normalize_tags(skill.tags + candidate.tags),
                    "review_history": [
                        *skill.review_history,
                        {
                            "review_id": review.review_id,
                            "decision": review.decision.value,
                            "rationale": review.rationale,
                            "safety_note": review.safety_note,
                            "candidate_id": candidate.candidate_id,
                        },
                    ],
                })
                updated_skills.append(merged_skill)
            if merged_skill is None:
                review = review.model_copy(update={
                    "decision": SkillReviewDecision.ADD_NEW,
                    "rationale": review.rationale + " Fallback to add_new because matched skill was not found.",
                })
            else:
                self._persist_nl_skills(updated_skills)
                self._write_skill_artifact(merged_skill, review)
                return SkillLifecycleResult(candidate=candidate, review=review, skill=merged_skill, action="merge_update")

        instructions = self._merge_instructions([], candidate.instructions)
        stable_id = self._skill_id("skill", candidate.domain.value, candidate.name.lower(), candidate.trigger_pattern)
        skill = NaturalLanguageSkill(
            skill_id=self._skill_id("nlskill", candidate.domain.value, candidate.trigger_pattern, candidate.name.lower()),
            stable_id=stable_id,
            version="0.1.0",
            domain=candidate.domain,
            trigger_pattern=candidate.trigger_pattern,
            name=candidate.name,
            description=candidate.description,
            when_to_use=candidate.when_to_use,
            instructions=instructions,
            rule_text=" ".join(instructions),
            source_trace=candidate.source_trace[:4000],
            source_stage=candidate.source_stage,
            confidence=candidate.confidence,
            tags=candidate.tags,
            review_history=[{
                "review_id": review.review_id,
                "decision": review.decision.value,
                "rationale": review.rationale,
                "safety_note": review.safety_note,
                "candidate_id": candidate.candidate_id,
            }],
        )
        skills.append(skill)
        self._persist_nl_skills(skills)
        self._write_skill_artifact(skill, review)
        return SkillLifecycleResult(candidate=candidate, review=review, skill=skill, action="add")

    def register_nl_skill(
        self,
        *,
        domain: SkillDomain | str,
        trigger_pattern: str,
        rule_text: str,
        source_trace: str,
        confidence: float = 0.55,
        tags: list[str] | None = None,
    ) -> NaturalLanguageSkill | None:
        candidate = self.extract_skill_candidate(
            domain=domain,
            trigger_pattern=trigger_pattern,
            source_trace=source_trace,
            tags=tags,
            confidence=confidence,
            instructions=[rule_text],
        )
        if candidate is None:
            return None
        return self.maintain_skill(candidate).skill

    def synthesize_nl_skill(
        self,
        *,
        domain: SkillDomain | str,
        trigger_pattern: str,
        source_trace: str,
        rule_text: str | None = None,
        confidence: float = 0.55,
        tags: list[str] | None = None,
        source_stage: str = "",
    ) -> SkillLifecycleResult | None:
        candidate = self.extract_skill_candidate(
            domain=domain,
            trigger_pattern=trigger_pattern,
            source_trace=source_trace,
            tags=tags,
            confidence=confidence,
            instructions=[rule_text] if (rule_text or "").strip() else None,
            source_stage=source_stage,
        )
        if candidate is None:
            return None
        return self.maintain_skill(candidate)

    def import_workspace_artifacts_for_skill_extraction(self, workspace_path: str | Path) -> list[SkillLifecycleResult]:
        workspace = Path(workspace_path)
        if not workspace.exists() or not self.enabled:
            return []
        tasks: list[tuple[SkillDomain, str, Path]] = [
            (SkillDomain.PLANNING, "planning_blueprint", workspace / "plans" / "experiment_blueprint.json"),
            (SkillDomain.EXPERIMENT, "experiment_strategy", workspace / "logs" / "experiment_strategy_summary.json"),
            (SkillDomain.EXPERIMENT, "failed_direction", workspace / "logs" / "failed_direction_summary.json"),
            (SkillDomain.REVIEW, "review_feedback", workspace / "drafts" / "review_output.json"),
            (SkillDomain.LITERATURE, "ideation_output", workspace / "papers" / "ideation_output.json"),
        ]
        results: list[SkillLifecycleResult] = []
        for domain, trigger_pattern, path in tasks:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                trace = json.dumps(payload, ensure_ascii=False)
            except Exception:
                continue
            lifecycle = self.synthesize_nl_skill(
                domain=domain,
                trigger_pattern=trigger_pattern,
                source_trace=trace,
                confidence=0.62,
                tags=[workspace.name, trigger_pattern, domain.value],
                source_stage=trigger_pattern,
            )
            if lifecycle is not None:
                results.append(lifecycle)
        return results

    def match_nl_skills(
        self,
        domain: SkillDomain | str,
        *,
        topic: str = "",
        text: str = "",
        tags: list[str] | None = None,
        top_k: int = 5,
    ) -> list[NaturalLanguageSkill]:
        if not self.enabled:
            return []
        domain = SkillDomain(domain)
        query_tokens = self._tokenize(" ".join([topic, text, " ".join(self._normalize_tags(tags))]))
        query_tags = set(self._normalize_tags(tags))
        scored: list[tuple[float, NaturalLanguageSkill]] = []
        for skill in self._load_all_nl_skills():
            if skill.domain != domain:
                continue
            overlap = len(self._skill_tokens(skill) & query_tokens)
            tag_overlap = len(set(skill.tags) & query_tags)
            score = skill.confidence * 2.0 + overlap * 0.45 + tag_overlap * 0.8 + min(skill.usage_count, 5) * 0.1
            if score >= 0.85:
                scored.append((score, skill))
        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [skill for _, skill in scored[:max(1, top_k)]]

    def render_nl_context(
        self,
        domain: SkillDomain | str,
        *,
        topic: str = "",
        text: str = "",
        tags: list[str] | None = None,
        top_k: int = 5,
    ) -> str:
        matches = self.match_nl_skills(domain, topic=topic, text=text, tags=tags, top_k=top_k)
        if not matches:
            return ""
        lines = []
        for skill in matches:
            instructions = "; ".join(skill.instructions[:3]) if skill.instructions else skill.rule_text
            lines.append(f"- [{skill.domain.value}/{skill.version}] {skill.name or skill.stable_id}: {instructions}")
        return (
            "\n\n=== EVOLVED RESEARCH SKILLS ===\n"
            "Apply these reusable behavioral rules distilled from prior failures, retries, reviews, and artifact maintenance.\n"
            + "\n".join(lines)
            + "\n=== END EVOLVED RESEARCH SKILLS ===\n"
        )

    def register_script_skill(
        self,
        *,
        category: ScriptSkillCategory | str,
        name: str,
        description: str,
        script_path: str,
        input_contract: str = "",
        output_contract: str = "",
        safe_to_autorun: bool = False,
        test_status: ScriptTestStatus | str = ScriptTestStatus.PROPOSED,
        scope: str = "project",
        tags: list[str] | None = None,
    ) -> ScriptSkill | None:
        if not self.enabled:
            return None
        category = ScriptSkillCategory(category)
        if category.value not in _ALLOWED_SCRIPT_CATEGORIES:
            raise ValueError(f"Unsupported script skill category: {category}")
        test_status = ScriptTestStatus(test_status)
        tags = self._normalize_tags(tags)
        script_id = self._skill_id("pyskill", category.value, name.strip().lower(), script_path)
        scripts = [item for item in self._load_models(self.script_file, ScriptSkill) if isinstance(item, ScriptSkill)]
        for index, skill in enumerate(scripts):
            if skill.skill_id != script_id:
                continue
            updated = skill.model_copy(update={
                "description": description,
                "input_contract": input_contract or skill.input_contract,
                "output_contract": output_contract or skill.output_contract,
                "safe_to_autorun": bool(safe_to_autorun and test_status == ScriptTestStatus.PASSED),
                "test_status": test_status,
                "tags": self._normalize_tags(skill.tags + tags),
                "scope": scope or skill.scope,
            })
            scripts[index] = updated
            self._save_models(self.script_file, scripts)
            return updated
        script_skill = ScriptSkill(
            skill_id=script_id,
            category=category,
            name=name.strip(),
            description=description.strip(),
            input_contract=input_contract,
            output_contract=output_contract,
            safe_to_autorun=bool(safe_to_autorun and test_status == ScriptTestStatus.PASSED),
            test_status=test_status,
            script_path=script_path,
            scope=scope,
            tags=tags,
        )
        scripts.append(script_skill)
        self._save_models(self.script_file, scripts)
        return script_skill

    def match_script_skills(
        self,
        domain: SkillDomain | str,
        *,
        tags: list[str] | None = None,
        top_k: int = 3,
        autorun_policy: str = "safe_only",
    ) -> list[ScriptSkill]:
        if not self.enabled:
            return []
        domain = SkillDomain(domain)
        domain_to_categories = {
            SkillDomain.EXPERIMENT: {ScriptSkillCategory.ENVIRONMENT_SETUP},
            SkillDomain.LITERATURE: {ScriptSkillCategory.LITERATURE_TRACKING},
            SkillDomain.WRITING: {ScriptSkillCategory.FIGURE_FORMATTING},
            SkillDomain.REVIEW: {ScriptSkillCategory.FIGURE_FORMATTING},
        }
        allowed = domain_to_categories.get(domain, set())
        query_tags = set(self._normalize_tags(tags))
        matches: list[tuple[float, ScriptSkill]] = []
        for skill in self._load_models(self.script_file, ScriptSkill):
            if not isinstance(skill, ScriptSkill) or skill.category not in allowed:
                continue
            if skill.test_status != ScriptTestStatus.PASSED:
                continue
            if autorun_policy == "off" and skill.safe_to_autorun:
                continue
            score = 1.0 + len(set(skill.tags) & query_tags) * 0.8 + (0.4 if skill.safe_to_autorun else 0.0)
            matches.append((score, skill))
        matches.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [skill for _, skill in matches[:max(1, top_k)]]

    def render_script_context(
        self,
        domain: SkillDomain | str,
        *,
        tags: list[str] | None = None,
        top_k: int = 3,
        autorun_policy: str = "safe_only",
    ) -> str:
        matches = self.match_script_skills(domain, tags=tags, top_k=top_k, autorun_policy=autorun_policy)
        if not matches:
            return ""
        lines = []
        for skill in matches:
            mode = "autorun" if skill.safe_to_autorun and autorun_policy != "off" else "recommended"
            lines.append(f"- [{skill.category.value}/{mode}] {skill.name}: {skill.description} ({skill.script_path})")
        return (
            "\n\n=== REGISTERED PYTHON SCRIPT SKILLS ===\n"
            "Prefer these tested low-risk automation hooks before asking the model to recreate repetitive setup or formatting work.\n"
            + "\n".join(lines)
            + "\n=== END REGISTERED PYTHON SCRIPT SKILLS ===\n"
        )

    def execute_script_skill(
        self,
        skill: ScriptSkill,
        *,
        args: list[str] | None = None,
        cwd: Path | None = None,
        autorun_policy: str = "safe_only",
    ) -> subprocess.CompletedProcess[str]:
        if skill.category.value not in _ALLOWED_SCRIPT_CATEGORIES:
            raise ValueError(f"Script skill category {skill.category.value} is not whitelisted")
        if skill.test_status != ScriptTestStatus.PASSED:
            raise ValueError(f"Script skill {skill.name} has not passed validation")
        if autorun_policy == "off" or (autorun_policy == "safe_only" and not skill.safe_to_autorun):
            raise ValueError(f"Autorun policy {autorun_policy} does not allow executing {skill.name}")
        command = ["python", skill.script_path, *(args or [])]
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
