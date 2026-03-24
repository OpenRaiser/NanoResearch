"""Evolved natural-language and script skill stores."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path

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


class ScriptSkillCategory(str, Enum):
    ENVIRONMENT_SETUP = "environment_setup"
    LITERATURE_TRACKING = "literature_tracking"
    FIGURE_FORMATTING = "figure_formatting"


class ScriptTestStatus(str, Enum):
    PROPOSED = "proposed"
    PASSED = "passed"
    FAILED = "failed"


class NaturalLanguageSkill(BaseModel):
    skill_id: str
    skill_type: str = "natural_language"
    domain: SkillDomain
    trigger_pattern: str
    rule_text: str
    source_trace: str
    confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)
    last_applied_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)


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


class SkillEvolutionStore:
    """Persistent adaptive skill registry under ``~/.nanoresearch/skills``."""

    def __init__(self, root: Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.root = root or (Path.home() / ".nanoresearch" / "skills")
        self.nl_file = self.root / "natural_language.json"
        self.script_file = self.root / "script_registry.json"
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _skill_id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
        return f"{prefix}-{digest[:12]}"

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
        if not self.enabled:
            return None
        domain = SkillDomain(domain)
        rule_text = (rule_text or "").strip()
        if not rule_text:
            return None
        trigger_pattern = (trigger_pattern or "adaptive_rule").strip()
        tags = self._normalize_tags(tags)
        skill_id = self._skill_id("nlskill", domain.value, trigger_pattern, rule_text[:300])
        skills = [item for item in self._load_models(self.nl_file, NaturalLanguageSkill) if isinstance(item, NaturalLanguageSkill)]
        for index, skill in enumerate(skills):
            if skill.skill_id != skill_id:
                continue
            updated = skill.model_copy(update={
                "confidence": max(skill.confidence, min(max(confidence, 0.0), 1.0)),
                "tags": self._normalize_tags(skill.tags + tags),
            })
            skills[index] = updated
            self._save_models(self.nl_file, skills)
            return updated
        skill = NaturalLanguageSkill(
            skill_id=skill_id,
            domain=domain,
            trigger_pattern=trigger_pattern,
            rule_text=rule_text,
            source_trace=(source_trace or "")[:4000],
            confidence=min(max(confidence, 0.0), 1.0),
            tags=tags,
        )
        skills.append(skill)
        self._save_models(self.nl_file, skills)
        return skill

    def synthesize_nl_skill(
        self,
        *,
        domain: SkillDomain | str,
        trigger_pattern: str,
        source_trace: str,
        rule_text: str | None = None,
        confidence: float = 0.55,
        tags: list[str] | None = None,
    ) -> NaturalLanguageSkill | None:
        synthesized = (rule_text or "").strip() or self._heuristic_rule(domain, source_trace)
        if not synthesized:
            return None
        return self.register_nl_skill(
            domain=domain,
            trigger_pattern=trigger_pattern,
            rule_text=synthesized,
            source_trace=source_trace,
            confidence=confidence,
            tags=tags,
        )

    def _heuristic_rule(self, domain: SkillDomain | str, trace: str) -> str:
        domain = SkillDomain(domain)
        trace_lower = (trace or "").lower()
        if any(token in trace_lower for token in ("seed", "variance", "std", "standard deviation")):
            return "Report aggregate metrics with at least three random seeds and include mean plus standard deviation before claiming improvements."
        if "ablation" in trace_lower:
            return "When planning or reviewing experiments, require one ablation per core module so each claimed component has an isolated removal study."
        if any(token in trace_lower for token in ("oom", "cuda out of memory", "memoryerror", "out of memory")):
            return "Before launching full experiments, run a reduced-scale dry run to validate memory usage, convergence direction, and logging artifacts."
        if any(token in trace_lower for token in ("citation", "bibtex", "cite")) and domain in {SkillDomain.WRITING, SkillDomain.REVIEW}:
            return "Use only citation keys that exist in the provided bibliography and tie each strong factual claim to an available source before revision."
        if any(token in trace_lower for token in ("environment", "dependency", "module not found", "importerror")):
            return "Validate the environment with a lightweight preflight dependency check before running the full experiment loop or submission pipeline."
        if domain == SkillDomain.LITERATURE:
            return "Broaden literature search queries with synonyms, adjacent task names, and review papers before committing to a gap or novelty claim."
        if domain == SkillDomain.PLANNING:
            return "Convert reviewer or retry failures into explicit planning constraints so the next blueprint bakes the fix in up front."
        if domain == SkillDomain.WRITING:
            return "Structure each academic section as prior work or context, the concrete limitation, and then the project-specific distinction or implication."
        if domain == SkillDomain.REVIEW:
            return "If a section scores poorly, preserve its strongest claims but require each revision request to name the concrete problem, the impact, and the exact fix."
        return "Turn repeated failure patterns into explicit checklists before the next attempt so the same mistake is not repeated."

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
        for skill in self._load_models(self.nl_file, NaturalLanguageSkill):
            if not isinstance(skill, NaturalLanguageSkill) or skill.domain != domain:
                continue
            overlap = len(self._tokenize(skill.rule_text + " " + skill.trigger_pattern) & query_tokens)
            tag_overlap = len(set(skill.tags) & query_tags)
            score = skill.confidence * 2.0 + overlap * 0.45 + tag_overlap * 0.8 + min(skill.usage_count, 5) * 0.1
            if score >= 0.85:
                scored.append((score, skill))
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
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
        lines = [f"- [{skill.domain.value}] {skill.rule_text}" for skill in matches]
        return (
            "\n\n=== EVOLVED RESEARCH SKILLS ===\n"
            "Apply these reusable behavioral rules distilled from prior failures, retries, and reviews.\n"
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
