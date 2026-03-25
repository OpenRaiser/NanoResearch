"""K-Dense scientific skill matching and context extraction.

Provides keyword-based matching of experiment blueprints / writing tasks
against the K-Dense claude-scientific-skills and claude-scientific-writer
skill repositories, then extracts high-value sections (Quick Start, Common
Pitfalls, Key Parameters) to inject as domain-expert context into agent
prompts.

Gracefully degrades: if the skill directories are missing or empty, all
public methods return empty/no-op results and the agents run unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from nanoresearch.evolution.skills import SkillDomain, SkillEvolutionStore

logger = logging.getLogger(__name__)

# ── Limits ────────────────────────────────────────────────────────────────
MAX_SKILLS = 5  # max matched skills per query
MAX_CHARS_PER_SKILL = 6000  # truncate each SKILL.md extract
MAX_TOTAL_CHARS = 24000  # hard cap on total injected context
MAX_ASSET_LINES = 80  # lines of each assets/*.py template to include
MIN_MATCH_SCORE = 2  # minimum keyword overlap to consider a match

# High-value section headings to extract from SKILL.md
_EXTRACT_HEADINGS = {
    "quick start",
    "common pitfalls",
    "key parameters",
    "best practices",
    "core capabilities",
    "workflow",
    "when to use this skill",
    "configuration",
    "parameters",
    "troubleshooting",
    "important notes",
}


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class SkillEntry:
    """Index entry for a single K-Dense skill."""

    name: str
    path: Path  # path to the directory containing SKILL.md
    keywords: set[str] = field(default_factory=set)
    description: str = ""
    asset_paths: list[Path] = field(default_factory=list)


@dataclass
class SkillContext:
    """Extracted skill context ready for prompt injection."""

    matched_skills: list[str] = field(default_factory=list)
    phase1_context: str = ""  # domain knowledge for project planning
    phase2_context: str = ""  # best-practices for file generation


# ── Keyword extraction helpers ────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z][a-z0-9_\-]{2,}")
_STOPWORDS = frozenset(
    "the and for are but not you all any can had her was one our out day get has "
    "him how its may new now old see way who did does from have into just like make "
    "many over such take than them very when will with that this these those which "
    "would been each more most some what about after could every first other their "
    "there where before should through while being during without between another "
    "because against during before after under above below using used uses also "
    "need needs must shall should will would might could when then than use "
    "skill file path description overview section example template code data "
    "model output input method function class based apply following".split()
)


def _extract_keywords(text: str, max_lines: int = 80) -> set[str]:
    """Extract meaningful keywords from text (first *max_lines* lines)."""
    lines = text.split("\n")[:max_lines]
    text_block = " ".join(lines).lower()
    words = set(_WORD_RE.findall(text_block))
    return words - _STOPWORDS


def _extract_yaml_frontmatter(text: str) -> tuple[str, str]:
    """Return (name, description) from YAML frontmatter, or ("", "")."""
    if not text.startswith("---"):
        return "", ""
    end = text.find("---", 3)
    if end == -1:
        return "", ""
    fm = text[3:end]
    name = desc = ""
    for line in fm.split("\n"):
        line = line.strip()
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("description:"):
            desc = line.split(":", 1)[1].strip().strip('"').strip("'")
    return name, desc


# ── Section extraction ────────────────────────────────────────────────────


def _extract_high_value_sections(md_text: str, max_chars: int = MAX_CHARS_PER_SKILL) -> str:
    """Extract high-value sections from a SKILL.md file.

    Keeps sections whose heading (lowered) matches *_EXTRACT_HEADINGS*.
    Returns concatenated section content, truncated to *max_chars*.
    """
    lines = md_text.split("\n")
    chunks: list[str] = []
    current_heading = ""
    current_lines: list[str] = []
    keep = False

    def _flush():
        nonlocal keep, current_lines
        if keep and current_lines:
            chunks.append("\n".join(current_lines))
        current_lines = []

    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            _flush()
            heading_text = line.lstrip("#").strip().lower()
            current_heading = heading_text
            keep = any(h in heading_text for h in _EXTRACT_HEADINGS)
            if keep:
                current_lines.append(line)
        else:
            if keep:
                current_lines.append(line)
    _flush()

    result = "\n\n".join(chunks)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... [truncated]"
    return result


# ── Main class ────────────────────────────────────────────────────────────


class SkillMatcher:
    """Index, match, and extract K-Dense scientific skills."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._index: list[SkillEntry] = []
        if skills_dir and skills_dir.is_dir():
            self._build_index(skills_dir)
            logger.info("SkillMatcher indexed %d skills from %s", len(self._index), skills_dir)
        else:
            logger.debug("SkillMatcher: no skills directory — running without skill injection")

    @property
    def skill_count(self) -> int:
        return len(self._index)

    # ── Indexing ──────────────────────────────────────────────────────────

    def _build_index(self, root: Path) -> None:
        """Walk *root* and index every directory containing a SKILL.md."""
        for skill_md in sorted(root.rglob("SKILL.md")):
            try:
                text = skill_md.read_text(errors="replace")
            except OSError:
                continue
            name, desc = _extract_yaml_frontmatter(text)
            if not name:
                name = skill_md.parent.name
            kw = _extract_keywords(text)
            # Also add the skill name tokens and description tokens
            kw |= set(_WORD_RE.findall(name.lower()))
            if desc:
                kw |= set(_WORD_RE.findall(desc.lower()))
            # Discover asset files (*.py templates)
            assets_dir = skill_md.parent / "assets"
            asset_paths = sorted(assets_dir.glob("*.py")) if assets_dir.is_dir() else []
            self._index.append(
                SkillEntry(
                    name=name,
                    path=skill_md.parent,
                    keywords=kw,
                    description=desc,
                    asset_paths=asset_paths,
                )
            )

    # ── Matching ──────────────────────────────────────────────────────────

    def match(self, blueprint: dict) -> list[tuple[SkillEntry, int]]:
        """Match an experiment blueprint against indexed skills.

        Returns up to *MAX_SKILLS* entries sorted by descending score
        (keyword overlap size). Only entries with score >= MIN_MATCH_SCORE
        are returned.
        """
        if not self._index:
            return []

        # Flatten blueprint into search terms
        search_tokens = self._blueprint_tokens(blueprint)
        if not search_tokens:
            return []

        scored: list[tuple[SkillEntry, int]] = []
        for entry in self._index:
            score = len(entry.keywords & search_tokens)
            if score >= MIN_MATCH_SCORE:
                scored.append((entry, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:MAX_SKILLS]

    def match_writing_skills(
        self, topic: str = "", template_format: str = ""
    ) -> list[tuple[SkillEntry, int]]:
        """Match writing-related skills by topic and format.

        Targeted matching: boosts venue-templates, scientific-writing,
        citation-management skills when their names match the query.
        """
        if not self._index:
            return []

        search_tokens = _extract_keywords(topic, max_lines=200)
        if template_format:
            search_tokens |= _extract_keywords(template_format, max_lines=10)
            # Add venue name variants
            fmt_lower = template_format.lower()
            search_tokens |= {fmt_lower, fmt_lower.replace("-", ""), fmt_lower.replace("_", "")}

        # Priority skill names that are always relevant for writing
        priority_names = {"scientific-writing", "venue-templates", "citation-management"}

        scored: list[tuple[SkillEntry, int]] = []
        for entry in self._index:
            score = len(entry.keywords & search_tokens)
            # Boost priority writing skills
            if entry.name in priority_names:
                score += 5
            if score >= MIN_MATCH_SCORE:
                scored.append((entry, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:MAX_SKILLS]

    # ── Context extraction ────────────────────────────────────────────────

    def extract_context(self, matches: Sequence[tuple[SkillEntry, int]]) -> SkillContext:
        """Extract prompt-injection context from matched skills.

        Returns a *SkillContext* with:
        - phase1_context: domain knowledge (high-value sections)
        - phase2_context: best practices + asset templates
        """
        if not matches:
            return SkillContext()

        ctx = SkillContext(matched_skills=[e.name for e, _ in matches])
        phase1_parts: list[str] = []
        phase2_parts: list[str] = []
        total_chars = 0

        for entry, score in matches:
            if total_chars >= MAX_TOTAL_CHARS:
                break

            skill_md = entry.path / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                raw = skill_md.read_text(errors="replace")
            except OSError:
                continue

            sections = _extract_high_value_sections(raw, MAX_CHARS_PER_SKILL)
            if not sections:
                continue

            block = f"### Skill: {entry.name}\n{sections}"
            total_chars += len(block)
            phase1_parts.append(block)

            # Collect asset snippets for phase 2
            asset_snippets: list[str] = []
            for ap in entry.asset_paths[:3]:
                try:
                    lines = ap.read_text(errors="replace").split("\n")[:MAX_ASSET_LINES]
                    asset_snippets.append(f"# {ap.name}\n" + "\n".join(lines))
                except OSError:
                    continue
            if asset_snippets:
                phase2_parts.append(
                    f"### Templates from {entry.name}:\n" + "\n\n".join(asset_snippets)
                )

        if phase1_parts:
            ctx.phase1_context = (
                "\n\n=== DOMAIN EXPERT KNOWLEDGE (K-Dense Skills) ===\n"
                "Use the following domain-specific knowledge as reference when "
                "designing the project structure and implementation approach.\n\n"
                + "\n\n".join(phase1_parts)
                + "\n=== END DOMAIN KNOWLEDGE ===\n"
            )

        if phase2_parts:
            ctx.phase2_context = (
                "\n\n=== BEST PRACTICES & TEMPLATES (K-Dense Skills) ===\n"
                "Reference these code templates and best practices when "
                "generating implementation files.\n\n"
                + "\n\n".join(phase2_parts)
                + "\n=== END BEST PRACTICES ===\n"
            )

        return ctx

    def extract_writing_context(
        self, matches: Sequence[tuple[SkillEntry, int]]
    ) -> str:
        """Extract writing-specific context for the WritingAgent.

        Returns a single block of text combining venue formatting rules,
        scientific writing guidelines, and citation management specs.
        """
        if not matches:
            return ""

        parts: list[str] = []
        total_chars = 0

        for entry, score in matches:
            if total_chars >= MAX_TOTAL_CHARS:
                break

            skill_md = entry.path / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                raw = skill_md.read_text(errors="replace")
            except OSError:
                continue

            sections = _extract_high_value_sections(raw, MAX_CHARS_PER_SKILL)
            if not sections:
                continue

            block = f"### Writing Guidance: {entry.name}\n{sections}"
            total_chars += len(block)
            parts.append(block)

        if not parts:
            return ""

        return (
            "\n\n=== ACADEMIC WRITING GUIDELINES (K-Dense Skills) ===\n"
            "Use the following writing guidelines and venue-specific "
            "formatting requirements as reference.\n\n"
            + "\n\n".join(parts)
            + "\n=== END WRITING GUIDELINES ===\n"
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _blueprint_tokens(blueprint: dict) -> set[str]:
        """Flatten an experiment blueprint dict into a set of search tokens."""
        parts: list[str] = []

        # Title
        parts.append(blueprint.get("title", ""))

        # Method name + description + components
        method = blueprint.get("proposed_method", {})
        if isinstance(method, dict):
            parts.append(method.get("name", ""))
            parts.append(method.get("description", ""))
            parts.extend(method.get("key_components", []))
        elif isinstance(method, str):
            parts.append(method)

        # Datasets
        for ds in blueprint.get("datasets", []):
            if isinstance(ds, dict):
                parts.append(ds.get("name", ""))
                parts.append(ds.get("description", ""))
            elif isinstance(ds, str):
                parts.append(ds)

        # Metrics
        for m in blueprint.get("metrics", []):
            if isinstance(m, dict):
                parts.append(m.get("name", ""))
            elif isinstance(m, str):
                parts.append(m)

        # Baselines
        for b in blueprint.get("baselines", []):
            if isinstance(b, dict):
                parts.append(b.get("name", ""))
                parts.append(b.get("description", ""))
            elif isinstance(b, str):
                parts.append(b)

        combined = " ".join(parts).lower()
        return _extract_keywords(combined, max_lines=10000)



@dataclass
class UnifiedSkillContext:
    """Combined static and evolved skill context for prompt injection."""

    matched_skills: list[str] = field(default_factory=list)
    static_context: str = ""
    evolved_context: str = ""
    script_context: str = ""

    @property
    def combined_context(self) -> str:
        parts = [part for part in (self.static_context, self.evolved_context, self.script_context) if part]
        return "\n\n".join(parts)


class UnifiedSkillMatcher:
    """Compose K-Dense static skills with evolved NL/script skills."""

    _TASK_TO_DOMAIN = {
        "literature": SkillDomain.LITERATURE,
        "ideation": SkillDomain.LITERATURE,
        "planning": SkillDomain.PLANNING,
        "experiment": SkillDomain.EXPERIMENT,
        "coding": SkillDomain.CODING,
        "writing": SkillDomain.WRITING,
        "review": SkillDomain.REVIEW,
    }

    def __init__(self, skills_dir: Path | None = None, *, evolution_store: SkillEvolutionStore | None = None, retrieval_top_k: int = 5, autorun_policy: str = "safe_only") -> None:
        self._static = SkillMatcher(skills_dir)
        self.evolution_store = evolution_store or SkillEvolutionStore(enabled=True)
        self._retrieval_top_k = max(1, retrieval_top_k)
        self._autorun_policy = autorun_policy

    def _domain_for_task(self, task_type: str) -> SkillDomain:
        return self._TASK_TO_DOMAIN.get(task_type, SkillDomain.PLANNING)

    def build_context(self, task_type: str, *, topic: str = "", blueprint: dict | None = None, text: str = "", tags: list[str] | None = None, template_format: str = "") -> UnifiedSkillContext:
        domain = self._domain_for_task(task_type)
        ctx = UnifiedSkillContext()

        if task_type in {"planning", "experiment", "coding"} and blueprint:
            matches = self._static.match(blueprint)
            static_ctx = self._static.extract_context(matches)
            ctx.static_context = "\n\n".join(part for part in (static_ctx.phase1_context, static_ctx.phase2_context) if part)
            ctx.matched_skills.extend(static_ctx.matched_skills)
        elif task_type in {"writing", "review"}:
            matches = self._static.match_writing_skills(topic=topic, template_format=template_format)
            ctx.static_context = self._static.extract_writing_context(matches)
            ctx.matched_skills.extend([entry.name for entry, _ in matches])

        text_payload = text
        if blueprint and not text_payload:
            try:
                text_payload = json.dumps(blueprint, ensure_ascii=False)
            except TypeError:
                text_payload = str(blueprint)

        ctx.evolved_context = self.evolution_store.render_nl_context(domain, topic=topic, text=text_payload, tags=tags, top_k=self._retrieval_top_k)
        ctx.script_context = self.evolution_store.render_script_context(domain, tags=tags, top_k=min(3, self._retrieval_top_k), autorun_policy=self._autorun_policy)

        nl_matches = self.evolution_store.match_nl_skills(domain, topic=topic, text=text_payload, tags=tags, top_k=self._retrieval_top_k)
        ctx.matched_skills.extend((skill.stable_id or skill.skill_id) for skill in nl_matches)
        script_matches = self.evolution_store.match_script_skills(domain, tags=tags, top_k=min(3, self._retrieval_top_k), autorun_policy=self._autorun_policy)
        ctx.matched_skills.extend(skill.name for skill in script_matches)
        return ctx
