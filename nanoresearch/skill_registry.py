"""Dynamic skill discovery, registration, and management.

Inspired by EvoScientist's SKILL.md format and skill marketplace.

Skills are directories containing a SKILL.md with YAML frontmatter:
    ---
    name: my-skill
    description: What this skill does
    tags: [figure, latex, analysis]
    version: 1.0.0
    ---
    # Skill Title
    ## Usage ...

Discovery locations:
    1. Built-in: nanoresearch/skills/ (shipped with package)
    2. User: ~/.nanoresearch/skills/ (user-installed)
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"
_USER_SKILLS_DIR = Path.home() / ".nanoresearch" / "skills"

# YAML frontmatter regex
_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


@dataclass
class SkillInfo:
    """Metadata for a discovered skill."""
    name: str
    description: str
    path: Path
    source: str  # "builtin" or "user"
    tags: list[str] = field(default_factory=list)
    version: str = "0.0.0"
    content: str = ""  # full SKILL.md content (without frontmatter)


class SkillRegistry:
    """Discover, register, and manage NanoResearch skills."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillInfo] = {}
        self._scanned = False

    def discover(self) -> list[SkillInfo]:
        """Scan all skill directories and return discovered skills."""
        self._skills.clear()

        # Scan built-in skills
        if _BUILTIN_SKILLS_DIR.is_dir():
            self._scan_dir(_BUILTIN_SKILLS_DIR, source="builtin")

        # Scan user-installed skills
        if _USER_SKILLS_DIR.is_dir():
            self._scan_dir(_USER_SKILLS_DIR, source="user")

        self._scanned = True
        return list(self._skills.values())

    def list_skills(self, tag: str | None = None) -> list[SkillInfo]:
        """List all skills, optionally filtered by tag."""
        if not self._scanned:
            self.discover()
        skills = list(self._skills.values())
        if tag:
            tag_lower = tag.lower()
            skills = [s for s in skills if tag_lower in [t.lower() for t in s.tags]]
        return skills

    def get_skill(self, name: str) -> SkillInfo | None:
        """Get a skill by name."""
        if not self._scanned:
            self.discover()
        return self._skills.get(name)

    def install_from_github(self, repo_url: str, skill_path: str = "") -> dict[str, Any]:
        """Install a skill from a GitHub repository.

        Args:
            repo_url: GitHub URL (e.g., https://github.com/owner/repo)
            skill_path: Optional subdirectory path within the repo

        Returns:
            {"success": bool, "message": str, "installed": [names]}
        """
        _USER_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        # Normalize URL
        repo_url = repo_url.rstrip("/")
        if "/tree/" in repo_url:
            # Extract path from tree URL
            parts = repo_url.split("/tree/")
            repo_url = parts[0]
            branch_and_path = parts[1]
            if "/" in branch_and_path:
                skill_path = branch_and_path.split("/", 1)[1]

        try:
            # Shallow clone to temp dir
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / "repo"
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(tmp_path)],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    return {"success": False, "message": f"Git clone failed: {result.stderr[:200]}"}

                # Find skill dirs
                search_root = tmp_path / skill_path if skill_path else tmp_path
                installed = []

                for skill_md in search_root.rglob("SKILL.md"):
                    skill_dir = skill_md.parent
                    info = self._parse_skill_md(skill_md, source="user")
                    if info:
                        dst = _USER_SKILLS_DIR / info.name
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(skill_dir, dst)
                        self._skills[info.name] = SkillInfo(
                            name=info.name, description=info.description,
                            path=dst, source="user", tags=info.tags,
                            version=info.version, content=info.content,
                        )
                        installed.append(info.name)
                        logger.info("Installed skill: %s -> %s", info.name, dst)

                if not installed:
                    return {"success": False, "message": "No SKILL.md found in repository"}
                return {"success": True, "message": f"Installed {len(installed)} skill(s)", "installed": installed}

        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Git clone timed out (60s)"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def uninstall(self, name: str) -> bool:
        """Uninstall a user-installed skill."""
        skill = self._skills.get(name)
        if not skill or skill.source != "user":
            return False
        try:
            if skill.path.exists():
                shutil.rmtree(skill.path)
            self._skills.pop(name, None)
            logger.info("Uninstalled skill: %s", name)
            return True
        except OSError as e:
            logger.warning("Failed to uninstall %s: %s", name, e)
            return False

    # ─── internal ───

    def _scan_dir(self, root: Path, source: str) -> None:
        """Scan a directory for skills (1-2 levels deep)."""
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            skill_md = child / "SKILL.md"
            if skill_md.is_file():
                info = self._parse_skill_md(skill_md, source)
                if info:
                    self._skills[info.name] = info
            else:
                # Level 2: grandchildren
                for grandchild in sorted(child.iterdir()):
                    if grandchild.is_dir():
                        gm = grandchild / "SKILL.md"
                        if gm.is_file():
                            info = self._parse_skill_md(gm, source)
                            if info:
                                self._skills[info.name] = info

    @staticmethod
    def _parse_skill_md(path: Path, source: str) -> SkillInfo | None:
        """Parse a SKILL.md file and extract metadata."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None

        # Extract YAML frontmatter
        match = _FRONTMATTER_RE.match(text)
        if not match:
            # No frontmatter — use directory name
            return SkillInfo(
                name=path.parent.name,
                description="",
                path=path.parent,
                source=source,
                content=text,
            )

        frontmatter_str = match.group(1)
        content = text[match.end():]

        # Simple YAML parsing (avoid pyyaml dependency for frontmatter)
        meta: dict[str, Any] = {}
        for line in frontmatter_str.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if val.startswith("[") and val.endswith("]"):
                    # Simple list parsing
                    items = val[1:-1].split(",")
                    meta[key] = [item.strip().strip("'\"") for item in items if item.strip()]
                else:
                    meta[key] = val.strip("'\"")

        name = meta.get("name", path.parent.name)
        return SkillInfo(
            name=name,
            description=meta.get("description", ""),
            path=path.parent,
            source=source,
            tags=meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
            version=meta.get("version", "0.0.0"),
            content=content.strip(),
        )
