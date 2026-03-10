"""Workspace directory management and manifest CRUD."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from nanoresearch.schemas.manifest import (
    ArtifactRecord,
    DEEP_ONLY_STAGES,
    PipelineMode,
    PipelineStage,
    StageRecord,
    WorkspaceManifest,
    processing_stages_for_mode,
)

logger = logging.getLogger(__name__)


_DEFAULT_ROOT = Path.home() / ".nanobot" / "workspace" / "research"

WORKSPACE_DIRS = ["papers", "plans", "drafts", "figures", "logs", "code"]


class Workspace:
    """Manages a single research session workspace on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._manifest_path = path / "manifest.json"
        self._manifest_cache: WorkspaceManifest | None = None

    # ---- creation --------------------------------------------------------

    @classmethod
    def create(
        cls,
        topic: str,
        config_snapshot: dict | None = None,
        root: Path = _DEFAULT_ROOT,
        session_id: str | None = None,
        pipeline_mode: PipelineMode = PipelineMode.STANDARD,
    ) -> "Workspace":
        sid = session_id or uuid.uuid4().hex[:12]
        ws_path = root / sid
        ws_path.mkdir(parents=True, exist_ok=True)
        for d in WORKSPACE_DIRS:
            (ws_path / d).mkdir(exist_ok=True)

        relevant_stages = [
            PipelineStage.INIT,
            *processing_stages_for_mode(pipeline_mode),
        ]

        manifest = WorkspaceManifest(
            session_id=sid,
            topic=topic,
            pipeline_mode=pipeline_mode,
            current_stage=PipelineStage.INIT,
            stages={
                stage.value: StageRecord(stage=stage)
                for stage in relevant_stages
            },
            config_snapshot=config_snapshot or {},
        )
        ws = cls(ws_path)
        ws._write_manifest(manifest)
        return ws

    @classmethod
    def load(cls, path: Path) -> "Workspace":
        if not path.exists():
            raise FileNotFoundError(f"Workspace directory not found: {path}")
        ws = cls(path)
        ws.manifest  # validate readable
        return ws

    # ---- manifest --------------------------------------------------------

    @property
    def manifest(self) -> WorkspaceManifest:
        if self._manifest_cache is not None:
            return self._manifest_cache
        if not self._manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest file not found: {self._manifest_path}"
            )
        try:
            raw = self._manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read manifest file {self._manifest_path}: {exc}"
            ) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Manifest file contains invalid JSON: {exc}"
            ) from exc
        data, normalized = self._normalize_manifest_data(data)
        manifest = WorkspaceManifest.model_validate(data)
        self._manifest_cache = manifest
        if normalized:
            self._write_manifest(manifest)
        return self._manifest_cache

    @staticmethod
    def _normalize_manifest_data(data: dict) -> tuple[dict, bool]:
        """Repair legacy manifests in-memory before validation."""

        if not isinstance(data, dict):
            return data, False

        normalized = False
        stages = data.get("stages")
        if not isinstance(stages, dict):
            return data, False

        deep_stage_names = {stage.value for stage in DEEP_ONLY_STAGES}
        inferred_deep = False
        current_stage = str(data.get("current_stage", ""))

        for stage_key, record in stages.items():
            try:
                stage_enum = PipelineStage(stage_key)
            except ValueError:
                continue

            if isinstance(record, dict):
                if record.get("stage") != stage_key:
                    record["stage"] = stage_key
                    normalized = True

                status = str(record.get("status", "pending"))
                if (
                    stage_key in deep_stage_names
                    and (
                        status != "pending"
                        or bool(record.get("output_path"))
                        or bool(record.get("error_message"))
                    )
                ):
                    inferred_deep = True
            elif stage_key in deep_stage_names:
                inferred_deep = True

            if current_stage == stage_enum.value and stage_enum in DEEP_ONLY_STAGES:
                inferred_deep = True

        if "pipeline_mode" not in data:
            data["pipeline_mode"] = (
                PipelineMode.DEEP.value if inferred_deep else PipelineMode.STANDARD.value
            )
            normalized = True

        return data, normalized

    def _write_manifest(self, m: WorkspaceManifest) -> None:
        """Atomic write: write to temp file then rename to avoid corruption."""
        m.updated_at = datetime.now(timezone.utc)
        self._manifest_cache = m
        content = m.model_dump_json(indent=2)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._manifest_path.parent), suffix=".tmp"
            )
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                fd = -1  # mark as closed
                # Atomic rename (on POSIX; best-effort on Windows)
                os.replace(tmp_path, str(self._manifest_path))
            except BaseException:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError:
            # Fallback to direct write if temp file approach fails
            self._manifest_path.write_text(content, encoding="utf-8")

    def update_manifest(self, **kwargs) -> WorkspaceManifest:
        m = self.manifest
        for k, v in kwargs.items():
            setattr(m, k, v)
        self._write_manifest(m)
        return m

    # ---- stage tracking --------------------------------------------------

    def mark_stage_running(self, stage: PipelineStage) -> None:
        m = self.manifest
        rec = m.stages.get(stage.value)
        if rec is None:
            rec = StageRecord(stage=stage)
            m.stages[stage.value] = rec
        rec.status = "running"
        rec.started_at = datetime.now(timezone.utc)
        m.current_stage = stage
        self._write_manifest(m)

    def mark_stage_completed(self, stage: PipelineStage, output_path: str = "") -> None:
        m = self.manifest
        rec = m.stages.get(stage.value)
        if rec is None:
            rec = StageRecord(stage=stage)
            m.stages[stage.value] = rec
        rec.status = "completed"
        rec.completed_at = datetime.now(timezone.utc)
        rec.output_path = output_path
        self._write_manifest(m)

    def mark_stage_failed(self, stage: PipelineStage, error: str) -> None:
        m = self.manifest
        rec = m.stages.get(stage.value)
        if rec is None:
            rec = StageRecord(stage=stage)
            m.stages[stage.value] = rec
        rec.status = "failed"
        rec.completed_at = datetime.now(timezone.utc)
        rec.error_message = error
        rec.retries += 1
        m.current_stage = PipelineStage.FAILED
        self._write_manifest(m)

    def increment_retry(self, stage: PipelineStage) -> int:
        m = self.manifest
        rec = m.stages.get(stage.value)
        if rec is None:
            rec = StageRecord(stage=stage)
            m.stages[stage.value] = rec
        rec.retries += 1
        rec.status = "pending"
        rec.error_message = ""
        self._write_manifest(m)
        return rec.retries

    # ---- artifacts -------------------------------------------------------

    def register_artifact(
        self, name: str, file_path: Path, stage: PipelineStage
    ) -> ArtifactRecord:
        checksum = ""
        if file_path.is_file():
            checksum = hashlib.md5(file_path.read_bytes()).hexdigest()
        record = ArtifactRecord(
            name=name,
            path=str(file_path.relative_to(self.path)),
            stage=stage,
            checksum=checksum,
        )
        m = self.manifest
        m.artifacts.append(record)
        self._write_manifest(m)
        return record

    # ---- convenience paths -----------------------------------------------

    @property
    def papers_dir(self) -> Path:
        return self.path / "papers"

    @property
    def plans_dir(self) -> Path:
        return self.path / "plans"

    @property
    def drafts_dir(self) -> Path:
        return self.path / "drafts"

    @property
    def figures_dir(self) -> Path:
        return self.path / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def code_dir(self) -> Path:
        return self.path / "code"

    # ---- utility ---------------------------------------------------------

    def write_json(self, subpath: str, data: dict | list) -> Path:
        p = self.path / subpath
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to write JSON to {p}: {exc}") from exc
        return p

    def read_json(self, subpath: str) -> dict | list:
        p = self.path / subpath
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {p}: {exc}") from exc

    def write_text(self, subpath: str, text: str) -> Path:
        p = self.path / subpath
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Failed to write to {p}: {exc}") from exc
        return p

    def read_text(self, subpath: str) -> str:
        p = self.path / subpath
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        return p.read_text(encoding="utf-8")

    # ---- export ----------------------------------------------------------

    def export(self, output_dir: Path | None = None) -> Path:
        """Export a clean, self-contained output folder.

        Structure:
            {topic_slug}/
            ├── paper.pdf
            ├── paper.tex
            ├── references.bib
            ├── figures/
            │   ├── fig1_architecture.pdf
            │   └── ...
            ├── code/
            │   └── experiment.py
            ├── data/
            │   ├── ideation_output.json
            │   └── experiment_blueprint.json
            └── manifest.json
        """
        # Build folder name from topic
        topic = self.manifest.topic
        slug = _slugify(topic)
        name = f"nanoresearch_{slug}_{self.manifest.session_id[:8]}"

        if output_dir is None:
            output_dir = Path.cwd()
        dest = output_dir / name

        if dest.exists():
            try:
                shutil.rmtree(dest)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot remove existing export dir {dest}: {exc}"
                ) from exc
        try:
            dest.mkdir(parents=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create export dir {dest}: {exc}"
            ) from exc

        # Copy paper outputs
        _copy_if_exists(self.drafts_dir / "paper.pdf", dest / "paper.pdf")
        _copy_if_exists(self.drafts_dir / "paper.tex", dest / "paper.tex")
        _copy_if_exists(self.drafts_dir / "references.bib", dest / "references.bib")

        # Copy figures
        fig_dest = dest / "figures"
        if self.figures_dir.exists() and any(self.figures_dir.iterdir()):
            shutil.copytree(self.figures_dir, fig_dest, dirs_exist_ok=True)
        _prepare_exported_paper_tex(
            dest / "paper.tex",
            has_figures=fig_dest.exists() and any(fig_dest.iterdir()),
        )

        # Copy code (skip .venv, __pycache__, node_modules to avoid long paths on Windows)
        code_dest = dest / "code"
        _skip_dirs = {".venv", "venv", "__pycache__", "node_modules", ".git", ".tox"}
        if self.code_dir.exists() and any(self.code_dir.iterdir()):
            shutil.copytree(
                self.code_dir, code_dest, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*_skip_dirs),
            )
        else:
            code_dest.mkdir(exist_ok=True)
            _copy_if_exists(self.plans_dir / "code_skeleton.py", code_dest / "experiment.py")

        # Copy structured data
        data_dest = dest / "data"
        data_dest.mkdir(exist_ok=True)
        _copy_if_exists(self.papers_dir / "ideation_output.json", data_dest / "ideation_output.json")
        _copy_if_exists(self.plans_dir / "experiment_blueprint.json", data_dest / "experiment_blueprint.json")
        _copy_if_exists(self.drafts_dir / "paper_skeleton.json", data_dest / "paper_skeleton.json")

        # Copy manifest
        _copy_if_exists(self._manifest_path, dest / "manifest.json")

        # Write a short README
        readme = (
            f"# {topic}\n\n"
            f"Auto-generated by NanoResearch (session: {self.manifest.session_id})\n\n"
            f"## Contents\n\n"
            f"- `paper.pdf` — Compiled paper\n"
            f"- `paper.tex` — LaTeX source\n"
            f"- `references.bib` — Bibliography\n"
            f"- `figures/` — All figures (PDF + PNG)\n"
            f"- `code/experiment.py` — Experiment code skeleton ({_count_lines(code_dest / 'experiment.py')} lines)\n"
            f"- `code/` — Complete runnable project (if multi-file generation enabled)\n"
            f"- `data/` — Structured intermediate outputs (JSON)\n"
            f"- `manifest.json` — Full pipeline execution record\n"
        )
        (dest / "README.md").write_text(readme, encoding="utf-8")

        return dest


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a filesystem-safe slug."""
    import re
    # Keep alphanumeric, Chinese chars, hyphens
    slug = re.sub(r'[^\w\u4e00-\u9fff-]', '_', text)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug[:max_len]


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            logger.warning("Failed to copy %s -> %s: %s", src, dst, exc)


def _prepare_exported_paper_tex(tex_path: Path, has_figures: bool) -> None:
    """Make exported ``paper.tex`` self-contained relative to the export root."""
    if not tex_path.is_file():
        return

    try:
        tex = tex_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read exported LaTeX %s: %s", tex_path, exc)
        return

    updated = tex
    uses_graphics = r"\includegraphics" in updated
    has_graphicspath = r"\graphicspath" in updated
    has_graphicx = re.search(
        r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\bgraphicx\b[^}]*\}",
        updated,
    ) is not None

    if uses_graphics and not has_graphicx:
        updated = _insert_into_preamble(updated, r"\usepackage{graphicx}")
    if uses_graphics and has_figures and not has_graphicspath:
        updated = _insert_into_preamble(updated, r"\graphicspath{{figures/}}")

    if updated == tex:
        return

    try:
        tex_path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to update exported LaTeX %s: %s", tex_path, exc)


def _insert_into_preamble(tex: str, snippet: str) -> str:
    """Insert a LaTeX preamble snippet before ``\\begin{document}``."""
    if snippet in tex:
        return tex

    marker = r"\begin{document}"
    index = tex.find(marker)
    line = f"{snippet}\n"
    if index == -1:
        return f"{line}{tex}"

    prefix = tex[:index]
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    return f"{prefix}{line}{tex[index:]}"


def _count_lines(path: Path) -> int:
    try:
        if path.is_file():
            return len(path.read_text(encoding="utf-8").splitlines())
    except OSError:
        pass
    return 0
