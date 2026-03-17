"""Resource matching and materialization for repair."""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .repair import RESOURCE_SUCCESS_STATUSES

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.repair_journal import capture_repair_snapshot, rollback_snapshot

logger = logging.getLogger(__name__)


class _RepairResourcesMixin:
    """Mixin — resource extraction, matching, and materialization."""

    @staticmethod
    def _extract_missing_resource_targets(error_text: str) -> list[str]:
        patterns = [
            r"""No such file or directory:\s*['"]([^'"]+)['"]""",
            r"""can't open file\s+['"]([^'"]+)['"]""",
            r"""does not exist:\s*['"]([^'"]+)['"]""",
            r"""FileNotFoundError:.*?['"]([^'"]+)['"]""",
            r"""Can't load [^'"]+ for ['"]([^'"]+)['"]""",
            r"""Incorrect path_or_model_id:\s*['"]([^'"]+)['"]""",
        ]
        targets: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                candidate = str(match.group(1)).strip()
                if candidate and candidate not in targets:
                    targets.append(candidate)
        return targets

    @staticmethod
    def _resource_kind_from_path(path_text: str) -> str:
        lower = path_text.lower()
        if any(token in lower for token in ("/models/", "\\models\\", ".pt", ".bin", ".ckpt", ".safetensors")):
            return "model"
        return "dataset"

    @staticmethod
    def _normalized_resource_key(path_text: str) -> str:
        name = Path(path_text).name.lower()
        for suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        else:
            for suffix in (".gz", ".bz2", ".xz", ".zip"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break

        while True:
            stem, ext = os.path.splitext(name)
            if ext.lower() in {
                ".csv",
                ".tsv",
                ".txt",
                ".json",
                ".jsonl",
                ".pkl",
                ".pickle",
                ".npy",
                ".npz",
                ".pt",
                ".pth",
                ".bin",
                ".ckpt",
                ".h5",
                ".hdf5",
                ".parquet",
                ".fa",
                ".fasta",
            }:
                name = stem
                continue
            break
        return name

    @classmethod
    def _collect_resource_candidates(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        candidates: list[dict[str, str]] = []
        seen_paths: set[str] = set()

        def add_candidate(path_value: str, kind: str, name: str) -> None:
            normalized = str(path_value or "").strip()
            if not normalized or normalized in seen_paths:
                return
            candidate_path = Path(normalized)
            if not candidate_path.exists():
                return
            seen_paths.add(normalized)
            candidates.append(
                {
                    "path": normalized,
                    "kind": kind,
                    "name": str(name or "").strip().lower(),
                    "basename": candidate_path.name.lower(),
                    "normalized_key": cls._normalized_resource_key(normalized),
                }
            )

        def scan_root(root_path: Path, kind: str) -> None:
            if not root_path.exists():
                return
            add_candidate(str(root_path), kind, root_path.name)
            try:
                children = sorted(root_path.iterdir())
            except OSError:
                return
            for child in children:
                add_candidate(str(child), kind, child.name)
                if child.is_dir():
                    try:
                        nested_children = sorted(child.iterdir())[:20]
                    except OSError:
                        continue
                    for nested in nested_children:
                        add_candidate(str(nested), kind, child.name)

        if isinstance(resource_context, dict):
            for resource in resource_context.get("downloaded_resources", []):
                if not isinstance(resource, dict):
                    continue
                if resource.get("status") not in RESOURCE_SUCCESS_STATUSES:
                    continue
                kind = str(resource.get("type", "dataset")).strip().lower()
                name = str(resource.get("name", "")).strip()
                for key in ("workspace_path", "path"):
                    value = resource.get(key)
                    if isinstance(value, str):
                        add_candidate(value, kind, name)
                for value in resource.get("workspace_files", []) or []:
                    if isinstance(value, str):
                        add_candidate(value, kind, name)

            for alias in resource_context.get("workspace_resource_aliases", []):
                if not isinstance(alias, dict):
                    continue
                kind = str(alias.get("type", "dataset")).strip().lower()
                name = str(alias.get("name", "")).strip()
                workspace_path = alias.get("workspace_path")
                if isinstance(workspace_path, str):
                    add_candidate(workspace_path, kind, name)
                for value in alias.get("workspace_files", []) or []:
                    if isinstance(value, str):
                        add_candidate(value, kind, name)

            for root_key, kind in (("data_dir", "dataset"), ("models_dir", "model")):
                root_value = str(resource_context.get(root_key, "")).strip()
                if not root_value:
                    continue
                root_path = Path(root_value)
                scan_root(root_path, kind)

        for root_path, kind in (
            (code_dir / "data", "dataset"),
            (code_dir / "datasets", "dataset"),
            (code_dir / "models", "model"),
            (code_dir / "checkpoints", "model"),
        ):
            scan_root(root_path, kind)

        for config_candidate in (
            code_dir / "config.py",
            code_dir / "config.yaml",
            code_dir / "config.yml",
            code_dir / "config.json",
            code_dir / "config.toml",
            code_dir / "config" / "default.yaml",
            code_dir / "config" / "default.yml",
            code_dir / "config" / "default.json",
            code_dir / "config" / "default.toml",
            code_dir / "configs" / "default.yaml",
            code_dir / "configs" / "default.yml",
            code_dir / "configs" / "default.json",
            code_dir / "configs" / "default.toml",
            code_dir / ".nanoresearch_autofix" / "config_auto.yaml",
            code_dir / ".nanoresearch_autofix" / "config_auto.json",
            code_dir / ".nanoresearch_autofix" / "config_auto.toml",
        ):
            if config_candidate.exists():
                add_candidate(str(config_candidate), "config", config_candidate.name)

        return candidates

    @classmethod
    def _match_resource_target(
        cls,
        code_dir: Path,
        missing_target: str,
        resource_context: dict[str, Any] | None,
    ) -> str | None:
        candidates = cls._collect_resource_candidates(code_dir, resource_context)
        if not candidates:
            return None

        missing_path = Path(missing_target)
        missing_name = missing_path.name.lower()
        missing_kind = (
            "config"
            if "config" in missing_target.lower()
            else cls._resource_kind_from_path(missing_target)
        )

        def filter_kind(items: list[dict[str, str]]) -> list[dict[str, str]]:
            typed = [item for item in items if item["kind"] == missing_kind]
            return typed or items

        cache_to_workspace = [
            ("cache_data_dir", "data_dir"),
            ("cache_models_dir", "models_dir"),
        ]
        for cache_key, workspace_key in cache_to_workspace:
            cache_dir = str(resource_context.get(cache_key, "") if isinstance(resource_context, dict) else "").strip()
            workspace_dir = str(resource_context.get(workspace_key, "") if isinstance(resource_context, dict) else "").strip()
            if cache_dir and workspace_dir and missing_target.startswith(cache_dir):
                suffix = missing_target[len(cache_dir):].lstrip("/\\")
                candidate = Path(workspace_dir) / suffix
                if candidate.exists():
                    return str(candidate)

        basename_matches = filter_kind(
            [item for item in candidates if item["basename"] == missing_name]
        )
        if len(basename_matches) == 1:
            return basename_matches[0]["path"]

        normalized_key = cls._normalized_resource_key(missing_target)
        normalized_matches = filter_kind(
            [item for item in candidates if item.get("normalized_key") == normalized_key]
        )
        if len(normalized_matches) == 1:
            return normalized_matches[0]["path"]

        name_matches = filter_kind(
            [item for item in candidates if item["name"] and item["name"] in missing_target.lower()]
        )
        if len(name_matches) == 1:
            return name_matches[0]["path"]

        if missing_kind == "config":
            config_files = [item for item in candidates if item["kind"] == "config" and Path(item["path"]).is_file()]
            if len(config_files) == 1:
                return config_files[0]["path"]

        if missing_kind == "dataset":
            dataset_files = [item for item in candidates if item["kind"] == "dataset" and Path(item["path"]).is_file()]
            if len(dataset_files) == 1:
                return dataset_files[0]["path"]

        return None

    @classmethod
    def _resource_replacement_map(
        cls,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
    ) -> dict[str, str]:
        if not isinstance(resource_context, dict):
            resource_context = {}

        replacements: dict[str, str] = {}
        for old_key, new_key in (("cache_data_dir", "data_dir"), ("cache_models_dir", "models_dir")):
            old_value = str(resource_context.get(old_key, "")).strip()
            new_value = str(resource_context.get(new_key, "")).strip()
            if old_value and new_value and old_value != new_value:
                replacements[old_value] = new_value

        for target in cls._extract_missing_resource_targets(error_text):
            replacement = cls._match_resource_target(code_dir, target, resource_context)
            if replacement and replacement != target:
                replacements[target] = replacement

        return replacements

    @classmethod
    def _materialize_missing_resource_targets(
        cls,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
    ) -> list[str]:
        if not isinstance(resource_context, dict):
            return []

        created: list[str] = []
        candidates = cls._collect_resource_candidates(code_dir, resource_context)
        for target_text in cls._extract_missing_resource_targets(error_text):
            target_path = Path(target_text)
            if target_path.exists():
                continue
            if target_path.suffix.lower() in {".gz", ".bz2", ".xz", ".zip"}:
                continue

            normalized_key = cls._normalized_resource_key(target_text)
            gz_matches = [
                item
                for item in candidates
                if Path(item["path"]).is_file()
                and item.get("normalized_key") == normalized_key
                and item["path"].lower().endswith(".gz")
                and not item["path"].lower().endswith(".tar.gz")
            ]
            if len(gz_matches) == 1:
                source_path = Path(gz_matches[0]["path"])
                try:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(source_path, "rb") as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except OSError:
                    pass
                else:
                    created.append(str(target_path))
                    continue

            zip_matches = [
                item
                for item in candidates
                if Path(item["path"]).is_file()
                and item["path"].lower().endswith(".zip")
            ]
            extracted = False
            for zip_candidate in zip_matches[:5]:
                source_path = Path(zip_candidate["path"])
                try:
                    with zipfile.ZipFile(source_path) as archive:
                        members = [
                            member
                            for member in archive.namelist()
                            if member and not member.endswith("/")
                        ]
                        matching_members = [
                            member
                            for member in members
                            if cls._normalized_resource_key(member) == normalized_key
                        ]
                        if len(matching_members) != 1:
                            continue
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(matching_members[0]) as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                except (OSError, zipfile.BadZipFile, KeyError):
                    continue
                created.append(str(target_path))
                extracted = True
                break
            if extracted:
                continue

        return created

    def _attempt_resource_path_repair(
        self,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        materialized = self._materialize_missing_resource_targets(code_dir, error_text, resource_context)
        replacements = self._resource_replacement_map(code_dir, error_text, resource_context)
        snapshot_batch: list[dict[str, Any]] = []
        for created_path_text in materialized:
            snapshot_batch.append(
                capture_repair_snapshot(
                    self.workspace.path,
                    Path(created_path_text),
                    namespace="resource_path_repair",
                    root_dir=self.workspace.path,
                    existed_before=False,
                    operation="create",
                )
            )
        if not replacements:
            self._record_snapshot_batch(
                mutation_kind="resource_path_repair",
                scope=scope or "resource_path_repair",
                snapshots=snapshot_batch,
                metadata={
                    "modified_files": [],
                    "materialized_files": list(materialized),
                },
            )
            return materialized

        text_suffixes = {".py", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".txt"}
        modified_files: list[str] = []
        for candidate in code_dir.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in text_suffixes:
                continue
            try:
                original = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            updated = original
            for old_value, new_value in replacements.items():
                if old_value and old_value in updated:
                    updated = updated.replace(old_value, new_value)

            if updated != original:
                snapshot = capture_repair_snapshot(
                    self.workspace.path,
                    candidate,
                    namespace="resource_path_repair",
                    root_dir=self.workspace.path,
                    operation="rewrite",
                )
                try:
                    candidate.write_text(updated, encoding="utf-8")
                except OSError:
                    continue

                if candidate.suffix.lower() == ".py" and not ExperimentAgent._check_syntax(candidate):
                    self.log(f"Resource-path repair produced invalid syntax in {candidate}, rolling back")
                    rollback_snapshot(self.workspace.path, candidate, snapshot)
                    snapshot["rolled_back"] = True
                    snapshot["rollback_reason"] = "syntax_error"
                    snapshot_batch.append(snapshot)
                    continue

                modified_files.append(str(candidate.relative_to(code_dir)))
                snapshot_batch.append(snapshot)

        self._record_snapshot_batch(
            mutation_kind="resource_path_repair",
            scope=scope or "resource_path_repair",
            snapshots=snapshot_batch,
            metadata={
                "modified_files": list(modified_files),
                "materialized_files": list(materialized),
            },
        )
        return [*materialized, *modified_files]

