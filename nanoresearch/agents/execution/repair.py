"""Error repair: resource matching, repair strategies, runtime remediation, and repair journal."""

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

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.project_runner import RUNNER_CONFIG_NAME
from nanoresearch.agents.repair_journal import (
    REPAIR_SNAPSHOT_JOURNAL_PATH,
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.agents.runtime_env import ExperimentExecutionPolicy, RuntimeEnvironmentManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants referenced by repair methods
# ---------------------------------------------------------------------------
REMEDIATION_LEDGER_PATH = "logs/execution_remediation_ledger.json"
RESOURCE_SUCCESS_STATUSES = {"downloaded", "full", "config_only"}
MODULE_PACKAGE_ALIASES = {
    "cv2": "opencv-python",
    "pil": "Pillow",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "bio": "biopython",
}
QUICK_EVAL_AUTO_OPTIONS = {
    "--quick-eval",
    "--epochs",
    "--num-epochs",
    "--max-steps",
    "--steps",
    "--batch-size",
    "--batch_size",
    "--num-workers",
    "--num_workers",
    "--workers",
    "--subset-size",
    "--subset_size",
    "--train-size",
    "--quick-eval-train-size",
    "--limit-train-batches",
    "--limit-val-batches",
}


class _RepairMixin:
    """Mixin providing error-repair, resource-matching, and runtime-remediation helpers."""

    @staticmethod
    def _repair_error_text(result: dict[str, Any]) -> str:
        stderr_text = str(result.get("stderr") or "").strip()
        if stderr_text:
            return stderr_text
        stdout_text = str(result.get("stdout") or "").strip()
        if stdout_text:
            return stdout_text
        return f"Process exited with return code {result.get('returncode', 'unknown')} and produced no output."

    @classmethod
    def _repair_error_signature(cls, result: dict[str, Any]) -> str:
        error_text = cls._repair_error_text(result)
        for raw_line in reversed(error_text.splitlines()):
            line = raw_line.strip()
            if not line or line.startswith("File ") or line.startswith("Traceback"):
                continue
            return f"rc={result.get('returncode', 'unknown')}|{line[:240]}"
        return f"rc={result.get('returncode', 'unknown')}|empty"

    @staticmethod
    def _repair_repeat_count(
        fix_history: list[dict[str, Any]],
        signature: str,
    ) -> int:
        for entry in fix_history:
            if entry.get("signature") == signature:
                return int(entry.get("repeat_count", 1)) + 1
        return 1

    @staticmethod
    def _record_repair_attempt(
        fix_history: list[dict[str, Any]],
        signature: str,
        error_text: str,
        cycle: int,
        modified: list[str],
    ) -> None:
        for entry in fix_history:
            if entry.get("signature") == signature:
                entry["repeat_count"] = int(entry.get("repeat_count", 1)) + 1
                entry["cycle"] = cycle
                entry["error_msg"] = error_text[:300]
                if modified:
                    seen = list(entry.get("fixed_files", []))
                    for rel_path in modified:
                        if rel_path not in seen:
                            seen.append(rel_path)
                    entry["fixed_files"] = seen
                return

        fix_history.append(
            {
                "signature": signature,
                "error_msg": error_text[:300],
                "cycle": cycle,
                "repeat_count": 1,
                "fixed_files": list(modified or []),
            }
        )

    @staticmethod
    def _append_remediation_entry(
        remediation_ledger: list[dict[str, Any]] | None,
        *,
        kind: str,
        status: str,
        scope: str,
        round_number: int | None = None,
        cycle: int | None = None,
        signature: str = "",
        reason: str = "",
        files: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if remediation_ledger is None:
            return

        entry: dict[str, Any] = {
            "entry_id": len(remediation_ledger) + 1,
            "kind": kind,
            "status": status,
            "scope": scope,
        }
        if round_number is not None:
            entry["round_number"] = round_number
        if cycle is not None:
            entry["cycle"] = cycle
        if signature:
            entry["signature"] = signature
        if reason:
            entry["reason"] = reason
        if files:
            entry["files"] = list(files)
        if details:
            entry["details"] = dict(details)
        remediation_ledger.append(entry)

    def _persist_remediation_ledger(
        self,
        remediation_ledger: list[dict[str, Any]] | None,
    ) -> str:
        payload = {
            "entry_count": len(remediation_ledger or []),
            "entries": list(remediation_ledger or []),
        }
        self.workspace.write_json(REMEDIATION_LEDGER_PATH, payload)
        return REMEDIATION_LEDGER_PATH

    def _repair_snapshot_journal_path(self) -> str:
        journal_path = self.workspace.path / REPAIR_SNAPSHOT_JOURNAL_PATH
        return REPAIR_SNAPSHOT_JOURNAL_PATH if journal_path.is_file() else ""

    def _record_snapshot_batch(
        self,
        *,
        mutation_kind: str,
        scope: str,
        snapshots: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not snapshots:
            self._remember_mutation_snapshot_entry(None)
            return None

        entry = append_snapshot_journal(
            self.workspace.path,
            agent=self.__class__.__name__,
            mutation_kind=mutation_kind,
            scope=scope,
            snapshots=snapshots,
            metadata=metadata,
        )
        self._remember_mutation_snapshot_entry(entry)
        return entry

    def _record_runtime_env_ledger(
        self,
        runtime_env: dict[str, Any],
        remediation_ledger: list[dict[str, Any]] | None,
    ) -> None:
        if remediation_ledger is None or not isinstance(runtime_env, dict):
            return

        if runtime_env.get("created"):
            self._append_remediation_entry(
                remediation_ledger,
                kind="runtime_env_create",
                status="applied",
                scope="local_environment",
                details={
                    "env_kind": runtime_env.get("kind", ""),
                    "env_name": runtime_env.get("env_name", ""),
                    "env_path": runtime_env.get("env_path", ""),
                    "recreated": bool(runtime_env.get("recreated", False)),
                },
            )

        dependency_install = runtime_env.get("dependency_install")
        if not isinstance(dependency_install, dict):
            dependency_install = {}
        status = str(dependency_install.get("status") or "").strip()
        if status:
            self._append_remediation_entry(
                remediation_ledger,
                kind="dependency_install",
                status=status,
                scope="local_environment",
                details={
                    "source": dependency_install.get("source", ""),
                    "manifest": dependency_install.get("manifest", ""),
                    "strategy": dependency_install.get("strategy", ""),
                    "error": dependency_install.get("error", ""),
                    "stderr": dependency_install.get("stderr", ""),
                    "returncode": dependency_install.get("returncode"),
                },
            )

        runtime_validation = runtime_env.get("runtime_validation")
        if not isinstance(runtime_validation, dict):
            return
        validation_status = str(runtime_validation.get("status") or "").strip()
        if not validation_status:
            return
        python_smoke = runtime_validation.get("python_smoke")
        pip_probe = runtime_validation.get("pip_probe")
        import_probe = runtime_validation.get("import_probe")
        self._append_remediation_entry(
            remediation_ledger,
            kind="runtime_env_validation",
            status=validation_status,
            scope="local_environment",
            details={
                "python_smoke_status": python_smoke.get("status", "") if isinstance(python_smoke, dict) else "",
                "python_executable": python_smoke.get("executable", "") if isinstance(python_smoke, dict) else "",
                "python_version": python_smoke.get("version", "") if isinstance(python_smoke, dict) else "",
                "pip_status": pip_probe.get("status", "") if isinstance(pip_probe, dict) else "",
                "pip_version": pip_probe.get("version", "") if isinstance(pip_probe, dict) else "",
                "import_status": import_probe.get("status", "") if isinstance(import_probe, dict) else "",
                "failed_imports": list(import_probe.get("failures", []) or []) if isinstance(import_probe, dict) else [],
                "skipped_reason": import_probe.get("skipped_reason", "") if isinstance(import_probe, dict) else "",
            },
        )

        runtime_validation_repair = runtime_env.get("runtime_validation_repair")
        if not isinstance(runtime_validation_repair, dict):
            return
        repair_status = str(runtime_validation_repair.get("status") or "").strip()
        if not repair_status or repair_status == "skipped":
            return
        self._append_remediation_entry(
            remediation_ledger,
            kind="runtime_env_repair",
            status=repair_status,
            scope="local_environment",
            details={
                "actions": list(runtime_validation_repair.get("actions", []) or []),
            },
        )

    def _record_launch_contract_ledger(
        self,
        launch_contract: dict[str, Any],
        remediation_ledger: list[dict[str, Any]] | None,
        *,
        round_number: int | None = None,
    ) -> None:
        if remediation_ledger is None or not isinstance(launch_contract, dict):
            return
        status = str(launch_contract.get("status") or "").strip()
        if not status:
            return
        self._append_remediation_entry(
            remediation_ledger,
            kind="launch_contract",
            status=status,
            scope="local_launch",
            round_number=round_number,
            details={
                "target_kind": launch_contract.get("target_kind", ""),
                "target": launch_contract.get("target", ""),
                "resolved_target": launch_contract.get("resolved_target", ""),
                "created_dirs": list(launch_contract.get("created_dirs", []) or []),
                "warnings": list(launch_contract.get("warnings", []) or []),
                "failures": list(launch_contract.get("failures", []) or []),
            },
        )

    def _record_launch_contract_repair_ledger(
        self,
        repair_result: dict[str, Any],
        remediation_ledger: list[dict[str, Any]] | None,
        *,
        round_number: int | None = None,
    ) -> None:
        if remediation_ledger is None or not isinstance(repair_result, dict):
            return
        status = str(repair_result.get("status") or "").strip()
        if not status or status == "skipped":
            return
        self._append_remediation_entry(
            remediation_ledger,
            kind="launch_contract_repair",
            status=status,
            scope="local_launch",
            round_number=round_number,
            details={
                "actions": list(repair_result.get("actions", []) or []),
                "files_modified": list(repair_result.get("files_modified", []) or []),
                "command": list(repair_result.get("command", []) or []),
                "initial_failures": list(
                    repair_result.get("initial_contract", {}).get("failures", [])
                    if isinstance(repair_result.get("initial_contract"), dict)
                    else []
                ),
                "final_failures": list(
                    repair_result.get("final_contract", {}).get("failures", [])
                    if isinstance(repair_result.get("final_contract"), dict)
                    else []
                ),
            },
        )

    def _build_repair_context(
        self,
        code_dir: Path,
        result: dict[str, Any],
        *,
        mode: str,
        repeat_count: int,
        resource_context: dict[str, Any] | None = None,
    ) -> str:
        report = PreflightChecker(code_dir).run_all()
        context_parts: list[str] = []

        stdout_text = str(result.get("stdout") or "").strip()
        stderr_text = str(result.get("stderr") or "").strip()
        if stdout_text and stdout_text != stderr_text:
            stdout_lines = stdout_text.splitlines()
            stdout_snippet = "\n".join(stdout_lines[-20:])[:1200]
            context_parts.append(f"Recent stdout ({mode}):\n{stdout_snippet}")

        if report.blocking_failures:
            context_parts.append(
                "Preflight blocking diagnostics:\n- " + "\n- ".join(report.blocking_failures[:8])
            )
        elif report.warning_messages:
            context_parts.append(
                "Preflight warnings:\n- " + "\n- ".join(report.warning_messages[:8])
            )

        if report.suggested_fixes:
            context_parts.append(
                "Suggested preflight fixes:\n- " + "\n- ".join(report.suggested_fixes[:8])
            )

        resource_summary = self._summarize_available_resources(code_dir, resource_context)
        if resource_summary:
            context_parts.append(resource_summary)

        if repeat_count > 1:
            context_parts.append(
                f"This failure signature has repeated {repeat_count} times. "
                "Do not repeat the same patch strategy; target a different root cause."
            )

        return "\n\n".join(part for part in context_parts if part)

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

    @staticmethod
    def _extract_missing_required_options(error_text: str) -> list[str]:
        options: list[str] = []
        patterns = [
            r"""the following arguments are required:\s*([^\n\r]+)""",
            r"""Missing option ['"]?(--[A-Za-z0-9][A-Za-z0-9_-]*)['"]?""",
            r"""argument\s+(--[A-Za-z0-9][A-Za-z0-9_-]*)\s*:\s*expected one argument""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                payload = str(match.group(1)).strip()
                if not payload:
                    continue
                if payload.startswith("--"):
                    extracted = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload)
                    if extracted:
                        for option in extracted:
                            if option not in options:
                                options.append(option)
                        continue
                    if payload not in options:
                        options.append(payload)
                    continue
                for option in re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload):
                    if option not in options:
                        options.append(option)
        return options

    @staticmethod
    def _extract_unrecognized_options(error_text: str) -> list[str]:
        options: list[str] = []
        patterns = [
            r"""unrecognized arguments:\s*([^\n\r]+)""",
            r"""No such option:\s*(--[A-Za-z0-9][A-Za-z0-9_-]*)""",
            r"""no such option:\s*(--[A-Za-z0-9][A-Za-z0-9_-]*)""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                payload = str(match.group(1)).strip()
                if not payload:
                    continue
                extracted = re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", payload)
                if extracted:
                    for option in extracted:
                        if option not in options:
                            options.append(option)
                    continue
                if payload.startswith("--") and payload not in options:
                    options.append(payload)
        return options

    @staticmethod
    def _strip_command_option(tokens: list[str], option: str) -> list[str]:
        updated: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == option:
                index += 1
                if index < len(tokens) and not tokens[index].startswith("--"):
                    index += 1
                continue
            if token.startswith(f"{option}="):
                index += 1
                continue
            updated.append(token)
            index += 1
        return updated

    @staticmethod
    def _command_option_present(tokens: list[str], options: list[str]) -> tuple[str, int, str]:
        for option in options:
            for index, token in enumerate(tokens):
                if token == option:
                    value = tokens[index + 1] if index + 1 < len(tokens) else ""
                    return option, index, value
                if token.startswith(f"{option}="):
                    return option, index, token.split("=", 1)[1]
        return "", -1, ""

    @staticmethod
    def _path_variants(code_dir: Path, path_value: str) -> set[str]:
        normalized = str(path_value or "").strip()
        if not normalized:
            return set()

        variants: set[str] = {os.path.normcase(os.path.normpath(normalized))}
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = code_dir / candidate
        variants.add(os.path.normcase(os.path.normpath(str(candidate))))
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        variants.add(os.path.normcase(os.path.normpath(str(resolved))))
        return variants

    @classmethod
    def _option_value_matches_missing_target(
        cls,
        code_dir: Path,
        option_value: str,
        missing_targets: list[str],
    ) -> bool:
        option_variants = cls._path_variants(code_dir, option_value)
        if not option_variants:
            return False
        for target in missing_targets:
            target_variants = cls._path_variants(code_dir, target)
            if option_variants & target_variants:
                return True
        return False

    @staticmethod
    def _command_entry_script(tokens: list[str], code_dir: Path) -> Path | None:
        for token in tokens:
            normalized = str(token or "").strip()
            if not normalized:
                continue
            if normalized in {"-m", "-c"}:
                return None
            if normalized.endswith(".py"):
                candidate = Path(normalized)
                return candidate if candidate.is_absolute() else code_dir / candidate
        return None

    @staticmethod
    def _entry_script_supports_flag(entry_script: Path | None, flag: str) -> bool:
        if not entry_script or not entry_script.exists():
            return False
        try:
            content = entry_script.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        normalized_flag = flag.lstrip("-").replace("-", "_")
        return flag in content or normalized_flag in content

    @classmethod
    def _resume_failure_signals(cls, error_text: str) -> list[str]:
        lower = str(error_text or "").lower()
        signals: list[str] = []
        signal_map = {
            "timed out": "timeout",
            "timeout": "timeout",
            "keyboardinterrupt": "keyboard_interrupt",
            "interrupted": "interrupted",
            "sigterm": "sigterm",
            "terminated": "terminated",
            "preempt": "preempted",
            "cancelled": "cancelled",
            "node_fail": "node_fail",
            "node fail": "node_fail",
        }
        for token, label in signal_map.items():
            if token in lower and label not in signals:
                signals.append(label)
        return signals

    @staticmethod
    def _choose_single_path(candidates: list[Path]) -> Path | None:
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        if len(unique) == 1:
            return unique[0]
        return None

    @staticmethod
    def _choose_latest_path(candidates: list[Path]) -> Path | None:
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        if not unique:
            return None

        def sort_key(path: Path) -> tuple[float, str]:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = -1.0
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            return mtime, str(resolved)

        return sorted(unique, key=sort_key, reverse=True)[0]

    @classmethod
    def _keyword_path_candidate(
        cls,
        candidates: list[Path],
        keywords: tuple[str, ...],
        *,
        files_only: bool = False,
        dirs_only: bool = False,
        allow_latest: bool = False,
    ) -> Path | None:
        normalized_keywords = tuple(str(keyword or "").strip().lower() for keyword in keywords if keyword)
        if not normalized_keywords:
            return None

        scored: list[tuple[int, Path]] = []
        for candidate in candidates:
            if files_only and not candidate.is_file():
                continue
            if dirs_only and not candidate.is_dir():
                continue
            haystacks = {
                candidate.name.lower(),
                cls._normalized_resource_key(str(candidate)),
            }
            parent = candidate.parent
            if parent != candidate:
                haystacks.add(parent.name.lower())
                haystacks.add(cls._normalized_resource_key(str(parent)))
            score = sum(1 for keyword in normalized_keywords if any(keyword in value for value in haystacks))
            if score > 0:
                scored.append((score, candidate))

        if not scored:
            return None
        best_score = max(score for score, _candidate in scored)
        best_candidates = [candidate for score, candidate in scored if score == best_score]
        match = cls._choose_single_path(best_candidates)
        if match is not None:
            return match
        if allow_latest:
            return cls._choose_latest_path(best_candidates)
        return None

    @classmethod
    def _runtime_config_candidate(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> str | None:
        resources = cls._collect_resource_candidates(code_dir, resource_context)
        config_paths = [
            Path(item["path"])
            for item in resources
            if item.get("kind") == "config" and Path(item["path"]).is_file()
        ]
        preferred = [
            path
            for path in config_paths
            if path.name.lower().startswith("config_auto")
            or path.name.lower().startswith("default.")
        ]
        match = cls._choose_single_path(preferred) or cls._choose_single_path(config_paths)
        return str(match) if match is not None else None

    @classmethod
    def _runtime_dataset_candidates(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> list[Path]:
        resources = cls._collect_resource_candidates(code_dir, resource_context)
        candidates = [
            Path(item["path"])
            for item in resources
            if item.get("kind") == "dataset" and Path(item["path"]).is_file()
        ]
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @classmethod
    def _runtime_dataset_directory_candidates(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> list[Path]:
        resources = cls._collect_resource_candidates(code_dir, resource_context)
        candidates = [
            Path(item["path"])
            for item in resources
            if item.get("kind") == "dataset" and Path(item["path"]).is_dir()
        ]
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @classmethod
    def _runtime_model_candidates(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> list[Path]:
        resources = cls._collect_resource_candidates(code_dir, resource_context)
        candidates = [
            Path(item["path"])
            for item in resources
            if item.get("kind") == "model" and Path(item["path"]).is_file()
        ]
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @classmethod
    def _runtime_option_candidate(
        cls,
        code_dir: Path,
        option: str,
        resource_context: dict[str, Any] | None,
    ) -> str | None:
        normalized = str(option or "").strip().lower()
        if not normalized:
            return None

        config_options = {"--config", "--config-path", "--cfg", "--config-file"}
        data_dir_options = {"--data-dir", "--data-root", "--dataset-dir", "--dataset-root", "--data", "--dataset"}
        data_file_options = {"--data-path", "--dataset-path", "--input-path", "--input-file", "--dataset-file"}
        train_file_options = {"--train-file", "--train-data", "--train-path"}
        val_file_options = {
            "--val-file",
            "--valid-file",
            "--validation-file",
            "--val-data",
            "--valid-data",
            "--validation-data",
            "--val-path",
            "--valid-path",
            "--dev-file",
            "--dev-data",
            "--dev-path",
        }
        test_file_options = {"--test-file", "--test-data", "--test-path"}
        labels_options = {"--labels-path", "--label-file", "--labels-file", "--label-path"}
        annotations_options = {"--annotations", "--annotation-file", "--annotation-path", "--annotations-file"}
        split_file_options = {"--split-file", "--splits-file", "--split-path", "--fold-file", "--folds-file"}
        metadata_options = {"--metadata-path", "--meta-path", "--metadata-file", "--meta-file"}
        image_dir_options = {"--image-dir", "--images-dir", "--image-root", "--images-root"}
        label_dir_options = {"--label-dir", "--labels-dir", "--label-root", "--labels-root"}
        model_dir_options = {"--model-dir", "--model-root"}
        model_file_options = {"--model-path", "--model-file", "--pretrained-model"}
        tokenizer_options = {"--tokenizer-path", "--tokenizer-name-or-path"}
        checkpoint_options = {"--checkpoint", "--ckpt", "--checkpoint-path"}
        resume_options = {"--resume", "--resume-from", "--resume-path"}
        checkpoint_dir_options = {"--checkpoint-dir", "--ckpt-dir"}
        output_dir_options = {"--output-dir", "--results-dir", "--save-dir"}
        log_dir_options = {"--log-dir", "--logging-dir"}

        if normalized in config_options:
            return cls._runtime_config_candidate(code_dir, resource_context)

        if normalized in output_dir_options:
            return str((code_dir / "results").resolve())
        if normalized in checkpoint_dir_options:
            return str((code_dir / "checkpoints").resolve())
        if normalized in log_dir_options:
            return str((code_dir / "logs").resolve())

        if normalized in data_dir_options:
            resource_dir = str(resource_context.get("data_dir", "")).strip() if isinstance(resource_context, dict) else ""
            if resource_dir and Path(resource_dir).exists():
                return str(Path(resource_dir).resolve())
            return_value = cls._choose_single_path([code_dir / "data", code_dir / "datasets"])
            return str(return_value.resolve()) if return_value is not None else None

        if normalized in model_dir_options:
            resource_dir = str(resource_context.get("models_dir", "")).strip() if isinstance(resource_context, dict) else ""
            if resource_dir and Path(resource_dir).exists():
                return str(Path(resource_dir).resolve())
            return_value = cls._choose_single_path([code_dir / "models", code_dir / "checkpoints"])
            return str(return_value.resolve()) if return_value is not None else None

        dataset_files = cls._runtime_dataset_candidates(code_dir, resource_context)
        dataset_dirs = cls._runtime_dataset_directory_candidates(code_dir, resource_context)
        if normalized in train_file_options:
            train_match = cls._keyword_path_candidate(dataset_files, ("train",), files_only=True)
            if train_match is not None:
                return str(train_match.resolve())
            fallback = cls._choose_single_path(dataset_files)
            return str(fallback.resolve()) if fallback is not None else None
        if normalized in val_file_options:
            val_match = cls._keyword_path_candidate(dataset_files, ("val", "valid", "validation", "dev"), files_only=True)
            return str(val_match.resolve()) if val_match is not None else None
        if normalized in test_file_options:
            test_match = cls._keyword_path_candidate(dataset_files, ("test",), files_only=True)
            return str(test_match.resolve()) if test_match is not None else None
        if normalized in labels_options:
            label_match = cls._keyword_path_candidate(dataset_files, ("label", "labels"), files_only=True)
            return str(label_match.resolve()) if label_match is not None else None
        if normalized in annotations_options:
            annotations_match = cls._keyword_path_candidate(
                dataset_files,
                ("annot", "annotation", "annotations", "anno"),
                files_only=True,
            )
            return str(annotations_match.resolve()) if annotations_match is not None else None
        if normalized in split_file_options:
            split_match = cls._keyword_path_candidate(
                dataset_files,
                ("split", "splits", "fold", "folds"),
                files_only=True,
            )
            return str(split_match.resolve()) if split_match is not None else None
        if normalized in metadata_options:
            meta_match = cls._keyword_path_candidate(dataset_files, ("meta", "metadata"), files_only=True)
            return str(meta_match.resolve()) if meta_match is not None else None
        if normalized in image_dir_options:
            image_dir_match = cls._keyword_path_candidate(dataset_dirs, ("image", "images", "img"), dirs_only=True)
            return str(image_dir_match.resolve()) if image_dir_match is not None else None
        if normalized in label_dir_options:
            label_dir_match = cls._keyword_path_candidate(
                dataset_dirs,
                ("label", "labels", "mask", "masks"),
                dirs_only=True,
            )
            return str(label_dir_match.resolve()) if label_dir_match is not None else None
        if normalized in data_file_options:
            fallback = cls._choose_single_path(dataset_files)
            return str(fallback.resolve()) if fallback is not None else None

        model_files = cls._runtime_model_candidates(code_dir, resource_context)
        if normalized in model_file_options:
            match = cls._choose_single_path(model_files)
            return str(match.resolve()) if match is not None else None
        if normalized in tokenizer_options:
            match = cls._choose_single_path([path for path in model_files if "token" in path.name.lower()])
            return str(match.resolve()) if match is not None else None
        if normalized in checkpoint_options or normalized in resume_options:
            checkpoint_files = [
                path for path in model_files if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
            ]
            preferred = [
                path
                for path in checkpoint_files
                if any(token in str(path).lower() for token in ("checkpoint", "checkpoints", "ckpt"))
            ]
            match = cls._choose_single_path(preferred)
            if match is None and preferred:
                match = cls._choose_latest_path(preferred)
            if match is None:
                match = cls._choose_single_path(checkpoint_files)
            if match is None and checkpoint_files:
                match = cls._choose_latest_path(checkpoint_files)
            if match is not None:
                return str(match.resolve())
            fallback = cls._choose_single_path(model_files)
            return str(fallback.resolve()) if fallback is not None else None

        return None

    @staticmethod
    def _upsert_command_option(tokens: list[str], option: str, value: str) -> list[str]:
        updated = list(tokens)
        for index, token in enumerate(updated):
            if token == option:
                if index + 1 < len(updated) and not updated[index + 1].startswith("--"):
                    updated[index + 1] = value
                else:
                    updated.insert(index + 1, value)
                return updated
            if token.startswith(f"{option}="):
                updated[index] = f"{option}={value}"
                return updated
        return [*updated, option, value]

    def _attempt_required_argument_repair(
        self,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        required_options = self._extract_missing_required_options(error_text)
        if not required_options:
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "required_options": []},
            )
            return []

        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        if not runner_config_path.exists():
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "required_options": list(required_options)},
            )
            return []

        try:
            payload = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "required_options": list(required_options)},
            )
            return []

        target_command = payload.get("target_command")
        if not isinstance(target_command, list):
            target_command = []
        updated_command = [str(token) for token in target_command]
        repairs: list[dict[str, str]] = []

        for option in required_options:
            candidate = self._runtime_option_candidate(code_dir, option, resource_context)
            if not candidate:
                continue
            new_command = self._upsert_command_option(updated_command, option, candidate)
            if new_command != updated_command:
                updated_command = new_command
                repairs.append({"option": option, "value": candidate})

        if not repairs:
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "required_options": list(required_options)},
            )
            return []

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            runner_config_path,
            namespace="required_argument_repair",
            root_dir=self.workspace.path,
            operation="rewrite",
        )
        payload["target_command"] = updated_command
        try:
            runner_config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, runner_config_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="required_argument_repair",
                scope=scope or "required_argument_repair",
                snapshots=[snapshot],
                metadata={"modified_files": [], "required_options": list(required_options)},
            )
            return []

        self._record_snapshot_batch(
            mutation_kind="required_argument_repair",
            scope=scope or "required_argument_repair",
            snapshots=[snapshot],
            metadata={
                "modified_files": [str(runner_config_path.relative_to(code_dir))],
                "required_options": list(required_options),
                "repairs": list(repairs),
            },
        )
        return [str(runner_config_path.relative_to(code_dir))]

    def _attempt_resume_repair(
        self,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        failure_signals = self._resume_failure_signals(error_text)
        if not failure_signals:
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "failure_signals": []},
            )
            return []

        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        if not runner_config_path.exists():
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "failure_signals": list(failure_signals)},
            )
            return []

        try:
            payload = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "failure_signals": list(failure_signals)},
            )
            return []

        target_command = payload.get("target_command")
        if not isinstance(target_command, list):
            target_command = []
        updated_command = [str(token) for token in target_command]
        entry_script = self._command_entry_script(updated_command, code_dir)

        option_groups = [
            ["--resume", "--resume-from", "--resume-path"],
            ["--checkpoint", "--ckpt", "--checkpoint-path"],
        ]
        repairs: list[dict[str, str]] = []
        for options in option_groups:
            existing_option, _index, current_value = self._command_option_present(updated_command, options)
            supported = [option for option in options if self._entry_script_supports_flag(entry_script, option)]
            chosen_option = existing_option or (supported[0] if supported else "")
            if not chosen_option:
                continue
            candidate = self._runtime_option_candidate(code_dir, chosen_option, resource_context)
            if not candidate:
                continue
            candidate_variants = self._path_variants(code_dir, candidate)
            current_variants = self._path_variants(code_dir, current_value)
            if current_value and current_variants & candidate_variants and Path(candidate).exists():
                continue
            new_command = self._upsert_command_option(updated_command, chosen_option, candidate)
            if new_command != updated_command:
                repairs.append(
                    {
                        "option": chosen_option,
                        "old_value": current_value,
                        "new_value": candidate,
                    }
                )
                updated_command = new_command
                break

        if not repairs:
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "failure_signals": list(failure_signals)},
            )
            return []

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            runner_config_path,
            namespace="resume_repair",
            root_dir=self.workspace.path,
            operation="rewrite",
        )
        payload["target_command"] = updated_command
        try:
            runner_config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, runner_config_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "resume_repair",
                snapshots=[snapshot],
                metadata={"modified_files": [], "failure_signals": list(failure_signals)},
            )
            return []

        self._record_snapshot_batch(
            mutation_kind="resume_repair",
            scope=scope or "resume_repair",
            snapshots=[snapshot],
            metadata={
                "modified_files": [str(runner_config_path.relative_to(code_dir))],
                "failure_signals": list(failure_signals),
                "repairs": list(repairs),
            },
        )
        return [str(runner_config_path.relative_to(code_dir))]

    def _attempt_cluster_resume_repair(
        self,
        code_dir: Path,
        final_status: str,
        results: dict[str, Any],
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        checkpoints = results.get("checkpoints") if isinstance(results.get("checkpoints"), list) else []
        if not checkpoints and not list((code_dir / "checkpoints").glob("*")):
            self._remember_mutation_snapshot_entry(None)
            self._record_snapshot_batch(
                mutation_kind="resume_repair",
                scope=scope or "cluster_resume_repair",
                snapshots=[],
                metadata={"modified_files": [], "reason": "no_checkpoints"},
            )
            return []

        error_text = "\n".join(
            part
            for part in [
                str(final_status or "").strip(),
                str(results.get("stdout_log") or "").strip(),
                str(results.get("stderr_log") or "").strip(),
            ]
            if part
        )
        return self._attempt_resume_repair(
            code_dir,
            error_text,
            resource_context,
            scope=scope or "cluster_resume_repair",
        )

    def _attempt_option_value_repair(
        self,
        code_dir: Path,
        error_text: str,
        resource_context: dict[str, Any] | None,
        *,
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        missing_targets = self._extract_missing_resource_targets(error_text)
        if not missing_targets:
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[],
                metadata={"modified_files": [], "missing_targets": []},
            )
            return []

        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        if not runner_config_path.exists():
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[],
                metadata={"modified_files": [], "missing_targets": list(missing_targets)},
            )
            return []

        try:
            payload = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[],
                metadata={"modified_files": [], "missing_targets": list(missing_targets)},
            )
            return []

        target_command = payload.get("target_command")
        if not isinstance(target_command, list):
            target_command = []
        updated_command = [str(token) for token in target_command]
        repairs: list[dict[str, str]] = []
        option_groups = [
            ["--config", "--config-path", "--cfg", "--config-file"],
            ["--data-dir", "--data-root", "--dataset-dir", "--dataset-root", "--data", "--dataset"],
            ["--data-path", "--dataset-path", "--input-path", "--input-file", "--dataset-file"],
            ["--train-file", "--train-data", "--train-path"],
            [
                "--val-file",
                "--valid-file",
                "--validation-file",
                "--val-data",
                "--valid-data",
                "--validation-data",
                "--val-path",
                "--valid-path",
                "--dev-file",
                "--dev-data",
                "--dev-path",
            ],
            ["--test-file", "--test-data", "--test-path"],
            ["--labels-path", "--label-file", "--labels-file", "--label-path"],
            ["--annotations", "--annotation-file", "--annotation-path", "--annotations-file"],
            ["--split-file", "--splits-file", "--split-path", "--fold-file", "--folds-file"],
            ["--metadata-path", "--meta-path", "--metadata-file", "--meta-file"],
            ["--image-dir", "--images-dir", "--image-root", "--images-root"],
            ["--label-dir", "--labels-dir", "--label-root", "--labels-root"],
            ["--model-dir", "--model-root"],
            ["--model-path", "--model-file", "--pretrained-model"],
            ["--tokenizer-path", "--tokenizer-name-or-path"],
            ["--checkpoint", "--ckpt", "--checkpoint-path"],
            ["--resume", "--resume-from", "--resume-path"],
        ]
        for options in option_groups:
            option, _index, current_value = self._command_option_present(updated_command, options)
            if not option or not current_value:
                continue
            if not self._option_value_matches_missing_target(code_dir, current_value, missing_targets):
                continue
            candidate = self._runtime_option_candidate(code_dir, option, resource_context)
            if not candidate:
                continue
            if self._path_variants(code_dir, current_value) & self._path_variants(code_dir, candidate):
                continue
            new_command = self._upsert_command_option(updated_command, option, candidate)
            if new_command != updated_command:
                repairs.append({"option": option, "old_value": current_value, "new_value": candidate})
                updated_command = new_command

        if not repairs:
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[],
                metadata={"modified_files": [], "missing_targets": list(missing_targets)},
            )
            return []

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            runner_config_path,
            namespace="option_value_repair",
            root_dir=self.workspace.path,
            operation="rewrite",
        )
        payload["target_command"] = updated_command
        try:
            runner_config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, runner_config_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="option_value_repair",
                scope=scope or "option_value_repair",
                snapshots=[snapshot],
                metadata={"modified_files": [], "missing_targets": list(missing_targets)},
            )
            return []

        self._record_snapshot_batch(
            mutation_kind="option_value_repair",
            scope=scope or "option_value_repair",
            snapshots=[snapshot],
            metadata={
                "modified_files": [str(runner_config_path.relative_to(code_dir))],
                "missing_targets": list(missing_targets),
                "repairs": list(repairs),
            },
        )
        return [str(runner_config_path.relative_to(code_dir))]

    def _attempt_unrecognized_argument_repair(
        self,
        code_dir: Path,
        error_text: str,
        *,
        mode: str = "",
        scope: str = "",
    ) -> list[str]:
        self._remember_mutation_snapshot_entry(None)
        unknown_options = self._extract_unrecognized_options(error_text)
        if not unknown_options:
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "unknown_options": []},
            )
            return []

        runner_config_path = code_dir / RUNNER_CONFIG_NAME
        if not runner_config_path.exists():
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "unknown_options": list(unknown_options)},
            )
            return []

        try:
            payload = json.loads(runner_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "unknown_options": list(unknown_options)},
            )
            return []

        target_command = payload.get("target_command")
        if not isinstance(target_command, list):
            target_command = []
        updated_command = [str(token) for token in target_command]
        blocked_quick_eval = {
            str(option).strip()
            for option in payload.get("quick_eval_blocked_options", [])
            if isinstance(option, str) and str(option).strip().startswith("--")
        }
        removed_options: list[str] = []
        blocked_options_added: list[str] = []

        for option in unknown_options:
            new_command = self._strip_command_option(updated_command, option)
            if new_command != updated_command:
                updated_command = new_command
                removed_options.append(option)
                continue
            if mode == "quick-eval" and option in QUICK_EVAL_AUTO_OPTIONS and option not in blocked_quick_eval:
                blocked_quick_eval.add(option)
                blocked_options_added.append(option)

        if not removed_options and not blocked_options_added:
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[],
                metadata={"modified_files": [], "unknown_options": list(unknown_options)},
            )
            return []

        snapshot = capture_repair_snapshot(
            self.workspace.path,
            runner_config_path,
            namespace="unrecognized_argument_repair",
            root_dir=self.workspace.path,
            operation="rewrite",
        )
        payload["target_command"] = updated_command
        if blocked_quick_eval:
            payload["quick_eval_blocked_options"] = sorted(blocked_quick_eval)
        else:
            payload.pop("quick_eval_blocked_options", None)
        try:
            runner_config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            rollback_snapshot(self.workspace.path, runner_config_path, snapshot)
            snapshot["rolled_back"] = True
            snapshot["rollback_reason"] = "write_error"
            self._record_snapshot_batch(
                mutation_kind="unrecognized_argument_repair",
                scope=scope or "unrecognized_argument_repair",
                snapshots=[snapshot],
                metadata={"modified_files": [], "unknown_options": list(unknown_options)},
            )
            return []

        self._record_snapshot_batch(
            mutation_kind="unrecognized_argument_repair",
            scope=scope or "unrecognized_argument_repair",
            snapshots=[snapshot],
            metadata={
                "modified_files": [str(runner_config_path.relative_to(code_dir))],
                "unknown_options": list(unknown_options),
                "removed_options": list(removed_options),
                "quick_eval_blocked_options": list(blocked_options_added),
            },
        )
        return [str(runner_config_path.relative_to(code_dir))]

    @staticmethod
    def _extract_missing_modules(error_text: str) -> list[str]:
        modules: list[str] = []
        patterns = [
            r"""No module named ['"]([A-Za-z0-9_.-]+)['"]""",
            r"""ModuleNotFoundError:\s*No module named ['"]([A-Za-z0-9_.-]+)['"]""",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, error_text):
                module_name = str(match.group(1)).strip().split(".")[0]
                if module_name and module_name not in modules:
                    modules.append(module_name)
        return modules

    @staticmethod
    def _extract_nltk_resources(error_text: str) -> list[str]:
        resources: list[str] = []
        for pattern in [
            r"""nltk\.download\(['"]([^'"]+)['"]\)""",
            r"""Resource\s+([A-Za-z0-9_./-]+)\s+not found""",
        ]:
            for match in re.finditer(pattern, error_text, re.IGNORECASE):
                resource_name = str(match.group(1)).strip().strip("/")
                if resource_name and resource_name not in resources:
                    resources.append(resource_name)
        return resources

    @classmethod
    def _candidate_package_names(
        cls,
        module_name: str,
        code_dir: Path,
    ) -> list[str]:
        normalized = module_name.strip()
        if not normalized or not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
            return []

        local_module = code_dir / f"{normalized}.py"
        local_package = code_dir / normalized
        if local_module.exists() or local_package.exists():
            return []

        candidates: list[str] = []
        alias = MODULE_PACKAGE_ALIASES.get(normalized.lower())
        if alias:
            candidates.append(alias)
        candidates.append(normalized)
        if "_" in normalized:
            candidates.append(normalized.replace("_", "-"))

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    async def _attempt_runtime_remediation(
        self,
        code_dir: Path,
        error_text: str,
        *,
        runtime_python: str,
        fix_history: list[dict[str, Any]] | None = None,
        execution_policy: ExperimentExecutionPolicy | None = None,
        remediation_ledger: list[dict[str, Any]] | None = None,
        mode: str = "",
        cycle: int | None = None,
        signature: str = "",
        round_number: int | None = None,
    ) -> list[str]:
        policy = execution_policy or RuntimeEnvironmentManager(
            self.config,
            self.log,
        ).build_execution_policy(code_dir)
        actions: list[str] = []

        nltk_resources = self._extract_nltk_resources(error_text)
        remaining_nltk_downloads = policy.remaining_nltk_downloads(fix_history)
        for resource_name in nltk_resources[:remaining_nltk_downloads]:
            result = await self._run_subprocess(
                [
                    runtime_python,
                    "-c",
                    (
                        "import nltk; "
                        f"nltk.download({resource_name!r}, quiet=True, raise_on_error=True)"
                    ),
                ],
                cwd=code_dir,
                timeout=300,
            )
            if result.get("returncode") == 0:
                actions.append(f"nltk:{resource_name}")
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="nltk_download",
                    status="applied",
                    scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    details={"resource": resource_name, "runtime_python": runtime_python},
                )

        if actions:
            return actions
        if nltk_resources and remaining_nltk_downloads <= 0:
            self.log("Skipped NLTK auto-download because the execution policy budget is exhausted")
            self._append_remediation_entry(
                remediation_ledger,
                kind="nltk_download",
                status="skipped",
                scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                round_number=round_number,
                cycle=cycle,
                signature=signature,
                reason="budget_exhausted",
                details={"resources": list(nltk_resources[:3])},
            )

        missing_modules = self._extract_missing_modules(error_text)
        remaining_package_installs = policy.remaining_runtime_auto_installs(fix_history)
        if missing_modules and not policy.runtime_auto_install_enabled:
            self.log("Skipped runtime pip auto-install because it is disabled by execution policy")
            self._append_remediation_entry(
                remediation_ledger,
                kind="pip_install",
                status="skipped",
                scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                round_number=round_number,
                cycle=cycle,
                signature=signature,
                reason="disabled_by_policy",
                details={"modules": list(missing_modules[:3])},
            )
            return actions
        if missing_modules and remaining_package_installs <= 0:
            self.log("Skipped runtime pip auto-install because the execution policy budget is exhausted")
            self._append_remediation_entry(
                remediation_ledger,
                kind="pip_install",
                status="skipped",
                scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                round_number=round_number,
                cycle=cycle,
                signature=signature,
                reason="budget_exhausted",
                details={"modules": list(missing_modules[:3])},
            )
            return actions

        for module_name in missing_modules[:3]:
            allowed_candidates = [
                package_name
                for package_name in self._candidate_package_names(module_name, code_dir)
                if policy.allows_runtime_package(
                    package_name,
                    module_name=module_name,
                    aliases=MODULE_PACKAGE_ALIASES,
                )
            ]
            if not allowed_candidates:
                self.log(
                    "Skipped runtime pip auto-install for missing module "
                    f"{module_name!r}: package is not declared or allowlisted"
                )
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="pip_install",
                    status="skipped",
                    scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    reason="not_declared_or_allowlisted",
                    details={"module": module_name},
                )
                continue
            for package_name in allowed_candidates:
                if remaining_package_installs <= 0:
                    break
                result = await self._run_subprocess(
                    [runtime_python, "-m", "pip", "install", package_name],
                    cwd=code_dir,
                    timeout=900,
                )
                if result.get("returncode") == 0:
                    actions.append(f"pip:{package_name}")
                    remaining_package_installs -= 1
                    self._append_remediation_entry(
                        remediation_ledger,
                        kind="pip_install",
                        status="applied",
                        scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                        round_number=round_number,
                        cycle=cycle,
                        signature=signature,
                        details={"module": module_name, "package": package_name},
                    )
                    break
                self._append_remediation_entry(
                    remediation_ledger,
                    kind="pip_install",
                    status="failed",
                    scope=f"local_{mode.replace('-', '_')}" if mode else "local_runtime",
                    round_number=round_number,
                    cycle=cycle,
                    signature=signature,
                    details={
                        "module": module_name,
                        "package": package_name,
                        "returncode": result.get("returncode"),
                        "stderr": str(result.get("stderr") or "")[:300],
                    },
                )

        return actions

    @classmethod
    def _summarize_available_resources(
        cls,
        code_dir: Path,
        resource_context: dict[str, Any] | None,
    ) -> str:
        candidates = cls._collect_resource_candidates(code_dir, resource_context)
        if not candidates:
            return ""

        lines: list[str] = []
        for item in candidates[:8]:
            lines.append(f"- [{item['kind']}] {item['path']}")
        return "Available workspace resources:\n" + "\n".join(lines)
