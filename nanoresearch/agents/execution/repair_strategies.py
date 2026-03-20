"""Argument repair strategies."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.project_runner import RUNNER_CONFIG_NAME

from nanoresearch.agents.repair_journal import capture_repair_snapshot, rollback_snapshot

logger = logging.getLogger(__name__)


class _RepairStrategiesMixin:
    """Mixin — argument and option repair strategies."""

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
