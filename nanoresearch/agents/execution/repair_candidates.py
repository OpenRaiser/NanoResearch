"""Path candidate finding for repair."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.project_runner import RUNNER_CONFIG_NAME

logger = logging.getLogger(__name__)


class _RepairCandidatesMixin:
    """Mixin — path candidate finding and option candidate resolution."""

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
            haystacks_str = " ".join(haystacks)
            score = sum(1 for keyword in normalized_keywords if keyword in haystacks_str)
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
