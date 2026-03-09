"""Tests for structured preflight diagnostics."""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from nanoresearch.agents.preflight import PreflightChecker


def test_run_all_collects_blocking_messages_and_suggested_fixes() -> None:
    tmp_dir = Path(f".test_preflight_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        (code_dir / "config").mkdir(parents=True)
        (code_dir / "config" / "default.yaml").write_text("model: demo\n", encoding="utf-8")
        (code_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        (code_dir / "main.py").write_text(
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--dry-run', action='store_true')\n"
            "parser.add_argument('--quick-eval', action='store_true')\n"
            "synthetic = True\n",
            encoding="utf-8",
        )

        report = PreflightChecker(code_dir).run_all()

        assert report.overall_status == "failed"
        assert report.blocking_check_names == ["config_yaml"]
        assert report.blocking_failures
        assert report.blocking_failures[0].startswith("config_yaml:")
        assert any("random_seed" in fix for fix in report.suggested_fixes)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_check_requirements_accepts_pyproject_manifest() -> None:
    tmp_dir = Path(f".test_preflight_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=61"]',
                    'build-backend = "setuptools.build_meta"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = PreflightChecker(code_dir).check_requirements()

        assert result.status == "passed"
        assert result.details["manifest"].endswith("pyproject.toml")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_check_requirements_warns_when_environment_yaml_has_no_pip_block() -> None:
    tmp_dir = Path(f".test_preflight_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "environment.yaml").write_text(
            "\n".join(
                [
                    "name: demo",
                    "dependencies:",
                    "  - python=3.10",
                    "  - pytorch",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = PreflightChecker(code_dir).check_requirements()

        assert result.status == "warning"
        assert result.details["manifest"].endswith("environment.yaml")
        assert result.details["suggested_fixes"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_check_main_entrypoint_reports_structured_missing_flags() -> None:
    tmp_dir = Path(f".test_preflight_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

        result = PreflightChecker(code_dir).check_main_entrypoint()

        assert result.status == "warning"
        assert result.details["entrypoint_path"].endswith("train.py")
        assert result.details["missing_flags"] == ["--dry-run", "--quick-eval"]
        assert result.details["suggested_fixes"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
