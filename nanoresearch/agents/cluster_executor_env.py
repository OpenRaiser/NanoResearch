"""Cluster executor env mixin -- conda env validation, repair, setup."""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any

from nanoresearch.agents.constants import (
    CLUSTER_ENV_VALIDATION_TIMEOUT,
    ENV_SETUP_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Re-import manifest constants from main module at call time to avoid circular
ENVIRONMENT_MANIFESTS = ("environment.yml", "environment.yaml")
PIP_MANIFESTS = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")


class _ClusterExecutorEnvMixin:
    """Mixin: conda env validation, repair, manifest resolution, and setup_env."""

    async def _validate_cluster_env(
        self,
        cluster_code_path: str,
        *,
        conda_sh: str,
        install_kind: str,
    ) -> dict:
        activate_prefix = self._activate_prefix(conda_sh, pipefail=True)

        python_script = (
            "import json, sys; "
            "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0]}))"
        )
        python_result = await self._run_cmd(
            f"{activate_prefix}python -c {shlex.quote(python_script)}",
            timeout=CLUSTER_ENV_VALIDATION_TIMEOUT,
        )
        python_payload = self._parse_json_tail(python_result.get("stdout", ""))
        python_smoke = {
            "status": "passed" if python_result.get("returncode") == 0 else "failed",
            "returncode": python_result.get("returncode"),
            "stderr": str(python_result.get("stderr") or "")[:300],
            "executable": str(python_payload.get("executable") or ""),
            "version": str(python_payload.get("version") or ""),
        }
        if python_smoke["status"] != "passed":
            return {
                "status": "failed",
                "python_smoke": python_smoke,
                "pip_probe": {"status": "skipped"},
                "import_probe": {"status": "skipped"},
            }

        pip_result = await self._run_cmd(
            f"{activate_prefix}python -m pip --version",
            timeout=CLUSTER_ENV_VALIDATION_TIMEOUT,
        )
        pip_probe = {
            "status": "passed" if pip_result.get("returncode") == 0 else "failed",
            "returncode": pip_result.get("returncode"),
            "version": str(pip_result.get("stdout") or "").strip()[:200],
            "stderr": str(pip_result.get("stderr") or "")[:300],
        }
        if pip_probe["status"] != "passed":
            return {
                "status": "failed",
                "python_smoke": python_smoke,
                "pip_probe": pip_probe,
                "import_probe": {"status": "skipped"},
            }

        probe_targets, skipped_reason = self._select_cluster_import_probe_targets(
            cluster_code_path,
            install_kind=install_kind,
        )
        if not probe_targets:
            import_probe = {
                "status": "skipped",
                "targets": [],
                "failures": [],
                "skipped_reason": skipped_reason,
            }
            return {
                "status": "ready",
                "python_smoke": python_smoke,
                "pip_probe": pip_probe,
                "import_probe": import_probe,
            }

        import_script = "\n".join(
            [
                "import importlib",
                "import json",
                f"targets = {json.dumps(probe_targets, ensure_ascii=False)}",
                "results = []",
                "for item in targets:",
                "    package = item['package']",
                "    module = item['module']",
                "    try:",
                "        importlib.import_module(module)",
                "        results.append({'package': package, 'module': module, 'status': 'passed'})",
                "    except Exception as exc:",
                "        results.append({",
                "            'package': package,",
                "            'module': module,",
                "            'status': 'failed',",
                "            'error': f'{exc.__class__.__name__}: {exc}',",
                "        })",
                "print(json.dumps({'results': results}, ensure_ascii=False))",
            ]
        )
        import_result = await self._run_cmd(
            f"{activate_prefix}python -c {shlex.quote(import_script)}",
            timeout=CLUSTER_ENV_VALIDATION_TIMEOUT,
        )
        parsed_payload = self._parse_json_tail(import_result.get("stdout", ""))
        parsed_results = parsed_payload.get("results", []) if isinstance(parsed_payload, dict) else []
        if not isinstance(parsed_results, list):
            parsed_results = []
        failures = [item for item in parsed_results if isinstance(item, dict) and item.get("status") != "passed"]
        import_status = "passed"
        if import_result.get("returncode") != 0 and not failures:
            import_status = "failed"
            failures = [
                {
                    "package": "",
                    "module": "",
                    "error": str(import_result.get("stderr") or "")[:300],
                }
            ]
        elif failures:
            import_status = "partial"

        return {
            "status": "ready" if import_status == "passed" else ("failed" if import_status == "failed" else "partial"),
            "python_smoke": python_smoke,
            "pip_probe": pip_probe,
            "import_probe": {
                "status": import_status,
                "targets": list(probe_targets),
                "results": list(parsed_results),
                "failures": list(failures),
                "stderr": str(import_result.get("stderr") or "")[:300],
                "skipped_reason": "",
            },
        }

    async def _repair_cluster_validation(
        self,
        cluster_code_path: str,
        *,
        conda_sh: str,
        install_kind: str,
        validation: dict,
    ) -> dict:
        from nanoresearch.agents.cluster_executor import MAX_CLUSTER_VALIDATION_REPAIR_PACKAGES
        failed_packages = self._extract_failed_import_packages(validation)
        if not failed_packages:
            return {"validation": validation, "repair": {"status": "skipped", "actions": []}}

        spec_index = self._manifest_repair_specs.get(cluster_code_path, {})
        repair_specs: list[str] = []
        unresolved: list[str] = []
        for package_name in failed_packages:
            spec = spec_index.get(package_name)
            if spec and spec not in repair_specs:
                repair_specs.append(spec)
            else:
                unresolved.append(package_name)
            if len(repair_specs) >= MAX_CLUSTER_VALIDATION_REPAIR_PACKAGES:
                break

        repair_actions: list[dict] = []
        current_validation = validation
        if repair_specs:
            install_cmd = (
                f"{self._activate_prefix(conda_sh, pipefail=True)}"
                f"pip install {' '.join(shlex.quote(spec) for spec in repair_specs)} 2>&1 | tail -40"
            )
            result = await self._run_cmd(install_cmd, timeout=ENV_SETUP_TIMEOUT)
            action = {
                "kind": "import_repair_install",
                "status": "installed" if result.get("returncode") == 0 else "failed",
                "specs": list(repair_specs),
                "returncode": result.get("returncode"),
                "stderr": str(result.get("stderr") or "")[:300],
            }
            repair_actions.append(action)
            if result.get("returncode") == 0:
                current_validation = await self._validate_cluster_env(
                    cluster_code_path,
                    conda_sh=conda_sh,
                    install_kind=install_kind,
                )
        elif unresolved:
            repair_actions.append(
                {
                    "kind": "import_repair_skipped",
                    "status": "skipped",
                    "packages": list(unresolved),
                }
            )

        final_status = str(current_validation.get("status") or "").strip()
        if not repair_actions:
            repair_status = "skipped"
        elif final_status == "ready":
            repair_status = "applied"
        elif any(action.get("status") == "failed" for action in repair_actions):
            repair_status = "failed"
        else:
            repair_status = "partial"
        return {
            "validation": current_validation,
            "repair": {
                "status": repair_status,
                "actions": repair_actions,
            },
        }

    @staticmethod
    def _probe_manifest_names(stdout: str) -> set[str]:
        return {line.strip() for line in stdout.splitlines() if line.strip()}

    def _resolve_manifest_policy(
        self,
        cluster_code_path: str,
        probe_stdout: str,
    ) -> tuple[str, str, str, str, str]:
        found = self._probe_manifest_names(probe_stdout)
        snapshot = self._manifest_snapshots.get(cluster_code_path)
        if snapshot is not None:
            expected = {
                name
                for name in (snapshot.environment_source, snapshot.install_source)
                if name
            }
            if not expected or expected.issubset(found):
                manifest_name = snapshot.install_source or snapshot.environment_source
                manifest_kind = "conda" if snapshot.install_kind in {"", "environment"} else "pip"
                environment_name = snapshot.environment_source
                self.log(
                    "Using cached local manifest policy for cluster env setup: "
                    f"install={snapshot.install_source or 'none'}, env={snapshot.environment_source or 'none'}"
                )
                return (
                    manifest_name,
                    manifest_kind if manifest_name else "",
                    environment_name,
                    "cached_local_manifest",
                    snapshot.install_kind,
                )
            self.log(
                "Cluster manifest probe does not match cached local policy; "
                "falling back to remote probe selection"
            )

        manifest_name, manifest_kind = self._select_manifest_from_probe(probe_stdout)
        environment_name = next(
            (name for name in ENVIRONMENT_MANIFESTS if name in found),
            "",
        )
        return manifest_name, manifest_kind, environment_name, "remote_probe", ""

    @staticmethod
    def _manifest_probe_command(cluster_code_path: str) -> str:
        quoted_dir = shlex.quote(cluster_code_path)
        checks = [*PIP_MANIFESTS, *ENVIRONMENT_MANIFESTS]
        probe_lines = [
            f'if [ -f {quoted_dir}/{name} ]; then echo "{name}"; fi'
            for name in checks
        ]
        return " ".join(probe_lines) or "true"

    @staticmethod
    def _select_manifest_from_probe(stdout: str) -> tuple[str, str]:
        found = {line.strip() for line in stdout.splitlines() if line.strip()}
        for name in PIP_MANIFESTS:
            if name in found:
                return name, "pip"
        for name in ENVIRONMENT_MANIFESTS:
            if name in found:
                return name, "conda"
        return "", ""

    async def setup_env(self, cluster_code_path: str) -> dict:
        """Create/update the cluster conda env and install project dependencies."""
        self.log(f"Setting up conda env '{self.conda_env}'...")
        quoted_code_path = shlex.quote(cluster_code_path)

        detect = (
            "CONDA_SH=$HOME/anaconda3/etc/profile.d/conda.sh; "
            "[ ! -f $CONDA_SH ] && CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh; "
            "[ ! -f $CONDA_SH ] && CONDA_SH=$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh; "
            "echo $CONDA_SH"
        )
        detect_result = await self._run_cmd(detect, timeout=15)
        conda_sh = detect_result["stdout"].strip()
        if not conda_sh or "No such" in conda_sh:
            conda_sh = "~/anaconda3/etc/profile.d/conda.sh"
        self._conda_sh = conda_sh
        self.log(f"Using conda: {conda_sh}")

        manifest_probe = await self._run_cmd(
            self._manifest_probe_command(cluster_code_path),
            timeout=10,
        )
        manifest_name, manifest_kind, environment_name, policy_source, install_kind = self._resolve_manifest_policy(
            cluster_code_path,
            manifest_probe.get("stdout", ""),
        )
        environment_path = (
            f"{cluster_code_path}/{environment_name}" if environment_name else ""
        )

        check_env = (
            f"source {conda_sh} 2>/dev/null && "
            f"conda env list | grep -w {self.conda_env} && echo ENV_EXISTS || echo ENV_MISSING"
        )
        check_result = await self._run_cmd(check_env, timeout=30)
        env_missing = "ENV_MISSING" in check_result.get("stdout", "")

        env_cmd = ""
        env_strategy = "existing"
        if env_missing:
            if environment_path:
                self.log(f"Creating conda env '{self.conda_env}' from {environment_name}...")
                env_cmd = (
                    "set -o pipefail; "
                    f"source {conda_sh} && "
                    f"conda env create -n {self.conda_env} -f {shlex.quote(environment_path)} "
                    f"2>&1 | tail -40"
                )
                env_strategy = "conda_env_create"
            else:
                self.log(f"Creating conda env '{self.conda_env}' (python={self.python_version})...")
                env_cmd = (
                    "set -o pipefail; "
                    f"source {conda_sh} && "
                    f"conda create -n {self.conda_env} python={self.python_version} -y "
                    f"2>&1 | tail -40"
                )
                env_strategy = "conda_create"
        elif environment_path:
            self.log(f"Updating conda env '{self.conda_env}' from {environment_name}...")
            env_cmd = (
                "set -o pipefail; "
                f"source {conda_sh} && "
                f"conda env update -n {self.conda_env} -f {shlex.quote(environment_path)} --prune "
                f"2>&1 | tail -40"
            )
            env_strategy = "conda_env_update"
        else:
            self.log(f"Conda env '{self.conda_env}' already exists")

        env_output = ""
        if env_cmd:
            env_result = await self._run_cmd(env_cmd, timeout=ENV_SETUP_TIMEOUT)
            env_output = (env_result.get("stdout", "") + "\n" + env_result.get("stderr", "")).strip()
            if env_result.get("returncode") != 0:
                self.log(f"Conda env preparation failed (rc={env_result['returncode']})")
                self.log(env_output[-500:])
                return {
                    "ok": False, "output": env_output, "manifest": environment_path,
                    "source": environment_name, "strategy": env_strategy,
                    "policy_source": policy_source,
                }

        async def finalize_success(*, output, manifest, source, strategy):
            rv = await self._validate_cluster_env(cluster_code_path, conda_sh=conda_sh, install_kind=install_kind)
            rr = await self._repair_cluster_validation(cluster_code_path, conda_sh=conda_sh, install_kind=install_kind, validation=rv)
            rv = rr["validation"]
            summary = self._format_runtime_validation_summary(rv, rr["repair"])
            combined = "\n".join(p for p in [output.strip(), summary] if p).strip()
            return {
                "ok": rv.get("status") == "ready", "output": combined,
                "manifest": manifest, "source": source, "strategy": strategy,
                "policy_source": policy_source, "runtime_validation": rv,
                "runtime_validation_repair": rr["repair"],
            }

        if not manifest_name:
            self.log("No dependency manifest found, skipping dependency install")
            return await finalize_success(output=env_output or "No dependency manifest", manifest="", source="", strategy=env_strategy)

        if manifest_kind == "conda" or install_kind == "environment":
            return await finalize_success(output=env_output or f"Applied {manifest_name}", manifest=environment_path, source=manifest_name, strategy=env_strategy)

        manifest_path = f"{cluster_code_path}/{manifest_name}"
        activate_prefix = (
            "set -o pipefail; "
            f"source {conda_sh} && "
            f"conda activate {self.conda_env} && "
            f"type proxy_on &>/dev/null && proxy_on; "
        )

        install_attempts: list[tuple[str, str]] = []
        if manifest_name == "requirements.txt":
            install_attempts.append(("requirements", f"{activate_prefix}pip install -r {shlex.quote(manifest_path)} 2>&1 | tail -40"))
        else:
            install_attempts.append(("editable", f"{activate_prefix}pip install -e {quoted_code_path} 2>&1 | tail -40"))
            install_attempts.append(("package", f"{activate_prefix}pip install {quoted_code_path} 2>&1 | tail -40"))

        last_output = env_output
        for strategy_name, install_cmd in install_attempts:
            self.log(f"Installing dependencies from {manifest_name} ({strategy_name})...")
            result = await self._run_cmd(install_cmd, timeout=ENV_SETUP_TIMEOUT)
            last_output = ((env_output + "\n") if env_output else "") + result.get("stdout", "") + "\n" + result.get("stderr", "")
            if result.get("returncode") == 0:
                self.log("Dependency install completed successfully")
                return await finalize_success(output=last_output.strip(), manifest=manifest_path, source=manifest_name, strategy=strategy_name)
            self.log(f"Dependency install failed via {manifest_name} ({strategy_name}), rc={result['returncode']}")
            self.log(last_output[-500:])

        return {
            "ok": False, "output": last_output.strip(),
            "manifest": manifest_path, "source": manifest_name,
            "strategy": install_attempts[-1][0], "policy_source": policy_source,
            "runtime_validation": {"status": "skipped"},
            "runtime_validation_repair": {"status": "skipped", "actions": []},
        }
