"""ReAct experiment mode — LLM drives experiment via tool calls."""
from __future__ import annotations

import json
import logging
from typing import Any

from nanoresearch.agents.experiment_tools import build_experiment_tools

logger = logging.getLogger(__name__)


class _ReactModeMixin:
    """Mixin — ReAct experiment mode."""

    async def _run_react_mode(
        self,
        blueprint_data: dict,
        reference_repos: list[dict],
    ) -> dict[str, Any]:
        """Run experiment in ReAct mode — LLM drives everything via tools."""
        self.log("Starting experiment in ReAct mode (LLM-driven)")

        code_dir = self.workspace.path / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        # Build allowed env set for isolation enforcement.
        # Only activate isolation when user has explicitly chosen an env.
        # None = no restriction; frozenset(...) = only these envs + nanoresearch_*
        _allowed: frozenset[str] | None = None
        if self.config.experiment_conda_env:
            _allowed = frozenset({self.config.experiment_conda_env, "base"})

        # Build tools
        tools = build_experiment_tools(work_dir=code_dir, allowed_envs=_allowed)

        # Build SLURM config block for system prompt
        partition = self.config.slurm_partition
        max_gpus = self.config.slurm_max_gpus
        wall_time = self.config.slurm_default_time
        if partition:
            slurm_config = (
                f"- Partition: `{partition}`\n"
                f"- Max GPUs per job: {max_gpus}\n"
                f"- Default wall time: {wall_time}\n"
                f"- Submit with: `sbatch your_script.sh`\n"
                f"- Check status: `squeue -u $(whoami)`\n"
                f"- Cancel job: `scancel <job_id>`\n"
                f"- View logs: read the SLURM output file (usually `slurm-<job_id>.out`)"
            )
        else:
            slurm_config = (
                "Not pre-configured. Run `sinfo` to check if SLURM is available.\n"
                f"If available, use at most {max_gpus} GPUs per job."
            )

        # Build conda env hint for system prompt
        conda_env = self.config.experiment_conda_env
        if conda_env:
            conda_env_hint = (
                f"\n**MANDATORY conda env**: `{conda_env}` — you MUST use this env "
                f"(activate with `conda activate {conda_env}`). "
                f"Do NOT use any other existing conda environment. "
                f"This env was explicitly chosen by the user. "
                f"If it lacks packages, install them into THIS env.\n"
            )
        else:
            conda_env_hint = (
                "\n**No pre-configured env** — create a new per-session venv or "
                "conda env (prefix name with `nanoresearch_`). "
                "Do NOT use any existing user conda environments.\n"
            )

        # Build container config block for system prompt
        container_image = self.config.container_image
        container_path = self.config.container_path
        container_bind = self.config.container_bind

        # Common explanation block
        _why = (
            "**WHY containers?** HPC clusters often have old glibc (e.g., 2.17 on CentOS 7).\n"
            "Modern PyTorch/CUDA needs glibc >= 2.28. Direct `pip install torch` fails with\n"
            "`GLIBC_2.xx not found`. Containers (e.g., Ubuntu 22.04) bundle glibc 2.35 inside.\n"
        )

        # Search paths for existing .sif files
        _search_dirs = "/mnt /opt /shared /data $HOME"

        container_lines = [_why]

        # Step 1: Search
        container_lines.extend([
            "**Step 1: Search for existing .sif files on the cluster**",
            f"```",
            f"find {_search_dirs} -name '*.sif' -maxdepth 4 2>/dev/null | head -20",
            f"```",
        ])
        if container_path:
            container_lines.append(
                f"Pre-configured path to check first: `{container_path}`"
            )

        # Step 2: Try each
        container_lines.extend([
            "",
            "**Step 2: Try each .sif file — test if it has a usable Python + PyTorch**",
            "For EACH .sif file found, test:",
            "```",
            "apptainer exec --nv FOUND.sif python3 -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"",
            "```",
            "- If it prints a torch version with `True` → **use this .sif, you are done!**",
            "- If it fails (no python, no torch, wrong CUDA) → try the next .sif",
            "- If no python3, try `python` instead",
        ])

        # Step 3: Download if none work
        container_lines.extend([
            "",
            "**Step 3: If NO usable .sif found → download a clean base image**",
        ])
        if container_image:
            container_lines.append(
                f"Pre-configured image: `{container_image}`"
            )
            sif_target = container_path or "ubuntu2204.sif"
            container_lines.append(
                f"```\napptainer pull {sif_target} {container_image}\n```"
            )
        else:
            container_lines.extend([
                "Download a clean Ubuntu 22.04 image (small, ~30MB, glibc 2.35):",
                "```",
                "apptainer pull ubuntu2204.sif docker://ubuntu:22.04",
                "```",
                "(use `timeout=1800` — first pull may be slow)",
            ])

        # Step 4: Install Python + deps inside
        container_lines.extend([
            "",
            "**Step 4: Install Python + PyTorch inside the clean container**",
            "Use `--writable-tmpfs` to allow temporary writes inside the read-only .sif:",
            "```",
            "# Test if python3 exists inside",
            "apptainer exec ubuntu2204.sif which python3",
            "",
            "# If no python3, install it (needs --writable-tmpfs or --fakeroot):",
            "apptainer exec --writable-tmpfs ubuntu2204.sif bash -c \\",
            '  "apt-get update -qq && apt-get install -y -qq python3 python3-pip > /dev/null && \\',
            '   pip3 install torch torchvision numpy -q && \\',
            '   python3 -c \\"import torch; print(torch.__version__)\\"" ',
            "```",
            "",
            "**BETTER: Build a reusable .sif with a definition file** (so you don't reinstall every time):",
            "```",
            "# Write a .def file",
            "cat > experiment.def << 'DEFEOF'",
            "Bootstrap: docker",
            "From: ubuntu:22.04",
            "",
            "%post",
            "    apt-get update -qq && apt-get install -y -qq python3 python3-pip git > /dev/null",
            "    pip3 install torch torchvision numpy scipy scikit-learn matplotlib -q",
            "",
            "%environment",
            "    export PATH=/usr/local/bin:/usr/bin:$PATH",
            "DEFEOF",
            "",
            "# Build the .sif (use --fakeroot on HPC clusters without root)",
            "apptainer build --fakeroot experiment.sif experiment.def",
            "```",
            "If `--fakeroot` fails, use `--writable-tmpfs` approach instead.",
        ])

        # Step 5: Usage
        container_lines.extend([
            "",
            "**Step 5: Use the container for ALL commands**",
            f"Bind mounts: `-B {container_bind}`",
            "```",
            "# Run python inside container",
            "apptainer exec --nv -B {bind} experiment.sif python3 main.py --quick-eval".format(
                bind=container_bind
            ),
            "",
            "# Install extra packages at runtime (--writable-tmpfs)",
            "apptainer exec --nv --writable-tmpfs -B {bind} experiment.sif bash -c \\".format(
                bind=container_bind
            ),
            '  "pip3 install -r requirements.txt -q && python3 main.py --quick-eval"',
            "```",
            "IMPORTANT: once in container mode, ALL python/pip must go through `apptainer exec`.",
        ])

        container_config = "\n".join(container_lines)

        # Fill variables in the loaded YAML template via str.replace
        system_prompt = self._REACT_SYSTEM_TEMPLATE
        for _key, _val in [
            ("slurm_config", slurm_config),
            ("conda_env_hint", conda_env_hint),
            ("container_config", container_config),
        ]:
            system_prompt = system_prompt.replace(f"{{{_key}}}", _val)

        # Build user prompt with blueprint
        blueprint_summary = json.dumps(blueprint_data, indent=2, ensure_ascii=False)
        if len(blueprint_summary) > 6000:
            blueprint_summary = blueprint_summary[:6000] + "\n... (truncated)"

        repo_context = self._build_repo_context(reference_repos)
        repo_block = f"\n\n## Reference code\n{repo_context}" if repo_context else ""

        user_prompt = f"""## Experiment Blueprint

{blueprint_summary}
{repo_block}

## Working directory
`{code_dir}`

Please start by discovering the environment (Phase 0), then implement and run this experiment.
The goal is to get real experimental results (metric numbers) — not placeholder code."""

        # Run the ReAct loop
        max_rounds = self.config.react_max_rounds
        self.log(f"ReAct loop: max {max_rounds} tool rounds")

        try:
            final_output = await self.generate_with_tools(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                max_tool_rounds=max_rounds,
                stage_override=self.config.for_stage("code_gen"),
                reminder_text=(
                    "[REMINDER] You are running an ML experiment. Stay focused:\n"
                    "- If the experiment is still running, check its status (squeue / read log file)\n"
                    "- If it failed, read the error and fix the code\n"
                    "- If it succeeded, collect the results and report metrics\n"
                    "- Do NOT start over from scratch unless absolutely necessary\n"
                    "- Your goal is REAL metric numbers, not placeholder code"
                ),
                reminder_interval=5,
            )
        except Exception as exc:
            logger.error("ReAct experiment failed: %s", exc, exc_info=True)
            final_output = f"ReAct experiment failed with error: {exc}"

        self.log(f"ReAct experiment completed. Output length: {len(final_output)}")
        self.workspace.write_text("logs/react_final_output.md", final_output)

        # Try to collect metrics from results/metrics.json
        metrics = self._parse_metrics_json(code_dir)
        experiment_status = "success" if metrics else "partial"

        result = {
            "code_project_plan": {"mode": "react"},
            "generated_files": [
                str(f.relative_to(code_dir))
                for f in code_dir.rglob("*")
                if f.is_file() and "__pycache__" not in str(f)
            ],
            "file_count": sum(
                1 for f in code_dir.rglob("*")
                if f.is_file() and "__pycache__" not in str(f)
            ),
            "code_execution": {"status": experiment_status},
            "experiment_results": metrics,
            "experiment_status": experiment_status,
            "react_output": final_output[:5000],
        }
        self.workspace.write_json("logs/experiment_output.json", result)
        return result
