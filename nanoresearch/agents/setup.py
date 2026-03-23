"""Setup agent — searches GitHub for relevant code, clones repos, downloads models/data.

Uses a global cache at ~/.nanoresearch/cache/ so models/data are shared across pipeline runs.
Downloads models from ModelScope first (faster in China), falls back to HuggingFace.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import re
import shlex
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

from .setup_search import _SetupSearchMixin
from .setup_github import _SetupGithubMixin

logger = logging.getLogger(__name__)

# Global cache directory — shared across all pipeline runs
GLOBAL_CACHE_DIR = Path.home() / ".nanoresearch" / "cache"
GLOBAL_MODELS_DIR = GLOBAL_CACHE_DIR / "models"
GLOBAL_DATA_DIR = GLOBAL_CACHE_DIR / "data"
SUCCESS_RESOURCE_STATUSES = {"downloaded", "full", "config_only"}

# Regex for GitHub repo URLs (not raw file links)
_GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+?)(?:\.git)?/?$"
)
# Patterns for extracting real download URLs from README / scripts inside a dataset repo
_DOWNLOAD_URL_RE = re.compile(
    r"(https?://(?:"
    r"drive\.google\.com/[^\s\)\]\"'>]+|"          # Google Drive
    r"docs\.google\.com/[^\s\)\]\"'>]+|"            # Google Docs exports
    r"dl\.fbaipublicfiles\.com/[^\s\)\]\"'>]+|"     # Meta / FAIR
    r"zenodo\.org/record[^\s\)\]\"'>]+|"            # Zenodo
    r"zenodo\.org/api/records[^\s\)\]\"'>]+|"        # Zenodo API
    r"huggingface\.co/datasets/[^\s\)\]\"'>]+|"     # HuggingFace datasets
    r"storage\.googleapis\.com/[^\s\)\]\"'>]+|"     # GCS
    r"s3\.amazonaws\.com/[^\s\)\]\"'>]+|"           # S3
    r"(?:[a-z0-9-]+\.)?s3[.-][^\s\)\]\"'>]+|"      # S3 regional
    r"dropbox\.com/[^\s\)\]\"'>]+|"                 # Dropbox
    r"figshare\.com/[^\s\)\]\"'>]+|"                # Figshare
    r"data\.dgl\.ai/[^\s\)\]\"'>]+|"                # DGL datasets
    r"people\.csail\.mit\.edu/[^\s\)\]\"'>]+|"      # MIT
    r"[^\s\)\]\"'>]+\.(?:zip|tar\.gz|tgz|tar\.bz2|gz|csv|tsv|json|jsonl|h5|hdf5|pt|pkl|npy|npz|parquet|txt)"
    r")"
    r")",
    re.IGNORECASE,
)


class SetupAgent(_SetupSearchMixin, _SetupGithubMixin, BaseResearchAgent):
    """Searches for relevant code repos, clones them, and downloads required resources."""

    stage = PipelineStage.SETUP

    @property
    def stage_config(self):
        """Reuse experiment-stage model routing for setup planning."""
        return self.config.for_stage("experiment")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        topic: str = inputs["topic"]
        ideation_output: dict = inputs.get("ideation_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})

        self.log("Starting setup: code search + resource download")

        # Step 1: Search GitHub for relevant repos
        search_plan = await self._plan_search(topic, ideation_output, experiment_blueprint)
        search_plan = self._augment_search_plan_with_blueprint_resources(
            search_plan,
            experiment_blueprint,
        )
        self.log(f"Search plan: {json.dumps(search_plan, indent=2)[:500]}")

        # Step 2: Search and clone repos
        cloned_repos = await self._search_and_clone(search_plan)
        self.log(f"Cloned {len(cloned_repos)} repos")

        # Step 3: Analyze cloned code
        code_analysis = await self._analyze_cloned_code(cloned_repos, experiment_blueprint)

        # Step 4: Download required resources (models, datasets)
        # Datasets → workspace-local `datasets/` dir (each task gets its own copy)
        # Models  → global cache (large, reusable across runs)
        datasets_dir = self.workspace.path / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        GLOBAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if self.config.auto_download_resources:
            resources = await self._download_resources(
                search_plan, datasets_dir, GLOBAL_MODELS_DIR
            )
        else:
            self.log("Automatic resource download disabled, skipping dataset/model fetch")
            resources = []

        # Workspace directories for generated code to reference
        data_dir = datasets_dir  # datasets live here directly, no symlink needed
        models_dir = self.workspace.path / "models"
        models_dir.mkdir(exist_ok=True)

        # Verify downloads — check file sizes
        verified_resources = []
        for r in resources:
            path = r.get("path", "")
            if path and Path(path).exists():
                if Path(path).is_dir():
                    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
                else:
                    size = Path(path).stat().st_size
                r["size_bytes"] = size
                if size == 0:
                    r["status"] = "empty"
                    self.log(f"WARNING: {r['name']} downloaded but file is empty!")
            verified_resources.append(r)

        # Check if all blueprint datasets were downloaded
        blueprint_datasets = {
            (ds.get("name", "") if isinstance(ds, dict) else str(ds)).lower().strip()
            for ds in experiment_blueprint.get("datasets", [])
        }
        downloaded_names = {
            r.get("name", "").lower().strip()
            for r in verified_resources
            if r.get("status") in ("downloaded", "full", "config_only")
        }
        missing_datasets = blueprint_datasets - downloaded_names
        if missing_datasets:
            self.log(f"WARNING: Blueprint datasets not downloaded: {missing_datasets}")
            # Add explicit entries so CODING knows these are unavailable
            for name in missing_datasets:
                if not any(r.get("name", "").lower().strip() == name for r in verified_resources):
                    verified_resources.append({
                        "name": name,
                        "type": "dataset",
                        "status": "not_downloaded",
                        "error": "Not found by SETUP agent",
                    })

        # Stage only models from cache → workspace (datasets are already local)
        staged_resources, workspace_aliases = self._stage_workspace_resources(
            verified_resources,
            data_dir,
            models_dir,
        )

        result = {
            "search_plan": search_plan,
            "cloned_repos": cloned_repos,
            "code_analysis": code_analysis,
            "downloaded_resources": staged_resources,
            "datasets_dir": str(datasets_dir),
            "data_dir": str(data_dir),
            "models_dir": str(models_dir),
            "cache_data_dir": str(GLOBAL_DATA_DIR),  # for repair.py cache→workspace path rewriting
            "cache_models_dir": str(GLOBAL_MODELS_DIR),
            "workspace_resource_aliases": workspace_aliases,
            "resource_download_enabled": self.config.auto_download_resources,
        }

        self.workspace.write_json("plans/setup_output.json", result)
        return result

    @staticmethod
    def _safe_alias_name(value: str, fallback: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
        return normalized or fallback

    @staticmethod
    def _stage_path(source: Path, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            return "existing"

        if source.is_dir():
            try:
                os.symlink(source, dest, target_is_directory=True)
                return "symlink"
            except OSError:
                shutil.copytree(source, dest)
                return "copytree"

        try:
            os.link(source, dest)
            return "hardlink"
        except OSError:
            try:
                os.symlink(source, dest)
                return "symlink"
            except OSError:
                shutil.copy2(source, dest)
                return "copy"

    @classmethod
    def _stage_workspace_resources(
        cls,
        resources: list[dict],
        data_dir: Path,
        models_dir: Path,
    ) -> tuple[list[dict], list[dict]]:
        staged_resources: list[dict] = []
        workspace_aliases: list[dict] = []

        for resource in resources:
            staged = dict(resource)
            status = str(resource.get("status", "")).strip()
            source_path = str(resource.get("path", "")).strip()
            resource_type = str(resource.get("type", "dataset")).strip().lower()
            target_root = models_dir if resource_type == "model" else data_dir

            if status not in SUCCESS_RESOURCE_STATUSES or not source_path:
                staged_resources.append(staged)
                continue

            source = Path(source_path)
            if not source.exists():
                staged_resources.append(staged)
                continue

            alias_details: dict[str, Any] = {
                "name": staged.get("name", ""),
                "type": resource_type,
                "cache_path": str(source),
            }

            if source.is_dir() and staged.get("files"):
                staged_file_paths: list[str] = []
                strategies: list[str] = []
                for file_name in staged.get("files", []):
                    candidate = source / str(file_name)
                    if not candidate.exists():
                        continue
                    dest = target_root / candidate.name
                    strategy = cls._stage_path(candidate, dest)
                    staged_file_paths.append(str(dest))
                    strategies.append(strategy)

                if staged_file_paths:
                    staged["cache_path"] = str(source)
                    staged["path"] = str(target_root)
                    staged["workspace_path"] = str(target_root)
                    staged["workspace_files"] = staged_file_paths
                    staged["staging_strategy"] = (
                        strategies[0] if len(set(strategies)) == 1 else "mixed"
                    )
                    alias_details.update(
                        {
                            "workspace_path": str(target_root),
                            "workspace_files": staged_file_paths,
                            "staging_strategy": staged["staging_strategy"],
                        }
                    )
                    workspace_aliases.append(alias_details)
                staged_resources.append(staged)
                continue

            alias_base = source.name or cls._safe_alias_name(
                str(staged.get("name", "resource")),
                "resource",
            )
            dest = target_root / alias_base
            strategy = cls._stage_path(source, dest)

            staged["cache_path"] = str(source)
            staged["path"] = str(dest)
            staged["workspace_path"] = str(dest)
            staged["staging_strategy"] = strategy
            alias_details.update(
                {
                    "workspace_path": str(dest),
                    "staging_strategy": strategy,
                }
            )
            workspace_aliases.append(alias_details)
            staged_resources.append(staged)

        return staged_resources, workspace_aliases

    @staticmethod
    def _augment_search_plan_with_blueprint_resources(
        search_plan: dict,
        blueprint: dict,
    ) -> dict:
        """Backfill downloadable dataset entries directly from the blueprint."""
        merged = dict(search_plan or {})
        datasets = list(merged.get("datasets", []))
        seen = {
            str(entry.get("name", "")).strip().lower()
            for entry in datasets
            if isinstance(entry, dict)
        }

        for dataset in blueprint.get("datasets", []):
            if not isinstance(dataset, dict):
                continue
            name = str(dataset.get("name", "")).strip()
            if not name or name.lower() in seen:
                continue
            source_url = str(dataset.get("source_url", "")).strip()
            if not source_url.startswith(("http://", "https://")):
                continue
            filename = source_url.split("/")[-1].split("?")[0] or f"{name.lower().replace(' ', '_')}.dat"
            datasets.append(
                {
                    "name": name,
                    "url": source_url,
                    "filename": filename,
                    "source": "blueprint",
                }
            )
            seen.add(name.lower())

        merged["datasets"] = datasets
        return merged

    async def _plan_search(
        self, topic: str, ideation: dict, blueprint: dict
    ) -> dict:
        """Use LLM to plan what to search, clone, and download."""
        system_prompt = (
            "You are a research engineer planning the setup phase for a deep learning experiment. "
            "Given a research topic and experiment blueprint, determine:\n"
            "1. What GitHub repos to search for (relevant codebases to build upon)\n"
            "2. What pretrained models to download (e.g., ESM, ProtBERT from HuggingFace)\n"
            "3. What datasets to download\n\n"
            "For datasets, you can provide:\n"
            "  - Direct download URLs (preferred): https://example.com/data.zip\n"
            "  - GitHub repo URLs: https://github.com/owner/dataset-repo (we will clone it "
            "and automatically extract real download links from README/scripts)\n"
            "  - wget/curl commands: wget https://... -O file.gz\n"
            "  - HuggingFace dataset URLs: https://huggingface.co/datasets/owner/name\n"
            "For models, use HuggingFace model IDs.\n"
            "Return JSON only."
        )

        method = blueprint.get("proposed_method", {})
        datasets = blueprint.get("datasets", [])
        hypothesis = ideation.get("selected_hypothesis", "")
        rationale = ideation.get("rationale", "")

        # Build explicit dataset checklist from blueprint
        dataset_checklist = ""
        for ds in datasets:
            if isinstance(ds, dict):
                name = ds.get("name", "")
                url = ds.get("source_url", "")
                dataset_checklist += f"  - {name} (known url: {url or 'FIND URL'})\n"
            else:
                dataset_checklist += f"  - {ds}\n"

        user_prompt = f"""Topic: {topic}

Hypothesis: {hypothesis}
Rationale: {rationale}

Proposed Method: {json.dumps(method, indent=2)[:1000]}
Datasets: {json.dumps(datasets, indent=2)[:500]}

IMPORTANT: The experiment blueprint requires ALL of the following datasets.
You MUST include ALL of them in your 'datasets' output with valid direct download URLs:
{dataset_checklist}
Do NOT skip any dataset from this list. If you cannot find a direct URL, provide the GitHub repo URL where the dataset is hosted — we will automatically clone it and extract the real download links.

Return a JSON object with:
{{
  "github_queries": ["query1", "query2", ...],  // 3-5 search queries for GitHub
  "target_repos": [  // specific repos to clone if known
    {{"owner": "...", "repo": "...", "reason": "..."}}
  ],
  "pretrained_models": [  // models to download from HuggingFace
    {{
      "name": "...",
      "source": "huggingface",
      "model_id": "facebook/esm2_t33_650M_UR50D",
      "download_weights": true,
      "reason": "..."
    }}
  ],
  "datasets": [  // datasets to download — url can be:
    // 1. Direct file URL: "https://example.com/data.zip"
    // 2. GitHub repo URL: "https://github.com/owner/dataset-repo"
    //    (will clone and auto-extract real download links from README/scripts)
    // 3. wget/curl command: "wget https://... -O file.gz"
    {{
      "name": "...",
      "url": "https://direct-download-url/file.gz OR https://github.com/owner/dataset-repo",
      "filename": "output_filename.gz",
      "reason": "..."
    }}
  ]
}}"""

        result = await self.generate_json(system_prompt, user_prompt)
        return result if isinstance(result, dict) else {}
