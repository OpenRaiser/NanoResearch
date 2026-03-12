"""Setup agent — searches GitHub for relevant code, clones repos, downloads models/data.

Uses a global cache at ~/.nanobot/cache/ so models/data are shared across pipeline runs.
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

logger = logging.getLogger(__name__)

# Global cache directory — shared across all pipeline runs
GLOBAL_CACHE_DIR = Path.home() / ".nanobot" / "cache"
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


class SetupAgent(BaseResearchAgent):
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

    async def _search_and_clone(self, search_plan: dict) -> list[dict]:
        """Search GitHub and clone relevant repos."""
        cloned = []
        repos_dir = self.workspace.path / "repos"
        repos_dir.mkdir(exist_ok=True)

        # Clone specific target repos first
        for repo_info in search_plan.get("target_repos", [])[:3]:
            owner = repo_info.get("owner", "")
            repo = repo_info.get("repo", "")
            if not owner or not repo:
                continue
            # Sanitize owner/repo to prevent command injection
            if not re.match(r'^[a-zA-Z0-9._-]+$', owner) or not re.match(r'^[a-zA-Z0-9._-]+$', repo):
                self.log(f"Skipping unsafe repo name: {owner}/{repo}")
                continue
            clone_url = f"https://github.com/{owner}/{repo}.git"
            dest = repos_dir / repo
            if dest.exists():
                cloned.append({"name": repo, "path": str(dest), "source": clone_url})
                continue
            try:
                result = await self._run_shell(
                    f"git clone --depth 1 {shlex.quote(clone_url)} {shlex.quote(str(dest))}", timeout=120
                )
                if dest.exists():
                    cloned.append({"name": repo, "path": str(dest), "source": clone_url})
                    self.log(f"Cloned {owner}/{repo}")
            except Exception as e:
                self.log(f"Failed to clone {owner}/{repo}: {e}")

        # Search GitHub API for additional repos
        for query in search_plan.get("github_queries", [])[:3]:
            if len(cloned) >= 5:
                break
            try:
                repos = await self._github_search(query)
                for r in repos[:2]:
                    if len(cloned) >= 5:
                        break
                    name = r.get("name", "")
                    clone_url = r.get("clone_url", "")
                    if not clone_url or (repos_dir / name).exists():
                        continue
                    dest = repos_dir / name
                    await self._run_shell(
                        f"git clone --depth 1 {shlex.quote(clone_url)} {shlex.quote(str(dest))}", timeout=120
                    )
                    if dest.exists():
                        cloned.append({
                            "name": name,
                            "path": str(dest),
                            "source": clone_url,
                            "stars": r.get("stargazers_count", 0),
                            "description": r.get("description", ""),
                        })
                        self.log(f"Cloned {name} ({r.get('stargazers_count', 0)} stars)")
            except Exception as e:
                self.log(f"GitHub search failed for '{query}': {e}")

        return cloned

    async def _github_search(self, query: str) -> list[dict]:
        """Search GitHub repos via API."""
        import urllib.parse
        encoded = urllib.parse.quote(query)
        cmd = (
            f'curl -s "https://api.github.com/search/repositories'
            f"?q={encoded}&sort=stars&per_page=5&order=desc\""
        )
        result = await self._run_shell(cmd, timeout=30)
        stdout = result.get("stdout", "")
        try:
            data = json.loads(stdout)
            return data.get("items", [])
        except json.JSONDecodeError:
            return []

    async def _analyze_cloned_code(
        self, cloned_repos: list[dict], blueprint: dict
    ) -> dict:
        """Analyze cloned repos to understand their structure and key components."""
        if not cloned_repos:
            return {"summary": "No repos cloned", "key_files": [], "reusable_components": []}

        # Collect file listings and key files from each repo
        repo_summaries = []
        for repo in cloned_repos[:3]:
            repo_path = Path(repo["path"])
            tree_result = await self._run_shell(
                f"find {repo_path} -maxdepth 3 -type f -name '*.py' | head -50",
                timeout=10,
            )
            files = tree_result.get("stdout", "").strip().split("\n")[:50]

            readme_content = ""
            for readme_name in ["README.md", "readme.md", "README.rst"]:
                readme_path = repo_path / readme_name
                if readme_path.exists():
                    readme_content = readme_path.read_text(errors="replace")[:3000]
                    break

            key_snippets = []
            for f in files:
                fname = Path(f).name.lower()
                if any(kw in fname for kw in ["model", "train", "config", "main", "run"]):
                    try:
                        content = Path(f).read_text(errors="replace")[:2000]
                        key_snippets.append({"file": f, "content": content})
                    except Exception as exc:
                        logger.debug("Failed to read repo snippet %s: %s", f, exc)
                    if len(key_snippets) >= 5:
                        break

            repo_summaries.append({
                "name": repo["name"],
                "files": files[:30],
                "readme": readme_content[:1500],
                "key_snippets": key_snippets,
            })

        system_prompt = (
            "You are a ML research engineer analyzing cloned code repositories. "
            "Identify reusable components, architecture patterns, training pipelines, "
            "and suggest how to build upon this code for the proposed experiment. "
            "Return JSON only."
        )

        method = blueprint.get("proposed_method", {})
        user_prompt = f"""Proposed method: {json.dumps(method, indent=2)[:800]}

Cloned repositories:
{json.dumps(repo_summaries, indent=2)[:8000]}

Return JSON:
{{
  "summary": "Overall analysis of available code...",
  "best_base_repo": "name of most relevant repo to build upon",
  "key_files": [
    {{"repo": "...", "file": "...", "purpose": "...", "reuse_plan": "..."}}
  ],
  "reusable_components": [
    {{"name": "...", "source_file": "...", "description": "...", "modifications_needed": "..."}}
  ],
  "missing_components": ["list of things we need to implement from scratch"],
  "recommended_approach": "How to combine/extend these codebases..."
}}"""

        result = await self.generate_json(system_prompt, user_prompt)
        return result if isinstance(result, dict) else {}

    async def _download_resources(
        self, search_plan: dict, data_dir: Path, models_dir: Path
    ) -> list[dict]:
        """Download pretrained models and datasets to global cache.

        Download priority for models:
        1. Check if already cached (skip download)
        2. Try ModelScope first (faster in China)
        3. Fall back to HuggingFace
        """
        downloaded = []

        # Download pretrained models
        for model_info in search_plan.get("pretrained_models", []):
            name = model_info.get("name", "unknown")
            model_id = model_info.get("model_id", "")
            source = model_info.get("source", "")
            download_weights = model_info.get("download_weights", True)

            if not model_id:
                continue

            safe_name = name.replace("/", "_").replace(" ", "_")
            dest = models_dir / safe_name

            # Check if already cached
            if dest.exists() and any(dest.iterdir()):
                existing_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
                if existing_size > 1000:  # more than just a few config files
                    self.log(f"Model already cached: {model_id} ({existing_size / 1024 / 1024:.0f} MB)")
                    status = "full" if existing_size > 100_000_000 else "config_only"
                    downloaded.append({
                        "name": name, "type": "model",
                        "path": str(dest), "source": model_id,
                        "status": status, "cached": True,
                    })
                    continue

            dest.mkdir(parents=True, exist_ok=True)
            self.log(f"Downloading model: {model_id}")

            # BUG-20 fix: validate model_id format before passing to shell.
            # Only allow characters valid in HuggingFace/ModelScope IDs.
            _MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9_\-./]+$")
            if not _MODEL_ID_RE.match(model_id):
                self.log(f"Invalid model_id format, skipping: {model_id!r}")
                downloaded.append({
                    "name": name, "type": "model",
                    "path": str(dest), "source": model_id,
                    "status": "failed", "error": "invalid model_id format",
                })
                continue

            # Try ModelScope first (convert HuggingFace ID to ModelScope format)
            modelscope_id = await self._hf_to_modelscope_id(model_id)
            success = False

            # BUG-20 fix: pass model_id/dest via environment variables
            # instead of embedding in f-string python code, preventing
            # shell/Python injection from untrusted LLM-generated IDs.
            if modelscope_id:
                if not _MODEL_ID_RE.match(modelscope_id):
                    self.log(f"Invalid modelscope_id format, skipping ModelScope: {modelscope_id!r}")
                else:
                    try:
                        self.log(f"Trying ModelScope (no proxy): {modelscope_id}")
                        ms_env = {
                            "_NR_MODEL_ID": modelscope_id,
                            "_NR_CACHE_DIR": str(dest.parent),
                        }
                        if download_weights:
                            result = await self._run_shell_no_proxy(
                                'python3 -c "'
                                'import os; '
                                'from modelscope import snapshot_download; '
                                'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                                'cache_dir=os.environ[\'_NR_CACHE_DIR\'], '
                                'revision=\'master\')"',
                                timeout=1800, env=ms_env,
                            )
                        else:
                            result = await self._run_shell_no_proxy(
                                'python3 -c "'
                                'import os; '
                                'from modelscope import snapshot_download; '
                                'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                                'cache_dir=os.environ[\'_NR_CACHE_DIR\'], '
                                'revision=\'master\', '
                                'ignore_file_pattern=[\'*.bin\', \'*.safetensors\', \'*.h5\', \'*.msgpack\'])"',
                                timeout=300, env=ms_env,
                            )
                        if result.get("returncode", 1) == 0:
                            success = True
                            self.log(f"Downloaded from ModelScope: {modelscope_id}")
                    except Exception as e:
                        self.log(f"ModelScope download failed: {e}")

            # Fall back to HuggingFace (official endpoint)
            if not success:
                try:
                    self.log(f"Trying HuggingFace: {model_id}")
                    hf_env = {
                        "_NR_MODEL_ID": model_id,
                        "_NR_LOCAL_DIR": str(dest),
                    }
                    if download_weights:
                        result = await self._run_shell(
                            'python3 -c "'
                            'import os; '
                            'from huggingface_hub import snapshot_download; '
                            'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                            'local_dir=os.environ[\'_NR_LOCAL_DIR\'])"',
                            timeout=1800, env=hf_env,
                        )
                    else:
                        result = await self._run_shell(
                            'python3 -c "'
                            'import os; '
                            'from huggingface_hub import snapshot_download; '
                            'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                            'local_dir=os.environ[\'_NR_LOCAL_DIR\'], '
                            'ignore_patterns=[\'*.bin\', \'*.safetensors\', \'*.h5\', \'*.msgpack\'])"',
                            timeout=300, env=hf_env,
                        )
                    if result.get("returncode", 1) == 0:
                        success = True
                        self.log(f"Downloaded from HuggingFace: {model_id}")
                except Exception as e:
                    self.log(f"HuggingFace download failed: {e}")

            # Fall back to hf-mirror.com (China mirror, no proxy needed)
            if not success:
                try:
                    self.log(f"Trying hf-mirror.com: {model_id}")
                    mirror_env = {
                        "_NR_MODEL_ID": model_id,
                        "_NR_LOCAL_DIR": str(dest),
                        "HF_ENDPOINT": "https://hf-mirror.com",
                    }
                    if download_weights:
                        result = await self._run_shell_no_proxy(
                            'python3 -c "'
                            'import os; '
                            'os.environ[\'HF_ENDPOINT\'] = \'https://hf-mirror.com\'; '
                            'from huggingface_hub import snapshot_download; '
                            'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                            'local_dir=os.environ[\'_NR_LOCAL_DIR\'])"',
                            timeout=1800, env=mirror_env,
                        )
                    else:
                        result = await self._run_shell_no_proxy(
                            'python3 -c "'
                            'import os; '
                            'os.environ[\'HF_ENDPOINT\'] = \'https://hf-mirror.com\'; '
                            'from huggingface_hub import snapshot_download; '
                            'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                            'local_dir=os.environ[\'_NR_LOCAL_DIR\'], '
                            'ignore_patterns=[\'*.bin\', \'*.safetensors\', \'*.h5\', \'*.msgpack\'])"',
                            timeout=300, env=mirror_env,
                        )
                    if result.get("returncode", 1) == 0:
                        success = True
                        self.log(f"Downloaded from hf-mirror.com: {model_id}")
                except Exception as e:
                    self.log(f"hf-mirror download failed: {e}")

            status = "full" if (download_weights and success) else ("config_only" if success else "failed")
            downloaded.append({
                "name": name, "type": "model",
                "path": str(dest), "source": model_id,
                "status": status,
            })

        # Download datasets
        for ds_info in search_plan.get("datasets", []):
            name = ds_info.get("name", "unknown")
            url = ds_info.get("url", "") or ds_info.get("download_cmd", "")
            filename = ds_info.get("filename", "")

            if not url:
                continue

            # Check if already cached
            if filename:
                cached_file = data_dir / filename
                decompressed_name = filename[:-3] if filename.endswith(".gz") else filename
                cached_decompressed = data_dir / decompressed_name
                if cached_decompressed.exists() and cached_decompressed.stat().st_size > 0:
                    self.log(f"Dataset already cached: {name} -> {cached_decompressed.name}")
                    downloaded.append({
                        "name": name, "type": "dataset",
                        "path": str(cached_decompressed),
                        "status": "downloaded", "cached": True,
                    })
                    continue
                if cached_file.exists() and cached_file.stat().st_size > 0:
                    self.log(f"Dataset already cached: {name} -> {cached_file.name}")
                    downloaded.append({
                        "name": name, "type": "dataset",
                        "path": str(cached_file),
                        "status": "downloaded", "cached": True,
                    })
                    continue

            self.log(f"Downloading dataset: {name}")

            # ── GitHub repo URL → clone + extract real data ──
            gh_match = self._is_github_repo_url(url)
            if gh_match:
                gh_owner, gh_repo = gh_match.group("owner"), gh_match.group("repo")
                ds_data_dir = data_dir / gh_repo
                ds_data_dir.mkdir(parents=True, exist_ok=True)
                result_entry = await self._handle_github_dataset(
                    name, gh_owner, gh_repo, ds_data_dir,
                )
                downloaded.append(result_entry)
                continue

            if url.startswith(("wget ", "curl ")):
                try:
                    # BUG-18 fix: sanitize LLM-generated download command.
                    # Tokenize with shlex and reject anything that isn't a
                    # flag or an http(s)/ftp URL to block shell injection.
                    try:
                        dl_parts = shlex.split(url)
                    except ValueError:
                        raise RuntimeError(f"Unparseable download command: {url[:200]}")
                    dl_cmd = dl_parts[0]
                    if dl_cmd not in ("wget", "curl"):
                        raise RuntimeError(f"Blocked download command: {dl_cmd}")
                    safe_dl = [dl_cmd]
                    for dl_arg in dl_parts[1:]:
                        if dl_arg.startswith("-"):
                            safe_dl.append(dl_arg)
                        elif dl_arg.startswith(("http://", "https://", "ftp://")):
                            safe_dl.append(dl_arg)
                        else:
                            logger.warning("Dropped suspicious arg in download cmd: %s", dl_arg[:120])
                    sanitized_dl = " ".join(shlex.quote(p) for p in safe_dl)
                    result = await self._run_shell(
                        f"cd {shlex.quote(str(data_dir))} && {sanitized_dl}", timeout=600
                    )
                    dl_files = list(data_dir.glob("*"))
                    downloaded.append({
                        "name": name, "type": "dataset",
                        "path": str(data_dir), "status": "downloaded",
                        "files": [f.name for f in dl_files],
                    })
                    self.log(f"Downloaded dataset: {name}")
                except Exception as e:
                    self.log(f"Failed to download dataset {name}: {e}")
                    downloaded.append({
                        "name": name, "type": "dataset",
                        "status": "failed", "error": str(e),
                    })
            elif url.startswith("http"):
                if not filename:
                    filename = url.split("/")[-1].split("?")[0]
                dest_file = data_dir / filename
                try:
                    result = await self._run_shell(
                        f"wget -q -O {shlex.quote(str(dest_file))} {shlex.quote(url)}", timeout=600
                    )
                    if dest_file.exists() and dest_file.stat().st_size > 0:
                        if filename.endswith(".gz") and not filename.endswith(".tar.gz"):
                            decompressed = data_dir / filename[:-3]
                            try:
                                with gzip.open(dest_file, 'rb') as f_in:
                                    with open(decompressed, 'wb') as f_out:
                                        shutil.copyfileobj(f_in, f_out)
                                self.log(f"Decompressed: {filename} -> {decompressed.name}")
                                downloaded.append({
                                    "name": name, "type": "dataset",
                                    "path": str(decompressed),
                                    "compressed_path": str(dest_file),
                                    "status": "downloaded",
                                })
                            except Exception:
                                downloaded.append({
                                    "name": name, "type": "dataset",
                                    "path": str(dest_file),
                                    "status": "downloaded",
                                })
                        else:
                            downloaded.append({
                                "name": name, "type": "dataset",
                                "path": str(dest_file),
                                "status": "downloaded",
                            })
                        self.log(f"Downloaded dataset: {name} -> {dest_file.name}")
                    else:
                        downloaded.append({
                            "name": name, "type": "dataset",
                            "status": "failed", "error": "Downloaded file is empty or missing",
                        })
                except Exception as e:
                    self.log(f"Failed to download dataset {name}: {e}")
                    downloaded.append({
                        "name": name, "type": "dataset",
                        "status": "failed", "error": str(e),
                    })

        return downloaded

    # ------------------------------------------------------------------
    # GitHub dataset repo handling
    # ------------------------------------------------------------------

    @staticmethod
    def _is_github_repo_url(url: str) -> re.Match | None:
        """Return a match object if *url* points to a GitHub repo (not a raw file)."""
        return _GITHUB_REPO_RE.match(url.strip())

    async def _clone_dataset_repo(self, owner: str, repo: str, dest: Path) -> bool:
        """Shallow-clone a GitHub dataset repo. Returns True on success."""
        if dest.exists():
            return True
        clone_url = f"https://github.com/{owner}/{repo}.git"
        try:
            result = await self._run_shell(
                f"git clone --depth 1 {shlex.quote(clone_url)} {shlex.quote(str(dest))}",
                timeout=180,
            )
            return dest.exists()
        except Exception as exc:
            self.log(f"Failed to clone dataset repo {owner}/{repo}: {exc}")
            return False

    @staticmethod
    def _extract_download_urls_from_repo(repo_dir: Path) -> list[str]:
        """Scan README, download scripts, and configs for real download URLs."""
        urls: list[str] = []
        seen: set[str] = set()

        # Files most likely to contain download URLs
        scan_patterns = [
            "README*", "readme*",
            "download*", "get_data*", "fetch*", "prepare*", "setup_data*",
            "scripts/download*", "scripts/get_data*", "scripts/prepare*",
            "data/download*", "data/get*", "data/prepare*",
            "*.sh", "*.py",
            "*.cfg", "*.yaml", "*.yml", "*.json",
        ]
        candidates: list[Path] = []
        for pattern in scan_patterns:
            candidates.extend(repo_dir.glob(pattern))
        # Also check one level down
        for pattern in ["**/download*", "**/get_data*", "**/prepare*"]:
            candidates.extend(repo_dir.glob(pattern))

        # Deduplicate and limit to avoid scanning huge repos
        candidate_set: set[Path] = set()
        for p in candidates:
            if p.is_file() and p.stat().st_size < 500_000:  # skip big files
                candidate_set.add(p)
        # Cap at 30 files
        for file_path in sorted(candidate_set)[:30]:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for m in _DOWNLOAD_URL_RE.finditer(content):
                url = m.group(0).rstrip(".,;:)>\"'")
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        return urls

    @staticmethod
    def _find_download_scripts(repo_dir: Path) -> list[Path]:
        """Find shell/Python scripts whose name suggests data downloading."""
        scripts: list[Path] = []
        keywords = {"download", "get_data", "fetch_data", "prepare_data", "setup_data"}
        for pattern in ["**/*.sh", "**/*.py"]:
            for p in repo_dir.glob(pattern):
                if not p.is_file():
                    continue
                name_lower = p.stem.lower()
                if any(kw in name_lower for kw in keywords):
                    scripts.append(p)
        return sorted(scripts)[:5]

    async def _handle_github_dataset(
        self,
        name: str,
        owner: str,
        repo: str,
        data_dir: Path,
    ) -> dict[str, Any]:
        """Clone a GitHub dataset repo, extract real download URLs, and fetch data."""
        repos_dir = self.workspace.path / "dataset_repos"
        repos_dir.mkdir(parents=True, exist_ok=True)
        repo_dest = repos_dir / repo

        # Step 1: Clone
        self.log(f"Dataset '{name}' is a GitHub repo — cloning {owner}/{repo} to find real data...")
        cloned = await self._clone_dataset_repo(owner, repo, repo_dest)
        if not cloned:
            return {
                "name": name, "type": "dataset",
                "status": "failed",
                "error": f"Failed to clone dataset repo github.com/{owner}/{repo}",
            }

        # Step 2: Check if the repo itself contains data files directly
        data_files: list[Path] = []
        data_extensions = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".npy", ".npz",
                          ".h5", ".hdf5", ".pt", ".pkl", ".parquet"}
        for f in repo_dest.rglob("*"):
            if f.is_file() and f.suffix.lower() in data_extensions and f.stat().st_size > 100:
                data_files.append(f)
        if data_files:
            # Copy data files to data_dir
            copied_files: list[str] = []
            total_size = 0
            for f in data_files[:20]:  # cap to avoid flooding
                dest = data_dir / f.relative_to(repo_dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(f, dest)
                copied_files.append(str(dest))
                total_size += f.stat().st_size
            if total_size > 1000:
                self.log(f"Found {len(copied_files)} data files directly in repo ({total_size / 1024:.0f} KB)")
                return {
                    "name": name, "type": "dataset",
                    "path": str(data_dir),
                    "status": "downloaded",
                    "source": f"github.com/{owner}/{repo}",
                    "files": copied_files[:10],
                    "strategy": "repo_data_files",
                }

        # Step 3: Try running download scripts
        download_scripts = self._find_download_scripts(repo_dest)
        for script in download_scripts:
            self.log(f"Running download script: {script.name}")
            if script.suffix == ".sh":
                cmd = f"cd {shlex.quote(str(data_dir))} && bash {shlex.quote(str(script))}"
            else:
                cmd = f"cd {shlex.quote(str(data_dir))} && python {shlex.quote(str(script))}"
            try:
                result = await self._run_shell(cmd, timeout=600)
                if result.get("returncode", 1) == 0:
                    dl_files = [f for f in data_dir.iterdir() if f.is_file() and f.stat().st_size > 0]
                    if dl_files:
                        self.log(f"Download script {script.name} succeeded — {len(dl_files)} files")
                        return {
                            "name": name, "type": "dataset",
                            "path": str(data_dir),
                            "status": "downloaded",
                            "source": f"github.com/{owner}/{repo} via {script.name}",
                            "files": [f.name for f in dl_files[:10]],
                            "strategy": "download_script",
                        }
            except Exception as exc:
                self.log(f"Download script {script.name} failed: {exc}")

        # Step 4: Extract download URLs from README / scripts and fetch
        extracted_urls = self._extract_download_urls_from_repo(repo_dest)
        if extracted_urls:
            self.log(f"Extracted {len(extracted_urls)} download URLs from repo files")
            for url in extracted_urls[:5]:
                filename = url.split("/")[-1].split("?")[0][:80]
                if not filename or len(filename) < 3:
                    filename = f"{name.replace(' ', '_')}_{hash(url) % 10000}.dat"
                dest_file = data_dir / filename
                if dest_file.exists() and dest_file.stat().st_size > 0:
                    continue
                try:
                    result = await self._run_shell(
                        f"wget -q -O {shlex.quote(str(dest_file))} {shlex.quote(url)}",
                        timeout=600,
                    )
                    if dest_file.exists() and dest_file.stat().st_size > 0:
                        self.log(f"Downloaded {filename} from extracted URL")
                    else:
                        dest_file.unlink(missing_ok=True)
                except Exception as exc:
                    self.log(f"Failed to download from extracted URL {url[:80]}: {exc}")
                    dest_file.unlink(missing_ok=True)

            dl_files = [f for f in data_dir.iterdir() if f.is_file() and f.stat().st_size > 0]
            if dl_files:
                return {
                    "name": name, "type": "dataset",
                    "path": str(data_dir),
                    "status": "downloaded",
                    "source": f"github.com/{owner}/{repo} (extracted URLs)",
                    "files": [f.name for f in dl_files[:10]],
                    "strategy": "extracted_urls",
                }

        # Step 5: Use LLM to analyze the repo and find download instructions
        readme_content = ""
        for readme_name in ["README.md", "readme.md", "README.rst", "README.txt", "README"]:
            readme_path = repo_dest / readme_name
            if readme_path.exists():
                readme_content = readme_path.read_text(errors="replace")[:6000]
                break

        if readme_content:
            llm_result = await self._llm_extract_download_info(
                name, owner, repo, readme_content, data_dir
            )
            if llm_result and llm_result.get("status") == "downloaded":
                return llm_result

        # All strategies failed — still return the cloned repo path as reference
        return {
            "name": name, "type": "dataset",
            "path": str(repo_dest),
            "status": "partial",
            "source": f"github.com/{owner}/{repo}",
            "strategy": "repo_cloned_only",
            "note": "Repo cloned but no direct data files or download URLs found. "
                    "Experiment code may need to load data from this repo directory.",
        }

    async def _llm_extract_download_info(
        self,
        dataset_name: str,
        owner: str,
        repo: str,
        readme_content: str,
        data_dir: Path,
    ) -> dict[str, Any] | None:
        """Ask LLM to read the README and extract the actual download command."""
        system_prompt = (
            "You are a data engineer. Given a dataset GitHub repo's README, extract the "
            "exact command(s) needed to download the dataset files. "
            "Return JSON only."
        )
        user_prompt = f"""Dataset: {dataset_name}
Repo: github.com/{owner}/{repo}

README content:
{readme_content}

Extract the download commands. Return JSON:
{{
  "download_commands": [
    "wget https://... -O filename.zip",
    "curl -L https://... -o data.tar.gz"
  ],
  "notes": "any special instructions (e.g., need to unzip, specific directory structure)"
}}

Rules:
- Only include wget/curl/gdown/python commands that directly download files
- For Google Drive links, use gdown: `gdown https://drive.google.com/uc?id=FILE_ID -O filename`
- For HuggingFace datasets, use: `wget https://huggingface.co/datasets/OWNER/REPO/resolve/main/FILE`
- Do NOT include pip install or git clone commands
- If the README says to use a Python API (e.g., `datasets.load_dataset()`), include that as a command:
  `python -c "from datasets import load_dataset; ds = load_dataset('name'); ds.save_to_disk('data/')"`
"""
        try:
            result = await self.generate_json(system_prompt, user_prompt)
        except Exception:
            return None

        if not isinstance(result, dict):
            return None

        commands = result.get("download_commands", [])
        if not isinstance(commands, list) or not commands:
            return None

        self.log(f"LLM extracted {len(commands)} download commands from README")
        for cmd in commands[:5]:
            if not isinstance(cmd, str) or not cmd.strip():
                continue
            cmd = cmd.strip()
            # Safety: only allow wget/curl/gdown/python commands
            if not cmd.startswith(("wget ", "curl ", "gdown ", "python ")):
                self.log(f"Skipping unsafe command: {cmd[:60]}")
                continue
            # BUG-18 fix (second site): sanitize via shlex tokenization
            try:
                cmd_parts = shlex.split(cmd)
            except ValueError:
                self.log(f"Skipping unparseable command: {cmd[:60]}")
                continue
            sanitized = " ".join(shlex.quote(p) for p in cmd_parts)
            try:
                await self._run_shell(
                    f"cd {shlex.quote(str(data_dir))} && {sanitized}",
                    timeout=600,
                )
            except Exception as exc:
                self.log(f"LLM download command failed: {exc}")

        dl_files = [f for f in data_dir.iterdir() if f.is_file() and f.stat().st_size > 0]
        if dl_files:
            return {
                "name": dataset_name, "type": "dataset",
                "path": str(data_dir),
                "status": "downloaded",
                "source": f"github.com/{owner}/{repo} (LLM-extracted commands)",
                "files": [f.name for f in dl_files[:10]],
                "strategy": "llm_readme_parse",
            }
        return None

    async def _hf_to_modelscope_id(self, hf_id: str) -> str:
        """Search ModelScope for a matching model (async, non-blocking).

        Uses ModelScope API to find if the model exists, rather than
        relying on a hardcoded mapping table.
        """
        # Try common org mappings first as search hints
        search_terms = []
        if "/" in hf_id:
            model_name = hf_id.split("/")[-1]
            search_terms.append(model_name)
        search_terms.append(hf_id)

        for term in search_terms:
            try:
                # Sanitize term to prevent code injection
                safe_term = re.sub(r"[^a-zA-Z0-9_\-./]", "", term)
                if not safe_term:
                    continue
                result = await self._run_shell_no_proxy(
                    f"python3 -c \""
                    f"from modelscope.hub.api import HubApi; "
                    f"api = HubApi(); "
                    f"models = api.list_models('{safe_term}', limit=3); "
                    f"print([m.model_id for m in models] if models else [])\"",
                    timeout=15,
                )
                if result.get("returncode") == 0 and result.get("stdout", "").strip():
                    import ast
                    ids = ast.literal_eval(result["stdout"].strip())
                    if ids:
                        self.log(f"Found ModelScope match: {ids[0]} for {hf_id}")
                        return ids[0]
            except Exception:
                pass

        return ""

    async def _run_shell_no_proxy(self, cmd: str, timeout: int = 60, env: dict | None = None) -> dict:
        """Run a shell command WITHOUT proxy (for domestic sources like ModelScope)."""
        _env = {k: v for k, v in __import__('os').environ.items()
               if 'proxy' not in k.lower()}
        if env:
            _env.update(env)
        env = _env
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace.path),
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def _run_shell(self, cmd: str, timeout: int = 60, env: dict | None = None) -> dict:
        """Run a shell command asynchronously with proxy environment."""
        _env = {**__import__('os').environ}
        proxy_url = _env.get("https_proxy") or _env.get("HTTPS_PROXY", "")
        if not proxy_url:
            import re as _re
            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                content = bashrc.read_text(errors="replace")
                m = _re.search(r"https_proxy=(http://[^\s;'\"]+)", content)
                if m:
                    proxy_url = m.group(1)
        if proxy_url:
            _env.update({
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
            })
        if env:
            _env.update(env)

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace.path),
            env=_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def close(self) -> None:
        pass
