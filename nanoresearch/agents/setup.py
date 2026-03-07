"""Setup agent — searches GitHub for relevant code, clones repos, downloads models/data.

Uses a global cache at ~/.nanobot/cache/ so models/data are shared across pipeline runs.
Downloads models from ModelScope first (faster in China), falls back to HuggingFace.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
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


class SetupAgent(BaseResearchAgent):
    """Searches for relevant code repos, clones them, and downloads required resources."""

    stage = PipelineStage.EXPERIMENT  # reuse stage config for LLM calls

    async def run(self, **inputs: Any) -> dict[str, Any]:
        topic: str = inputs["topic"]
        ideation_output: dict = inputs.get("ideation_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})

        self.log("Starting setup: code search + resource download")

        # Step 1: Search GitHub for relevant repos
        search_plan = await self._plan_search(topic, ideation_output, experiment_blueprint)
        self.log(f"Search plan: {json.dumps(search_plan, indent=2)[:500]}")

        # Step 2: Search and clone repos
        cloned_repos = await self._search_and_clone(search_plan)
        self.log(f"Cloned {len(cloned_repos)} repos")

        # Step 3: Analyze cloned code
        code_analysis = await self._analyze_cloned_code(cloned_repos, experiment_blueprint)

        # Step 4: Download required resources (models, datasets)
        # Use global cache so models/data are shared across pipeline runs
        GLOBAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        GLOBAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

        resources = await self._download_resources(
            search_plan, GLOBAL_DATA_DIR, GLOBAL_MODELS_DIR
        )

        # Also create workspace-local symlinks for convenience
        data_dir = self.workspace.path / "data"
        models_dir = self.workspace.path / "models"
        for d in (data_dir, models_dir):
            if not d.exists():
                d.mkdir(exist_ok=True)

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

        result = {
            "search_plan": search_plan,
            "cloned_repos": cloned_repos,
            "code_analysis": code_analysis,
            "downloaded_resources": verified_resources,
            "data_dir": str(GLOBAL_DATA_DIR),
            "models_dir": str(GLOBAL_MODELS_DIR),
        }

        self.workspace.write_json("plans/setup_output.json", result)
        return result

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
            "For datasets, provide DIRECT download URLs (not API endpoints). "
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
Do NOT skip any dataset from this list. If you cannot find a direct URL, provide your best guess.

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
  "datasets": [  // datasets to download
    {{
      "name": "...",
      "url": "https://direct-download-url/file.gz",
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
                    except Exception:
                        pass
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
        for model_info in search_plan.get("pretrained_models", [])[:2]:
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

            # Try ModelScope first (convert HuggingFace ID to ModelScope format)
            modelscope_id = await self._hf_to_modelscope_id(model_id)
            success = False

            if modelscope_id:
                try:
                    self.log(f"Trying ModelScope (no proxy): {modelscope_id}")
                    if download_weights:
                        result = await self._run_shell_no_proxy(
                            f"python3 -c \""
                            f"from modelscope import snapshot_download; "
                            f"snapshot_download('{modelscope_id}', cache_dir='{dest.parent}', "
                            f"revision='master')\"",
                            timeout=1800,
                        )
                    else:
                        result = await self._run_shell_no_proxy(
                            f"python3 -c \""
                            f"from modelscope import snapshot_download; "
                            f"snapshot_download('{modelscope_id}', cache_dir='{dest.parent}', "
                            f"revision='master', "
                            f"ignore_file_pattern=['*.bin', '*.safetensors', '*.h5', '*.msgpack'])\"",
                            timeout=300,
                        )
                    if result.get("returncode", 1) == 0:
                        success = True
                        self.log(f"Downloaded from ModelScope: {modelscope_id}")
                except Exception as e:
                    self.log(f"ModelScope download failed: {e}")

            # Fall back to HuggingFace
            if not success:
                try:
                    self.log(f"Trying HuggingFace: {model_id}")
                    if download_weights:
                        result = await self._run_shell(
                            f"python3 -c \""
                            f"from huggingface_hub import snapshot_download; "
                            f"snapshot_download('{model_id}', local_dir='{dest}')\"",
                            timeout=1800,
                        )
                    else:
                        result = await self._run_shell(
                            f"python3 -c \""
                            f"from huggingface_hub import snapshot_download; "
                            f"snapshot_download('{model_id}', local_dir='{dest}', "
                            f"ignore_patterns=['*.bin', '*.safetensors', '*.h5', '*.msgpack'])\"",
                            timeout=300,
                        )
                    if result.get("returncode", 1) == 0:
                        success = True
                        self.log(f"Downloaded from HuggingFace: {model_id}")
                except Exception as e:
                    self.log(f"HuggingFace download failed: {e}")

            status = "full" if (download_weights and success) else ("config_only" if success else "failed")
            downloaded.append({
                "name": name, "type": "model",
                "path": str(dest), "source": model_id,
                "status": status,
            })

        # Download datasets
        for ds_info in search_plan.get("datasets", [])[:3]:
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

            if url.startswith(("wget ", "curl ")):
                try:
                    # Only allow wget/curl commands, sanitize the data_dir path
                    result = await self._run_shell(
                        f"cd {shlex.quote(str(data_dir))} && {url}", timeout=600
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

    async def _run_shell_no_proxy(self, cmd: str, timeout: int = 60) -> dict:
        """Run a shell command WITHOUT proxy (for domestic sources like ModelScope)."""
        env = {k: v for k, v in __import__('os').environ.items()
               if 'proxy' not in k.lower()}
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

    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict:
        """Run a shell command asynchronously with proxy environment."""
        env = {**__import__('os').environ}
        proxy_url = env.get("https_proxy") or env.get("HTTPS_PROXY", "")
        if not proxy_url:
            import re as _re
            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                content = bashrc.read_text(errors="replace")
                m = _re.search(r"https_proxy=(http://[^\s;'\"]+)", content)
                if m:
                    proxy_url = m.group(1)
        if proxy_url:
            env.update({
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
            })

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

    async def close(self) -> None:
        pass
