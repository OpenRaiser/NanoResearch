"""Setup agent GitHub integration and shell utilities mixin."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+?)(?:\.git)?/?$"
)
_DOWNLOAD_URL_RE = re.compile(
    r"(https?://(?:"
    r"drive\.google\.com/[^\s\)\]\"'>]+|"
    r"docs\.google\.com/[^\s\)\]\"'>]+|"
    r"dl\.fbaipublicfiles\.com/[^\s\)\]\"'>]+|"
    r"zenodo\.org/record[^\s\)\]\"'>]+|"
    r"zenodo\.org/api/records[^\s\)\]\"'>]+|"
    r"huggingface\.co/datasets/[^\s\)\]\"'>]+|"
    r"storage\.googleapis\.com/[^\s\)\]\"'>]+|"
    r"s3\.amazonaws\.com/[^\s\)\]\"'>]+|"
    r"(?:[a-z0-9-]+\.)?s3[.-][^\s\)\]\"'>]+|"
    r"dropbox\.com/[^\s\)\]\"'>]+|"
    r"figshare\.com/[^\s\)\]\"'>]+|"
    r"data\.dgl\.ai/[^\s\)\]\"'>]+|"
    r"people\.csail\.mit\.edu/[^\s\)\]\"'>]+|"
    r"[^\s\)\]\"'>]+\.(?:zip|tar\.gz|tgz|tar\.bz2|gz|csv|tsv|json|jsonl|h5|hdf5|pt|pkl|npy|npz|parquet|txt)"
    r")"
    r")",
    re.IGNORECASE,
)


class _SetupGithubMixin:
    """Mixin — GitHub integration, LLM extraction, and shell utilities."""

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
