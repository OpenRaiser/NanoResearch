"""PDF compilation tool — wraps tectonic or pdflatex + bibtex."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum time (seconds) for a single subprocess invocation
_SUBPROCESS_TIMEOUT = 120


async def compile_pdf(
    tex_path: str | Path,
    output_dir: str | Path | None = None,
    bibtex: bool = True,
) -> dict[str, str]:
    """Compile a .tex file to PDF using tectonic (preferred) or pdflatex.

    Args:
        tex_path: Path to the .tex file.
        output_dir: Output directory (defaults to same as tex_path).
        bibtex: Whether to run bibtex for references.

    Returns:
        Dict with 'pdf_path' on success, or 'error' with compilation log.
    """
    tex_path = Path(tex_path).resolve()
    if not tex_path.is_file():
        return {"error": f"File not found: {tex_path}"}

    work_dir = tex_path.parent
    out_dir = Path(output_dir).resolve() if output_dir else work_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    job_name = tex_path.stem

    # Prefer tectonic (handles bibtex automatically in a single command)
    if shutil.which("tectonic"):
        return await _compile_tectonic(tex_path, out_dir, job_name)

    if shutil.which("pdflatex"):
        return await _compile_pdflatex(tex_path, out_dir, job_name, bibtex, work_dir)

    return {"error": "No LaTeX compiler found. Install tectonic or TeX Live (pdflatex)."}


async def _compile_tectonic(
    tex_path: Path, out_dir: Path, job_name: str,
) -> dict[str, str]:
    """Compile with tectonic — handles bibtex cycle automatically."""
    cmd = ["tectonic", str(tex_path)]
    result = await _run(cmd, tex_path.parent)

    pdf_path = out_dir / f"{job_name}.pdf"
    if pdf_path.is_file():
        return {"pdf_path": str(pdf_path), "compiler": "tectonic"}

    stderr = result.get("stderr", "")
    stdout = result.get("stdout", "")
    return {"error": f"tectonic compilation failed:\n{stderr}\n{stdout}"}


async def _compile_pdflatex(
    tex_path: Path, out_dir: Path, job_name: str,
    bibtex: bool, work_dir: Path,
) -> dict[str, str]:
    """Compile with pdflatex + bibtex (3-pass)."""
    base_cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={out_dir}",
        str(tex_path),
    ]

    # First pass
    result = await _run(base_cmd, work_dir)
    if result["returncode"] != 0 and "Fatal" in result.get("stderr", ""):
        return {"error": f"pdflatex failed (pass 1):\n{result.get('stderr', '')}\n{result.get('stdout', '')}"}

    # BibTeX pass
    if bibtex:
        bib_cmd = ["bibtex", str(out_dir / job_name)]
        await _run(bib_cmd, work_dir)

    # Second and third passes for references
    for pass_num in (2, 3):
        result = await _run(base_cmd, work_dir)
        if result["returncode"] != 0 and "Fatal" in result.get("stderr", ""):
            return {"error": f"pdflatex failed (pass {pass_num}):\n{result.get('stderr', '')}"}

    pdf_path = out_dir / f"{job_name}.pdf"
    if pdf_path.is_file():
        return {"pdf_path": str(pdf_path), "compiler": "pdflatex"}
    return {"error": f"PDF not generated at {pdf_path}"}


async def _run(cmd: list[str], cwd: Path) -> dict[str, str | int]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_SUBPROCESS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        proc.kill()
        logger.warning("Subprocess timed out after %ds: %s", _SUBPROCESS_TIMEOUT, cmd)
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Process timed out after {_SUBPROCESS_TIMEOUT}s",
        }
    return {
        "returncode": proc.returncode or 0,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }
