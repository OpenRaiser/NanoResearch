"""Tectonic-based PDF compilation + deterministic error fixing."""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)

# ── Package requirement map ──────────────────────────────────────────────────
_COMMAND_PACKAGES: dict[str, str] = {
    r"\toprule": "booktabs",
    r"\midrule": "booktabs",
    r"\bottomrule": "booktabs",
    r"\cmidrule": "booktabs",
    r"\multirow": "multirow",
    r"\multicolumn": "multirow",
    r"\xcolor": "xcolor",
    r"\textcolor": "xcolor",
    r"\color": "xcolor",
    r"\url": "url",
    r"\href": "hyperref",
    r"\FloatBarrier": "placeins",
    r"\resizebox": "graphicx",
    r"\subcaption": "subcaption",
    r"\subfigure": "subfigure",
}


@dataclass
class CompileResult:
    success: bool
    pdf_path: Path | None = None
    error_log: str = ""


def compile_pdf(paper_dir: Path, config: Config | None = None) -> CompileResult:
    """Compile paper.tex to PDF using tectonic.

    Copies figures into the same directory as paper.tex for compilation.
    """
    tex_path = paper_dir / "paper.tex"
    if not tex_path.exists():
        return CompileResult(success=False, error_log="paper.tex not found")

    tectonic = (config.tectonic_path if config else "tectonic")

    # Copy figures alongside tex for tectonic
    figures_dir = paper_dir / "figures"
    if figures_dir.exists():
        for ext in ("*.pdf", "*.png", "*.jpg", "*.jpeg"):
            for f in figures_dir.glob(ext):
                dst = paper_dir / f.name
                try:
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(str(f), str(dst))
                except OSError:
                    pass

    try:
        result = subprocess.run(
            [tectonic, str(tex_path), "--keep-logs"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            cwd=str(paper_dir),
        )

        pdf_path = tex_path.with_suffix(".pdf")

        if result.returncode == 0 and pdf_path.exists():
            logger.info("PDF compiled successfully: %s", pdf_path)
            return CompileResult(success=True, pdf_path=pdf_path)
        else:
            error = result.stderr or result.stdout or "Unknown compilation error"
            logger.warning("Compilation failed: %s", error[:500])
            return CompileResult(success=False, error_log=error)

    except FileNotFoundError:
        return CompileResult(
            success=False,
            error_log=f"Tectonic not found at: {tectonic}",
        )
    except subprocess.TimeoutExpired:
        return CompileResult(success=False, error_log="Compilation timed out (120s)")
    except Exception as exc:
        return CompileResult(success=False, error_log=str(exc))


def deterministic_fix(tex: str, error_log: str) -> tuple[str, int]:
    """Apply deterministic fixes based on compilation error log.

    Returns (fixed_tex, number_of_fixes_applied).
    """
    fixes = 0

    # 1. Missing $ inserted — bare _ or ^ outside math mode
    if "Missing $ inserted" in error_log or "Missing \\$ inserted" in error_log:
        # Find bare _ not inside math, commands, or labels
        # Only fix _ in prose lines (not starting with \)
        lines = tex.split('\n')
        new_lines = []
        in_math = False
        for line in lines:
            stripped = line.lstrip()
            # Skip lines in math environments, commands, comments
            if stripped.startswith('%') or stripped.startswith('\\begin{') or stripped.startswith('\\end{'):
                new_lines.append(line)
                continue
            if re.search(r'\\begin\{(?:equation|align|gather|math)', stripped):
                in_math = True
            if re.search(r'\\end\{(?:equation|align|gather|math)', stripped):
                in_math = False
            if not in_math and not stripped.startswith('\\'):
                original = line
                # Escape bare _ not in \command{} or $...$
                line = re.sub(
                    r'(?<!\\)(?<!\$)(?<![{])_(?![{}])',
                    r'\\_',
                    line,
                )
                if line != original:
                    fixes += 1
            new_lines.append(line)
        tex = '\n'.join(new_lines)

    # 2. Undefined control sequence — auto-add missing packages
    if "Undefined control sequence" in error_log:
        preamble_end = tex.find(r'\begin{document}')
        if preamble_end > 0:
            preamble = tex[:preamble_end]
            for cmd, pkg in _COMMAND_PACKAGES.items():
                if cmd in tex and pkg not in preamble:
                    # Insert usepackage before \begin{document}
                    tex = tex[:preamble_end] + f"\\usepackage{{{pkg}}}\n" + tex[preamble_end:]
                    preamble_end += len(f"\\usepackage{{{pkg}}}\n")
                    fixes += 1
                    logger.info("Auto-added \\usepackage{%s} for %s", pkg, cmd)

    # 3. Too many unprocessed floats — insert \clearpage
    if "Too many unprocessed floats" in error_log:
        # Find sections with many consecutive floats and add \clearpage
        float_pat = re.compile(r'\\end\{(?:figure|table)\*?\}')
        positions = [m.end() for m in float_pat.finditer(tex)]
        # Insert \clearpage after every 3rd consecutive float
        if len(positions) >= 4:
            # Find clusters of 3+ floats within 200 chars of each other
            insertions = []
            cluster_count = 1
            for i in range(1, len(positions)):
                between = tex[positions[i-1]:positions[i]].strip()
                # Check if there's text between floats
                non_float = re.sub(
                    r'\\begin\{(?:figure|table)\*?\}.*?\\end\{(?:figure|table)\*?\}',
                    '', between, flags=re.DOTALL,
                ).strip()
                if not non_float:
                    cluster_count += 1
                    if cluster_count >= 3:
                        insertions.append(positions[i])
                        cluster_count = 0
                else:
                    cluster_count = 1

            for pos in reversed(insertions):
                tex = tex[:pos] + "\n\\clearpage\n" + tex[pos:]
                fixes += 1

    # 4. Misplaced \noalign — triple backslash \\\ in tabular rows
    if "Misplaced" in error_log and "noalign" in error_log:
        # Inside tabular environments, replace \\\ (triple) with \\ (double)
        def _fix_triple_backslash(m: re.Match) -> str:
            body = m.group(2)
            original = body
            body = re.sub(r'\\\\\\(?!\\)', r'\\\\', body)
            if body != original:
                nonlocal fixes
                fixes += 1
            return m.group(1) + body + m.group(3)

        tex = re.sub(
            r'(\\begin\{tabular[*x]?\}(?:\[[^\]]*\])?\{[^}]*\})'
            r'(.*?)'
            r'(\\end\{tabular[*x]?\})',
            _fix_triple_backslash,
            tex,
            flags=re.DOTALL,
        )

    # 5. Package hyperref Error — fix \href format
    if "Package hyperref Error" in error_log:
        # Fix \href without braces: \href URL{text} → \href{URL}{text}
        tex = re.sub(
            r'\\href\s+([^\s{]+)\{',
            r'\\href{\1}{',
            tex,
        )
        fixes += 1

    if fixes:
        logger.info("Deterministic fix: applied %d fix(es) from error log", fixes)
    return tex, fixes


def llm_fix_loop(
    tex: str,
    error_log: str,
    llm_client,
    max_attempts: int = 3,
) -> str:
    """Use LLM to fix compilation errors via search-replace edits."""
    for attempt in range(max_attempts):
        system = (
            "You are a LaTeX compilation error fixer. "
            "Given the error log and LaTeX source, output a JSON array of "
            "search-replace edits to fix the error.\n\n"
            "OUTPUT FORMAT — respond with ONLY a JSON array:\n"
            '[{"old": "broken text", "new": "fixed text"}]\n\n'
            "RULES:\n"
            "- Each 'old' string must appear EXACTLY in the source\n"
            "- Make minimal, targeted changes\n"
            "- Fix the root cause, not symptoms"
        )

        # Truncate error log
        if len(error_log) > 3000:
            error_log = error_log[:1500] + "\n...\n" + error_log[-1500:]

        # Truncate tex for context
        tex_snippet = tex
        if len(tex_snippet) > 15000:
            tex_snippet = tex_snippet[:7500] + "\n...\n" + tex_snippet[-7500:]

        user = (
            f"## Error Log\n```\n{error_log}\n```\n\n"
            f"## LaTeX Source\n```\n{tex_snippet}\n```"
        )

        response = llm_client.generate(system, user, json_mode=True)
        edits = llm_client.safe_parse_json(response, [])

        if isinstance(edits, dict):
            edits = [edits]
        if not isinstance(edits, list):
            logger.warning("LLM returned invalid edits format")
            break

        applied = 0
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            old = edit.get("old", "")
            new = edit.get("new", "")
            if old and old in tex and old != new:
                tex = tex.replace(old, new, 1)
                applied += 1

        if applied == 0:
            logger.info("LLM fix attempt %d: no edits applied", attempt + 1)
            break

        logger.info("LLM fix attempt %d: applied %d edit(s)", attempt + 1, applied)

    return tex
