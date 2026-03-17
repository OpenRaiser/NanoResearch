"""Main pipeline: orchestrates all fixes and compilation."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import Config
from .compiler import compile_pdf, deterministic_fix, llm_fix_loop, CompileResult
from .fixers import run_all_fixes
from .fixers.bibtex import fix_bibtex
from .fixers.figure_trim import trim_all_figures
from .validators import run_all_checks
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def _create_llm_client(config: Config) -> LLMClient | None:
    """Create LLM client if configured."""
    if not config.llm.api_key:
        logger.info("No LLM API key configured, running without LLM")
        return None
    try:
        return LLMClient(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            model=config.llm.model,
            vision_model=config.llm.vision_model,
            temperature=config.llm.temperature,
            timeout=config.llm.timeout,
        )
    except Exception as exc:
        logger.warning("Failed to create LLM client: %s", exc)
        return None


def backup_paper(paper_dir: Path) -> Path:
    """Create .bak/ backup of original files. Returns backup dir."""
    bak_dir = paper_dir / ".bak"
    bak_dir.mkdir(exist_ok=True)

    tex_path = paper_dir / "paper.tex"
    if tex_path.exists():
        shutil.copy2(tex_path, bak_dir / "paper.tex")

    bib_path = paper_dir / "references.bib"
    if bib_path.exists():
        shutil.copy2(bib_path, bak_dir / "references.bib")

    figures_dir = paper_dir / "figures"
    if figures_dir.exists():
        shutil.copytree(figures_dir, bak_dir / "figures", dirs_exist_ok=True)

    logger.info("Backup created at %s", bak_dir)
    return bak_dir


def restore_paper(paper_dir: Path) -> bool:
    """Restore from .bak/ backup. Returns True if restored."""
    bak_dir = paper_dir / ".bak"
    if not bak_dir.exists():
        logger.error("No backup found at %s", bak_dir)
        return False

    bak_tex = bak_dir / "paper.tex"
    if bak_tex.exists():
        shutil.copy2(bak_tex, paper_dir / "paper.tex")

    bak_bib = bak_dir / "references.bib"
    if bak_bib.exists():
        shutil.copy2(bak_bib, paper_dir / "references.bib")

    bak_figs = bak_dir / "figures"
    if bak_figs.exists():
        dest_figs = paper_dir / "figures"
        if dest_figs.exists():
            shutil.rmtree(dest_figs)
        shutil.copytree(bak_figs, dest_figs)

    logger.info("Restored from backup %s", bak_dir)
    return True


def fix_paper(
    paper_dir: Path,
    *,
    use_llm: bool = True,
    do_compile: bool = True,
    dry_run: bool = False,
    config: Config | None = None,
) -> dict:
    """Main entry point: fix all paper formatting issues.

    Args:
        paper_dir: Directory containing paper.tex, references.bib, figures/
        use_llm: Whether to use LLM for vision trim and error fixing
        do_compile: Whether to compile PDF after fixes
        dry_run: If True, only report what would be fixed without writing

    Returns:
        dict with keys: fixes, compile_result, trimmed_figures
    """
    config = config or Config.load()
    llm_client = _create_llm_client(config) if use_llm else None

    tex_path = paper_dir / "paper.tex"
    bib_path = paper_dir / "references.bib"
    figures_dir = paper_dir / "figures"

    # Fallback: if figures/ subdir doesn't exist, check if paper_dir itself has images
    if not figures_dir.exists():
        _img_exts = {".png", ".pdf", ".jpg", ".jpeg"}
        has_images = any(f.suffix.lower() in _img_exts for f in paper_dir.iterdir() if f.is_file())
        if has_images:
            figures_dir = paper_dir
            logger.info("No figures/ subdir, using paper_dir as figures_dir")

    if not tex_path.exists():
        logger.error("paper.tex not found in %s", paper_dir)
        return {"error": "paper.tex not found"}

    # Backup
    backup_paper(paper_dir)

    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""

    all_fixes: list[str] = []

    # 1. Figure whitespace trimming (operates on image files)
    trimmed: list[str] = []
    if figures_dir.exists():
        trimmed = trim_all_figures(figures_dir, llm_client)
        if trimmed:
            all_fixes.append(f"figure_trim: trimmed {len(trimmed)} figures")

    # 2. LaTeX deterministic fixes
    tex, tex_fixes = run_all_fixes(tex, bib, figures_dir if figures_dir.exists() else None)
    all_fixes.extend(tex_fixes)

    # 3. BibTeX fixes
    if bib:
        bib, bib_fixes = fix_bibtex(bib)
        all_fixes.extend(bib_fixes)

    # 4. Cross-reference validation + auto-fix
    tex, ref_fixes = run_all_checks(
        tex, bib, figures_dir if figures_dir.exists() else None,
    )
    all_fixes.extend(ref_fixes)

    if dry_run:
        print_report(all_fixes, None)
        return {"fixes": all_fixes, "dry_run": True}

    # 5. Write fixed files
    tex_path.write_text(tex, encoding="utf-8")
    if bib:
        bib_path.write_text(bib, encoding="utf-8")

    # 6. Compile PDF
    result: CompileResult | None = None
    if do_compile:
        result = compile_pdf(paper_dir, config)

        # 7. Deterministic error fix if compilation failed
        if not result.success:
            logger.info("Compilation failed, trying deterministic fixes...")
            tex = tex_path.read_text(encoding="utf-8")
            tex, det_count = deterministic_fix(tex, result.error_log)
            if det_count > 0:
                tex_path.write_text(tex, encoding="utf-8")
                result = compile_pdf(paper_dir, config)
                if result.success:
                    all_fixes.append(f"deterministic_fix: fixed {det_count} compile error(s)")

        # 8. LLM error fix loop if still failing
        if not result.success and llm_client:
            logger.info("Compilation still failing, attempting LLM fix...")
            tex = tex_path.read_text(encoding="utf-8")
            tex = llm_fix_loop(tex, result.error_log, llm_client)
            tex_path.write_text(tex, encoding="utf-8")
            result = compile_pdf(paper_dir, config)
            if result.success:
                all_fixes.append("llm_fix: LLM fixed compilation errors")

    # 9. Report
    print_report(all_fixes, result)

    return {
        "fixes": all_fixes,
        "compile_result": result,
        "trimmed_figures": trimmed,
    }


def print_report(fixes: list[str], result: CompileResult | None) -> None:
    """Print a summary report."""
    print("\n" + "=" * 60)
    print("  Project_P Fix Report")
    print("=" * 60)

    if fixes:
        print(f"\n  Fixes applied ({len(fixes)}):")
        for i, fix in enumerate(fixes, 1):
            print(f"    {i}. {fix}")
    else:
        print("\n  No fixes needed.")

    if result is not None:
        print()
        if result.success:
            print(f"  PDF compiled successfully: {result.pdf_path}")
        else:
            print(f"  PDF compilation FAILED")
            if result.error_log:
                # Print first 10 lines of error
                lines = result.error_log.strip().split("\n")[:10]
                for line in lines:
                    print(f"    {line}")

    print("\n" + "=" * 60 + "\n")
