"""Figure placement fixes — the core of Project_P.

Handles: misplaced figures, missing figures, consecutive figures, post-bib figures,
figure height cap, and intro figure relocation.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .._helpers import escape_latex_text, find_section_end, find_bib_position
from ..skeleton import parse_skeleton, insert_at, remove_block

logger = logging.getLogger(__name__)

# ── Figure type classification ───────────────────────────────────────────────

# Section-hint map: keyword in label/filename → target section.
# Order matters: more specific first.
_FIGURE_SECTION_HINTS: list[tuple[str, str]] = [
    # Experiments (check first — most figures belong here)
    ("result", "Experiments"),
    ("comparison", "Experiments"),
    ("performance", "Experiments"),
    ("ablation", "Experiments"),
    ("efficiency", "Experiments"),
    ("tradeoff", "Experiments"),
    ("training", "Experiments"),
    ("convergence", "Experiments"),
    ("qualitative", "Experiments"),
    ("visualization", "Experiments"),
    ("accuracy", "Experiments"),
    ("loss", "Experiments"),
    ("bar", "Experiments"),
    ("curve", "Experiments"),
    ("case", "Experiments"),
    # Introduction
    ("overview", "Introduction"),
    ("task", "Introduction"),
    ("motivation", "Introduction"),
    ("teaser", "Introduction"),
    ("intuition", "Introduction"),
    ("illustration", "Introduction"),
    # Method (check last — "model" is ambiguous)
    ("architecture", "Method"),
    ("arch", "Method"),
    ("framework", "Method"),
    ("pipeline", "Method"),
    ("diagram", "Method"),
    ("workflow", "Method"),
    ("detail", "Method"),
    ("model", "Method"),
]

# Labels legitimate in Introduction (architecture/overview figures)
_INTRO_KEEP_LABELS = re.compile(
    r'arch|overview|model|framework|pipeline|system|fig1|fig_1',
    re.IGNORECASE,
)


def fix_figure_placement(tex: str, figures_dir: Path | None) -> str:
    """Master figure fix pipeline."""
    # Step 1: Extract figures from list environments
    tex = _extract_figures_from_lists(tex)

    # Step 2: Relocate non-architecture figures out of Introduction
    tex = _relocate_intro_figures(tex)

    # Step 2b: Relocate misplaced figures to correct section (skeleton-based)
    tex = _relocate_misplaced_figures(tex)

    # Step 3: Fix \end{document} placement (ensures bib is correct)
    tex = _fix_end_document_placement(tex)

    # Step 4: Relocate figures stranded after bibliography
    tex = _relocate_post_bib_figures(tex)

    # Step 5: Inject missing figures from figures/ directory
    if figures_dir and figures_dir.exists():
        tex = _inject_missing_figures(tex, figures_dir)

    # Step 6: Deduplicate figure blocks referencing same image file
    tex = _dedup_figures(tex)

    # Step 7: Remove figure blocks with no valid \includegraphics
    tex = remove_empty_figure_blocks(tex)

    # Step 8: Enforce height cap on all \includegraphics
    tex = _enforce_figure_height_cap(tex)

    # Step 9: Detect and remove placeholder/failure figures
    tex = _remove_placeholder_figures(tex)

    # Step 10: Spread consecutive figures
    tex = _spread_consecutive_figures(tex)

    return tex


# ── Smart placement ──────────────────────────────────────────────────────────

def _insert_figure_near_ref(
    content: str, fig_key: str, figure_block: str,
) -> tuple[str, bool]:
    r"""Insert figure_block after the paragraph that first references fig_key.

    Returns (new_content, was_inserted).
    """
    pattern = re.compile(
        rf'\\(?:ref|autoref|cref)\{{fig:{re.escape(fig_key)}\}}',
        re.IGNORECASE,
    )
    match = pattern.search(content)
    if not match:
        return content, False

    search_start = match.end()
    para_end = re.search(
        r'\n\s*\n|\\(?:sub){0,2}section\{|\\paragraph\{'
        r'|\\begin\{table\*?\}|\\begin\{figure\*?\}',
        content[search_start:],
    )
    if para_end:
        insert_pos = search_start + para_end.start()
    else:
        insert_pos = len(content)

    # Don't insert after bibliography
    bib_pos = find_bib_position(content)
    end_doc = content.find(r'\end{document}')
    if end_doc >= 0:
        bib_pos = min(bib_pos, end_doc)
    if insert_pos > bib_pos:
        insert_pos = bib_pos

    new_content = content[:insert_pos] + "\n\n" + figure_block + "\n" + content[insert_pos:]
    return new_content, True


def _extract_fig_label(figure_block: str) -> str:
    """Extract the figure label suffix from a figure block."""
    label_m = re.search(r'\\label\{fig:([^}]+)\}', figure_block)
    if label_m:
        return label_m.group(1)
    label_m = re.search(r'\\label\{([^}]+)\}', figure_block)
    if label_m:
        raw = label_m.group(1)
        return raw[4:] if raw.startswith("fig:") else raw
    return ""


def _classify_figure_section(fig_label: str, figure_block: str) -> str:
    """Determine which section a figure belongs to based on label/filename hints."""
    fig_key_lower = fig_label.lower() if fig_label else ""
    incl_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', figure_block)
    file_hint = incl_m.group(1).lower() if incl_m else ""

    for keyword, section in _FIGURE_SECTION_HINTS:
        if keyword in fig_key_lower or keyword in file_hint:
            return section
    return "Experiments"


def smart_place_figure(content: str, figure_block: str) -> str:
    r"""Place a figure block at the best position in the document.

    Uses skeleton-based Edit mode for precise placement:
    1. Parse document skeleton to know exact section boundaries
    2. Near first \ref{fig:label} (paragraph-end precision)
    3. At end of correct section (skeleton-based)
    4. Before bibliography (last resort)
    """
    fig_label = _extract_fig_label(figure_block)

    # Parse skeleton for precise positioning
    try:
        skel = parse_skeleton(content)
    except Exception:
        skel = None

    # Strategy 1: near first \ref (skeleton-aware)
    if fig_label and skel:
        pos = skel.find_insert_point_after_ref(content, fig_label)
        if pos is not None:
            content = insert_at(content, pos, figure_block)
            logger.info("Placed figure (label=%s) near \\ref (skeleton mode)", fig_label)
            return content

    # Fallback: regex-based ref placement
    if fig_label:
        new_content, placed = _insert_figure_near_ref(content, fig_label, figure_block)
        if placed:
            return new_content

    # Strategy 2: end of appropriate section (skeleton-based)
    target_section = _classify_figure_section(fig_label, figure_block)

    if skel:
        sec_node = skel.find_section(target_section)
        if sec_node:
            pos = sec_node.end
            # Back up past trailing whitespace
            while pos > 0 and content[pos - 1] in ('\n', '\r', ' ', '\t'):
                pos -= 1
            content = insert_at(content, pos, figure_block)
            logger.info("Placed figure (label=%s) at end of \\section{%s} (skeleton mode)",
                        fig_label, target_section)
            return content

    # Fallback: regex-based section end
    sec_end = find_section_end(content, target_section)
    if sec_end is not None:
        while sec_end > 0 and content[sec_end - 1] in ('\n', '\r', ' ', '\t'):
            sec_end -= 1
        content = content[:sec_end] + "\n\n" + figure_block + "\n\n" + content[sec_end:]
        logger.info("Placed figure (label=%s) at end of \\section{%s}", fig_label, target_section)
        return content

    # Strategy 3: before bibliography
    for anchor in (r'\bibliographystyle{', r'\bibliography{',
                    r'\begin{thebibliography}', r'\end{document}'):
        pos = content.find(anchor)
        if pos >= 0:
            content = content[:pos] + "\n\n" + figure_block + "\n\n" + content[pos:]
            logger.info("Placed figure (label=%s) before bibliography (last resort)", fig_label)
            return content

    return content + "\n\n" + figure_block + "\n"


# ── Skeleton-based misplaced figure relocation ───────────────────────────────

def _relocate_misplaced_figures(tex: str) -> str:
    """Use skeleton to detect figures whose label/filename hint doesn't match
    their current section, and relocate them to the correct section."""
    try:
        skel = parse_skeleton(tex)
    except Exception:
        return tex

    if not skel.sections or not skel.floats:
        return tex

    # Build section-name lookup: for each float, find which section it's in
    def _section_of(pos: int) -> str | None:
        for sec in skel.sections:
            if sec.start <= pos < sec.end:
                return sec.title
        return None

    relocations: list[tuple[int, int, str, str]] = []  # (start, end, block_text, target_section)

    for fb in skel.floats:
        if not fb.env_type.startswith("figure"):
            continue
        fig_label = _extract_fig_label(tex[fb.start:fb.end])
        target = _classify_figure_section(fig_label, tex[fb.start:fb.end])
        current_sec = _section_of(fb.start)

        if current_sec is None:
            continue

        # Check if the figure is in the wrong section
        current_lower = current_sec.lower()
        target_lower = target.lower()

        # Don't relocate if current section matches target (even partially)
        if target_lower in current_lower or current_lower in target_lower:
            continue

        # Don't relocate Introduction-appropriate figures
        if _INTRO_KEEP_LABELS.search(fig_label or ""):
            continue

        # Don't relocate if there's a \ref to this figure in the current section
        if fig_label:
            current_node = skel.find_section(current_sec)
            if current_node:
                sec_text = tex[current_node.start:current_node.end]
                ref_pat = re.compile(
                    rf'\\(?:ref|autoref|cref)\{{fig:{re.escape(fig_label)}\}}',
                )
                if ref_pat.search(sec_text):
                    continue

        # Only relocate if the target section exists in the skeleton
        target_node = skel.find_section(target)
        if target_node is None:
            continue

        relocations.append((fb.start, fb.end, tex[fb.start:fb.end], target))

    if not relocations:
        return tex

    # Remove blocks in reverse order, then re-place them
    for start, end, block_text, target in reversed(relocations):
        tex = remove_block(tex, start, end)
        logger.info("Relocating misplaced figure from current section to %s", target)

    # Re-parse skeleton after removals, then place figures
    # Place directly at start of target section (not smart_place which may
    # find a \ref back in the original section and undo the relocation)
    for _, _, block_text, target in relocations:
        try:
            skel2 = parse_skeleton(tex)
        except Exception:
            skel2 = None

        placed = False
        if skel2:
            target_node = skel2.find_section(target)
            if target_node:
                # Insert right after the section heading line
                heading_end = tex.find('\n', target_node.start)
                if heading_end < 0:
                    heading_end = target_node.start + 50
                insert_pos = heading_end + 1
                tex = insert_at(tex, insert_pos, block_text)
                logger.info("Placed figure at start of \\section{%s}", target)
                placed = True

        if not placed:
            tex = smart_place_figure(tex, block_text)

    return tex


# ── Extract figures from lists ───────────────────────────────────────────────

def _extract_figures_from_lists(text: str) -> str:
    """Move figure blocks out of itemize/enumerate environments."""
    fig_pattern = re.compile(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.DOTALL)
    for env in ('itemize', 'enumerate'):
        pat = re.compile(
            rf'(\\begin\{{{env}\}})'
            rf'((?:(?!\\begin\{{{env}\}})(?!\\end\{{{env}\}}).)*?)'
            rf'(\\end\{{{env}\}})',
            re.DOTALL,
        )

        def _move_figs(m: re.Match) -> str:
            body = m.group(2)
            figs = list(fig_pattern.finditer(body))
            if not figs:
                return m.group(0)
            extracted: list[str] = []
            new_body = body
            for fm in reversed(figs):
                extracted.insert(0, fm.group(0))
                new_body = new_body[:fm.start()] + new_body[fm.end():]
            new_body = re.sub(r'\n{3,}', '\n\n', new_body)
            return m.group(1) + new_body + m.group(3) + '\n\n' + '\n\n'.join(extracted)

        text = pat.sub(_move_figs, text)
    return text


# ── Relocate intro figures ───────────────────────────────────────────────────

def _relocate_intro_figures(text: str) -> str:
    """Move non-architecture figures out of Introduction."""
    intro_match = re.search(r'(\\section\{Introduction\})', text, re.IGNORECASE)
    if not intro_match:
        return text

    intro_start = intro_match.end()
    next_section = re.search(r'\\section\{', text[intro_start:])
    if not next_section:
        return text

    intro_end = intro_start + next_section.start()
    intro_text = text[intro_start:intro_end]

    fig_pattern = re.compile(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.DOTALL)
    figures_in_intro = list(fig_pattern.finditer(intro_text))
    if not figures_in_intro:
        return text

    to_relocate: list[tuple[str, str]] = []
    new_intro = intro_text
    for m in reversed(figures_in_intro):
        fig_block = m.group(0)
        label_match = re.search(r'\\label\{([^}]+)\}', fig_block)
        label = label_match.group(1) if label_match else ""

        if _INTRO_KEEP_LABELS.search(label):
            continue

        to_relocate.insert(0, (label, fig_block))
        new_intro = new_intro[:m.start()] + new_intro[m.end():]

    if not to_relocate:
        return text

    text = text[:intro_start] + new_intro + text[intro_end:]
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Find updated intro boundary for ref placement
    intro_match2 = re.search(r'\\section\{Introduction\}', text, re.IGNORECASE)
    next_sec2 = re.search(r'\\section\{', text[intro_match2.end():]) if intro_match2 else None
    intro_end_pos = (intro_match2.end() + next_sec2.start()) if (intro_match2 and next_sec2) else 0

    for _label, fig_block in to_relocate:
        lm = re.search(r'\\label\{fig:([^}]+)\}', fig_block)
        fig_label = lm.group(1) if lm else ""

        placed = False
        if fig_label:
            ref_pat = re.compile(r'\\ref\{fig:' + re.escape(fig_label) + r'\}')
            for ref_m in ref_pat.finditer(text):
                if ref_m.start() < intro_end_pos:
                    continue
                para_end = text.find('\n\n', ref_m.end())
                if para_end == -1:
                    para_end = len(text)
                text = text[:para_end] + "\n\n" + fig_block + "\n" + text[para_end:]
                placed = True
                break

        if not placed:
            exp_end = find_section_end(text, "Experiments")
            if exp_end is not None:
                text = text[:exp_end] + "\n\n" + fig_block + "\n" + text[exp_end:]
            else:
                anchor = re.search(
                    r'\\bibliographystyle\{|\\bibliography\{|'
                    r'\\begin\{thebibliography\}|\\end\{document\}',
                    text,
                )
                pos = anchor.start() if anchor else len(text)
                text = text[:pos] + "\n\n" + fig_block + "\n\n" + text[pos:]

    return text


# ── Fix \end{document} placement ─────────────────────────────────────────────

def _fix_end_document_placement(text: str) -> str:
    r"""Ensure exactly one \end{document} at the end, with bibliography before it."""
    if r'\begin{document}' not in text:
        return text

    end_doc_positions = [m.start() for m in re.finditer(r'\\end\{document\}', text)]
    if not end_doc_positions:
        text = text.rstrip()
        text += "\n\n\\bibliographystyle{plainnat}"
        text += "\n\\bibliography{references}"
        text += "\n\n\\end{document}\n"
        return text

    if len(end_doc_positions) == 1:
        end_pos = end_doc_positions[0]
        has_bib_before = (
            re.search(r'\\bibliography\{', text[:end_pos])
            or re.search(r'\\begin\{thebibliography\}', text[:end_pos])
        )
        if has_bib_before:
            return text

    # Extract bibliography commands
    bib_style_m = re.search(r'\\bibliographystyle\{([^}]+)\}', text)
    bib_file_m = re.search(r'\\bibliography\{([^}]+)\}', text)
    bib_style = bib_style_m.group(1) if bib_style_m else "plainnat"
    bib_file = bib_file_m.group(1) if bib_file_m else "references"

    inline_bib_m = re.search(
        r'(\\begin\{thebibliography\}.*?\\end\{thebibliography\})',
        text, re.DOTALL,
    )
    inline_bib = inline_bib_m.group(1) if inline_bib_m else ""

    # Remove all and re-append
    text = re.sub(r'\\end\{document\}\s*', '', text)
    text = re.sub(r'\\bibliographystyle\{[^}]*\}\s*', '', text)
    text = re.sub(r'\\bibliography\{[^}]*\}\s*', '', text)
    if inline_bib:
        text = text.replace(inline_bib, '')

    text = text.rstrip()

    if inline_bib:
        text += "\n\n" + inline_bib
    else:
        text += f"\n\n\\bibliographystyle{{{bib_style}}}"
        text += f"\n\\bibliography{{{bib_file}}}"
    text += "\n\n\\end{document}\n"

    return text


# ── Relocate post-bibliography figures ────────────────────────────────────────

def _relocate_post_bib_figures(text: str) -> str:
    """Move figure blocks after bibliography back to proper positions."""
    bib_pos = find_bib_position(text)
    if bib_pos >= len(text):
        return text

    after_bib = text[bib_pos:]
    fig_blocks = list(re.finditer(
        r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',
        after_bib, re.DOTALL,
    ))
    if not fig_blocks:
        return text

    extracted: list[str] = []
    for m in reversed(fig_blocks):
        extracted.insert(0, m.group(0))
        start = bib_pos + m.start()
        end = bib_pos + m.end()
        while end < len(text) and text[end] in (' ', '\n', '\r'):
            end += 1
        text = text[:start] + text[end:]

    for fig_block in extracted:
        text = smart_place_figure(text, fig_block)

    logger.info("Relocated %d figure(s) from after bibliography", len(extracted))
    return text


# ── Inject missing figures ───────────────────────────────────────────────────

def _inject_missing_figures(tex: str, figures_dir: Path) -> str:
    """Scan figures/ for image files not referenced in tex; inject emergency blocks."""
    # Exclude known non-figure files (e.g., paper.pdf compile output)
    _EXCLUDE_STEMS = {"paper", "main", "output", "draft"}
    existing_images: list[str] = []
    for ext in ("*.png", "*.pdf", "*.jpg", "*.jpeg"):
        existing_images.extend(
            f.name for f in figures_dir.glob(ext)
            if f.stem.lower() not in _EXCLUDE_STEMS
        )

    if not existing_images:
        return tex

    # Find all \includegraphics references — compare by stem to avoid .png/.pdf duplication
    included = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex))
    included_stems = {Path(p).stem.lower() for p in included}

    # Deduplicate by stem: keep only one extension per stem (prefer .pdf)
    seen_stems: set[str] = set()
    unique_images: list[str] = []
    for img in sorted(existing_images, key=lambda x: (0 if x.endswith('.pdf') else 1)):
        s = Path(img).stem.lower()
        if s not in seen_stems:
            seen_stems.add(s)
            unique_images.append(img)

    missing = [img for img in unique_images if Path(img).stem.lower() not in included_stems]

    for img_name in missing:
        stem = Path(img_name).stem
        # Check if there's a \ref{fig:stem} — if so, it was meant to be included
        has_ref = re.search(rf'\\ref\{{fig:{re.escape(stem)}\}}', tex)
        if not has_ref:
            # Also check for partial match
            has_ref = re.search(rf'\\ref\{{fig:[^}}]*{re.escape(stem)}[^}}]*\}}', tex)

        if not has_ref:
            continue  # No reference to this figure, skip

        label = f"fig:{stem}"
        caption = stem.replace("_", " ").replace("-", " ").title()
        block = (
            "\\begin{figure}[t!]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.85\\textwidth, "
            f"height=0.32\\textheight, keepaspectratio]"
            f"{{{img_name}}}\n"
            f"\\caption{{{escape_latex_text(caption)}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{figure}"
        )
        tex = smart_place_figure(tex, block)
        logger.info("Injected missing figure: %s", img_name)

    return tex


# ── Figure deduplication ─────────────────────────────────────────────────────

def _dedup_figures(tex: str) -> str:
    """Remove duplicate figure blocks that include the same image file.

    When duplicates exist, keep the one with the longer caption (more
    informative). Rewrite \\ref commands pointing to removed labels.
    """
    fig_pat = re.compile(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.DOTALL)
    blocks = list(fig_pat.finditer(tex))
    if not blocks:
        return tex

    # Group by includegraphics stem
    stem_groups: dict[str, list[tuple[re.Match, str, str, int]]] = {}
    for m in blocks:
        block = m.group(0)
        inc_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
        if not inc_m:
            continue
        stem = Path(inc_m.group(1)).stem.lower()
        label_m = re.search(r'\\label\{([^}]+)\}', block)
        label = label_m.group(1) if label_m else ""
        cap_m = re.search(r'\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', block)
        cap_len = len(cap_m.group(1)) if cap_m else 0
        stem_groups.setdefault(stem, []).append((m, label, block, cap_len))

    removals: list[tuple[int, int, str]] = []  # (start, end, label_to_remove)
    ref_rewrites: dict[str, str] = {}  # old_label → kept_label

    for stem, group in stem_groups.items():
        if len(group) <= 1:
            continue

        # Keep the one with longest caption
        group.sort(key=lambda x: x[3], reverse=True)
        kept = group[0]
        kept_label = kept[1]

        for dup_match, dup_label, _, _ in group[1:]:
            removals.append((dup_match.start(), dup_match.end(), dup_label))
            if dup_label and kept_label:
                ref_rewrites[dup_label] = kept_label
            logger.info(
                "Dedup: removing duplicate figure (stem=%s, label=%s), keeping label=%s",
                stem, dup_label, kept_label,
            )

    if not removals:
        return tex

    # Remove blocks in reverse order
    for start, end, _ in sorted(removals, key=lambda x: x[0], reverse=True):
        # Also remove surrounding blank lines
        while end < len(tex) and tex[end] in ('\n', '\r', ' '):
            end += 1
        tex = tex[:start] + tex[end:]

    # Rewrite refs pointing to removed labels
    for old_label, new_label in ref_rewrites.items():
        tex = re.sub(
            rf'(\\(?:ref|autoref|cref|eqref))\{{{re.escape(old_label)}\}}',
            rf'\1{{{new_label}}}',
            tex,
        )

    tex = re.sub(r'\n{3,}', '\n\n', tex)
    return tex


# ── Remove empty figure blocks ──────────────────────────────────────────────

def remove_empty_figure_blocks(tex: str) -> str:
    """Remove figure blocks that have no valid (uncommented) \\includegraphics.

    Also removes dangling \\ref commands that point to removed labels.
    """
    fig_pat = re.compile(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.DOTALL)
    blocks = list(fig_pat.finditer(tex))

    to_remove: list[tuple[int, int, str]] = []
    for m in blocks:
        block = m.group(0)
        # Check for at least one UNCOMMENTED \includegraphics
        has_valid = any(
            'includegraphics' in line and not line.strip().startswith('%')
            for line in block.split('\n')
        )
        if has_valid:
            continue

        label_m = re.search(r'\\label\{([^}]+)\}', block)
        label = label_m.group(1) if label_m else ""
        to_remove.append((m.start(), m.end(), label))
        logger.info("Removing empty figure block (label=%s)", label)

    if not to_remove:
        return tex

    for start, end, label in sorted(to_remove, key=lambda x: x[0], reverse=True):
        # Remove surrounding blank lines too
        while end < len(tex) and tex[end] in ('\n', '\r', ' '):
            end += 1
        tex = tex[:start] + tex[end:]

        # Clean up \ref commands to removed label
        if label:
            esc = re.escape(label)
            _figref = (rf'(?:Figure|Fig\.?)~?'
                       rf'\\(?:ref|autoref|cref)\{{{esc}\}}')
            # Pattern 1: sentence whose subject is Figure~\ref{X}
            # Remove from "Figure~\ref{X}" to the first ". " + uppercase
            # (sentence boundary), handling decimals like "2.0" safely.
            tex = re.sub(
                _figref + r'.*?\.(?=\s+[A-Z]|\s*\n|\s*$)\s*',
                '',
                tex,
            )
            # Pattern 2: remaining "Figure~\ref{X}" mid-sentence
            tex = re.sub(_figref, '', tex)
            # Pattern 3: standalone \ref{X}
            tex = re.sub(
                rf'\\(?:ref|autoref|cref)\{{{esc}\}}',
                '',
                tex,
            )

    tex = re.sub(r'\n{3,}', '\n\n', tex)
    return tex


# ── Height cap ───────────────────────────────────────────────────────────────

def _enforce_figure_height_cap(latex: str) -> str:
    r"""Ensure every \includegraphics has a height cap."""
    _HEIGHT_OPTS = "height=0.32\\textheight, keepaspectratio"

    def _add_height(m: re.Match) -> str:
        opts = m.group(1)
        if "height=" in opts:
            return m.group(0)
        new_opts = opts + ", " + _HEIGHT_OPTS
        return f"\\includegraphics[{new_opts}]" + m.group(2)

    latex = re.sub(
        r'\\includegraphics\[([^\]]+)\](\{[^}]+\})',
        _add_height,
        latex,
    )

    def _add_full_opts(m: re.Match) -> str:
        filename = m.group(1)
        return (f"\\includegraphics"
                f"[width=0.85\\textwidth, {_HEIGHT_OPTS}]"
                f"{{{filename}}}")

    latex = re.sub(r'\\includegraphics\{([^}]+)\}', _add_full_opts, latex)

    return latex


# ── Spread consecutive figures ───────────────────────────────────────────────

def _spread_consecutive_figures(text: str) -> str:
    """Detect consecutive figure blocks with no text between and spread them apart.

    Avoids spreading figures near the end of the paper (close to bibliography)
    because separating them there creates pages with a single figure and large
    whitespace — worse than keeping them stacked.
    """
    fig_env = re.compile(
        r'(\\begin\{figure\*?\})(.*?)(\\end\{figure\*?\})',
        re.DOTALL,
    )

    # Find bibliography position — don't spread figures close to it
    bib_pos = find_bib_position(text)

    max_passes = 20
    pass_count = 0
    i = 0
    while pass_count < max_passes:
        figures = list(fig_env.finditer(text))
        if i >= len(figures) - 1:
            break
        pass_count += 1
        fig_a = figures[i]
        fig_b = figures[i + 1]
        between = text[fig_a.end():fig_b.start()]
        if between.strip() != "":
            i += 1
            continue

        # Skip spreading if these figures are near end of paper.
        # Check: is there <1500 chars of text between fig_b and bibliography?
        # If so, there's not enough content to fill a page, so keep them together.
        text_after_b = text[fig_b.end():bib_pos]
        text_content = re.sub(
            r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '', text_after_b, flags=re.DOTALL
        ).strip()
        if len(text_content) < 1500:
            logger.info(
                "Keeping consecutive figures together (near end of paper, only %d chars after)",
                len(text_content),
            )
            i += 1
            continue

        block_b = fig_b.group(0)
        text = text[:fig_b.start()] + text[fig_b.end():]
        text = re.sub(r'\n{3,}', '\n\n', text)

        text = smart_place_figure(text, block_b)

        # Check if still consecutive after re-placement
        figures_new = list(fig_env.finditer(text))
        still_consecutive = False
        for j in range(len(figures_new) - 1):
            if figures_new[j + 1].group(0) == block_b:
                gap = text[figures_new[j].end():figures_new[j + 1].start()]
                if gap.strip() == "":
                    still_consecutive = True
                break

        if still_consecutive:
            # Insert \FloatBarrier between them so LaTeX processes them
            # in order, but both keep [t!] placement
            for j in range(len(figures_new) - 1):
                if figures_new[j + 1].group(0) == block_b:
                    insert_pos = figures_new[j].end()
                    text = text[:insert_pos] + "\n\\FloatBarrier\n" + text[insert_pos:]
                    break
            i += 1
        # else: don't advance, re-check

    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text


# ── Placeholder / failure caption detection ──────────────────────────────────

_PLACEHOLDER_CAPTION_PATS = [
    re.compile(r'placeholder', re.IGNORECASE),
    re.compile(r'experiment\s+fail', re.IGNORECASE),
    re.compile(r'training\s+fail', re.IGNORECASE),
    re.compile(r'failure\s+summary', re.IGNORECASE),
    re.compile(r'error\s+source', re.IGNORECASE),
    re.compile(r'no\s+real\s+training', re.IGNORECASE),
    re.compile(r'synthetic\s+data\s+only', re.IGNORECASE),
    re.compile(r'dummy\s+(?:data|figure|result)', re.IGNORECASE),
    re.compile(r'\bTODO\b'),
    re.compile(r'\bFIXME\b'),
    re.compile(r'\bTBD\b'),
    re.compile(r'to\s+be\s+(?:replaced|updated|added)', re.IGNORECASE),
]


def _remove_placeholder_figures(tex: str) -> str:
    """Detect and remove figure blocks with placeholder/failure captions."""
    fig_pat = re.compile(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.DOTALL)
    blocks = list(fig_pat.finditer(tex))

    to_remove: list[tuple[int, int, str]] = []

    for m in blocks:
        block = m.group(0)
        cap_m = re.search(
            r'\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}',
            block,
        )
        if not cap_m:
            continue
        caption = cap_m.group(1)

        # Check caption text
        matched_reason = None
        for pat in _PLACEHOLDER_CAPTION_PATS:
            if pat.search(caption):
                matched_reason = f"caption matched '{pat.pattern}'"
                break

        # Check filename for "placeholder" if caption didn't match
        if not matched_reason:
            incl_m = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', block)
            if incl_m:
                fname = incl_m.group(1).lower()
                if 'placeholder' in fname:
                    matched_reason = f"filename '{incl_m.group(1)}' contains 'placeholder'"

        if matched_reason:
            label_m = re.search(r'\\label\{([^}]+)\}', block)
            label = label_m.group(1) if label_m else ""
            logger.warning(
                "Removing placeholder figure (label=%s, %s)",
                label or "(none)", matched_reason,
            )
            to_remove.append((m.start(), m.end(), label))

    if not to_remove:
        return tex

    for start, end, label in sorted(to_remove, key=lambda x: x[0], reverse=True):
        while end < len(tex) and tex[end] in ('\n', '\r', ' '):
            end += 1
        tex = tex[:start] + tex[end:]

        # Clean up \ref commands to removed label
        if label:
            esc = re.escape(label)
            # "Figure~\ref{X} ..." sentence → remove
            _figref = (rf'(?:Figure|Fig\.?)~?'
                       rf'\\(?:ref|autoref|cref)\{{{esc}\}}')
            tex = re.sub(
                _figref + r'.*?\.(?=\s+[A-Z]|\s*\n|\s*$)\s*',
                '', tex,
            )
            tex = re.sub(_figref, '', tex)
            tex = re.sub(
                rf'\\(?:ref|autoref|cref)\{{{esc}\}}',
                '', tex,
            )

    tex = re.sub(r'\n{3,}', '\n\n', tex)
    return tex
