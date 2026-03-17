"""Float placement and table overflow fixes."""
from __future__ import annotations

import re


def fix_floats(text: str) -> str:
    """Normalize float placement specifiers and fix table overflow."""
    text = _ensure_placeins_package(text)
    text = _normalize_placement(text)
    text = _fix_inline_table_start(text)
    text = _fix_table_overflow(text)
    text = _collapse_math_blank_lines(text)
    return text


def _ensure_placeins_package(text: str) -> str:
    r"""Ensure \usepackage{placeins} is present (needed for \FloatBarrier)."""
    if 'placeins' in text:
        return text
    # Insert after last \usepackage line
    m = re.search(r'(\\usepackage(?:\[[^\]]*\])?\{[^}]+\}[^\n]*\n)', text)
    if m:
        # Find the last \usepackage
        last_pos = 0
        for pkg_m in re.finditer(r'\\usepackage(?:\[[^\]]*\])?\{[^}]+\}[^\n]*\n', text):
            last_pos = pkg_m.end()
        if last_pos > 0:
            text = text[:last_pos] + "\\usepackage[section]{placeins}\n" + text[last_pos:]
    return text


def _normalize_placement(text: str) -> str:
    """Normalize [H], [h], [h!], [b], [b!] → [t!] for figures and tables.

    [b!] causes half-empty pages when the figure is near the end of a section.
    [t!] with placeins package is the safest choice for academic papers.
    """
    # Figures: normalize any single-letter specifier to [t!]
    text = re.sub(r'\\begin\{figure\}\s*\[[HhBb]!?\]', r'\\begin{figure}[t!]', text)
    text = re.sub(r'\\begin\{figure\}(?!\[)', r'\\begin{figure}[t!]', text)
    text = re.sub(r'\\begin\{figure\*\}\s*\[[HhBb]!?\]', r'\\begin{figure*}[t!]', text)
    text = re.sub(r'\\begin\{figure\*\}(?!\s*\[)', r'\\begin{figure*}[t!]', text)

    # Tables
    text = re.sub(r'\\begin\{table\}\s*\[[Hh]!?\]', r'\\begin{table}[t!]', text)
    text = re.sub(r'\\begin\{table\}(?!\s*\[)', r'\\begin{table}[t!]', text)
    text = re.sub(r'\\begin\{table\*\}\s*\[[Hh]!?\]', r'\\begin{table*}[t!]', text)
    text = re.sub(r'\\begin\{table\*\}(?!\s*\[)', r'\\begin{table*}[t!]', text)

    return text


def _fix_table_overflow(text: str) -> str:
    r"""Inject \small, \tabcolsep, @{}, resizebox, and precision fixes into tables."""

    def _patch_table(match: re.Match) -> str:
        block = match.group(0)
        # Inject \small
        if "\\small" not in block:
            block = re.sub(
                r'(\\begin\{table\*?\}\[[^\]]*\])',
                r'\1\n\\small',
                block,
            )
        # Inject \tabcolsep
        if "\\tabcolsep" not in block:
            block = re.sub(
                r'\\begin\{tabular\*?\}',
                r'\\setlength{\\tabcolsep}{4pt}\n\g<0>',
                block,
                count=1,
            )
        # Add @{} to column spec
        block = _fix_tabular_at_braces(block)
        # Auto-wrap wide tables with \resizebox
        block = _auto_resizebox(block)
        # Escape bare % in table cells
        block = _escape_percent_in_tables(block)
        # Truncate excessive numeric precision
        block = _truncate_precision(block)
        return block

    text = re.sub(
        r'\\begin\{table\*?\}.*?\\end\{table\*?\}',
        _patch_table,
        text,
        flags=re.DOTALL,
    )
    return text


def _auto_resizebox(block: str) -> str:
    r"""Wrap tabular in \resizebox if column count > 5 and not already wrapped."""
    if "\\resizebox" in block:
        return block

    # Count columns by finding the first data row (contains &) after \toprule or \midrule
    # and counting & separators.  This is more robust than parsing the column spec
    # which may contain nested braces like @{}.
    lines = block.split('\n')
    col_count = 0
    for line in lines:
        stripped = line.strip()
        if '&' in stripped and stripped.endswith('\\\\'):
            col_count = stripped.count('&') + 1
            break

    if col_count <= 5:
        return block

    # Wrap \begin{tabular}...\end{tabular} with \resizebox
    tabular_pat = re.compile(
        r'(\\begin\{tabular\*?\}\{[^}]+\}.*?\\end\{tabular\*?\})',
        re.DOTALL,
    )
    block = tabular_pat.sub(
        r'\\resizebox{\\textwidth}{!}{%\n\1\n}',
        block,
        count=1,
    )
    return block


def _truncate_precision(block: str) -> str:
    """Truncate numbers with >4 decimal places inside tables."""
    def _trunc(m: re.Match) -> str:
        full = m.group(0)
        integer = m.group(1)
        decimals = m.group(2)
        if len(decimals) > 4:
            return f"{integer}.{decimals[:4]}"
        return full

    # Match numbers like 0.12345678 but not inside commands like \citep{...}
    block = re.sub(r'(?<![{\\a-zA-Z])(\d+)\.(\d{5,})(?![}\w])', _trunc, block)
    return block


def _fix_tabular_at_braces(text: str) -> str:
    """Add @{} to tabular column spec if missing."""
    result = []
    i = 0
    tag = "\\begin{tabular}{"
    while i < len(text):
        pos = text.find(tag, i)
        if pos == -1:
            result.append(text[i:])
            break
        result.append(text[i:pos])
        brace_start = pos + len(tag) - 1
        depth = 0
        brace_end = brace_start
        for j in range(brace_start, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    brace_end = j
                    break
        if brace_end <= brace_start:
            result.append(tag)
            i = pos + len(tag)
            continue
        spec = text[brace_start + 1:brace_end]
        while "@{@{}}" in spec:
            spec = spec.replace("@{@{}}", "@{}")
        if not spec.startswith("@{}"):
            spec = "@{}" + spec
        if not spec.endswith("@{}"):
            spec = spec + "@{}"
        result.append(f"\\begin{{tabular}}{{{spec}}}")
        i = brace_end + 1
    return "".join(result)


def _collapse_math_blank_lines(text: str) -> str:
    """Collapse blank lines before/after math environments."""
    _math_envs = r'(?:equation|align|gather|multline|eqnarray)\*?'
    text = re.sub(
        rf'\n[ \t]*\n([ \t]*\\begin\{{{_math_envs}\}})',
        r'\n\1',
        text,
    )
    text = re.sub(
        rf'(\\end\{{{_math_envs}\}})[ \t]*\n[ \t]*\n',
        r'\1\n',
        text,
    )
    return text


def _fix_inline_table_start(text: str) -> str:
    r"""Fix \begin{table} embedded inside a paragraph (missing line break).

    LLMs sometimes generate text like:
        ...the baseline landscape in \begin{table}[t!]
    which causes LaTeX errors.  This fixer splits them apart.
    """
    # Match text immediately before \begin{table} on the same line
    text = re.sub(
        r'([^\n]{10,})\s*(\\begin\{table\*?\}\s*\[[^\]]*\])',
        r'\1\n\n\2',
        text,
    )
    return text


def _escape_percent_in_tables(block: str) -> str:
    r"""Escape bare % characters inside table cells.

    In LaTeX, % starts a comment.  Inside table cells this silently
    truncates the row, causing 'Misplaced \noalign' errors.
    """
    lines = block.split('\n')
    result = []
    in_tabular = False
    for line in lines:
        if '\\begin{tabular' in line:
            in_tabular = True
        if in_tabular and '&' in line:
            # Escape % that are not already escaped (not preceded by \)
            line = re.sub(r'(?<!\\)%', r'\\%', line)
        if '\\end{tabular' in line:
            in_tabular = False
        result.append(line)
    return '\n'.join(result)
