"""BibTeX fixes: brace-depth splitting, dedup, escaping, Unicode."""
from __future__ import annotations

import html as _html
import logging
import re

logger = logging.getLogger(__name__)

_URL_FIELDS = {"url", "doi", "eprint", "howpublished", "note"}


def fix_bibtex(bib: str) -> tuple[str, list[str]]:
    """Fix BibTeX content. Returns (fixed_bib, fix_descriptions)."""
    if not bib.strip():
        return bib, []

    fixes: list[str] = []
    original = bib

    # Phase 0: repair structurally broken entries (missing closing braces, fields)
    bib, repair_fixes = _repair_incomplete_entries(bib)
    fixes.extend(repair_fixes)

    bib = _sanitize_bibtex(bib)

    if bib != original:
        fixes.append("bibtex: dedup, escaping, and Unicode fixes applied")

    return bib, fixes


def split_bibtex_entries(bib: str) -> list[str]:
    """Split BibTeX string into individual entries using brace-depth counting."""
    entries: list[str] = []
    i = 0
    entry_start_re = re.compile(r'@\w+\s*\{')
    while i < len(bib):
        m = entry_start_re.search(bib, i)
        if not m:
            break
        start = m.start()
        open_brace = m.end() - 1
        depth = 1
        j = open_brace + 1
        while j < len(bib) and depth > 0:
            if bib[j] == '\\' and j + 1 < len(bib) and bib[j + 1] in '{}':
                j += 2
                continue
            if bib[j] == '{':
                depth += 1
            elif bib[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            entries.append(bib[start:j].strip())
        i = j
    return entries


def _sanitize_bibtex(bib: str) -> str:
    """Fix common issues in BibTeX entries."""
    # Deduplicate entries by key
    key_pattern = re.compile(r'@\w+\s*\{\s*([^,\s]+)\s*,')
    entries = split_bibtex_entries(bib)
    if entries:
        seen: set[str] = set()
        deduped: list[str] = []
        for entry in entries:
            key_match = key_pattern.match(entry.strip())
            if key_match:
                bib_key = key_match.group(1).strip()
                if bib_key in seen:
                    continue
                seen.add(bib_key)
            deduped.append(entry.strip())
        bib = "\n\n".join(deduped) + "\n"

    # HTML entity decoding
    bib = _html.unescape(bib)

    # Escape & in text fields
    bib = _escape_in_text_fields(bib, r'(?<!\\)&', r'\\&')

    # Escape # and % in text fields
    bib = _escape_in_text_fields(bib, r'(?<!\\)#', r'\\#')
    bib = _escape_in_text_fields(bib, r'(?<!\\)%', r'\\%')

    # Unicode → LaTeX
    unicode_map = {
        "\u00e9": r"{\'e}", "\u00e8": r"{\`e}", "\u00eb": r'{\"e}',
        "\u00fc": r'{\"u}', "\u00f6": r'{\"o}', "\u00e4": r'{\"a}',
        "\u00df": r"{\ss}", "\u00e7": r"{\c{c}}", "\u00c7": r"{\c{C}}",
        "\u00f1": r"{\~n}", "\u011f": r"{\u{g}}", "\u0131": r"{\i}",
        "\u015f": r"{\c{s}}", "\u0151": r"{\H{o}}", "\u0171": r"{\H{u}}",
        "\u017e": r"{\v{z}}", "\u0161": r"{\v{s}}", "\u0107": r"{\'c}",
        "\u2014": "---", "\u2013": "--",
    }
    for char, repl in unicode_map.items():
        bib = bib.replace(char, repl)

    # Fix bare underscores in title fields
    def _fix_title_underscores(m: re.Match) -> str:
        key = m.group(1)
        val = m.group(2)
        val = re.sub(r'(?<!\\)_', r'\\_', val)
        return f'{key} = {{{val}}}'

    bib = re.sub(
        r'((?:book)?title)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
        _fix_title_underscores,
        bib,
        flags=re.IGNORECASE,
    )

    # Detect entry type (article vs inproceedings)
    bib = _detect_bib_entry_type(bib)

    return bib


def _escape_in_text_fields(bib: str, pattern: str, replacement: str) -> str:
    """Escape a pattern only in non-URL BibTeX fields."""
    def _field_repl(fm: re.Match) -> str:
        field_name = fm.group(1).strip().lower()
        if field_name in _URL_FIELDS:
            return fm.group(0)
        field_body = fm.group(2)
        escaped = re.sub(pattern, replacement, field_body)
        return fm.group(0).replace(field_body, escaped)

    return re.sub(
        r'(\b\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*")',
        _field_repl, bib,
    )


_CONFERENCE_KEYWORDS = frozenset({
    "conference", "proceedings", "proc.", "workshop", "symposium",
    "icml", "neurips", "nips", "iclr", "cvpr", "iccv", "eccv",
    "aaai", "ijcai", "acl", "emnlp", "naacl", "coling",
    "sigir", "kdd", "www", "chi", "uist", "siggraph",
    "interspeech", "icassp",
})


def _detect_bib_entry_type(bib: str) -> str:
    """Fix @article that should be @inproceedings based on booktitle keywords."""
    def _fix_entry(m: re.Match) -> str:
        entry = m.group(0)
        if not entry.lower().startswith("@article"):
            return entry
        # Check if booktitle field exists (indicates conference paper)
        if re.search(r'booktitle\s*=', entry, re.IGNORECASE):
            bt_m = re.search(
                r'booktitle\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
                entry, re.IGNORECASE,
            )
            if bt_m:
                bt_val = bt_m.group(1).lower()
                if any(kw in bt_val for kw in _CONFERENCE_KEYWORDS):
                    return re.sub(r'^@article', '@inproceedings', entry, flags=re.IGNORECASE)
        return entry

    return re.sub(r'@\w+\s*\{[^@]*', _fix_entry, bib)


# ── Structural repair for broken entries ─────────────────────────────────────

def _repair_incomplete_entries(bib: str) -> tuple[str, list[str]]:
    """Repair BibTeX entries with missing closing braces and required fields.

    NanoResearch sometimes produces truncated bib files where entries have only
    a title field, no author/year/journal, and no closing ``}``. The standard
    brace-depth splitter silently drops these entries. This function detects
    and repairs them *before* the main sanitizer runs.
    """
    fixes: list[str] = []

    # Split on entry boundaries (lookahead for @type{)
    raw_blocks = re.split(r'(?=@\w+\s*\{)', bib.strip())
    raw_blocks = [b for b in raw_blocks if b.strip()]

    if not raw_blocks:
        return bib, fixes

    # Quick check: if all entries already have balanced braces, skip repair
    needs_repair = False
    for block in raw_blocks:
        if not block.strip().startswith('@'):
            continue
        depth = sum(1 if c == '{' else (-1 if c == '}' else 0) for c in block)
        if depth > 0:
            needs_repair = True
            break
        # Check for missing author field
        if block.strip().startswith('@') and not re.search(r'\bauthor\s*=', block, re.IGNORECASE):
            needs_repair = True
            break

    if not needs_repair:
        return bib, fixes

    repaired: list[str] = []
    repair_count = 0

    for block in raw_blocks:
        block = block.strip()
        if not block.startswith('@'):
            continue

        # Extract entry type and key
        header_m = re.match(r'@(\w+)\s*\{\s*([^,\s]+)\s*,?', block)
        if not header_m:
            repaired.append(block)
            continue

        entry_type = header_m.group(1).lower()
        cite_key = header_m.group(2)

        # Check brace balance
        depth = sum(1 if c == '{' else (-1 if c == '}' else 0) for c in block)

        if depth > 0:
            # Missing closing brace(s)
            block = block.rstrip()
            if not block.rstrip().endswith(','):
                # Ensure last field value is closed
                # Find if there's an unclosed { in the last field
                last_eq = block.rfind('=')
                if last_eq >= 0:
                    after_eq = block[last_eq:]
                    open_b = after_eq.count('{')
                    close_b = after_eq.count('}')
                    if open_b > close_b:
                        block += '}' * (open_b - close_b)
                block += ','
            block += '\n}'
            repair_count += 1

        block_lower = block.lower()

        # Check for missing author field
        if not re.search(r'\bauthor\s*=', block_lower):
            # Insert after title field
            title_m = re.search(
                r'(title\s*=\s*\{(?:[^{}]|\{[^{}]*\})*\},?)',
                block, re.IGNORECASE,
            )
            if title_m:
                insert_pos = title_m.end()
                block = block[:insert_pos] + '\n  author = {Unknown},' + block[insert_pos:]
                repair_count += 1

        # Check for missing year field
        if not re.search(r'\byear\s*=', block.lower()):
            year_m = re.search(r'(\d{4})', cite_key)
            year_val = year_m.group(1) if year_m else "n.d."
            # Insert after author field
            author_m = re.search(
                r'(author\s*=\s*\{[^}]*\},?)',
                block, re.IGNORECASE,
            )
            if author_m:
                insert_pos = author_m.end()
                block = block[:insert_pos] + f'\n  year = {{{year_val}}},' + block[insert_pos:]
                repair_count += 1

        # Check for missing journal/booktitle
        if entry_type == 'article' and not re.search(r'\bjournal\s*=', block.lower()):
            year_m2 = re.search(
                r'(year\s*=\s*\{[^}]*\},?)',
                block, re.IGNORECASE,
            )
            if year_m2:
                insert_pos = year_m2.end()
                block = block[:insert_pos] + '\n  journal = {arXiv preprint},' + block[insert_pos:]
                repair_count += 1
        elif entry_type == 'inproceedings' and not re.search(r'\bbooktitle\s*=', block.lower()):
            year_m2 = re.search(
                r'(year\s*=\s*\{[^}]*\},?)',
                block, re.IGNORECASE,
            )
            if year_m2:
                insert_pos = year_m2.end()
                block = block[:insert_pos] + '\n  booktitle = {Proceedings},' + block[insert_pos:]
                repair_count += 1

        repaired.append(block)

    if repair_count > 0:
        fixes.append(f"bibtex: repaired {repair_count} structural issues "
                      f"(missing braces/fields) in {len(repaired)} entries")
        logger.info("BibTeX structural repair: %d fixes across %d entries",
                     repair_count, len(repaired))

    return '\n\n'.join(repaired) + '\n', fixes
