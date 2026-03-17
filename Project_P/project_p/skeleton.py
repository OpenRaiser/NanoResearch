"""Document skeleton parser — Edit-mode infrastructure for Project_P.

Parses a LaTeX document into a structured skeleton of sections, subsections,
paragraphs with their exact character positions. Enables precise, position-based
editing (insert/move/delete at exact locations) instead of regex-based text scanning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TexNode:
    """A node in the document skeleton tree."""
    level: int          # 0=section, 1=subsection, 2=subsubsection, 3=paragraph
    title: str          # heading text (e.g., "Introduction")
    label: str          # \label{...} if present, else ""
    start: int          # char position of \section{...} command start
    heading_end: int    # char position right after the heading + optional label
    end: int            # char position of the end of this section's content
    children: list[TexNode] = field(default_factory=list)


@dataclass
class FloatBlock:
    """A figure or table environment."""
    env_type: str       # "figure", "figure*", "table", "table*"
    start: int          # char position of \begin{...}
    end: int            # char position right after \end{...}
    label: str          # \label{...} if present
    caption: str        # caption text
    image_file: str     # \includegraphics filename (figures only)

    @property
    def text(self) -> str:
        """Not stored — retrieve from document."""
        return ""


@dataclass
class DocumentSkeleton:
    """Complete parsed structure of a LaTeX document."""
    preamble_end: int           # position of \begin{document}
    body_start: int             # position right after \begin{document}
    body_end: int               # position of \end{document}
    bib_start: int              # position of \bibliographystyle or \bibliography
    sections: list[TexNode]     # top-level sections
    floats: list[FloatBlock]    # all figure/table blocks

    def find_section(self, title_hint: str) -> TexNode | None:
        """Find a section by keyword match on title."""
        hint = title_hint.lower()
        stem = hint.rstrip('s')
        for sec in self.sections:
            t = sec.title.lower()
            if t == hint or stem in t or hint in t:
                return sec
        return None

    def find_insert_point_after_ref(
        self, content: str, label: str, *, avoid_floats: bool = True,
    ) -> int | None:
        r"""Find the best insertion point after the first \ref{label} in content.

        Returns the position at the end of the paragraph containing the ref,
        or None if no ref is found.
        """
        ref_pat = re.compile(
            rf'\\(?:ref|autoref|cref)\{{(?:fig:)?{re.escape(label)}\}}',
            re.IGNORECASE,
        )
        m = ref_pat.search(content)
        if not m:
            return None

        # Find end of paragraph (blank line or next heading/float)
        after = m.end()
        para_end = re.search(
            r'\n\s*\n|\\(?:sub){0,2}section\{|\\paragraph\{'
            r'|\\begin\{(?:table|figure)\*?\}',
            content[after:],
        )
        if para_end:
            pos = after + para_end.start()
        else:
            pos = len(content)

        # Don't insert past bibliography
        if pos > self.bib_start:
            pos = self.bib_start

        return pos

    def section_end_pos(self, section: TexNode) -> int:
        """Get the clean insertion point at the end of a section's content.

        Backs up past trailing whitespace to get a tight position.
        """
        pos = section.end
        return pos


_LEVEL_COMMANDS = {
    "section": 0, "section*": 0,
    "subsection": 1, "subsection*": 1,
    "subsubsection": 2, "subsubsection*": 2,
    "paragraph": 3, "paragraph*": 3,
}

# Pattern matches \section{...}, \subsection*{...}, etc. with nested braces
_HEADING_RE = re.compile(
    r'\\((?:sub){0,2}section\*?|paragraph\*?)'
    r'\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}',
)

_LABEL_AFTER_HEADING = re.compile(r'\s*\\label\{([^}]+)\}')

_FLOAT_RE = re.compile(
    r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}',
    re.DOTALL,
)


def parse_skeleton(tex: str) -> DocumentSkeleton:
    """Parse a LaTeX document into a DocumentSkeleton."""
    # Find document boundaries
    begin_doc = tex.find(r'\begin{document}')
    preamble_end = begin_doc if begin_doc >= 0 else 0
    body_start = begin_doc + len(r'\begin{document}') if begin_doc >= 0 else 0

    end_doc = tex.rfind(r'\end{document}')
    body_end = end_doc if end_doc >= 0 else len(tex)

    # Find bibliography start
    bib_start = len(tex)
    for pat in (r'\bibliographystyle{', r'\bibliography{', r'\begin{thebibliography}'):
        pos = tex.find(pat)
        if pos >= 0 and pos < bib_start:
            bib_start = pos

    # Parse headings
    headings: list[tuple[int, int, str, int, str]] = []  # (start, heading_end, title, level, label)
    for m in _HEADING_RE.finditer(tex, body_start, body_end):
        cmd = m.group(1)
        title = m.group(2)
        level = _LEVEL_COMMANDS.get(cmd, 0)
        heading_end = m.end()

        # Check for \label right after heading
        label = ""
        label_m = _LABEL_AFTER_HEADING.match(tex, heading_end)
        if label_m:
            label = label_m.group(1)
            heading_end = label_m.end()

        headings.append((m.start(), heading_end, title, level, label))

    # Build tree: assign end positions and parent-child relationships
    sections: list[TexNode] = []
    all_nodes: list[TexNode] = []

    for i, (start, heading_end, title, level, label) in enumerate(headings):
        # End position: start of next heading at same or higher level, or bib/end
        end = bib_start
        for j in range(i + 1, len(headings)):
            if headings[j][3] <= level:
                end = headings[j][0]
                break

        node = TexNode(
            level=level, title=title, label=label,
            start=start, heading_end=heading_end, end=end,
        )
        all_nodes.append(node)

    # Assign children
    for i, node in enumerate(all_nodes):
        for j in range(i + 1, len(all_nodes)):
            child = all_nodes[j]
            if child.start >= node.end:
                break
            if child.level == node.level + 1:
                node.children.append(child)

    # Top-level sections
    sections = [n for n in all_nodes if n.level == 0]

    # Parse float blocks
    floats: list[FloatBlock] = []
    for m in _FLOAT_RE.finditer(tex, body_start, body_end):
        block_text = m.group(0)
        env_type = m.group(1)

        label_m = re.search(r'\\label\{([^}]+)\}', block_text)
        label = label_m.group(1) if label_m else ""

        cap_m = re.search(
            r'\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}',
            block_text,
        )
        caption = cap_m.group(1) if cap_m else ""

        img_m = re.search(
            r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',
            block_text,
        )
        image_file = img_m.group(1) if img_m else ""

        floats.append(FloatBlock(
            env_type=env_type, start=m.start(), end=m.end(),
            label=label, caption=caption, image_file=image_file,
        ))

    return DocumentSkeleton(
        preamble_end=preamble_end,
        body_start=body_start,
        body_end=body_end,
        bib_start=bib_start,
        sections=sections,
        floats=floats,
    )


def insert_at(tex: str, pos: int, content: str) -> str:
    """Insert content at exact position with surrounding blank lines."""
    # Ensure clean separation
    before = tex[:pos].rstrip()
    after = tex[pos:].lstrip()
    return before + "\n\n" + content.strip() + "\n\n" + after


def remove_block(tex: str, start: int, end: int) -> str:
    """Remove a block from tex and clean up surrounding whitespace."""
    while end < len(tex) and tex[end] in ('\n', '\r', ' ', '\t'):
        end += 1
    return tex[:start] + tex[end:]
