"""Environment fixes: abstract double-wrapping, mismatched \\begin/\\end."""
from __future__ import annotations

import re


def fix_environments(text: str) -> str:
    """Fix environment-related issues."""
    text = _collapse_double_abstract(text)
    text = _fix_mismatched_environments(text)
    return text


def _collapse_double_abstract(text: str) -> str:
    """Collapse double-nested abstract environments."""
    text = re.sub(
        r'\\begin\{abstract\}\s*\\begin\{abstract\}',
        r'\\begin{abstract}',
        text,
    )
    text = re.sub(
        r'\\end\{abstract\}\s*\\end\{abstract\}',
        r'\\end{abstract}',
        text,
    )
    return text


def _fix_mismatched_environments(text: str) -> str:
    r"""Fix mismatched \begin{X}...\end{Y} pairs.

    Scans for \begin and \end tags, tracks a stack, and fixes
    mismatches by replacing \end{wrong} with \end{correct}.
    """
    # Collect all \begin and \end positions
    begin_pat = re.compile(r'\\begin\{([^}]+)\}')
    end_pat = re.compile(r'\\end\{([^}]+)\}')

    # Simple stack-based approach: find unmatched ends and fix them
    stack: list[tuple[str, int]] = []  # (env_name, position)
    replacements: list[tuple[int, int, str]] = []  # (start, end, replacement)

    for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', text):
        cmd = m.group(1)
        env = m.group(2)

        if cmd == "begin":
            stack.append((env, m.start()))
        elif cmd == "end":
            if stack and stack[-1][0] == env:
                stack.pop()
            elif stack:
                # Mismatch: \end{env} doesn't match top of stack
                expected = stack[-1][0]
                # Replace this \end with the expected one
                replacements.append((
                    m.start(),
                    m.end(),
                    f"\\end{{{expected}}}",
                ))
                stack.pop()
            # else: extra \end with empty stack — leave it

    # Apply replacements in reverse order
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]

    return text
