"""Shared utilities for cleaning LLM-generated code output."""

from __future__ import annotations


def _strip_code_fences(content: str) -> str:
    """Robustly strip markdown code fences from LLM-generated code.

    Handles edge cases that simple first/last fence stripping misses:
    - LLM self-correction mid-output ("Wait, I need to also write...")
    - Multiple code blocks in a single response
    - Stray ``` markers embedded in the middle of code

    Strategy: if multiple fenced code blocks exist, pick the longest
    complete block (most likely the full file).  If no fenced blocks are
    detected, just strip any stray ``` lines.
    """
    content = (content or "").strip()
    if not content:
        return content

    # --- Phase 1: extract code blocks delimited by ```...``` ---
    blocks: list[str] = []
    lines = content.split("\n")
    inside = False
    block_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not inside and stripped.startswith("```"):
            inside = True
            block_lines = []
            continue
        if inside and stripped == "```":
            inside = False
            blocks.append("\n".join(block_lines))
            continue
        if inside:
            block_lines.append(line)

    if blocks:
        # Prefer the longest block (most likely the complete file)
        best = max(blocks, key=len)
        return best.strip()

    # --- Phase 2: no matched pairs — handle single opening fence ---
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
        # Also remove trailing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            return cleaned

    # --- Phase 3: remove any remaining stray ``` lines ---
    cleaned_lines = [l for l in lines if l.strip() != "```"]
    return "\n".join(cleaned_lines).strip()
