"""PDF full-text extraction tool using PyMuPDF."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from mcp_server.utils import get_http_client

logger = logging.getLogger(__name__)

# Common section heading patterns in academic papers
_SECTION_PATTERNS = [
    re.compile(
        r"^\s*(?:\d+\.?\s+)?(Introduction|Related\s+Work|Background|"
        r"Method(?:s|ology)?|Approach|Experiment(?:s|al\s+(?:Setup|Results))?|"
        r"Results?(?:\s+and\s+Discussion)?|Discussion|Conclusion(?:s)?|"
        r"Abstract|Acknowledgment(?:s)?|References|Appendix)",
        re.IGNORECASE | re.MULTILINE,
    ),
]


async def download_and_extract(
    pdf_url: str, max_pages: int = 30
) -> dict[str, Any]:
    """Download a PDF from URL and extract its full text.

    Args:
        pdf_url: URL to the PDF file.
        max_pages: Maximum pages to process.

    Returns:
        Dict with keys: full_text, sections, method_text, experiment_text, page_count.
    """
    try:
        async with get_http_client(timeout=60.0) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
            pdf_bytes = resp.content
    except httpx.TimeoutException:
        logger.warning("PDF download timed out for: %s", pdf_url[:200])
        return {"full_text": "", "sections": {}, "method_text": "", "experiment_text": "", "page_count": 0}
    except httpx.HTTPStatusError as exc:
        logger.warning("PDF download HTTP %d for: %s", exc.response.status_code, pdf_url[:200])
        return {"full_text": "", "sections": {}, "method_text": "", "experiment_text": "", "page_count": 0}
    except httpx.HTTPError as exc:
        logger.warning("PDF download network error: %s", exc)
        return {"full_text": "", "sections": {}, "method_text": "", "experiment_text": "", "page_count": 0}

    return extract_text_from_bytes(pdf_bytes, max_pages)


def extract_text_from_bytes(
    pdf_bytes: bytes, max_pages: int = 30
) -> dict[str, Any]:
    """Extract structured text from PDF bytes.

    Args:
        pdf_bytes: Raw PDF file content.
        max_pages: Maximum pages to process.

    Returns:
        Dict with keys: full_text, sections, method_text, experiment_text, page_count.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed; returning empty extraction")
        return {
            "full_text": "",
            "sections": {},
            "method_text": "",
            "experiment_text": "",
            "page_count": 0,
        }

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.warning("Failed to open PDF: %s", e)
        return {
            "full_text": "",
            "sections": {},
            "method_text": "",
            "experiment_text": "",
            "page_count": 0,
        }

    try:
        page_count = min(len(doc), max_pages)
        pages_text: list[str] = []
        for page_num in range(page_count):
            page = doc[page_num]
            pages_text.append(page.get_text())
    finally:
        doc.close()

    full_text = "\n".join(pages_text)
    sections = _split_sections(full_text)

    method_text = ""
    experiment_text = ""
    for name, content in sections.items():
        name_lower = name.lower()
        if "method" in name_lower or "approach" in name_lower:
            method_text = content
        elif "experiment" in name_lower or "result" in name_lower:
            experiment_text = content

    return {
        "full_text": full_text,
        "sections": sections,
        "method_text": method_text,
        "experiment_text": experiment_text,
        "page_count": page_count,
    }


def _split_sections(text: str) -> dict[str, str]:
    """Split full text into sections based on heading patterns."""
    sections: dict[str, str] = {}
    current_heading = "Preamble"
    current_lines: list[str] = []

    for line in text.split("\n"):
        matched = False
        for pattern in _SECTION_PATTERNS:
            m = pattern.match(line.strip())
            if m:
                # Save previous section
                if current_lines:
                    sections[current_heading] = "\n".join(current_lines).strip()
                current_heading = m.group(1).strip()
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    # Save last section
    if current_lines:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections
