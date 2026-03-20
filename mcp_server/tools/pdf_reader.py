"""PDF full-text extraction tool using PyMuPDF."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from mcp_server.utils import get_http_client

logger = logging.getLogger(__name__)

# Common section heading patterns in academic papers
# Broad enough to catch variants like "Model Architecture", "Proposed Framework", etc.
_SECTION_PATTERNS = [
    re.compile(
        r"^\s*(?:\d+\.?\s+)?(Introduction|Related\s+Work|Background|Preliminaries|"
        r"Problem\s+(?:Definition|Formulation|Statement|Setup)|"
        r"Method(?:s|ology)?|Approach|(?:Proposed\s+)?(?:Model|Framework|System|Architecture)|"
        r"(?:Our\s+)?(?:Method|Approach|Framework|Model)|Technical\s+Approach|"
        r"Experiment(?:s|al)?(?:\s+(?:Setup|Results|Settings|Details))?|"
        r"(?:Main\s+)?Results?(?:\s+and\s+(?:Discussion|Analysis))?|"
        r"Evaluation|Analysis|Ablation(?:\s+Study)?|"
        r"Discussion|Conclusion(?:s)?|Limitations?|Broader\s+Impact|"
        r"Abstract|Acknowledgment(?:s)?|References|Appendix|"
        r"Implementation\s+Details|Training\s+Details|Datasets?)",
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
    _MAX_PDF_SIZE = 50 * 1024 * 1024  # 50 MB
    try:
        async with get_http_client(timeout=60.0) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > _MAX_PDF_SIZE:
                logger.warning("PDF too large (%s bytes): %s", content_length, pdf_url[:200])
                return {"full_text": "", "sections": {}, "method_text": "", "experiment_text": "", "page_count": 0}
            pdf_bytes = resp.content
            if len(pdf_bytes) > _MAX_PDF_SIZE:
                logger.warning("PDF body too large (%d bytes): %s", len(pdf_bytes), pdf_url[:200])
                return {"full_text": "", "sections": {}, "method_text": "", "experiment_text": "", "page_count": 0}
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
            try:
                pages_text.append(page.get_text())
            except Exception as e:
                logger.warning("Failed to extract text from page %d: %s", page_num + 1, e)
                pages_text.append("")
    finally:
        doc.close()

    full_text = "\n".join(pages_text)
    sections = _split_sections(full_text)

    method_text = ""
    experiment_text = ""
    for name, content in sections.items():
        name_lower = name.lower()
        if any(kw in name_lower for kw in (
            "method", "approach", "framework", "architecture", "model",
            "technical", "proposed", "our method",
        )):
            method_text += ("\n\n" + content if method_text else content)
        elif any(kw in name_lower for kw in (
            "experiment", "result", "evaluation", "ablation",
            "implementation", "training detail",
        )):
            experiment_text += ("\n\n" + content if experiment_text else content)

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
