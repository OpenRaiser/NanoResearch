"""Citation fact-checking: verify claims against source paper abstracts."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Max citation checks per review (limits LLM cost)
MAX_CITATION_CHECKS = 15


async def verify_citation_claims(
    agent,  # BaseResearchAgent instance (for LLM calls)
    paper_tex: str,
    bibtex_keys_to_papers: dict[str, dict],
) -> list[dict]:
    """Verify factual accuracy of citation claims in the paper.

    For each sentence containing a \\cite{}, compare the claim against the
    source paper's abstract/title.

    Args:
        agent: Agent instance for LLM calls (uses generate_json).
        paper_tex: Full LaTeX source.
        bibtex_keys_to_papers: Mapping from BibTeX key to paper dict
            (must have 'title' and/or 'abstract' keys).

    Returns:
        List of verification dicts with keys:
        - sentence, cite_key, source_title, accurate (bool), issue (str|None)
    """
    cite_sentences = _extract_cite_sentences(paper_tex)
    if not cite_sentences:
        return []

    checked_keys: set[str] = set()
    verifications: list[dict] = []

    for sentence, cite_keys in cite_sentences:
        if len(verifications) >= MAX_CITATION_CHECKS:
            break
        for key in cite_keys:
            if key in checked_keys or len(verifications) >= MAX_CITATION_CHECKS:
                continue
            checked_keys.add(key)

            paper = bibtex_keys_to_papers.get(key)
            if not paper or not isinstance(paper, dict):
                continue
            abstract = paper.get("abstract", "") or ""
            title = paper.get("title", "") or ""
            if not abstract and not title:
                continue

            try:
                result = await agent.generate_json(
                    system=(
                        "You are a citation fact-checker. Compare the claim "
                        "in the paper against the source's title and abstract. "
                        'Return JSON: {"accurate": true/false, '
                        '"issue": null or string describing the inaccuracy}'
                    ),
                    user=(
                        f'Claim in paper: "{sentence[:500]}"\n\n'
                        f'Source paper title: "{title}"\n'
                        f'Source paper abstract: "{abstract[:1500]}"\n\n'
                        "Is the claim accurately representing the source?"
                    ),
                )
            except Exception as exc:
                logger.warning("Citation check failed for %s: %s", key, exc)
                continue

            if not isinstance(result, dict):
                continue

            verifications.append({
                "sentence": sentence[:200],
                "cite_key": key,
                "source_title": title[:200],
                "accurate": bool(result.get("accurate", True)),
                "issue": result.get("issue"),
            })

    return verifications


def _extract_cite_sentences(tex: str) -> list[tuple[str, list[str]]]:
    r"""Extract sentences containing \cite commands.

    Returns list of (sentence_text, [cite_key1, cite_key2, ...]).
    Only returns sentences from the document body (after \begin{document}).
    """
    # Only search in document body
    body_match = re.search(r'\\begin\{document\}', tex)
    if body_match:
        tex = tex[body_match.end():]
    end_match = re.search(r'\\end\{document\}', tex)
    if end_match:
        tex = tex[:end_match.start()]

    # Split into rough sentences (period + space + capital or backslash)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', tex)

    results: list[tuple[str, list[str]]] = []
    for sent in sentences:
        # Match \cite{}, \citet{}, \citep{}, \citeauthor{}, \citeyear{}, etc.
        cite_matches = re.findall(r'\\cite[a-z]*\{([^}]+)\}', sent)
        if not cite_matches:
            continue
        keys: list[str] = []
        for match in cite_matches:
            keys.extend(k.strip() for k in match.split(",") if k.strip())
        if keys:
            # Clean up LaTeX noise from the sentence for readability
            clean = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', sent)
            clean = clean.strip()
            if len(clean) > 20:  # skip tiny fragments
                results.append((clean, keys))

    return results
