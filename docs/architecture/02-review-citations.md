# 3. P0: Multi-Model Review Committee

## Problem

Current REVIEW uses the same LLM (or family) that wrote the paper to review it.
This is fundamentally flawed — it's like having a student grade their own exam.

## Solution: Multi-Reviewer Architecture

**File**: `nanoresearch/agents/review.py` — modify `_review_paper()` method.

### 3.1 Config Changes (`config.py`)

Add to `ResearchConfig`:

```python
# In StageModelConfig, add reviewer profiles
review_committee: list[dict] = [
    # Default: use 2 different models
    # Each reviewer has a role, model config, and weight
]
```

Config example (`~/.nanobot/config.json`):

```json
{
  "research": {
    "review_committee": [
      {
        "role": "Methodology Expert",
        "focus": "technical soundness, mathematical rigor, proof correctness, novelty assessment",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "weight": 0.40
      },
      {
        "role": "Empirical Reviewer",
        "focus": "experiment design, statistical significance, reproducibility, baselines fairness",
        "model": "claude-sonnet-4-20250514",
        "base_url": "https://api.anthropic.com/v1",
        "weight": 0.35
      },
      {
        "role": "Writing Quality Reviewer",
        "focus": "clarity, logical flow, grammar, figure quality, related work coverage",
        "model": "gemini-2.5-pro",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "weight": 0.25
      }
    ]
  }
}
```

### 3.2 Implementation Changes (`review.py`)

```python
async def _multi_reviewer_assessment(self, paper_tex: str,
                                      sections: list[tuple[str, str]]) -> dict:
    """Run parallel reviews from multiple model personas."""
    committee = self.config.review_committee
    if not committee or len(committee) < 2:
        # Fallback to single-model review (existing behavior)
        return await self._review_paper(sections)

    import asyncio
    review_tasks = []
    for reviewer in committee:
        review_tasks.append(
            self._review_as_role(paper_tex, sections, reviewer)
        )
    reviews = await asyncio.gather(*review_tasks, return_exceptions=True)

    # Filter out failures
    valid_reviews = []
    weights = []
    for review, reviewer in zip(reviews, committee):
        if isinstance(review, Exception):
            self.log(f"Reviewer '{reviewer['role']}' failed: {review}")
            continue
        valid_reviews.append(review)
        weights.append(reviewer.get("weight", 1.0 / len(committee)))

    if not valid_reviews:
        self.log("All reviewers failed, falling back to single-model")
        return await self._review_paper(sections)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted score
    overall = sum(r["overall_score"] * w for r, w in zip(valid_reviews, weights))

    # Union of all issues (deduplicated by content similarity)
    all_issues = []
    seen_issues = set()
    for review in valid_reviews:
        for section_review in review.get("section_reviews", []):
            for issue in section_review.get("issues", []):
                issue_key = issue.strip().lower()[:80]
                if issue_key not in seen_issues:
                    seen_issues.add(issue_key)
                    all_issues.append(issue)

    # Per-section: take lowest score (most critical reviewer wins)
    merged_section_reviews = self._merge_section_reviews(valid_reviews)

    return {
        "overall_score": round(overall, 2),
        "section_reviews": merged_section_reviews,
        "individual_reviews": valid_reviews,  # Keep for transparency
        "num_reviewers": len(valid_reviews),
    }


async def _review_as_role(self, paper_tex: str,
                           sections: list[tuple[str, str]],
                           reviewer: dict) -> dict:
    """Run review from a specific reviewer persona."""
    role = reviewer["role"]
    focus = reviewer["focus"]

    # Create a temporary model config for this reviewer
    from .multi_model import ModelDispatcher
    reviewer_config = StageModelConfig(
        model=reviewer["model"],
        base_url=reviewer.get("base_url", self.config.base_url),
        api_key=reviewer.get("api_key", self.config.api_key),
        temperature=0.3,
        max_tokens=4096,
    )

    system = (
        f"You are a top-tier {role} at a major ML conference (NeurIPS/ICML/ICLR). "
        f"Your primary focus: {focus}. "
        f"Review the paper section by section. For each section, provide:\n"
        f"- score (1-10)\n"
        f"- issues (list of specific problems)\n"
        f"- suggestions (list of specific improvements)\n"
        f"Return JSON: {{\"overall_score\": float, \"section_reviews\": [...]}}"
    )

    # Use the reviewer's model for the LLM call
    result = await self.dispatcher.generate(
        config=reviewer_config,
        system_prompt=system,
        user_prompt=f"Review this paper:\n\n{paper_tex[:20000]}",
        json_mode=True,
    )

    return self._parse_review_json(result)


def _merge_section_reviews(self, reviews: list[dict]) -> list[dict]:
    """Merge section reviews from multiple reviewers.

    Strategy: for each section, take the LOWEST score (most critical reviewer
    wins) and union all issues/suggestions.
    """
    section_map: dict[str, dict] = {}
    for review in reviews:
        for sr in review.get("section_reviews", []):
            name = sr.get("section", "").lower().strip()
            if name not in section_map:
                section_map[name] = {
                    "section": sr.get("section", name),
                    "score": sr.get("score", 5),
                    "issues": list(sr.get("issues", [])),
                    "suggestions": list(sr.get("suggestions", [])),
                }
            else:
                existing = section_map[name]
                # Take minimum score (strictest reviewer)
                existing["score"] = min(existing["score"],
                                         sr.get("score", 5))
                # Union issues and suggestions (dedup by first 80 chars)
                seen = {i[:80].lower() for i in existing["issues"]}
                for issue in sr.get("issues", []):
                    if issue[:80].lower() not in seen:
                        existing["issues"].append(issue)
                        seen.add(issue[:80].lower())
                seen_s = {s[:80].lower() for s in existing["suggestions"]}
                for sug in sr.get("suggestions", []):
                    if sug[:80].lower() not in seen_s:
                        existing["suggestions"].append(sug)
                        seen_s.add(sug[:80].lower())
    return list(section_map.values())
```

### 3.3 Safety Constraints

- **Backward compatible**: If `review_committee` is empty or has < 2 entries, fall back to existing single-model review.
- **Graceful degradation**: If any reviewer fails (API error, timeout), continue with remaining reviewers.
- **No new dependencies**: Uses existing `ModelDispatcher` infrastructure.
- **Revision still uses original model**: Only the REVIEW scoring uses multiple models. The REVISION (rewriting) step stays on the configured revision model.

---

# 4. P0: Citation Fact-Checking

## Problem

The pipeline generates citations like "Smith et al. [15] achieved 95.2% accuracy on
ImageNet" — but never verifies if this matches what paper [15] actually says. This can
produce factual errors in published papers.

## Solution

Add `nanoresearch/agents/review_citation_checker.py`:

```python
"""Citation fact-checking: verify claims against source abstracts."""
import re
from typing import Optional


async def verify_citation_claims(
    agent,  # BaseResearchAgent instance (for LLM calls)
    paper_tex: str,
    papers: list[dict],
    bibtex_keys_to_papers: dict[str, dict],
) -> list[dict]:
    """Verify factual accuracy of citation claims in the paper.

    For each sentence containing a \\cite{}, compare the claim against the
    source paper's abstract/title.

    Args:
        agent: Agent instance for LLM calls.
        paper_tex: Full LaTeX source.
        papers: List of paper dicts from ideation.
        bibtex_keys_to_papers: Mapping from BibTeX key to paper dict.

    Returns:
        List of verification results.
    """
    # 1. Extract sentences with citations
    cite_sentences = _extract_cite_sentences(paper_tex)
    if not cite_sentences:
        return []

    # 2. Batch verify — cap at 15 checks to limit LLM cost.
    #    Group by cite_key to avoid re-verifying same source.
    checked_keys: set[str] = set()
    verifications = []
    MAX_CHECKS = 15
    for sentence, cite_keys in cite_sentences:
        if len(verifications) >= MAX_CHECKS:
            break
        for key in cite_keys:
            if key in checked_keys or len(verifications) >= MAX_CHECKS:
                continue
            checked_keys.add(key)
            paper = bibtex_keys_to_papers.get(key)
            if not paper:
                continue
            abstract = paper.get("abstract", "")
            title = paper.get("title", "")
            if not abstract and not title:
                continue

            # 3. LLM verification
            result = await agent.generate_json(
                system=(
                    "You are a citation fact-checker. Compare the claim in the "
                    "paper against the source's title and abstract. "
                    "Return JSON: {\"accurate\": true/false, "
                    "\"issue\": null or string describing the inaccuracy}"
                ),
                user=(
                    f"Claim in paper: \"{sentence}\"\n\n"
                    f"Source paper title: \"{title}\"\n"
                    f"Source paper abstract: \"{abstract[:1500]}\"\n\n"
                    f"Is the claim accurately representing the source?"
                ),
            )

            verifications.append({
                "sentence": sentence[:200],
                "cite_key": key,
                "source_title": title,
                "accurate": result.get("accurate", True),
                "issue": result.get("issue"),
            })

    return verifications


def _extract_cite_sentences(tex: str) -> list[tuple[str, list[str]]]:
    """Extract sentences containing \\cite commands.

    NOTE: This uses a rough sentence splitter that may not work perfectly
    inside LaTeX environments (e.g., \\caption{}, math mode). For better
    accuracy, consider stripping LaTeX commands before splitting.
    """
    # Split into sentences (rough: period + space + capital)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', tex)
    results = []
    for sent in sentences:
        cite_matches = re.findall(r'\\cite[tp]?\{([^}]+)\}', sent)
        if cite_matches:
            keys = []
            for match in cite_matches:
                keys.extend(k.strip() for k in match.split(","))
            results.append((sent.strip(), keys))
    return results
```

### Integration into `review.py`

In `run()`, after the main review and before revisions:

```python
# Citation fact-checking
from .review_citation_checker import verify_citation_claims
citation_verifications = await verify_citation_claims(
    self, paper_tex, papers, bibtex_key_map)
inaccurate = [v for v in citation_verifications if not v["accurate"]]
if inaccurate:
    self.log(f"Citation fact-check: {len(inaccurate)} inaccurate claims found")
    # Add to consistency_issues for the revision loop to fix
    for v in inaccurate:
        consistency_issues.append({
            "type": "citation_inaccuracy",
            "description": f"Claim about [{v['cite_key']}] may be inaccurate: {v['issue']}",
            "sentence": v["sentence"],
        })
```

> **IMPORTANT**: This adds issues to the existing consistency_issues list, which the
> existing revision loop already processes. No structural changes to the revision flow.
