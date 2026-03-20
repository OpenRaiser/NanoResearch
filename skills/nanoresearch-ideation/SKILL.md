---
name: nanoresearch-ideation
description: Search academic literature and generate research hypotheses
version: 0.1.0
---

# Ideation Skill

## Purpose
Search arXiv and Semantic Scholar for papers related to a research topic, perform gap analysis, and generate novel hypotheses.

## Tools Required
- `search_arxiv`: Search arXiv for papers
- `search_semantic_scholar`: Search Semantic Scholar for papers and citations

## Input
- `topic`: The research topic or question to investigate

## Process
1. Generate 5-8 diverse search queries from the topic
2. Search arXiv and Semantic Scholar using each query
3. Deduplicate and rank papers by relevance
4. Analyze the collected papers to identify research gaps
5. Generate 2-4 novel hypotheses that address the identified gaps
6. Select the most promising hypothesis with justification

## Output
Produces `papers/ideation_output.json` containing:
- Retrieved papers with metadata
- Survey summary
- Gap analysis
- Generated hypotheses
- Selected hypothesis with rationale
