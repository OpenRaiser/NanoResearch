"""Pytest fixtures for NanoResearch tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Workspace:
    """Create a temporary workspace for testing."""
    return Workspace.create(
        topic="Test topic: graph neural networks for protein folding",
        root=tmp_path,
        session_id="test_session_001",
    )


@pytest.fixture
def config() -> ResearchConfig:
    """Default research config for testing (with dummy credentials)."""
    return ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")


@pytest.fixture
def sample_ideation_output() -> dict:
    """Sample ideation output for downstream stage tests."""
    return {
        "topic": "Graph neural networks for protein folding",
        "search_queries": ["GNN protein folding", "graph network structure prediction"],
        "papers": [
            {
                "paper_id": "2401.00001",
                "title": "GNN-Fold: Graph Neural Network for Protein Structure Prediction",
                "authors": ["Alice Smith", "Bob Jones"],
                "year": 2024,
                "abstract": "We propose GNN-Fold...",
                "venue": "NeurIPS",
                "citation_count": 42,
                "url": "https://arxiv.org/abs/2401.00001",
                "bibtex": "",
                "relevance_score": 0.9,
            }
        ] * 5,
        "survey_summary": "The field of protein structure prediction has seen significant advances...",
        "gaps": [
            {
                "gap_id": "GAP-001",
                "description": "Limited exploration of equivariant message passing for side-chain prediction",
                "supporting_refs": ["2401.00001"],
                "severity": "high",
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "HYP-001",
                "statement": "Equivariant GNNs with side-chain attention improve folding accuracy",
                "gap_refs": ["GAP-001"],
                "novelty_justification": "No prior work combines equivariance with side-chain specific attention",
                "feasibility_notes": "Can build on existing SE(3)-Transformer codebase",
            }
        ],
        "selected_hypothesis": "HYP-001",
        "rationale": "Most promising due to high gap severity and clear implementation path",
        "evidence": {
            "extracted_metrics": [
                {
                    "paper_id": "2401.00001",
                    "paper_title": "GNN-Fold: Graph Neural Network for Protein Structure Prediction",
                    "dataset": "CASP14",
                    "metric_name": "GDT-TS",
                    "value": 92.4,
                    "unit": "",
                    "context": "achieves 92.4 GDT-TS on CASP14",
                    "method_name": "AlphaFold2",
                    "higher_is_better": True,
                }
            ],
            "extraction_notes": "Extracted 1 metric from test papers",
            "coverage_warnings": [],
        },
    }


@pytest.fixture
def sample_blueprint() -> dict:
    """Sample experiment blueprint for downstream stage tests."""
    return {
        "title": "Equivariant GNN with Side-Chain Attention for Protein Folding",
        "hypothesis_ref": "HYP-001",
        "datasets": [
            {
                "name": "CASP14",
                "description": "Critical Assessment of protein Structure Prediction 14",
                "source_url": "https://predictioncenter.org/casp14/",
                "size_info": "87 target proteins",
                "preprocessing_notes": "Standard preprocessing pipeline",
            }
        ],
        "baselines": [
            {
                "name": "AlphaFold2",
                "description": "State-of-the-art protein structure prediction",
                "reference_paper_id": "2401.00001",
                "expected_performance": {"GDT-TS": 92.4},
            }
        ],
        "proposed_method": {
            "name": "EquiFold",
            "description": "Equivariant GNN with side-chain attention",
            "key_components": ["SE(3)-equivariant layers", "side-chain attention"],
            "architecture": "Message-passing GNN with equivariant updates",
        },
        "metrics": [
            {"name": "GDT-TS", "description": "Global Distance Test", "higher_is_better": True, "primary": True},
            {"name": "RMSD", "description": "Root Mean Square Deviation", "higher_is_better": False, "primary": False},
        ],
        "ablation_groups": [
            {
                "group_name": "Component Ablation",
                "description": "Remove each key component",
                "variants": [
                    {"name": "no_equivariance", "description": "Remove SE(3) equivariance"},
                    {"name": "no_side_chain_attn", "description": "Remove side-chain attention"},
                ],
            }
        ],
        "compute_requirements": {"gpu_type": "A100", "num_gpus": 4, "estimated_hours": 48},
    }
