"""Tests for writing agent grounding: GroundingPacket, table builders, completeness."""

from __future__ import annotations

from nanoresearch.agents.writing import GroundingPacket, WritingAgent


# ---- completeness classification ----

def test_classify_completeness_full():
    result = WritingAgent._classify_completeness(
        "COMPLETED",
        [{"method_name": "M", "metrics": [{"metric_name": "acc", "value": 0.9}]}],
        {"converged": True},
    )
    assert result == "full"


def test_classify_completeness_none_no_results():
    assert WritingAgent._classify_completeness("COMPLETED", [], {}) == "none"


def test_classify_completeness_none_failed():
    assert WritingAgent._classify_completeness(
        "FAILED",
        [{"method_name": "M", "metrics": []}],
        {},
    ) == "none"


def test_classify_completeness_quick_eval():
    result = WritingAgent._classify_completeness(
        "QUICK_EVAL_COMPLETED",
        [{"method_name": "M", "metrics": [{"metric_name": "acc", "value": 0.8}]}],
        {"summary": "Quick-eval finished in 30s"},
    )
    assert result == "quick_eval"


def test_classify_completeness_partial():
    result = WritingAgent._classify_completeness(
        "COMPLETED",
        [{"method_name": "M", "metrics": [{"metric_name": "acc", "value": 0.5}]}],
        {"converged": False},
    )
    assert result == "partial"


# ---- grounding packet ----

def _make_packet(**overrides) -> GroundingPacket:
    defaults = dict(
        experiment_results={"accuracy": 0.93, "loss": 0.12},
        experiment_status="COMPLETED",
        experiment_analysis={
            "summary": "Model converged well.",
            "final_metrics": {"accuracy": 0.93, "loss": 0.12},
            "converged": True,
            "comparison_with_baselines": {
                "our_method": {"accuracy": 0.93},
                "BaselineA": {"accuracy": 0.88},
            },
            "ablation_results": [
                {"variant_name": "Full model", "metrics": [{"metric_name": "accuracy", "value": 0.93}]},
                {"variant_name": "w/o Attn", "metrics": [{"metric_name": "accuracy", "value": 0.87}]},
            ],
            "key_findings": ["Attention is critical"],
            "limitations": ["Only tested on one dataset"],
        },
        experiment_summary="# Summary\nModel converged.",
        blueprint={
            "proposed_method": {"name": "DeepMethod", "key_components": ["Fusion"]},
            "datasets": [{"name": "DemoSet"}],
            "metrics": [{"name": "accuracy"}],
            "baselines": [{"name": "BaselineA"}],
            "ablation_groups": [{"group_name": "Fusion"}],
        },
    )
    defaults.update(overrides)
    return WritingAgent._build_grounding_packet(**defaults)


def test_grounding_packet_full_results():
    pkt = _make_packet()
    assert pkt.result_completeness == "full"
    assert pkt.has_real_results is True
    assert len(pkt.main_results) >= 1
    assert pkt.comparison_with_baselines
    assert "BaselineA" in pkt.comparison_with_baselines
    assert pkt.key_findings == ["Attention is critical"]
    assert pkt.limitations == ["Only tested on one dataset"]


def test_grounding_packet_no_results():
    pkt = _make_packet(
        experiment_results={},
        experiment_status="FAILED",
        experiment_analysis={},
    )
    assert pkt.result_completeness == "none"
    assert pkt.has_real_results is False
    assert "No experiment results available" in pkt.evidence_gaps


def test_grounding_packet_no_ablation_gap():
    pkt = _make_packet(
        experiment_analysis={
            "summary": "OK",
            "final_metrics": {"acc": 0.9},
            "converged": True,
            "ablation_results": [],
        },
    )
    assert "No ablation study results" in pkt.evidence_gaps


def test_grounding_packet_output_dict():
    pkt = _make_packet()
    d = pkt.to_output_dict()
    assert d["result_completeness"] == "full"
    assert d["has_real_results"] is True
    assert d["has_baseline_comparison"] is True
    assert isinstance(d["evidence_gaps"], list)


# ---- deterministic table builders ----

def test_build_main_table_latex_with_data():
    main_results = [
        {
            "method_name": "DeepMethod",
            "dataset": "DemoSet",
            "is_proposed": True,
            "metrics": [
                {"metric_name": "Accuracy", "value": 0.93},
                {"metric_name": "F1", "value": 0.91},
            ],
        },
    ]
    comparison = {
        "BaselineA": {"Accuracy": 0.88, "F1": 0.85},
        "BaselineB": {"Accuracy": 0.90, "F1": 0.87},
    }
    table = WritingAgent._build_main_table_latex(main_results, comparison, {})
    assert "\\begin{table}" in table
    assert "\\label{tab:main_results}" in table
    assert "BaselineA" in table
    assert "BaselineB" in table
    assert "DeepMethod (Ours)" in table
    assert "\\textbf{0.93}" in table  # best accuracy is bolded
    assert "\\bottomrule" in table


def test_build_main_table_latex_empty():
    assert WritingAgent._build_main_table_latex([], {}, {}) == ""


def test_build_ablation_table_latex():
    ablation = [
        {"variant_name": "Full model", "metrics": [{"metric_name": "Accuracy", "value": 0.93}]},
        {"variant_name": "w/o Attn", "metrics": [{"metric_name": "Accuracy", "value": 0.87}]},
    ]
    table = WritingAgent._build_ablation_table_latex(ablation, {})
    assert "\\label{tab:ablation}" in table
    assert "Full model" in table
    assert "w/o Attn" in table
    assert "0.93" in table
    assert "0.87" in table


def test_build_ablation_table_latex_empty():
    assert WritingAgent._build_ablation_table_latex([], {}) == ""


def test_grounding_packet_has_prebuilt_tables():
    pkt = _make_packet()
    assert "\\begin{table}" in pkt.main_table_latex
    assert "\\label{tab:main_results}" in pkt.main_table_latex
    assert "\\begin{table}" in pkt.ablation_table_latex
    assert "\\label{tab:ablation}" in pkt.ablation_table_latex


def test_grounding_packet_scaffold_tables_when_no_results():
    """When experiment fails, scaffold tables are built from blueprint."""
    pkt = _make_packet(
        experiment_results={},
        experiment_status="FAILED",
        experiment_analysis={},
    )
    # Scaffold tables should be generated (not empty)
    assert "\\label{tab:main_results}" in pkt.main_table_latex
    assert "\\label{tab:ablation}" in pkt.ablation_table_latex
    # Proposed method row uses "--" placeholder
    assert "--" in pkt.main_table_latex
    # Blueprint baseline name appears
    assert "BaselineA" in pkt.main_table_latex
    # V2.1+: scaffold uses "--" placeholders instead of "pending"
    assert "--" in pkt.main_table_latex


def test_grounding_packet_no_scaffold_without_metrics():
    """If blueprint has no metrics, scaffold tables can't be built."""
    pkt = _make_packet(
        experiment_results={},
        experiment_status="FAILED",
        experiment_analysis={},
        blueprint={"baselines": [], "metrics": []},
    )
    assert pkt.main_table_latex == ""
    assert pkt.ablation_table_latex == ""


# ---- baseline comparison context ----

def test_baseline_comparison_context():
    pkt = _make_packet()
    ctx = WritingAgent._build_baseline_comparison_context(pkt)
    assert "BASELINE COMPARISON" in ctx
    assert "BaselineA" in ctx


def test_baseline_comparison_context_empty():
    pkt = _make_packet(experiment_analysis={})
    ctx = WritingAgent._build_baseline_comparison_context(pkt)
    assert ctx == ""


# ---- grounding status context ----

def test_grounding_status_context_full():
    pkt = _make_packet()
    ctx = WritingAgent._build_grounding_status_context(pkt)
    assert "FULL" in ctx
    assert "Use exact numbers" in ctx


def test_grounding_status_context_none():
    pkt = _make_packet(
        experiment_results={},
        experiment_status="FAILED",
        experiment_analysis={},
    )
    ctx = WritingAgent._build_grounding_status_context(pkt)
    assert "NONE" in ctx
    # V2.1+: stronger anti-fabrication wording
    assert "Do NOT write" in ctx or "Do NOT fabricate" in ctx


def test_grounding_status_context_quick_eval():
    pkt = _make_packet(experiment_status="QUICK_EVAL_COMPLETED")
    ctx = WritingAgent._build_grounding_status_context(pkt)
    assert "QUICK" in ctx


# ---- table verification / injection ----

def test_verify_and_inject_tables_injects_missing_main_table():
    pkt = _make_packet()
    # Simulate section content without main results table
    content = "We compare our method with baselines.\n\nThe main results show improvements."
    from pathlib import Path
    import shutil, uuid
    tmp_dir = Path(f".test_tbl_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        from nanoresearch.pipeline.workspace import Workspace
        from nanoresearch.config import ResearchConfig
        ws = Workspace.create(topic="t", root=tmp_dir, session_id="tbl001")
        agent = WritingAgent(ws, ResearchConfig(base_url="http://x/v1/", api_key="k"))
        result = agent._verify_and_inject_tables(content, pkt, "Experiments")
        assert "\\label{tab:main_results}" in result
        assert "\\label{tab:ablation}" in result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_verify_and_inject_tables_skips_existing():
    pkt = _make_packet()
    content = (
        "Some text.\n\n"
        "\\begin{table}[t!]\n\\label{tab:main_results}\n\\end{table}\n\n"
        "\\begin{table}[t!]\n\\label{tab:ablation}\n\\end{table}"
    )
    from pathlib import Path
    import shutil, uuid
    tmp_dir = Path(f".test_tbl_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        from nanoresearch.pipeline.workspace import Workspace
        from nanoresearch.config import ResearchConfig
        ws = Workspace.create(topic="t", root=tmp_dir, session_id="tbl002")
        agent = WritingAgent(ws, ResearchConfig(base_url="http://x/v1/", api_key="k"))
        result = agent._verify_and_inject_tables(content, pkt, "Experiments")
        # Should not inject duplicates
        assert result.count("\\label{tab:main_results}") == 1
        assert result.count("\\label{tab:ablation}") == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---- review grounding block ----

def test_review_grounding_block():
    from nanoresearch.agents.review import ReviewAgent
    from nanoresearch.pipeline.workspace import Workspace
    from nanoresearch.config import ResearchConfig
    from pathlib import Path
    import shutil, uuid
    tmp_dir = Path(f".test_rev_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        ws = Workspace.create(topic="t", root=tmp_dir, session_id="rev001")
        agent = ReviewAgent(ws, ResearchConfig(base_url="http://x/v1/", api_key="k"))
        agent._writing_grounding = {
            "result_completeness": "full",
            "has_real_results": True,
        }
        agent._experiment_analysis = {"final_metrics": {"accuracy": 0.93}}
        agent._experiment_status = "COMPLETED"
        block = agent._build_revision_grounding_block()
        assert "FULL RESULTS" in block
        assert "accuracy = 0.93" in block
        assert "PRESERVE" in block
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_review_grounding_block_no_results():
    from nanoresearch.agents.review import ReviewAgent
    from nanoresearch.pipeline.workspace import Workspace
    from nanoresearch.config import ResearchConfig
    from pathlib import Path
    import shutil, uuid
    tmp_dir = Path(f".test_rev_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        ws = Workspace.create(topic="t", root=tmp_dir, session_id="rev002")
        agent = ReviewAgent(ws, ResearchConfig(base_url="http://x/v1/", api_key="k"))
        agent._writing_grounding = {
            "result_completeness": "none",
            "has_real_results": False,
        }
        agent._experiment_analysis = {}
        agent._experiment_status = "FAILED"
        block = agent._build_revision_grounding_block()
        assert "NO REAL RESULTS" in block
        assert "Do NOT add" in block
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
