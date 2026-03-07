"""Standalone test: run ONLY the experiment stage.

Usage:
    python test_experiment_only.py [session_id]

If no session_id given, uses the latest session with a blueprint.
Cleans the code/ directory and re-runs experiment from scratch.
"""

import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging to see agent output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.agents.experiment import ExperimentAgent


async def main():
    # 1. Find workspace
    root = Path.home() / ".nanobot" / "workspace" / "research"
    positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if positional_args:
        session_id = positional_args[0]
    else:
        # Find latest session with a blueprint
        candidates = []
        for d in sorted(root.iterdir(), reverse=True):
            bp = d / "plans" / "experiment_blueprint.json"
            if bp.exists():
                candidates.append(d.name)
        if not candidates:
            print("No sessions with blueprints found")
            sys.exit(1)
        session_id = candidates[0]

    ws_path = root / session_id
    print(f"Using session: {session_id}")
    print(f"Workspace: {ws_path}")

    # 2. Load config and workspace
    config = ResearchConfig.load()

    # Create manifest if it doesn't exist (for manually-created sessions)
    manifest_path = ws_path / "manifest.json"
    if not manifest_path.exists():
        bp_path_check = ws_path / "plans" / "experiment_blueprint.json"
        topic = "test"
        if bp_path_check.exists():
            bp_data = json.loads(bp_path_check.read_text(encoding="utf-8"))
            topic = bp_data.get("title", "test")
        ws = Workspace.create(topic=topic, session_id=session_id)
    else:
        ws = Workspace(ws_path)

    # 3. Load blueprint
    bp_path = ws_path / "plans" / "experiment_blueprint.json"
    blueprint = json.loads(bp_path.read_text(encoding="utf-8"))
    print(f"Blueprint: {blueprint.get('title', '?')}")

    # 4. Clean old code directory AND iteration checkpoint for fresh experiment
    keep_code = "--keep-code" in sys.argv
    code_dir = ws_path / "code"
    if not keep_code:
        if code_dir.exists():
            print(f"Cleaning old code directory: {code_dir}")
            shutil.rmtree(code_dir)
        code_dir.mkdir(parents=True, exist_ok=True)

        # Clean old results
        results_dir = code_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("Keeping existing code directory (--keep-code)")
        code_dir.mkdir(parents=True, exist_ok=True)

    # Remove iteration checkpoint so agent starts from round 1
    checkpoint = ws_path / "logs" / "iteration_checkpoint.json"
    if checkpoint.exists():
        print("Removing old iteration checkpoint")
        checkpoint.unlink()

    # 5. Create agent and run
    agent = ExperimentAgent(workspace=ws, config=config)

    print(f"\n{'='*60}")
    print(f"Running EXPERIMENT stage (max {config.experiment_max_rounds} rounds)")
    print(f"Conda env: {config.experiment_conda_env or 'auto'}")
    print(f"Quick-eval timeout: {config.quick_eval_timeout}s")
    print(f"{'='*60}\n")

    try:
        result = await agent.run(experiment_blueprint=blueprint)
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RESULT:")
        print(f"  Status: {result.get('experiment_status', '?')}")
        print(f"  Metrics: {json.dumps(result.get('experiment_results', {}), indent=2)[:500]}")
        print(f"{'='*60}")

        # Check if metrics.json was produced
        metrics_path = code_dir / "results" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            print(f"\nmetrics.json exists! Keys: {list(metrics.keys())}")
            if "main_results" in metrics:
                print(f"  main_results: {len(metrics['main_results'])} entries")
            if "ablation_results" in metrics:
                print(f"  ablation_results: {len(metrics['ablation_results'])} entries")
            if "training_log" in metrics:
                print(f"  training_log: {len(metrics['training_log'])} epochs")
        else:
            print("\nWARNING: metrics.json NOT found!")

    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
