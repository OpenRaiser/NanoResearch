---
name: nanoresearch-experiment
description: Generate a Python code skeleton from an experiment blueprint
version: 0.1.0
---

# Experiment Skill

## Purpose
Take the experiment blueprint and produce a runnable Python code skeleton that implements the proposed method, baselines, training loops, evaluation harness, and ablation configurations.

## Tools Required
None. This skill operates entirely through LLM code generation based on the experiment blueprint.

## Input
- `experiment_blueprint`: Path to `papers/experiment_blueprint.json` produced by the planning skill

## Process
1. Parse the experiment blueprint for datasets, baselines, metrics, and ablation groups
2. Generate the project directory structure (data loaders, models, training, evaluation, configs)
3. Produce data loading and preprocessing code for each specified dataset
4. Implement model architecture stubs for the proposed method and each baseline
5. Generate training loop with logging, checkpointing, and early stopping
6. Implement the evaluation harness computing all specified metrics
7. Create configuration files for each ablation group
8. Add a main entry point that accepts a config and runs the full train-evaluate pipeline

## Output
Produces `experiments/` directory containing:
- `data/`: Data loading and preprocessing modules
- `models/`: Model architecture implementations (proposed method and baselines)
- `training/`: Training loop and optimization utilities
- `evaluation/`: Metric computation and result aggregation
- `configs/`: YAML configuration files for each experiment and ablation variant
- `run.py`: Main entry point for launching experiments
- `requirements.txt`: Python dependencies
