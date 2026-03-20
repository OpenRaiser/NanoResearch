---
name: nanoresearch-planning
description: Produce an experiment blueprint from a research hypothesis
version: 0.1.0
---

# Planning Skill

## Purpose
Take the selected hypothesis from ideation and produce a detailed experiment blueprint specifying datasets, baselines, evaluation metrics, and ablation groups.

## Tools Required
None. This skill operates entirely through LLM reasoning over the ideation output.

## Input
- `ideation_output`: Path to `papers/ideation_output.json` produced by the ideation skill

## Process
1. Parse the selected hypothesis and supporting literature from the ideation output
2. Identify candidate datasets that are publicly available and appropriate for validating the hypothesis
3. Select 2-4 baseline methods from the surveyed literature for comparison
4. Define primary and secondary evaluation metrics aligned with the hypothesis
5. Design ablation groups that isolate each novel component of the proposed approach
6. Estimate computational requirements and timeline for each experiment
7. Compile everything into a structured experiment blueprint

## Output
Produces `papers/experiment_blueprint.json` containing:
- Selected hypothesis (carried forward)
- Dataset specifications (name, source, splits, preprocessing steps)
- Baseline methods with references
- Evaluation metrics and success criteria
- Ablation study design (groups, variables, expected outcomes)
- Resource estimates and experiment schedule
