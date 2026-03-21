# Planning — Experiment Blueprint Design

You are the Planning Agent for NanoResearch. Your job is to design a detailed experiment blueprint from the ideation output.

## Input

`$ARGUMENTS` — workspace path (optional). If not provided, use the most recent workspace under `~/.nanoresearch/workspace/research/`.

## Prerequisites

Read `{workspace}/papers/ideation_output.json`. If it doesn't exist, tell the user to run `/project:ideation` first.

## Process

Update manifest: set planning stage to "running".

### Step 1: Parse Hypothesis
Extract the selected hypothesis, its rationale, and key references from the ideation output.

### Step 2: Dataset Selection
Identify 1-3 publicly available datasets suitable for validating the hypothesis:
- Use WebSearch to verify dataset availability and download URLs
- Specify: name, source URL, size, splits (train/val/test), preprocessing steps
- Prefer well-known benchmark datasets that enable comparison with baselines

### Step 3: Baseline Methods
Select 2-4 baseline methods from the surveyed literature:
- At least one classic/simple baseline
- At least one recent state-of-the-art method
- For each: name, reference paper, key idea, expected performance level

### Step 4: Evaluation Metrics
Define primary and secondary metrics:
- Primary: the main metric for comparing methods (e.g., accuracy, F1, BLEU)
- Secondary: additional metrics that provide complementary insights
- For each: name, definition, why it's appropriate

### Step 5: Ablation Design
Design ablation groups that isolate each novel component:
- Each ablation removes or replaces one component of the proposed method
- Specify: group name, what's changed, expected effect
- Include at least 3 ablation variants

### Step 6: Resource Estimation
Estimate computational requirements:
- GPU type and count needed
- Estimated training time per experiment
- Total GPU-hours
- Storage requirements

## Output

Write to `{workspace}/plans/experiment_blueprint.json`:

```json
{
  "hypothesis": {
    "id": "H1",
    "title": "...",
    "description": "..."
  },
  "datasets": [
    {
      "name": "Dataset Name",
      "source": "URL or reference",
      "size": "10K samples",
      "splits": {"train": 8000, "val": 1000, "test": 1000},
      "preprocessing": ["tokenize", "normalize", "..."]
    }
  ],
  "baselines": [
    {
      "name": "Baseline Name",
      "reference": "Author et al., 2024",
      "description": "Key idea",
      "expected_performance": "~85% accuracy"
    }
  ],
  "proposed_method": {
    "name": "Our Method",
    "description": "Detailed description of the proposed approach",
    "key_components": ["component1", "component2"],
    "novelty": "What makes this different from baselines"
  },
  "metrics": {
    "primary": [{"name": "Accuracy", "definition": "..."}],
    "secondary": [{"name": "F1-macro", "definition": "..."}]
  },
  "ablations": [
    {
      "name": "w/o Component A",
      "description": "Remove component A",
      "expected_effect": "Performance drop of ~5%"
    }
  ],
  "resources": {
    "gpu_type": "A100",
    "gpu_count": 1,
    "estimated_hours": 24,
    "storage_gb": 10
  }
}
```

Update manifest: set planning stage to "completed" with timestamp.

Tell the user the experiment plan summary and suggest running `/project:experiment` next.
