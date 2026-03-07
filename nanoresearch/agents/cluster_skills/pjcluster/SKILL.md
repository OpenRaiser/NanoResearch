---
name: pjcluster
description: Submit, monitor, and manage GPU training jobs on the PJLab SLURM cluster (8xA100 nodes)
version: 0.1.0
---

# PJLab Cluster Experiment Skill

## Purpose
Automate experiment execution on the PJLab SLURM cluster. Covers resource checking, job submission, GPU monitoring, and troubleshooting across `raise` and `belt_road` partitions (8xA100 per node).

## Cluster Quick Reference

| Item | Value |
|------|-------|
| GPU per node | 8x A100 |
| Partitions | `raise`, `belt_road` |
| Quota types | `reserved` (guaranteed), `spot` (preemptible) |
| Bastion | `jump.pjlab.org.cn` → `10.140.37.162/163/164` |
| Proxy | `proxy_on` / `proxy_off` (pre-configured in ~/.bashrc) |
| Code path | `/mnt/petrelfs/$USER/` (fast SSD, ~150G-512G) |
| Data path | `/mnt/dhwfile/tancheng/$USER/` (NFS, ~2.3PB shared) |

## Workflow

```
1. Check resources    →  bash check_cluster.sh -p raise
2. Submit job         →  bash submit_training.sh --partition raise --gpus 8 --env myenv --script "python train.py"
3. Monitor            →  squeue -u $USER / swatch -n {node} nv_always
4. Debug if failed    →  swatch examine {job_id}
```

## Input
- `training_script`: The Python command to run (e.g., `python train.py --config x.yaml`)
- `conda_env`: Conda environment name
- `partition`: `raise` or `belt_road`
- `num_gpus`: 1-8
- `quota_type`: `reserved` (default) or `spot`
- `container` (optional): Apptainer .sif path for container-wrapped execution

## Output
- SLURM job ID
- Training logs at `logs/{job_id}.log`
- Model checkpoints at user-specified save directory

## Key Rules
1. NEVER run GPU tasks on bastion (no GPU there)
2. ALWAYS match `--gres=gpu:N` to actual GPU usage (no card hogging)
3. Use `proxy_on` before network ops (pip, git, wget); `proxy_off` before S3 mount
4. Default to `reserved` quota; fall back to `spot` when reserved is full
5. Do NOT download models in parallel (bandwidth contention)

## Files
- `references/job_submission.md` — srun/sbatch patterns, Apptainer, torchrun
- `references/common_commands.md` — GPU monitoring, task management, bash shortcuts
- `scripts/submit_training.sh` — One-command job submission with auto-generated sbatch
- `scripts/check_cluster.sh` — Cluster resource overview
