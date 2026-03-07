# SLURM Job Submission

## Partitions & Quotas

| Partition | Nodes | GPU |
|-----------|-------|-----|
| `raise` | ~12 nodes | 8xA100 each |
| `belt_road` | ~66 nodes | 8xA100 each |

| Quota | Behavior |
|-------|----------|
| `reserved` | Guaranteed, dedicated |
| `spot` | Preemptible, more available but job may be killed |

Check available resources:
```bash
svp list -p raise
svp list -p belt_road
```

## Interactive Debug

```bash
# Single GPU session
srun -p raise --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --quotatype=reserved /bin/bash

# Multi GPU session
srun -p belt_road --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --quotatype=reserved /bin/bash

# Attach to a running job's node
srun -p raise -w {node_name} --pty bash

# CPU-only (no GPU needed)
srun -p raise --gres=gpu:0 --quotatype=reserved /bin/bash

# Exit: Ctrl+D
```

## Batch Submission (sbatch)

### Basic single-node training
```bash
#!/bin/bash
#SBATCH --partition=raise
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --quotatype=reserved
#SBATCH --job-name=train_exp
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=72:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

python train.py --config configs/default.yaml
```

### Multi-GPU with torchrun (DDP)
```bash
#!/bin/bash
#SBATCH --partition=belt_road
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --quotatype=reserved
#SBATCH --job-name=ddp_train
#SBATCH --output=logs/%j.log

source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

torchrun --nproc_per_node=8 --master_port=29500 \
    train.py --config configs/ddp.yaml
```

### With Apptainer container (for flash-attn etc.)
```bash
# Direct srun
srun -p raise --gres=gpu:8 --quotatype=reserved \
    apptainer exec --nv -B /mnt:/mnt \
    /mnt/petrelfs/$USER/ubuntu_22.04.sif \
    bash -c "source ~/anaconda3/etc/profile.d/conda.sh && \
             conda activate myenv && python train.py"

# Or use wrapper script (~/run.sh)
srun -p raise --gres=gpu:8 --quotatype=reserved \
    bash ~/run.sh python train.py
```

Apptainer wrapper (`~/run.sh`):
```bash
#!/bin/bash
CONTAINER="/mnt/petrelfs/$USER/ubuntu_22.04.sif"
CONDA_SH="~/anaconda3/etc/profile.d/conda.sh"
ENV_NAME="myenv"

apptainer exec --nv -B /mnt:/mnt ${CONTAINER} \
    bash -c "source ${CONDA_SH} && conda activate ${ENV_NAME} && $*"
```

### Apptainer key parameters

| Parameter | Purpose | Without it |
|-----------|---------|------------|
| `--nv` | GPU passthrough | GPU invisible |
| `-B /mnt:/mnt` | Mount host filesystem | Cannot find code/data/conda |
| `.sif` path | Container image | Must specify |

## Using submit_training.sh (Recommended)

```bash
# Basic
bash submit_training.sh \
    --partition raise --gpus 8 --env myenv \
    --script "python train.py --config default.yaml"

# With Apptainer
bash submit_training.sh \
    --partition raise --gpus 8 --env myenv \
    --script "python train.py" \
    --container /mnt/petrelfs/$USER/ubuntu_22.04.sif

# Spot quota, custom time limit
bash submit_training.sh \
    --partition belt_road --gpus 4 --env myenv \
    --script "python eval.py" \
    --quota spot --time 24:00:00 --job-name eval_run
```

## Node Status

| Status | Meaning |
|--------|---------|
| `IDLE` | 0/8 GPU used |
| `MIXED` | Partially used (can squeeze in) |
| `ALLOCATED` | 8/8 GPU used |

## Decision: Reserved vs Spot

```
svp list -p {partition}
→ RESERVED_IDLE > 0  →  use --quotatype=reserved
→ RESERVED_IDLE = 0  →  use --quotatype=spot (may be preempted)
```

## Important Rules
- `--gres=gpu:N` MUST match actual usage. Do NOT hog cards.
- Do NOT manually set `CUDA_VISIBLE_DEVICES`. SLURM handles GPU mapping.
- Use `-x {node}` to exclude bad nodes; `-w {node}` to request specific ones.
