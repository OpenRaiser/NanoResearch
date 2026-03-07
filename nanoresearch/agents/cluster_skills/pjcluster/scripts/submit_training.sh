#!/bin/bash
# ============================================================
# submit_training.sh - Generate and submit a SLURM training job
#
# Usage:
#   bash submit_training.sh \
#     --partition raise \
#     --gpus 8 \
#     --env myenv \
#     --script "python train.py --config default.yaml" \
#     [--quota spot] \
#     [--time 72:00:00] \
#     [--container /path/to/ubuntu_22.04.sif] \
#     [--job-name my_experiment] \
#     [--log-dir logs]
# ============================================================

set -euo pipefail

# === Defaults ===
PARTITION="raise"
GPUS=8
QUOTA="reserved"
ENV_NAME=""
SCRIPT_CMD=""
TIME="72:00:00"
CONTAINER=""
JOB_NAME="train"
LOG_DIR="logs"

# === Parse arguments ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition)  PARTITION="$2";  shift 2 ;;
        --gpus)       GPUS="$2";       shift 2 ;;
        --env)        ENV_NAME="$2";   shift 2 ;;
        --script)     SCRIPT_CMD="$2"; shift 2 ;;
        --quota)      QUOTA="$2";      shift 2 ;;
        --time)       TIME="$2";       shift 2 ;;
        --container)  CONTAINER="$2";  shift 2 ;;
        --job-name)   JOB_NAME="$2";   shift 2 ;;
        --log-dir)    LOG_DIR="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# === Validate ===
if [ -z "$ENV_NAME" ]; then echo "Error: --env required"; exit 1; fi
if [ -z "$SCRIPT_CMD" ]; then echo "Error: --script required"; exit 1; fi

mkdir -p "$LOG_DIR"

# === Detect conda ===
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[ ! -f "$CONDA_SH" ] && CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
    echo "Error: conda.sh not found at ~/anaconda3 or ~/miniconda3"
    exit 1
fi

# === Build run command ===
if [ -n "$CONTAINER" ]; then
    RUN_CMD="apptainer exec --nv -B /mnt:/mnt ${CONTAINER} bash -c 'source ${CONDA_SH} && conda activate ${ENV_NAME} && ${SCRIPT_CMD}'"
else
    RUN_CMD="source ${CONDA_SH} && conda activate ${ENV_NAME} && ${SCRIPT_CMD}"
fi

# === Generate sbatch script ===
SBATCH_FILE="${LOG_DIR}/job_${JOB_NAME}_$(date +%Y%m%d_%H%M%S).sh"

cat > "$SBATCH_FILE" << EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$((GPUS * 8))
#SBATCH --quotatype=${QUOTA}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%j.log
#SBATCH --error=${LOG_DIR}/%j.err
#SBATCH --time=${TIME}

echo "=== Job \$SLURM_JOB_ID on \$SLURM_NODELIST | ${GPUS} GPUs | $(date) ==="

${RUN_CMD}

echo "=== Done: exit \$? at \$(date) ==="
EOF

chmod +x "$SBATCH_FILE"
echo "Script: $SBATCH_FILE"
sbatch "$SBATCH_FILE"
