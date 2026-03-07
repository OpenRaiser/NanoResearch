# Common Cluster Commands

## Resource Monitoring

```bash
# Available GPUs per partition
svp list -p raise
svp list -p belt_road

# Node status (IDLE/MIXED/ALLOCATED)
cinfo -p belt_road occupy-reserved

# Count available GPUs on MIXED nodes
cinfo -p belt_road | sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g" | awk '{
    if ($4 == "MIXED") {
        split($2, g, "/"); avail += g[2]-g[1]; total += g[2]; n++
    }
} END {
    if (n>0) printf "MIXED nodes: %d | Free GPUs: %d / %d | Usage: %.1f%%\n", n, avail, total, (total-avail)*100/total
}'

# Total reserved idle vs spot
svp list | awk 'NR>1 { idle+=$(NF-2); spot+=$(NF-1) }
END { printf "RESERVED_IDLE: %d  SPOT_USED: %d\n", idle, spot }'
```

## Job Management

```bash
# My jobs
squeue -u $USER

# All jobs on partition
squeue -p raise

# Job detail (GPU mapping etc.)
scontrol show job {job_id} -d

# Why did my job fail?
swatch examine {job_id}

# Cancel one job
scancel {job_id}

# Cancel all my Python jobs
squeue -u $USER | grep python | awk '{print $1}' | xargs scancel

# Job stats by user on a partition
squeue -p belt_road -o "%.10u %.8T" | tail -n +2 | sort | uniq -c | sort -rn
```

## GPU Monitoring

```bash
# One-shot GPU status on a node
swatch -n {node_name} nv

# Continuous monitoring
swatch -n {node_name} nv_always

# List processes on node
swatch -n {node_name} list_program

# Clean leftover processes
swatch -n {node_name} clean_process

# Release GPU memory
swatch -n {node_name} memory_release

# Current node GPU count
python -c "import torch; print(torch.cuda.device_count())"

# Current GPU device mapping
echo $CUDA_VISIBLE_DEVICES
```

## Storage Quick Check

```bash
# Home quota
petrelfs-ctl --getquota --uid $USER

# Filesystem usage
df -Th /mnt/petrelfs
df -Th /mnt/dhwfile
```

## Bash Shortcuts (add to ~/.bashrc)

```bash
# Quick debug on belt_road node by suffix (e.g., srun_oc 60)
srun_oc() {
    srun -p belt_road --job-name=debug --quotatype=reserved \
         --gres=gpu:0 --ntasks=1 --cpus-per-task=8 \
         -w "SH-IDC1-10-140-24-$1" --pty /bin/bash
}

# Watch GPU on node
srun_watch() { swatch -n "SH-IDC1-10-140-24-$1" nv_always; }

# Clean GPU on node
srun_clean() { swatch -n "SH-IDC1-10-140-24-$1" clean_process; }

# Status overview
srun_stat() {
    squeue -u $USER
    echo "---"
    svp list -p raise
    svp list -p belt_road
}

# Kill all local GPU processes
kill_gpu() {
    fuser -v /dev/nvidia* 2>/dev/null | awk '{for(i=1;i<=NF;i++) print "kill -9 " $i}' | sh
}
```

## Troubleshooting

```bash
# Job stuck in PENDING — check reason
scontrol show job {job_id} -d | grep Reason

# Exclude problematic node
srun -p raise --gres=gpu:8 -x SH-IDC1-10-140-24-XX ...

# Request specific node
srun -p raise --gres=gpu:8 -w SH-IDC1-10-140-24-YY ...

# Task control
# Ctrl+Z → suspend; fg → resume; bg → background; jobs → list
```
