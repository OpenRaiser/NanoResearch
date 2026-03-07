#!/bin/bash
# ============================================================
# check_cluster.sh - Quick cluster status overview
#
# Usage:
#   bash check_cluster.sh              # Both partitions
#   bash check_cluster.sh -p raise     # Specific partition
#   bash check_cluster.sh -u username  # Specific user's jobs
# ============================================================

PARTITION=""
USERNAME="$USER"

while [[ $# -gt 0 ]]; do
    case $1 in
        -p) PARTITION="$2"; shift 2 ;;
        -u) USERNAME="$2"; shift 2 ;;
        *)  shift ;;
    esac
done

echo "==========================================="
echo "  PJLab Cluster Status  $(date '+%Y-%m-%d %H:%M')"
echo "==========================================="

# === Partition Resources ===
if [ -z "$PARTITION" ]; then
    PARTS=("raise" "belt_road")
else
    PARTS=("$PARTITION")
fi

for P in "${PARTS[@]}"; do
    echo ""
    echo "--- $P ---"
    svp list -p "$P" 2>/dev/null || echo "  (svp unavailable)"

    if command -v cinfo &>/dev/null; then
        cinfo -p "$P" 2>/dev/null | sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[m|K]//g" | awk -v p="$P" '{
            if ($4 == "MIXED") {
                split($2, g, "/"); avail += g[2]-g[1]; total += g[2]; n++
            }
        } END {
            if (n>0) printf "  MIXED: %d nodes | Free GPUs: %d/%d (%.0f%% busy)\n", n, avail, total, (total-avail)*100/total
        }'
    fi
done

# === My Jobs ===
echo ""
echo "--- Jobs ($USERNAME) ---"
squeue -u "$USERNAME" -o "%.10i %.15j %.8T %.4D %.6C %.10M %.20R" 2>/dev/null | head -20 || echo "  (none)"

# === Storage ===
echo ""
echo "--- Storage ---"
petrelfs-ctl --getquota --uid "$USERNAME" 2>/dev/null || true
df -Th /mnt/petrelfs 2>/dev/null | awk 'NR==2{print "  petrelfs: "$3" / "$2" ("$5" used)"}'
df -Th /mnt/dhwfile 2>/dev/null | awk 'NR==2{print "  dhwfile:  "$3" / "$2" ("$5" used)"}'
echo "==========================================="
