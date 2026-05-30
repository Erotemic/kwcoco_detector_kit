#!/usr/bin/env bash
# Submit the whole gen002 pipeline as one chain:
#
#   prep  (CPU, ~25 min)      builds full-res universal tiles + WDS shards
#     ├── pup_vs_nonpup        starts when prep finishes, GPU
#     ├── single_sealion       starts when prep finishes, GPU
#     └── lifestage_6cls       starts when prep finishes, GPU
#
# The three training jobs queue concurrently with --dependency=afterok
# on the prep job, so slurm starts them in parallel the moment prep
# completes successfully. If prep fails, none of them run.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_gen002_pipeline.sh
#
# To submit a subset (e.g. only pup + lifestage), pass scheme names:
#   bash projects/.../submit_gen002_pipeline.sh pup_vs_nonpup lifestage_6cls
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: all three schemes. Override with positional args.
DEFAULT_SCHEMES=(pup_vs_nonpup single_sealion lifestage_6cls)
if [ $# -eq 0 ]; then
    SCHEMES=("${DEFAULT_SCHEMES[@]}")
else
    SCHEMES=("$@")
fi

echo "=== gen002 pipeline ==="
echo "  prep job submits first; ${#SCHEMES[@]} training job(s) queue with --dependency=afterok"
echo

# 1. Prep — the script's sbatch echoes "  job: <id>" to stderr;
#    capture from the parsable jobid line on stdout.
prep_jobid=$(bash "$SCRIPT_DIR/submit_prep_gen002.sh" 2>&1 | tee /dev/stderr | grep -oE '^  job:[[:space:]]+[0-9]+' | awk '{print $2}' | tail -1)
if [ -z "$prep_jobid" ]; then
    echo "ERROR: failed to capture prep job id from submit_prep_gen002.sh output" >&2
    exit 1
fi
echo
echo "  -> prep_jobid=$prep_jobid"

# 2. Each scheme's training submit script picks up KCD_DEPENDS_ON and
#    adds --dependency=afterok:$prep_jobid via _submit_train.sh.
export KCD_DEPENDS_ON="$prep_jobid"

train_jobids=()
for scheme in "${SCHEMES[@]}"; do
    submit_script="$SCRIPT_DIR/submit_train_${scheme}_deimv2_hgnetv2_n_1gpu_arisia_gen002.sh"
    if [ ! -f "$submit_script" ]; then
        echo "ERROR: no submit script for scheme '$scheme' at $submit_script" >&2
        exit 1
    fi
    echo
    echo "=== submitting $scheme ==="
    jobid=$(bash "$submit_script" 2>&1 | tee /dev/stderr | grep -oE '^  job:[[:space:]]+[0-9]+' | awk '{print $2}' | tail -1)
    train_jobids+=("$jobid")
done

echo
echo "=== gen002 pipeline queued ==="
echo "  prep:      $prep_jobid"
for i in "${!SCHEMES[@]}"; do
    echo "  ${SCHEMES[$i]}: ${train_jobids[$i]}  (depends on $prep_jobid)"
done
echo
echo "  squeue -j $prep_jobid,$(IFS=,; echo "${train_jobids[*]}")"
