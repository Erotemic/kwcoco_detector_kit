#!/usr/bin/env bash
# Submit a single-GPU baseline job and follow its stdout until done.
# Mirror of submit_pup_vs_nonpup.sh but for the parameterized baseline
# launcher (sbatch_baseline.sh -> launch_baseline.sh).
#
# Defaults to: deimv2_hgnetv2_n on pup_vs_nonpup, 30 epochs, 1 GPU.
#
# Usage (assuming cwd = ~/code/kwcoco_detector_kit):
#   bash projects/viame_sealions_2026/scripts/submit_baseline.sh
#
#   KCD_SCHEME=single_sealion bash projects/.../scripts/submit_baseline.sh
#   KCD_VARIANT=deimv2_hgnetv2_pico bash projects/.../scripts/submit_baseline.sh
#   FOLLOW=0 bash projects/.../scripts/submit_baseline.sh   # submit and detach
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

LOG_DPATH="${LOG_DPATH:-$KCD_SLURM_LOG_DPATH}"
FOLLOW="${FOLLOW:-auto}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"

FOLLOW_SCRIPT="$KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py"
kcd_require_path "kit follow_job.py" "$FOLLOW_SCRIPT" || {
    echo "  Override the path with KCD_KIT_DPATH=/path/to/kwcoco_detector_kit" >&2
    exit 1
}

mkdir -p "$LOG_DPATH"

# Forward baseline-specific env vars to sbatch so the slurm script
# (which executes inside a fresh shell on the compute node) can see them.
forward_vars=(KCD_REPO_ROOT="$KCD_REPO_ROOT" KCD_KIT_DPATH="$KCD_KIT_DPATH")
for v in KCD_SCHEME KCD_VARIANT KCD_INPUT_HW KCD_TRAIN_POLICY \
         KCD_CATEGORY_NAMES KCD_NUM_EPOCHS KCD_PER_GPU_BATCH \
         KCD_INIT_CHECKPOINT KCD_TRAIN_FROM_SCRATCH KCD_NCCL_DEBUG \
         KCD_DEV_MOUNT_DEIMV2 KCD_IMAGE; do
    val="${!v:-}"
    [ -n "$val" ] && forward_vars+=("$v=$val")
done

export_str="ALL"
for kv in "${forward_vars[@]}"; do
    export_str="$export_str,$kv"
done

# Encode variant + scheme in the job name so squeue / log files are
# self-identifying when many baselines are queued.
job_scheme="${KCD_SCHEME:-pup_vs_nonpup}"
job_variant="${KCD_VARIANT:-deimv2_hgnetv2_n}"
job_short_variant="${job_variant#deimv2_}"  # hgnetv2_n / dinov3_s / etc.
job_name="baseline-${job_short_variant}-${job_scheme}"

sbatch_args=(
    --parsable
    --job-name="$job_name"
    --output="$LOG_DPATH/%x-%j.out"
    --error="$LOG_DPATH/%x-%j.out"
    --export="$export_str"
    --chdir="$KCD_REPO_ROOT"
)
[ -n "$SLURM_PARTITION" ] && sbatch_args+=(--partition="$SLURM_PARTITION")
[ -n "$ACCOUNT" ]         && sbatch_args+=(--account="$ACCOUNT")

echo "Submitting $SCRIPT_DIR/sbatch_baseline.sh ..." >&2
echo "  scheme=$job_scheme  variant=$job_variant" >&2
jobid="$(sbatch "${sbatch_args[@]}" "$SCRIPT_DIR/sbatch_baseline.sh")"
jobid="${jobid%%;*}"

stdout_fpath="$LOG_DPATH/${job_name}-${jobid}.out"
echo "  job:  $jobid" >&2
echo "  log:  $stdout_fpath" >&2

# Auto-follow only when stdout is a tty.
if [ "$FOLLOW" = "auto" ]; then
    if [ -t 1 ]; then FOLLOW=1; else FOLLOW=0; fi
fi

if [ "$FOLLOW" = "1" ] || [ "$FOLLOW" = "true" ]; then
    reattach_cmd="bash $SCRIPT_DIR/follow_job.sh $jobid"
    set +e
    python3 "$FOLLOW_SCRIPT" "$jobid" --stdout "$stdout_fpath"
    follow_rc=$?
    job_state="$(squeue -h -j "$jobid" -o '%T' 2>/dev/null | head -n1 || true)"
    set -e
    if [ -n "$job_state" ]; then
        echo >&2
        echo "[follow detached] job $jobid is still on slurm (state: $job_state)." >&2
        echo "  reattach: $reattach_cmd" >&2
        echo "  cancel:   scancel $jobid" >&2
    fi
    exit "$follow_rc"
fi

echo "$jobid"
echo "  reattach with: bash $SCRIPT_DIR/follow_job.sh $jobid" >&2
