#!/usr/bin/env bash
# Submit the pup_vs_nonpup sbatch job and follow its stdout until the
# job terminates. Returns the job's exit code, so this feels like a
# foreground command despite running through slurm.
#
# The actual job spec lives in scripts/sbatch_pup_vs_nonpup.sh — this
# is just submit-and-tail glue.
#
# Reuses the dep-free follow_job.py utility from the kit (no copy):
#   $KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py
#
# Usage:
#   bash scripts/submit_pup_vs_nonpup.sh                 # submit + follow
#   FOLLOW=0 bash scripts/submit_pup_vs_nonpup.sh        # submit and detach
#   ACCOUNT=foo SLURM_PARTITION=gpu bash scripts/submit_pup_vs_nonpup.sh
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

sbatch_args=(
    --parsable
    --output="$LOG_DPATH/%x-%j.out"
    --error="$LOG_DPATH/%x-%j.out"  # merge stderr into stdout — easier to tail
    # Forward the resolved repo + kit paths because slurm copies the
    # sbatch script to its own staging dir before running it, so
    # ${BASH_SOURCE[0]} inside the script can't locate paths.sh.
    --export=ALL,KCD_REPO_ROOT="$KCD_REPO_ROOT",KCD_KIT_DPATH="$KCD_KIT_DPATH"
    --chdir="$KCD_REPO_ROOT"
)
[ -n "$SLURM_PARTITION" ] && sbatch_args+=(--partition="$SLURM_PARTITION")
[ -n "$ACCOUNT" ]         && sbatch_args+=(--account="$ACCOUNT")

echo "Submitting $SCRIPT_DIR/sbatch_pup_vs_nonpup.sh ..." >&2
jobid="$(sbatch "${sbatch_args[@]}" "$SCRIPT_DIR/sbatch_pup_vs_nonpup.sh")"
jobid="${jobid%%;*}"

stdout_fpath="$LOG_DPATH/sealion-pup-nonpup-${jobid}.out"
echo "  job:  $jobid" >&2
echo "  log:  $stdout_fpath" >&2

# Auto-follow only when stdout is a tty (interactive shell). When
# called from a CI/non-interactive context, default to detach so the
# caller can poll separately.
if [ "$FOLLOW" = "auto" ]; then
    if [ -t 1 ]; then FOLLOW=1; else FOLLOW=0; fi
fi

if [ "$FOLLOW" = "1" ] || [ "$FOLLOW" = "true" ]; then
    # Don't exec — after the tail returns, check squeue and print a
    # reattach hint if the job is still in flight. follow_job.py's
    # exit code can't distinguish "user detached without cancel" from
    # "job finished naturally" (both return 0), so the squeue check is
    # the authoritative signal.
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
