#!/usr/bin/env bash
# Attach (or re-attach) a follower to an existing pup_vs_nonpup slurm job.
#
# Use this when:
#   - you Ctrl+C'd `scripts/submit_pup_vs_nonpup.sh`'s tail but the job
#     is still running and you want to watch it again,
#   - you ran with FOLLOW=0 and want to follow now,
#   - you're on a different shell/session from the original submitter.
#
# Usage (assuming cwd = ~/code/kwcoco_detector_kit):
#   bash projects/viame_sealions_2026/scripts/follow_pup_vs_nonpup.sh <jobid>
#   bash projects/viame_sealions_2026/scripts/follow_pup_vs_nonpup.sh        # auto: latest by mtime
#
# The log path is resolved the same way submit_pup_vs_nonpup.sh writes
# it, so this works regardless of cwd.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

LOG_DPATH="${LOG_DPATH:-$KCD_REPO_ROOT/training_runs/slurm_logs}"
FOLLOW_SCRIPT="$KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py"

kcd_require_path "kit follow_job.py" "$FOLLOW_SCRIPT" || {
    echo "  Override the path with KCD_KIT_DPATH=/path/to/kwcoco_detector_kit" >&2
    exit 1
}

JOBID="${1:-}"
if [ -z "$JOBID" ]; then
    # Pick the most-recently-modified sealion-pup-nonpup-*.out file in
    # the log dir. squeue would be more authoritative (only running
    # jobs) but requires the user to be on the slurm node; mtime is
    # cheap and usually right.
    latest_log="$(ls -t "$LOG_DPATH"/sealion-pup-nonpup-*.out 2>/dev/null | head -n1 || true)"
    if [ -z "$latest_log" ]; then
        echo "ERROR: no slurm log found under $LOG_DPATH" >&2
        echo "       Pass the jobid explicitly: $0 <jobid>" >&2
        exit 1
    fi
    # Extract jobid from sealion-pup-nonpup-<jobid>.out.
    JOBID="${latest_log##*-}"
    JOBID="${JOBID%.out}"
    echo "  (auto-detected jobid: $JOBID)" >&2
fi

stdout_fpath="$LOG_DPATH/sealion-pup-nonpup-${JOBID}.out"
echo "  job:  $JOBID" >&2
echo "  log:  $stdout_fpath" >&2
echo "  (Ctrl+C detaches; rerun this command to reattach.)" >&2

exec python3 "$FOLLOW_SCRIPT" "$JOBID" --stdout "$stdout_fpath"
