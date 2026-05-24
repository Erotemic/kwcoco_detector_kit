#!/usr/bin/env bash
# Attach (or re-attach) a follower to an existing sealion slurm job.
#
# Works for both the big-run jobs (`sealion-pup-nonpup-<id>.out`) and
# the parameterized baselines (`baseline-<variant>-<scheme>-<id>.out`).
#
# Use this when:
#   - you Ctrl+C'd a submit_*.sh tail but the job is still running,
#   - you ran with FOLLOW=0 and want to follow now,
#   - you're on a different shell/session from the original submitter.
#
# Usage (assuming cwd = ~/code/kwcoco_detector_kit):
#   bash projects/viame_sealions_2026/scripts/follow_job.sh <jobid>
#   bash projects/viame_sealions_2026/scripts/follow_job.sh           # auto: latest *.out by mtime
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

LOG_DPATH="${LOG_DPATH:-$KCD_SLURM_LOG_DPATH}"
FOLLOW_SCRIPT="$KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py"

kcd_require_path "kit follow_job.py" "$FOLLOW_SCRIPT" || {
    echo "  Override the path with KCD_KIT_DPATH=/path/to/kwcoco_detector_kit" >&2
    exit 1
}

JOBID="${1:-}"
stdout_fpath=""
if [ -z "$JOBID" ]; then
    # Pick the most-recently-modified *.out file in the log dir.
    # squeue would be authoritative (only running jobs) but mtime is
    # cheap and works for any of our naming conventions.
    latest_log="$(ls -t "$LOG_DPATH"/*.out 2>/dev/null | head -n1 || true)"
    if [ -z "$latest_log" ]; then
        echo "ERROR: no slurm log found under $LOG_DPATH" >&2
        echo "       Pass the jobid explicitly: $0 <jobid>" >&2
        exit 1
    fi
    # Filename format: <jobname>-<jobid>.out
    JOBID="${latest_log##*-}"
    JOBID="${JOBID%.out}"
    stdout_fpath="$latest_log"
    echo "  (auto-detected jobid: $JOBID from $(basename "$latest_log"))" >&2
fi

# If we don't have the path from auto-detect, search for it by jobid.
# Avoids hardcoding a single jobname pattern.
if [ -z "$stdout_fpath" ]; then
    stdout_fpath="$(ls "$LOG_DPATH"/*-${JOBID}.out 2>/dev/null | head -n1 || true)"
    if [ -z "$stdout_fpath" ]; then
        echo "ERROR: no log matching *-${JOBID}.out under $LOG_DPATH" >&2
        echo "       Check that the job was submitted via scripts/submit_*.sh" >&2
        exit 1
    fi
fi

echo "  job:  $JOBID" >&2
echo "  log:  $stdout_fpath" >&2
echo "  (Ctrl+C detaches; rerun this command to reattach.)" >&2

exec python3 "$FOLLOW_SCRIPT" "$JOBID" --stdout "$stdout_fpath"
