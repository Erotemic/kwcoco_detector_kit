#!/usr/bin/env bash
# Health-check a training run's slurm log.
#
# Companion to the sea-lion project's follow_job.sh. That one prints the log
# verbatim, which is right when you are watching a run start; this one answers
# "is it OK" for a job that has been going for hours. It flags the failure
# modes that have cost this project GPU-days -- NaN zombies, mid-epoch stalls,
# AP collapse, NCCL watchdog aborts, OOM, rank crashes -- and prints per-epoch
# AP and wall time with a remaining-time estimate.
#
# The Python half is stdlib-only and is invoked BY PATH with the system
# interpreter, exactly like follow_job.py: the kit is not pip-installed on
# aiq's login node, so triage must not require a container.
#
# This wrapper is deliberately trivial and project-local only because it reads
# KCD_SLURM_LOG_DPATH from this project's paths.sh. It is a straight copy into
# any other project.
#
# Usage (from the kit root):
#   bash projects/viame_fish_2026/scripts/run_health.sh              # newest log
#   bash projects/viame_fish_2026/scripts/run_health.sh 490          # by jobid
#   bash projects/viame_fish_2026/scripts/run_health.sh 490 --watch  # stream events
#
# Any extra arguments are forwarded, so --num_epochs / --stall_seconds /
# --fail_on_findings all work. --fail_on_findings exits non-zero on a fatal
# finding, which makes this usable as a check in a script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

LOG_DPATH="${LOG_DPATH:-$KCD_SLURM_LOG_DPATH}"
HEALTH_SCRIPT="$KCD_KIT_DPATH/kwcoco_detector_kit/monitoring/log_health.py"
kcd_require_path "kit log_health.py" "$HEALTH_SCRIPT" || {
    echo "  Override the kit location with KCD_KIT_DPATH=/path/to/kwcoco_detector_kit" >&2
    exit 1
}

JOBID=""
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    JOBID="$1"
    shift
fi

if [ -n "$JOBID" ]; then
    LOG_FPATH="$(ls "$LOG_DPATH"/*-"${JOBID}".out 2>/dev/null | head -n1 || true)"
    if [ -z "$LOG_FPATH" ]; then
        echo "ERROR: no log matching *-${JOBID}.out under $LOG_DPATH" >&2
        exit 1
    fi
else
    # Newest by mtime. squeue would be authoritative for "still running", but
    # this tool is just as useful on a finished run, so mtime is the right
    # cheap default -- and it works for every naming convention we use.
    LOG_FPATH="$(ls -t "$LOG_DPATH"/*.out 2>/dev/null | head -n1 || true)"
    if [ -z "$LOG_FPATH" ]; then
        echo "ERROR: no slurm log found under $LOG_DPATH" >&2
        echo "       Pass a jobid explicitly: $0 <jobid>" >&2
        exit 1
    fi
    echo "  (newest log: $(basename "$LOG_FPATH"))" >&2
fi

exec python3 "$HEALTH_SCRIPT" "$LOG_FPATH" "$@"
