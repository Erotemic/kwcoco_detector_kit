#!/usr/bin/env bash
# Submit the smoke ladder as dependent Slurm jobs. By default this stops after
# the VIAME subset test; set MAX_STAGE=04 to queue the full run too.
set -euo pipefail

THIS_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DPATH="$(cd "$THIS_DPATH/../../.." && pwd)"
MAX_STAGE="${MAX_STAGE:-03}"
FOLLOW="${FOLLOW:-auto}"
LOG_DPATH="${LOG_DPATH:-$ROOT_DPATH/smoketests/dino_v2_4x/slurm/logs}"

stages=(00 01 02 03)
if [ "$MAX_STAGE" = "04" ]; then
    stages+=(04)
elif [ "$MAX_STAGE" != "03" ]; then
    stages=()
    for s in 00 01 02 03 04; do
        stages+=("$s")
        [ "$s" = "$MAX_STAGE" ] && break
    done
fi

dependency=""
last_stage=""
jobids=()
stage_ids=()
for stage in "${stages[@]}"; do
    if [ -n "$dependency" ]; then
        jobid="$(FOLLOW=0 DEPENDENCY="afterok:$dependency" "$THIS_DPATH/submit_stage.sh" "$stage")"
    else
        jobid="$(FOLLOW=0 "$THIS_DPATH/submit_stage.sh" "$stage")"
    fi
    echo "stage $stage -> job $jobid"
    dependency="$jobid"
    last_stage="$stage"
    jobids+=("$jobid")
    stage_ids+=("$stage")
done

echo
echo "Queued through stage $last_stage"

if [ "$FOLLOW" = "auto" ]; then
    if [ -t 1 ]; then
        FOLLOW=1
    else
        FOLLOW=0
    fi
fi

if [ "$FOLLOW" = "1" ] || [ "$FOLLOW" = "true" ]; then
    echo
    echo "Following queued stages. Ctrl-C stops following; Slurm jobs keep running."
    for idx in "${!jobids[@]}"; do
        stage="${stage_ids[$idx]}"
        jobid="${jobids[$idx]}"
        stdout_fpath="$LOG_DPATH/kcd-dino2-${stage}-${jobid}.out"
        echo
        echo "=== Follow stage $stage job $jobid ==="
        set +e
        python "$THIS_DPATH/follow_job.py" "$jobid" --stdout "$stdout_fpath"
        rc="$?"
        set -e
        if [ "$rc" != "0" ]; then
            echo "stage $stage job $jobid failed with rc=$rc" >&2
            exit "$rc"
        fi
    done
fi
