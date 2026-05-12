#!/usr/bin/env bash
# Submit the smoke ladder as dependent Slurm jobs. By default this stops after
# the VIAME subset test; set MAX_STAGE=04 to queue the full run too.
set -euo pipefail

THIS_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_STAGE="${MAX_STAGE:-03}"

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
for stage in "${stages[@]}"; do
    if [ -n "$dependency" ]; then
        jobid="$(DEPENDENCY="afterok:$dependency" "$THIS_DPATH/submit_stage.sh" "$stage")"
    else
        jobid="$("$THIS_DPATH/submit_stage.sh" "$stage")"
    fi
    echo "stage $stage -> job $jobid"
    dependency="$jobid"
    last_stage="$stage"
done

echo
echo "Queued through stage $last_stage"
