#!/usr/bin/env bash
# Submit the VIAME Sea Lions OGDino full-data reproduce run as a Slurm job
# that launches the Docker payload at run_in_docker.sh.
set -euo pipefail

THIS_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DPATH="$(cd "$THIS_DPATH/../../.." && pwd)"

LOG_DPATH="${LOG_DPATH:-$ROOT_DPATH/reproduce/viame_sealions_2026/slurm/logs}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
FOLLOW="${FOLLOW:-auto}"
DEPENDENCY="${DEPENDENCY:-}"

GPUS="${GPUS:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-180G}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
JOB_NAME="${JOB_NAME:-kcd-viame-ogdino-full}"

KCD_REPRODUCE_TAG="${KCD_REPRODUCE_TAG:-ogdino_swint_full}"

mkdir -p "$LOG_DPATH"

sbatch_args=(
    --parsable
    --job-name="$JOB_NAME"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --time="$TIME_LIMIT"
    --output="$LOG_DPATH/%x-%j.out"
    --error="$LOG_DPATH/%x-%j.out"
    --export=ALL,NUM_GPUS_REQUESTED="$GPUS",HOST_KIT_DPATH="$ROOT_DPATH",KCD_REPRODUCE_TAG="$KCD_REPRODUCE_TAG"
)
stdout_template="$LOG_DPATH/${JOB_NAME}-%j.out"

if [ "$GPUS" != "0" ]; then
    sbatch_args+=(--gres="gpu:$GPUS")
fi
if [ -n "$SLURM_PARTITION" ]; then
    sbatch_args+=(--partition="$SLURM_PARTITION")
fi
if [ -n "$ACCOUNT" ]; then
    sbatch_args+=(--account="$ACCOUNT")
fi
if [ -n "$DEPENDENCY" ]; then
    sbatch_args+=(--dependency="$DEPENDENCY")
fi

echo "Submitting reproduce run: tag=$KCD_REPRODUCE_TAG gpus=$GPUS cpus=$CPUS_PER_TASK mem=$MEM time=$TIME_LIMIT" >&2
jobid="$(sbatch "${sbatch_args[@]}" "$THIS_DPATH/run_in_docker.sh")"
jobid="${jobid%%;*}"
echo "$jobid"

stdout_fpath="${stdout_template//%j/$jobid}"
echo "job $jobid" >&2
echo "log: $stdout_fpath" >&2

if [ "$FOLLOW" = "auto" ]; then
    if [ -t 1 ]; then
        FOLLOW=1
    else
        FOLLOW=0
    fi
fi

if [ "$FOLLOW" = "1" ] || [ "$FOLLOW" = "true" ]; then
    # Reuse the smoketests follow_job.py — same Slurm semantics, no need
    # to duplicate.
    python "$ROOT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py" "$jobid" --stdout "$stdout_fpath"
fi
