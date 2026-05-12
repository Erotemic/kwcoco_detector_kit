#!/usr/bin/env bash
# Submit one DINOv2/OpenGroundingDINO smoke stage as a Slurm job that runs
# inside the built Docker image.
set -euo pipefail

THIS_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DPATH="$(cd "$THIS_DPATH/../../.." && pwd)"

STAGE="${1:-00}"
DEPENDENCY="${DEPENDENCY:-}"
LOG_DPATH="${LOG_DPATH:-$ROOT_DPATH/smoketests/dino_v2_4x/slurm/logs}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
FOLLOW="${FOLLOW:-auto}"

mkdir -p "$LOG_DPATH"

case "$STAGE" in
    00|00_mock|mock)
        stage_id="00"
        stage_script="00_mock_demo_dataload.sh"
        gpus=0
        cpus="${CPUS_PER_TASK:-4}"
        mem="${MEM:-16G}"
        time_limit="${TIME_LIMIT:-00:20:00}"
        ;;
    01|01_1gpu|1gpu)
        stage_id="01"
        stage_script="01_ogdino_demo_1gpu.sh"
        gpus=1
        cpus="${CPUS_PER_TASK:-8}"
        mem="${MEM:-48G}"
        time_limit="${TIME_LIMIT:-01:00:00}"
        ;;
    02|02_4gpu_demo|4gpu_demo)
        stage_id="02"
        stage_script="02_ogdino_demo_4gpu.sh"
        gpus=4
        cpus="${CPUS_PER_TASK:-24}"
        mem="${MEM:-160G}"
        time_limit="${TIME_LIMIT:-01:30:00}"
        ;;
    03|03_viame_subset|subset)
        stage_id="03"
        stage_script="03_ogdino_viame_subset_4gpu.sh"
        gpus=4
        cpus="${CPUS_PER_TASK:-24}"
        mem="${MEM:-192G}"
        time_limit="${TIME_LIMIT:-02:00:00}"
        ;;
    04|04_viame_full|full)
        stage_id="04"
        stage_script="04_ogdino_viame_full_4gpu.sh"
        gpus=4
        cpus="${CPUS_PER_TASK:-32}"
        mem="${MEM:-192G}"
        time_limit="${TIME_LIMIT:-72:00:00}"
        ;;
    *)
        echo "Unknown stage: $STAGE" >&2
        echo "Known stages: 00, 01, 02, 03, 04" >&2
        exit 1
        ;;
esac

sbatch_args=(
    --parsable
    --job-name="kcd-dino2-${stage_id}"
    --cpus-per-task="$cpus"
    --mem="$mem"
    --time="$time_limit"
    --output="$LOG_DPATH/%x-%j.out"
    --error="$LOG_DPATH/%x-%j.out"
    --export=ALL,STAGE_SCRIPT="$stage_script",NUM_GPUS_REQUESTED="$gpus",HOST_KIT_DPATH="$ROOT_DPATH"
)
stdout_template="$LOG_DPATH/kcd-dino2-${stage_id}-%j.out"
stdout_fpath=""

if [ "$gpus" != "0" ]; then
    sbatch_args+=(--gres="gpu:$gpus")
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

echo "Submitting stage $stage_id ($stage_script): gpus=$gpus cpus=$cpus mem=$mem time=$time_limit" >&2
jobid="$(sbatch "${sbatch_args[@]}" "$THIS_DPATH/run_stage_in_docker.sh")"
jobid="${jobid%%;*}"
echo "$jobid"

stdout_fpath="${stdout_template//%j/$jobid}"
echo "stage $stage_id -> job $jobid" >&2
echo "log: $stdout_fpath" >&2

if [ "$FOLLOW" = "auto" ]; then
    if [ -t 1 ]; then
        FOLLOW=1
    else
        FOLLOW=0
    fi
fi

if [ "$FOLLOW" = "1" ] || [ "$FOLLOW" = "true" ]; then
    python "$THIS_DPATH/follow_job.py" "$jobid" --stdout "$stdout_fpath"
fi
