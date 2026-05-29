#!/usr/bin/env bash
# Internal boilerplate. Submit a training run to slurm and tail its
# stdout until done. Called from the descriptive submit_train_*.sh
# wrappers — not invoked directly.
#
# Contract: the wrapper must have exported these KCD_* env vars before
# sourcing this script:
#   KCD_RUN_NAME       experiment id; used for slurm job name + KCD_ROOT
#   KCD_SCHEME         scheme name (drives kwcoco paths)
#   KCD_VARIANT        DEIMv2 variant
#   KCD_NUM_GPUS       1..N — controls sbatch --gres + DDP launch
#   KCD_PER_GPU_BATCH  per-GPU batch size (total = NUM_GPUS * PER_GPU_BATCH)
#   KCD_NUM_EPOCHS
#   KCD_INPUT_HW       e.g. "640,640"
#   KCD_TRAIN_POLICY   fixed | multiscale | multiscale_<lo>_<hi>
#   KCD_LR             head LR
#   KCD_BACKBONE_LR    backbone LR
#   KCD_USE_AMP        true|false
# Optional:
#   KCD_INIT_CHECKPOINT     auto-resolved from variant when unset
#   KCD_TRAIN_FROM_SCRATCH=1 to skip init checkpoint
#   KCD_TIME_LIMIT     slurm walltime (default 72:00:00 for 4+gpu, 48:00:00 otherwise)
#   KCD_CPUS_PER_TASK  sbatch --cpus-per-task (default scales with num_gpus)
#   KCD_MEM            sbatch --mem (default scales with num_gpus)
#   KCD_DEV_MOUNT_DEIMV2=1   mount host tpl/DEIMv2 over the image's copy
#   KCD_NCCL_DEBUG     0|1|verbose (default 1 = flight recorder only)
#   KCD_IMAGE          docker image tag
#   FOLLOW             1|0|auto (default auto = follow if stdout is a tty)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

: "${KCD_RUN_NAME:?_submit_train.sh: KCD_RUN_NAME must be exported by the wrapper}"
: "${KCD_SCHEME:?_submit_train.sh: KCD_SCHEME must be exported}"
: "${KCD_VARIANT:?_submit_train.sh: KCD_VARIANT must be exported}"
: "${KCD_NUM_GPUS:?_submit_train.sh: KCD_NUM_GPUS must be exported}"

LOG_DPATH="${LOG_DPATH:-$KCD_SLURM_LOG_DPATH}"
FOLLOW="${FOLLOW:-auto}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"

FOLLOW_SCRIPT="$KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py"
kcd_require_path "kit follow_job.py" "$FOLLOW_SCRIPT" || exit 1

mkdir -p "$LOG_DPATH"

# Scale slurm resource requests with GPU count if not overridden.
default_cpus=$(( 4 * KCD_NUM_GPUS ))
default_mem_gb=$(( 32 * KCD_NUM_GPUS ))
KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-$default_cpus}"
KCD_MEM="${KCD_MEM:-${default_mem_gb}G}"
if [ "$KCD_NUM_GPUS" -ge 4 ]; then
    KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-72:00:00}"
else
    KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-48:00:00}"
fi

job_name="$KCD_RUN_NAME"

# sbatch's --export uses commas as the entry separator, so any value
# containing a comma (KCD_CATEGORY_NAMES, KCD_INPUT_HW, KCD_TILE_SOURCE_SCALES,
# ...) silently truncates. Write the env to a sourceable file with
# shell-quoted values (printf %q), pass ONLY the file path via --export,
# then source it in the sbatch script.
ENV_FPATH="$LOG_DPATH/${job_name}.env"
: > "$ENV_FPATH"
# Snapshot every KCD_* var (exported or not). Wildcard (vs the old
# explicit whitelist) means: adding a new KCD_* knob in the launcher
# never requires also remembering to add it here. The KCD_ namespace
# is reserved for our config so this won't sweep in unrelated state.
# `compgen -v` (vs `-e`) covers vars that paths.sh set without an
# explicit `export` — defense against future regressions of that kind.
while IFS= read -r v; do
    val="${!v:-}"
    if [ -n "$val" ]; then
        printf 'export %s=%q\n' "$v" "$val" >> "$ENV_FPATH"
    fi
done < <(compgen -v | grep -E '^KCD_' | sort -u)
echo "[_submit_train.sh] wrote env to $ENV_FPATH ($(wc -l < "$ENV_FPATH") vars)"

sbatch_args=(
    --parsable
    --job-name="$job_name"
    --gres="gpu:${KCD_NUM_GPUS}"
    --cpus-per-task="$KCD_CPUS_PER_TASK"
    --mem="$KCD_MEM"
    --time="$KCD_TIME_LIMIT"
    --nodes=1
    --ntasks=1
    --output="$LOG_DPATH/%x-%j.out"
    --error="$LOG_DPATH/%x-%j.out"
    --export=ALL,KCD_ENV_FPATH="$ENV_FPATH"
    --chdir="$KCD_REPO_ROOT"
)
[ -n "$SLURM_PARTITION" ] && sbatch_args+=(--partition="$SLURM_PARTITION")
[ -n "$ACCOUNT" ]         && sbatch_args+=(--account="$ACCOUNT")

echo "Submitting training run $KCD_RUN_NAME ..." >&2
echo "  scheme=$KCD_SCHEME  variant=$KCD_VARIANT  gpus=$KCD_NUM_GPUS  epochs=${KCD_NUM_EPOCHS:-?}" >&2
echo "  walltime=$KCD_TIME_LIMIT  cpus=$KCD_CPUS_PER_TASK  mem=$KCD_MEM" >&2

jobid="$(sbatch "${sbatch_args[@]}" "$SCRIPT_DIR/_sbatch_train.sh")"
jobid="${jobid%%;*}"

stdout_fpath="$LOG_DPATH/${job_name}-${jobid}.out"
echo "  job:  $jobid" >&2
echo "  log:  $stdout_fpath" >&2

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
        echo "[follow detached] job $jobid still on slurm (state: $job_state)." >&2
        echo "  reattach: $reattach_cmd" >&2
        echo "  cancel:   scancel $jobid" >&2
    fi
    exit "$follow_rc"
fi

echo "$jobid"
echo "  reattach with: bash $SCRIPT_DIR/follow_job.sh $jobid" >&2
