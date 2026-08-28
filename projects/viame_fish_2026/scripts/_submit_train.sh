#!/usr/bin/env bash
# Internal boilerplate. Submit a fish training run to slurm on aiq-gpu.
# Called from the descriptive submit_train_*.sh wrappers, never directly.
#
# Contract: the wrapper must export these before sourcing this script:
#   KCD_RUN_NAME       experiment id; slurm job name + KCD_ROOT
#   KCD_VARIANT        DEIMv2 variant
#   KCD_NUM_GPUS       1..N
#   KCD_PER_GPU_BATCH  per-GPU batch (total = NUM_GPUS * PER_GPU_BATCH)
#   KCD_NUM_EPOCHS
#   KCD_INPUT_HW       e.g. "[1024, 1024]"
#   KCD_LR / KCD_BACKBONE_LR
# Optional: KCD_TIME_LIMIT, KCD_CPUS_PER_TASK, KCD_MEM, KCD_IMAGE, FOLLOW.
#
# ## Why this reaches into the sea-lion project
#
# `_sbatch_train.sh` is genuinely project-agnostic already: it derives the
# config from $KCD_REPO_ROOT (which paths.sh to source) and $KCD_LAUNCH_SCRIPT
# (which launcher to run), and everything else in it is host hardening we very
# much want -- per-job GPU pinning by UUID, zombie-container cleanup traps,
# GPU-leak detection, shm sizing. Reimplementing ~380 lines of that for fish
# would mean maintaining two copies of the part most likely to bite us.
#
# It happens to live under projects/viame_sealions_2026/scripts/. Promoting it
# to a shared location is the right cleanup, but that edits the sea-lion
# project, so it is left as a follow-up rather than done as a side effect of
# this run. Override KCD_SHARED_SBATCH if it moves.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

: "${KCD_RUN_NAME:?_submit_train.sh: KCD_RUN_NAME must be exported by the wrapper}"
: "${KCD_VARIANT:?_submit_train.sh: KCD_VARIANT must be exported}"
: "${KCD_NUM_GPUS:?_submit_train.sh: KCD_NUM_GPUS must be exported}"

KCD_SHARED_SBATCH="${KCD_SHARED_SBATCH:-$KCD_KIT_DPATH/projects/viame_sealions_2026/scripts/_sbatch_train.sh}"
kcd_require_path "shared _sbatch_train.sh" "$KCD_SHARED_SBATCH" || exit 1

# Make the shared sbatch script load THIS project's paths.sh and launcher.
export KCD_REPO_ROOT="$VF_PROJECT_DPATH"
# Respect a wrapper's choice of launcher so the same submit/sbatch/docker
# machinery can drive a non-training job (e.g. _launch_export_score.sh).
export KCD_LAUNCH_SCRIPT="${KCD_LAUNCH_SCRIPT:-_launch_train.sh}"

# Fail on the host, before paying for a job start or a container start.
kcd_require_init_checkpoint "$KCD_VARIANT" || exit 1
kcd_require_train_inputs || exit 1

# No-slurm hosts run the job directly via `docker run` in the foreground.
# aiq-gpu has no slurm at all -- no sbatch binary, slurmd/slurmctld inactive --
# so without this branch every submit_train_*.sh here dies on `sbatch: command
# not found` AFTER passing all its preflight checks. The sea-lion project's
# _submit_train.sh has had this branch; the fish one did not.
#
# _run_standalone.sh is shared exactly like _sbatch_train.sh, and honours the
# KCD_REPO_ROOT exported above, so it loads THIS project's paths.sh.
KCD_SHARED_STANDALONE="${KCD_SHARED_STANDALONE:-$KCD_KIT_DPATH/projects/viame_sealions_2026/scripts/_run_standalone.sh}"
if [ "${KCD_NO_SLURM:-0}" = "1" ]; then
    kcd_require_path "shared _run_standalone.sh" "$KCD_SHARED_STANDALONE" || exit 1
    exec bash "$KCD_SHARED_STANDALONE"
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found and KCD_NO_SLURM is not 1." >&2
    echo "  This host has no slurm. Re-run with KCD_NO_SLURM=1 to launch" >&2
    echo "  directly via docker, or set it in the submit_*.sh wrapper." >&2
    exit 1
fi

LOG_DPATH="${LOG_DPATH:-$KCD_SLURM_LOG_DPATH}"
FOLLOW="${FOLLOW:-auto}"
mkdir -p "$LOG_DPATH"

default_cpus=$(( 8 * KCD_NUM_GPUS ))
default_mem_gb=$(( 48 * KCD_NUM_GPUS ))
KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-$default_cpus}"
KCD_MEM="${KCD_MEM:-${default_mem_gb}G}"
KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-72:00:00}"

job_name="$KCD_RUN_NAME"

# sbatch --export uses commas as its entry separator, so any value containing
# a comma (KCD_INPUT_HW, KCD_CATEGORY_NAMES) would silently truncate. Write a
# sourceable env file with shell-quoted values and pass only its path.
ENV_FPATH="$LOG_DPATH/${job_name}.env"
: > "$ENV_FPATH"
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
    --gres="${KCD_GRES:-gpu:${KCD_NUM_GPUS}}"
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
[ -n "${SLURM_PARTITION:-}" ] && sbatch_args+=(--partition="$SLURM_PARTITION")
[ -n "${ACCOUNT:-}" ]         && sbatch_args+=(--account="$ACCOUNT")
if [ -n "${KCD_DEPENDS_ON:-}" ]; then
    if [[ "$KCD_DEPENDS_ON" == *:* ]]; then
        dep_arg="$KCD_DEPENDS_ON"
    else
        dep_arg="afterok:$(echo "$KCD_DEPENDS_ON" | tr ' ' ':')"
    fi
    sbatch_args+=(--dependency="$dep_arg")
fi

echo "Submitting fish training run $KCD_RUN_NAME ..." >&2
echo "  variant=$KCD_VARIANT  gpus=$KCD_NUM_GPUS  epochs=${KCD_NUM_EPOCHS:-?}  input_hw=${KCD_INPUT_HW:-?}" >&2
echo "  walltime=$KCD_TIME_LIMIT  cpus=$KCD_CPUS_PER_TASK  mem=$KCD_MEM" >&2

jobid="$(sbatch "${sbatch_args[@]}" "$KCD_SHARED_SBATCH")"
jobid="${jobid%%;*}"

stdout_fpath="$LOG_DPATH/${job_name}-${jobid}.out"
echo "  job:  $jobid" >&2
echo "  log:  $stdout_fpath" >&2

if [ "$FOLLOW" = "auto" ]; then
    if [ -t 1 ]; then FOLLOW=1; else FOLLOW=0; fi
fi

FOLLOW_SCRIPT="$KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py"
if { [ "$FOLLOW" = "1" ] || [ "$FOLLOW" = "true" ]; } && [ -f "$FOLLOW_SCRIPT" ]; then
    set +e
    python3 "$FOLLOW_SCRIPT" "$jobid" --stdout "$stdout_fpath"
    follow_rc=$?
    job_state="$(squeue -h -j "$jobid" -o '%T' 2>/dev/null | head -n1 || true)"
    set -e
    if [ -n "$job_state" ]; then
        echo >&2
        echo "[follow detached] job $jobid still on slurm (state: $job_state)." >&2
        echo "  reattach: tail -f $stdout_fpath" >&2
    fi
    exit "$follow_rc"
fi

echo "$jobid"
echo "  follow with: tail -f $stdout_fpath" >&2
