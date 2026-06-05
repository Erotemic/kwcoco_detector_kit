#!/usr/bin/env bash
# Internal boilerplate. Submit a tile-build job to slurm.
# Called from submit_build_tiles.sh — not invoked directly.
#
# Differs from _submit_train.sh:
#   * required vars list is smaller (no scheme/variant)
#   * defaults to KCD_GRES=none (no GPU needed)
#   * forces KCD_LAUNCH_SCRIPT=_launch_tiles.sh
#   * default walltime is shorter (4h vs 48h)
#
# Reuses _sbatch_train.sh as the on-compute-node entrypoint — that
# script's docker boilerplate (NCCL flags, dev mounts, leak watchdog)
# is harmless when KCD_LAUNCH_SCRIPT routes to _launch_tiles.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

: "${KCD_RUN_NAME:?_submit_tiles.sh: KCD_RUN_NAME must be exported}"

LOG_DPATH="${LOG_DPATH:-$KCD_SLURM_LOG_DPATH}"
FOLLOW="${FOLLOW:-auto}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"

FOLLOW_SCRIPT="$KCD_KIT_DPATH/smoketests/dino_v2_4x/slurm/follow_job.py"
kcd_require_path "kit follow_job.py" "$FOLLOW_SCRIPT" || exit 1

mkdir -p "$LOG_DPATH"

# Tile build is CPU + I/O bound; no GPU. Modest defaults; users can
# override per-host (arisia has 64+ cores; namek has 16).
KCD_GRES="${KCD_GRES:-none}"
KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-16}"
KCD_MEM="${KCD_MEM:-64G}"
KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-08:00:00}"

# Dummy KCD_NUM_GPUS so _sbatch_train.sh's pre-checks pass. The
# KCD_GRES=none above overrides the actual sbatch GPU request.
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-0}"

# Route _sbatch_train.sh to the tile entrypoint.
export KCD_LAUNCH_SCRIPT=_launch_tiles.sh

# Dummy values for vars _sbatch_train.sh references but _launch_tiles.sh
# ignores. Setting them here so we never silently use undef-var defaults.
export KCD_SCHEME="${KCD_SCHEME:-universal}"
export KCD_VARIANT="${KCD_VARIANT:-tiles}"

job_name="$KCD_RUN_NAME"

# Write KCD_* env to a sourceable file (same pattern as _submit_train.sh)
ENV_FPATH="$LOG_DPATH/${job_name}.env"
: > "$ENV_FPATH"
while IFS= read -r v; do
    val="${!v:-}"
    if [ -n "$val" ]; then
        printf 'export %s=%q\n' "$v" "$val" >> "$ENV_FPATH"
    fi
done < <(compgen -v | grep -E '^KCD_' | sort -u)
echo "[_submit_tiles.sh] wrote env to $ENV_FPATH ($(wc -l < "$ENV_FPATH") vars)"

sbatch_args=(
    --parsable
    --job-name="$job_name"
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
if [ "$KCD_GRES" != "none" ]; then
    sbatch_args+=(--gres="$KCD_GRES")
fi
[ -n "$SLURM_PARTITION" ] && sbatch_args+=(--partition="$SLURM_PARTITION")
[ -n "$ACCOUNT" ]         && sbatch_args+=(--account="$ACCOUNT")

if [ -n "${KCD_DEPENDS_ON:-}" ]; then
    if [[ "$KCD_DEPENDS_ON" == *:* ]]; then
        dep_arg="$KCD_DEPENDS_ON"
    else
        dep_arg="afterok:$(echo "$KCD_DEPENDS_ON" | tr ' ' ':')"
    fi
    sbatch_args+=(--dependency="$dep_arg")
    echo "  depends-on: $dep_arg" >&2
fi

echo "Submitting tile-build job $KCD_RUN_NAME ..." >&2
echo "  walltime=$KCD_TIME_LIMIT  cpus=$KCD_CPUS_PER_TASK  mem=$KCD_MEM  gres=$KCD_GRES" >&2
echo "  source:  $KCD_UNIVERSAL_TRAIN_KWCOCO" >&2
echo "  cache:   $KCD_TILE_CACHE_DPATH" >&2

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
