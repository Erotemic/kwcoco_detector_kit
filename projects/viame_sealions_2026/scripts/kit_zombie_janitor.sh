#!/usr/bin/env bash
# Catch-all cleanup for kit containers whose slurm job is gone.
#
# The in-trap cleanup (_sbatch_train.sh) can lose the race against
# SIGKILL when slurm enforces walltime: bash dies before "docker
# stop" returns, leaving the container running orphaned in the
# daemon. This janitor sweeps after the fact.
#
# Logic:
#   1. Find every running container whose name starts with "kcd-"
#      OR whose kcd.user label matches the current user.
#   2. Extract its slurm job id (from --label kcd.slurm_job_id, or
#      from the kcd-<jobid>-<run> name pattern).
#   3. If the slurm job is no longer in `squeue`, the container is
#      a zombie -- kill it.
#
# Safety: only cleans up the current user's containers (via
# kcd.user label OR /proc inspection). Skips containers without a
# resolvable slurm job id (manual runs, foreign containers).
#
# Usage:
#   bash projects/viame_sealions_2026/scripts/kit_zombie_janitor.sh        # clean
#   KCD_DRY_RUN=1 bash .../kit_zombie_janitor.sh                           # just print
#
# Recommend running this:
#   - Periodically (cron, every 15-30 min)
#   - Before rsync_from_arisia.sh (avoid syncing zombie writes)
#   - After a known scancel / walltime event

set -u

DRY_RUN="${KCD_DRY_RUN:-0}"
MY_USER="$(whoami)"

# Find candidates: container name starts with "kcd-" OR has our
# user label. The label-based check catches containers from a
# future naming scheme; the name-based check catches today's runs.
mapfile -t CANDIDATES < <(
    {
        docker ps --filter "name=kcd-" --format '{{.ID}}|{{.Names}}'
        docker ps --filter "label=kcd.user=$MY_USER" --format '{{.ID}}|{{.Names}}'
    } | sort -u
)

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
    echo "kit_zombie_janitor: no kit containers found."
    exit 0
fi

declare -a ZOMBIES=()
declare -a LIVE=()
declare -a SKIPPED=()

for line in "${CANDIDATES[@]}"; do
    cid="${line%%|*}"
    name="${line##*|}"

    # Prefer the explicit label (set by --label kcd.slurm_job_id).
    sjid="$(docker inspect "$cid" --format '{{index .Config.Labels "kcd.slurm_job_id"}}' 2>/dev/null)"
    if [ -z "$sjid" ] || [ "$sjid" = "<no value>" ]; then
        # Fallback: parse kcd-<jobid>-<run> name.
        sjid="$(echo "$name" | sed -n 's/^kcd-\([0-9][0-9]*\)-.*$/\1/p')"
    fi

    if [ -z "$sjid" ] || [ "$sjid" = "manual" ]; then
        SKIPPED+=("$cid $name (no slurm job id - manual or foreign run)")
        continue
    fi

    # squeue with -h -j <id> is silent if the job isn't queued/running.
    if squeue -h -j "$sjid" 2>/dev/null | grep -q .; then
        LIVE+=("$cid $name slurm=$sjid")
    else
        ZOMBIES+=("$cid|$name|$sjid")
    fi
done

if [ "${#LIVE[@]}" -gt 0 ]; then
    echo "kit_zombie_janitor: LIVE containers (slurm job still queued/running):"
    for s in "${LIVE[@]}"; do echo "  $s"; done
fi
if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo "kit_zombie_janitor: SKIPPED (no slurm id):"
    for s in "${SKIPPED[@]}"; do echo "  $s"; done
fi

if [ "${#ZOMBIES[@]}" -eq 0 ]; then
    echo "kit_zombie_janitor: no zombies found."
    exit 0
fi

echo
echo "kit_zombie_janitor: ZOMBIES (slurm job gone):"
declare -a TO_KILL=()
for z in "${ZOMBIES[@]}"; do
    cid="${z%%|*}"
    rest="${z#*|}"
    name="${rest%%|*}"
    sjid="${rest##*|}"
    age="$(docker inspect "$cid" --format '{{.State.StartedAt}}' 2>/dev/null)"
    echo "  $cid  $name  slurm=$sjid  started=$age"
    TO_KILL+=("$cid")
done

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "(KCD_DRY_RUN=1 set; not actually killing. Unset and re-run to clean.)"
    exit 0
fi

echo
echo "kit_zombie_janitor: killing ${#TO_KILL[@]} zombie(s)..."
docker rm -f "${TO_KILL[@]}"
echo "kit_zombie_janitor: done."

# Post-cleanup GPU check.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "GPU state after cleanup:"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
fi
