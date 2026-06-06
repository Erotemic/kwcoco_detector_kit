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
# DEFAULT MODE IS REPORT-ONLY. The janitor will identify zombies
# and print them, but will NOT kill anything unless you set
# KCD_KILL_ZOMBIES=1 explicitly. Killing is opt-in because a
# correctly-set-up training run should never produce zombies in
# the first place — if zombies appear, that's a signal worth
# investigating before papering over.
#
# Usage:
#   bash projects/viame_sealions_2026/scripts/kit_zombie_janitor.sh        # report only
#   KCD_KILL_ZOMBIES=1 bash .../kit_zombie_janitor.sh                      # report + kill
#
# Recommend running this:
#   - After a known scancel / walltime event, with KCD_KILL_ZOMBIES=1
#     once you've confirmed the trap-cleanup didn't fire.
#   - As a periodic SANITY CHECK (cron, every 15-30 min, no
#     KCD_KILL_ZOMBIES) — silent on no-op runs; if a zombie shows up,
#     the cron output reveals the bug to investigate.

set -u

KILL="${KCD_KILL_ZOMBIES:-0}"
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

if [ "$KILL" != "1" ]; then
    echo
    echo "(default mode: report only; set KCD_KILL_ZOMBIES=1 to actually kill these.)"
    echo "Zombies appearing here without a trap-cleanup having fired is a signal" >&2
    echo "worth investigating before just nuking — check _sbatch_train.sh's trap" >&2
    echo "and the slurm log for the orphaned job(s)." >&2
    exit 0
fi

echo
echo "kit_zombie_janitor: KCD_KILL_ZOMBIES=1 set; killing ${#TO_KILL[@]} zombie(s)..."
docker rm -f "${TO_KILL[@]}"
echo "kit_zombie_janitor: done."

# Post-cleanup GPU check.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "GPU state after cleanup:"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
fi
