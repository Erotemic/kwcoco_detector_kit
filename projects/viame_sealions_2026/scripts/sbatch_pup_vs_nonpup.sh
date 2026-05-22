#!/usr/bin/env bash
# Slurm wrapper around launch_pup_vs_nonpup_arisia.sh.
#
# Submits a single 4xGPU job that runs the launch script inside the
# kwcoco-detector-kit docker image so the full toolchain (torch+CUDA,
# DEIMv2 submodule, kit dependencies, baked tests) is reproducible
# regardless of the compute node's host environment.
#
# Submit:
#     sbatch scripts/sbatch_pup_vs_nonpup.sh
#
# Override slurm + image config via env (set in your shell or `sbatch
# --export=ALL,VAR=value ...`):
#
#   KCD_IMAGE        docker image tag (default: kwcoco-detector-kit:ogdino-cu132-arisia)
#   KCD_PARTITION    slurm partition (default: arisia's default queue)
#   KCD_GRES         GPU resource string (default: gpu:4)
#   KCD_TIME_LIMIT   walltime (default: 72:00:00)
#   KCD_NUM_EPOCHS   epochs (default: 30)
#   KCD_PER_GPU_BATCH per-GPU batch size (default: 16)

#SBATCH --job-name=sealion-pup-nonpup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Under slurm, ${BASH_SOURCE[0]} points at slurm's staging copy
# (/var/lib/slurm-llnl/slurmd/jobNNN/slurm_script), so we can't use it
# to find paths.sh. submit_pup_vs_nonpup.sh forwards $KCD_REPO_ROOT
# via --export; fall back to $SLURM_SUBMIT_DIR (slurm's canonical name
# for the original submission directory) when invoked directly.
if [ -z "${KCD_REPO_ROOT:-}" ]; then
    KCD_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
source "$KCD_REPO_ROOT/scripts/paths.sh"

KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-arisia}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-16}"

echo "=== Slurm context ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<manual>}"
echo "HOSTNAME=$(hostname)"
echo "REPO=$KCD_REPO_ROOT"
echo "IMAGE=$KCD_IMAGE"
nvidia-smi -L || true
echo

# All four GPUs allocated by --gres become visible to the docker run
# via --gpus all (docker reads slurm's CUDA_VISIBLE_DEVICES). Mount
# the user's data tree so the kit can see kwcoco files, pretrained
# checkpoints, and write training output.
docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=32g \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    -e NUM_EPOCHS="$NUM_EPOCHS" \
    -e KCD_PER_GPU_BATCH="$KCD_PER_GPU_BATCH" \
    -e SLURM_JOB_ID="${SLURM_JOB_ID:-manual}" \
    -v "$KCD_DATA_ROOT:$KCD_DATA_ROOT" \
    -w "$KCD_REPO_ROOT" \
    "$KCD_IMAGE" \
    bash scripts/launch_pup_vs_nonpup_arisia.sh

echo
echo "Done. Output under: $KCD_ROOT_PUP_VS_NONPUP"
