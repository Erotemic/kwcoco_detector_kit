#!/usr/bin/env bash
# Generation 2 -- warm start from gen001's epoch-12 weights, corrected schedule.
#
# ## Why a warm start rather than resuming gen001
#
# Resuming gen001 was tried (slurm job 296) and does not work. Its tensorboard
# events show the scheduler coming back at step 52,320 -- exactly 5 epochs, not
# 13 -- with Lr/pg_0 pinned at its undecayed base of 5.0e-05 for the whole 2.3 h
# it ran, while Loss/total climbed from 35.2 to 39.9. It was making the model
# worse, 8.4x slower than the original run (3.51 vs 0.42 s/step), and never
# reached an epoch boundary so nothing was checkpointed.
#
# A supporting clue that gen001 never had a clean resume point: `last.pth`,
# which DEIMv2 rewrites every epoch, is frozen at epoch 0. Checkpoint
# bookkeeping was already anomalous, so best_stg2.pth almost certainly does not
# carry consistent scheduler/step state.
#
# `--init_checkpoint` is a different, well-exercised path: it loads weights and
# starts a fresh schedule, which is exactly what we want.
#
# ## What changed from gen001
#
# 1. WARM START. Initialised from gen001's best checkpoint (vali AP 0.5440,
#    held-out test AP50 0.7272) rather than from COCO.
#
# 2. LOWER LR. 2e-4 instead of 1e-3. gen001's 1e-3 drove a NaN excursion at
#    epoch 4 that took an epoch to recover from, and that was from a COCO init.
#    Re-applying it to already-fine-tuned weights would be worse: a large LR on
#    a converged model destroys what it learned before it improves anything.
#    Backbone scaled in the same 1/20 ratio: 1e-5.
#
# 3. SHORTER SCHEDULE. 12 epochs, not 20. Warm-started from a near-converged
#    model, the value is in completing a schedule -- the cosine tail and the
#    clean final epochs -- not in raw epoch count. ~1.4 h/epoch => ~17 h.
#
# 4. THE AUGMENTATION FIX. gen001 emitted upstream's raw policy boundaries
#    [4, 78, 148], which assume a ~150-epoch run. At 20 epochs that made stages
#    3 and 4 unreachable: training entered the Mosaic stage at epoch 4 and never
#    left, so it never got the clean fine-tuning phase the recipe is built
#    around. trainers/deimv2.py now scales those boundaries by num_epochs, so
#    this run gets [1, 6, 11] -- warmup, Mosaic, mid, then a genuine NoAug final
#    epoch. THIS REQUIRES A REBUILT IMAGE; the kit is baked, not mounted.
#
# Batch size is deliberately unchanged at 6/GPU. gen001 measured only
# ~16.5 GB of 96 GB per GPU, so there is room for 16-24, but raising batch and
# lowering LR at the same time confounds the result. Take the schedule fix
# first; batch is a gen003 lever.
#
# Submit (from the kit root, on aiq-gpu, AFTER rebuilding the image):
#   bash projects/viame_fish_2026/scripts/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen002_warmstart.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

GEN001_WORKDIR="$KCD_RUNS_DPATH/fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001/runs/deimv2_dinov3_x_1024x1024_fixed"
export KCD_INIT_CHECKPOINT="${KCD_INIT_CHECKPOINT:-$GEN001_WORKDIR/best_stg2.pth}"
kcd_require_path "gen001 warm-start checkpoint" "$KCD_INIT_CHECKPOINT" || exit 1

# ============================================================
# Hyperparameters
# ============================================================
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-6}"      # total = 24, as gen001
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-12}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_TRAIN_POLICY="${KCD_TRAIN_POLICY:-fixed}"
export KCD_LR="${KCD_LR:-2e-4}"                          # was 1e-3
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-1e-5}"        # was 5e-5
export KCD_USE_AMP=true

# ============================================================
# Eval: whole-image, matching how the model trains.
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-False}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Slurm on aiq
# ============================================================
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# Let the NCCL watchdog kill a stalled collective after 600s instead of hanging
# forever. See the gen001 submit script for the full account -- job 293 sat in a
# spinning all-reduce for two days because TORCH_NCCL_BLOCKING_WAIT=1 prevents
# the async watchdog from preempting it.
export KCD_NCCL_BLOCKING_WAIT="${KCD_NCCL_BLOCKING_WAIT:-0}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

echo "gen002 warm start"
echo "  init:    $KCD_INIT_CHECKPOINT"
echo "  epochs:  $KCD_NUM_EPOCHS   lr: $KCD_LR (backbone $KCD_BACKBONE_LR)"
echo "  NOTE: needs an image built after the aug-policy fix in trainers/deimv2.py."
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
