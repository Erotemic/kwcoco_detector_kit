#!/usr/bin/env bash
# Generation 4 -- gen003's shape with double the cosine tail.
#
# Supersedes the first gen004 attempt (slurm job 492, 48 epochs), which was
# killed at epoch 21 having never beaten its own epoch-1 score. Two things went
# wrong there, and both are fixed below rather than worked around.
#
# ## 1. The restore-the-best branch was restoring epoch 0
#
# DEIMv2 reloads a checkpoint after every non-improving eval
# (det_solver.py:213) -- model, optimizer, GradScaler, EMA and LR warmup, all
# of it. It reloads `best_stg1.pth`, which is only ever WRITTEN while
# `epoch < stop_epoch`. Upstream puts stop_epoch late, so for them that file
# accumulates the whole stage-1 phase and the reload means "go back to the
# best". The kit pins stop_epoch=1 for train_policy=fixed
# (trainers/deimv2.py:287), which freezes best_stg1.pth at EPOCH 0 forever.
#
# Job 492 took that reset at epochs 2, 7, 13 and 19. Its AP sawtoothed back to
# ~0.52 -- epoch-0 level -- after every one, and best_stg2.pth was never
# updated past epoch 1. gen003 survived the same mechanism only because its
# cosine phase improved monotonically, so the branch rarely fired.
#
# Fixed in the fork: reload best_stg2.pth (the global best, since top1 is
# monotonic) and fall back to best_stg1.pth only when it does not exist yet.
#
# ## 2. The flat-LR phase was half the run
#
# flat_epoch was hardcoded to num_epochs//2, so asking for 48 epochs bought 24
# epochs of CONSTANT LR -- precisely the regime where the model oscillates,
# trips the reload branch, and accumulates nothing. Every gain gen003 made came
# from its cosine tail.
#
# flat_epoch is now overridable via KCD_FLAT_EPOCH, so a longer schedule buys
# cosine epochs instead of flat ones:
#
#   gen003        24 epochs, flat 12 -> 12 cosine   vali 0.5406
#   gen004 (492)  48 epochs, flat 24 -> 21 flat epochs run, killed at 0.5342
#   THIS RUN      36 epochs, flat 12 -> 24 cosine   (same flat phase as gen003,
#                                                    twice the annealing tail)
#
# Vali only. The held-out test split does not inform schedule choices -- see
# the holdout-discipline note in paths.sh.
#
# Holding the flat phase at gen003's proven 12 epochs and spending the extra
# budget entirely on annealing is the smallest change consistent with where the
# gains actually came from.
#
# ## Everything else is deliberately gen003's configuration
#
#   fp16          DEIMv2's native precision. gen004/492 confirmed fp16 and bf16
#                 are indistinguishable through 8 epochs (max delta 0.004 AP,
#                 no consistent sign), so this costs nothing and matches the
#                 recipe the published numbers come from.
#   batch 16/GPU  peak 37.8 GB of 96 measured. Batch 32 would fit, but doubling
#                 it buys 16% more images/hour and halves the optimizer steps.
#   lr 5e-4/1e-5  upstream's tuned pair; 24 epochs with no excursion.
#   fresh COCO    also the missing control -- no completed run has yet combined
#                 fp16 with the collision fix.
#   policy        [2, 19, 35] against stop_epoch=1. No collision.
#
# ## Budget
#
# 36 epochs x 54:49 measured = ~32.9 h. Walltime is set well above that so a
# slow patch cannot kill the job near the end.
#
# ## Before submitting
#
#   nvidia-smi    # a stray vLLM server is still the leading explanation for
#                 # job 296's 8.4x slow steps and job 489's 2 h stall.
#
# ## While it runs
#
#   bash projects/viame_fish_2026/scripts/run_health.sh --num_epochs 36
#
# Watch for "Refresh EMA at epoch N ... (restored best_stg2.pth)" -- the new
# message confirms the fix is live. If it says best_stg1.pth after epoch 1, the
# image predates the fix.
#
# Submit (from the kit root, on aiq-gpu, AFTER rebuilding the image):
#   bash projects/viame_fish_2026/scripts/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen004_cosine.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Fresh from COCO: KCD_INIT_CHECKPOINT stays unset so _launch_train.sh resolves
# $KCD_DEIMV2_DINOV3_X_COCO_PTH. KCD_RESUME_CKPT defaults to "auto", which after
# a partial run would silently continue it, so pin it.
kcd_require_init_checkpoint "deimv2_dinov3_x" || exit 1
export KCD_RESUME_CKPT="${KCD_RESUME_CKPT:-fresh}"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-16}"     # total 64; 37.8/96 GB
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-36}"           # policy -> [2, 19, 35]
export KCD_FLAT_EPOCH="${KCD_FLAT_EPOCH:-12}"           # 24 cosine epochs
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_TRAIN_POLICY="${KCD_TRAIN_POLICY:-fixed}"
export KCD_LR="${KCD_LR:-5e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-1e-5}"
export KCD_USE_AMP=true
export KCD_AMP_DTYPE="${KCD_AMP_DTYPE:-float16}"        # DEIMv2's own recipe

# ============================================================
# Eval: whole-image, matching how the model trains.
# ============================================================
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-False}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

# ============================================================
# Slurm on aiq
# ============================================================
export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-48:00:00}"     # ~33 h expected
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# Let the NCCL watchdog kill a stalled collective instead of hanging forever.
export KCD_NCCL_BLOCKING_WAIT="${KCD_NCCL_BLOCKING_WAIT:-0}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

echo "gen004 -- gen003's shape, double the cosine tail"
echo "  init:      COCO pretrained (resolved by kcd_resolve_init_checkpoint)"
echo "  resume:    $KCD_RESUME_CKPT"
echo "  amp:       $KCD_AMP_DTYPE"
echo "  batch:     $KCD_PER_GPU_BATCH/gpu x $KCD_NUM_GPUS = $(( KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))"
echo "  epochs:    $KCD_NUM_EPOCHS  (flat $KCD_FLAT_EPOCH, cosine $(( KCD_NUM_EPOCHS - KCD_FLAT_EPOCH )))"
echo "  lr:        $KCD_LR (backbone $KCD_BACKBONE_LR)"
echo "  expect:    ~33 h at gen003's measured 54:49/epoch"
echo "  NOTE: needs an image built after the best_stg2 reload fix."
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
