#!/usr/bin/env bash
# Generation 3 -- fresh from COCO, bf16, decoupled augmentation schedule,
# batch sized to the cards.
#
# ## What killed gen002 (slurm job 489)
#
# Partway through epoch 1 the forward went all-NaN and never came back. Every
# loss component read exactly 0.0000 for the remaining 11 epochs, so the
# gradients were 0 and FINITE -- GradScaler never intervened, upstream's
# `math.isfinite(loss_value)` guard never fired, and the run trained happily to
# AP 0.000 while dumping an 819 MB NaN.pth every single step. It held 4 GPUs
# for ~11 hours producing nothing.
#
# Epoch 1 is where two unrelated things collided:
#
#   1. `train_policy=fixed` pins collate_fn.stop_epoch = 1, and at
#      `epoch == stop_epoch` DEIMv2 reloads model, optimizer, GradScaler and
#      EMA state from best_stg1.pth (det_solver.py:83-86).
#   2. The augmentation policy [1, 6, 11] turned Mosaic/ZoomOut/IoUCrop/
#      PhotometricDistort on for the first time at epoch 1.
#
# So the optimizer and fp16 loss scale were reset to their epoch-0 (NoAug)
# values at the exact step the input distribution changed. gen001 hit a NaN
# excursion at its own augmentation boundary too -- epoch 4 of [4, 78, 148] --
# but its stop_epoch reload was 3 epochs earlier, and it RECOVERED by epoch 5.
# Two runs, two NaN events, both in the first epoch with augmentation enabled.
#
# ## What changed
#
# 1. bf16 INSTEAD OF fp16. torch.autocast on CUDA defaults to float16 and
#    DEIMv2 never passed a dtype, so every kit run has trained at fp16, whose
#    ~65504 ceiling turns one activation excursion into inf and then NaN.
#    bfloat16 has float32's exponent range. Blackwell supports it natively at
#    the same memory cost. Set KCD_AMP_DTYPE=float16 to A/B against the old
#    behaviour.
#
# 2. THE COLLISION IS DECOUPLED. _aug_policy_epochs now refuses to place any
#    boundary on stop_epoch, so 24 epochs yields [2, 12, 23]: NoAug warmup,
#    the checkpoint reload alone at epoch 1, augmentation from epoch 2, and a
#    genuine NoAug final epoch. Fixes the sea-lion project too -- every
#    schedule there is short enough to have been colliding.
#
# 3. TRAINING ABORTS ON NaN. det_engine.py now raises instead of dumping and
#    continuing. A numerical failure costs minutes, not a weekend.
#
# 4. FRESH FROM COCO, NOT WARM-STARTED. gen002's best_stg2.pth (vali AP
#    0.5443) is the best model we have, but it descends from gen001, which
#    trained through a NaN excursion and a broken augmentation schedule.
#    Rather than carry unknown ill-conditioning forward into the run that is
#    meant to settle the question, gen003 starts from the COCO-pretrained
#    dinov3_x -- the same starting point as gen001, now with all three fixes.
#    KCD_INIT_CHECKPOINT is deliberately unset so kcd_resolve_init_checkpoint
#    picks $KCD_DEIMV2_DINOV3_X_COCO_PTH.
#
# 5. BATCH 6 -> 16 PER GPU (total 24 -> 64). gen001 measured ~16.5 GB of 96 GB
#    per GPU at batch 6 -- about 2.75 GB/sample -- so batch 16 lands near
#    44 GB with real headroom for eval and fragmentation. Batch 24 (~66 GB)
#    is reachable but leaves nothing to absorb a surprise.
#
# 6. LR 5e-4 / BACKBONE 1e-5. Upstream's tuned pair for this exact config
#    (configs/deimv2/deimv2_dinov3_x_coco.yml). gen001's 1e-3 drove a NaN
#    excursion at epoch 4 and the same value made sea-lion gen006 diverge
#    without recovering. The kit passes lr straight through to AdamW -- there
#    is no batch scaling anywhere in the stack -- so at batch 64 this is half
#    gen001's LR over 2.7x the batch, i.e. a much smaller per-sample step.
#
# 7. 24 EPOCHS. A 2.7x larger batch means 2.7x fewer optimizer steps per
#    epoch, so a longer schedule is needed to get the same number of updates.
#    gen001's core problem was never completing a schedule; 24 completes.
#
# ## Time budget
#
# gen001 ran clean at ~1.4 h/epoch at batch 6 (0.42 s/step x 10464 steps, plus
# ~10 min for the two evals DEIMv2 runs per epoch -- raw model and EMA). At
# batch 64 the epoch is 3,924 steps; expect ~1.1-1.4 h/epoch depending on how
# much the larger batch improves utilisation, so 24 epochs is ~27-34 h.
#
# CHECK AFTER EPOCH 1 and do the arithmetic before walking away. If epochs are
# coming in at 2.5 h+, that is the contention signature from job 296 and job
# 489's own 2-hour stall between steps 4000 and 4500 -- look for a stray vLLM
# server with `nvidia-smi` rather than assuming it is this config.
#
# Submit (from the kit root, on aiq-gpu, AFTER rebuilding the image):
#   bash projects/viame_fish_2026/scripts/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen003_bf16_fresh.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Fresh start: no init checkpoint of ours, and no resume. KCD_INIT_CHECKPOINT
# is left unset on purpose -- _launch_train.sh resolves the COCO-pretrained
# dinov3_x via kcd_resolve_init_checkpoint. KCD_RESUME_CKPT defaults to "auto",
# which would pick up a checkpoint from a re-run, so pin it.
kcd_require_init_checkpoint "deimv2_dinov3_x" || exit 1
export KCD_RESUME_CKPT="${KCD_RESUME_CKPT:-fresh}"

# ============================================================
# Hyperparameters
# ============================================================
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-16}"     # total = 64, was 24
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-24}"           # policy -> [2, 12, 23]
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_TRAIN_POLICY="${KCD_TRAIN_POLICY:-fixed}"
export KCD_LR="${KCD_LR:-5e-4}"                         # upstream's tuned pair
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-1e-5}"
export KCD_USE_AMP=true
export KCD_AMP_DTYPE="${KCD_AMP_DTYPE:-bfloat16}"       # set float16 to A/B

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
# forever. See the gen001 submit script for the full account.
export KCD_NCCL_BLOCKING_WAIT="${KCD_NCCL_BLOCKING_WAIT:-0}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

echo "gen003 fresh / bf16 / decoupled aug schedule"
echo "  init:      COCO pretrained (resolved by kcd_resolve_init_checkpoint)"
echo "  resume:    $KCD_RESUME_CKPT"
echo "  amp:       $KCD_AMP_DTYPE"
echo "  batch:     $KCD_PER_GPU_BATCH/gpu x $KCD_NUM_GPUS gpus = $(( KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))"
echo "  epochs:    $KCD_NUM_EPOCHS   lr: $KCD_LR (backbone $KCD_BACKBONE_LR)"
echo "  NOTE: needs an image built after the bf16 + aug-decoupling fixes."
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
