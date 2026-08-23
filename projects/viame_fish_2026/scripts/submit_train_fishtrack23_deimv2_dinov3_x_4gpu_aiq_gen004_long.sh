#!/usr/bin/env bash
# Generation 4 -- the long fp16 run. Every lesson from gen001-003, one changed
# variable: schedule length.
#
# ## Why length, and not a bigger batch
#
# gen003 left hard telemetry (MetricLogger's "max mem", job 490):
#
#   peak allocated   37.8 GB of 96 per GPU at batch 16
#   working set       1.26 GB (model + EMA + optimizer)
#   activations       2.28 GB/sample
#   growth            36.9 GB at epoch 0 step 0 -> 37.8 GB by epoch 14, then
#                     flat. No spike when augmentation turned on at epoch 2
#                     (~100 MB), no fragmentation creep over 24 epochs.
#
# So VRAM is nowhere near binding -- batch 32/GPU would fit in ~77 GB. It is
# still the wrong lever. Throughput scales sub-linearly with batch while the
# step count scales inversely, so in a fixed 48 h:
#
#   total batch  img/s  epoch cycle  epochs   optimizer steps
#         24     57.1    78.8 min      36          377k
#         64     88.9    52.6 min      54          212k   <- gen003's batch
#        128    103.0    46.1 min      62          122k
#
# Doubling to 128 buys 16% more images/hour and HALVES the updates. And updates
# are what we appear to be short of: gen001 reached vali AP 0.5440 on ~136k
# steps, gen003 only 0.5406 on ~94k. gen003 was update-starved, not
# under-fed -- which also explains the AP plateau from epoch 6 onward.
#
# 48 epochs at batch 64 gives ~212k steps: 1.6x gen001, 2.3x gen003.
#
# ## What is inherited (do not re-litigate)
#
# 1. fp16, DEIMv2's native precision. It mentions bfloat16 nowhere in its
#    training path, builds a GradScaler unconditionally -- fp16 underflow
#    machinery, a vestige under bf16 -- and every published COCO number comes
#    from plain `--use-amp`. gen003's bf16 finished BELOW both fp16 runs and
#    gives up three mantissa bits in a model whose localization is
#    fine-grained-distribution based (reg_max=32, loss_fgl, loss_ddf).
#    Pinned explicitly so a future default change cannot silently alter this
#    run.
#
# 2. The augmentation boundary is kept off collate_fn.stop_epoch. At 48 epochs
#    the policy is [2, 25, 47] against stop_epoch=1 -- the checkpoint/optimizer
#    /GradScaler reload at epoch 1 happens alone, augmentation starts at 2, and
#    epoch 47 is a genuine NoAug finish. The collision is what killed gen002.
#
# 3. Training aborts on non-finite pred_boxes instead of dumping an 819 MB
#    NaN.pth every step and training on. A numerical failure now costs minutes.
#
# 4. Batch 16/GPU, lr 5e-4 / backbone 1e-5 -- unchanged from gen003, which ran
#    24 epochs with no excursion. flat_epoch is 24, so LR holds at 5e-4 for
#    half the run and then cosines down over the back half.
#
# 5. Fresh from COCO. This is also the missing control: no run has yet been
#    done with fp16 AND the collision fixed. Warm-starting from gen003 would
#    confound that and would restart a cosine on already-converged weights.
#
# ## Budget
#
# gen003 measured 54:49 per epoch end-to-end (47:27 train + eval + save).
# 48 epochs => ~43.8 h, inside 48 with ~4 h of slack. KCD_TIME_LIMIT is set
# well above that so a slow patch cannot get the job killed near the end;
# shorten KCD_NUM_EPOCHS rather than the walltime if the budget tightens.
#
# ## Before submitting
#
#   nvidia-smi        # a stray vLLM server explains job 296's 8.4x slow steps
#                     # and job 489's 2 h mid-epoch stall. Check, do not assume.
#
# ## While it runs
#
#   bash projects/viame_fish_2026/scripts/run_health.sh --num_epochs 48
#   bash projects/viame_fish_2026/scripts/run_health.sh --watch --num_epochs 48
#
# Submit (from the kit root, on aiq-gpu, AFTER rebuilding the image):
#   bash projects/viame_fish_2026/scripts/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen004_long.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

# Fresh from COCO: KCD_INIT_CHECKPOINT stays unset so _launch_train.sh resolves
# $KCD_DEIMV2_DINOV3_X_COCO_PTH. KCD_RESUME_CKPT defaults to "auto", which after
# any partial run would silently continue it, so pin it.
kcd_require_init_checkpoint "deimv2_dinov3_x" || exit 1
export KCD_RESUME_CKPT="${KCD_RESUME_CKPT:-fresh}"

# ============================================================
# Hyperparameters -- only KCD_NUM_EPOCHS differs from gen003
# ============================================================
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-4}"
export KCD_PER_GPU_BATCH="${KCD_PER_GPU_BATCH:-16}"     # total 64; 37.8/96 GB
export KCD_VAL_BATCH_MULT="${KCD_VAL_BATCH_MULT:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-48}"           # policy -> [2, 25, 47]
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
export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-60:00:00}"     # ~44 h expected
export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"
export KCD_TRAIN_NUM_WORKERS="${KCD_TRAIN_NUM_WORKERS:-8}"
export KCD_VAL_NUM_WORKERS="${KCD_VAL_NUM_WORKERS:-4}"

# Let the NCCL watchdog kill a stalled collective instead of hanging forever.
# TORCH_NCCL_BLOCKING_WAIT=1 is what turned job 293's 10-minute fault into a
# two-day one: a blocking wait is precisely what the async watchdog cannot
# preempt.
export KCD_NCCL_BLOCKING_WAIT="${KCD_NCCL_BLOCKING_WAIT:-0}"

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

echo "gen004 long fp16 run"
echo "  init:      COCO pretrained (resolved by kcd_resolve_init_checkpoint)"
echo "  resume:    $KCD_RESUME_CKPT"
echo "  amp:       $KCD_AMP_DTYPE"
echo "  batch:     $KCD_PER_GPU_BATCH/gpu x $KCD_NUM_GPUS = $(( KCD_PER_GPU_BATCH * KCD_NUM_GPUS ))  (~37.8 GB of 96 measured)"
echo "  epochs:    $KCD_NUM_EPOCHS   lr: $KCD_LR (backbone $KCD_BACKBONE_LR)"
echo "  expect:    ~44 h at gen003's measured 54:49/epoch"
echo "  NOTE: needs an image built after the fp16 revert and the collision fix."
echo

exec bash "$SCRIPT_DIR/_submit_train.sh"
