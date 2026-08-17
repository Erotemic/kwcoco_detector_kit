#!/usr/bin/env bash
# Resume gen001 from its epoch-12 checkpoint and finish epochs 13-19.
#
# gen001 deadlocked in an NCCL all-reduce at epoch 13 and was lost. The
# checkpoint it left carries full training state -- `optimizer`, `last_epoch`,
# `ema`, `scaler`, `lr_warmup_scheduler` -- so this is a genuine continuation of
# the same schedule, not a fine-tune of the weights.
#
# ## Why this needs no trickery
#
# `sweep` refuses to skip training when the workdir has a checkpoint but no
# `.train_complete` marker, and retrains instead. That is exactly what we want
# here: with --resume set, "retraining" IS the continuation. The guard and the
# intent agree, so nothing has to be faked. On completion the marker is written
# and the normal eval/export stages run.
#
# ## What changed from the original run
#
# KCD_NCCL_BLOCKING_WAIT=0 (inherited from the gen001 submit script) so that a
# repeat of the same stall aborts on the 600s heartbeat instead of hanging
# indefinitely -- worth ~10 minutes of loss rather than two days.
#
# Everything else is deliberately identical to gen001, because the point is to
# finish that run's schedule, not to start a different experiment. Batch size
# and LR in particular are unchanged: the measured 16.5 GB/GPU leaves room for
# a much larger batch, but changing it mid-schedule would invalidate the
# comparison with epochs 0-12, and LR=1e-3 already produced a NaN excursion at
# epoch 4. Those belong in gen002.
#
# Submit (from the kit root, on aiq-gpu):
#   bash projects/viame_fish_2026/scripts/submit_resume_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

RUN_NAME="fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001"
WORKDIR="$KCD_RUNS_DPATH/$RUN_NAME/runs/deimv2_dinov3_x_1024x1024_fixed"

# Resume from the best checkpoint on disk. best_stg2.pth is epoch 12
# (vali AP 0.5440); last.pth stopped being updated at epoch 0 in this run, so
# it is NOT the right resume point despite the name.
export KCD_RESUME_CKPT="${KCD_RESUME_CKPT:-$WORKDIR/best_stg2.pth}"
kcd_require_path "resume checkpoint" "$KCD_RESUME_CKPT" || exit 1

echo "Resuming $RUN_NAME from:"
echo "  $KCD_RESUME_CKPT"
echo "  (sweep will report 'no completion marker -> retraining'; with --resume"
echo "   set that is the continuation, not a restart from scratch.)"
echo

# Delegate to the gen001 submit script so every hyperparameter, the NCCL fix,
# and the run identity come from one place. KCD_RESUME_CKPT is already
# exported, and _launch_train.sh passes it through as --resume; pareto_sweep
# then ignores init_checkpoint, since the resume checkpoint already carries the
# fine-tuned weights.
exec bash "$SCRIPT_DIR/submit_train_fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen001.sh"
