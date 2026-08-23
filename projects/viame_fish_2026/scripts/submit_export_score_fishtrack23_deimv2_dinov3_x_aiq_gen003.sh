#!/usr/bin/env bash
# Export + score gen003's finished checkpoint.
#
# gen003 trained all 24 epochs cleanly (`.train_complete` is written,
# best_stg2.pth holds epoch 21 at vali AP 0.5406) and then died on the FIRST
# image of the scoring pass:
#
#   predict_image -> scores[0].cpu().numpy()
#   TypeError: Got unsupported ScalarType BFloat16
#
# numpy has float16 but no bfloat16, so every .numpy() downstream of the eval
# autocast worked for exactly as long as that autocast was fp16. Fixed by
# casting floating outputs back to float32 in DEIMv2Predictor._forward. This
# script re-runs only the scoring; no training is repeated.
#
# REQUIRES AN IMAGE REBUILT AFTER THAT FIX. Without one, override the eval
# precision instead -- both avoid bf16 reaching numpy in the current image:
#
#   KCD_EVAL_AMP=0   ... fp32 eval, numerically exact, ~2x slower
#   KCD_AMP_DTYPE=float16 ... what every run before gen003 used
#
# _launch_export_score.sh detects a complete predictions file and skips
# inference (725304d), so a re-run after a mid-eval failure is cheap.
#
# One GPU and a short walltime -- inference over 33,434 test images plus an
# ONNX trace, not training.
#
# Submit (from the kit root, on aiq-gpu):
#   bash projects/viame_fish_2026/scripts/submit_export_score_fishtrack23_deimv2_dinov3_x_aiq_gen003.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The run whose checkpoint we are exporting. KCD_ROOT is derived from this, so
# it must match the training run's name exactly.
export KCD_RUN_NAME="${KCD_RUN_NAME:-fishtrack23_deimv2_dinov3_x_4gpu_aiq_gen003_bf16_fresh}"

# Drive the export/score launcher instead of the training one, reusing the
# same sbatch + docker + GPU-pinning machinery.
export KCD_LAUNCH_SCRIPT=_launch_export_score.sh

# _submit_train.sh's contract still applies (it pre-flights the bundles and the
# init checkpoint), so these have to be set even though nothing trains. They
# mirror gen003's training values so the regenerated config matches the
# checkpoint's architecture.
export KCD_VARIANT=deimv2_dinov3_x
export KCD_CATEGORY_NAMES=fish
export KCD_NUM_GPUS="${KCD_NUM_GPUS:-1}"
export KCD_NUM_EPOCHS="${KCD_NUM_EPOCHS:-24}"
export KCD_INPUT_HW="${KCD_INPUT_HW:-[1024, 1024]}"
export KCD_TRAIN_POLICY="${KCD_TRAIN_POLICY:-fixed}"
export KCD_LR="${KCD_LR:-5e-4}"
export KCD_BACKBONE_LR="${KCD_BACKBONE_LR:-1e-5}"

# Pinned to bfloat16 because gen003 is the one run that TRAINED in bf16. The
# kit default has since gone back to fp16 (DEIMv2's own recipe), so this has
# to be explicit to score this checkpoint the way it was trained.
# Overridable per the header when running against an image that predates the
# float32 cast in _forward.
export KCD_AMP_DTYPE="${KCD_AMP_DTYPE:-bfloat16}"

# Whole-image eval, matching how the model was trained. Tiled eval would
# measure something the model never saw.
export KCD_TILED_EVAL="${KCD_TILED_EVAL:-False}"
export KCD_EVAL_DEVICE="${KCD_EVAL_DEVICE:-cuda}"

export KCD_TIME_LIMIT="${KCD_TIME_LIMIT:-08:00:00}"
export KCD_CPUS_PER_TASK="${KCD_CPUS_PER_TASK:-16}"
export KCD_MEM="${KCD_MEM:-64G}"

export KCD_NO_SLURM="${KCD_NO_SLURM:-0}"
export KCD_DOCKER_GPU_MODE="${KCD_DOCKER_GPU_MODE:-gpus}"
export KCD_IMAGE="${KCD_IMAGE:-kwcoco-detector-kit:ogdino-cu132-aiq}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
