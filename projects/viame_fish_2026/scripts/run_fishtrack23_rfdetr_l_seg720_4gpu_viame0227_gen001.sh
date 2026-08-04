#!/usr/bin/env bash
# Generation 1: retry the FishTrack23 RF-DETR Large 720 segmentation recipe
# using VIAME v0.22.7 after the v0.22.6 attempt hung.
#
# Experiment-defining choices belong in this file. For a changed config,
# binary, data subset, or hyperparameter, copy this file to gen002 instead of
# silently editing the historical run definition.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

export VF_RUN_NAME=fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001
export VF_ENTRYPOINT_FPATH="${BASH_SOURCE[0]}"
export VF_CUDA_VISIBLE_DEVICES=0,1,2,3
export VF_INPUT_DPATH="${VF_INPUT_DPATH:-$VF_DATA_DPATH/Train}"
export VF_PROJECT_CONFIG="$VF_PROJECT_DPATH/configs/train_detector_rf_detr_l_720_90gb.conf"

bash "$SCRIPT_DIR/check_setup.sh" || exit 1
exec bash "$SCRIPT_DIR/_launch_viame_train.sh"
