#!/usr/bin/env bash
# Generation 1: retry the FishTrack23 RF-DETR Large 720 segmentation recipe
# using VIAME v0.22.7-rc2 after the earlier v0.22.6 attempt hung.
#
# setup_config.sh selects the config. This run definition requires the bundled
# train_detector_rf_detr_l_seg_720.conf recipe and snapshots it into every
# attempt directory before training starts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/paths.sh"

export VF_RUN_NAME=fishtrack23_rfdetr_l_seg720_4gpu_viame0227_gen001
export VF_ENTRYPOINT_FPATH="${BASH_SOURCE[0]}"
export VF_CUDA_VISIBLE_DEVICES=0,1,2,3
export VF_INPUT_DPATH="${VF_INPUT_DPATH:-$VF_DATA_DPATH/Train}"
export VF_EXPECTED_CONFIG_NAME=train_detector_rf_detr_l_seg_720.conf

bash "$SCRIPT_DIR/check_setup.sh" || exit 1
exec bash "$SCRIPT_DIR/_launch_viame_train.sh"
