#!/usr/bin/env bash
# Canonical paths for the VIAME fish detector project.
# Override any VF_* variable in the calling shell when a host differs.

export VF_PROJECT_DPATH="${VF_PROJECT_DPATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export VF_WORK_DPATH="${VF_WORK_DPATH:-/data/users/${USER}/fish}"
export VF_DOWNLOAD_DPATH="${VF_DOWNLOAD_DPATH:-$VF_WORK_DPATH/downloads}"
export VF_SOFTWARE_DPATH="${VF_SOFTWARE_DPATH:-$VF_WORK_DPATH/software}"
export VF_RUNS_DPATH="${VF_RUNS_DPATH:-$VF_WORK_DPATH/runs}"
export VF_LOGS_DPATH="${VF_LOGS_DPATH:-$VF_WORK_DPATH/logs}"

export VF_DATA_SOURCE="${VF_DATA_SOURCE:-numenor:/data/Public/NOAA/FishTrack23-Latest/}"
export VF_DATA_DPATH="${VF_DATA_DPATH:-$HOME/ssd-data/FishTrack23-Latest}"
export VF_INPUT_DPATH="${VF_INPUT_DPATH:-$VF_DATA_DPATH/Train}"

export VF_VIAME_VERSION="${VF_VIAME_VERSION:-0.22.7}"
export VF_VIAME_ARCHIVE="${VF_VIAME_ARCHIVE:-$VF_DOWNLOAD_DPATH/VIAME-v${VF_VIAME_VERSION}-Linux-64Bit.tar.gz}"
export VF_CURRENT_VIAME_LINK="${VF_CURRENT_VIAME_LINK:-$VF_WORK_DPATH/viame-current}"

export VF_CONFIG_NAME="${VF_CONFIG_NAME:-train_detector_rf_detr_l_720_90gb.conf}"
export VF_PROJECT_CONFIG="${VF_PROJECT_CONFIG:-$VF_PROJECT_DPATH/configs/$VF_CONFIG_NAME}"
