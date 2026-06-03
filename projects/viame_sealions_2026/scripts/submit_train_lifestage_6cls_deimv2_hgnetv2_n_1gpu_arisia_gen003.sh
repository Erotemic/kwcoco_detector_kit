#!/usr/bin/env bash
# Generation 3 — fixes the empty-tile filter that broke gen002.
#
#   scheme:   single_sealion (P0)
#   variant:  deimv2_hgnetv2_n
#   gpus:     1 (arisia)
#   gen:      003
#
# What changed from gen002 (single variable at a time):
#
#   1. KCD_WDS_SKIP_EMPTY=0 (the new default in DEIMv2 e7a6c57).
#      gen002 silently filtered samples whose annotations were empty
#      after scheme collapse, causing each positive tile to be
#      oversampled ~38x per epoch and zero true-background gradient
#      signal. gen002 single_sealion got kit AP 0.024 vs v5's 0.177
#      (journal 2026-06-01). gen003 keeps empties so the stream
#      composition matches v5's gen001 corpus (78.6% empty / 21.4%
#      positive, naturally).
#
#   2. KCD_WDS_SHARDS_DPATH points at the SSD-mounted shards. Same
#      content as gen002's HDD shards, just on faster storage (135 GB
#      free, 27 GB needed). With WDS streaming + page cache the speed
#      difference is small for steady-state training, but the cold-
#      start epoch 0 should be noticeably faster.
#
# Everything else (model, LR, batch, augmentation, schedule) matches
# gen002 so the change in result is attributable to the data fix.
#
# Reproducibility note: experiment-defining knobs are SET HERE, not
# passed via the command line. Only performance knobs that don't
# affect results (KCD_TRAIN_NUM_WORKERS, KCD_CPUS_PER_TASK, KCD_MEM)
# may be overridden when submitting.
#
# Submit:
#   bash projects/viame_sealions_2026/scripts/submit_train_single_sealion_deimv2_hgnetv2_n_1gpu_arisia_gen003.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters — match single_sealion_gen002 exactly
# ============================================================
export KCD_SCHEME=lifestage_6cls
export KCD_CATEGORY_NAMES=bull,subadult_male,female,juvenile,pup,northern_fur_seal
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=12
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=4.9e-4
export KCD_BACKBONE_LR=2.45e-4
export KCD_USE_AMP=true

# ============================================================
# Tile params — unchanged from gen002 (the bundle is shared)
# ============================================================
export KCD_TILE_SIZE=320
export KCD_TILE_SOURCE_SCALES=1.0
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

export KCD_USE_WEBDATASET=1

# ============================================================
# gen003-specific knobs — these define the experiment
# ============================================================
# Empty tiles flow through as legitimate negative samples; matches
# v5 gen001 distribution. The new DEIMv2 default is already 0, but
# set it explicitly here so the script alone tells the whole story
# (env-state independence).
export KCD_WDS_SKIP_EMPTY=0

# Shards on SSD. Same content as the HDD copy, just faster storage.
# Mirror-copy required before first submit:
#   mkdir -p /data/users/jon.crall/kcd_sealion/ssd-data/tile_cache/_universal
#   cp -a /data/users/jon.crall/kcd_sealion/tile_cache/_universal/fd353b1c \
#         /data/users/jon.crall/kcd_sealion/ssd-data/tile_cache/_universal/
# Bind-mount the SSD target into the docker container. Without this,
# the symlink at $KCD_WDS_SHARDS_DPATH resolves to a path that does
# not exist inside the container (/home/local/KHQ is not mounted),
# and `mkdir -p` fails with EEXIST on the symlink itself.
export KCD_EXTRA_MOUNTS=/home/local/KHQ/jon.crall/ssd-data

export KCD_WDS_SHARDS_DPATH=/data/users/jon.crall/kcd_sealion/ssd-data/tile_cache/_universal/fd353b1c/shards

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
