#!/usr/bin/env bash
# Generation 2 — full-resolution tiles + WebDataset training input.
#
#   scheme:   pup_vs_nonpup (P1)
#   variant:  deimv2_hgnetv2_n
#   gpus:     1 (arisia)
#   gen:      002
#
# Hypothesis: pup AP=0.000 in v6 came from training on tiles cut at
# downsampled scales (source_scales=1.0,0.5,0.25,0.125). At the 0.125
# scale a typical pup is ~2-4 px, undetectable at any model capacity.
# gen002 tiles at source_scales=1.0 only (native resolution) with the
# same 320x320 model input -- the matcher sees pups at their full
# source-pixel size. Stride stays 0.5 for direct comparability with v6.
#
# Also: KCD_USE_WEBDATASET=1 routes training reads through
# kwcoco_dataloader's HDD-friendly tar-shard stream (per namek
# benchmarks, 1.67x faster on rotational disk vs. random JPEG opens).
# The shard build is a one-time amortized step done once per tile
# bundle hash; subsequent gen002 runs reuse it.
#
# Submit (from kit root):
#   bash projects/viame_sealions_2026/scripts/submit_train_pup_vs_nonpup_deimv2_hgnetv2_n_1gpu_arisia_gen002.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Hyperparameters — match v6 except where called out
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1
export KCD_PER_GPU_BATCH=16
export KCD_VAL_BATCH_MULT=1
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5.66e-4
export KCD_BACKBONE_LR=2.83e-4
export KCD_USE_AMP=true

# ============================================================
# gen002 tile params — FULL RESOLUTION ONLY, no downsampling
# ============================================================
export KCD_TILE_SIZE=320
export KCD_TILE_SOURCE_SCALES=1.0
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# ============================================================
# WebDataset shards path — built once per universal tile bundle.
# Auto-resolved to "$TILE_DIR/shards" by _launch_train.sh; left empty
# here so the cache lives next to the bundle and shards survive across
# gen002 runs.
# ============================================================
export KCD_USE_WEBDATASET=1

# ============================================================
# Run identity
# ============================================================
RUN_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
export KCD_RUN_NAME="${RUN_NAME#submit_train_}"

exec bash "$SCRIPT_DIR/_submit_train.sh"
