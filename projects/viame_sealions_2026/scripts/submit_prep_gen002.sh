#!/usr/bin/env bash
# Build the shared gen002 data artifacts once: full-resolution
# universal tile bundle + WebDataset shards. NO GPU. Training jobs
# fan out from this via slurm --dependency=afterok and skip prep
# steps because the cache markers are present.
#
# Workflow:
#
#   prep_jobid=$(KCD_PRINT_JOBID=1 bash projects/viame_sealions_2026/scripts/submit_prep_gen002.sh)
#   KCD_DEPENDS_ON=$prep_jobid bash projects/.../submit_train_pup_vs_nonpup_..._gen002.sh
#   KCD_DEPENDS_ON=$prep_jobid bash projects/.../submit_train_single_sealion_..._gen002.sh
#   KCD_DEPENDS_ON=$prep_jobid bash projects/.../submit_train_lifestage_6cls_..._gen002.sh
#
# Or use submit_gen002_pipeline.sh which orchestrates all four.
#
# This script reuses _submit_train.sh's plumbing for env-file write +
# sbatch submission. KCD_DATA_PREP_ONLY=1 makes _launch_train.sh stop
# after the shard build (step 1b) and skip apply_scheme + sweep.
#
# Tile parameters MUST match the per-scheme gen002 scripts so the
# cache hash agrees. Anything that affects KCD_TILE_* must be kept
# in lockstep across this file and submit_train_*_gen002.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Prep marker — these vars are needed but the values don't matter
# for the shared artifacts. We pick pup_vs_nonpup as the
# "anchor scheme" so the env file passes scheme validation. The
# prep job exits before apply_scheme runs.
# ============================================================
export KCD_SCHEME=pup_vs_nonpup
export KCD_CATEGORY_NAMES=pup,nonpup_sealion
export KCD_VARIANT=deimv2_hgnetv2_n
export KCD_NUM_GPUS=1                # sbatch requires --gres count; we ask
                                     # for 0 below via KCD_GRES override
export KCD_PER_GPU_BATCH=16
export KCD_NUM_EPOCHS=30
export KCD_INPUT_HW='[320, 320]'
export KCD_TRAIN_POLICY=fixed
export KCD_LR=5.66e-4
export KCD_BACKBONE_LR=2.83e-4
export KCD_USE_AMP=true

# ============================================================
# gen002 tile params — must match submit_train_*_gen002.sh
# ============================================================
export KCD_TILE_SIZE=320
export KCD_TILE_SOURCE_SCALES=1.0
export KCD_TILE_STRIDE_FRAC=0.5
export KCD_TILE_MIN_GT_AREA_FRAC=0.0005
export KCD_TILE_MIN_KEEP_FRACTION=0.20
export KCD_TILE_OVERSIZE_FACTOR=1.2
export KCD_TILE_KEEP_NEGATIVE=true

# ============================================================
# Prep-only mode + WebDataset
# ============================================================
export KCD_USE_WEBDATASET=1
export KCD_DATA_PREP_ONLY=1

# CPU-only resources. Tile generation is image-decode bound.
export KCD_GRES="none"               # no GPU (kit's _submit_train.sh
                                     # honors KCD_GRES override)
export KCD_CPUS_PER_TASK=8
export KCD_MEM=32G
export KCD_TIME_LIMIT=02:00:00

# ============================================================
# Run identity. Slightly different from a train run — prep's
# output lives in the tile cache, not under runs/<run_name>/.
# ============================================================
export KCD_RUN_NAME=gen002_data_prep

exec bash "$SCRIPT_DIR/_submit_train.sh"
