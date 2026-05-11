# sealion_aerial — NOAA Steller Sea Lion Population Count

Aerial single-class detection on the NOAA Steller Sea Lion dataset (Kaggle 2017 challenge). Each source image is ~5616×3744 RGB; annotations are colored dot markers per class. Trains a DEIMv2 DINOv3-S detector at 640×640 input, tier L/XL single-GPU or 4×L DDP.

## Files

| file | role |
|---|---|
| `README.md` | this file |
| `prepare_kwcoco.py` | NOAA dataset → kwcoco bundle. Converts each colored dot to a 32×32 fixed-size bbox centered on the dot. Multi-class collapses to a single `sealion` category by default. |
| `config.yaml` | scriptconfig-compatible overlay for `python -m kwcoco_detector_kit run-all`. |
| `run_all.sh` | end-to-end driver. Tiles → trains → exports → evals → manifest. |

## Inputs

The Kaggle dataset distribution gives you:

```
Train.zip                ~947 raw aerial images (.jpg)
Train.csv               counts per image per class
TrainDotted.zip         the same images with colored dots painted on
MismatchedTrainImages.txt  list of files to exclude (count mismatches)
```

`prepare_kwcoco.py` consumes `Train/`, `TrainDotted/`, and `Train.csv`, drops the mismatched files, and emits:

```
$KCD_ROOT/sealion/
  raw.kwcoco.zip          full-image bundle (single category 'sealion')
  raw_assets/             symlinks back to the source jpegs
```

After that, `run_all.sh` runs the kit's standard pipeline: tile (multi-scale, oversized for crop-aug), train, export, eval, bench, manifest.

## Quick start (tier L single-GPU)

```bash
# 1. Convert NOAA -> kwcoco (CPU-bound, ~5 min)
python examples/sealion_aerial/prepare_kwcoco.py \
    --train_dpath /path/to/Train \
    --dotted_dpath /path/to/TrainDotted \
    --counts_csv /path/to/Train.csv \
    --mismatched /path/to/MismatchedTrainImages.txt \
    --dst /scratch/sealion/raw.kwcoco.zip

# 2. End-to-end
KCD_ROOT=/scratch/kcd_sealion bash examples/sealion_aerial/run_all.sh
```

## Quick start (4×L DDP)

```bash
KCD_ROOT=/scratch/kcd_sealion \
    KCD_NUM_GPUS=4 \
    KCD_DISTRIBUTED=1 \
    bash examples/sealion_aerial/run_all.sh
```

The kit's `_tier.py` auto-detects aggregate VRAM and picks tier `2-4xL`. Multi-GPU all-reduce is gated on PCIe link width (failure #17); pass `KCD_TIER=L` to force single-GPU when peer mismatch is suspected.

## Acceptance

End-to-end run completes without crash; produces a populated eligibility manifest with at least one `HOST_PROMISING` candidate of `candidate_kind=real`. Specific AP is not gated by acceptance — sealion is the "does the kit actually train a real detector" smoke, not a quality benchmark.

See [`docs/multi_gpu.md`](../../docs/multi_gpu.md) for the DDP recipe + the PCIe-link-width caveat.
