# 2026-05-12 — VIAME Sea Lions 2021-2024 OpenGroundingDINO Bring-up

## Purpose

Get the VIAME Steller sea lion aerial imagery converted into training-ready
`kwcoco`, then start tuning a detector from a foundation OpenGroundingDINO
Swin-T checkpoint on a single RTX 3090 host (`namek`). This entry is a broad
chronological journal, not just a bug postmortem. It should give a future agent
enough context to resume work without rediscovering the same edges.

## Repositories And Important Paths

Main data repo:

```bash
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026
```

Main code repo:

```bash
/home/joncrall/code/kwcoco_detector_kit
```

Historical converter that was inspected but not used as a dependency:

```bash
/home/joncrall/code/bioharn/dev/data_tools/viame_csv_to_kwcoco.py
/home/joncrall/code/bioharn/bioharn/io/viame_csv.py
```

Host Python env used for training:

```bash
/home/joncrall/.local/uv/envs/uvpy3.13.2
```

Host CUDA toolkit after cleanup:

```bash
CUDA_HOME=/usr/local/cuda-13.0
```

## Data Conversion History

The old `bioharn` converter was too old for this dataset. It expects a VIAME
CSV with alternating class/score columns and delegates to
`bioharn.io.viame_csv`. The current CSV files are headered:

```text
ID,IMAGE,FRAME,TL_X,TL_Y,BR_X,BR_Y,REVIEW_D,TARGET_LENGTH,CLASS,REVIEW_C,ATTRIBUTE
```

The `.venv` in the data repo had `kwcoco`, `kwimage`, `PIL`, and friends, but
not an installed `bioharn`. We intentionally avoided depending on `bioharn`.

Created standalone converter in the data repo:

```bash
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/scripts/convert_sealions_csv_to_kwcoco.py
```

Converted output:

```bash
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/sealions_2021_2024.kwcoco.zip
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/sealions_2021_2024_sample40.kwcoco.zip
```

Important notes:

- `IncludesNewAnnotations` has 2022, 2023, and 2024, but no 2021 CSV.
- Used `Redundant/2021_annotations.csv` for 2021.
- Used `IncludesNewAnnotations/2022_annotations.csv`,
  `IncludesNewAnnotations/2023_annotations.csv`, and
  `IncludesNewAnnotations/2024_annotations.csv` for the other years.
- 2022 and 2023 had filename mismatches where CSV used `AMAKROCKS` and images
  used `AMAK+ROCKS`. Every mismatch resolved uniquely by normalized basename.
- 2021 and 2024 matched directly.
- The full kwcoco has `1642` images and `89955` annotations.
- Original category codes:

```text
B: 5947
DN: 19
DP: 301
F: 20500
J: 19478
NFS: 15965
O: 5909
P: 18634
S: 3202
```

Validation:

```bash
kwcoco validate sealions_2021_2024.kwcoco.zip
```

passed. There were 132 boxes slightly outside image bounds, but only by about
13 px max in the samples checked. This looked like edge annotation noise, not
image/annotation mismatch.

## Training-Ready Collapsed Dataset

The current kit training/export path is easiest for single-category training,
so the first model treats all source codes as one `sealion` class while
preserving the original code on each annotation as `source_category`.

Added:

```bash
/home/joncrall/code/kwcoco_detector_kit/examples/viame_sealions_2026/prepare_training_kwcoco.py
```

Prepared output:

```bash
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/training_ready_v1/all_collapsed.kwcoco.zip
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/training_ready_v1/train.kwcoco.zip
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/training_ready_v1/vali.kwcoco.zip
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/training_ready_v1/test.kwcoco.zip
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/training_ready_v1/prepare_report.json
```

Split sizes:

```text
train: 1314 images, 70287 annotations
vali:   164 images,  9215 annotations
test:   164 images, 10453 annotations
```

The split script writes absolute image paths. This was a deliberate fix:
writing the split bundles into `training_ready_v1` while preserving old relative
paths made `kwcoco validate --missing` look for images under
`training_ready_v1/Redacted_Imagery/...`.

## Kit Example Added

Added an example directory:

```bash
/home/joncrall/code/kwcoco_detector_kit/examples/viame_sealions_2026
```

Files:

```bash
README.md
prepare_training_kwcoco.py
setup_host_env.sh
run_3090_opengroundingdino.sh
```

The runner started conservative:

- trainer: `opengroundingdino`
- variant: `opengroundingdino_swint`
- tile size: `800`
- source scales: `1.0,0.5,0.25`
- positive-only tiles
- default batch initially `2`, later overnight recommendation used `8`
- export/bench disabled for first pass
- eval off by default unless `KCD_DO_EVAL=1`

Smoke mode was added after the full tile set proved too slow for iteration:

```bash
SMOKE=1 NUM_EPOCHS=1 BATCH_SIZE=1 bash examples/viame_sealions_2026/run_3090_opengroundingdino.sh
```

Smoke subsets are regenerated each time and use absolute image paths. This
fixed a failure where OpenGroundingDINO looked for smoke assets under
`smoke_tiles/train_tiles_pos_assets/...` even though the JPEGs lived under the
original tile bundle directory.

## Tile Cache / Run Output Separation

We discovered that changing `KCD_ROOT` caused the runner to retile because
`TILE_DPATH` defaulted to `$KCD_ROOT/tiles`. This is bad for experiment
hygiene: run roots should be disposable, while tiles and pretrained weights
should be reusable.

Patched the runner to distinguish:

```bash
KCD_ROOT       # per-run output: runs/, sweeps/, evals/
KCD_CACHE_ROOT # reusable artifacts: pretrained weights, tile bundles
```

The runner now prints:

```bash
KCD_ROOT=...
KCD_CACHE_ROOT=...
TILE_DPATH=...
PRETRAIN_MODEL_PATH=...
```

It also falls back to the original legacy tile cache:

```bash
$DATA_DPATH/training_runs/ogdino_swint_3090/tiles
```

if the new cache path does not yet exist and the old cache does.

## Host Environment Bring-up

The host has RTX 3090, driver `580.142`, and `nvidia-smi` reports CUDA runtime
support `13.0`.

Initial problem:

- `nvcc` was missing.
- Installing Ubuntu's `nvidia-cuda-toolkit` installed CUDA `11.5`, which was
  wrong for the CUDA 13.0 PyTorch wheel.
- Fixed by removing old toolkit and installing NVIDIA's `cuda-toolkit-13-0`.

Important distinction recorded:

- `nvidia-smi CUDA Version` is driver runtime capability.
- `nvcc --version` is the toolkit compiler version.
- PyTorch can run CUDA without `nvcc`, but OpenGroundingDINO's CUDA extension
  requires `nvcc`.

`setup_host_env.sh` now checks `torch.version.cuda` against `nvcc --version`
before compiling OpenGroundingDINO's extension.

OpenGroundingDINO deformable attention extension compiled successfully after
CUDA 13.0 toolkit was on `PATH`.

The setup script downloaded:

```bash
groundingdino_swint_ogc.pth
```

from the IDEA-Research GroundingDINO release. It is now intended to live in:

```bash
$KCD_CACHE_ROOT/pretrained/groundingdino_swint_ogc.pth
```

with a legacy fallback under the first `ogdino_swint_3090` root.

## Dependency Discoveries

The `[opengroundingdino]` extra and `setup_audit` were expanded as missing
runtime dependencies were encountered.

Added:

```text
pycocotools
matplotlib
timm
```

Existing OpenGroundingDINO extras already included:

```text
transformers
addict
yapf
jsonlines
```

`pycocotools` was needed by both `tools/coco2odvg.py` and
`datasets/coco.py`.

`matplotlib` and `timm` were discovered during actual trainer startup.

## Upstream OpenGroundingDINO Patches / Hacks

These are local submodule modifications and should eventually be upstreamed,
vendored as patches, or replaced with a cleaner wrapper.

Patched:

```bash
tpl/Open-GroundingDino/train_dist.sh
tpl/Open-GroundingDino/util/slconfig.py
tpl/Open-GroundingDino/main.py
```

### `train_dist.sh`

Problems:

- It hardcoded `python`, ignoring the active env.
- It always used `torch.distributed.launch`, even for one GPU.
- It ignored the kit's `use_amp` setting.

Fixes:

- Uses `${PYTHON_BIN:-python}`.
- Uses plain `main.py` for `GPU_NUM=1`; only uses distributed launch for
  multi-GPU.
- Adds `--amp` when `USE_AMP=1` or `USE_AMP=true`.

Note: DEIMv2 still needs torchrun even for single GPU due to its own upstream
distributed assumptions. This change is specific to OpenGroundingDINO, whose
`setup_distributed()` has a non-distributed path when `WORLD_SIZE` is absent.

### `util/slconfig.py`

Problem:

```text
TypeError: FormatCode() got an unexpected keyword argument 'verify'
```

Cause:

- Newer `yapf` removed the `verify` kwarg.

Fix:

- Try `FormatCode(..., verify=True)`.
- On `TypeError` mentioning `verify`, retry without `verify`.

### `main.py`

Problem:

PyTorch 2.6+ changed `torch.load()` default to `weights_only=True`. Resuming an
OpenGroundingDINO checkpoint failed with:

```text
_pickle.UnpicklingError: Weights only load failed
Unsupported global: GLOBAL argparse.Namespace
```

Cause:

- OpenGroundingDINO checkpoints are full training checkpoints, not pure tensor
  state dicts.

Fix:

- Use `torch.load(..., weights_only=False)` for:
  - frozen weights
  - resume checkpoints
  - pretrained model path

Security note:

- This is safe only for trusted local/release checkpoints. The setup uses the
  known GroundingDINO release checkpoint and local training checkpoints.

## Training Timeline And Current Status

Full tiling with default settings produced:

```text
train: 34553 tiles, 473477 annotations
vali:   4409 tiles,  61917 annotations
test:   4630 tiles,  69660 annotations
```

The first batch-1 full run showed:

```text
34553 steps / epoch
eta about 4 hours per epoch
max mem about 3.4 GB initially
```

We stopped/avoided treating that as a useful smoke because one epoch was too
long. Added smoke mode.

Smoke after fixes:

- Ran successfully.
- Max memory about `3.7 GB`.

Because smoke used only about `3.7/24 GB`, an overnight full run was started
with:

```bash
BATCH_SIZE=8
VAL_BATCH_SIZE=8
NUM_EPOCHS=8
LR=1e-4
BACKBONE_LR=1e-5
```

At batch 8, validation reported:

```text
max mem: 13555 MB
```

Validation metrics observed around epoch 5/6:

```text
AP@[IoU=0.50:0.95] all    = 0.514
AP50                     = 0.782
AP75                     = 0.583
AP small                 = 0.374
AP medium                = 0.693
AP large                 = 0.802
AR all maxDets=100       = 0.591
AR small maxDets=100     = 0.446
AR medium maxDets=100    = 0.772
AR large maxDets=100     = 0.856
```

This is a strong sign that the pipeline is not merely running but learning.
Small-object AP is the obvious weak spot.

The run had already reached:

```text
Epoch: [6] [0/4319]
lr: 0.000010
```

Concern:

- The inherited upstream scheduler drops LR too early for an 8-epoch run.
- The run is still useful, but next run should probably set scheduler milestones
  relative to `NUM_EPOCHS`.

## Overnight Run Suggestions From Current State

Current batch 8 run uses about 13.6 GB, leaving headroom on a 24 GB 3090.

Candidate next run:

```bash
BATCH_SIZE=12
VAL_BATCH_SIZE=12
NUM_EPOCHS=16
LR=1e-4
BACKBONE_LR=1e-5
```

But before launching that, inspect final metrics from the current run and
decide whether to fix the LR schedule first.

Potential next config improvements:

- Patch OpenGroundingDINO config generation so `lr_drop` / `lr_drop_list` are
  sensible for short fine-tunes.
- Consider `BATCH_SIZE=12` or `16` after verifying memory at batch 8 remains
  stable.
- Consider changing source scales / tile size for small-object AP.
- Add hard negatives after the positive-only detector is stable.

## Dockerization

User requested a robust and reproducible training workflow, but also wants it
to remain possible to work directly from the library.

Added Docker scaffold:

```bash
docker/opengroundingdino/Dockerfile
docker/opengroundingdino/README.md
.dockerignore
```

Design:

- Docker is optional, not required.
- Base image and PyTorch CUDA wheel index are build args.
- It should be possible to build images like:
  - `kwcoco-detector-kit:ogdino-cu130`
  - `kwcoco-detector-kit:ogdino-cu128`
- The Docker image precompiles OpenGroundingDINO's CUDA extension.
- The key invariant remains: toolkit CUDA version must match
  `torch.version.cuda`.

Open issue:

- If bind-mounting the editable repo over `/workspace/kwcoco_detector_kit`,
  the precompiled extension in `/opt/kwcoco_detector_kit` is not automatically
  present in the mounted checkout. Options:
  1. use baked `/opt/kwcoco_detector_kit` for training,
  2. copy the `.so` into the bind mount,
  3. rerun setup in the mounted checkout,
  4. improve the container entrypoint to handle this.

## Things That Are Hacks Or Temporary

- The OpenGroundingDINO submodule is patched in-place. These changes are
  practical but should eventually become formal patches or a fork.
- The current first-pass training collapses all source classes to `sealion`.
  This is intentional for detector bring-up, but not the final biological
  taxonomy story.
- Tile generation is positive-only. This is fine for first-pass training but
  hard-negative mining should be added after we have a usable checkpoint.
- Smoke subsets use the first N tile image ids, not a stratified/random sample.
  This is deterministic and simple, but not necessarily representative.
- The current runner names the candidate
  `opengroundingdino_swint_800x800_fixed`, so changing batch/epochs but reusing
  the same `KCD_ROOT` can collide with/resume older runs. Use distinct
  `KCD_ROOT` per experiment.
- `scriptconfig` comma-string warnings are noisy (`"1.0,0.5,0.25"` smartcasts
  to a list). The tile parser tolerates both forms, but the warning remains.

## Follow-ups

High priority:

- Make OpenGroundingDINO LR schedule generation aware of `NUM_EPOCHS`.
- Ensure run IDs include hyperparameters or a run tag to avoid accidental
  resume/collision.
- After current run finishes, record final metrics and checkpoint path.
- Add a quick command to export/visualize predictions on random held-out full
  frames.
- Add hard-negative mining / negative tile inclusion after first checkpoint.

Medium priority:

- Improve smoke subset selection to sample across years/sites/source images.
- Add a "reuse cache" test for the shell runner, or port shell logic to a small
  Python CLI where behavior is easier to unit test.
- Add a dependency lock or constraints file for the host env.
- Make Docker entrypoint handle editable bind mounts and precompiled extension
  placement cleanly.
- Build and test the `ogdino-cu130` Docker image on the host.

Later:

- Decide whether to keep single-class detection or train subclasses / age-sex
  categories. The original codes are preserved as `source_category`.
- Investigate small-object AP improvements: tile scale mix, higher resolution,
  inference tiling, hard negatives, and perhaps smaller stride.
- Consider multi-class / hierarchy support in `kwcoco_detector_kit` export and
  trainer config generation.

## Useful Commands

Activate host env in a new tmux window:

```bash
cd /home/joncrall/code/kwcoco_detector_kit
source /home/joncrall/.local/uv/envs/uvpy3.13.2/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export KIT_DPATH=/home/joncrall/code/kwcoco_detector_kit
export DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026
export KCD_CACHE_ROOT="$DATA_DPATH/training_runs/cache/ogdino_swint"
export KCD_OPENGROUNDINGDINO_REPO_DPATH="$KIT_DPATH/tpl/Open-GroundingDino"
export PRETRAIN_MODEL_PATH="$KCD_CACHE_ROOT/pretrained/groundingdino_swint_ogc.pth"
export PYTHON_BIN=python
```

Check CUDA/PyTorch:

```bash
which python
which nvcc
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version | grep release
```

Smoke:

```bash
SMOKE=1 NUM_EPOCHS=1 BATCH_SIZE=1 \
    bash examples/viame_sealions_2026/run_3090_opengroundingdino.sh
```

Overnight-style run reusing cache:

```bash
DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026

KCD_ROOT="$DATA_DPATH/training_runs/ogdino_swint_b8_e8" \
KCD_CACHE_ROOT="$DATA_DPATH/training_runs/cache/ogdino_swint" \
NUM_EPOCHS=8 \
BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
LR=1e-4 \
BACKBONE_LR=1e-5 \
bash examples/viame_sealions_2026/run_3090_opengroundingdino.sh
```
