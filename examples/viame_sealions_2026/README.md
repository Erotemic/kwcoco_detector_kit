# VIAME Sea Lions 2021-2024

This example starts from the converted VIAME kwcoco bundle:

```bash
/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/sealions_2021_2024.kwcoco.zip
```

The first-pass training target is a single `sealion` detector. The source
annotation codes (`B`, `F`, `J`, `NFS`, etc.) are preserved on each annotation
as `source_category`, but all selected boxes train as one class.

## Prepare Splits

This has already been run once on the VM:

```bash
PYTHON_BIN=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026/.venv/bin/python
DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026

$PYTHON_BIN examples/viame_sealions_2026/prepare_training_kwcoco.py \
    --src "$DATA_DPATH/sealions_2021_2024.kwcoco.zip" \
    --dst_dpath "$DATA_DPATH/training_ready_v1"
```

Outputs:

```bash
$DATA_DPATH/training_ready_v1/all_collapsed.kwcoco.zip
$DATA_DPATH/training_ready_v1/train.kwcoco.zip
$DATA_DPATH/training_ready_v1/vali.kwcoco.zip
$DATA_DPATH/training_ready_v1/test.kwcoco.zip
$DATA_DPATH/training_ready_v1/prepare_report.json
```

Current split sizes:

| split | images | annotations |
|---|---:|---:|
| train | 1314 | 70287 |
| vali | 164 | 9215 |
| test | 164 | 10453 |

## Host Setup

Run this on the host with the RTX 3090, inside the Python environment you want
to train with:

```bash
cd /home/joncrall/code/kwcoco_detector_kit
bash examples/viame_sealions_2026/setup_host_env.sh
```

The setup script installs the kit with OpenGroundingDINO extras, builds the
OpenGroundingDINO deformable-attention extension, and downloads the Swin-T
GroundingDINO checkpoint.

The scripts separate per-run output from reusable cache:

```bash
KCD_ROOT       # run outputs: runs/, sweeps/, evals/
KCD_CACHE_ROOT # reusable artifacts: pretrained weights, tile bundles
```

Changing `KCD_ROOT` for a new experiment should not force retile as long as
`KCD_CACHE_ROOT` is unchanged. The runner also falls back to the initial
`ogdino_swint_3090/tiles` cache if it already exists.

## Start Training

```bash
cd /home/joncrall/code/kwcoco_detector_kit
bash examples/viame_sealions_2026/run_3090_opengroundingdino.sh
```

Useful knobs:

```bash
SMOKE=1 NUM_EPOCHS=1 BATCH_SIZE=1 bash examples/viame_sealions_2026/run_3090_opengroundingdino.sh
KCD_DO_EVAL=1 bash examples/viame_sealions_2026/run_3090_opengroundingdino.sh
TILE_SIZE=640 BATCH_SIZE=2 bash examples/viame_sealions_2026/run_3090_opengroundingdino.sh
```

Default training uses positive-only multiscale tiles at `800x800`, batch size
`2`, and skips export/benchmark/eval so the first GPU run answers the most
important question: does the trainer start and make checkpoints on this data?
Use `SMOKE=1` first; it subsets the already-built tiles to 512 train / 128 val
images so one epoch is short enough to debug the trainer loop.

## Slurm: 4x RTX A6000

For a single-node Slurm system with 4x NVIDIA RTX A6000 cards:

```bash
cd /home/joncrall/code/kwcoco_detector_kit
sbatch examples/viame_sealions_2026/run_slurm_a6000_opengroundingdino.sbatch
```

Useful overrides:

```bash
sbatch --export=ALL,CUDA_MODULE=cuda/13.0,NUM_EPOCHS=24,BATCH_SIZE=4 \
    examples/viame_sealions_2026/run_slurm_a6000_opengroundingdino.sbatch
```

If `nvidia-smi` reports driver `595.58.03` and `CUDA Version: 13.2`, still
match the loaded `nvcc` toolkit to `torch.version.cuda`, not to the driver
banner. A CUDA 13.2-capable driver can run a PyTorch `cu130` wheel, but the
OpenGroundingDINO CUDA extension should be compiled with CUDA 13.0 in that
case. The setup script checks this before building the extension.

## Docker

The editable host workflow remains supported and is the fastest way to develop.
For a pinned CUDA/PyTorch/compiler stack with the OpenGroundingDINO CUDA
extension prebuilt, see:

```bash
docker/opengroundingdino/README.md
```
