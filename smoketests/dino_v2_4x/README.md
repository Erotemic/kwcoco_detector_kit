# DINOv2 / OpenGroundingDINO 4x GPU Smoke Ladder

This ladder tests the Docker / CUDA / kwcoco / distributed-training stack from
cheap to expensive.

## Canonical Run: Slurm -> Docker

These smoketests are meant to run as Slurm jobs that launch the Docker image
built by `docker/opengroundingdino/build_arisia_cuda132.sh`.

Submit one stage:

```bash
bash smoketests/dino_v2_4x/slurm/submit_stage.sh 00
bash smoketests/dino_v2_4x/slurm/submit_stage.sh 01
bash smoketests/dino_v2_4x/slurm/submit_stage.sh 02
bash smoketests/dino_v2_4x/slurm/submit_stage.sh 03
```

When run from an interactive terminal, `submit_stage.sh` follows the Slurm
stdout log until the job finishes and returns the Slurm exit code. Use
`FOLLOW=0` to only submit and print the job id.

Submit a dependent ladder through the VIAME subset test:

```bash
MAX_STAGE=03 bash smoketests/dino_v2_4x/slurm/submit_ladder.sh
```

Submit the full ladder, including the real run:

```bash
MAX_STAGE=04 bash smoketests/dino_v2_4x/slurm/submit_ladder.sh
```

Useful outer knobs:

```bash
IMAGE_TAG=kwcoco-detector-kit:ogdino-cu132-arisia
DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026
KCD_SMOKE_ROOT_HOST=${SCRATCH}/kcd_smoketests/dino_v2_4x
KCD_CACHE_ROOT_HOST=${SCRATCH}/kcd_smoketests/cache/opengroundingdino
SLURM_PARTITION=<partition-name>
ACCOUNT=<account-name>
```

Logs are written to:

```bash
smoketests/dino_v2_4x/slurm/logs/
```

To attach to an already-submitted job:

```bash
python smoketests/dino_v2_4x/slurm/follow_job.py <jobid>
```

## Debug Run Inside The Container

For interactive debugging, launch the container manually:

```bash
DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026

docker run --rm -it --gpus all --ipc=host --shm-size=32g \
    -v "$DATA_DPATH:$DATA_DPATH" \
    -v /home/joncrall/code/kwcoco_detector_kit:/workspace/kwcoco_detector_kit \
    -w /workspace/kwcoco_detector_kit \
    -e KIT_DPATH=/workspace/kwcoco_detector_kit \
    -e DATA_DPATH="$DATA_DPATH" \
    -e KCD_CACHE_ROOT="$DATA_DPATH/training_runs/cache/ogdino_swint" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    kwcoco-detector-kit:ogdino-cu132-arisia \
    bash
```

Then run stages one at a time:

```bash
bash smoketests/dino_v2_4x/00_mock_demo_dataload.sh
bash smoketests/dino_v2_4x/01_ogdino_demo_1gpu.sh
bash smoketests/dino_v2_4x/02_ogdino_demo_4gpu.sh
bash smoketests/dino_v2_4x/03_ogdino_viame_subset_4gpu.sh
bash smoketests/dino_v2_4x/04_ogdino_viame_full_4gpu.sh
```

Or run the ladder through the subset test:

```bash
MAX_STAGE=03 bash smoketests/dino_v2_4x/run_ladder.sh
```

Set `MAX_STAGE=04` only when you want the full real run.

## Stages

| stage | purpose |
|---|---|
| `00_mock_demo_dataload.sh` | Tiny kwcoco demo data with `mock_tiny`; proves imports, image loading, and sweep plumbing. |
| `01_ogdino_demo_1gpu.sh` | OpenGroundingDINO on tiny demo data, one GPU, one tiny epoch. |
| `02_ogdino_demo_4gpu.sh` | Same demo data with 4-GPU distributed launch. |
| `03_ogdino_viame_subset_4gpu.sh` | Small absolute-path VIAME subset with 4 GPUs. |
| `04_ogdino_viame_full_4gpu.sh` | Full VIAME tiled-data 4-GPU run. |

## Useful Knobs

```bash
KCD_SMOKE_ROOT=/tmp/kcd_smoketests/dino_v2_4x
KCD_CACHE_ROOT=/path/to/cache
PRETRAIN_MODEL_PATH=/path/to/groundingdino_swint_ogc.pth
INPUT_SIZE=320        # demo stages; subset/full can use 800
BATCH_SIZE=1
NUM_EPOCHS=1
VIAME_SUBSET_TRAIN_IMAGES=16
VIAME_SUBSET_VALI_IMAGES=8
VIAME_SUBSET_TEST_IMAGES=8
```

The first OpenGroundingDINO stage downloads the Swin-T checkpoint into
`$KCD_CACHE_ROOT/pretrained/` if `PRETRAIN_MODEL_PATH` does not already exist.
