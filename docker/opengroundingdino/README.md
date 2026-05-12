# OpenGroundingDINO Docker Images

The normal workflow can run directly from the editable checkout. Docker is an
optional reproducibility layer for pinning the CUDA/PyTorch/compiler stack and
prebuilding OpenGroundingDINO's `MultiScaleDeformableAttention` extension.

The Dockerfile uses BuildKit cache mounts for `apt`, `uv`, `pip`, and torch
extension builds. The helper scripts set `DOCKER_BUILDKIT=1` automatically.

## Recommended Profiles

### Stable CUDA 13.0 / PyTorch cu130

This image is usually the safest reproducible training environment today. It
runs on any host whose NVIDIA driver is new enough for CUDA 13.x, including a
driver that reports `CUDA Version: 13.2`.

```bash
cd /home/joncrall/code/kwcoco_detector_kit

docker build \
    -f docker/opengroundingdino/Dockerfile \
    --build-arg BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
    -t kwcoco-detector-kit:ogdino-cu130 .
```

### Arisia CUDA 13.2 / PyTorch Nightly cu132

The arisia host reports:

```text
Docker version 28.0.4
nvcc CUDA release 13.2, V13.2.51
nvidia-smi Driver Version: 595.58.03  CUDA Version: 13.2
```

To make the container toolkit match that CUDA 13.2 stack, build:

```bash
cd /home/joncrall/code/kwcoco_detector_kit
bash docker/opengroundingdino/build_arisia_cuda132.sh
```

Equivalent explicit command:

```bash
docker build \
    -f docker/opengroundingdino/Dockerfile \
    --build-arg BASE_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu24.04 \
    --build-arg PYTHON_VERSION=3.11 \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu132 \
    --build-arg TORCH_PRE=1 \
    -t kwcoco-detector-kit:ogdino-cu132-arisia .
```

This profile uses PyTorch nightly CUDA 13.2 wheels. Prefer the stable cu130
image when reproducibility matters more than matching the host toolkit exactly;
prefer the cu132 image when you want `nvcc` 13.2 inside the container.

During build, the Dockerfile verifies:

```text
torch.version.cuda == nvcc --version release
```

and fails immediately if they disagree.

## Build-Time Caching

The Dockerfile is ordered so expensive layers survive normal source edits:

1. Base CUDA image + apt packages.
2. Seeded Python venv.
3. PyTorch / torchvision / torchaudio from the selected CUDA wheel index.
4. Third-party Python dependencies from `pyproject.toml`.
5. OpenGroundingDINO source + compiled CUDA extension.
6. The fast-changing `kwcoco_detector_kit`, docs, examples, and Docker helper
   files.

That means editing toolkit Python files should only rerun the final editable
install / env-check layer, not the PyTorch download or CUDA extension build.

The `.dockerignore` excludes local `.venv`, `__pycache__`, compiled `.so`
files, tile assets, and kwcoco bundles so the build context stays small.

## Run Interactively

Run with the data repo mounted:

```bash
DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026

docker run --rm -it --gpus all --ipc=host \
    -v "$DATA_DPATH:$DATA_DPATH" \
    -v /home/joncrall/code/kwcoco_detector_kit:/workspace/kwcoco_detector_kit \
    -w /workspace/kwcoco_detector_kit \
    kwcoco-detector-kit:ogdino-cu132-arisia
```

Inside the container, either use the baked copy at `/opt/kwcoco_detector_kit`
or the bind-mounted editable checkout at `/workspace/kwcoco_detector_kit`.
The entrypoint copies the precompiled `MultiScaleDeformableAttention*.so` from
the baked copy into the bind-mounted checkout if the mounted checkout does not
already have it.

Quick smoke inside the container:

```bash
python -m kwcoco_detector_kit check-env \
    --groups core,opengroundingdino \
    --strict_import

python - <<'PY'
import shutil
import subprocess
import torch
print(torch.__version__)
print(torch.version.cuda)
print(subprocess.check_output([shutil.which('nvcc'), '--version'], text=True))
PY
```

## Run VIAME Training

After mounting the VIAME data repo:

```bash
DATA_DPATH=/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026

docker run --rm -it --gpus all --ipc=host \
    --shm-size=32g \
    -v "$DATA_DPATH:$DATA_DPATH" \
    -v /home/joncrall/code/kwcoco_detector_kit:/workspace/kwcoco_detector_kit \
    -w /workspace/kwcoco_detector_kit \
    -e KIT_DPATH=/workspace/kwcoco_detector_kit \
    -e DATA_DPATH="$DATA_DPATH" \
    -e KCD_ROOT="$DATA_DPATH/training_runs/docker_ogdino_a6000x4" \
    -e KCD_CACHE_ROOT="$DATA_DPATH/training_runs/cache/ogdino_swint" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -e NUM_GPUS=4 \
    -e KCD_DISTRIBUTED=1 \
    -e SCALE_TIER=2-4xL \
    -e BATCH_SIZE=4 \
    -e VAL_BATCH_SIZE=4 \
    kwcoco-detector-kit:ogdino-cu132-arisia \
    bash examples/viame_sealions_2026/run_3090_opengroundingdino.sh
```

Despite the historical `run_3090` filename, the script now accepts
`NUM_GPUS=4`, `KCD_DISTRIBUTED=1`, and `SCALE_TIER=2-4xL`; the defaults remain
single-GPU so existing host workflows keep working.

## CUDA Variants

The Dockerfile is parameterized:

```bash
docker build \
    -f docker/opengroundingdino/Dockerfile \
    --build-arg BASE_IMAGE=nvidia/cuda:<cuda-tag>-devel-ubuntu24.04 \
    --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/<torch-cuda-index> \
    -t kwcoco-detector-kit:ogdino-<cuda-tag> .
```

The key invariant is the same as the host workflow: the CUDA toolkit in the
image must match `torch.version.cuda`, because OpenGroundingDINO compiles a
CUDA extension.
