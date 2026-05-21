# OpenGroundingDINO Docker Images

The normal workflow can run directly from the editable checkout. Docker is an
optional reproducibility layer for pinning the CUDA/PyTorch/compiler stack and
prebuilding OpenGroundingDINO's `MultiScaleDeformableAttention` extension.

The Dockerfile uses BuildKit cache mounts for `apt`, `uv`, `pip`, and torch
extension builds. The helper scripts set `DOCKER_BUILDKIT=1` automatically.

## Auto-Detect The Build Profile

Use `build_auto.sh` when you want the host to pick the CUDA profile and tag the
image with a stable name:

```bash
cd /home/joncrall/code/kwcoco_detector_kit
bash docker/opengroundingdino/build_auto.sh
```

The script reads `nvidia-smi`'s reported `CUDA Version` and chooses the highest
supported container CUDA profile that does not exceed it:

- CUDA >= 13.2 -> `cu132`, also tagged `kwcoco-detector-kit:ogdino-cu132-arisia`
- CUDA >= 13.0 -> `cu130`, also tagged `kwcoco-detector-kit:ogdino-cu130`

Both cases are tagged as `kwcoco-detector-kit:ogdino-auto`, so run commands can
use the same image name on different machines:

```bash
docker run --rm --gpus all kwcoco-detector-kit:ogdino-auto python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

Override detection for testing or cluster policy:

```bash
HOST_CUDA_VERSION=13.0 KCD_DOCKER_DRYRUN=1 bash docker/opengroundingdino/build_auto.sh
KCD_DOCKER_CUDA_PROFILE=cu130 bash docker/opengroundingdino/build_auto.sh
KCD_DOCKER_CUDA_PROFILE=cu132 bash docker/opengroundingdino/build_auto.sh
```

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
6. The fast-changing `kwcoco_detector_kit`, docs, examples, tests, and Docker
   helper files.
7. `pytest tests/unit -m "not requires_gpu"` verification (set
   `--build-arg RUN_TESTS=0` to skip).

That means editing toolkit Python files should only rerun the final editable
install / env-check layer, not the PyTorch download or CUDA extension build.

The `.dockerignore` excludes local `.venv`, `__pycache__`, compiled `.so`
files, tile assets, and kwcoco bundles so the build context stays small.

The helper scripts also ensure the OpenGroundingDINO submodule is initialized
before invoking Docker. If the build context is only a few KB and the build
fails at `tpl/Open-GroundingDino/models/GroundingDINO/ops`, run:

```bash
git submodule update --init tpl/Open-GroundingDino
```

The OpenGroundingDINO CUDA op is built during `docker build`, where no GPU is
usually visible. The image sets `FORCE_CUDA=1` and defaults
`TORCH_CUDA_ARCH_LIST=8.6` for RTX A6000. Override
`TORCH_CUDA_ARCH_LIST` if you build the image for a different GPU family.

## Run Interactively

Run with the data repo mounted:

```bash
DATA_DPATH=/data/users/jon.crall/dvc-repos/viame_sealions_2026
KCD_EXPT_DPATH=/data/users/jon.crall/dvc-repos/viame_sealions_2026_expt

docker run --rm -it --gpus all --ipc=host \
    -v "$DATA_DPATH:$DATA_DPATH:ro" \
    -v "$KCD_EXPT_DPATH:$KCD_EXPT_DPATH" \
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

## Run Tests Against The Baked Image

The `tests/` tree is copied into the image and the unit suite is executed at
build time as a verification step (skip with `--build-arg RUN_TESTS=0`).

To re-run the same suite against an already-built image — useful for catching
regressions in the deploy environment without rebuilding from scratch:

```bash
# Full unit suite, no GPU needed.
docker run --rm kwcoco-detector-kit:ogdino-auto \
    kcd-pytest tests/unit -m "not requires_gpu"

# Subset.
docker run --rm kwcoco-detector-kit:ogdino-auto \
    kcd-pytest tests/unit/test_multiclass_pipeline.py -v

# GPU-aware integration tests (forward --gpus all so requires_gpu tests run).
docker run --rm --gpus all kwcoco-detector-kit:ogdino-auto \
    kcd-pytest tests/unit
```

If you bind-mount an editable checkout at `/workspace/kwcoco_detector_kit` to
iterate on tests interactively, point the helper at it:

```bash
docker run --rm --gpus all \
    -v /home/joncrall/code/kwcoco_detector_kit:/workspace/kwcoco_detector_kit \
    -e KCD_TEST_REPO=/workspace/kwcoco_detector_kit \
    kwcoco-detector-kit:ogdino-auto kcd-pytest tests/unit
```

## Run VIAME Training

After mounting the VIAME data repo:

```bash
DATA_DPATH=/data/users/jon.crall/dvc-repos/viame_sealions_2026
KCD_EXPT_DPATH=/data/users/jon.crall/dvc-repos/viame_sealions_2026_expt

docker run --rm -it --gpus all --ipc=host \
    --shm-size=32g \
    -v "$DATA_DPATH:$DATA_DPATH:ro" \
    -v "$KCD_EXPT_DPATH:$KCD_EXPT_DPATH" \
    -v /home/joncrall/code/kwcoco_detector_kit:/workspace/kwcoco_detector_kit \
    -w /workspace/kwcoco_detector_kit \
    -e KIT_DPATH=/workspace/kwcoco_detector_kit \
    -e DATA_DPATH="$DATA_DPATH" \
    -e KCD_ROOT="$KCD_EXPT_DPATH/training_runs/docker_ogdino_a6000x4" \
    -e KCD_CACHE_ROOT="$KCD_EXPT_DPATH/cache/ogdino_swint" \
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
