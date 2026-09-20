# RF-DETR Docker image

RF-DETR should normally run through this container rather than requiring a host
Python environment to carry the RF-DETR / PyTorch / CUDA dependency stack. The
image pins the KDK checkout and `tpl/rf-detr` submodule, installs RF-DETR's training extras, and records both revisions in image labels and
`/etc/kcd_provenance.json`.

RF-DETR has no custom CUDA extension in this image. The CUDA profile selects a
compatible CUDA runtime / PyTorch wheel pair; it does not compile a
GPU-architecture-specific RF-DETR kernel.

The runtime image also installs the optional image-I/O and geometry
accelerators used by source-space prediction:

- `kwimage_ext` is installed from PyPI so KWCoco/KWImage NMS can use the
  compiled backend instead of falling back to pure Python/NumPy paths;
- GDAL is installed with `python -m kwcoco finish_install --with_gdal=True`
  through Kitware's large-image wheel index. This enables `delayed_image` to
  use region-readable raster paths without adding a distro GDAL development
  stack to the image.

The Docker build imports both `kwimage_ext` and `osgeo.gdal` after
installation, so an image is not published as usable if either fast-path
dependency failed to install.

## Normal build: auto profile

Use the generic builder:

```bash
cd ~/code/kwcoco_detector_kit
bash docker/rfdetr/build_auto.sh
```

The builder considers both the driver CUDA level and GPU compute capability.
The current policy is deliberately different for workstation Ampere and
Blackwell hosts:

- compute capability `< 12.0` with a CUDA-13-capable driver -> stable PyTorch
  `cu130`, tagged `kwcoco-detector-kit:rfdetr-cu130`;
- compute capability `>= 12.0` and host CUDA >= 13.2 -> PyTorch nightly
  `cu132`, tagged `kwcoco-detector-kit:rfdetr-cu132`.

Both are also tagged `kwcoco-detector-kit:rfdetr-auto`. Thus an RTX 3090
(compute capability 8.6) deliberately selects stable `cu130` even when its
newer driver reports CUDA 13.2. A newer NVIDIA driver can run an older CUDA
runtime image; there is no benefit here to selecting a nightly cu132 wheel
merely because the driver advertises 13.2.

Override explicitly when reproducing a known environment:

```bash
KCD_RFDETR_CUDA_PROFILE=cu130 bash docker/rfdetr/build_auto.sh
KCD_RFDETR_CUDA_PROFILE=cu132 bash docker/rfdetr/build_auto.sh
```

The historical production helper remains available for the aiq Blackwell
profile:

```bash
bash docker/rfdetr/build_aiq_cuda132_blackwell.sh
```

That helper is an explicit production-machine profile; its name does not imply
that `Dockerfile` itself contains Blackwell-only RF-DETR code.

For a named stable workstation build:

```bash
bash docker/rfdetr/build_stable_cuda130.sh
```

## Normal execution: `kcd-rfdetr`

Do not recreate a long `docker run` command for each package/export/prediction.
Use the checked-in wrapper:

```bash
docker/rfdetr/kcd-rfdetr image-info

docker/rfdetr/kcd-rfdetr package-build \
    --workdir="$HOME/data/example_snapshot" \
    --trainer=rfdetr \
    --out="$HOME/data/models/example.zip"

docker/rfdetr/kcd-rfdetr predict \
    --model="$HOME/data/models/example.zip" \
    --src="$HOME/data/example.kwcoco.zip" \
    --dst="$HOME/data/example.pred.kwcoco.zip" \
    --device=cuda:0 \
    --backend=torch \
    --windowed=true
```

The wrapper:

- builds `kwcoco-detector-kit:rfdetr-auto` automatically if it is absent;
- defaults to physical host GPU 0 (`KCD_RFDETR_GPU=0`);
- exposes only that GPU, so it appears as `cuda:0` inside the container;
- bind-mounts `$HOME` at the identical path so KWCoco asset paths and package
  provenance remain meaningful;
- automatically bind-mounts absolute `--key=/path` CLI paths that resolve
  outside `$HOME`, and accepts newline-separated `KCD_RFDETR_EXTRA_MOUNTS`
  for KWCoco asset roots referenced from inside manifests;
- uses a UID-safe Python installation under `/opt/uv-python`, so running as the
  host uid/gid never depends on traversing `/root`;
- mounts the current KDK checkout over the baked editable-install location so
  local source changes are actually exercised;
- preserves the historical `bash -lc` RF-DETR image entrypoint used by existing
  production launch files, while shell-quoting argv safely for normal commands;
- runs as the host uid/gid so generated packages and predictions are not
  root-owned.

Override the physical device when needed:

```bash
KCD_RFDETR_GPU=1 docker/rfdetr/kcd-rfdetr predict ...
```

Use `exec` for a non-KDK command inside the same runtime:

```bash
docker/rfdetr/kcd-rfdetr exec python -c 'import torch, rfdetr; print(torch.cuda.get_device_name(0))'
```

Use `shell` for interactive diagnosis:

```bash
docker/rfdetr/kcd-rfdetr shell
```

## ONNX note

The default RF-DETR image is deliberately the proven PyTorch runtime used for
training and native-mask prediction. It does not yet claim a pinned GPU ONNX
Runtime stack. The ShitSpotter local-review workflow therefore defaults to
`--backend=torch`. ONNX export/parity remains an opt-in KDK package feature and
should get a separately tested container profile before it becomes the default
local prediction backend.

## Dry-run / CI checks

No GPU or Docker build is required to verify profile selection and command
construction:

```bash
HOST_CUDA_VERSION=13.2 HOST_COMPUTE_CAP=8.6 KCD_DOCKER_DRYRUN=1 \
    bash docker/rfdetr/build_auto.sh

KCD_DOCKER_DRYRUN=1 docker/rfdetr/kcd-rfdetr predict \
    --model=/tmp/model.zip --windowed=true
```
