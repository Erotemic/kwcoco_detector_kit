# syntax=docker/dockerfile:1.7
#
# Cache layout (BuildKit-managed, NOT host bind mounts):
#   /root/.cache/uv               — uv's resolver + wheel cache (shared across
#                                   all install steps below).
#   /root/.cache/torch_extensions — torch CUDA extension build cache (used by
#                                   the OpenGroundingDino MultiScaleDeformableAttention
#                                   compile step only).
# Both are declared via `--mount=type=cache,target=...` which BuildKit stores
# in its own volume. The host's `~/.cache/uv` / `~/.cache/pip` are never
# touched by `docker build`, so host file ownership is never modified.
# (Don't bind-mount the host cache in build scripts either — see
# build_auto.sh.)
ARG BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_PRE=0
ARG UV_CACHE_DIR=/root/.cache/uv
ARG TORCH_CUDA_ARCH_LIST="8.6"

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHON_BIN=/opt/venv/bin/python
ENV KCD_OPENGROUNDINGDINO_REPO_DPATH=/opt/kwcoco_detector_kit/tpl/Open-GroundingDino
ENV KCD_DEIMV2_REPO_DPATH=/opt/kwcoco_detector_kit/tpl/DEIMv2
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ENV UV_CACHE_DIR=${UV_CACHE_DIR}
# BuildKit cache mounts live on overlayfs; uv's default hardlink mode
# fails across filesystems, so force copy mode. (Slower than hardlink
# on a real workstation but invisible inside the cache-mounted layer.)
ENV UV_LINK_MODE=copy
# Always install for the venv's python and precompile .pyc to drop
# first-import latency.
ENV UV_PYTHON=${PYTHON_BIN}
ENV UV_COMPILE_BYTECODE=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    pkg-config \
    python3 \
    python3-venv

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --seed --python ${PYTHON_VERSION} ${VIRTUAL_ENV} && \
    uv --version

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -U pip wheel setuptools && \
    if [ "${TORCH_PRE}" = "1" ]; then \
        uv pip install --prerelease=allow --index-url ${TORCH_INDEX_URL} torch torchvision torchaudio; \
    else \
        uv pip install --index-url ${TORCH_INDEX_URL} torch torchvision torchaudio; \
    fi && \
    ${PYTHON_BIN} - <<'PY'
import re
import shutil
import subprocess
import sys

import torch

torch_cuda = torch.version.cuda
nvcc = shutil.which("nvcc")
text = subprocess.check_output([nvcc, "--version"], text=True, stderr=subprocess.STDOUT)
match = re.search(r"release\s+([0-9]+\.[0-9]+)", text)
nvcc_cuda = match.group(1) if match else None
print(f"torch.__version__={torch.__version__}")
print(f"torch.version.cuda={torch_cuda}")
print(f"nvcc={nvcc}")
print(f"nvcc CUDA release={nvcc_cuda}")
if not torch_cuda or not nvcc_cuda or torch_cuda != nvcc_cuda:
    raise SystemExit(
        "CUDA ABI mismatch inside Docker image: "
        f"torch.version.cuda={torch_cuda!r}, nvcc={nvcc_cuda!r}. "
        "Use matching BASE_IMAGE and TORCH_INDEX_URL build args."
    )
PY

WORKDIR /opt/kwcoco_detector_kit

# Install third-party deps before copying fast-changing source. This keeps the
# expensive dependency layers cached across ordinary toolkit code edits.
COPY pyproject.toml README.md LICENSE /opt/kwcoco_detector_kit/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        "kwcoco>=0.8" \
        "kwimage>=0.11" \
        "kwutil>=0.3" \
        "ubelt>=1.4" \
        "kwconf>=0.11.0" \
        "numpy>=1.24" \
        "pyyaml>=6.0" \
        "onnx>=1.16" \
        "onnxruntime>=1.18" \
        "onnxscript>=0.1" \
        "onnxsim>=0.4" \
        "opencv-python-headless>=4.8" \
        "Pillow>=10.0" \
        "transformers>=4.35,<4.47" \
        "huggingface_hub>=0.27" \
        addict \
        yapf \
        pycocotools \
        matplotlib \
        timm \
        colorlog \
        jsonlines \
        "pytest>=8" \
        "pytest-cov>=4" \
        "faster-coco-eval>=1.6.7" \
        "tensorboard" \
        "scipy" \
        "calflops" \
        "safetensors"

# GDAL via Kitware's large-image wheel index (girder.github.io). The
# wheels bundle libgdal so we don't need libgdal-dev / gdal-bin on the
# system. kwcoco's `finish_install --with_gdal` resolves to the right
# pinned version + finds it on Kitware's wheel index.
# The kit's _preimport scaffold then loads osgeo.gdal before torch,
# which is what delayed_image needs to take its fast path.
RUN --mount=type=cache,target=/root/.cache/uv \
    ${PYTHON_BIN} -m kwcoco finish_install --with_gdal=True --strict=False

# Copy and build the upstream trainer before copying the rest of the kit. The
# CUDA extension depends on torch/nvcc + OpenGroundingDINO source, not on most
# kit edits, so this expensive layer now survives normal package changes.
COPY tpl/Open-GroundingDino /opt/kwcoco_detector_kit/tpl/Open-GroundingDino
RUN --mount=type=cache,target=/root/.cache/torch_extensions \
    test -f tpl/Open-GroundingDino/models/GroundingDINO/ops/setup.py || \
        (echo "Open-GroundingDino submodule contents are missing from Docker build context. Run: git submodule update --init tpl/Open-GroundingDino" >&2; exit 1) && \
    cd tpl/Open-GroundingDino/models/GroundingDINO/ops && \
    ${PYTHON_BIN} setup.py build_ext --inplace -v && \
    cp MultiScaleDeformableAttention*.so /opt/kwcoco_detector_kit/tpl/Open-GroundingDino/

# DEIMv2 is pure Python — no CUDA extension to compile. Copy after the
# heavier OGDino build so kit edits don't invalidate the OGDino layer.
COPY tpl/DEIMv2 /opt/kwcoco_detector_kit/tpl/DEIMv2
RUN test -f /opt/kwcoco_detector_kit/tpl/DEIMv2/train.py || \
        (echo "DEIMv2 submodule contents are missing from Docker build context. Run: git submodule update --init tpl/DEIMv2" >&2; exit 1)

# kwcoco_dataloader — pinned via submodule so the exact commit is baked
# into the image (provenance). Soft-imported by the WebDataset training
# path; the optional `webdataset` + `wids` deps install with it via the
# package's own pyproject extras. KCD_DEV_MOUNT_DATALOADER=1 at runtime
# overlays the host checkout the same way KCD_DEV_MOUNT_KIT does.
COPY tpl/kwcoco_dataloader /opt/kwcoco_detector_kit/tpl/kwcoco_dataloader
RUN test -f /opt/kwcoco_detector_kit/tpl/kwcoco_dataloader/kwcoco_dataloader/__init__.py || \
        (echo "kwcoco_dataloader submodule contents are missing. Run: git submodule update --init tpl/kwcoco_dataloader" >&2; exit 1)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e /opt/kwcoco_detector_kit/tpl/kwcoco_dataloader && \
    uv pip install "webdataset>=0.2" "wids>=0.1" braceexpand "line_profiler>=4.0"

COPY kwcoco_detector_kit /opt/kwcoco_detector_kit/kwcoco_detector_kit
COPY docs /opt/kwcoco_detector_kit/docs
COPY examples /opt/kwcoco_detector_kit/examples
COPY docker /opt/kwcoco_detector_kit/docker
COPY dev /opt/kwcoco_detector_kit/dev
COPY smoketests /opt/kwcoco_detector_kit/smoketests
COPY tests /opt/kwcoco_detector_kit/tests

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e "." && \
    ${PYTHON_BIN} -m kwcoco_detector_kit check-env \
        --groups core,opengroundingdino \
        --strict_import

# Run the fast unit tests inside the image to verify the build. Tests
# marked `requires_gpu` skip automatically (no GPU during docker build).
# Set --build-arg RUN_TESTS=0 to skip this step in CI / for quick rebuilds.
ARG RUN_TESTS=1
RUN if [ "${RUN_TESTS}" = "1" ]; then \
        cd /opt/kwcoco_detector_kit && \
        ${PYTHON_BIN} -m pytest tests/unit -m "not requires_gpu" -q --no-header; \
    else \
        echo "skipping in-image pytest run (RUN_TESTS=${RUN_TESTS})"; \
    fi

# Bake build provenance: the git SHAs of the kit + each submodule, the
# Dockerfile's own hash, and the build time. _provenance.py reads
# /etc/kcd_provenance.json, so a RUNNING container (and the eval metrics
# JSON) report exactly what produced the image — even if the image hash
# itself is later lost (never published, retagged, etc.). The LABELs make
# the same data visible via `docker inspect` without running the image.
# ARGs declared HERE (not at the top) so changing them only rebuilds this
# trailing layer, never the expensive dependency layers.
ARG KCD_KIT_SHA=unknown
ARG KCD_DEIMV2_SHA=unknown
ARG KCD_OGDINO_SHA=unknown
ARG KCD_DATALOADER_SHA=unknown
ARG KCD_DOCKERFILE_SHA=unknown
ARG KCD_BUILD_TIME=unknown
RUN printf '{\n  "kit_sha": "%s",\n  "deimv2_sha": "%s",\n  "opengroundingdino_sha": "%s",\n  "kwcoco_dataloader_sha": "%s",\n  "dockerfile_sha256": "%s",\n  "build_time": "%s"\n}\n' \
        "${KCD_KIT_SHA}" "${KCD_DEIMV2_SHA}" "${KCD_OGDINO_SHA}" \
        "${KCD_DATALOADER_SHA}" "${KCD_DOCKERFILE_SHA}" "${KCD_BUILD_TIME}" \
        > /etc/kcd_provenance.json && cat /etc/kcd_provenance.json
LABEL kcd.kit_sha="${KCD_KIT_SHA}" \
      kcd.deimv2_sha="${KCD_DEIMV2_SHA}" \
      kcd.opengroundingdino_sha="${KCD_OGDINO_SHA}" \
      kcd.kwcoco_dataloader_sha="${KCD_DATALOADER_SHA}" \
      kcd.dockerfile_sha256="${KCD_DOCKERFILE_SHA}" \
      kcd.build_time="${KCD_BUILD_TIME}"

COPY docker/opengroundingdino/entrypoint.sh /usr/local/bin/kcd-ogdino-entrypoint
COPY docker/opengroundingdino/kcd-pytest /usr/local/bin/kcd-pytest
RUN chmod +x /usr/local/bin/kcd-ogdino-entrypoint /usr/local/bin/kcd-pytest

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/kcd-ogdino-entrypoint"]
CMD ["/bin/bash"]
