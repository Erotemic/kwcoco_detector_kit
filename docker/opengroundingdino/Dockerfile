# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_PRE=0
ARG PIP_CACHE_DIR=/root/.cache/pip
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
ENV PIP_CACHE_DIR=${PIP_CACHE_DIR}
ENV UV_CACHE_DIR=${UV_CACHE_DIR}
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
    ${PYTHON_BIN} -m pip --version

RUN --mount=type=cache,target=/root/.cache/pip \
    ${PYTHON_BIN} -m pip install -U pip wheel setuptools && \
    if [ "${TORCH_PRE}" = "1" ]; then \
        ${PYTHON_BIN} -m pip install --pre torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}; \
    else \
        ${PYTHON_BIN} -m pip install torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}; \
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
RUN --mount=type=cache,target=/root/.cache/pip \
    ${PYTHON_BIN} -m pip install \
        "kwcoco>=0.8" \
        "kwimage>=0.11" \
        "kwutil>=0.3" \
        "ubelt>=1.4" \
        "scriptconfig>=0.9" \
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

COPY kwcoco_detector_kit /opt/kwcoco_detector_kit/kwcoco_detector_kit
COPY docs /opt/kwcoco_detector_kit/docs
COPY examples /opt/kwcoco_detector_kit/examples
COPY docker /opt/kwcoco_detector_kit/docker
COPY dev /opt/kwcoco_detector_kit/dev
COPY smoketests /opt/kwcoco_detector_kit/smoketests
COPY tests /opt/kwcoco_detector_kit/tests

RUN --mount=type=cache,target=/root/.cache/pip \
    ${PYTHON_BIN} -m pip install --no-deps -e "." && \
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

COPY docker/opengroundingdino/entrypoint.sh /usr/local/bin/kcd-ogdino-entrypoint
COPY docker/opengroundingdino/kcd-pytest /usr/local/bin/kcd-pytest
RUN chmod +x /usr/local/bin/kcd-ogdino-entrypoint /usr/local/bin/kcd-pytest

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/kcd-ogdino-entrypoint"]
CMD ["/bin/bash"]
