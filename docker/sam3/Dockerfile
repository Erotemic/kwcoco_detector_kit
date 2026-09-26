# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV PATH=/opt/venv/bin:/root/.local/bin:${PATH}
ENV UV_LINK_MODE=copy UV_PYTHON=/opt/venv/bin/python UV_COMPILE_BYTECODE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TOKENIZERS_PARALLELISM=false

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl git libgl1 libglib2.0-0 python3 python3-venv && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

RUN --mount=type=cache,target=/root/.cache/uv \
    umask 022 && \
    uv python install --install-dir ${UV_PYTHON_INSTALL_DIR} ${PYTHON_VERSION} && \
    uv venv --seed --python ${PYTHON_VERSION} ${VIRTUAL_ENV} && \
    python_real="$(readlink -f ${VIRTUAL_ENV}/bin/python)" && \
    python_bin_dir="$(dirname "$python_real")" && \
    python_root="$(dirname "$python_bin_dir")" && \
    test ! "$python_real" = /root/* && \
    chmod a+rx /opt "${UV_PYTHON_INSTALL_DIR}" "$python_root" "$python_bin_dir" \
      "$python_real" "${VIRTUAL_ENV}" "${VIRTUAL_ENV}/bin" && \
    uv pip install --index-url ${TORCH_INDEX_URL} \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

WORKDIR /opt/kwcoco_detector_kit
COPY pyproject.toml README.md LICENSE ./
COPY tpl/sam3 ./tpl/sam3
RUN test -f tpl/sam3/sam3/model_builder.py || \
      (echo 'Initialize tpl/sam3 before building' >&2; exit 1)

# SAM3 is intentionally installed from KDK's pinned fork.  Its pyproject owns
# the inference dependency versions (including numpy<2); the checkpoint itself
# remains in the host Hugging Face cache and is not baked into this image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e ./tpl/sam3 && \
    uv pip install 'pycocotools>=2.0.7' 'pytest>=8' 'pytest-cov>=4'

COPY kwcoco_detector_kit ./kwcoco_detector_kit
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e .

ARG RUN_TESTS=1
RUN if [ "${RUN_TESTS}" = 1 ]; then \
      python -m pytest tests/unit/test_sam3_predictor_adapter.py -q; \
    fi

ARG KCD_KIT_SHA=unknown
ARG KCD_SAM3_SHA=unknown
ARG KCD_DOCKERFILE_SHA=unknown
ARG KCD_BUILD_TIME=unknown
RUN <<'EOF_PROV'
set -e
KCD_KIT_SHA="${KCD_KIT_SHA}" \
KCD_SAM3_SHA="${KCD_SAM3_SHA}" \
KCD_DOCKERFILE_SHA="${KCD_DOCKERFILE_SHA}" \
KCD_BUILD_TIME="${KCD_BUILD_TIME}" \
python - <<'PY'
import importlib.metadata as md
import json
import os
from pathlib import Path
import torch

prov = {
    'kit_sha': os.environ['KCD_KIT_SHA'],
    'sam3_sha': os.environ['KCD_SAM3_SHA'],
    'dockerfile_sha256': os.environ['KCD_DOCKERFILE_SHA'],
    'build_time': os.environ['KCD_BUILD_TIME'],
    'torch_version': torch.__version__,
    'torch_cuda_version': torch.version.cuda,
    'python_packages': {
        name: md.version(name)
        for name in ['kwcoco', 'kwconf', 'delayed-image', 'sam3']
    },
}
Path('/etc/kcd_provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
print(json.dumps(prov, indent=2))
PY
EOF_PROV

LABEL kcd.kit_sha="${KCD_KIT_SHA}" kcd.sam3_sha="${KCD_SAM3_SHA}" \
      kcd.dockerfile_sha256="${KCD_DOCKERFILE_SHA}" kcd.build_time="${KCD_BUILD_TIME}" \
      kcd.runtime_uid_safe="1" kcd.sam3_admission="1"

WORKDIR /opt/kwcoco_detector_kit
CMD ["bash"]
