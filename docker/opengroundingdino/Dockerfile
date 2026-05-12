ARG BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_PRE=0

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PYTHON_BIN=/opt/venv/bin/python
ENV KCD_OPENGROUNDINGDINO_REPO_DPATH=/opt/kwcoco_detector_kit/tpl/Open-GroundingDino
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    pkg-config \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

RUN uv venv --seed --python ${PYTHON_VERSION} ${VIRTUAL_ENV} && \
    ${PYTHON_BIN} -m pip --version

RUN ${PYTHON_BIN} -m pip install -U pip wheel setuptools && \
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
COPY . /opt/kwcoco_detector_kit

RUN ${PYTHON_BIN} -m pip install -e ".[opengroundingdino]" && \
    ${PYTHON_BIN} -m kwcoco_detector_kit check-env \
        --groups core,opengroundingdino \
        --strict_import

RUN cd tpl/Open-GroundingDino/models/GroundingDINO/ops && \
    ${PYTHON_BIN} setup.py build_ext --inplace -v && \
    cp MultiScaleDeformableAttention*.so /opt/kwcoco_detector_kit/tpl/Open-GroundingDino/

COPY docker/opengroundingdino/entrypoint.sh /usr/local/bin/kcd-ogdino-entrypoint
RUN chmod +x /usr/local/bin/kcd-ogdino-entrypoint

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/kcd-ogdino-entrypoint"]
CMD ["/bin/bash"]
