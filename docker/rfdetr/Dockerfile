# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvidia/cuda:13.2.0-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu132
ARG TORCH_PRE=1
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV PATH=/opt/venv/bin:/root/.local/bin:${PATH}
ENV UV_LINK_MODE=copy UV_PYTHON=/opt/venv/bin/python UV_COMPILE_BYTECODE=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl git python3 python3-venv && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

RUN --mount=type=cache,target=/root/.cache/uv \
    umask 022 && \
    uv python install --install-dir ${UV_PYTHON_INSTALL_DIR} ${PYTHON_VERSION} && \
    uv venv --seed --python ${PYTHON_VERSION} ${VIRTUAL_ENV} && \
    python_real="$(readlink -f ${VIRTUAL_ENV}/bin/python)" && \
    python_bin_dir="$(dirname "$python_real")" && \
    python_root="$(dirname "$python_bin_dir")" && \
    test ! "$python_real" = /root/* && \
    chmod a+rx \
      /opt \
      "${UV_PYTHON_INSTALL_DIR}" \
      "$python_root" \
      "$python_bin_dir" \
      "$python_real" \
      "${VIRTUAL_ENV}" \
      "${VIRTUAL_ENV}/bin" && \
    if [ "${TORCH_PRE}" = 1 ]; then \
      uv pip install --prerelease=allow --index-url ${TORCH_INDEX_URL} torch torchvision; \
    else \
      uv pip install --index-url ${TORCH_INDEX_URL} torch torchvision; \
    fi

WORKDIR /opt/kwcoco_detector_kit
COPY pyproject.toml README.md LICENSE ./
COPY docker/rfdetr/requirements-kwstack.txt ./docker/rfdetr/requirements-kwstack.txt
COPY tpl/rf-detr ./tpl/rf-detr
RUN test -f tpl/rf-detr/src/rfdetr/__init__.py || \
      (echo 'Initialize tpl/rf-detr before building' >&2; exit 1)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e './tpl/rf-detr[train,loggers]' && \
    uv pip install -r docker/rfdetr/requirements-kwstack.txt && \
    uv pip install 'kwimage>=0.11' 'kwutil>=0.3' 'ubelt>=1.4' \
      'opencv-python-headless>=4.8' 'pytest>=8' 'pytest-cov>=4'

# Install GDAL through KWCoco's supported large-image wheel route. KWCoco's
# finish_install prefers uv when it is available and points GDAL at Kitware's
# large-image wheel index, avoiding a distro libgdal-dev toolchain. Isolate the
# installer from host/container pip configuration so ambient index settings do
# not silently change how the GDAL wheel is resolved.
RUN --mount=type=cache,target=/root/.cache/uv <<'EOF'
set -e
python -m kwcoco finish_install \
    --with_gdal=True \
    --with_cv2_headless=False \
    --strict=False \
    --prefer_uv=True \
    --isolated_installer=True
python - <<'PY'
import importlib.metadata as md
from osgeo import gdal

for name in ['kwcoco', 'kwimage-ext', 'kwconf', 'delayed-image']:
    print(f'{name}: {md.version(name)}')
print('gdal:', gdal.VersionInfo())
PY
EOF

COPY kwcoco_detector_kit ./kwcoco_detector_kit
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps -e .

ARG RUN_TESTS=1
RUN if [ "${RUN_TESTS}" = 1 ]; then \
      python -m pytest tests/unit/test_rfdetr_config_gen.py \
        tests/unit/test_tile_geometry.py tests/unit/test_tile_cache.py -q; \
    fi

ARG KCD_KIT_SHA=unknown
ARG KCD_RFDETR_SHA=unknown
ARG KCD_DOCKERFILE_SHA=unknown
ARG KCD_BUILD_TIME=unknown
RUN <<'EOF'
set -e
python - <<'PY'
import importlib.metadata as md
import json
import os
from pathlib import Path
from osgeo import gdal

prov = {
    'kit_sha': os.environ['KCD_KIT_SHA'],
    'rfdetr_sha': os.environ['KCD_RFDETR_SHA'],
    'dockerfile_sha256': os.environ['KCD_DOCKERFILE_SHA'],
    'build_time': os.environ['KCD_BUILD_TIME'],
    'python_packages': {
        name: md.version(name)
        for name in ['kwcoco', 'kwimage-ext', 'kwconf', 'delayed-image']
    },
    'gdal_version': gdal.VersionInfo(),
}
Path('/etc/kcd_provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
print(json.dumps(prov, indent=2))
PY
EOF

LABEL kcd.kit_sha="${KCD_KIT_SHA}" kcd.rfdetr_sha="${KCD_RFDETR_SHA}" \
      kcd.dockerfile_sha256="${KCD_DOCKERFILE_SHA}" kcd.build_time="${KCD_BUILD_TIME}" \
      kcd.runtime_uid_safe="1" kcd.prediction_fastpaths="1"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["exec /bin/bash"]
