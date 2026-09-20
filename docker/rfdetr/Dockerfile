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
COPY tpl/rf-detr ./tpl/rf-detr
RUN test -f tpl/rf-detr/src/rfdetr/__init__.py || \
      (echo 'Initialize tpl/rf-detr before building' >&2; exit 1)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e './tpl/rf-detr[train,loggers]' && \
    uv pip install 'kwcoco>=0.8' 'kwimage>=0.11' 'kwimage_ext>=0.3.3' \
      'kwutil>=0.3' 'ubelt>=1.4' 'kwconf>=0.11.0' \
      'opencv-python-headless>=4.8' 'pytest>=8' 'pytest-cov>=4'

# Install GDAL through KWCoco's supported large-image wheel route. This uses
# Kitware's wheel index so the Python binding and bundled libgdal stay aligned
# without adding a distro libgdal-dev toolchain to the detector image.
RUN --mount=type=cache,target=/root/.cache/uv \
    python -m kwcoco finish_install \
      --with_gdal=True --with_cv2_headless=False --strict=False && \
    python -c "import kwimage_ext; from osgeo import gdal; print('kwimage_ext:', kwimage_ext.__file__); print('gdal:', gdal.VersionInfo())"

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
RUN printf '{\n  "kit_sha": "%s",\n  "rfdetr_sha": "%s",\n  "dockerfile_sha256": "%s",\n  "build_time": "%s"\n}\n' \
      "${KCD_KIT_SHA}" "${KCD_RFDETR_SHA}" "${KCD_DOCKERFILE_SHA}" "${KCD_BUILD_TIME}" \
      > /etc/kcd_provenance.json
LABEL kcd.kit_sha="${KCD_KIT_SHA}" kcd.rfdetr_sha="${KCD_RFDETR_SHA}" \
      kcd.dockerfile_sha256="${KCD_DOCKERFILE_SHA}" kcd.build_time="${KCD_BUILD_TIME}" \
      kcd.runtime_uid_safe="1" kcd.prediction_fastpaths="1"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["exec /bin/bash"]
