#!/usr/bin/env bash
# Host-side environment bootstrap for the VIAME sea-lion OpenGroundingDINO run.
#
# Run this on the machine with the RTX 3090. It intentionally does not run
# training; it installs Python deps, builds OpenGroundingDINO's deformable
# attention extension, and downloads the Swin-T GroundingDINO checkpoint.
set -euo pipefail

KIT_DPATH="${KIT_DPATH:-/home/joncrall/code/kwcoco_detector_kit}"
DATA_DPATH="${DATA_DPATH:-/media/joncrall/raid/home/joncrall/data/dvc-repos/viame_sealions_2026}"
KCD_ROOT="${KCD_ROOT:-$DATA_DPATH/training_runs/ogdino_swint_3090}"
KCD_CACHE_ROOT="${KCD_CACHE_ROOT:-$DATA_DPATH/training_runs/cache/ogdino_swint}"
PYTHON_BIN="${PYTHON_BIN:-python}"
KCD_TORCH_INDEX_URL="${KCD_TORCH_INDEX_URL:-}"

mkdir -p "$KCD_CACHE_ROOT/pretrained"

echo "=== Install kwcoco-detector-kit + OpenGroundingDINO deps ==="
"$PYTHON_BIN" -m pip install -U pip wheel setuptools
if [ -n "$KCD_TORCH_INDEX_URL" ]; then
    "$PYTHON_BIN" -m pip install --force-reinstall torch torchvision torchaudio \
        --index-url "$KCD_TORCH_INDEX_URL"
fi
"$PYTHON_BIN" -m pip install -e "$KIT_DPATH[opengroundingdino]"
"$PYTHON_BIN" -m kwcoco_detector_kit check-env \
    --groups core,opengroundingdino \
    --install \
    --strict_import

echo
echo "=== Check PyTorch / nvcc CUDA ABI match ==="
"$PYTHON_BIN" - <<'PY'
import re
import shutil
import subprocess
import sys

import torch

torch_cuda = torch.version.cuda
nvcc = shutil.which('nvcc')
if not nvcc:
    raise SystemExit(
        'nvcc was not found on PATH. Install a CUDA toolkit, or set CUDA_HOME '
        'and PATH so nvcc is visible before building OpenGroundingDINO ops.'
    )
text = subprocess.check_output([nvcc, '--version'], text=True, stderr=subprocess.STDOUT)
match = re.search(r'release\s+([0-9]+\.[0-9]+)', text)
nvcc_cuda = match.group(1) if match else None
print(f'torch.__version__ = {torch.__version__}')
print(f'torch.version.cuda = {torch_cuda}')
print(f'nvcc = {nvcc}')
print(f'nvcc CUDA release = {nvcc_cuda}')
if not torch_cuda or not nvcc_cuda or torch_cuda != nvcc_cuda:
    nvcc_cuda_compact = str(nvcc_cuda).replace('.', '')
    raise SystemExit(
        '\nCUDA mismatch: the PyTorch wheel was compiled for CUDA '
        f'{torch_cuda}, but nvcc is CUDA {nvcc_cuda}.\n'
        'Option A: put a matching CUDA toolkit first on PATH. For your '
        'current PyTorch this means nvcc should report CUDA '
        f'{torch_cuda}.\n'
        'Option B: reinstall PyTorch to match the nvcc you already have:\n'
        f'  {sys.executable} -m pip install --force-reinstall '
        'torch torchvision torchaudio '
        f'--index-url https://download.pytorch.org/whl/cu{nvcc_cuda_compact}\n'
        'Rerun this setup script after either change.'
    )
PY

echo
echo "=== Build OpenGroundingDINO deformable attention extension ==="
pushd "$KIT_DPATH/tpl/Open-GroundingDino/models/GroundingDINO/ops" >/dev/null
"$PYTHON_BIN" setup.py build_ext --inplace -v
popd >/dev/null
cp "$KIT_DPATH"/tpl/Open-GroundingDino/models/GroundingDINO/ops/MultiScaleDeformableAttention*.so \
   "$KIT_DPATH"/tpl/Open-GroundingDino/

echo
echo "=== Download Swin-T GroundingDINO checkpoint if needed ==="
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-$KCD_CACHE_ROOT/pretrained/groundingdino_swint_ogc.pth}"
if [ ! -f "$PRETRAIN_MODEL_PATH" ]; then
    if command -v curl >/dev/null 2>&1; then
        curl -L \
            -o "$PRETRAIN_MODEL_PATH" \
            https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    else
        wget \
            -O "$PRETRAIN_MODEL_PATH" \
            https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    fi
fi

echo
echo "Ready."
echo "  export KIT_DPATH=$KIT_DPATH"
echo "  export KCD_ROOT=$KCD_ROOT"
echo "  export KCD_CACHE_ROOT=$KCD_CACHE_ROOT"
echo "  export PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH"
