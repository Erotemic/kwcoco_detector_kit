#!/usr/bin/env bash
# Developer setup for kwcoco_detector_kit.
#
# Assumes you have ALREADY activated the python env you want to use
# (e.g. via `uv venv && source .venv/bin/activate`, conda activate,
# pyenv, etc.). This script just:
#
#   1. Initialises and updates the git submodules we actually use
#      (tpl/DEIMv2, tpl/kwcoco_dataloader; Open-GroundingDino is
#      legacy and skipped — clone it manually if you need it).
#   2. Pins each submodule to its tracking branch so a later
#      `git push` from the kit root recurses cleanly.
#   3. Installs the kit in editable mode with all runtime + dev
#      extras (deimv2 hidden deps, webdataset, kwcoco-dataloader,
#      tests).
#   4. Installs the local tpl/kwcoco_dataloader checkout in
#      editable mode AFTER the kit's pip resolve, so the dev copy
#      shadows whatever pypi version got pulled by `[kwcoco-dataloader]`.
#
# Re-runnable. Safe to invoke after a `git pull` to pick up new deps
# or a bumped submodule pin.
#
# Usage (from kit root, with your venv active):
#   bash run_developer_setup.sh
#
# Override the python to install into:
#   PYTHON_BIN=/path/to/python bash run_developer_setup.sh
#
# Skip a stage (useful for fast re-runs):
#   SETUP_SKIP_SUBMODULES=1 bash run_developer_setup.sh
#   SETUP_SKIP_PIP=1        bash run_developer_setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: no python found on PATH. Activate your venv first, or set PYTHON_BIN." >&2
    exit 1
fi

echo "=== developer setup for kwcoco_detector_kit ==="
echo "  repo: $REPO_ROOT"
echo "  python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo

# ---------- 1. submodules ----------------------------------------------
if [ "${SETUP_SKIP_SUBMODULES:-0}" = "1" ]; then
    echo "=== 1. submodules: skipped (SETUP_SKIP_SUBMODULES=1) ==="
else
    echo "=== 1. submodules ==="
    # We actively develop against these two; Open-GroundingDino is
    # legacy and is skipped to avoid pulling its large weights every
    # time someone runs this script.
    ACTIVE_SUBMODULES=(
        tpl/DEIMv2
        tpl/kwcoco_dataloader
    )
    for sm in "${ACTIVE_SUBMODULES[@]}"; do
        echo "  init/update $sm"
        git submodule update --init --recursive "$sm"
    done

    # Apply branch tracking (same logic as scripts/setup_git.sh step 6)
    # so submodule pushes target the right branch on the fork.
    echo "  configuring submodule tracking branches"
    git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | while read -r key path; do
        # Only touch submodules we just initialised.
        skip=1
        for sm in "${ACTIVE_SUBMODULES[@]}"; do
            if [ "$path" = "$sm" ]; then
                skip=0
                break
            fi
        done
        [ "$skip" = "1" ] && continue

        name=$(echo "$key" | sed -e 's/^submodule\.//' -e 's/\.path$//')
        cfg_branch=$(git config -f .gitmodules --get "submodule.${name}.branch" 2>/dev/null || true)
        if [ -z "$cfg_branch" ]; then
            echo "    $path: no branch declared, skipping"
            continue
        fi
        (
            cd "$path"
            git fetch --quiet origin "$cfg_branch" 2>/dev/null || true
            pinned=$(git rev-parse HEAD)
            git checkout --quiet -B "$cfg_branch" "$pinned"
            git branch --set-upstream-to="origin/$cfg_branch" "$cfg_branch" 2>/dev/null || true
            echo "    $path: on $cfg_branch at $(git rev-parse --short HEAD)"
        )
    done
fi

# ---------- 2. pip install --------------------------------------------
if [ "${SETUP_SKIP_PIP:-0}" = "1" ]; then
    echo
    echo "=== 2. pip install: skipped (SETUP_SKIP_PIP=1) ==="
else
    echo
    echo "=== 2. tpl/kwcoco_dataloader (editable, FIRST) ==="
    # MUST install this before the kit. The kit's [kwcoco-dataloader]
    # extra pins kwcoco-dataloader>=0.1.3 — but the highest pypi-published
    # version is 0.1.2, so resolving the extra against pypi fails with
    #   ERROR: No matching distribution found for kwcoco-dataloader>=0.1.3
    # Installing the local 0.1.3 checkout first satisfies the constraint
    # AND lets you edit reader/writer code with changes picked up live.
    "$PYTHON_BIN" -m pip install -e tpl/kwcoco_dataloader

    echo
    echo "=== 3. kit (editable + all extras) ==="
    # The deimv2 extra carries the hidden transitives DEIMv2 imports
    # (calflops → transformers, tensorboard, faster_coco_eval, scipy).
    # The webdataset extra pulls webdataset + braceexpand for the WDS
    # reader. The dev extra adds pytest. We OMIT [kwcoco-dataloader]
    # here on purpose — it would try to redownload from pypi over our
    # editable install. The opengroundingdino extra is also omitted
    # (heavy + legacy); install manually if you need it.
    "$PYTHON_BIN" -m pip install -e ".[deimv2,webdataset,dev]"
fi

# ---------- 3. verify --------------------------------------------------
echo
echo "=== 4. verify imports ==="
"$PYTHON_BIN" - <<'PYEOF'
import importlib, sys, traceback

REQUIRED = [
    ("kwcoco_detector_kit", "kit package"),
    ("kwcoco_dataloader",   "WDS reader/writer (editable from tpl/)"),
    ("torch",               "deep learning"),
    ("torchvision",         "ABI-matched torchvision"),
    ("kwcoco",              "kwcoco core"),
    ("kwimage",             "kwimage"),
    ("tensorboard",         "training logs"),
    ("transformers",        "calflops dep"),
    ("calflops",            "DEIMv2 FLOPs profiler"),
    ("onnx",                "ONNX repack"),
    ("onnxscript",          "torch.onnx.export"),
    ("onnxruntime",         "ONNX inference / bench"),
    ("webdataset",          "WDS reader backend"),
    ("PIL",                 "image decode"),
    ("pytest",              "test runner"),
]

missing = []
for mod_name, why in REQUIRED:
    try:
        mod = importlib.import_module(mod_name)
        loc = getattr(mod, "__file__", "<builtin>") or "<builtin>"
        ver = getattr(mod, "__version__", "")
        print(f"  OK  {mod_name:24s} {ver:10s} {loc}")
    except Exception as e:
        missing.append((mod_name, why, e))
        print(f"  FAIL {mod_name:24s} {why}: {e}")

if missing:
    print()
    print(f"{len(missing)} import(s) failed. Re-run after fixing the env, or set", file=sys.stderr)
    print("PYTHON_BIN to a different python.", file=sys.stderr)
    sys.exit(1)

# Sanity-check that DEIMv2 is sys.path-reachable. We don't pip install
# it (it has no setup.py), so the test/training code does
# sys.path.insert(0, 'tpl/DEIMv2') at use-time. Confirm the file is
# present so a stale submodule fails fast here.
from pathlib import Path
deimv2_marker = Path("tpl/DEIMv2/engine/data/dataset/wds_coco_dataset.py")
if deimv2_marker.exists():
    print(f"  OK  tpl/DEIMv2 checkout present (wds_coco_dataset.py found)")
else:
    print(f"  FAIL tpl/DEIMv2 missing wds_coco_dataset.py — re-run with SETUP_SKIP_SUBMODULES=0")
    sys.exit(1)

print()
print("All checks passed.")
PYEOF

echo
echo "=== ready ==="
echo
echo "Quick next steps:"
echo "  - Smoke the WDS pipeline (CPU, ~17s):"
echo "      bash dev/wds_e2e_demo/run_demo.sh"
echo "  - Run integration tests:"
echo "      $PYTHON_BIN -m pytest tests/integration/ -v"
echo "  - See projects/viame_sealions_2026/AGENT.md for the sealion workflow."
