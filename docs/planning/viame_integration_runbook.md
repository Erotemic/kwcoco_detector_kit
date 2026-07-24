# VIAME integration runbook — build, run, and score a kit ONNX model

Operational companion to [viame_integration.md](viame_integration.md). That
doc is the design; this is the "how do I actually run it on namek" checklist.

The integration is intentionally thin: the kit exports a self-describing ONNX
package, and a VIAME kwiver plugin (`kwcoco_detector_kit_detector.py`) runs it
through `onnxruntime` with **no PyTorch at inference time**.

**VIAME inference does not depend on `kwcoco_detector_kit` being installed.** The
`OnnxPredictor` is *vendored* into VIAME — the kit owns the canonical copy and a
vendoring tool stamps provenance into the VIAME copy, so VIAME can pin/resync
deliberately instead of tracking the fast-moving kit live (the kit may break
ONNX-package compatibility on purpose as it evolves).

- Plugin + template + the **vendored predictor** + tests live on the VIAME branch
  **`kwcoco-detector-kit-main`** (based on VIAME `main`):
  - `plugins/pytorch/kwcoco_detector_kit_detector.py`
  - `plugins/pytorch/kwcoco_detector_kit_onnx_predictor.py` ← vendored `OnnxPredictor`
  - `configs/pipelines/templates/detector_kwcoco_detector_kit.pipe`
  - both `.py` registered (unguarded) in `plugins/pytorch/CMakeLists.txt`
- The kit side owns the canonical predictor, the vendoring tool, the runtime-dep
  setup script, and a standalone smoke test:
  - `kwcoco_detector_kit/predictors/onnx.py` → `OnnxPredictor` (**canonical**)
  - `dev/vendor_onnx_to_viame.py` → re-vendors the predictor into VIAME (`--check` detects drift)
  - `dev/viame_container_setup.sh` → installs onnxruntime + kwconf into VIAME python
  - `dev/viame_onnx_smoke.py`

## 0. Prerequisite — an exported ONNX package

On the training host, export a checkpoint to a package directory:

```bash
# on the GPU box where the workdir lives
python -m kwcoco_detector_kit export-onnx /data/users/jon.crall/kcd_sealion/<workdir>
# → writes <workdir>/export/<model>.onnx + <model>.modelspec.json + <model>.labels.txt
```

The `.modelspec.json` sidecar carries input size, preprocessing, score
threshold, and `category_names`, so the VIAME side needs no extra config.

## 1. Build the dev image (namek)

The build clones whatever branch the host repo currently has checked out, so
check out the integration branch first.

```bash
# namek
cd $HOME/code/VIAME
git checkout kwcoco-detector-kit-main

DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "viame:viame-gpu-local" \
    -f docker/viame_gpu_local.docker .
```

This is a long, full-source VIAME + torch build (hours, first time). The
`kwcoco_detector_kit_detector.py` plugin is baked in because its CMake
registration is unguarded — no extra `VIAME_ENABLE_*` flag is required.

## 2. Run the container with BOTH repos mounted

The kit is a separate repo, so mount it alongside VIAME.

```bash
# namek
docker run --gpus=all \
    --shm-size=8g \
    --volume "$HOME/code/VIAME:/host-viame" \
    --volume "$HOME/code/kwcoco_detector_kit:/host-kwcoco-detector-kit" \
    --volume "/data/users/jon.crall:/data/users/jon.crall" \
    -it viame:viame-gpu-local bash
```

The third mount makes the exported ONNX package (under `/data/...`) visible at
the same path inside the container.

## 3. Install runtime deps (one-time per container)

The plugin and the vendored predictor are already baked into the image by the
build. Only two torch-free packages are missing from a stock VIAME python —
`onnxruntime` and `kwconf` (the plugin's config lib):

```bash
# inside the container
source /opt/noaa/viame/setup_viame.sh
bash viame_container_setup.sh   # (copy the script in, or mount the kit just for this)
```

No kit install, no editable mount, no symlinks. The script verifies the
vendored predictor imports (printing its `kit_git_sha` provenance) and the
plugin registers. If `onnxruntime-gpu` mismatches the container's cuDNN, rerun
with `KCD_ORT_PACKAGE=onnxruntime` for a CPU-only plumbing test.

## 4. Standalone smoke test (no pipeline)

Fastest signal that the model loads and runs end-to-end:

```bash
# inside the container
python /host-kwcoco-detector-kit/dev/viame_onnx_smoke.py \
    --package /data/users/jon.crall/kcd_sealion/<workdir>/export \
    --viame
```

`--viame` additionally drives the kwiver plugin (`set_configuration` +
`detect`). Add `--image <path>` to run on a real frame instead of synthetic
noise.

## 5. Run the VIAME pipeline → detections CSV

Instantiate the template with a real package path and an image list.

```bash
# inside the container
cd /opt/noaa/viame/examples/object_detection

cp /opt/noaa/viame/configs/pipelines/templates/detector_kwcoco_detector_kit.pipe \
   demo_kwcoco_detector_kit.pipe

PKG=/data/users/jon.crall/kcd_sealion/<workdir>/export
sed -i "s|\[-PACKAGE-\]|$PKG|g" demo_kwcoco_detector_kit.pipe

# point the pipe at an image list (one image path per line). Use your own
# sealion eval list; the bundled small set just proves the plumbing.
viame demo_kwcoco_detector_kit.pipe \
    -s input:video_filename=input_image_list_small_set.txt \
    -s detector1:detector:kwcoco_detector_kit:device=cuda \
    -s detector_writer:file_name=computed_detections.csv
```

Output is `computed_detections.csv` in VIAME CSV format. On non-sealion
imagery expect few/no detections — that still validates the full path
(reader → preprocess → ONNX → kwiver detections → NMS → CSV writer).

## 6. Score (next milestone)

Scoring compares `computed_detections.csv` against a ground-truth CSV with
VIAME's detection scorer (`score_results` / kwant). The missing piece is GT in
VIAME CSV format:

- Convert the eval kwcoco bundle's annotations to VIAME CSV (kwcoco geometry →
  VIAME CSV rows), keeping `category_names` order aligned with the export's
  `labels.txt`.
- Remember the project scoring conventions from kit memory: **drop NFS from
  GT+pred before scoring**, and **class-agnostic detection AP is the selection
  criterion** (per-class AP is diagnostic only).
- Then run VIAME's scorer on (GT csv, computed_detections.csv).

The kwcoco→VIAME-CSV converter is the only remaining code to write for a
closed scoring loop; inference (steps 1–5) is complete.

## Editing the predictor → re-vendor

The canonical predictor is `kwcoco_detector_kit/predictors/onnx.py` in the kit.
After editing it, re-vendor the copy into VIAME and rebuild (or re-copy the file
into the container's `viame.pytorch` package):

```bash
# in the kwcoco_detector_kit repo
python dev/vendor_onnx_to_viame.py --viame-root ~/code/VIAME

# CI / pre-commit guard — fail if the vendored copy drifted from canonical:
python dev/vendor_onnx_to_viame.py --viame-root ~/code/VIAME --check
```

The vendored file carries a provenance header (`__vendored_provenance__` with
the kit git SHA + a content hash of the source); `--check` recomputes the hash
to detect drift. Never edit the vendored copy by hand.
