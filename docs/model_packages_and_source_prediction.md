# Portable model packages and source-space prediction

KDK's normal inference boundary is a self-describing model package plus a
source KWCoco dataset. Prediction does not require the original training
workdir or a pre-materialized tile corpus.

## Package contract

`kwcoco-detector-kit package-build` asks the trainer plugin to select its
canonical checkpoint. The selected basename is preserved in the package (for
RF-DETR this matters because `checkpoint_best_total.pth`,
`checkpoint_best_ema.pth`, and `last.ckpt` have explicit loader semantics).
Package schema `kwcoco_detector_kit.package.v2` records:

- trainer/backend and variant;
- exact category order;
- box/mask capabilities;
- checkpoint filename, SHA-256, and size;
- generated training config and policy;
- inference defaults (window size, overlap, NMS);
- postprocess defaults;
- optional ONNX artifact, SHA-256, contract, mask capability, and parity result;
- KDK/framework versions and source commit when available;
- optional training-manifest paths/hashes as provenance only.

Absolute training paths may appear as provenance, but no package load path
depends on the original workdir.

For RF-DETR segmentation, export uses upstream's native segmentation ONNX
contract. ONNX is not preferred merely because an `.onnx` file exists. Auto
backend selection requires a passing postprocessed parity report and every
capability requested by the package. A segmentation package therefore cannot
silently fall back to box-only ONNX.

Representative packaging command:

```bash
kwcoco-detector-kit package-build \
    --workdir /path/to/immutable/run-snapshot \
    --trainer rfdetr \
    --variant seg_2xlarge \
    --category-names poop \
    --export-onnx \
    --score-thresh 0.01 \
    --verify-parity \
    --parity-src /path/to/train.kwcoco.zip \
    --export-device cuda:0 \
    --out /path/to/model.zip
```

Parity compares postprocessed labels, confidence scores, boxes, and masks when
the ONNX contract claims mask support. Real KWCoco windows should be supplied
for release packages; synthetic inputs are only a CI fallback.

## No-cache source-space tiled prediction

```bash
kwcoco-detector-kit predict \
    --model /path/to/model.zip \
    --src /path/to/source.kwcoco.zip \
    --dst /path/to/pred.kwcoco.zip \
    --device cuda:0 \
    --windowed=true \
    --window 768 \
    --overlap 0.25 \
    --batch-size 16
```

The package can supply those window defaults, so `--windowed`, `--window`, and
`--overlap` may usually be omitted.

`SourceWindowReader` realizes each source image for only the duration of the
prediction pass:

- JPEG/PNG/WebP-like assets use `decode_once`: finalize once, serve all windows
  as NumPy slices, then release the image;
- TIFF/COG-like assets use `delayed_region`: request delayed crops so a
  region-readable backend can avoid whole-image decode;
- if regional crop realization is unsupported, the reader switches to
  decode-once rather than repeatedly decoding the same source.

The promoted `predictors.TiledPredictor` owns window planning, batched detector
calls, coordinate translation, native-mask reconstruction, per-window
reduction, and cross-window per-class NMS. Evaluation imports the same
implementation through a compatibility shim; it does not maintain a second
geometry engine.

The ordinary `predict` path is pipelined by default. It uses three bounded
stages while keeping CUDA ownership on the main thread:

```text
source/window I/O threads
        ↓ bounded prefetch
GPU detector inference (main thread only)
        ↓ bounded completed-source queue
CPU merge/NMS + mask polygonization workers
        ↓ deterministic in-order commit
prediction KWCoco
```

This deliberately uses threads rather than multiprocessing for source/window
realization. The expensive codecs/GDAL/NumPy operations release the GIL, while
threads keep decoded images and 768-window NumPy arrays in shared memory instead
of serializing tens of MiB per model batch through process IPC. The window stage
uses one producer per active source reader, avoiding assumptions about concurrent
access to a single GDAL/delayed-image object. CUDA models are never copied into a
worker.

All queues are bounded. `source_prefetch` and `window_prefetch` mean the number
of future items retained beyond the item currently owned by the GPU thread;
`postprocess_inflight` bounds completed source results waiting on CPU work. CPU
workers may finish out of order, but annotations are committed in source order so
output is deterministic and worker exceptions surface at a stable boundary.

The defaults are conservative and suitable for a single local GPU:

```text
pipeline=true
source_workers=2
source_prefetch=2
window_prefetch=2
postprocess_workers=1
postprocess_inflight=2
```

Tune without changing code, for example:

```bash
kwcoco-detector-kit predict \
    --model=model.zip \
    --src=source.kwcoco.zip \
    --dst=pred.kwcoco.zip \
    --device=cuda:0 \
    --windowed=true \
    --batch-size=16 \
    --source-workers=4 \
    --source-prefetch=3 \
    --window-prefetch=3 \
    --postprocess-workers=2 \
    --postprocess-inflight=3
```

Use `--pipeline=false` for a serial diagnostic baseline. The emitted
`*.profile.json` separates stage *work* time from main-thread *wait* time for
source preparation, window realization, GPU inference, CPU postprocessing, and
annotation commit. Because stages overlap, work-time totals are not expected to
sum to wall time. High `source_prefetch_wait_seconds` or
`window_prefetch_wait_seconds` means the GPU is starving for pixels; high
`postprocess_wait_seconds` means CPU finalization is applying backpressure.

For native segmentation, crop masks remain crop-sized through score filtering,
per-window reduction, and cross-window box NMS. KDK only reconstructs a
full-source mask for detections that survive NMS/max-detection capping. This is
important for low-score annotation-QA passes, where expanding every candidate
mask before suppression can dominate host memory and CPU time.

Prediction KWCoco preserves source image IDs and source assets, contains only
prediction annotations, and records package/checkpoint/ONNX hashes, backend
actually used, source dataset hash, resolved inference configuration, and timing
in dataset `info`.

## Truth semantics

A source dataset may contain categories that are not detector classes. KDK
represents that explicitly with `TruthSemantics`:

- `target`: emitted as positive detector supervision;
- `background`: known non-target/distractor annotation; not emitted as a model
  class but remains legal negative evidence;
- `ignore`: uncertain annotation; intersecting training/mining windows are not
  admitted as trusted background.

Dataset-specific names belong in campaign config. KDK never hardcodes
ShitSpotter categories.

`TileConfig` and `CandidateConfig` share the same fields:

```yaml
category_names: poop
ignore_categories: unknown,ignore
uncategorized_annotation_policy: ignore
default_non_target_policy: background
unclassified_category_policy: ignore
```

An ignore annotation with no usable localization fails closed for negative
mining because no source window can be proven disjoint from it. This source
uncertainty is distinct from the existing `tile_role="ignore"` concept for a
target annotation that cannot be safely represented after crop/visibility
rules.

Candidate manifests embed the normalized truth semantics in their policy
fingerprint. Changing truth semantics therefore invalidates stale candidate
indexes by design.

## Truth-aware prediction review

`prediction-review` compares source-coordinate predictions against current
source truth without modifying it:

```bash
kwcoco-detector-kit prediction-review \
    --true source.kwcoco.zip \
    --pred pred.kwcoco.zip \
    --dst-dpath review \
    --target-categories poop \
    --ignore-categories unknown,ignore
```

Each ranked row is classified as `matched_target`, `known_distractor`,
`uncertain_region`, or `unexplained_prediction` and retains the source image,
adjacent LabelMe path, prediction geometry/score, and overlapping source
annotations. The review KWCoco uses a dedicated proposal category and is
diagnostic only.
