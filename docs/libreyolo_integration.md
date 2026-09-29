# LibreYOLO integration

`trainer=libreyolo` is KDK's general model-engine backend. It deliberately does
less than the older per-family trainer adapters: KDK prepares the data view and
LibreYOLO trains the model.

## Boundary

```text
rich source kwcoco
    |
    | KDK: channels / delayed assets / windows / truth policy / sampling
    v
prepared KDK kwcoco view
    |
    | KDK bridge: COCO annotations only (absolute image references)
    v
LibreYOLO model engine
    |
    | train / DDP / checkpoint / native predict / native export
    v
KDK predictor records
    |
    | PredictionSpace / tiled merge / source-space eval / mining / packaging
    v
source kwcoco coordinates
```

The bridge is intentionally not a new tile store. `detector_prepared/
libreyolo_coco/` contains `train.mscoco.json`, `vali.mscoco.json`,
`dataset.yaml`, and a content receipt. The COCO `file_name` fields point at the
already-prepared KDK images. For instance segmentation, KDK's existing COCO
exporter preserves holes as RLE where necessary.

## Initial model catalog

The catalog lives in `kwcoco_detector_kit/trainers/libreyolo.py:VARIANTS`.
Initial entries cover:

- DEIMv2: Atto, Femto, Pico, N, S, M, L, X detection;
- D-FINE: N/S/M/L/X detection and instance segmentation;
- RF-DETR: N/S/M/L detection and N/S/M/L/X/XX instance segmentation;
- YOLOv9: T/S/M/C detection;
- GTR: S/M/L/X detection and instance segmentation; and
- TinyFormer: S/M/L/X/XL detection.

Variant names are KDK names such as `rfdetr_xx_seg` or `deimv2_atto`; the
catalog maps these to LibreYOLO checkpoint names and model metadata.

The catalog is intentionally narrower than LibreYOLO's full task list. KDK's
current `DetectorTrainer` contract is boxes + optional instance masks, so pose,
OBB, semantic segmentation, depth, VLMs, and other LibreYOLO tasks are not
advertised through this backend yet.

## Install

Initialize the pinned source checkout:

```bash
git submodule update --init --recursive tpl/libreyolo
pip install -e ".[libreyolo]"
```

The source checkout is preferred over an unrelated site-packages installation.
Override its location with `KCD_LIBREYOLO_REPO_DPATH=/path/to/libreyolo`.

Audit the relevant dependencies with:

```bash
kwcoco-detector-kit check-env --groups core,onnx,libreyolo --runtime
```

## Training

Recipes and sweeps use one trainer name while selecting the model family in the
variant:

```yaml
sweep:
  trainer: libreyolo
  matrix:
    - variant: rfdetr_xx_seg
      input_hw: [768, 768]
      train_policy: fixed
    - variant: dfine_x_seg
      input_hw: [640, 640]
      train_policy: multiscale
    - variant: tinyformer_xl
      input_hw: [640, 640]
      train_policy: multiscale
```

KDK passes `device=[0, 1, ...]` for multi-GPU training and lets LibreYOLO own
its DDP spawn. KDK must not wrap this backend in `torchrun`, which would create
nested process groups.

The common bridge currently accepts `train_policy: fixed` or `multiscale`.
`multiscale` means the selected LibreYOLO family's native multiscale recipe;
KDK's exact-range policy spellings are rejected because they cannot be mapped
generically without changing family semantics. Variants whose LibreYOLO trainer
does not expose multiscale input variation are recorded/coerced to `fixed`.

Family-specific tuning knobs can be supplied under
`extra.libreyolo_train_kwargs`; KDK orchestration fields are never forwarded
implicitly. This keeps the boundary auditable.

## Rich kwcoco data

The LibreYOLO catalog currently consumes RGB tensors. That is a property of the
admitted model configurations, not of KDK's source data model. Multispectral,
auxiliary, multi-resolution, or COG-backed source data remains in kwcoco and is
resolved by KDK before the model-engine boundary.

Do not convert the canonical dataset to YOLO format to use this backend. The
prepared training view is disposable and reproducible from kwcoco. Source
coordinates remain authoritative during evaluation and mining.

A future lazy bridge may pass KDK samples directly into LibreYOLO's trainer to
avoid even annotation materialization. It should be an optimization of this
contract, not a change in ownership.

## Inference and export

`LibreYOLOPredictor` converts LibreYOLO `Results` into KDK's detector records.
KDK's existing tiled/source-window machinery therefore works without teaching
LibreYOLO about large images.

ONNX export calls LibreYOLO's native exporter. KDK writes its own modelspec and
then checks postprocessed checkpoint-vs-ONNX parity through LibreYOLO's ONNX
backend. The package records this as `libreyolo_native_v1`; KDK does not pretend
the graph uses its older two-input processed-detection ONNX contract.

A `libreyolo_native_v1` package intentionally uses LibreYOLO's backend for
pre/postprocessing, so package inference requires the pinned LibreYOLO source
checkout (or `KCD_LIBREYOLO_REPO_DPATH`) in addition to KDK. The ONNX graph is
still copied into the package and fingerprinted; the runtime dependency is
explicit rather than silently reimplementing family-specific decoding in KDK.
