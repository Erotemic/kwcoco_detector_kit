# VIAME Integration Plan

**Status**: planning  
**Date**: 2026-06-25  
**Scope**: ONNX-based deployment of kit-trained detectors into VIAME's PyTorch plugin system

---

## Goal

A user trains a `pup_vs_nonpup` (or any kit scheme) model, exports an ONNX package, and drops it into a VIAME pipeline — without VIAME needing to know about PyTorch, DEIMv2, or any of the kit's training dependencies.

---

## Architecture

```
  Training machine                 Deployment machine (VIAME)
  ─────────────────                ──────────────────────────
  kwcoco_detector_kit              kwcoco_detector_kit  ←  inference-only subset
  └─ export-onnx CLI                  (onnxruntime, numpy, kwimage — no PyTorch)
       │
       ├── deimv2_h1280_w1280.onnx
       ├── deimv2_h1280_w1280.modelspec.json
       └── deimv2_h1280_w1280.labels.txt   ←─ VIAME-native format
             │
             ▼
  VIAME/plugins/pytorch/kwcoco_detector_kit_detector.py
     KwcocoDetectorKitConfig(scfg.DataConfig)
     KwcocoDetectorKitDetector(ImageObjectDetector)
       _build_model():
         from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
         self._predictor = OnnxPredictor(package_path)
       detect(image_data):
         rgb = image_to_rgb_numpy(image_data)
         dets = self._predictor.predict_image_kwimage(rgb)
         return kwimage_to_kwiver_detections(dets)
```

---

## Dependency budget

| Layer | Dependencies |
|-------|-------------|
| Kit training | PyTorch, DEIMv2 submodule, kwcoco, kwimage, onnxruntime, scriptconfig |
| Kit inference (`predictors/onnx.py`) | **onnxruntime, numpy, kwimage only** |
| VIAME plugin | kwcoco_detector_kit (inference-only), kwiver, scriptconfig, kwimage |

The kit's inference path must import-guard everything PyTorch-specific. `kwcoco_detector_kit.predictors.onnx` must be importable in an environment with only `onnxruntime + numpy + kwimage` installed.

---

## Phase 1 — Kit: `OnnxPredictor` class

**File to create**: `kwcoco_detector_kit/predictors/onnx.py`

### What it does

1. Accepts a **package path** (directory or `.zip` archive) that contains:
   - `<name>.onnx` — the exported model
   - `<name>.modelspec.json` — all inference params
2. Reads `modelspec.json` for:
   - `input.shape_hw` → `[H, W]`
   - `preprocess.scale` → divide by 255 (already in modelspec; typically `0.003921...`)
   - `preprocess.normalize_mean/std` → currently zeros/ones (no normalization), but must be read so future models that do normalize work correctly
   - `postprocess.score_thresh` → default detection threshold
   - `meta.category_names` → list of string class names
3. Creates an `onnxruntime.InferenceSession` with the `.onnx` file
4. Exposes two methods:
   - `predict_image(image_np, orig_size=None) → list[dict]` — matches `DetectorPredictor` protocol; returns `[{'label': int, 'bbox_xyxy': [...], 'score': float}, ...]`
   - `predict_image_kwimage(image_np, orig_size=None) → kwimage.Detections` — richer return type used by VIAME plugin; has `.classes` populated from `category_names`

### Interface

```python
class OnnxPredictor:
    def __init__(
        self,
        package: str | Path,
        *,
        device: str = "cpu",          # "cpu" | "cuda" | "cuda:N"
        score_thresh: float | None = None,   # overrides modelspec default
        nms_thresh: float | None = None,
        providers: list[str] | None = None,  # passed to ort.InferenceSession
    ): ...

    @property
    def eval_spatial_size(self) -> tuple[int, int]: ...  # (H, W)

    @property
    def category_names(self) -> list[str]: ...

    def predict_image(
        self,
        image_np: np.ndarray,    # HxWx3 uint8 RGB
        orig_size=None,          # (W, H); inferred from image_np if None
    ) -> list[dict]: ...
    # dict: {'label': int, 'bbox_xyxy': [x0,y0,x1,y1], 'score': float}

    def predict_image_kwimage(
        self,
        image_np: np.ndarray,
        orig_size=None,
    ) -> "kwimage.Detections": ...
    # .boxes: kwimage.Boxes (ltrb format)
    # .scores: np.ndarray float32
    # .class_idxs: np.ndarray int
    # .classes: kwcoco.CategoryTree or list[str]
```

### Implementation notes

- **Preprocessing** inside `predict_image`:
  1. Resize `image_np` to `(eval_h, eval_w)` using `cv2.resize` or `kwimage.imresize` (area interpolation preferred for downscale)
  2. `img_f32 = (resized.astype(np.float32) / 255.0 - mean) / std` — mean/std from modelspec (currently no-ops)
  3. Transpose to `(1, 3, H, W)` NCHW
  4. Build `orig_target_sizes = np.array([[orig_W, orig_H]], dtype=np.int64)`
  5. Run `session.run(None, {"images": img_f32, "orig_target_sizes": orig_sizes})`
  6. `labels, boxes, scores = outputs[:3]` — boxes are `[N, 300, 4]` in `xyxy` format in original image coordinates (DEIMv2 postprocessor handles rescaling to orig_size internally)
  7. Apply `score_thresh` filter
  8. Return list of dicts

- **Device → providers mapping**:
  ```python
  _DEVICE_TO_PROVIDERS = {
      "cpu": ["CPUExecutionProvider"],
      "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  }
  # "cuda:N" → set OrtValue options; or pass as CUDA device ID
  ```

- **No PyTorch imports anywhere in this file.** No `from kwcoco_detector_kit.trainers import ...`. The only kit import is `from kwcoco_detector_kit.export.modelspec import ...` (read-only, no torch dep).

- **Package resolution**: Reuse `kwcoco_detector_kit.export.package.open_package` context manager to handle `.zip` archives and directories uniformly. `open_package` already handles temp extraction. If there's a concern about importing `package.py` pulling in heavy deps, audit it — if it's clean (just `pathlib`, `json`, `shutil`, `zipfile`, `yaml`) it's fine to import.

---

## Phase 2 — Export contract: labels.txt + ONNX metadata

Two additions to `write_modelspec` (or as a separate helper called from `_export_deimv2` and `_export_inproc`):

### 2a. `labels.txt`

**File**: written to `<onnx_fpath>.labels.txt` (i.e., `deimv2_h1280_w1280.labels.txt`)

**Format**: one category name per line, 0-indexed, matching the model's output label indices. No background class (our DEIMv2 models output labels 0…N-1 for N real classes).

```
pup
nonpup_sealion
```

**Open question for the commenter**: VIAME's own training pipeline writes `labels.txt` with `__background__` at index 0 for internally-trained models. Externally-supplied ONNX models may not follow this convention. Confirm whether VIAME's ONNX detector path assumes background-at-0 or reads our labels.txt literally. If VIAME assumes background-at-0, we prepend it:
```
__background__
pup
nonpup_sealion
```
and the DEIMv2 postprocessor's label 0 → kwiver class `"pup"` mapping must shift by 1. **Resolve this before implementing** — wrong offset silently misclassifies every detection.

**Implementation**: add to `write_modelspec` after writing the JSON:

```python
labels_fpath = onnx_fpath.with_suffix(".labels.txt")
labels_fpath.write_text("\n".join(category_names) + "\n")
return spec_fpath   # (already returns spec path; caller gets labels_fpath implicitly)
```

Or extract as `write_labels_txt(onnx_fpath, category_names)` called from `_export_deimv2` and `_export_inproc` so `write_modelspec` stays a pure JSON writer.

### 2b. ONNX metadata embedding

**Why**: keeps the ONNX file self-contained; survives copies that lose sidecars.

**Implementation** — add to `_export_deimv2` after moving/repacking the `.onnx`:

```python
import onnx as _onnx
_model = _onnx.load(str(out_fpath), load_external_data=True)
_model.metadata_props.append(
    _onnx.StringStringEntryProto(key="category_names",
                                  value=",".join(category_names))
)
_model.metadata_props.append(
    _onnx.StringStringEntryProto(key="score_thresh",
                                  value=str(score_thresh))
)
_model.metadata_props.append(
    _onnx.StringStringEntryProto(key="input_hw",
                                  value=f"{H},{W}")
)
_onnx.save(_model, str(out_fpath), save_as_external_data=False)
```

`OnnxPredictor` then reads these as a fallback when `.modelspec.json` is absent:

```python
meta = {p.key: p.value for p in session.get_modelmeta().custom_metadata_map.items()}
```

Note: `onnxruntime` exposes metadata via `InferenceSession.get_modelmeta().custom_metadata_map` — no separate `onnx` package needed at inference time.

**When to embed**: after `onnx.save` in the repack step that already exists in `_export_deimv2`. If `onnx` package is unavailable, skip silently (sidecar is the primary contract).

---

## Phase 3 — VIAME plugin

**File to create**: `/home/joncrall/code/VIAME/plugins/pytorch/kwcoco_detector_kit_detector.py`

### Full skeleton

```python
# BSD 3-Clause License — see VIAME/LICENSE.txt
"""
VIAME ImageObjectDetector wrapping a kit-exported ONNX package.

The kit's OnnxPredictor handles all inference; this file is a thin kwiver
adapter. No PyTorch or DEIMv2 dependency — only onnxruntime + kwimage.
"""

from kwiver.vital.algo import ImageObjectDetector
import scriptconfig as scfg

from viame.pytorch.utilities import (
    image_to_rgb_numpy,
    kwimage_to_kwiver_detections,
    register_vital_algorithm,
    vital_config_update,
)


class KwcocoDetectorKitConfig(scfg.DataConfig):
    """Configuration for KwcocoDetectorKitDetector."""
    package = scfg.Value(None, help=(
        'Path to the exported ONNX package directory or .zip archive. '
        'Must contain a .onnx file and a .modelspec.json sidecar.'
    ))
    device = scfg.Value('cpu', help='onnxruntime device: "cpu", "cuda", "cuda:0"')
    score_thresh = scfg.Value(None, type=float, help=(
        'Detection score threshold. Defaults to modelspec value when None.'
    ))
    nms_thresh = scfg.Value(None, type=float, help=(
        'NMS IoU threshold. Defaults to modelspec value when None.'
    ))

    def __post_init__(self):
        super().__post_init__()
        if self.score_thresh is not None:
            self.score_thresh = float(self.score_thresh)
        if self.nms_thresh is not None:
            self.nms_thresh = float(self.nms_thresh)


class KwcocoDetectorKitDetector(ImageObjectDetector):
    """
    VIAME detector wrapping a kwcoco_detector_kit ONNX export.

    Point ``package`` at the directory produced by ``python -m kwcoco_detector_kit
    export-onnx`` — it contains a ``.onnx`` model and a ``.modelspec.json``
    sidecar that describe all inference parameters (input size, preprocessing,
    category names).  No PyTorch or DEIMv2 installation required.

    Example:
        >>> # xdoctest: +REQUIRES(env:VIAME_SMOKE)
        >>> import sys, pathlib
        >>> sys.path.insert(0, str(pathlib.Path('~/code/VIAME/plugins/pytorch').expanduser()))
        >>> from kwcoco_detector_kit_detector import *  # NOQA
        >>> package = '/data/users/jon.crall/kcd_sealion/workdirs/.../export/'
        >>> image_data = KwcocoDetectorKitDetector.demo_image()
        >>> self = KwcocoDetectorKitDetector()
        >>> self.set_configuration(dict(package=package, device='cpu'))
        >>> detected_objects = self.detect(image_data)
        >>> print(f'found {len(detected_objects)} detections')
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._config = KwcocoDetectorKitConfig()
        self._predictor = None

    # ------------------------------------------------------------------
    # kwiver config protocol
    # ------------------------------------------------------------------

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value) if value is not None else '')
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            raw = cfg.get_value(key)
            self._config[key] = None if raw == '' else raw
        self._config.__post_init__()
        self._build_model()
        return True

    def check_configuration(self, cfg):
        return cfg.has_value('package') and cfg.get_value('package') != ''

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_model(self):
        from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
        self._predictor = OnnxPredictor(
            self._config.package,
            device=self._config.device or 'cpu',
            score_thresh=self._config.score_thresh,
            nms_thresh=self._config.nms_thresh,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, image_data):
        if self._predictor is None:
            raise RuntimeError('KwcocoDetectorKitDetector: call set_configuration first')
        rgb = image_to_rgb_numpy(image_data)
        dets = self._predictor.predict_image_kwimage(rgb)
        dets = dets.numpy()
        return kwimage_to_kwiver_detections(dets)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def demo_image(cls):
        from PIL import Image as PILImage
        from kwiver.vital.util import VitalPIL
        from kwiver.vital.types import ImageContainer
        import kwimage
        pil_img = PILImage.open(kwimage.grab_test_image_fpath())
        return ImageContainer(VitalPIL.from_pil(pil_img))


def __vital_algorithm_register__():
    register_vital_algorithm(
        KwcocoDetectorKitDetector,
        "kwcoco_detector_kit",
        "DEIMv2 / kwcoco_detector_kit ONNX detector",
    )
```

### Pipe template

Create `/home/joncrall/code/VIAME/configs/pipelines/templates/detector_kwcoco_detector_kit.pipe`:

```
# ================================================================
# kwcoco_detector_kit ONNX detector — edit [-PACKAGE-] and thresholds
# ================================================================
process detector
  :: image_object_detector
  :detector:type                     kwcoco_detector_kit
  :detector:kwcoco_detector_kit:package       [-PACKAGE-]
  :detector:kwcoco_detector_kit:device        cpu
  :detector:kwcoco_detector_kit:score_thresh  0.30
  :detector:kwcoco_detector_kit:nms_thresh    0.50
```

---

## Files to create / modify

### Kit (`kwcoco_detector_kit/`)

| File | Action | Notes |
|------|--------|-------|
| `predictors/onnx.py` | **CREATE** | `OnnxPredictor`; onnxruntime-only, no PyTorch |
| `predictors/__init__.py` | modify | export `OnnxPredictor` |
| `export/modelspec.py` | modify | also write `labels.txt` sidecar (or call helper) |
| `export/onnx.py` | modify | embed ONNX metadata after repack |

### VIAME (`VIAME/plugins/pytorch/`)

| File | Action | Notes |
|------|--------|-------|
| `kwcoco_detector_kit_detector.py` | **CREATE** | full skeleton above |
| `configs/pipelines/templates/detector_kwcoco_detector_kit.pipe` | **CREATE** | pipe template |

---

## Implementation order

1. **`OnnxPredictor`** — implement and unit-test standalone (no VIAME, no PyTorch). Test: load gen007's exported ONNX, run a synthetic image, check output shape.

2. **`labels.txt` + ONNX metadata** — add to `write_modelspec` / `_export_deimv2`. Re-export gen006 2-GPU and gen007 to pick up the new sidecars. Resolve the background-at-0 question first.

3. **VIAME plugin** — implement `kwcoco_detector_kit_detector.py` using the skeleton above. Test by running VIAME's plugin loader against the file with a dev install of the kit.

4. **Pipe template** — add alongside the other detector templates in VIAME.

---

## Testing strategy

### Kit-side (no VIAME)

```python
# tests/unit/test_onnx_predictor.py
from kwcoco_detector_kit.predictors.onnx import OnnxPredictor
import numpy as np

def test_onnx_predictor_synthetic(tmp_path):
    # Build a tiny mock ONNX (mock_tiny trainer) and modelspec, check output shape
    ...

def test_onnx_predictor_no_torch():
    # Assert onnxruntime but NOT torch is imported when OnnxPredictor is used
    import sys
    assert 'torch' not in sys.modules
```

### VIAME smoke (requires VIAME install)

```bash
# xdoctest kwcoco_detector_kit_detector.py --style=google
cd ~/code/VIAME/plugins/pytorch
python -m pytest kwcoco_detector_kit_detector.py --doctest-modules
```

---

## Open questions (resolve before implementing)

1. **`labels.txt` background-at-0**: Does VIAME's ONNX detector pipeline (which will call our plugin) expect `__background__` at line 0? If yes, our `predict_image_kwimage` must offset label indices by +1 so label 0 (pup) maps to line 1. If no, write category names verbatim from 0.

2. **`open_package` dep audit**: `package.py` imports `yaml`; check whether VIAME's inference environment has `pyyaml` or if we need to read `package.yaml` with `json.loads` fallback.

3. **NMS**: DEIMv2's ONNX model already runs NMS internally (the postprocessor is baked into the traced graph). The `nms_thresh` config param in the plugin is then only advisory — we cannot re-run NMS on the ONNX output unless we strip the postprocessor from the ONNX at export time. Decide: expose `nms_thresh` as a no-op with a warning, or strip the postprocessor and run NMS in `OnnxPredictor`. The current export includes the postprocessor (label, box, score format), so it is a no-op for now.

4. **CUDA providers**: `onnxruntime-gpu` vs `onnxruntime` package name differs by platform/CUDA version. The `OnnxPredictor` should handle `ImportError` on `CUDAExecutionProvider` gracefully and fall back to CPU.

5. **Tiled inference in VIAME**: The VIAME plugin will receive whole frames (potentially 4K+). The kit's tiled eval pipeline uses `tiled_predictor.py` in `eval/`. If VIAME's frame resolution is much larger than the model's `eval_spatial_size` (1280px), detection of small objects (pups) will suffer. A tiled variant of the VIAME plugin (or a VIAME tiling wrapper) may be needed as a follow-up.
