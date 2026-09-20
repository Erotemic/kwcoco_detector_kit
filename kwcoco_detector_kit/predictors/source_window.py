"""No-cache source-image window realization for tiled inference."""
from __future__ import annotations

import time
from pathlib import Path

from kwcoco_detector_kit.predictors.space import PredictionSpace


_DECODE_ONCE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".avif", ".heic",
}
_REGION_READABLE_EXTS = {".tif", ".tiff", ".cog"}


def _coerce_rgb(image):
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def choose_source_read_strategy(coco_img, requested="auto") -> str:
    """Choose between one full decode and independent delayed crop reads.

    Ordinary compressed photographic formats do not provide useful random
    region reads, so re-decoding one JPEG per tile is avoided. TIFF-like
    imagery is allowed to stay lazy so GDAL/raster backends can service crop
    requests without realizing the full source.
    """
    requested = str(requested or "auto")
    if requested not in {"auto", "decode_once", "delayed_region"}:
        raise ValueError(f"unknown source read strategy: {requested!r}")
    if requested != "auto":
        return requested
    file_name = str(coco_img.img.get("file_name", ""))
    suffix = Path(file_name).suffix.lower()
    if suffix in _REGION_READABLE_EXTS:
        return "delayed_region"
    if suffix in _DECODE_ONCE_EXTS:
        return "decode_once"
    # Unknown sources favor correctness and avoiding repeated decode work.
    return "decode_once"


class SourceWindowReader:
    """Realize prediction-space windows without a persistent tile cache.

    The reader owns an explicit :class:`PredictionSpace`. Delayed operations
    are expressed in that space before finalization, allowing ``delayed_image``
    to optimize downscales against GDAL overviews when the underlying asset
    supports them. Detector outputs are mapped back to native image space by
    the prediction writer; callers never have to treat scaled coordinates as
    canonical annotation coordinates.

    ``decode_once`` finalizes the full scaled delayed image at most once and
    serves all windows as NumPy slices. ``delayed_region`` keeps the scaled
    delayed graph lazy and finalizes each requested crop independently.
    """

    def __init__(self, coco_img, *, strategy="auto", prediction_scale=1.0):
        self.coco_img = coco_img
        self.strategy = choose_source_read_strategy(coco_img, strategy)
        self._coerce_rgb = _coerce_rgb
        self._delayed = None
        self._prediction_delayed = None
        self._full = None
        img = coco_img.img
        self._native_shape = None
        if img.get("height") is not None and img.get("width") is not None:
            self._native_shape = (int(img["height"]), int(img["width"]))
        if self._native_shape is None:
            # KWCoco images normally carry width/height. If metadata is absent,
            # we must discover native shape before a prediction-space transform
            # can be defined.
            native = self._coerce_rgb(self._ensure_delayed().finalize())
            self._native_shape = tuple(map(int, native.shape[:2]))
            self._native_full_fallback = native
        else:
            self._native_full_fallback = None
        self.space = PredictionSpace.from_scale(self._native_shape, prediction_scale)
        self.t_decode = 0.0
        self.t_window_read = 0.0
        self.full_decode_count = 0
        self.region_read_count = 0

    @property
    def source_hw(self):
        """Native source-image dimensions; canonical KWCoco annotation space."""
        return self.space.native_hw

    @property
    def prediction_hw(self):
        """Dimensions in the scaled space consumed by the detector."""
        return self.space.prediction_hw

    def prepare(self, *, force_full=False):
        """Prepare this source on an I/O worker before GPU consumption.

        JPEG-like ``decode_once`` sources are finalized here so decode/resize
        overlaps inference on an earlier source image. Region-readable sources
        stay lazy unless a caller explicitly needs the whole image.
        """
        if force_full or self.strategy == "decode_once":
            self.read_full()
        return self

    def _ensure_delayed(self):
        if self._delayed is None:
            self._delayed = self.coco_img.imdelay()
        return self._delayed

    def _ensure_prediction_delayed(self):
        if self._prediction_delayed is None:
            delayed = self._ensure_delayed()
            if not self.space.is_native:
                req_sx, req_sy = self.space.requested_scale_xy
                scale = req_sx if req_sx == req_sy else (req_sx, req_sy)
                delayed = delayed.scale(scale)
            self._prediction_delayed = delayed
        return self._prediction_delayed

    def read_full(self):
        """Finalize the entire image in prediction space."""
        if self._full is None:
            t0 = time.perf_counter()
            if self._native_full_fallback is not None and self.space.is_native:
                full = self._native_full_fallback
                self._native_full_fallback = None
            else:
                full = self._coerce_rgb(self._ensure_prediction_delayed().finalize())
            expected = self.prediction_hw
            actual = tuple(map(int, full.shape[:2]))
            if actual != expected:
                raise RuntimeError(
                    "delayed-image prediction scale produced an unexpected shape: "
                    f"expected={expected}, actual={actual}, space={self.space.to_dict()}"
                )
            self._full = full
            self.t_decode += time.perf_counter() - t0
            self.full_decode_count += 1
        return self._full

    def read_window(self, x0, y0, x1, y1):
        """Read one crop whose coordinates are in prediction space."""
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        if self.strategy == "decode_once":
            full = self.read_full()
            t0 = time.perf_counter()
            arr = full[y0:y1, x0:x1]
            self.t_window_read += time.perf_counter() - t0
            return arr

        t0 = time.perf_counter()
        space_slice = (slice(y0, y1), slice(x0, x1))
        try:
            delayed_crop = self._ensure_prediction_delayed().crop(space_slice)
            arr = self._coerce_rgb(delayed_crop.finalize())
            self.region_read_count += 1
        except Exception:
            # A nominally region-readable asset can still be backed by a
            # delayed node that cannot crop efficiently. Correctness wins:
            # fall back to one scaled decode, and never repeatedly decode it.
            self.strategy = "decode_once"
            full = self.read_full()
            arr = full[y0:y1, x0:x1]
        self.t_window_read += time.perf_counter() - t0
        return arr

    def read_windows(self, offsets, window_hw):
        """Realize one bounded batch of prediction-space windows."""
        win_h, win_w = map(int, window_hw)
        H, W = map(int, self.prediction_hw)
        return [
            self.read_window(x0, y0, min(x0 + win_w, W), min(y0 + win_h, H))
            for x0, y0 in offsets
        ]
