"""No-cache source-image window realization for tiled inference."""
from __future__ import annotations

import time
from pathlib import Path


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
    region reads, so re-decoding one JPEG per tile is avoided.  TIFF-like
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
    """Realize source-space windows without a persistent tile cache.

    ``decode_once`` finalizes the full delayed image at most once and serves all
    windows as NumPy slices. ``delayed_region`` keeps the delayed graph and
    finalizes each requested crop independently, which lets region-readable
    backends avoid a full source decode.
    """

    def __init__(self, coco_img, *, strategy="auto"):
        self.coco_img = coco_img
        self.strategy = choose_source_read_strategy(coco_img, strategy)
        self._coerce_rgb = _coerce_rgb
        # Build the delayed graph lazily. Region-readable sources are prepared
        # on one source-prefetch thread but consumed on the dedicated window
        # thread; delaying graph construction avoids carrying backend-specific
        # handles across threads. Decode-once sources create/finalize the graph
        # in the source worker and subsequently share only the NumPy array.
        self._delayed = None
        self._full = None
        img = coco_img.img
        self._shape = None
        if img.get("height") is not None and img.get("width") is not None:
            self._shape = (int(img["height"]), int(img["width"]))
        self.t_decode = 0.0
        self.t_window_read = 0.0
        self.full_decode_count = 0
        self.region_read_count = 0

    @property
    def source_hw(self):
        if self._shape is None:
            arr = self.read_full()
            self._shape = tuple(map(int, arr.shape[:2]))
        return self._shape

    def prepare(self, *, force_full=False):
        """Prepare this source on an I/O worker before GPU consumption.

        JPEG-like ``decode_once`` sources are finalized here so their decode
        overlaps inference on an earlier source image.  Region-readable
        sources stay lazy unless a caller explicitly needs the whole image.
        """
        if force_full or self.strategy == "decode_once" or self._shape is None:
            self.read_full()
        return self

    def _ensure_delayed(self):
        if self._delayed is None:
            self._delayed = self.coco_img.imdelay()
        return self._delayed

    def read_full(self):
        if self._full is None:
            t0 = time.perf_counter()
            self._full = self._coerce_rgb(self._ensure_delayed().finalize())
            self.t_decode += time.perf_counter() - t0
            self.full_decode_count += 1
            self._shape = tuple(map(int, self._full.shape[:2]))
        return self._full

    def read_window(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        if self.strategy == "decode_once":
            # Full-source decode has its own timing bucket.  Do not count it a
            # second time as window-read work when the first window triggers
            # lazy preparation.
            full = self.read_full()
            t0 = time.perf_counter()
            arr = full[y0:y1, x0:x1]
            self.t_window_read += time.perf_counter() - t0
            return arr

        t0 = time.perf_counter()
        space_slice = (slice(y0, y1), slice(x0, x1))
        try:
            delayed_crop = self._ensure_delayed().crop(space_slice)
            arr = self._coerce_rgb(delayed_crop.finalize())
            self.region_read_count += 1
        except Exception:
            # A nominally region-readable asset can still be backed by a
            # delayed node that cannot crop efficiently. Correctness wins:
            # fall back to one decode, and never repeatedly decode it.
            self.strategy = "decode_once"
            full = self.read_full()
            arr = full[y0:y1, x0:x1]
        self.t_window_read += time.perf_counter() - t0
        return arr

    def read_windows(self, offsets, window_hw):
        """Realize one bounded batch of windows in offset order."""
        win_h, win_w = map(int, window_hw)
        H, W = map(int, self.source_hw)
        return [
            self.read_window(x0, y0, min(x0 + win_w, W), min(y0 + win_h, H))
            for x0, y0 in offsets
        ]
