"""
TileLoader — torch ``IterableDataset`` wrapping any ``TileStore``.

Phase 3's load-time crop augmentation lives here. The on-disk tile is
typically larger than the model input (``oversize_factor > 1`` in
``data.tile``); the loader random-crops to the model input at load time,
which gives the trainer scale + position jitter without needing the
input to fit exactly on disk.

Two modes:

  - ``augment=True`` (training)  random crop + horizontal flip; bboxes
                                  warped through the crop.
  - ``augment=False`` (eval)      center crop, no flip.

Multispectral support: the loader respects ``store.metadata["channels"]``
and applies per-channel normalization when ``normalize`` is set
(typically from ``data.stats.compute_per_channel_stats``).

The loader stays out of DDP / sampler decisions — it's an
``IterableDataset``, and the trainer's DataLoader / DistributedSampler
handles rank-aware sharding. For ``WebdatasetStore``, DDP-aware shard
splitting can be added via the upstream ``webdataset`` library's
splitter; documented in ``docs/webdataset.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from kwcoco_detector_kit.data.tile_store import TileRecord, TileStore


@dataclass
class TileBatch:
    """A single (image, targets) sample after load-time transforms.

    Tensor-shaped fields are torch tensors when ``torch`` is installed;
    otherwise numpy arrays.
    """
    image: "np.ndarray | object"          # 3xHxW float in [0,1] (or HxWxC numpy)
    bboxes_xywh: "np.ndarray | object"    # Mx4 in model-input pixel coords
    category_ids: "np.ndarray | object"   # M int
    metadata: dict


class TileLoader:
    """Iterable wrapper around a ``TileStore`` that applies load-time aug."""

    def __init__(
        self,
        store: TileStore,
        *,
        model_input_hw: Optional[Tuple[int, int]] = None,
        augment: bool = True,
        normalize: Optional[dict] = None,
        seed: int = 0,
        return_tensors: bool = True,
    ):
        """
        Args:
            store: a TileStore (any backend).
            model_input_hw: (H, W) the model expects. If ``None``, read
                from ``record.metadata['tile_model_input_size']`` per-tile.
            augment: ``True`` for random crop + flip; ``False`` for
                center crop / no flip (eval).
            normalize: optional ``{"mean": [...], "std": [...]}`` for
                per-channel normalization. ``None`` returns uint8-in-[0,255]
                divided by 255 (basic float scaling).
            seed: RNG seed.
            return_tensors: ``True`` returns torch tensors (CHW float32);
                ``False`` returns numpy (HWC float32). Tests use False.
        """
        self._store = store
        self._model_input_hw = (
            tuple(model_input_hw) if model_input_hw is not None else None
        )
        self._augment = bool(augment)
        self._normalize = normalize
        self._rng = np.random.RandomState(int(seed))
        self._return_tensors = bool(return_tensors)

    @property
    def metadata(self) -> dict:
        return self._store.metadata

    def __iter__(self) -> Iterator[TileBatch]:
        for record in self._store:
            yield self._transform(record)

    # ----- transforms -----

    def _resolve_model_input(self, record: TileRecord) -> Tuple[int, int]:
        if self._model_input_hw is not None:
            return self._model_input_hw
        meta_hw = record.metadata.get("tile_model_input_size")
        if meta_hw and len(meta_hw) == 2:
            return (int(meta_hw[0]), int(meta_hw[1]))
        # Fall back to the on-disk tile size.
        H, W = record.image_np.shape[:2]
        return (H, W)

    def _crop(self, record: TileRecord, target_hw: Tuple[int, int]):
        """Random-crop (augment) or center-crop (eval) to target_hw."""
        H, W = record.image_np.shape[:2]
        tH, tW = int(target_hw[0]), int(target_hw[1])
        if (H, W) == (tH, tW):
            return record.image_np, record.bboxes_xywh
        if H < tH or W < tW:
            # Tile too small — center-pad with zeros and emit as-is. This
            # handles the edge tiles from multiscale extraction where the
            # source image's far-right column is narrower than tile_size.
            pad_h = max(0, tH - H)
            pad_w = max(0, tW - W)
            padded = np.zeros((H + pad_h, W + pad_w, record.image_np.shape[-1]),
                              dtype=record.image_np.dtype)
            padded[:H, :W] = record.image_np
            return padded[:tH, :tW], record.bboxes_xywh.copy()

        if self._augment:
            top = int(self._rng.randint(0, H - tH + 1))
            left = int(self._rng.randint(0, W - tW + 1))
        else:
            top = (H - tH) // 2
            left = (W - tW) // 2

        image = record.image_np[top:top + tH, left:left + tW]
        bboxes = record.bboxes_xywh.copy()
        if bboxes.size:
            bboxes[:, 0] -= left
            bboxes[:, 1] -= top
            # Clip to the crop window
            x1 = bboxes[:, 0]
            y1 = bboxes[:, 1]
            x2 = x1 + bboxes[:, 2]
            y2 = y1 + bboxes[:, 3]
            x1 = np.clip(x1, 0, tW)
            y1 = np.clip(y1, 0, tH)
            x2 = np.clip(x2, 0, tW)
            y2 = np.clip(y2, 0, tH)
            new_w = x2 - x1
            new_h = y2 - y1
            keep = (new_w > 1) & (new_h > 1)
            bboxes = np.stack([x1, y1, new_w, new_h], axis=-1)[keep]
        return image, bboxes

    def _flip(self, image: np.ndarray, bboxes: np.ndarray):
        """Horizontal flip; warp bboxes."""
        W = image.shape[1]
        image = image[:, ::-1, :].copy()
        if bboxes.size:
            bboxes = bboxes.copy()
            bboxes[:, 0] = W - bboxes[:, 0] - bboxes[:, 2]
        return image, bboxes

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """uint8 -> float32 [0,1] -> (x - mean) / std per channel."""
        x = image.astype(np.float32) / 255.0
        if self._normalize:
            mean = np.asarray(self._normalize["mean"], dtype=np.float32)
            std = np.asarray(self._normalize["std"], dtype=np.float32)
            if mean.ndim == 1:
                mean = mean.reshape(1, 1, -1)
            if std.ndim == 1:
                std = std.reshape(1, 1, -1)
            x = (x - mean) / np.where(std > 1e-6, std, 1.0)
        return x

    def _transform(self, record: TileRecord) -> TileBatch:
        target_hw = self._resolve_model_input(record)
        image, bboxes = self._crop(record, target_hw)
        if self._augment and self._rng.rand() < 0.5:
            image, bboxes = self._flip(image, bboxes)
        x = self._normalize_image(image)
        category_ids = record.category_ids

        if self._return_tensors:
            try:
                import torch
                x_t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
                b_t = torch.from_numpy(np.ascontiguousarray(bboxes))
                c_t = torch.from_numpy(np.ascontiguousarray(category_ids))
                return TileBatch(image=x_t, bboxes_xywh=b_t, category_ids=c_t,
                                 metadata=dict(record.metadata))
            except ImportError:
                pass
        return TileBatch(image=x, bboxes_xywh=bboxes, category_ids=category_ids,
                         metadata=dict(record.metadata))
