"""
kwcoco_dataloader integration for single-frame bounding-box detection.

This module wraps :class:`~kwcoco_dataloader.tasks.fusion.datamodules.\
kwcoco_dataset.KWCocoVideoDataset` — the richest kwcoco sampling engine
available — for the kit's detection use case.

Why bother?  The native ``KWCocoVideoDataset`` was designed for multi-frame
multi-sensor satellite fusion, so its batch items are complex dicts of
per-frame, per-modality tensors.  For a single-frame RGB detector, 95 % of
that structure is noise.  This module:

* Configures ``KWCocoVideoDataset`` with detection-appropriate defaults
  (``time_steps=1``, ``requested_tasks={boxes:True, class/saliency/change:False}``).
* Exposes a clean per-image item dict:
  ``{image, boxes_ltrb, class_idxs, class_names, meta}``.
* Provides ``detection_collate_fn`` — a batch collator that stacks images
  and returns ragged lists of boxes (variable M per image).
* Exposes the full ``KWCocoVideoDatasetConfig`` surface area for power users
  (balanced sampling, ``select_images`` JQ filters, augmentation knobs, etc.).

Requires the ``kwcoco_dataloader`` package::

    pip install kwcoco-dataloader  # or install from source

Quick start::

    from kwcoco_detector_kit.data.kwcoco_sampler import KwcocoDetectionDataset

    train_ds = KwcocoDetectionDataset(
        'train.kwcoco.zip',
        chip_dims=(512, 512),
        channels='r|g|b',
        balance_options=[{'attribute': 'contains_annotation'}],
        use_centered_positives=True,
    )
    loader = train_ds.make_loader(batch_size=4, num_workers=4, shuffle=True)
    for batch in loader:
        images = batch['image']          # (B, 3, H, W) float32
        boxes  = batch['boxes_ltrb']     # list of B tensors, each (M, 4)
        cids   = batch['class_idxs']     # list of B tensors, each (M,)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

_MISSING = object()


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def detection_collate_fn(batch: list) -> dict:
    """Collate a list of per-image dicts into a batch dict.

    Images are stacked (all must have the same CHW shape); boxes and
    class_idxs are returned as ragged lists (one tensor per image).

    Args:
        batch: list of dicts, each with keys ``image``, ``boxes_ltrb``,
            ``class_idxs``, ``class_names``, ``meta``.

    Returns:
        dict with stacked ``image`` tensor and list fields for the rest.
    """
    import torch
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes_ltrb'] for item in batch]
    class_idxs = [item['class_idxs'] for item in batch]
    class_names = batch[0]['class_names']
    metas = [item['meta'] for item in batch]
    return {
        'image': images,
        'boxes_ltrb': boxes,
        'class_idxs': class_idxs,
        'class_names': class_names,
        'meta': metas,
    }


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class KwcocoDetectionDataset:
    """Single-frame detection dataset backed by :class:`KWCocoVideoDataset`.

    All ``KWCocoVideoDatasetConfig`` options can be passed as keyword
    arguments.  The following defaults are set automatically:

    * ``time_steps=1`` — single frame
    * ``output_type='rgb'`` — single-sensor RGB output wrapper
    * ``requested_tasks`` — only ``boxes`` is enabled; class/saliency/change
      rasters are disabled to save compute

    Args:
        kwcoco_src: Path to a kwcoco file, bundle directory, or a
            :class:`kwcoco.CocoDataset` instance.
        chip_dims: Spatial window size ``(H, W)`` or a single int.
            Use ``'full'`` to always sample the entire image.
        channels: SensorChanSpec string, e.g. ``'r|g|b'`` (default) or
            ``'B02|B03|B04'`` for multispectral data.
        balance_options: YAML-coercible list of dicts understood by
            ``KWCocoVideoDatasetConfig.balance_options``.  Example::

                [{'attribute': 'contains_annotation'},
                 {'attribute': 'class', 'weights': {'widget': 1.0}}]

        mode: ``'fit'`` (train) or ``'predict'`` (eval/test).
        **kwargs: Forwarded verbatim to ``KWCocoVideoDataset``.

    Raises:
        ImportError: if ``kwcoco_dataloader`` is not installed.
    """

    def __init__(
        self,
        kwcoco_src: Union[str, Path, "kwcoco.CocoDataset"],
        *,
        chip_dims: Union[int, Tuple[int, int], str] = (512, 512),
        channels: str = "r|g|b",
        balance_options=None,
        use_centered_positives: bool = True,
        use_grid_positives: bool = True,
        use_grid_negatives: bool = True,
        mode: str = "fit",
        **kwargs,
    ):
        try:
            from kwcoco_dataloader.tasks.fusion.datamodules.kwcoco_dataset import (
                KWCocoVideoDataset,
            )
        except ImportError as ex:
            raise ImportError(
                "kwcoco_dataloader is required for KwcocoDetectionDataset. "
                "Install it with: pip install kwcoco-dataloader"
            ) from ex

        # Requested tasks for detection: only boxes, no sseg rasters.
        detection_tasks = {
            "boxes": True,
            "class": False,
            "saliency": False,
            "change": False,
        }

        # Build the inner dataset.  Let requested_tasks config override ours
        # in case the user passes it explicitly.
        kv_kwargs = dict(
            time_steps=1,
            chip_dims=chip_dims,
            channels=channels,
            balance_options=balance_options,
            use_centered_positives=use_centered_positives,
            use_grid_positives=use_grid_positives,
            use_grid_negatives=use_grid_negatives,
            output_type="rgb",
            mode=mode,
            requested_tasks=detection_tasks,
            reduce_item_size=True,
        )
        kv_kwargs.update(kwargs)

        self._inner: KWCocoVideoDataset = KWCocoVideoDataset(kwcoco_src, **kv_kwargs)
        # Ensure box task is on even if user's requested_tasks dict
        # overrode the 'boxes' key through the config path.
        self._inner.requested_tasks["boxes"] = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> List[str]:
        """Ordered list of class names from the underlying dataset."""
        return list(self._inner.classes)

    @property
    def sample_grid(self) -> dict:
        """The space-time sample grid built by the inner dataset."""
        return self._inner.sample_grid

    @property
    def n_positives(self) -> int:
        return len(self._inner.sample_grid.get("positives_indexes", []))

    @property
    def n_negatives(self) -> int:
        grid = self._inner.sample_grid
        n_pos = len(grid.get("positives_indexes", []))
        return len(grid.get("targets", [])) - n_pos

    # ------------------------------------------------------------------
    # torch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._inner.sample_grid["targets"])

    def __getitem__(self, index) -> dict:
        """Return a single detection sample.

        Returns:
            dict with keys:

            * ``image``: ``(C, H, W)`` float32 torch tensor in ``[0, 1]``.
            * ``boxes_ltrb``: ``(M, 4)`` float32 tensor — left/top/right/bottom
              in output-pixel coordinates.  M may be 0.
            * ``class_idxs``: ``(M,)`` int64 tensor — indices into
              :attr:`class_names`.
            * ``class_names``: list of str (same every item — convenience ref).
            * ``meta``: dict with ``gid``, ``ann_aids``.
        """
        import torch

        item = self._inner[index]

        # RGBImageBatchItem.imdata_chw → (C, H, W) float32 tensor
        image = item.imdata_chw
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(np.asarray(image, dtype=np.float32))

        frame = item["frames"][0]

        raw_boxes = frame.get("box_ltrb", None)
        raw_cids = frame.get("box_cidxs", None)

        if raw_boxes is None or (hasattr(raw_boxes, "__len__") and len(raw_boxes) == 0):
            boxes_ltrb = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_ltrb = torch.as_tensor(
                np.asarray(raw_boxes, dtype=np.float32)
            )

        if raw_cids is None or (hasattr(raw_cids, "__len__") and len(raw_cids) == 0):
            class_idxs = torch.zeros((0,), dtype=torch.int64)
        else:
            class_idxs = torch.as_tensor(
                np.asarray(raw_cids, dtype=np.int64)
            )

        meta = {
            "gid": frame.get("gid"),
            "ann_aids": frame.get("ann_aids") or [],
        }

        return {
            "image": image,
            "boxes_ltrb": boxes_ltrb,
            "class_idxs": class_idxs,
            "class_names": self.class_names,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def make_loader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        collate_fn=None,
    ):
        """Return a DataLoader with detection-appropriate defaults.

        Args:
            batch_size: samples per batch.
            num_workers: background worker count.
            shuffle: shuffle the sample grid before each epoch.
            pin_memory: pin host memory for faster GPU transfer.
            collate_fn: override the default :func:`detection_collate_fn`.

        Returns:
            :class:`torch.utils.data.DataLoader`
        """
        import torch.utils.data as torch_data
        from kwcoco_dataloader.tasks.fusion.datamodules.kwcoco_dataset import (
            worker_init_fn,
        )

        if collate_fn is None:
            collate_fn = detection_collate_fn

        return torch_data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    # ------------------------------------------------------------------
    # Balanced sampler access (for BalanceMixin)
    # ------------------------------------------------------------------

    def make_balanced_loader(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Return a DataLoader that uses the inner dataset's balanced sampler.

        Uses :meth:`KWCocoVideoDataset.make_balanced_loader`, which builds a
        ``BalancedSampleForest`` over the sample grid.

        Returns:
            :class:`torch.utils.data.DataLoader`
        """
        import torch.utils.data as torch_data
        from kwcoco_dataloader.tasks.fusion.datamodules.kwcoco_dataset import (
            worker_init_fn,
        )

        inner_loader = self._inner.make_loader(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
            collate_fn=detection_collate_fn,
        )
        return inner_loader

    # ------------------------------------------------------------------
    # Visualisation (best-effort)
    # ------------------------------------------------------------------

    def draw_item(self, index, **kwargs):
        """Delegate to the inner dataset's rich visualiser."""
        item = self._inner[self._inner.sample_grid["targets"][index]]
        return self._inner.draw_item(item, **kwargs)

    def __repr__(self) -> str:
        n = len(self)
        pos = self.n_positives
        return (
            f"KwcocoDetectionDataset("
            f"n={n}, n_positives={pos}, n_negatives={n - pos}, "
            f"classes={self.class_names})"
        )
