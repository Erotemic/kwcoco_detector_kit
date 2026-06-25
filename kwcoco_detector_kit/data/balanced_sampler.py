"""Dataloader-level class balancing via per-index sample weights.

The alternative to :mod:`kwcoco_detector_kit.data.balance_mscoco` (which
*duplicates image rows* in a static file): compute one weight per dataset
index from ``kwcoco_dataloader``'s ``BalancedSampleForest`` hierarchical
stratified sampler, save them to a sidecar JSON at launch time, and draw
epochs with a weighted sampler inside the dataloader. Same images on
disk, fresh draw every epoch, balance is a knob instead of a bake.

Strictly **optional**: nothing here runs unless a weights file is wired
into the training config (``kcd_sample_weights_fpath`` in the generated
DEIMv2 yml). The file-duplication path remains the default.

Pipeline:

1. :func:`build_sample_grid_from_mscoco` — one grid item per image (in
   dataset-index order) with a multi-label ``classes`` count dict;
   annotation-free images get the ``<empty>`` sentinel (same convention
   as ``KCD_BALANCE_TARGET_JSON``).
2. :func:`compute_index_weights` — feeds the grid to
   ``BalancedSampleForest``, applies ``subdivide`` per configured key
   (optionally with per-class target weights), then asks the forest for
   flat per-index weights via ``.index_weights()``.

   .. note:: ``.index_weights()`` is the agreed contract with
      ``kwcoco_dataloader`` (flattening the hierarchical strata to one
      weight per leaf index). Until the submodule ships it, this module
      raises a clear error naming the missing method. The
      ``forest_factory`` hook keeps everything testable meanwhile.

3. :func:`write_balance_weights` / :func:`load_balance_weights` — the
   sidecar JSON (weights + provenance meta).
4. :class:`DistributedWeightedRandomSampler` — rank-aware weighted
   draw-with-replacement; deterministic per ``(seed, epoch, rank)``;
   ``set_epoch`` reseeds. DEIMv2's patched solver builds it from the
   sidecar via :func:`sampler_from_weights_file`.

CLI (run at launch time, before the sweep)::

    python -m kwcoco_detector_kit.data.balanced_sampler \\
        --src train_unbalanced.mscoco.json \\
        --dst balance_weights.json \\
        --class_weights '{"<empty>": 0.25, "pup": 0.20, ...}' \\
        --seed 0
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import scriptconfig as scfg

__all__ = [
    "EMPTY_KEY",
    "build_sample_grid_from_mscoco",
    "compute_index_weights",
    "write_balance_weights",
    "load_balance_weights",
    "DistributedWeightedRandomSampler",
    "sampler_from_weights_file",
]

EMPTY_KEY = "<empty>"


# ---------------------------------------------------------------------------
# Weight computation (launch-time, CPU, no torch needed)
# ---------------------------------------------------------------------------

def build_sample_grid_from_mscoco(mscoco_fpath) -> List[Dict[str, Any]]:
    """One forest grid item per image, in dataset-index order.

    Order matters: DEIMv2's ``CocoDetection`` indexes images in the json
    ``images`` list order, and the sampler's weights are positional.
    """
    data = json.loads(Path(mscoco_fpath).read_text())
    cat_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    per_image: Dict[Any, Dict[str, int]] = {
        img["id"]: {} for img in data.get("images", [])
    }
    for ann in data.get("annotations", []):
        name = cat_name.get(ann.get("category_id"))
        gid = ann.get("image_id")
        if name is None or gid not in per_image:
            continue
        per_image[gid][name] = per_image[gid].get(name, 0) + 1
    grid = []
    for img in data.get("images", []):
        classes = per_image[img["id"]] or {EMPTY_KEY: 1}
        grid.append({"classes": classes})
    return grid


def _default_forest_factory(sample_grid, rng):
    try:
        from kwcoco_dataloader.tasks.fusion.datamodules.balanced_sampling import (
            BalancedSampleForest,
        )
    except ImportError as ex:
        raise ImportError(
            "balanced sampler-mode needs kwcoco_dataloader (the "
            "tpl/kwcoco_dataloader submodule / baked image dependency); "
            "file-duplication balance (balance_mscoco) does not."
        ) from ex
    return BalancedSampleForest(sample_grid, rng=rng)


def compute_index_weights(
    mscoco_fpath,
    *,
    class_weights: Optional[Dict[str, float]] = None,
    subdivide_keys: Sequence[str] = ("classes",),
    seed: int = 0,
    max_oversample: Optional[int] = None,
    forest_factory=_default_forest_factory,
) -> List[float]:
    """Flat per-index sample weights from hierarchical stratified balance.

    ``class_weights`` (e.g. the ``KCD_BALANCE_TARGET_JSON`` dict, with
    ``<empty>`` for annotation-free images) biases the ``classes``
    subdivision; omitted classes get the forest's default handling.

    ``max_oversample`` caps how many times any single index may appear per
    epoch on average.  The cap is ``max_oversample / N`` per index (where
    ``N`` is the dataset size); after capping, weights are renormalized.
    This mirrors the ``max_oversample`` parameter in file-mode
    ``balance_mscoco``: both prevent data-starved strata from dominating
    the epoch and guard against per-tile memorization.  ``None`` means no
    cap (forest weights used as-is, which can produce very high oversample
    ratios for rare classes against large datasets).

    Weights are normalized to sum to 1.
    """
    grid = build_sample_grid_from_mscoco(mscoco_fpath)
    if not grid:
        raise ValueError(f"no images in {mscoco_fpath}")
    forest = forest_factory(grid, rng=seed)
    for key in subdivide_keys:
        w = class_weights if key == "classes" else None
        forest.subdivide(key, weights=w)

    if not hasattr(forest, "index_weights"):
        raise NotImplementedError(
            "BalancedSampleForest has no .index_weights() yet — sampler-"
            "mode balancing needs the kwcoco_dataloader version that "
            "flattens the hierarchical strata to per-index weights. "
            "Update tpl/kwcoco_dataloader (or use the default "
            "file-duplication balance mode meanwhile)."
        )
    weights = [float(w) for w in forest.index_weights()]
    if len(weights) != len(grid):
        raise ValueError(
            f"forest.index_weights() returned {len(weights)} weights for "
            f"{len(grid)} images"
        )
    if any((w < 0 or math.isnan(w)) for w in weights):
        raise ValueError("index weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("index weights sum to zero")
    weights = [w / total for w in weights]

    if max_oversample is not None:
        k = int(max_oversample)
        if k <= 0:
            raise ValueError(f"max_oversample must be > 0; got {k}")
        # Iterative cap (numpy): after each renorm some weights may creep back
        # above the cap (renorm divides by total < 1 when indices were capped).
        # Repeat until stable — converges in O(distinct-strata) iterations,
        # usually 2-3 in practice (worst case ~30 for highly skewed data).
        # Use numpy throughout so each O(N) pass is vectorised, not a Python
        # loop.  Also bound the outer loop at a constant: the Python-list
        # version used range(len(weights)+1) which could run 900k+ times once
        # FP rounding prevented strict convergence, tying up the process for
        # many hours on large datasets.
        import numpy as np
        w_arr = np.array(weights, dtype=np.float64)
        cap = k / len(w_arr)
        n_capped_total = 0
        for _ in range(512):
            over = w_arr > cap
            if not over.any():
                break
            n_capped_total += int(over.sum())
            np.minimum(w_arr, cap, out=w_arr)
            total = w_arr.sum()
            if total <= 0:
                raise ValueError(
                    f"all weights capped to zero with max_oversample={k}")
            w_arr /= total
        weights = w_arr.tolist()
        if n_capped_total:
            print(f"[balanced_sampler] max_oversample={k}: capped "
                  f"{n_capped_total} index-iterations; renormalized.")

    return weights


def write_balance_weights(fpath, weights: Sequence[float],
                          meta: Optional[Dict[str, Any]] = None) -> Path:
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps({
        "weights": [float(w) for w in weights],
        "meta": dict(meta or {}),
    }))
    return fpath


def load_balance_weights(fpath) -> List[float]:
    data = json.loads(Path(fpath).read_text())
    return [float(w) for w in data["weights"]]


# ---------------------------------------------------------------------------
# The sampler (training-time, torch)
# ---------------------------------------------------------------------------

class DistributedWeightedRandomSampler:
    """Rank-aware weighted sampling with replacement.

    Each rank draws ``ceil(num_samples_total / world_size)`` indices per
    epoch from the same weight vector but an independent, deterministic
    stream seeded by ``(seed, epoch, rank)`` — so a resumed run replays
    identical epochs and ranks never duplicate a stream. ``set_epoch``
    must be called per epoch (DEIMv2's patched DataLoader/solver does).

    Duck-types ``torch.utils.data.Sampler`` (``__iter__``/``__len__``);
    no inheritance needed.
    """

    def __init__(
        self,
        weights: Sequence[float],
        *,
        num_samples_total: Optional[int] = None,
        seed: int = 0,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        import torch
        if rank is None or world_size is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            else:
                rank, world_size = 0, 1
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.epoch = 0
        self._weights = torch.as_tensor(list(weights), dtype=torch.double)
        if self._weights.numel() == 0:
            raise ValueError("empty weight vector")
        if (self._weights < 0).any() or self._weights.sum() <= 0:
            raise ValueError("weights must be non-negative with positive sum")
        total = int(num_samples_total or self._weights.numel())
        self.num_samples = int(math.ceil(total / self.world_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        import torch
        g = torch.Generator()
        g.manual_seed(
            self.seed * 1_000_003 + self.epoch * 1_009 + self.rank)
        idx = torch.multinomial(
            self._weights, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples


def sampler_from_weights_file(
    weights_fpath,
    *,
    dataset_len: int,
    epoch_length: Optional[int] = None,
    seed: int = 0,
) -> DistributedWeightedRandomSampler:
    """Build the sampler from a sidecar written at launch time.

    ``dataset_len`` must equal the weight count — a mismatch means the
    weights were computed from a different file than the dataset the
    loader is actually serving, which is exactly the silent-skew bug
    this check exists to make loud.
    """
    weights = load_balance_weights(weights_fpath)
    if len(weights) != int(dataset_len):
        raise ValueError(
            f"balance weights ({len(weights)} entries from "
            f"{weights_fpath}) do not match the train dataset "
            f"({dataset_len} images) — the weights were computed from a "
            "different annotation file"
        )
    return DistributedWeightedRandomSampler(
        weights,
        num_samples_total=(int(epoch_length) if epoch_length else None),
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class BalancedSamplerConfig(scfg.DataConfig):
    """Compute per-index balance weights for dataloader-level sampling."""

    src = scfg.Value(None, required=True, help="unbalanced .mscoco.json")
    dst = scfg.Value(None, required=True, help="output balance_weights.json")
    class_weights = scfg.Value(None, type=str, help=(
        "JSON dict of per-class target weights (KCD_BALANCE_TARGET_JSON "
        "semantics; use '<empty>' for annotation-free images)"))
    subdivide_keys = scfg.Value("classes", type=str, help=(
        "CSV of grid keys to stratify over, outermost first"))
    max_oversample = scfg.Value(None, type=int, help=(
        "Cap per-index weight at max_oversample/N before normalizing. "
        "Mirrors balance_mscoco's max_oversample: prevents data-starved "
        "strata from dominating and limits per-tile repetition. "
        "None = no cap (forest weights as-is)."))
    seed = scfg.Value(0, type=int)

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        class_weights = (
            json.loads(config.class_weights) if config.class_weights else None
        )
        keys = [k.strip() for k in str(config.subdivide_keys).split(",")
                if k.strip()]
        weights = compute_index_weights(
            config.src,
            class_weights=class_weights,
            subdivide_keys=keys,
            max_oversample=config.max_oversample,
            seed=int(config.seed),
        )
        out = write_balance_weights(config.dst, weights, meta={
            "src": str(config.src),
            "class_weights": class_weights,
            "subdivide_keys": keys,
            "seed": int(config.seed),
            "n_images": len(weights),
        })
        nz = sum(1 for w in weights if w > 0)
        print(f"[balanced_sampler] wrote {len(weights)} weights "
              f"({nz} nonzero) -> {out}")
        return 0


__cli__ = BalancedSamplerConfig

if __name__ == "__main__":
    raise SystemExit(BalancedSamplerConfig.main())
